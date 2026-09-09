import cv2
import numpy as np
import joblib
import mysql.connector
import os
import uuid
import re
import base64
from io import BytesIO
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from skimage.feature import hog
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import remove_small_objects
from tagalog_to_baybayin import TagalogToBaybayin

# Optional HEIC/HEIF support for iPhone photos. Harmless if not installed;
# `pip install pillow-heif` to enable.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_SUPPORT = True
except Exception:
    _HEIF_SUPPORT = False

app = Flask(__name__)
CORS(app)

ttb_translator = TagalogToBaybayin()

# --- 1. AI & ARCHIVE PATH CONFIG ---
ARCHIVE_ROOT = 'open_archival_dataset'
TEMP_ROOT = 'temp_crops'
BASE_DIR = Path(__file__).resolve().parent


def load_joblib_artifact(*candidate_names):
    for name in candidate_names:
        artifact_path = BASE_DIR / name
        if artifact_path.exists():
            return joblib.load(artifact_path)
    return None

for folder in [ARCHIVE_ROOT, TEMP_ROOT]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Weighted HOG + spatial SVM (see backend/MODEL_README.txt) ---
# Feature vector = HOG features ++ spatial features. The two blocks are scaled by
# SEPARATE scalers; the spatial block is then multiplied by SPATIAL_WEIGHT before
# the blocks are concatenated for the SVM. Lengths + the training glyph size are
# read back from the scalers below, so a model retrained at a different
# resolution (e.g. 96x96) drops in with no code change.
HOG_FEATURE_LEN = 1764      # 64x64 default; overridden from hog_scaler
SPATIAL_FEATURE_LEN = 26    # overridden from spatial_scaler

model = load_joblib_artifact('weighted_svm.pkl', 'svm_model.pkl',
                             'baybayin_svm_model.pkl', 'baybayin_svm_model.sav')
hog_scaler = load_joblib_artifact('hog_scaler.pkl', 'baybayin_scaler.pkl', 'baybayin_scaler.sav')
spatial_scaler = load_joblib_artifact('spatial_scaler.pkl')
_raw_weight = load_joblib_artifact('best_weight.pkl', 'spatial_weight.pkl')
SPATIAL_WEIGHT = float(_raw_weight) if _raw_weight is not None else 6.0
label_encoder = load_joblib_artifact('label_encoder.pkl', 'baybayin_classes.pkl', 'baybayin_classes.sav')

if hog_scaler is not None and getattr(hog_scaler, 'n_features_in_', None):
    HOG_FEATURE_LEN = int(hog_scaler.n_features_in_)
if spatial_scaler is not None and getattr(spatial_scaler, 'n_features_in_', None):
    SPATIAL_FEATURE_LEN = int(spatial_scaler.n_features_in_)
# HOG length with ppc=8, cpb=2, orient=9 is ((T/8 - 1)^2) * 36, so:
_cells = int(round((HOG_FEATURE_LEN / 36.0) ** 0.5)) + 1
DERIVED_TARGET_SIZE = _cells * 8  # 1764 -> 64, 4356 -> 96

# Isotonic probability calibrator wrapping `model` (built by calibrate_model.py).
# When present, classify_glyph() takes both the label and the confidence from
# its predict_proba(); otherwise it falls back to a softmax over the raw SVM
# decision_function, which is poorly calibrated (see calibrate_model.py docstring).
calibrated_model = load_joblib_artifact('weighted_svm_calibrated.pkl')

# --- Path A: 6-class kudlit/virama "mark corrector" (optional) -----------------
# A small separate SVM (train_mark_corrector.py) that reads ONLY the mark region.
# It never touches the monolith's BASE letter - it only proposes a different
# mark, and reconcile_mark() applies that ONLY when it is confident AND the
# monolith itself was unsure. Missing files -> feature stays off, monolith
# behaviour is byte-identical.
mark_model = load_joblib_artifact('mark_svm.pkl')
mark_calibrated = load_joblib_artifact('mark_svm_calibrated.pkl')
mark_scaler = load_joblib_artifact('mark_scaler.pkl')
MARK_CLASSES = ["none", "virama", "dot_above", "dash_above", "dot_below", "dash_below"]
# only override the monolith when the corrector is at least this confident ...
MARK_OVERRIDE_MIN = 0.80
# ... and the monolith's own confidence was below this (its sure calls are ~95%+
# right, so don't second-guess them).
MARK_TRUST_MONO_ABOVE = 0.90

# `class_names` stays a plain list so the rest of the file is unchanged: it maps
# the SVM's integer output back to a glyph name.
if label_encoder is not None and hasattr(label_encoder, 'classes_'):
    class_names = [str(c) for c in label_encoder.classes_]
elif isinstance(label_encoder, (list, tuple, np.ndarray)):
    class_names = [str(c) for c in label_encoder]
else:
    class_names = []

# scaler alias kept so any external import of `scaler` still resolves
scaler = hog_scaler

_missing = [n for n, v in (
    ('weighted_svm.pkl', model),
    ('hog_scaler.pkl', hog_scaler),
    ('spatial_scaler.pkl', spatial_scaler),
    ('label_encoder.pkl', label_encoder),
) if v is None]

if _missing or not class_names:
    print(
        "❌ Critical Error: Missing Baybayin classifier artifacts in backend/: "
        + ", ".join(_missing or ['label_encoder classes'])
        + ". See backend/MODEL_README.txt for the expected file set."
    )
else:
    _conf_mode = "isotonic-calibrated" if calibrated_model is not None else "softmax proxy (uncalibrated)"
    print(
        f"✅ AI System Online. Loaded {len(class_names)} classes "
        f"(spatial weight = {SPATIAL_WEIGHT:g}, confidence = {_conf_mode})."
    )

# --- 2. DATABASE CONFIG ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '', 
    'database': 'dayaw' 
}

# --- 3. DATABASE HELPERS ---

def get_db_connection():
    return mysql.connector.connect(**db_config)

def start_processing_session(ip_address):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "INSERT INTO processing_sessions (status, ip_address) VALUES ('Processing', %s)"
        cursor.execute(query, (ip_address,))
        new_id = cursor.lastrowid 
        conn.commit()
        return new_id
    except Exception as e:
        print(f"❌ DB Session Error: {e}")
        return 0
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def log_detections(session_id, detections_list):
    if not detections_list or session_id == 0: return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        formatted_logs = [(session_id, d['char'], d['confidence']) for d in detections_list]
        query = "INSERT INTO detection_logs (session_id, detected_char, confidence_score) VALUES (%s, %s, %s)"
        cursor.executemany(query, formatted_logs)
        conn.commit()
    except Exception as e:
        print(f"❌ Log Error: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def update_session_status(session_id, status):
    if session_id == 0: return
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "UPDATE processing_sessions SET status = %s, end_time = CURRENT_TIMESTAMP WHERE session_id = %s"
        cursor.execute(query, (status, session_id))
        conn.commit()
    except Exception as e:
        print(f"❌ Update Error: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# --- 4. IMAGE PROCESSING & AUTO-CROP ENGINE (SMART PARAGRAPH SEGMENTATION + SVM PREDICTION) ---

# Adaptive segmentation constants for multi-line Baybayin paragraphs.
# Based on training/smart_segmentation.py, then nudged from testing on real
# handwriting: WORD_GAP_RATIO 1.5 -> 0.78 (larger values merged loosely spaced
# words, e.g. "ako kane" -> "akokane"); H_DILATE 0.24 -> 0.20 (0.24 over-merged
# adjacent glyphs). Within-word gaps run ~0-0.12 w, word gaps ~0.8 w+.
# Note: connected cursive is near the ceiling of what this morphology
# approach can do - some cuts will land wrong no matter the constants.
STANDARD_WIDTH = 1600
MIN_CHAR_AREA_RATIO = 0.0003
MIN_BOX_HEIGHT_RATIO = 0.28
MIN_BOX_WIDTH_RATIO = 0.18
MIN_BOX_AREA_RATIO = 0.22
EDGE_MARGIN_PX = 10
# 0.45 (was 0.38): bridge the gap from a glyph body to its o/u kudlit dot /
# virama so they stay ONE contour - otherwise the mark splits off, and a lost
# mark reads "no" as "na" / "n" as "na".
V_DILATE_RATIO = 0.45
H_DILATE_RATIO = 0.20
V_DILATE_FALLBACK = 45
H_DILATE_FALLBACK = 12
ROW_GROUPING_RATIO = 0.65
WORD_GAP_RATIO = 0.78
MIN_COMPONENT_HEIGHT = 15
SPLIT_WIDTH_RATIO = 1.7
SPLIT_SEARCH_WINDOW = 0.35
SPLIT_MIN_GAP_RATIO = 0.32
MIN_SEGMENT_WIDTH_RATIO = 0.55
BG_BLUR_KERNEL = 101


def normalize_image_size(img, standard_width=STANDARD_WIDTH, upscale_only=False):
    """Bring the working image to `standard_width` so every ratio-based
    threshold downstream is predictable AND - more importantly - so each glyph
    lands ~80-120px before the per-glyph crop->64 resize. Leaving a big photo
    at full size makes glyphs 200px+, so the crop downscale is *more* brutal on
    the mark, not less (AUDIT #4 first tried upscale-only and it regressed
    "sa labas" -> "sa lobas"; reverted). upscale_only=True (white-paper mode)
    still skips the shrink for already-clean scans.
    """
    h, w = img.shape[:2]
    if upscale_only and w >= standard_width:
        return img
    scale = standard_width / w
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (standard_width, new_h), interpolation=interp)


def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 20:
        return img, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.3 or abs(angle) > 20:
        return img, angle

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def flatten_background(gray, blur_kernel=BG_BLUR_KERNEL):
    """CamScanner-style "clean to white": divide the image by a heavily blurred
    copy of itself to cancel uneven lighting, shadows and off-white paper.

    AUDIT #5: on an already-clean, evenly-lit scan the divide is a no-op but the
    final NORM_MINMAX contrast-stretch can blow a faint kudlit / virama toward
    white and drop it below Otsu. So the stretch now runs ONLY when the
    background is genuinely uneven (blurred-bg spread is wide). A clean image
    passes through with just the divide.
    """
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    bg_uneven = float(bg.std()) > 18.0          # ~uniform paper -> std is small
    bg = np.where(bg < 1, 1, bg).astype(np.uint8)
    norm = cv2.divide(gray, bg, scale=255)
    if bg_uneven:
        return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    return norm


# ----------------------------------------------------------------------------
# Pen vs marker. The dataset is marker-weight; a thin ballpen stroke loses its
# kudlit dot to remove_small_objects and to the 64px downscale. "pen" mode
# (a) grows the ink ~1px so strokes approach marker weight and the kudlit
# survives the resize, and (b) keeps the noise floor low so the dot isn't
# erased. "marker" mode is the current behaviour, unchanged.
PEN_MIN_NOISE = 5            # remove_small_objects floor in pen mode (marker: 20)
STROKE_THICKEN_ITERS = 1     # 2x2 dilation of the dark ink in pen mode


def _thicken_ink(gray, iters=STROKE_THICKEN_ITERS):
    """Grow dark strokes by ~1px. Eroding a grayscale image expands the dark
    (ink) regions, so a thin pen line gets closer to marker weight without
    changing the glyph's shape."""
    return cv2.erode(gray, np.ones((2, 2), np.uint8), iterations=iters)


def measure_average_char_size(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    light_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(thresh, light_kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    heights = [
        stats[i, cv2.CC_STAT_HEIGHT]
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_HEIGHT] > MIN_COMPONENT_HEIGHT
    ]
    widths = [
        stats[i, cv2.CC_STAT_WIDTH]
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_HEIGHT] > MIN_COMPONENT_HEIGHT
    ]

    if heights and widths:
        avg_height = int(np.median(heights))
        avg_width = int(np.median(widths))
    else:
        avg_height = None
        avg_width = None

    return avg_height, avg_width


def merge_by_proximity(boxes, avg_width, avg_height,
                       x_gap_ratio=H_DILATE_RATIO, y_gap_ratio=V_DILATE_RATIO):
    """Non-destructive stand-in for morphological dilation (white-paper mode).

    Union two boxes when their bounding rects nearly touch AND share a column -
    i.e. a kudlit / virama sitting above or below its body, or a broken cursive
    stroke - without altering a single pixel, so the glyph shape the classifier
    sees is exactly what was on the paper. Iterates to a fixed point."""
    boxes = [list(b) for b in boxes]
    x_gap = x_gap_ratio * avg_width
    y_gap = y_gap_ratio * avg_height
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax, ay, aw, ah = boxes[i]
                bx, by, bw, bh = boxes[j]
                dx = max(ax, bx) - min(ax + aw, bx + bw)   # >0 => horizontal gap
                dy = max(ay, by) - min(ay + ah, by + bh)   # >0 => vertical gap
                x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
                if dx <= x_gap and dy <= y_gap and x_overlap > -0.25 * avg_width:
                    nx0, ny0 = min(ax, bx), min(ay, by)
                    nx1, ny1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    boxes[i] = [nx0, ny0, nx1 - nx0, ny1 - ny0]
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]


def detect_character_boxes(img, edge_margin=EDGE_MARGIN_PX, white_paper=False,
                           pen_type="marker", stages=None):
    img = normalize_image_size(img, upscale_only=white_paper)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if stages is not None:
        stages["1_normalized"] = img.copy()
    gray = flatten_background(gray)  # normalise lighting before any thresholding
    if pen_type == "pen":
        gray = _thicken_ink(gray)   # thin ballpen -> ~marker weight (see _thicken_ink)
    if stages is not None:
        stages["2_flattened"] = gray.copy()
    h_img, w_img = gray.shape

    avg_height, avg_width = measure_average_char_size(gray)
    if avg_height and avg_width:
        v_dilate = max(15, int(avg_height * V_DILATE_RATIO))
        h_dilate = max(5, int(avg_width * H_DILATE_RATIO))
    else:
        v_dilate = V_DILATE_FALLBACK
        h_dilate = H_DILATE_FALLBACK
        avg_height = 80
        avg_width = 60

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if white_paper:
        # no pixel-fattening: contour the clean threshold as-is, then glue
        # kudlit/virama to their body by proximity afterwards.
        source = thresh
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_dilate, v_dilate))
        source = cv2.dilate(thresh, kernel, iterations=1)
    if stages is not None:
        stages["3_binary"] = thresh.copy()
        stages["4_grouped"] = source.copy()  # dilated, or raw threshold in white-paper mode

    min_area = MIN_CHAR_AREA_RATIO * h_img * w_img
    min_box_height = avg_height * MIN_BOX_HEIGHT_RATIO
    min_box_width = avg_width * MIN_BOX_WIDTH_RATIO
    min_box_area = (avg_width * avg_height) * MIN_BOX_AREA_RATIO
    if white_paper:
        # a lone kudlit dot / virama is a real box here (no dilation glued it
        # on yet), so don't let the size filters delete it before the merge
        min_box_height *= 0.35
        min_box_width *= 0.35
        min_box_area *= 0.12

    contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) <= min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)

        if h < min_box_height:
            continue
        if w < min_box_width:
            continue
        if (w * h) < min_box_area:
            continue

        # AUDIT #6: don't delete a glyph just for touching the frame - a kudlit
        # near the edge used to take the whole character with it. Clamp to the
        # frame, and only drop it if most of the box was actually outside.
        cx0, cy0 = max(x, edge_margin), max(y, edge_margin)
        cx1 = min(x + w, w_img - edge_margin)
        cy1 = min(y + h, h_img - edge_margin)
        cw, ch = cx1 - cx0, cy1 - cy0
        if cw < min_box_width or ch < min_box_height:
            continue
        if cw * ch < 0.55 * w * h:          # >45% of the box was off-frame -> junk
            continue

        boxes.append((cx0, cy0, cw, ch))

    if white_paper:
        boxes = merge_by_proximity(boxes, avg_width, avg_height)

    return boxes, gray, img, avg_height, avg_width


def split_merged_box(box, gray, avg_width, split_ratio=SPLIT_WIDTH_RATIO,
                     search_window=SPLIT_SEARCH_WINDOW, min_gap_ratio=SPLIT_MIN_GAP_RATIO,
                     min_segment_ratio=MIN_SEGMENT_WIDTH_RATIO):
    x, y, w, h = box
    n_expected = max(1, round(w / avg_width))

    if w < avg_width * split_ratio or n_expected <= 1:
        return [box]

    crop = gray[y:y + h, x:x + w]
    _, crop_bin = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_density = crop_bin.sum(axis=0)
    peak_density = col_density.max() if col_density.max() > 0 else 1

    split_points = []
    for i in range(1, n_expected):
        expected_x = int(w * i / n_expected)
        window = int(avg_width * search_window)
        lo = max(1, expected_x - window)
        hi = min(w - 1, expected_x + window)
        if lo >= hi:
            continue
        local_slice = col_density[lo:hi]
        best_offset = int(np.argmin(local_slice))
        candidate_x = lo + best_offset
        candidate_density = col_density[candidate_x]

        if candidate_density < min_gap_ratio * peak_density:
            split_points.append(candidate_x)

    if not split_points:
        return [box]

    split_points = sorted(set([0] + split_points + [w]))
    sub_boxes = []
    for i in range(len(split_points) - 1):
        seg_x0, seg_x1 = split_points[i], split_points[i + 1]
        seg_w = seg_x1 - seg_x0
        if seg_w < avg_width * 0.3:
            continue
        sub_boxes.append((x + seg_x0, y, seg_w, h))

    if not sub_boxes:
        return [box]

    return sub_boxes


def resolve_merge_or_split(box, candidate_pieces, gray, score_fn):
    """Ask the classifier whether a geometrically-split box was really one glyph.

    `score_fn(gray_patch) -> confidence in [0, 1]`. Keep the whole box unless
    EVERY candidate piece is classified at least as confidently as the whole -
    i.e. only accept a split the model is sure about. This stops a single
    cursive glyph (e.g. the "nga" ligature) from being chopped into two
    low-confidence fragments.
    """
    if len(candidate_pieces) <= 1:
        return [box]

    x, y, w, h = box
    whole_score = score_fn(gray[y:y + h, x:x + w])
    sub_scores = [score_fn(gray[sy:sy + sh, sx:sx + sw])
                  for (sx, sy, sw, sh) in candidate_pieces]

    if not sub_scores or whole_score >= min(sub_scores):
        return [box]
    return candidate_pieces


def split_all_merged_boxes(boxes, gray, avg_width, score_fn=None):
    result = []
    for box in boxes:
        candidate_pieces = split_merged_box(box, gray, avg_width)
        if len(candidate_pieces) <= 1:
            result.extend(candidate_pieces)
            continue

        if score_fn is not None:
            final_pieces = resolve_merge_or_split(box, candidate_pieces, gray, score_fn)
        else:
            final_pieces = candidate_pieces

        result.extend(final_pieces)
    return result


def drop_stray_marks(boxes, avg_height, avg_width):
    """Fold a detached diacritic box (virama "krus", e/i/o/u kudlit dot) back
    into the base glyph it belongs to instead of leaving it as a phantom
    character: a tiny box (< 25% median area AND short side) that shares a
    column with a real box is unioned into it; a tiny box overlapping no base
    box (a free-standing speck) is dropped.

    NOTE: an earlier "net 2" also folded small-ish (< 55% median) boxes that
    hung off a neighbour's band - it ate genuinely small thin glyphs (Ha, I,
    U): "bahay" -> "ba y". Reverted. Bold-marker viramas can still surface as a
    phantom "Ge"; group_into_lines' line-merge keeps them off their own line."""
    if len(boxes) < 4:
        return boxes

    areas = sorted(w * h for (_, _, w, h) in boxes)
    median_area = areas[len(areas) // 2]
    small_side = 0.66 * min(avg_width, avg_height)

    def x_overlap(a, b):
        return max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))

    def v_gap(m, base):
        my0, my1, by0, by1 = m[1], m[1] + m[3], base[1], base[1] + base[3]
        if my1 <= by0:
            return by0 - my1
        if my0 >= by1:
            return my0 - by1
        return 0

    marks, bases = [], []
    for b in boxes:
        _, _, w, h = b
        (marks if (w * h < 0.25 * median_area and min(w, h) < small_side)
         else bases).append(b)

    if not marks or not bases:
        return boxes

    def centre(b):
        return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)

    merged = list(bases)
    for m in marks:
        # 1st choice: a base in the same column (kudlit sits directly above/below)
        cand = [i for i, base in enumerate(merged)
                if x_overlap(m, base) >= 0.35 * m[2]
                and v_gap(m, base) < 1.3 * avg_height]
        if not cand:
            # AUDIT #7: don't drop a real detached mark. Attach it to the
            # NEAREST base within ~1.5 glyph-widths (segmentation may have
            # placed it a little off). Only a mark isolated from everything
            # (> 1.5 avg_width) is discarded as a true speck.
            mcx, mcy = centre(m)
            near = [(i, ((mcx - centre(b)[0]) ** 2 + (mcy - centre(b)[1]) ** 2) ** 0.5)
                    for i, b in enumerate(merged)]
            i, dist = min(near, key=lambda t: t[1])
            if dist > 1.5 * avg_width:
                continue                       # truly free-standing -> drop
            cand = [i]
        j = min(cand, key=lambda i: v_gap(m, merged[i]))
        bx, by, bw, bh = merged[j]
        nx0, ny0 = min(bx, m[0]), min(by, m[1])
        nx1, ny1 = max(bx + bw, m[0] + m[2]), max(by + bh, m[1] + m[3])
        merged[j] = (nx0, ny0, nx1 - nx0, ny1 - ny0)

    return merged


def group_into_lines(boxes, avg_height, row_ratio=ROW_GROUPING_RATIO):
    if not boxes:
        return []

    row_threshold = avg_height * row_ratio
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    lines = []

    for box in boxes_sorted:
        placed = False
        for line in lines:
            avg_y = np.mean([b[1] + b[3] / 2 for b in line])
            box_y_center = box[1] + box[3] / 2
            if abs(avg_y - box_y_center) < row_threshold:
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    # merge lines whose vertical bands overlap - small or wavy writing otherwise
    # fragments one physical row into several (pilipino x4: 4 rows -> 15).
    def _cy(ln):
        return float(np.median([b[1] + b[3] / 2 for b in ln]))

    lines.sort(key=_cy)
    fused = []
    for line in lines:
        if fused and abs(_cy(line) - _cy(fused[-1])) < avg_height * 0.9:
            fused[-1].extend(line)
        else:
            fused.append(line)
    lines = fused

    for line in lines:
        line.sort(key=lambda b: b[0])

    return lines


def insert_word_breaks(line_boxes, avg_width, word_gap_ratio=WORD_GAP_RATIO):
    """Split a line into words. A fixed `word_gap_ratio * avg_width` cut can't fit
    every hand at once (0.78 merged "sa labas"; lower it and "salamat" splits).
    So we also read THIS line's own gap distribution: within-word gaps cluster
    small, word gaps sit above the widest ratio-jump in the sorted gaps. The
    adaptive threshold can only lower the fixed one, never raise it, and is
    floored so a faint jump can't over-split."""
    if len(line_boxes) < 2:
        return line_boxes

    gaps = []
    for i in range(1, len(line_boxes)):
        prev_right = line_boxes[i - 1][0] + line_boxes[i - 1][2]
        gaps.append(max(0.0, line_boxes[i][0] - prev_right))

    threshold = avg_width * word_gap_ratio
    if len(gaps) >= 3:
        ordered = sorted(gaps)
        best_ratio, split_at = 1.0, None
        for lo, hi in zip(ordered, ordered[1:]):
            if hi >= 0.30 * avg_width and lo > 1e-6 and hi / lo > best_ratio:
                best_ratio, split_at = hi / lo, 0.5 * (lo + hi)
        if split_at is not None and best_ratio >= 1.8:
            threshold = min(threshold, max(split_at, 0.30 * avg_width))

    result = [line_boxes[0]]
    for i in range(1, len(line_boxes)):
        if gaps[i - 1] > threshold:
            result.append(None)
        result.append(line_boxes[i])
    return result


# --- WEIGHTED MODEL v2: preprocessing + feature extraction ---
# Direct port of the Colab reference (inference.py -> preprocess_v2 /
# hog_extract_v2). Every step here must match training exactly; the README
# and reference both warn that a mismatch silently wrecks accuracy.

# Follows whatever size the loaded model was trained at (64 or 96), inferred
# from the HOG scaler above.
TARGET_SIZE = DERIVED_TARGET_SIZE
MIN_NOISE_SIZE = 20
PAD_RATIO = 0.12
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'


def _despeckle(binary, min_noise_size):
    """remove_small_objects, but a no-op when min_noise_size <= 0 (white-paper
    mode keeps every ink blob so a thin-pen kudlit dot is never erased)."""
    if not min_noise_size or min_noise_size <= 0:
        return binary
    return (remove_small_objects(binary > 0, min_size=min_noise_size) * 255).astype(np.uint8)


def _tight_box_from_gray(gray_patch, min_noise_size=MIN_NOISE_SIZE):
    """Return the tight ink-bounds inside a grayscale patch, in patch-local coords."""
    if gray_patch is None or gray_patch.size == 0:
        return None

    _, binary = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned = _despeckle(binary, min_noise_size)
    if cleaned.sum() == 0:
        return None

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w < 3 or h < 3:
        return None

    return x, y, w, h


def _prepare_character_crop(gray_patch, target_size=TARGET_SIZE,
                            min_noise_size=MIN_NOISE_SIZE, pad_ratio=PAD_RATIO):
    """Raw grayscale glyph patch -> clean `target_size` image (port of
    inference.preprocess_image). Otsu-invert -> remove small objects ->
    tight crop -> pad to square (ratio 0.12) -> resize. The result is NOT
    re-thresholded: HOG runs on the anti-aliased resize, exactly as in
    training."""
    if gray_patch is None or gray_patch.size == 0:
        return None

    _, binary = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned = _despeckle(binary, min_noise_size)
    if cleaned.sum() == 0:
        return None

    tight = _tight_box_from_gray(gray_patch, min_noise_size=min_noise_size)
    if tight is None:
        return None

    x, y, w, h = tight
    tight_patch = cleaned[y:y + h, x:x + w]
    if tight_patch.size == 0:
        return None

    side = max(w, h)
    pad = int(side * pad_ratio)
    canvas_side = side + 2 * pad
    canvas = np.zeros((canvas_side, canvas_side), dtype=np.uint8)
    y_off, x_off = (canvas_side - h) // 2, (canvas_side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = tight_patch

    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)


def _grid_density_features(binary_img, grid_size):
    """Port of inference.grid_density_features."""
    h, w = binary_img.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    densities = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0 = gy * cell_h
            y1 = (gy + 1) * cell_h if gy < grid_size - 1 else h
            x0 = gx * cell_w
            x1 = (gx + 1) * cell_w if gx < grid_size - 1 else w
            cell = binary_img[y0:y1, x0:x1]
            densities.append(cell.sum() / (cell.size * 255) if cell.size > 0 else 0)
    return densities


def _kudlit_component_features(binary_img):
    """Port of inference.kudlit_component_features: connected-component stats
    capturing base-glyph vs kudlit-mark geometry."""
    labeled = sk_label(binary_img > 0)
    regions = regionprops(labeled)
    n_components = len(regions)

    if n_components == 0:
        return [0, 0, 0, 0, 0]

    regions_sorted = sorted(regions, key=lambda r: -r.area)
    total_ink_area = sum(r.area for r in regions_sorted)
    largest = regions_sorted[0]
    largest_area_frac = largest.area / total_ink_area if total_ink_area > 0 else 0

    if n_components >= 2:
        secondary = regions_sorted[1]
        secondary_area_ratio = secondary.area / largest.area if largest.area > 0 else 0
        h, w = binary_img.shape
        dy = (secondary.centroid[0] - largest.centroid[0]) / h
        dx = (secondary.centroid[1] - largest.centroid[1]) / w
    else:
        secondary_area_ratio, dy, dx = 0.0, 0.0, 0.0

    return [n_components, largest_area_frac, secondary_area_ratio, dy, dx]


KUDLIT_STRIP_RATIO = 0.22  # top / bottom band width, as a fraction of glyph height


def _kudlit_strip_features(binary):
    """Fraction of the glyph's ink that sits in the top band vs the bottom band.

    The o/u kudlit is a mark *below* the body, the e/i kudlit is *above* it, and
    a bare consonant / -a form has neither. HOG can't tell Go from G (same main
    stroke) and the connected-component features miss a kudlit that touches the
    body; these two numbers fire regardless (ink low in the frame -> o/u, ink
    high -> e/i). They are appended last so an older 26-feature scaler still
    slices cleanly."""
    h = binary.shape[0]
    strip = max(1, int(round(h * KUDLIT_STRIP_RATIO)))
    total = float(binary.sum()) or 1.0
    top_frac = float(binary[:strip].sum() / total)
    bot_frac = float(binary[h - strip:].sum() / total)
    return [top_frac, bot_frac]


KUDLIT_SHAPE_BAND = 0.16  # top / bottom band, fraction of glyph height


def _kudlit_shape_features(binary):
    """Shape of the mark in the top band vs the bottom band:
    [top_aspect, top_relwidth, bot_aspect, bot_relwidth].

    A *dash* kudlit is wide and flat (aspect w/h > ~2, spans a big share of the
    glyph width); a *dot* kudlit is compact (aspect ~1, narrow). Density and
    position features fire the same for both, so this is the signal that
    separates Ne/Ni (dash vs dot above) and Nu/No (dash vs dot below).
    Appended last so a shorter scaler still slices cleanly."""
    h, w = binary.shape
    band = max(1, int(round(h * KUDLIT_SHAPE_BAND)))
    cols_all = np.where(binary.sum(axis=0) > 0)[0]
    glyph_w = float(cols_all[-1] - cols_all[0] + 1) if cols_all.size else float(w)

    def strip_shape(strip):
        cs = np.where(strip.sum(axis=0) > 0)[0]
        rs = np.where(strip.sum(axis=1) > 0)[0]
        if cs.size == 0 or rs.size == 0:
            return [0.0, 0.0]
        mw = float(cs[-1] - cs[0] + 1)
        mh = float(rs[-1] - rs[0] + 1)
        return [min(6.0, mw / max(1.0, mh)),
                min(1.5, mw / max(1.0, glyph_w))]

    return strip_shape(binary[:band]) + strip_shape(binary[h - band:])


def _extract_spatial_features(preprocessed_img):
    """Spatial feature vector: overall density (1) + 2x2 grid (4) + 4x4 grid (16)
    + kudlit component stats (5) + kudlit top/bottom strip fractions (2)
    + kudlit top/bottom mark shape (4) = 32.
    The first 26 are the Colab inference.extract_spatial_features port; the
    trailing 6 are new (see _kudlit_strip_features / _kudlit_shape_features).
    _build_feature_vector slices to whatever the loaded spatial_scaler expects,
    so this is safe with a 26-, 28- or 32-feature model."""
    _, binary = cv2.threshold(preprocessed_img, 127, 255, cv2.THRESH_BINARY)
    overall_density = [binary.sum() / (binary.size * 255)]
    quadrant = _grid_density_features(binary, grid_size=2)
    fine_grid = _grid_density_features(binary, grid_size=4)
    kudlit_feats = _kudlit_component_features(binary)
    strip_feats = _kudlit_strip_features(binary)
    shape_feats = _kudlit_shape_features(binary)
    return (overall_density + quadrant + fine_grid + kudlit_feats
            + strip_feats + shape_feats)


def _build_feature_vector(preprocessed_img):
    """64x64 glyph -> scaled, spatially-weighted 1790-dim SVM input
    (port of inference.extract_combined_features + the scaling block of
    inference.predict_character)."""
    hog_features = hog(
        preprocessed_img, orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL, cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM, feature_vector=True,
    )
    spatial_features = np.asarray(_extract_spatial_features(preprocessed_img), dtype=np.float64)

    features_hog = hog_features[:HOG_FEATURE_LEN].reshape(1, -1)
    # slice to what the loaded model expects: 26 (V1-V4) or 28 (kudlit-strip model)
    features_spatial = spatial_features[:SPATIAL_FEATURE_LEN].reshape(1, -1)

    hog_scaled = hog_scaler.transform(features_hog)
    spatial_scaled = spatial_scaler.transform(features_spatial) * SPATIAL_WEIGHT
    return np.hstack([hog_scaled, spatial_scaled])


def _decode_label(pred_int):
    if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
        return str(label_encoder.inverse_transform([int(pred_int)])[0])
    if 0 <= int(pred_int) < len(class_names):
        return class_names[int(pred_int)]
    return str(pred_int)


def classify_glyph(img64, top_k=3):
    """Return (glyph_name, confidence in [0, 1], alternatives) for a preprocessed
    glyph. `alternatives` is the top-`top_k` [name, confidence] pairs, most
    likely first (so alternatives[0] == (glyph_name, confidence)) - the UI shows
    these so a user can tap the runner-up when the top guess is wrong (the
    kudlit confusions like K/Ko almost always have the right answer at #2).

    Feature extraction + label decoding match inference.predict_character. When
    weighted_svm_calibrated.pkl is loaded the scores are isotonic-calibrated
    probabilities; otherwise they are a softmax over the raw SVM margins
    (poorly calibrated - a placeholder until the calibrator is built).
    """
    vec = _build_feature_vector(img64)

    if calibrated_model is not None:
        scores = calibrated_model.predict_proba(vec)[0]
        classes = calibrated_model.classes_
    else:
        margins = model.decision_function(vec)[0]
        exp = np.exp(margins - np.max(margins))
        scores = exp / exp.sum()
        classes = model.classes_

    order = np.argsort(scores)[::-1][:max(1, top_k)]
    alternatives = [[_decode_label(classes[i]), float(scores[i])] for i in order]
    return alternatives[0][0], alternatives[0][1], alternatives


# --- mark corrector (Path A) -------------------------------------------------
MARK_BAND = 0.42          # top / bottom band of the 64px glyph fed to the mark HOG
MARK_CROP = 32
_STANDALONE_VOWELS = {"A", "E", "I", "O", "U"}
_MARK_FROM_TAIL = {"a": "none", "e": "dash_above", "i": "dot_above",
                   "o": "dot_below", "u": "dash_below"}
_MARK_TO_TAIL = {"none": "a", "dash_above": "e", "dot_above": "i",
                 "dot_below": "o", "dash_below": "u"}


def _split_label(name):
    """95-class label -> (base, mark). "Ko"->("K","dot_below"), "K"->("K","virama"),
    "Ka"->("K","none"), "Nga"->("Ng","none"), "A"->("A","none")."""
    if name in _STANDALONE_VOWELS:
        return name, "none"
    low = name.lower()
    if not any(v in low for v in "aeiou"):
        return name, "virama"
    return name[:-1], _MARK_FROM_TAIL[low[-1]]


def _join_label(base, mark):
    return base if mark == "virama" else base + _MARK_TO_TAIL[mark]


def _mark_features(img64):
    """Mark-region feature vector - MUST match train_mark_corrector.mark_features:
    HOG of a 32x32 top-band crop + HOG of a 32x32 bottom-band crop + strip(2)
    + shape(4) + component(5) = 659."""
    _, binary = cv2.threshold(img64, 127, 255, cv2.THRESH_BINARY)
    h = binary.shape[0]
    band = int(round(h * MARK_BAND))
    top = cv2.resize(img64[:band, :], (MARK_CROP, MARK_CROP), interpolation=cv2.INTER_AREA)
    bot = cv2.resize(img64[h - band:, :], (MARK_CROP, MARK_CROP), interpolation=cv2.INTER_AREA)
    hog_kw = dict(orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
                  cells_per_block=HOG_CELLS_PER_BLOCK, block_norm=HOG_BLOCK_NORM,
                  feature_vector=True)
    return np.concatenate([
        hog(top, **hog_kw), hog(bot, **hog_kw),
        _kudlit_strip_features(binary), _kudlit_shape_features(binary),
        _kudlit_component_features(binary),
    ]).astype(np.float64).reshape(1, -1)


def _read_mark(img64):
    """(mark_name, confidence in [0,1]) from the mark corrector."""
    feat = mark_scaler.transform(_mark_features(img64))
    est = mark_calibrated if mark_calibrated is not None else mark_model
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(feat)[0]
        i = int(np.argmax(proba))
        return MARK_CLASSES[int(est.classes_[i])], float(proba[i])
    return MARK_CLASSES[int(est.predict(feat)[0])], 1.0


def reconcile_mark(img64, mono_char, mono_conf):
    """If the mark corrector is loaded, let it override ONLY the mark of the
    monolith's guess, and ONLY when it is confident (>= MARK_OVERRIDE_MIN) and
    the monolith itself was unsure (< MARK_TRUST_MONO_ABOVE). Returns
    (char, info_dict|None). Never changes the base letter."""
    if mark_model is None or mark_scaler is None:
        return mono_char, None
    base, mono_mark = _split_label(mono_char)
    if base in _STANDALONE_VOWELS or mono_conf >= MARK_TRUST_MONO_ABOVE:
        return mono_char, None
    try:
        corr_mark, corr_conf = _read_mark(img64)
    except Exception:
        return mono_char, None
    info = {"base": base, "mono_mark": mono_mark, "corr_mark": corr_mark,
            "corr_conf": round(corr_conf * 100, 1), "applied": False}
    if corr_mark == mono_mark or corr_conf < MARK_OVERRIDE_MIN:
        return mono_char, info
    new_char = _join_label(base, corr_mark)
    if new_char not in class_names:
        return mono_char, info
    info["applied"] = True
    return new_char, info


class UnsupportedImageError(Exception):
    """Raised when uploaded bytes cannot be decoded to an image by any backend."""


def _decode_image_bytes(image_bytes):
    """Uploaded image bytes -> BGR uint8 ndarray.

    Tries OpenCV first (JPEG, PNG, BMP, WEBP, TIFF, PPM/PGM, ...). Falls back to
    Pillow (plus pillow-heif when installed) for HEIC/HEIF and AVIF, honouring
    EXIF orientation on that path. Raises UnsupportedImageError if nothing can
    read it (e.g. PDF, SVG, camera RAW, or a corrupt file).
    """
    if not image_bytes:
        raise UnsupportedImageError("empty upload")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    try:
        with Image.open(BytesIO(image_bytes)) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb = np.asarray(pil_img.convert("RGB"))
    except Exception as exc:
        raise UnsupportedImageError(str(exc)) from exc

    if rgb.size == 0:
        raise UnsupportedImageError("decoded image is empty")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _score_gray_crop(gray_patch, min_noise_size=MIN_NOISE_SIZE):
    """A "looks like one clean glyph" score for `gray_patch`, used only to
    arbitrate box splits (resolve_merge_or_split).

    Uses the RAW SVM top one-vs-rest margin, not the calibrated probability:
    a merged two-glyph blob falls between decision boundaries and gets a low
    top margin, whereas the isotonic calibration would report it at ~0.95 and
    wrongly keep the merge.
    """
    crop = _prepare_character_crop(gray_patch, min_noise_size=min_noise_size)
    if crop is None:
        return -1e9
    try:
        vec = _build_feature_vector(crop)
        return float(np.max(model.decision_function(vec)[0]))
    except Exception:
        _, conf, _ = classify_glyph(crop)
        return conf


def _titlecase_words(text):
    """"PuMuNTa AKKaNe" -> "Pumunta Akkane": first letter of each word upper,
    the rest lower. Baybayin class labels are per-syllable (Pu, Mu, Ta, Ng...),
    so a plain concatenation looks like camelCase - fix it at assembly time."""
    return " ".join(w[:1].upper() + w[1:].lower() for w in text.split())


def _encode_stage(img, max_w=1100, quality=72):
    """ndarray (grayscale or BGR) -> base64 JPEG string for the debug viewer,
    downscaled so the whole stage strip stays a sensible payload."""
    if img is None or getattr(img, "size", 0) == 0:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, max(1, int(h * max_w / w))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf).decode("ascii") if ok else None


def _montage(tiles, cols=10, cell=84, labels=None):
    """Grid of same-content tiles (each resized to `cell`) for the Visualize
    Process view - the per-glyph crops, HOG maps, predictions."""
    if not tiles:
        return None
    lab_h = 16 if labels else 0
    rows = (len(tiles) + cols - 1) // cols
    pad = 4
    W = cols * (cell + pad) + pad
    H = rows * (cell + lab_h + pad) + pad
    canvas = np.full((H, W, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        if t is None:
            continue
        if t.ndim == 2:
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        t = cv2.resize(t, (cell, cell), interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, cols)
        y0 = pad + r * (cell + lab_h + pad)
        x0 = pad + c * (cell + pad)
        canvas[y0:y0 + cell, x0:x0 + cell] = t
        if labels and i < len(labels) and labels[i]:
            cv2.putText(canvas, str(labels[i])[:12], (x0, y0 + cell + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 180), 1)
    return canvas


def _boxes_overlay(gray, box_groups):
    """gray frame -> BGR with boxes drawn. box_groups: list of (boxes, color)."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for boxes, color in box_groups:
        for (x, y, w, h) in boxes:
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 3)
    return vis


def preprocess_and_predict(image_bytes, session_id, white_paper=False,
                           pen_type="marker", visualize=False):
    img = _decode_image_bytes(image_bytes)  # raises UnsupportedImageError

    # White-paper mode: the background is guaranteed clean, so skip the steps
    # that trade detail for noise-robustness - no downscaling, no morphological
    # dilation (merge boxes by proximity instead), no remove_small_objects
    # (which eats thin-pen kudlit dots). See detect_character_boxes / _despeckle.
    pen_type = "pen" if str(pen_type).lower() == "pen" else "marker"
    if white_paper:
        mns = 0
    elif pen_type == "pen":
        mns = PEN_MIN_NOISE          # don't erase a thin-ballpen kudlit dot
    else:
        mns = MIN_NOISE_SIZE

    stage_imgs = {"0_raw": img.copy()}
    img, deskew_angle = deskew_image(img)
    stage_imgs["0b_deskewed"] = img.copy()
    boxes, gray, _, avg_height, avg_width = detect_character_boxes(
        img, white_paper=white_paper, pen_type=pen_type, stages=stage_imgs)
    proc_h, proc_w = gray.shape[:2]
    # base64 previews so the app can show WHAT THE COMPUTER SEES at each stage,
    # not the raw capture. The bbox coords in `detections` are in the
    # "2_flattened" frame's pixel space (same as processed_size).
    stages_b64 = {k: _encode_stage(stage_imgs[k]) for k in sorted(stage_imgs)}
    # Below ~70px/glyph in the 1600-wide working frame the whole pipeline frays
    # (phantom boxes, over-splits, kudlits gone). Warn the user to shoot closer.
    capture_warning = None
    if avg_width and avg_width < 70:
        capture_warning = (
            f"Small capture - glyphs are about {int(avg_width)}px wide. Hold the "
            "camera closer or crop tighter (aim for 90px+); segmentation and "
            "kudlit detection degrade sharply below ~70px.")
    quality_notes = [capture_warning] if capture_warning else []
    if abs(deskew_angle) > 12:
        quality_notes.append(
            f"Page is tilted about {abs(deskew_angle):.0f} deg - only 0.3-20 deg "
            "is corrected, and never per line. Keep the paper straight.")
    meta = {"processed_size": [int(proc_w), int(proc_h)],
            "deskew_angle": round(float(deskew_angle), 3),
            "white_paper": bool(white_paper),
            "pen_type": pen_type,
            "avg_glyph_px": [int(avg_width or 0), int(avg_height or 0)],
            "capture_warning": capture_warning,
            "quality_notes": quality_notes,
            "processed_b64": stages_b64.get("2_flattened"),
            "stages_b64": stages_b64}

    # Ordered "Visualize Process" strip: one card per pipeline step.
    viz = [] if visualize else None

    def _viz(part, title, caption, image):
        if viz is not None:
            viz.append({"part": part, "title": title, "caption": caption,
                        "img": _encode_stage(image, max_w=1000, quality=62)})

    _viz("PART 1 - find the glyphs", "1. normalize_image_size",
         f"Whole photo resized to a {proc_w}px working width so every "
         "ratio-based threshold below is predictable.", stage_imgs.get("1_normalized"))
    _viz("PART 1 - find the glyphs", "2. deskew_image",
         f"Rotated by {meta['deskew_angle']} deg (only corrects 0.3-20 deg).",
         stage_imgs.get("0b_deskewed"))
    _viz("PART 1 - find the glyphs", "3. flatten_background",
         "Divide by a blurred copy of itself: shadows / off-white paper gone. "
         "This is what Otsu thresholds; the boxes are measured on this frame.",
         stage_imgs.get("2_flattened"))
    _viz("PART 1 - find the glyphs", "4. Otsu threshold -> binary",
         "One black/white cutoff. This exact pixel set is what findContours traces.",
         stage_imgs.get("3_binary"))
    _viz("PART 1 - find the glyphs",
         "5. " + ("merge-by-proximity" if white_paper else "dilate"),
         ("White-paper mode: boxes grouped by gap, no pixels changed."
          if white_paper else
          f"Ink fattened by ~{int(avg_width * H_DILATE_RATIO)}x"
          f"{int(avg_height * V_DILATE_RATIO)}px so a glyph + its kudlit + its "
          "virama fuse into one blob."),
         stage_imgs.get("4_grouped"))
    _viz("PART 1 - find the glyphs", "6. findContours + size / border filters",
         f"{len(boxes)} boxes survive (avg glyph {avg_width}x{avg_height}px).",
         _boxes_overlay(gray, [(boxes, (0, 0, 230))]) if visualize else None)

    if not boxes:
        if viz is not None:
            meta["visualize_b64"] = viz
        return "No characters detected", 0.0, [], meta

    boxes_before_split = list(boxes)
    boxes = split_all_merged_boxes(
        boxes, gray, avg_width,
        score_fn=lambda p: _score_gray_crop(p, min_noise_size=mns))
    _viz("PART 1 - find the glyphs", "7. split_all_merged_boxes",
         f"Boxes wider than {SPLIT_WIDTH_RATIO}x avg are cut at an ink valley if "
         f"the classifier likes both halves. {len(boxes_before_split)} -> {len(boxes)}.",
         _boxes_overlay(gray, [(boxes, (0, 0, 230))]) if visualize else None)

    boxes = drop_stray_marks(boxes, avg_height, avg_width)
    _viz("PART 1 - find the glyphs", "8. drop_stray_marks",
         f"A detached kudlit / virama box is folded into its glyph. -> {len(boxes)} boxes.",
         _boxes_overlay(gray, [(boxes, (0, 150, 0))]) if visualize else None)

    lines = group_into_lines(boxes, avg_height)
    if visualize and lines:
        palette = [(0, 0, 230), (0, 150, 0), (200, 120, 0), (170, 0, 170),
                   (0, 160, 200), (120, 90, 0)]
        _viz("PART 1 - find the glyphs", "9. group_into_lines + insert_word_breaks",
             f"{len(lines)} line(s). A gap wider than {WORD_GAP_RATIO}x avg width "
             "becomes a space.",
             _boxes_overlay(gray, [(ln, palette[i % len(palette)])
                                   for i, ln in enumerate(lines)]))
    if not lines:
        if viz is not None:
            meta["visualize_b64"] = viz
        return "No characters detected", 0.0, [], meta

    # --- limitation checks: tell the user what to fix next time ---
    overlaps, wide = 0, 0
    issue_px = []
    for line in lines:
        for a, b in zip(line, line[1:]):
            if b[0] - (a[0] + a[2]) < -0.12 * avg_width:  # boxes clearly intersect
                overlaps += 1
                issue_px.append([int(a[0]), int(a[1]), int(b[0] + b[2] - a[0]),
                                 int(max(a[3], b[3]))])
        for (x, y, w, h) in line:
            if w > 1.9 * avg_width:                       # never got split
                wide += 1
                issue_px.append([int(x), int(y), int(w), int(h)])
    if overlaps:
        quality_notes.append(
            f"{overlaps} pair(s) of character boxes overlap - two glyphs were "
            "written too close and may be read as one. Leave a clear gap "
            "between every character.")
    if wide:
        quality_notes.append(
            f"{wide} box(es) are far wider than one glyph - characters are "
            "likely merged. Space them apart so each stands alone.")
    meta["quality_notes"] = quality_notes
    meta["issue_bbox_px"] = issue_px

    # Isotonic-calibrated proba is well-scaled but weakly discriminative, so a
    # low floor just drops the few genuinely hopeless glyphs. The raw softmax
    # fallback lives around ~0.63, so keep the old 0.23 floor when uncalibrated.
    CONF_LIMIT = 0.50 if calibrated_model is not None else 0.23
    WORD_GAP_THRESHOLD = max(20, int(avg_width * 1.5))
    full_sentence_text = []
    confidences = []
    detections = []

    session_temp_dir = os.path.join(TEMP_ROOT, f"session_{session_id}")
    os.makedirs(session_temp_dir, exist_ok=True)

    viz_tight = []      # raw ROI per glyph (PART 2 input)
    viz_crops = []      # final 64x64 per glyph (model input)
    viz_labels = []     # "<char> <conf>%"

    crop_index = 0
    for line_idx, line in enumerate(lines):
        line_chars = []
        tokens = insert_word_breaks(line, avg_width)
        pending_space = False
        for i, token in enumerate(tokens):
            if token is None:
                line_chars.append(" ")
                pending_space = True
                continue

            x, y, w, h = token
            roi_gray = gray[max(0, y):min(proc_h, y + h), max(0, x):min(proc_w, x + w)]
            if roi_gray.size == 0:
                continue

            ink_bounds = _tight_box_from_gray(roi_gray, min_noise_size=mns)
            if ink_bounds is None:
                continue

            ink_x, ink_y, ink_w, ink_h = ink_bounds
            tight_roi = roi_gray[ink_y:ink_y + ink_h, ink_x:ink_x + ink_w]
            if tight_roi.size == 0:
                continue

            img_final = _prepare_character_crop(tight_roi, min_noise_size=mns)
            if img_final is None:
                continue

            char, conf, alternatives = classify_glyph(img_final)
            char, mark_info = reconcile_mark(img_final, char, conf)

            if visualize:
                viz_tight.append(tight_roi.copy())
                viz_crops.append(img_final.copy())
                viz_labels.append(f"{char} {conf * 100:.0f}%")

            abs_x = int(x + ink_x)
            abs_y = int(y + ink_y)
            abs_w = int(max(1, ink_w))
            abs_h = int(max(1, ink_h))

            temp_filename = f"{crop_index}_{uuid.uuid4().hex[:8]}.jpg"
            temp_path = os.path.join(session_temp_dir, temp_filename)
            cv2.imwrite(temp_path, img_final)

            detections.append({
                "char": char,
                "confidence": round(conf * 100, 2),
                "is_eligible": conf >= CONF_LIMIT,
                "temp_path": temp_path,
                # top-3 [name, percent] guesses, best first; the UI lets the
                # user tap a runner-up when the top pick is wrong.
                "alternatives": [[c, round(s * 100, 2)] for c, s in alternatives],
                # mark-corrector trace (null unless mark_svm.pkl is loaded)
                "mark": mark_info,
                # layout so the client can rebuild the formatted text after edits
                "line": line_idx,
                "space_before": pending_space,
                # tighter box that follows the actual ink footprint of the glyph
                "bbox_px": [abs_x, abs_y, abs_w, abs_h],
                "bbox": [round(abs_x / proc_w, 5), round(abs_y / proc_h, 5),
                         round(abs_w / proc_w, 5), round(abs_h / proc_h, 5)],
            })
            pending_space = False
            confidences.append(conf)
            crop_index += 1

            if conf >= CONF_LIMIT:
                line_chars.append(char)

        line_text = _titlecase_words("".join(line_chars).strip())
        full_sentence_text.append(line_text)

    final_text = " | ".join(line for line in full_sentence_text if line)
    avg_conf = round(np.mean(confidences) * 100, 2) if confidences else 0.0

    if viz is not None:
        _viz("PART 2 - clean each box", "10. tight crop per glyph",
             "Each box re-cut to its exact ink footprint (in reading order).",
             _montage(viz_tight, cols=10, cell=90))
        _viz("PART 2 - clean each box",
             "11. " + ("despeckle skipped" if white_paper else
                       f"remove_small_objects (min {MIN_NOISE_SIZE}px)") +
             " -> pad -> resize 64x64",
             "The exact 64x64 grayscale images handed to the model. A kudlit dot "
             "is only a few pixels here.", _montage(viz_crops, cols=10, cell=64))
        try:
            hog_tiles = []
            for cr in viz_crops:
                _, hi = hog(cr, orientations=HOG_ORIENTATIONS,
                            pixels_per_cell=HOG_PIXELS_PER_CELL,
                            cells_per_block=HOG_CELLS_PER_BLOCK,
                            block_norm=HOG_BLOCK_NORM, visualize=True)
                hog_tiles.append(cv2.normalize(hi, None, 0, 255, cv2.NORM_MINMAX)
                                 .astype(np.uint8))
            _viz("PART 3 - classify", "12. HOG features (1764 numbers per glyph)",
                 "The edge-direction map the SVM actually sees - stroke shape is "
                 "vivid, a small dot barely registers.",
                 _montage(hog_tiles, cols=10, cell=64))
        except Exception:
            pass
        _viz("PART 3 - classify", "13. model prediction",
             f"HOG + 26 spatial features -> SVM -> 1 of 95 classes. "
             f"Result: \"{final_text}\"  (avg {avg_conf}%).",
             _montage(viz_crops, cols=10, cell=90, labels=viz_labels))
        meta["visualize_b64"] = viz

    return final_text, avg_conf, detections, meta

# --- 5. API ROUTES ---

@app.route('/api/translate', methods=['POST'])
def translate():
    session_id = start_processing_session(request.remote_addr)
    mode = request.form.get('mode') if 'mode' in request.form else request.json.get('mode')

    try:
        if mode == 'Baybayin to Tagalog':
            if model is None or hog_scaler is None or spatial_scaler is None or not class_names:
                update_session_status(session_id, 'Model_Unavailable')
                return jsonify({
                    "error": "Baybayin-to-Tagalog model files are missing or failed to load on the server."
                }), 503

            if 'file' not in request.files:
                update_session_status(session_id, 'No_File')
                return jsonify({"error": "No image uploaded"}), 400
            
            image_bytes = request.files['file'].read()
            white_paper = str(request.form.get('white_paper', '')).strip().lower() \
                in ('1', 'true', 'yes', 'on')
            visualize = str(request.form.get('visualize', '')).strip().lower() \
                in ('1', 'true', 'yes', 'on')
            pen_type = str(request.form.get('pen_type', 'marker')).strip().lower()
            try:
                text, conf, results, meta = preprocess_and_predict(
                    image_bytes, session_id, white_paper=white_paper,
                    pen_type=pen_type, visualize=visualize)
            except UnsupportedImageError as exc:
                update_session_status(session_id, 'Unsupported_Format')
                return jsonify({
                    "error": "Unsupported or unreadable image. Please upload a JPEG, PNG, "
                             "WEBP, BMP, TIFF"
                             + (", HEIC" if _HEIF_SUPPORT else "")
                             + " image.",
                    "detail": str(exc),
                    "session_id": session_id,
                }), 415

            log_detections(session_id, results)
            status = "Success" if conf > 60 else "Low_Confidence"
            update_session_status(session_id, status)

            return jsonify({
                "translated_text": text,
                "confidence": conf,
                "status": status,
                "individual_detections": results,
                "session_id": session_id,
                # size of the image the bbox coords in individual_detections
                # refer to (deskewed, width-normalised), plus the applied skew.
                "processed_size": meta["processed_size"],
                "deskew_angle": meta["deskew_angle"],
                "white_paper": meta.get("white_paper", False),
                "pen_type": meta.get("pen_type", "marker"),
                "avg_glyph_px": meta.get("avg_glyph_px"),
                "capture_warning": meta.get("capture_warning"),
                "quality_notes": meta.get("quality_notes", []),
                "issue_bbox_px": meta.get("issue_bbox_px", []),
                # what the computer actually sees, stage by stage (base64 JPEG)
                "processed_b64": meta.get("processed_b64"),
                "stages_b64": meta.get("stages_b64", {}),
                # full ordered pipeline walkthrough (only when visualize=1)
                "visualize_b64": meta.get("visualize_b64", []),
            })

        elif mode == 'Tagalog to Baybayin':
            input_text = request.form.get('text') if 'text' in request.form else request.json.get('text')
            if not input_text:
                update_session_status(session_id, 'No_Text')
                return jsonify({"error": "No text provided"}), 400
            
            translated_result, confidence = ttb_translator.translate(input_text)
            update_session_status(session_id, "Success")
            
            return jsonify({
                "translated_text": translated_result,
                "confidence": confidence,
                "session_id": session_id
            })

    except Exception as e:
        update_session_status(session_id, 'Error')
        return jsonify({"error": str(e)}), 500

@app.route('/api/archive_bulk', methods=['POST'])
def archive_bulk():
    data = request.json
    session_id = data.get('session_id')
    detections = data.get('detections', [])

    if not detections:
        return jsonify({"status": "Ignored", "message": "No detections to archive"}), 200

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        saved_count = 0
        archive_data = []

        for d in detections:
            char = d.get('char')
            confidence = d.get('confidence')
            temp_path = d.get('temp_path')
            is_eligible = d.get('is_eligible', False)

            if not char or not temp_path or not os.path.exists(temp_path) or not is_eligible:
                continue

            char_dir = os.path.join(ARCHIVE_ROOT, char)
            os.makedirs(char_dir, exist_ok=True)

            final_filename = f"sess{session_id}_{uuid.uuid4().hex[:8]}.jpg"
            final_path = os.path.join(char_dir, final_filename)

            os.rename(temp_path, final_path)
            archive_data.append((session_id, char, confidence, True))
            saved_count += 1

        if archive_data:
            query = """
                INSERT INTO open_archival 
                (session_id, char_label, confidence_score, verified_by_user) 
                VALUES (%s, %s, %s, %s)
            """
            cursor.executemany(query, archive_data)
            conn.commit()

        # Cleanup
        session_temp_dir = os.path.join(TEMP_ROOT, f"session_{session_id}")
        if os.path.exists(session_temp_dir):
            for file in os.listdir(session_temp_dir):
                os.remove(os.path.join(session_temp_dir, file))
            os.rmdir(session_temp_dir)

        return jsonify({"status": "Success", "message": f"Archived {saved_count} entries"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)