import cv2
import numpy as np
import joblib
import mysql.connector
import os
import uuid
import re
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
# Feature vector = 1764 HOG features ++ 26 spatial features (1790 total).
# The two blocks are scaled by SEPARATE scalers; the spatial block is then
# multiplied by SPATIAL_WEIGHT before the blocks are concatenated for the SVM.
HOG_FEATURE_LEN = 1764
SPATIAL_FEATURE_LEN = 26

model = load_joblib_artifact('weighted_svm.pkl', 'svm_model.pkl',
                             'baybayin_svm_model.pkl', 'baybayin_svm_model.sav')
hog_scaler = load_joblib_artifact('hog_scaler.pkl', 'baybayin_scaler.pkl', 'baybayin_scaler.sav')
spatial_scaler = load_joblib_artifact('spatial_scaler.pkl')
_raw_weight = load_joblib_artifact('best_weight.pkl', 'spatial_weight.pkl')
SPATIAL_WEIGHT = float(_raw_weight) if _raw_weight is not None else 6.0
label_encoder = load_joblib_artifact('label_encoder.pkl', 'baybayin_classes.pkl', 'baybayin_classes.sav')

# Isotonic probability calibrator wrapping `model` (built by calibrate_model.py).
# When present, classify_glyph() takes both the label and the confidence from
# its predict_proba(); otherwise it falls back to a softmax over the raw SVM
# decision_function, which is poorly calibrated (see calibrate_model.py docstring).
calibrated_model = load_joblib_artifact('weighted_svm_calibrated.pkl')

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

# Adaptive segmentation constants for multi-line Baybayin paragraphs
# First tuning pass: slightly more tolerant of genuine handwriting while
# reducing over-merging and false splits on paragraph-style input.
STANDARD_WIDTH = 1600
MIN_CHAR_AREA_RATIO = 0.0003
MIN_BOX_HEIGHT_RATIO = 0.28
MIN_BOX_WIDTH_RATIO = 0.18
MIN_BOX_AREA_RATIO = 0.22
EDGE_MARGIN_PX = 10
V_DILATE_RATIO = 0.35   
H_DILATE_RATIO = 0.20
V_DILATE_FALLBACK = 45
H_DILATE_FALLBACK = 12
ROW_GROUPING_RATIO = 0.7
# Word break when the gap to the next glyph exceeds this fraction of the mean
# glyph width. Handwriting packs glyphs within a word almost edge-to-edge
# (~0-0.1 w) while word gaps run ~0.8w+, so 0.75 separates them cleanly;
# the old 1.2 missed loosely-spaced words.
WORD_GAP_RATIO = 0.75
MIN_COMPONENT_HEIGHT = 15
SPLIT_WIDTH_RATIO = 1.8
SPLIT_SEARCH_WINDOW = 0.35
SPLIT_MIN_GAP_RATIO = 0.35
MIN_SEGMENT_WIDTH_RATIO = 0.55
BG_BLUR_KERNEL = 101


def normalize_image_size(img, standard_width=STANDARD_WIDTH):
    h, w = img.shape[:2]
    scale = standard_width / w
    new_h = int(h * scale)
    return cv2.resize(img, (standard_width, new_h), interpolation=cv2.INTER_AREA)


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


def detect_character_boxes(img, edge_margin=EDGE_MARGIN_PX):
    img = normalize_image_size(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_dilate, v_dilate))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    min_area = MIN_CHAR_AREA_RATIO * h_img * w_img
    min_box_height = avg_height * MIN_BOX_HEIGHT_RATIO
    min_box_width = avg_width * MIN_BOX_WIDTH_RATIO
    min_box_area = (avg_width * avg_height) * MIN_BOX_AREA_RATIO

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        touches_border = (
            x <= edge_margin or y <= edge_margin or
            x + w >= w_img - edge_margin or y + h >= h_img - edge_margin
        )
        if touches_border:
            continue

        boxes.append((x, y, w, h))

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
    """Remove boxes that are almost certainly diacritics, not base glyphs -
    the virama "krus" (x) and e/i/o/u kudlit dots. They are far smaller than a
    real glyph; keeping them adds phantom characters (and sometimes a phantom
    one-glyph line). Conservative: only drops clear area+size outliers."""
    if len(boxes) < 4:
        return boxes
    areas = sorted(w * h for (_, _, w, h) in boxes)
    median_area = areas[len(areas) // 2]
    small_side = 0.66 * min(avg_width, avg_height)
    kept = [
        (x, y, w, h) for (x, y, w, h) in boxes
        if not (w * h < 0.25 * median_area and min(w, h) < small_side)
    ]
    return kept if kept else boxes


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

    lines.sort(key=lambda line: np.mean([b[1] for b in line]))
    for line in lines:
        line.sort(key=lambda b: b[0])

    return lines


def insert_word_breaks(line_boxes, avg_width, word_gap_ratio=WORD_GAP_RATIO):
    if len(line_boxes) < 2:
        return line_boxes

    gap_threshold = avg_width * word_gap_ratio
    result = [line_boxes[0]]
    for i in range(1, len(line_boxes)):
        prev_right = line_boxes[i - 1][0] + line_boxes[i - 1][2]
        gap = line_boxes[i][0] - prev_right
        if gap > gap_threshold:
            result.append(None)
        result.append(line_boxes[i])
    return result


# --- WEIGHTED MODEL v2: preprocessing + feature extraction ---
# Direct port of the Colab reference (inference.py -> preprocess_v2 /
# hog_extract_v2). Every step here must match training exactly; the README
# and reference both warn that a mismatch silently wrecks accuracy.

TARGET_SIZE = 64
MIN_NOISE_SIZE = 20
PAD_RATIO = 0.12
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'


def _tight_box_from_gray(gray_patch):
    """Return the tight ink-bounds inside a grayscale patch, in patch-local coords."""
    if gray_patch is None or gray_patch.size == 0:
        return None

    _, binary = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned_bool = remove_small_objects(binary > 0, min_size=MIN_NOISE_SIZE)
    cleaned = (cleaned_bool * 255).astype(np.uint8)
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
    cleaned_bool = remove_small_objects(binary > 0, min_size=min_noise_size)
    cleaned = (cleaned_bool * 255).astype(np.uint8)
    if cleaned.sum() == 0:
        return None

    tight = _tight_box_from_gray(gray_patch)
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


def _extract_spatial_features(preprocessed_img):
    """Port of inference.extract_spatial_features (26 features): overall
    density (1) + 2x2 grid (4) + 4x4 grid (16) + kudlit stats (5)."""
    _, binary = cv2.threshold(preprocessed_img, 127, 255, cv2.THRESH_BINARY)
    overall_density = [binary.sum() / (binary.size * 255)]
    quadrant = _grid_density_features(binary, grid_size=2)
    fine_grid = _grid_density_features(binary, grid_size=4)
    kudlit_feats = _kudlit_component_features(binary)
    return overall_density + quadrant + fine_grid + kudlit_feats


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
    features_spatial = spatial_features.reshape(1, -1)

    hog_scaled = hog_scaler.transform(features_hog)
    spatial_scaled = spatial_scaler.transform(features_spatial) * SPATIAL_WEIGHT
    return np.hstack([hog_scaled, spatial_scaled])


def _decode_label(pred_int):
    if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
        return str(label_encoder.inverse_transform([int(pred_int)])[0])
    if 0 <= int(pred_int) < len(class_names):
        return class_names[int(pred_int)]
    return str(pred_int)


def classify_glyph(img64):
    """Return (glyph_name, confidence in [0, 1]) for a preprocessed 64x64 glyph.

    Feature extraction + label decoding match inference.predict_character exactly.
    Confidence:
      * if weighted_svm_calibrated.pkl is loaded, both the label and the
        confidence come from its isotonic-calibrated predict_proba() - an
        honestly scaled probability (see calibrate_model.py).
      * otherwise, fall back to argmax of the raw SVM plus a softmax over its
        one-vs-rest decision_function margins. That proxy is poorly calibrated
        (test-set AUROC ~0.49 for right-vs-wrong) and is only a placeholder
        until the calibrator is built.
    """
    vec = _build_feature_vector(img64)

    if calibrated_model is not None:
        proba = calibrated_model.predict_proba(vec)[0]
        best = int(np.argmax(proba))
        char = _decode_label(calibrated_model.classes_[best])
        return char, float(proba[best])

    char = _decode_label(model.predict(vec)[0])
    try:
        margins = model.decision_function(vec)[0]
        exp = np.exp(margins - np.max(margins))
        conf = float(np.max(exp / exp.sum()))
    except Exception:
        conf = 1.0
    return char, conf


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


def _score_gray_crop(gray_patch):
    """Confidence in [0, 1] that `gray_patch` is a single clean glyph.
    Used to validate geometric box splits (see resolve_merge_or_split)."""
    crop = _prepare_character_crop(gray_patch)
    if crop is None:
        return -1.0
    _, conf = classify_glyph(crop)
    return conf


def _titlecase_words(text):
    """"PuMuNTa AKKaNe" -> "Pumunta Akkane": first letter of each word upper,
    the rest lower. Baybayin class labels are per-syllable (Pu, Mu, Ta, Ng...),
    so a plain concatenation looks like camelCase - fix it at assembly time."""
    return " ".join(w[:1].upper() + w[1:].lower() for w in text.split())


def preprocess_and_predict(image_bytes, session_id):
    img = _decode_image_bytes(image_bytes)  # raises UnsupportedImageError

    img, deskew_angle = deskew_image(img)
    boxes, gray, _, avg_height, avg_width = detect_character_boxes(img)
    proc_h, proc_w = gray.shape[:2]
    meta = {"processed_size": [int(proc_w), int(proc_h)],
            "deskew_angle": round(float(deskew_angle), 3)}

    if not boxes:
        return "No characters detected", 0.0, [], meta

    boxes = split_all_merged_boxes(boxes, gray, avg_width, score_fn=_score_gray_crop)
    boxes = drop_stray_marks(boxes, avg_height, avg_width)
    lines = group_into_lines(boxes, avg_height)
    if not lines:
        return "No characters detected", 0.0, [], meta

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

    crop_index = 0
    for line in lines:
        line_chars = []
        tokens = insert_word_breaks(line, avg_width)
        for i, token in enumerate(tokens):
            if token is None:
                line_chars.append(" ")
                continue

            x, y, w, h = token
            roi_gray = gray[max(0, y):min(proc_h, y + h), max(0, x):min(proc_w, x + w)]
            if roi_gray.size == 0:
                continue

            ink_bounds = _tight_box_from_gray(roi_gray)
            if ink_bounds is None:
                continue

            ink_x, ink_y, ink_w, ink_h = ink_bounds
            tight_roi = roi_gray[ink_y:ink_y + ink_h, ink_x:ink_x + ink_w]
            if tight_roi.size == 0:
                continue

            img_final = _prepare_character_crop(tight_roi)
            if img_final is None:
                continue

            char, conf = classify_glyph(img_final)

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
                # tighter box that follows the actual ink footprint of the glyph
                "bbox_px": [abs_x, abs_y, abs_w, abs_h],
                "bbox": [round(abs_x / proc_w, 5), round(abs_y / proc_h, 5),
                         round(abs_w / proc_w, 5), round(abs_h / proc_h, 5)],
            })
            confidences.append(conf)
            crop_index += 1

            if conf >= CONF_LIMIT:
                line_chars.append(char)

        line_text = _titlecase_words("".join(line_chars).strip())
        full_sentence_text.append(line_text)

    final_text = " | ".join(line for line in full_sentence_text if line)
    avg_conf = round(np.mean(confidences) * 100, 2) if confidences else 0.0
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
            try:
                text, conf, results, meta = preprocess_and_predict(image_bytes, session_id)
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