"""
REFERENCE COPY - the Colab paragraph segmenter that backend/app.py's
segmentation is derived from. Not imported by the server (matplotlib
visualization, Colab paths).

backend/app.py contains ports of every function here (deskew_image,
measure_average_char_size, detect_character_boxes, split_merged_box,
resolve_merge_or_split, split_all_merged_boxes, group_into_lines,
insert_word_breaks) plus additions not in this file: drop_stray_marks,
_titlecase_words, per-glyph bbox in the API response, and a raw-margin
(not calibrated-probability) score for resolve_merge_or_split.

The CONFIG constants in app.py were re-tuned against real handwriting and
differ from the values below - see the comment block above the constants in
app.py. Connected cursive is near the ceiling of this morphology approach
regardless of the constants.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
from PIL import Image


# ============================================================
# CONFIG
# ============================================================
STANDARD_WIDTH = 1600
MIN_CHAR_AREA_RATIO = 0.0003
MIN_BOX_HEIGHT_RATIO = 0.30
MIN_BOX_WIDTH_RATIO = 0.20
MIN_BOX_AREA_RATIO = 0.25
EDGE_MARGIN_PX = 10
V_DILATE_RATIO = 0.40
H_DILATE_RATIO = 0.24
V_DILATE_FALLBACK = 45
H_DILATE_FALLBACK = 12
ROW_GROUPING_RATIO = 0.6
WORD_GAP_RATIO = 1.5
MIN_COMPONENT_HEIGHT = 15
SPLIT_WIDTH_RATIO = 1.6
SPLIT_SEARCH_WINDOW = 0.35
SPLIT_MIN_GAP_RATIO = 0.30
MIN_SEGMENT_WIDTH_RATIO = 0.55
BG_BLUR_KERNEL = 101


def load_image_robust(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return img
    try:
        pil_img = Image.open(image_path).convert("RGB")
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[load error] Could not read {image_path}: {e}")
        return None


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
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def measure_average_char_size(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    light_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(thresh, light_kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)
               if stats[i, cv2.CC_STAT_HEIGHT] > MIN_COMPONENT_HEIGHT]
    widths = [stats[i, cv2.CC_STAT_WIDTH] for i in range(1, num_labels)
              if stats[i, cv2.CC_STAT_HEIGHT] > MIN_COMPONENT_HEIGHT]
    if heights and widths:
        return int(np.median(heights)), int(np.median(widths))
    return None, None


def detect_character_boxes(img, edge_margin=EDGE_MARGIN_PX):
    img = normalize_image_size(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    avg_height, avg_width = measure_average_char_size(gray)
    if avg_height and avg_width:
        v_dilate = max(15, int(avg_height * V_DILATE_RATIO))
        h_dilate = max(5, int(avg_width * H_DILATE_RATIO))
    else:
        v_dilate, h_dilate = V_DILATE_FALLBACK, H_DILATE_FALLBACK
        avg_height, avg_width = 80, 60
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
        if h < min_box_height or w < min_box_width or (w * h) < min_box_area:
            continue
        if (x <= edge_margin or y <= edge_margin or
                x + w >= w_img - edge_margin or y + h >= h_img - edge_margin):
            continue
        boxes.append((x, y, w, h))
    return boxes, gray, img, avg_height, avg_width


# include dilation here


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
        best_offset = int(np.argmin(col_density[lo:hi]))
        candidate_x = lo + best_offset
        if col_density[candidate_x] < min_gap_ratio * peak_density:
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
    return sub_boxes or [box]


def resolve_merge_or_split(box, candidate_pieces, gray, pipeline, predict_fn):
    if len(candidate_pieces) <= 1:
        return [box]
    x, y, w, h = box
    whole = predict_fn(gray[y:y+h, x:x+w], pipeline, top_k=1,
                       already_preprocessed=False, verbose=False)
    whole_score = whole["top_k"][0][1] if "top_k" in whole else -999
    sub_scores = []
    for (sx, sy, sw, sh) in candidate_pieces:
        r = predict_fn(gray[sy:sy+sh, sx:sx+sw], pipeline, top_k=1,
                       already_preprocessed=False, verbose=False)
        sub_scores.append(r["top_k"][0][1] if "top_k" in r else -999)
    split_confidence = min(sub_scores) if sub_scores else -999
    return [box] if whole_score >= split_confidence else candidate_pieces


def split_all_merged_boxes(boxes, gray, avg_width, pipeline=None, predict_fn=None):
    result = []
    for box in boxes:
        pieces = split_merged_box(box, gray, avg_width)
        if len(pieces) <= 1:
            result.extend(pieces)
            continue
        if pipeline is not None and predict_fn is not None:
            result.extend(resolve_merge_or_split(box, pieces, gray, pipeline, predict_fn))
        else:
            result.extend(pieces)
    return result


def group_into_lines(boxes, avg_height, row_ratio=ROW_GROUPING_RATIO):
    if not boxes:
        return []
    row_threshold = avg_height * row_ratio
    lines = []
    for box in sorted(boxes, key=lambda b: b[1]):
        placed = False
        for line in lines:
            avg_y = np.mean([b[1] + b[3] / 2 for b in line])
            if abs(avg_y - (box[1] + box[3] / 2)) < row_threshold:
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
        if line_boxes[i][0] - prev_right > gap_threshold:
            result.append(None)
        result.append(line_boxes[i])
    return result


def segment_paragraph(image_path, pipeline=None, predict_fn=None, visualize=True):
    img = load_image_robust(image_path)
    if img is None:
        return None, None
    img = normalize_image_size(img)
    img, angle = deskew_image(img)
    boxes, gray, img_final, avg_height, avg_width = detect_character_boxes(img)
    if not boxes:
        return None, None
    boxes = split_all_merged_boxes(boxes, gray, avg_width, pipeline=pipeline, predict_fn=predict_fn)
    lines = group_into_lines(boxes, avg_height)
    if visualize:
        vis = img_final.copy()
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
        counter = 0
        for li, line in enumerate(lines):
            for (x, y, w, h) in line:
                cv2.rectangle(vis, (x, y), (x + w, y + h), colors[li % len(colors)], 2)
                counter += 1
        plt.figure(figsize=(14, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    return lines, gray
