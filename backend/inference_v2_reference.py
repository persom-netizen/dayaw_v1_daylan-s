"""
!!! STALE - FROZEN AT THE 26-FEATURE V2/V3 PIPELINE. DO NOT USE AS A REFERENCE.
The live spec is backend/training/train_weighted_model.py (kudlit features,
--kudlit-augment, pen/marker thicken, etc). backend/app.py must match THAT
file, not this one. Kept only for the original V2 provenance / regression
checks against the very first model. Delete once V3 is retired.
--------------------------------------------------------------------------------
Baybayin BTL inference (v2) - single character prediction using the
final, stronger model (91.14% test accuracy).

Config: 64x64 preprocessing, HOG(9,8x8,2x2)=1764 dims + 26 spatial dims,
spatial weight=6x, SVM(C=20, gamma='scale', class_weight='balanced').

This is the function smart_segment.py's predict_text() calls per detected
character crop - passing already_preprocessed=False for raw photo crops.

REFERENCE COPY. This is the Colab pipeline that backend/app.py's
_prepare_character_crop / _extract_spatial_features / _build_feature_vector /
classify_glyph are ported from. Kept here for provenance and regression
checks. It is NOT imported by the running server (the MODEL_DIR /
HOG_FEATURES_DIR paths below are Colab Drive paths).
"""

import os
import cv2
import numpy as np

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
from PIL import Image
from skimage.feature import hog
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import joblib


# ============================================================
# CONFIG - must match V2 training exactly
# ============================================================
MODEL_DIR = "/content/drive/MyDrive/WEIGHTED_MODEL_V2"
HOG_FEATURES_DIR = "/content/drive/MyDrive/HOG_FEATURES_V2"

TARGET_SIZE = 64
MIN_NOISE_SIZE = 20
PAD_RATIO = 0.12

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
N_HOG_FEATURES = 1764


# ============================================================
# LOAD MODEL + SCALERS
# ============================================================
def load_pipeline_v2(model_dir=MODEL_DIR, hog_features_dir=HOG_FEATURES_DIR):
    model = joblib.load(os.path.join(model_dir, "weighted_svm.pkl"))
    hog_scaler = joblib.load(os.path.join(model_dir, "hog_scaler.pkl"))
    spatial_scaler = joblib.load(os.path.join(model_dir, "spatial_scaler.pkl"))
    weight = joblib.load(os.path.join(model_dir, "best_weight.pkl"))
    label_encoder = joblib.load(os.path.join(hog_features_dir, "label_encoder.pkl"))
    return {
        "model": model, "hog_scaler": hog_scaler, "spatial_scaler": spatial_scaler,
        "weight": weight, "label_encoder": label_encoder,
    }


# ============================================================
# IMAGE LOADING
# ============================================================
def load_grayscale_robust(image_path_or_array, verbose=True):
    if isinstance(image_path_or_array, np.ndarray):
        img = image_path_or_array
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    path = image_path_or_array
    if not os.path.exists(path):
        if verbose:
            print(f"  [load error] Path does not exist: {path}")
        return None

    img = cv2.imread(path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if not HEIC_SUPPORT and path.lower().endswith(('.heic', '.heif')):
        if verbose:
            print(f"  [load error] HEIC file but pillow_heif not active this session.")
        return None

    try:
        pil_img = Image.open(path).convert("RGB")
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        if verbose:
            print(f"  [load error] {repr(e)}")
        return None


# ============================================================
# PREPROCESSING (matches preprocess_v2.py exactly)
# ============================================================
def preprocess_image(gray, target_size=TARGET_SIZE, min_noise_size=MIN_NOISE_SIZE,
                       pad_ratio=PAD_RATIO):
    if gray is None or gray.size == 0:
        return None

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_bool = binary > 0
    cleaned_bool = remove_small_objects(binary_bool, min_size=min_noise_size)
    cleaned = (cleaned_bool * 255).astype(np.uint8)

    if cleaned.sum() == 0:
        return None

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w < 3 or h < 3:
        return None

    tight = cleaned[y:y+h, x:x+w]
    side = max(w, h)
    pad = int(side * pad_ratio)
    canvas_side = side + 2 * pad
    canvas = np.zeros((canvas_side, canvas_side), dtype=np.uint8)
    y_off, x_off = (canvas_side - h) // 2, (canvas_side - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = tight

    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)


# ============================================================
# FEATURE EXTRACTION (matches hog_extract_v2.py exactly)
# ============================================================
def grid_density_features(binary_img, grid_size):
    h, w = binary_img.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    densities = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = gy * cell_h, (gy+1)*cell_h if gy < grid_size-1 else h
            x0, x1 = gx * cell_w, (gx+1)*cell_w if gx < grid_size-1 else w
            cell = binary_img[y0:y1, x0:x1]
            densities.append(cell.sum() / (cell.size * 255) if cell.size > 0 else 0)
    return densities


def kudlit_component_features(binary_img):
    binary_bool = binary_img > 0
    labeled = label(binary_bool)
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


def extract_spatial_features(gray_img):
    _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    overall_density = [binary.sum() / (binary.size * 255)]
    quadrant = grid_density_features(binary, grid_size=2)
    fine_grid = grid_density_features(binary, grid_size=4)
    kudlit_feats = kudlit_component_features(binary)
    return overall_density + quadrant + fine_grid + kudlit_feats


def extract_combined_features(preprocessed_img):
    hog_features = hog(
        preprocessed_img, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK, block_norm=HOG_BLOCK_NORM, feature_vector=True,
    )
    spatial_features = extract_spatial_features(preprocessed_img)
    return np.concatenate([hog_features, spatial_features])


# ============================================================
# PREDICT
# ============================================================
def predict_character(image_path_or_array, pipeline, top_k=3, already_preprocessed=False, verbose=True):
    """
    already_preprocessed=True: input is already a clean 64x64 image (e.g.
    from PREPROCESSED_DATASET_V2 or paths_test).
    already_preprocessed=False (default): raw crop from a photo/segmentation
    - runs full preprocessing. THIS is what smart_segment.py should use.
    """
    gray = load_grayscale_robust(image_path_or_array, verbose=verbose)
    if gray is None:
        return {"error": "could not read image"}

    if already_preprocessed:
        preprocessed = gray
    else:
        preprocessed = preprocess_image(gray)
        if preprocessed is None:
            return {"error": "no ink content found after preprocessing"}

    features = extract_combined_features(preprocessed)
    features_hog = features[:N_HOG_FEATURES].reshape(1, -1)
    features_spatial = features[N_HOG_FEATURES:].reshape(1, -1)

    weight = pipeline["weight"]
    features_hog_scaled = pipeline["hog_scaler"].transform(features_hog)
    features_spatial_scaled = pipeline["spatial_scaler"].transform(features_spatial) * weight
    features_combined = np.hstack([features_hog_scaled, features_spatial_scaled])

    model = pipeline["model"]
    label_encoder = pipeline["label_encoder"]

    pred_idx = model.predict(features_combined)[0]
    pred_class = label_encoder.inverse_transform([pred_idx])[0]

    result = {"prediction": pred_class, "preprocessed_image": preprocessed}

    if hasattr(model, "decision_function"):
        scores = model.decision_function(features_combined)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        result["top_k"] = [
            (label_encoder.inverse_transform([idx])[0], float(scores[idx]))
            for idx in top_indices
        ]

    return result


# ============================================================
# USAGE
# ============================================================
if __name__ == "__main__":
    print(f"HEIC support active: {HEIC_SUPPORT}")
    pipeline = load_pipeline_v2()
    print(f"Pipeline loaded. Spatial weight: {pipeline['weight']}")
    print("\nReady to be used by smart_segment.py's predict_text() for full")
    print("paragraph reading, or called directly for single-character tests.")
