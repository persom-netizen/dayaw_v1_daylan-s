#!/usr/bin/env python3
r"""
Train the weighted HOG + spatial SVM for DAYAW Baybayin OCR, end to end, from a
folder of harvested character images.

This reproduces the exact pipeline that backend/app.py and
backend/inference_v2_reference.py expect, so the artifacts it writes drop
straight into backend/ with NO code changes (as long as the CONFIG block below
is left at its defaults).

--------------------------------------------------------------------------------
DATASET LAYOUT (folder per class, class name = folder name):

    ALL_DATASET/
        A/     img001.png img002.png ...
        Ba/    ...
        Be/    ...
        ...
        Nga/   ...

Class names become the labels. LabelEncoder sorts them, so the integer order is
deterministic and matches label_encoder.pkl.

--------------------------------------------------------------------------------
QUICK START - see backend/training/README.md for the full Colab cells.

  CLI:
    python train_weighted_model.py --data ALL_DATASET --out WEIGHTED_MODEL_V3

  Notebook (paste this whole file into a cell, then in the NEXT cell):
    run(data="/content/drive/MyDrive/ALL_DATASET",
        out="/content/drive/MyDrive/WEIGHTED_MODEL_V3")
    # optional: augment=2, weight_grid="2,4,6,8,10,12,15,20,25", search_c="10,20,50"

--------------------------------------------------------------------------------
OUTPUTS (in --out):

  drop into backend/ :
    weighted_svm.pkl              SVC(C, gamma='scale', class_weight='balanced')
    weighted_svm_calibrated.pkl   + isotonic CalibratedClassifierCV (confidence)
    hog_scaler.pkl                StandardScaler on the 1764 HOG features
    spatial_scaler.pkl            StandardScaler on the 26 spatial features
    best_weight.pkl               int, spatial-block multiplier
    label_encoder.pkl            LabelEncoder (int <-> class name)

  drop into backend/tests/reports/ :
    test_predictions.npy  test_true_labels.npy  test_confusion_matrix.npy

  keep on Drive (lets you re-run backend/calibrate_model.py locally, and do
  error analysis) :
    X.npy  y.npy  splits/{X,y}_{train,val,test}.npy  splits/paths_*.npy
    metrics.json   MANIFEST.json   MODEL_README.txt
"""

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np

# skimage >= 0.26 renamed remove_small_objects(min_size=...) and nags about it.
# backend/requirements.txt pins scikit-image < 0.26 so the semantics match
# inference; silence the noise here in case a newer skimage sneaks in.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from skimage.feature import hog
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ============================================================================
# CONFIG. backend/app.py reads the glyph size + feature lengths back from the
# scalers this script produces, so --target-size drops in with no app.py edit.
# The HOG params and the 26 spatial features must NOT change without matching
# edits in app.py's _build_feature_vector / _extract_spatial_features.
# ============================================================================
TARGET_SIZE = 64           # overridden by --target-size (64 or 96)
MIN_NOISE_SIZE = 20
PAD_RATIO = 0.12

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
# 26 = the Colab port; +2 = top/bottom kudlit-strip ink fractions;
# +4 = top/bottom kudlit mark shape (aspect + relative width) - the dash-vs-dot
# signal for Ne/Ni, Nu/No. See extract_spatial_features / app.py. app.py reads
# this length back from the spatial scaler, so 28- or 32-feature models drop in
# unchanged.
N_SPATIAL_FEATURES = 32


def _hog_len(target_size):
    cells = target_size // HOG_PIXELS_PER_CELL[0]
    blocks = cells - (HOG_CELLS_PER_BLOCK[0] - 1)
    return blocks * blocks * (HOG_CELLS_PER_BLOCK[0] ** 2) * HOG_ORIENTATIONS


N_HOG_FEATURES = _hog_len(TARGET_SIZE)          # 64 -> 1764, 96 -> 4356
N_FEATURES = N_HOG_FEATURES + N_SPATIAL_FEATURES

SEED = 42
TEST_FRAC = 0.15          # fraction of everything held out for the test report
VAL_FRAC = 0.15           # fraction of everything used to fit scalers search + calibrator
SVC_C_DEFAULT = 20.0
SVC_GAMMA = "scale"
DEFAULT_WEIGHT_GRID = [1, 2, 3, 4, 6, 8, 10, 12, 15]
WEIGHT_SEARCH_SUBSAMPLE = 8000   # stratified subsample size for the weight/C search
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================================
# PREPROCESSING + FEATURES
# Copied verbatim from backend/inference_v2_reference.py. backend/app.py is a
# byte-identical port of these (verified: reproduces test_predictions.npy on
# all 4560 old test samples). DO NOT edit one without the others.
# ============================================================================
def preprocess_image(gray, target_size=None, min_noise_size=MIN_NOISE_SIZE,
                     pad_ratio=PAD_RATIO):
    if target_size is None:
        target_size = TARGET_SIZE  # module global, may be set by --target-size
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

    tight = cleaned[y:y + h, x:x + w]
    side = max(w, h)
    pad = int(side * pad_ratio)
    canvas_side = side + 2 * pad
    canvas = np.zeros((canvas_side, canvas_side), dtype=np.uint8)
    y_off, x_off = (canvas_side - h) // 2, (canvas_side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = tight
    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)


def grid_density_features(binary_img, grid_size):
    h, w = binary_img.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    densities = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = gy * cell_h, (gy + 1) * cell_h if gy < grid_size - 1 else h
            x0, x1 = gx * cell_w, (gx + 1) * cell_w if gx < grid_size - 1 else w
            cell = binary_img[y0:y1, x0:x1]
            densities.append(cell.sum() / (cell.size * 255) if cell.size > 0 else 0)
    return densities


def kudlit_component_features(binary_img):
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


KUDLIT_STRIP_RATIO = 0.22  # top / bottom band width, fraction of glyph height


def kudlit_strip_features(binary):
    """Fraction of ink in the top band vs the bottom band. o/u kudlit sits
    below the body, e/i above, bare/-a has neither - and unlike the connected-
    component features these still fire when the mark touches the glyph.
    Appended last so a 26-feature model still slices cleanly (see app.py)."""
    h = binary.shape[0]
    strip = max(1, int(round(h * KUDLIT_STRIP_RATIO)))
    total = float(binary.sum()) or 1.0
    return [float(binary[:strip].sum() / total),
            float(binary[h - strip:].sum() / total)]


KUDLIT_SHAPE_BAND = 0.16  # top / bottom band, fraction of glyph height


def kudlit_shape_features(binary):
    """Shape of the mark in the top / bottom band: [top_aspect, top_relwidth,
    bot_aspect, bot_relwidth]. A dash kudlit is wide + flat (aspect > ~2, wide
    share of the glyph); a dot is compact (aspect ~1). Separates Ne/Ni and
    Nu/No, which the density / position features cannot. Appended last (see
    app.py._kudlit_shape_features - keep identical)."""
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
        return [min(6.0, mw / max(1.0, mh)), min(1.5, mw / max(1.0, glyph_w))]

    return strip_shape(binary[:band]) + strip_shape(binary[h - band:])


def extract_spatial_features(pre_img):
    _, binary = cv2.threshold(pre_img, 127, 255, cv2.THRESH_BINARY)
    overall_density = [binary.sum() / (binary.size * 255)]
    quadrant = grid_density_features(binary, grid_size=2)
    fine_grid = grid_density_features(binary, grid_size=4)
    kudlit_feats = kudlit_component_features(binary)
    strip_feats = kudlit_strip_features(binary)
    shape_feats = kudlit_shape_features(binary)
    return (overall_density + quadrant + fine_grid + kudlit_feats
            + strip_feats + shape_feats)


def extract_combined_features(pre_img):
    hog_features = hog(
        pre_img, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK, block_norm=HOG_BLOCK_NORM, feature_vector=True,
    )
    spatial_features = extract_spatial_features(pre_img)
    return np.concatenate([hog_features, spatial_features]).astype(np.float64)


# ============================================================================
# AUGMENTATION  (train split only; --augment N adds N variants per image)
# Deliberately mild so a kudlit dot/mark stays in place and e<->i / o<->u are
# not blurred into each other. No flips.
# ============================================================================
def augment_once(pre_img, rng):
    h, w = pre_img.shape
    angle = rng.uniform(-5, 5)
    scale = rng.uniform(0.92, 1.08)
    tx, ty = rng.integers(-3, 4), rng.integers(-3, 4)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    out = cv2.warpAffine(pre_img, m, (w, h), flags=cv2.INTER_AREA,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mode = rng.integers(0, 3)
    if mode == 1:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    elif mode == 2:
        out = cv2.erode(out, np.ones((2, 2), np.uint8), iterations=1)
    return out


_STANDALONE_VOWELS = {"A", "E", "I", "O", "U"}


def has_kudlit(class_name):
    """True for classes that carry a mark: bare consonant (virama) or +e/i/o/u.
    False for the inherent -a forms and the standalone vowels."""
    if class_name in _STANDALONE_VOWELS:
        return False
    low = class_name.lower()
    if not any(v in low for v in "aeiou"):
        return True                       # bare consonant -> virama
    return low[-1] in "eiou"              # +e/i/o/u kudlit; +a has none


def augment_kudlit(pre_img, rng):
    """Heavier augment, only for mark-bearing classes: wider affine so the
    kudlit lands in more positions/sizes relative to the body, and always a
    stroke-thickness change. The dash-vs-dot / present-vs-absent boundary is
    razor-thin, so this widens it without flips or big rotations."""
    h, w = pre_img.shape
    angle = rng.uniform(-6, 6)
    scale = rng.uniform(0.85, 1.18)
    tx, ty = rng.integers(-5, 6), rng.integers(-5, 6)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    out = cv2.warpAffine(pre_img, m, (w, h), flags=cv2.INTER_AREA,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.integers(0, 2):
        out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    else:
        out = cv2.erode(out, np.ones((2, 2), np.uint8), iterations=1)
    return out


# ============================================================================
# DATASET
# ============================================================================
def load_grayscale(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        from PIL import Image
        with Image.open(path) as im:
            return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    except Exception:
        return None


def discover(root, limit_per_class=None):
    root = Path(root)
    if not root.is_dir():
        sys.exit(f"ERROR: --data {root} is not a directory")
    items = []
    class_counts = {}
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if limit_per_class:
            files = files[:limit_per_class]
        if not files:
            continue
        class_counts[cls_dir.name] = len(files)
        items.extend((f, cls_dir.name) for f in files)
    if not items:
        sys.exit(f"ERROR: no images found under {root} (expected {root}/<ClassName>/*.png)")
    return items, class_counts


def _process_one(path, label, n_aug, kudlit_aug, seed):
    warnings.filterwarnings("ignore")  # also silence inside joblib worker processes
    gray = load_grayscale(path)
    pre = preprocess_image(gray)
    if pre is None:
        return []
    rows = [(extract_combined_features(pre), label, str(path))]
    k_aug = kudlit_aug if (kudlit_aug and has_kudlit(label)) else 0
    if n_aug or k_aug:
        rng = np.random.default_rng(abs(hash((str(path), seed))) % (2**32))
        for k in range(n_aug):
            rows.append((extract_combined_features(augment_once(pre, rng)),
                         label, f"{path}#aug{k}"))
        for k in range(k_aug):
            rows.append((extract_combined_features(augment_kudlit(pre, rng)),
                         label, f"{path}#kaug{k}"))
    return rows


def build_matrix(items, n_aug, kudlit_aug, seed, n_jobs):
    t0 = time.time()
    batches = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(_process_one)(p, lab, n_aug, kudlit_aug, seed) for p, lab in items
    )
    feats, labels, paths = [], [], []
    for b in batches:
        for f, lab, pth in b:
            feats.append(f)
            labels.append(lab)
            paths.append(pth)
    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels)
    paths = np.asarray(paths)
    assert X.shape[1] == N_FEATURES, f"feature length {X.shape[1]} != {N_FEATURES}"
    print(f"built X={X.shape} in {time.time()-t0:.0f}s "
          f"({len(items)} source images, +{n_aug}/img global aug, "
          f"+{kudlit_aug}/img kudlit-class aug, "
          f"{X.shape[0]-sum(1 for b in batches if b)} augmented rows added, "
          f"{len(items)-sum(1 for b in batches if b)} unreadable/blank)")
    return X, y, paths


# ============================================================================
# SCALING + SEARCH + TRAIN
# ============================================================================
def scale_blocks(X, hog_scaler, spatial_scaler, weight):
    hog_block = hog_scaler.transform(X[:, :N_HOG_FEATURES])
    spatial_block = spatial_scaler.transform(X[:, N_HOG_FEATURES:]) * weight
    return np.hstack([hog_block, spatial_block])


def stratified_subsample(X, y, n, seed):
    if len(y) <= n:
        return X, y
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, random_state=seed, stratify=y)
    return X[idx], y[idx]


def search_weight(Xtr, ytr, Xva, yva, hog_scaler, spatial_scaler, grid, C, seed):
    Xs, ys = stratified_subsample(Xtr, ytr, WEIGHT_SEARCH_SUBSAMPLE, seed)
    print(f"\nspatial-weight search on {len(ys)} train / {len(yva)} val samples:")
    best_w, best_acc = grid[0], -1.0
    for w in grid:
        clf = SVC(C=C, gamma=SVC_GAMMA, class_weight="balanced", cache_size=1000)
        clf.fit(scale_blocks(Xs, hog_scaler, spatial_scaler, w), ys)
        acc = clf.score(scale_blocks(Xva, hog_scaler, spatial_scaler, w), yva)
        flag = "  <-- best" if acc > best_acc else ""
        print(f"   weight {w:>3}: val acc {acc:.4f}{flag}")
        if acc > best_acc:
            best_w, best_acc = w, acc
    print(f"picked spatial weight = {best_w}  (val acc {best_acc:.4f})")
    return int(best_w)


def search_c(Xtr, ytr, Xva, yva, hog_scaler, spatial_scaler, weight, grid, seed):
    Xs, ys = stratified_subsample(Xtr, ytr, WEIGHT_SEARCH_SUBSAMPLE, seed)
    Xva_s = scale_blocks(Xva, hog_scaler, spatial_scaler, weight)
    Xs_s = scale_blocks(Xs, hog_scaler, spatial_scaler, weight)
    print(f"\nC search on {len(ys)} train / {len(yva)} val samples:")
    best_c, best_acc = grid[0], -1.0
    for c in grid:
        clf = SVC(C=c, gamma=SVC_GAMMA, class_weight="balanced", cache_size=1000)
        clf.fit(Xs_s, ys)
        acc = clf.score(Xva_s, yva)
        flag = "  <-- best" if acc > best_acc else ""
        print(f"   C {c:>5}: val acc {acc:.4f}{flag}")
        if acc > best_acc:
            best_c, best_acc = c, acc
    print(f"picked C = {best_c}  (val acc {best_acc:.4f})")
    return float(best_c)


# ============================================================================
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(data, out, augment=0, kudlit_augment=0, C=SVC_C_DEFAULT, search_c="",
        weight_grid="", fixed_weight=0, limit_per_class=0, n_jobs=-1,
        target_size=64):
    """Notebook entry point - call this instead of using the CLI, e.g.:
        run(data="/content/drive/MyDrive/ALL_DATASET",
            out="/content/drive/MyDrive/WEIGHTED_MODEL_V6",
            kudlit_augment=3)          # sanity check: mark-class augment only
    """
    argv = ["--data", str(data), "--out", str(out), "--augment", str(augment),
            "--kudlit-augment", str(kudlit_augment),
            "--C", str(C), "--n-jobs", str(n_jobs), "--target-size", str(target_size)]
    if search_c:
        argv += ["--search-c", str(search_c)]
    if weight_grid:
        argv += ["--weight-grid", str(weight_grid)]
    if fixed_weight:
        argv += ["--fixed-weight", str(fixed_weight)]
    if limit_per_class:
        argv += ["--limit-per-class", str(limit_per_class)]
    main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="ALL_DATASET root (folder per class)")
    ap.add_argument("--out", required=True, help="output dir for artifacts")
    ap.add_argument("--augment", type=int, default=0,
                    help="augmented copies per training image (0 = off; 2 is a good "
                         "start for kudlit robustness, ~2-3x training time)")
    ap.add_argument("--kudlit-augment", type=int, default=0,
                    help="extra augmented copies for MARK-BEARING classes only "
                         "(bare consonant + e/i/o/u forms): wider affine so the "
                         "kudlit varies in position/size. Sanity check for "
                         "whether mark data - not architecture - is the fix.")
    ap.add_argument("--C", type=float, default=SVC_C_DEFAULT)
    ap.add_argument("--search-c", default="", help="comma list, e.g. 10,20,50,100")
    ap.add_argument("--weight-grid", default="",
                    help="comma list overriding the spatial-weight search grid")
    ap.add_argument("--fixed-weight", type=int, default=0,
                    help="skip the weight search and use this value")
    ap.add_argument("--limit-per-class", type=int, default=0, help="smoke-test cap")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--target-size", type=int, default=64, choices=[48, 64, 80, 96, 128],
                    help="glyph resize (default 64). 96 makes the kudlit dot/mark "
                         "~2x bigger for the model; app.py auto-adapts from the scaler.")
    args = ap.parse_args(argv)

    global TARGET_SIZE, N_HOG_FEATURES, N_FEATURES
    TARGET_SIZE = args.target_size
    N_HOG_FEATURES = _hog_len(TARGET_SIZE)
    N_FEATURES = N_HOG_FEATURES + N_SPATIAL_FEATURES
    print(f"target_size {TARGET_SIZE} -> {N_HOG_FEATURES} HOG + {N_SPATIAL_FEATURES} "
          f"spatial = {N_FEATURES} features")

    out = Path(args.out)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    # ---- 1. dataset ----
    items, class_counts = discover(args.data, args.limit_per_class or None)
    n_classes = len(class_counts)
    counts = np.array(list(class_counts.values()))
    med = int(np.median(counts))
    print(f"{n_classes} classes, {len(items)} images "
          f"(min {counts.min()}, median {med}, max {counts.max()} per class)")
    weak = {c: n for c, n in class_counts.items() if n < 0.5 * med}
    if weak:
        print(f"WARNING: {len(weak)} class(es) have < 50% of the median count "
              f"(kudlit variants are a common culprit): {weak}")

    # ---- 2. features ----
    X_all_noaug, y_all_str, paths_all = build_matrix(items, 0, 0, SEED, args.n_jobs)

    le = LabelEncoder().fit(sorted(class_counts))
    y_all = le.transform(y_all_str)
    print(f"label order: {list(le.classes_)}")

    # ---- 3. split (stratified, deterministic) ----
    idx = np.arange(len(y_all))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_FRAC, random_state=SEED, stratify=y_all)
    val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=val_rel, random_state=SEED, stratify=y_all[idx_tr])
    print(f"split: train {len(idx_tr)}  val {len(idx_va)}  test {len(idx_te)}")

    # augment TRAIN only
    if args.augment or args.kudlit_augment:
        train_items = [items[i] for i in idx_tr]
        Xtr, ytr_str, ptr = build_matrix(
            train_items, args.augment, args.kudlit_augment, SEED, args.n_jobs)
        ytr = le.transform(ytr_str)
    else:
        Xtr, ytr, ptr = X_all_noaug[idx_tr], y_all[idx_tr], paths_all[idx_tr]
    Xva, yva, pva = X_all_noaug[idx_va], y_all[idx_va], paths_all[idx_va]
    Xte, yte, pte = X_all_noaug[idx_te], y_all[idx_te], paths_all[idx_te]

    # ---- 4. scalers (fit on TRAIN only) ----
    hog_scaler = StandardScaler().fit(Xtr[:, :N_HOG_FEATURES])
    spatial_scaler = StandardScaler().fit(Xtr[:, N_HOG_FEATURES:])

    # ---- 5. spatial-weight search ----
    if args.fixed_weight:
        weight = int(args.fixed_weight)
        print(f"\nusing fixed spatial weight = {weight}")
    else:
        grid = ([int(x) for x in args.weight_grid.split(",")]
                if args.weight_grid else DEFAULT_WEIGHT_GRID)
        weight = search_weight(Xtr, ytr, Xva, yva, hog_scaler, spatial_scaler, grid, args.C, SEED)

    # ---- 6. optional C search ----
    C = args.C
    if args.search_c:
        C = search_c(Xtr, ytr, Xva, yva, hog_scaler, spatial_scaler, weight,
                     [float(x) for x in args.search_c.split(",")], SEED)

    # ---- 7. final SVM on full train ----
    print(f"\ntraining final SVC(C={C}, gamma='{SVC_GAMMA}', class_weight='balanced') "
          f"on {len(ytr)} samples ...")
    t0 = time.time()
    Xtr_s = scale_blocks(Xtr, hog_scaler, spatial_scaler, weight)
    model = SVC(C=C, gamma=SVC_GAMMA, class_weight="balanced", cache_size=2000)
    model.fit(Xtr_s, ytr)
    print(f"  fit in {time.time()-t0:.0f}s, {model.support_vectors_.shape[0]} support vectors")

    # ---- 8. isotonic calibrator on val ----
    print("fitting isotonic CalibratedClassifierCV on the val split ...")
    t0 = time.time()
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(scale_blocks(Xva, hog_scaler, spatial_scaler, weight), yva)
    print(f"  fit in {time.time()-t0:.0f}s")

    # ---- 9. evaluate on test ----
    Xte_s = scale_blocks(Xte, hog_scaler, spatial_scaler, weight)
    pred = model.predict(Xte_s)
    proba = calibrated.predict_proba(Xte_s)
    pred_cal = calibrated.classes_[proba.argmax(1)]
    acc = accuracy_score(yte, pred)
    acc_cal = accuracy_score(yte, pred_cal)
    conf = proba.max(1)
    correct = pred_cal == yte
    mp, mr, mf, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(yte, pred, labels=list(range(n_classes)))

    print(f"\n=== TEST ===  acc {acc:.4f}  (calibrated {acc_cal:.4f})  "
          f"macro-F1 {mf:.4f}  weighted-F1 {wf:.4f}")
    print(classification_report(yte, pred, target_names=list(le.classes_),
                                zero_division=0, digits=4))
    # kudlit-focused confusions
    conf_pairs = []
    for t in range(n_classes):
        for p in range(n_classes):
            if t != p and cm[t, p]:
                conf_pairs.append((int(cm[t, p]), le.classes_[t], le.classes_[p]))
    conf_pairs.sort(reverse=True)
    print("top confusions:", ", ".join(f"{a}->{b}({n})" for n, a, b in conf_pairs[:12]))

    # ---- 10. save artifacts ----
    joblib.dump(model, out / "weighted_svm.pkl")
    joblib.dump(calibrated, out / "weighted_svm_calibrated.pkl")
    joblib.dump(hog_scaler, out / "hog_scaler.pkl")
    joblib.dump(spatial_scaler, out / "spatial_scaler.pkl")
    joblib.dump(int(weight), out / "best_weight.pkl")
    joblib.dump(le, out / "label_encoder.pkl")

    reports = out  # test .npy live alongside; copy them into backend/tests/reports/
    np.save(reports / "test_predictions.npy", pred)
    np.save(reports / "test_true_labels.npy", yte)
    np.save(reports / "test_confusion_matrix.npy", cm)

    np.save(out / "X.npy", X_all_noaug)
    np.save(out / "y.npy", y_all)
    np.save(out / "filepaths.npy", paths_all)
    np.save(out / "splits" / "X_train.npy", Xtr)
    np.save(out / "splits" / "y_train.npy", ytr)
    np.save(out / "splits" / "X_val.npy", Xva)
    np.save(out / "splits" / "y_val.npy", yva)
    np.save(out / "splits" / "X_test.npy", Xte)
    np.save(out / "splits" / "y_test.npy", yte)
    np.save(out / "splits" / "paths_train.npy", ptr)
    np.save(out / "splits" / "paths_val.npy", pva)
    np.save(out / "splits" / "paths_test.npy", pte)

    per_class = {}
    pp, pr, pf, ps = precision_recall_fscore_support(
        yte, pred, labels=list(range(n_classes)), average=None, zero_division=0)
    for i, c in enumerate(le.classes_):
        per_class[c] = {"precision": round(float(pp[i]), 4), "recall": round(float(pr[i]), 4),
                        "f1": round(float(pf[i]), 4), "support": int(ps[i])}

    metrics = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_classes": n_classes,
        "classes": list(le.classes_),
        "source_images": len(items),
        "augment_per_image": args.augment,
        "kudlit_augment": args.kudlit_augment,
        "train/val/test": [int(len(ytr)), int(len(yva)), int(len(yte))],
        "svc": {"C": C, "gamma": SVC_GAMMA, "class_weight": "balanced",
                "n_support_vectors": int(model.support_vectors_.shape[0])},
        "spatial_weight": int(weight),
        "target_size": TARGET_SIZE,
        "n_features": N_FEATURES,
        "test_accuracy": round(float(acc), 4),
        "test_accuracy_calibrated": round(float(acc_cal), 4),
        "macro_f1": round(float(mf), 4), "weighted_f1": round(float(wf), 4),
        "macro_precision": round(float(mp), 4), "macro_recall": round(float(mr), 4),
        "calibrated_conf>=0.90_keep": round(float((conf >= 0.9).mean()), 4),
        "calibrated_conf>=0.90_acc": round(float(correct[conf >= 0.9].mean()), 4)
        if (conf >= 0.9).any() else None,
        "class_counts": class_counts,
        "per_class": per_class,
        "top_confusions": [{"true": a, "pred": b, "count": n} for n, a, b in conf_pairs[:30]],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    manifest = {"trained_at": metrics["trained_at"], "config": {
        "target_size": TARGET_SIZE, "hog": [HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL,
        HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM], "n_features": N_FEATURES,
        "spatial_weight": int(weight), "C": C}, "sha256": {}}
    for name in ["weighted_svm.pkl", "weighted_svm_calibrated.pkl", "hog_scaler.pkl",
                 "spatial_scaler.pkl", "best_weight.pkl", "label_encoder.pkl",
                 "test_predictions.npy", "test_true_labels.npy", "test_confusion_matrix.npy"]:
        manifest["sha256"][name] = sha256(out / name)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    (out / "MODEL_README.txt").write_text(
        f"DAYAW weighted Baybayin model - trained {metrics['trained_at']}\n"
        f"Test accuracy: {acc:.4f}  (calibrated {acc_cal:.4f})\n"
        f"{n_classes} classes, {len(items)} source images, "
        f"augment x{args.augment}, kudlit-augment x{args.kudlit_augment}\n\n"
        f"Pipeline (must match backend/app.py):\n"
        f"  preprocess: grayscale -> Otsu invert -> remove_small_objects(min_size={MIN_NOISE_SIZE})\n"
        f"    -> tight crop -> pad square (ratio {PAD_RATIO}) -> resize {TARGET_SIZE}x{TARGET_SIZE}\n"
        f"  HOG: orientations={HOG_ORIENTATIONS} ppc={HOG_PIXELS_PER_CELL} "
        f"cpb={HOG_CELLS_PER_BLOCK} block_norm={HOG_BLOCK_NORM} -> {N_HOG_FEATURES}\n"
        f"  spatial: overall density(1) + 2x2 grid(4) + 4x4 grid(16) + kudlit stats(5) "
        f"+ top/bottom strip(2) + kudlit mark shape(4) -> {N_SPATIAL_FEATURES}\n"
        f"  scale HOG block with hog_scaler; scale spatial block with spatial_scaler "
        f"then x {weight}; concat -> {N_FEATURES}\n"
        f"  SVC(C={C}, gamma='{SVC_GAMMA}', class_weight='balanced'); "
        f"label_encoder.inverse_transform for names\n"
        f"  confidence: weighted_svm_calibrated.pkl .predict_proba (isotonic, fit on val)\n"
    )

    print(f"\nsaved to {out}")
    print("\nINSTALL INTO THE APP:")
    print(f"  cp {out}/{{weighted_svm,weighted_svm_calibrated,hog_scaler,spatial_scaler,"
          f"best_weight,label_encoder}}.pkl   dayawanalisa/backend/")
    print(f"  cp {out}/test_{{predictions,true_labels,confusion_matrix}}.npy   "
          f"dayawanalisa/backend/tests/reports/")
    print("  cd dayawanalisa/backend && python tests/generate_model_report.py")
    print(f"  (default CONFIG unchanged -> no app.py edits needed; spatial weight "
          f"{weight} is read from best_weight.pkl)")


def _running_in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


# Run the CLI only for `python train_weighted_model.py ...`. When this file is
# pasted into a Jupyter/Colab cell `__name__` is also "__main__", so guard on the
# notebook check too and let the user call run(data=..., out=...) instead.
if __name__ == "__main__" and not _running_in_notebook():
    main()
