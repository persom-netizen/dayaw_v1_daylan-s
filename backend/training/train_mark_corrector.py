#!/usr/bin/env python3
r"""
train_mark_corrector.py  --  Path A: a small 6-class SVM that reads ONLY the
kudlit / virama mark, to correct the 95-class monolith's #1 error (the -o dot
lost: Ko->K, No->N, ...).

Still SVM + HOG. The monolith is NOT retrained or touched. This adds:
    mark_svm.pkl  mark_svm_calibrated.pkl  mark_scaler.pkl  mark_meta.json

app.py loads them automatically if present (reconcile_mark); missing -> the
monolith runs exactly as before.

Colab (paste this whole file into a cell, then in the next cell):
    run_mark(data="/content/drive/MyDrive/ALL_DATASET",
             out="/content/drive/MyDrive/MARK_CORRECTOR_V1")

--------------------------------------------------------------------------------
V1 RESULT (2026-09-08, 34,810 images): mark test acc 0.929, but dot_below F1
only 0.867 - it confuses the o-dot with the virama "x" the same way the
monolith does (both are "a mark below the body"). On the real test sheets its
confident overrides fired in the wrong direction; app.py's confidence gate
(MARK_TRUST_MONO_ABOVE) then blocked it whenever the monolith was confidently
wrong. Net: no help. See training/README.md section 4 - the recommendation is
Path B (classical Baybayin, e==i and o==u), where the corrector is only a
3/4-class "above / below / none" call, which V1 already does well
(none F1 0.95, dash_above 0.96).
"""

import argparse, json, sys, time, warnings
from pathlib import Path

import cv2, joblib, numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from skimage.feature import hog
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import remove_small_objects
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---- CONFIG (preprocess + feature layout MUST match backend/app.py) ----
TARGET_SIZE = 64
MIN_NOISE_SIZE = 20
PAD_RATIO = 0.12
HOG_KW = dict(orientations=9, pixels_per_cell=(8, 8),
              cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
MARK_BAND = 0.42
CROP = 32
KUDLIT_STRIP_RATIO = 0.22
KUDLIT_SHAPE_BAND = 0.16
SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

MARK_CLASSES = ["none", "virama", "dot_above", "dash_above", "dot_below", "dash_below"]
_VOWELS = {"A", "E", "I", "O", "U"}
# THIS font's convention: e=dash above, i=dot above, o=dot below, u=dash below
_TAIL = {"a": "none", "e": "dash_above", "i": "dot_above",
         "o": "dot_below", "u": "dash_below"}


def mark_label(cls):
    if cls in _VOWELS:
        return "none"
    low = cls.lower()
    if not any(v in low for v in "aeiou"):
        return "virama"
    return _TAIL[low[-1]]


def preprocess_image(gray):
    if gray is None or gray.size == 0:
        return None
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned = (remove_small_objects(b > 0, min_size=MIN_NOISE_SIZE) * 255).astype(np.uint8)
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
    pad = int(side * PAD_RATIO)
    cs = side + 2 * pad
    canvas = np.zeros((cs, cs), np.uint8)
    yo, xo = (cs - h) // 2, (cs - w) // 2
    canvas[yo:yo + h, xo:xo + w] = tight
    return cv2.resize(canvas, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)


def kudlit_strip_features(binary):
    h = binary.shape[0]
    s = max(1, int(round(h * KUDLIT_STRIP_RATIO)))
    tot = float(binary.sum()) or 1.0
    return [float(binary[:s].sum() / tot), float(binary[h - s:].sum() / tot)]


def kudlit_shape_features(binary):
    h, w = binary.shape
    band = max(1, int(round(h * KUDLIT_SHAPE_BAND)))
    cols = np.where(binary.sum(axis=0) > 0)[0]
    gw = float(cols[-1] - cols[0] + 1) if cols.size else float(w)

    def ss(strip):
        cs = np.where(strip.sum(axis=0) > 0)[0]
        rs = np.where(strip.sum(axis=1) > 0)[0]
        if cs.size == 0 or rs.size == 0:
            return [0.0, 0.0]
        mw = float(cs[-1] - cs[0] + 1); mh = float(rs[-1] - rs[0] + 1)
        return [min(6.0, mw / max(1.0, mh)), min(1.5, mw / max(1.0, gw))]
    return ss(binary[:band]) + ss(binary[h - band:])


def kudlit_component_features(binary):
    reg = regionprops(sk_label(binary > 0))
    if not reg:
        return [0, 0, 0, 0, 0]
    reg.sort(key=lambda r: -r.area)
    tot = sum(r.area for r in reg)
    lg = reg[0]
    laf = lg.area / tot if tot else 0
    if len(reg) >= 2:
        sc = reg[1]
        sar = sc.area / lg.area if lg.area else 0
        h, w = binary.shape
        dy = (sc.centroid[0] - lg.centroid[0]) / h
        dx = (sc.centroid[1] - lg.centroid[1]) / w
    else:
        sar = dy = dx = 0.0
    return [len(reg), laf, sar, dy, dx]


def mark_features(pre_img):
    _, b = cv2.threshold(pre_img, 127, 255, cv2.THRESH_BINARY)
    h = b.shape[0]
    band = int(round(h * MARK_BAND))
    top = cv2.resize(pre_img[:band, :], (CROP, CROP), interpolation=cv2.INTER_AREA)
    bot = cv2.resize(pre_img[h - band:, :], (CROP, CROP), interpolation=cv2.INTER_AREA)
    return np.concatenate([
        hog(top, **HOG_KW), hog(bot, **HOG_KW),
        kudlit_strip_features(b), kudlit_shape_features(b),
        kudlit_component_features(b),
    ]).astype(np.float64)


def load_gray(p):
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is not None:
        return im
    try:
        from PIL import Image
        with Image.open(p) as x:
            return cv2.cvtColor(np.asarray(x.convert("RGB")), cv2.COLOR_RGB2GRAY)
    except Exception:
        return None


def _one(path, y6):
    warnings.filterwarnings("ignore")
    pre = preprocess_image(load_gray(path))
    if pre is None:
        return None
    return mark_features(pre), y6


def build(items, n_jobs):
    t0 = time.time()
    rows = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
        joblib.delayed(_one)(p, y) for p, y in items)
    rows = [r for r in rows if r is not None]
    X = np.asarray([r[0] for r in rows], np.float64)
    y = np.asarray([r[1] for r in rows])
    print(f"built X={X.shape} in {time.time()-t0:.0f}s ({len(items)-len(rows)} unreadable)")
    return X, y


def run_mark(data, out, C=10.0, n_jobs=-1, limit_per_class=0):
    argv = ["--data", str(data), "--out", str(out), "--C", str(C),
            "--n-jobs", str(n_jobs)]
    if limit_per_class:
        argv += ["--limit-per-class", str(limit_per_class)]
    main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--limit-per-class", type=int, default=0)
    a = ap.parse_args(argv)

    root = Path(a.data)
    if not root.is_dir():
        sys.exit(f"ERROR: --data {root} is not a directory (mount Drive?)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    items, seen = [], {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        y6 = mark_label(d.name)
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if a.limit_per_class:
            files = files[:a.limit_per_class]
        items += [(f, y6) for f in files]
        seen[d.name] = (y6, len(files))
    print(f"{len(seen)} source classes -> 6 mark classes, {len(items)} images")
    by6 = {}
    for _, (y6, n) in seen.items():
        by6[y6] = by6.get(y6, 0) + n
    print("mark-class image counts:", by6)

    X, y = build(items, a.n_jobs)
    yc = np.array([MARK_CLASSES.index(v) for v in y])

    idx = np.arange(len(yc))
    itr, ite = train_test_split(idx, test_size=TEST_FRAC, random_state=SEED, stratify=yc)
    itr, iva = train_test_split(itr, test_size=VAL_FRAC / (1 - TEST_FRAC),
                                random_state=SEED, stratify=yc[itr])
    print(f"split: train {len(itr)}  val {len(iva)}  test {len(ite)}")

    sc = StandardScaler().fit(X[itr])
    Xtr, Xva, Xte = sc.transform(X[itr]), sc.transform(X[iva]), sc.transform(X[ite])

    print(f"\nfitting SVC(C={a.C}) on {len(itr)} ...")
    t0 = time.time()
    clf = SVC(C=a.C, gamma="scale", class_weight="balanced", cache_size=1500)
    clf.fit(Xtr, yc[itr])
    print(f"  fit in {time.time()-t0:.0f}s, {clf.support_vectors_.shape[0]} SVs")

    cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(Xva, yc[iva])

    pred = clf.predict(Xte)
    acc = float((pred == yc[ite]).mean())
    print(f"\n=== MARK TEST ===  acc {acc:.4f}\n")
    print(classification_report(yc[ite], pred, target_names=MARK_CLASSES,
                                digits=4, zero_division=0))
    cm = confusion_matrix(yc[ite], pred, labels=list(range(6)))
    print("confusion matrix (rows=true, cols=pred):")
    print("            " + "  ".join(f"{c[:6]:>6}" for c in MARK_CLASSES))
    for i, c in enumerate(MARK_CLASSES):
        print(f"{c:>11} " + "  ".join(f"{cm[i, j]:>6}" for j in range(6)))
    di, vi, ni = (MARK_CLASSES.index(k) for k in ("dot_below", "virama", "none"))
    print(f"\ndot_below recall: {cm[di, di] / cm[di].sum():.3f}   "
          f"(missed as virama {cm[di, vi]}, as none {cm[di, ni]})")
    print(f"virama->dot_below: {cm[vi, di]}    none->dot_below: {cm[ni, di]}")

    joblib.dump(clf, out / "mark_svm.pkl")
    joblib.dump(cal, out / "mark_svm_calibrated.pkl")
    joblib.dump(sc, out / "mark_scaler.pkl")
    (out / "mark_meta.json").write_text(json.dumps({
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mark_classes": MARK_CLASSES,
        "feature_config": {"target_size": TARGET_SIZE, "mark_band": MARK_BAND,
                           "crop": CROP, "n_features": int(X.shape[1])},
        "test_accuracy": round(acc, 4),
        "source_class_map": {k: v[0] for k, v in seen.items()},
    }, indent=2))
    print(f"\nsaved to {out}")
    print("INSTALL: cp {mark_svm,mark_svm_calibrated,mark_scaler}.pkl -> backend/  "
          "(app.py loads them automatically; delete them to disable)")


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _in_notebook():
    main()
