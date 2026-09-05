"""
Fit an isotonic probability calibrator on top of the prefit weighted SVM and
save it as backend/weighted_svm_calibrated.pkl (what app.py's classify_glyph
loads for confidence scores).

Why: weighted_svm.pkl was trained probability=False. Its raw decision_function
margins carry almost no "how sure am I" signal (softmax proxy AUROC ~0.49 on the
test set). CalibratedClassifierCV(cv="prefit", method="isotonic") fit on the
held-out validation split leaves predict() essentially unchanged (91.14% ->
~91.1%, ~0.4% of predictions shift) but turns predict_proba() into an honestly
scaled probability.

Needs the Colab feature splits (HOG_FEATURES_V2/splits/X_val.npy etc). They are
too large to vendor, so point FEATURES_DIR at wherever they live:

    python calibrate_model.py
    FEATURES_DIR="D:/somewhere/HOG_FEATURES_V2" python calibrate_model.py

Re-run this whenever weighted_svm.pkl / the scalers are retrained.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "tests" / "reports"

DEFAULT_FEATURES_DIR = r"C:\Users\RAIN\Downloads\HOG_FEATURES_V2-20260831T220228Z-1-001\HOG_FEATURES_V2"
FEATURES_DIR = Path(os.environ.get("FEATURES_DIR", DEFAULT_FEATURES_DIR))

HOG_FEATURE_LEN = 1764
OUT_PATH = BASE_DIR / "weighted_svm_calibrated.pkl"


def load_split(name):
    p = FEATURES_DIR / "splits" / name
    if not p.exists():
        sys.exit(
            f"ERROR: {p} not found.\n"
            f"Set FEATURES_DIR to the folder that holds splits/X_val.npy etc."
        )
    return np.load(p)


def weighted_scale(X, hog_scaler, spatial_scaler, weight):
    hog_block = hog_scaler.transform(X[:, :HOG_FEATURE_LEN])
    spatial_block = spatial_scaler.transform(X[:, HOG_FEATURE_LEN:]) * weight
    return np.hstack([hog_block, spatial_block])


def expected_calibration_error(conf, correct, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not m.any():
            continue
        acc = float(correct[m].mean())
        avg_conf = float(conf[m].mean())
        ece += (m.mean()) * abs(acc - avg_conf)
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                     "predicted": round(avg_conf, 4), "actual": round(acc, 4)})
    return float(ece), rows


def main():
    model = joblib.load(BASE_DIR / "weighted_svm.pkl")
    hog_scaler = joblib.load(BASE_DIR / "hog_scaler.pkl")
    spatial_scaler = joblib.load(BASE_DIR / "spatial_scaler.pkl")
    weight = joblib.load(BASE_DIR / "best_weight.pkl")
    label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

    X_val, y_val = load_split("X_val.npy"), load_split("y_val.npy")
    X_test, y_test = load_split("X_test.npy"), load_split("y_test.npy")
    print(f"val {X_val.shape}  test {X_test.shape}  spatial_weight={weight}")

    Xv = weighted_scale(X_val, hog_scaler, spatial_scaler, weight)
    Xt = weighted_scale(X_test, hog_scaler, spatial_scaler, weight)

    base_pred = model.predict(Xt)
    base_acc = float((base_pred == y_test).mean())

    print("fitting isotonic calibrator on the validation split ...")
    t0 = time.time()
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(Xv, y_val)
    print(f"  done in {time.time() - t0:.0f}s")

    proba = calibrated.predict_proba(Xt)
    cal_pred = calibrated.classes_[proba.argmax(axis=1)]
    cal_acc = float((cal_pred == y_test).mean())
    conf = proba.max(axis=1)
    correct = cal_pred == y_test
    shifted = int((cal_pred != base_pred).sum())

    ece, reliability = expected_calibration_error(conf, correct)

    print(f"\ntest accuracy: {base_acc:.4f} (raw)  ->  {cal_acc:.4f} (calibrated)")
    print(f"predictions shifted by calibration: {shifted}/{len(y_test)} "
          f"({shifted / len(y_test) * 100:.2f}%)")
    print(f"expected calibration error (10-bin): {ece:.4f}")
    print("\nreliability:")
    for r in reliability:
        print(f"  {r['bin']:>10}  n={r['n']:4d}  predicted {r['predicted']:.3f}  actual {r['actual']:.3f}")
    for thr in (0.5, 0.7, 0.8, 0.9, 0.95):
        keep = conf >= thr
        if keep.any():
            print(f"  conf>={thr:.2f}: keeps {keep.mean() * 100:5.1f}%   acc-on-kept {correct[keep].mean() * 100:5.1f}%")

    joblib.dump(calibrated, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nsaved -> {OUT_PATH}  ({size_mb:.0f} MB)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "calibrator": "CalibratedClassifierCV(method='isotonic', cv='prefit')",
        "fit_on": "validation split (never used to train weighted_svm.pkl)",
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "test_accuracy_raw": round(base_acc, 4),
        "test_accuracy_calibrated": round(cal_acc, 4),
        "predictions_shifted": shifted,
        "expected_calibration_error": round(ece, 4),
        "reliability_bins": reliability,
        "num_classes": len(label_encoder.classes_),
    }
    (REPORTS_DIR / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {REPORTS_DIR / 'calibration_report.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["predicted"] for r in reliability]
        ys = [r["actual"] for r in reliability]
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        plt.plot(xs, ys, "o-", color="#8B5E3C", label="calibrated weighted SVM")
        plt.xlabel("predicted confidence")
        plt.ylabel("actual accuracy")
        plt.title(f"Reliability (ECE={ece:.3f}, test acc={cal_acc:.3f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "calibration_reliability.png", dpi=120)
        print(f"wrote {REPORTS_DIR / 'calibration_reliability.png'}")
    except Exception as e:
        print(f"(skipped reliability plot: {e})")


if __name__ == "__main__":
    main()
