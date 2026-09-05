"""
Regenerate the effectiveness reports for the weighted V2 Baybayin model from
the saved Colab test outputs in tests/reports/:

    test_predictions.npy   - model's predicted class index per test sample (4560)
    test_true_labels.npy   - ground-truth class index per test sample
    test_confusion_matrix.npy - 95x95 confusion matrix (rows = true, cols = pred)

No images or the SVM itself are needed - these arrays already are the test-set
results (91.14% accuracy). Run after copying fresh .npy files from Colab.

    python tests/generate_model_report.py

Outputs (all in tests/reports/):
    model_metrics.json            - overwrites the stale 17-class / 4.7% file
    classification_report.txt     - per-class precision/recall/F1
    confusion_matrix.png          - heatmap
    top_confusions.json           - most frequent true->pred mistakes
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "tests" / "reports"


def main():
    y_pred = np.load(REPORTS_DIR / "test_predictions.npy")
    y_true = np.load(REPORTS_DIR / "test_true_labels.npy")
    cm_saved = np.load(REPORTS_DIR / "test_confusion_matrix.npy")

    label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")
    classes = [str(c) for c in label_encoder.classes_]
    n_classes = len(classes)
    labels = list(range(n_classes))
    try:
        spatial_weight = int(joblib.load(BASE_DIR / "best_weight.pkl"))
    except Exception:
        spatial_weight = "?"

    accuracy = float((y_pred == y_true).mean())
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    wgt_p, wgt_r, wgt_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = {
        classes[i]: {
            "precision": round(float(per_p[i]), 4),
            "recall": round(float(per_r[i]), 4),
            "f1": round(float(per_f[i]), 4),
            "support": int(per_s[i]),
        }
        for i in range(n_classes)
    }

    metrics = {
        "model": f"weighted_svm.pkl (HOG 1764 + spatial 26, spatial weight {spatial_weight})",
        "source": "tests/reports/test_predictions.npy vs test_true_labels.npy",
        "n_test_samples": int(len(y_true)),
        "num_classes": n_classes,
        "accuracy": round(accuracy, 4),
        # keep the legacy top-level keys (weighted averages) so old readers work
        "precision": round(float(wgt_p), 4),
        "recall": round(float(wgt_r), 4),
        "f1_score": round(float(wgt_f), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f), 4),
        "weighted_precision": round(float(wgt_p), 4),
        "weighted_recall": round(float(wgt_r), 4),
        "weighted_f1": round(float(wgt_f), 4),
        "classes": classes,
        "per_class": per_class,
    }
    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"accuracy {accuracy:.4f}  macro-F1 {macro_f:.4f}  weighted-F1 {wgt_f:.4f}")
    print(f"wrote {REPORTS_DIR / 'model_metrics.json'}")

    txt = classification_report(
        y_true, y_pred, labels=labels, target_names=classes, zero_division=0, digits=4
    )
    (REPORTS_DIR / "classification_report.txt").write_text(txt)
    print(f"wrote {REPORTS_DIR / 'classification_report.txt'}")

    # confusion matrix from the raw predictions (sanity-check against the saved one)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    if cm.shape == cm_saved.shape and np.array_equal(cm, cm_saved):
        print("confusion matrix matches the saved test_confusion_matrix.npy")
    else:
        print("NOTE: rebuilt confusion matrix differs from saved .npy "
              f"(saved sum={int(cm_saved.sum())}, rebuilt sum={int(cm.sum())})")

    worst = []
    for t in range(n_classes):
        for p in range(n_classes):
            if t != p and cm[t, p] > 0:
                worst.append({
                    "true": classes[t], "pred": classes[p], "count": int(cm[t, p]),
                    "rate": round(float(cm[t, p] / max(1, cm[t].sum())), 3),
                })
    worst.sort(key=lambda d: -d["count"])
    (REPORTS_DIR / "top_confusions.json").write_text(json.dumps(worst[:30], indent=2))
    print(f"wrote {REPORTS_DIR / 'top_confusions.json'}  (top: "
          + ", ".join(f"{w['true']}->{w['pred']}({w['count']})" for w in worst[:5]) + ")")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        plt.figure(figsize=(22, 20))
        sns.heatmap(cm_norm, xticklabels=classes, yticklabels=classes,
                    cmap="magma", square=True, cbar_kws={"shrink": 0.5})
        plt.title(f"Weighted V2 - normalized confusion matrix (test acc {accuracy:.4f})")
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=130)
        print(f"wrote {REPORTS_DIR / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"(skipped confusion_matrix.png: {e})")


if __name__ == "__main__":
    main()
