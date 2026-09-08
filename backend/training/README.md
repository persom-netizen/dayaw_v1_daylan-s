# Retraining the weighted Baybayin model (Google Colab)

`train_weighted_model.py` takes your harvested character images and produces a
full artifact set that drops straight into `backend/` — the same 6 pkl files the
app loads plus the 3 test `.npy` files the report script reads.

It reproduces the **exact** preprocessing + feature pipeline in
`backend/app.py` / `backend/inference_v2_reference.py`, so if you leave the
`CONFIG` block at its defaults there are **no code changes** on the app side.

---

## 1. Dataset layout

One folder per class, on your Drive. Folder name = class label.

```
MyDrive/ALL_DATASET/
    A/     001.png 002.png ...
    Ba/    ...
    Be/    ...
    Bi/    ...
    ...
    Nga/   ...
```

Accepted image types: png, jpg, jpeg, bmp, webp, tif. Any size; they get
Otsu-binarized, cropped, padded and resized to 64×64 like every other image in
the pipeline. This is the same layout the app already writes to
`backend/open_archival_dataset/<char>/`, so archived samples can be folded in.

`LabelEncoder` sorts the class names, so the integer↔name mapping is
deterministic and `label_encoder.pkl` stays consistent with the app (which reads
`class_names` straight from `label_encoder.classes_`).

---

## 2. Colab cells

```python
# cell 1 — mount
from google.colab import drive
drive.mount('/content/drive')
```

```python
# cell 2 — deps. Pin scikit-image < 0.26 so remove_small_objects() behaves
# exactly like backend/requirements.txt; pin sklearn into the app's range.
!pip -q install "scikit-learn==1.6.1" "scikit-image<0.26" opencv-python-headless joblib
```

```python
# cell 3 — paste the ENTIRE contents of train_weighted_model.py here and run.
#   It only DEFINES things (the `if __name__ == "__main__"` guard is False in a
#   notebook), so nothing runs yet and there is no argparse error. No output.
```

```python
# cell 4 — train (baseline, drop-in compatible)
run(data="/content/drive/MyDrive/ALL_DATASET",
    out="/content/drive/MyDrive/WEIGHTED_MODEL_V3")

# kudlit-tuned run, after you've seen the baseline + per-class counts:
# run(data="/content/drive/MyDrive/ALL_DATASET",
#     out="/content/drive/MyDrive/WEIGHTED_MODEL_V3_aug",
#     augment=2, weight_grid="2,4,6,8,10,12,15,20,25")
```

```python
# cell 5 — look at the results
import json
m = json.load(open("/content/drive/MyDrive/WEIGHTED_MODEL_V3/metrics.json"))
print("test acc:", m["test_accuracy"], " macro-F1:", m["macro_f1"],
      " spatial weight:", m["spatial_weight"])
med = sorted(m["class_counts"].values())[len(m["class_counts"]) // 2]
print("thin classes:", {k: v for k, v in m["class_counts"].items() if v < 0.5 * med})
print("top confusions:", m["top_confusions"][:12])
```

> Prefer a file? In cell 3 make the first line `%%writefile train_weighted_model.py`
> above the pasted script, then in cell 4 run
> `!python train_weighted_model.py --data "…" --out "…"`.

Runtime: feature extraction is parallel and fast (a few minutes for ~30k
images). The RBF SVM fit and the spatial-weight search are the slow part
(~10–40 min total on a Colab CPU, depending on dataset size and `augment`). A
GPU runtime does **not** help `sklearn`'s SVC.

`run(...)` accepts: `data`, `out`, `augment`, `C`, `search_c="10,20,50"`,
`weight_grid="2,4,6,8,10"`, `fixed_weight`, `limit_per_class` (smoke test),
`n_jobs`.

---

## 3. Install into the app

From the `--out` folder, copy:

| file(s) | destination |
|---|---|
| `weighted_svm.pkl`, `weighted_svm_calibrated.pkl`, `hog_scaler.pkl`, `spatial_scaler.pkl`, `best_weight.pkl`, `label_encoder.pkl` | `dayawanalisa/backend/` |
| `test_predictions.npy`, `test_true_labels.npy`, `test_confusion_matrix.npy` | `dayawanalisa/backend/tests/reports/` |

Then:

```bash
cd dayawanalisa/backend
python tests/generate_model_report.py     # refresh model_metrics.json, confusion_matrix.png, ...
```

All six pkl names match what `load_joblib_artifact` already looks for, and the
spatial weight is read from `best_weight.pkl` at startup — nothing in `app.py`
needs editing. Restart the Flask server; the log line should read
`confidence = isotonic-calibrated`.

Keep the rest of the `--out` folder (`X.npy`, `splits/`, `metrics.json`,
`MANIFEST.json`) on Drive — `backend/calibrate_model.py` can re-fit the
calibrator from `splits/X_val.npy` if you ever tweak it, and `splits/paths_*`
let you trace misclassifications back to source images.

---

## 4. Fixing the kudlit errors (`Ko`↔`K`, `Ni`↔`Ne`, …)

That confusion is the model's #1 error class (base glyph vs its e/i / o/u kudlit
variant). Options, cheapest first — the first three need **no `app.py` change**:

### a. Balance the dataset
The script prints per-class counts and warns about any class below 50% of the
median. Kudlit variants (`Be Bi Bo Bu`, `De Di Do Du`, …) are usually the
thin ones. Add samples there first — it's the highest-leverage fix.

### b. Wider spatial-weight search
The 5 "kudlit" spatial features (component count + the secondary-component
centroid offset `dy,dx` = where the kudlit sits) are the model's positional
signal, but at weight 6 they're swamped by 1764 HOG dims. Push the search up:

```
--weight-grid "2,4,6,8,10,12,15,20,25"
```

`best_weight.pkl` carries the chosen value, so the app picks it up automatically.

### c. Mild augmentation
```
--augment 2        # or 3
```
Adds small rotations (±5°), scale (0.92–1.08), ±3 px shifts and stroke
thickness jitter — kept gentle on purpose so a kudlit dot doesn't rotate/shift
out of its cell and `e`↔`i` don't blur together. Roughly `1 + N`× the training
set, so `--augment 2` ≈ 2–3× the SVM fit time.

### d. Optional `--search-c "10,20,50,100"`
Small grid over the SVM's `C`. Won't fix kudlit on its own but worth a pass once
(a) and (b) are done.

### Bigger changes (these **do** need matching `app.py` edits)

| change | why it helps kudlit | app.py edits |
|---|---|---|
| `TARGET_SIZE = 96` | a kudlit dot goes from ~a few px to ~2× — HOG can actually see it | set `TARGET_SIZE = 96` in `app.py`; `HOG_FEATURE_LEN` becomes `(96/8−1)²·2²·9 = 2916` — update that constant and `training`'s `N_HOG_FEATURES`. Retrain (scalers change dim). |
| **(done)** +2 features: top-strip / bottom-strip ink fraction | "is there a mark above / below the body" | `kudlit_strip_features` in both files; `N_SPATIAL_FEATURES` 26→28 |
| **(done)** +4 features: kudlit **mark shape** — aspect ratio + relative width of the mark in the top band and the bottom band | this is the **dash-vs-dot** signal (`Ne`↔`Ni`, `Nu`↔`No`). A dash → aspect > ~2, wide; a dot → aspect ~1, narrow. Density / position features fire the same for both. | `kudlit_shape_features` in **both** `train_weighted_model.py` and `app.py` (keep identical), appended last; `N_SPATIAL_FEATURES` 28→**32**. `app.py` reads the length back from the scaler, so a 26/28/32-feature model all drop in. **Retrain to activate.** |
| two-stage head: base consonant, then kudlit | removes the imbalance problem entirely | larger `app.py` change (two models); only if the above aren't enough |

Keep the training script's `CONFIG` block and `app.py`'s constants identical —
that invariant is what makes the artifacts drop-in.

---

## 5. Flags reference

```
--data PATH            ALL_DATASET root (required)
--out PATH             artifact output dir (required)
--augment N            augmented copies per training image (default 0)
--C FLOAT              SVM C (default 20)
--search-c "a,b,c"     grid-search C on a val subsample instead
--weight-grid "a,b,c"  override the spatial-weight search grid
--fixed-weight N       skip the weight search, use N
--limit-per-class N    only read N images/class (smoke test)
--n-jobs N             parallel workers for feature extraction (default all cores)
```
