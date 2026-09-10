DAYAW weighted Baybayin model - trained 2026-09-10T06:15:59
Test accuracy: 0.9803  (calibrated 0.9790)
95 classes, 34855 source images, augment x0, kudlit-augment x2, pen-aug x1

Pipeline (must match backend/app.py):
  preprocess: grayscale -> Otsu invert -> remove_small_objects(min_size=8)
    -> tight crop -> pad square (ratio 0.12) -> resize 64x64
    (app.py auto-uses min_size 8 for a 36-feature spatial scaler)
  HOG: orientations=9 ppc=(8, 8) cpb=(2, 2) block_norm=L2-Hys -> 1764
  spatial: overall density(1) + 2x2 grid(4) + 4x4 grid(16) + kudlit stats(5) + body-isolated kudlit mark descriptor above+below(10) -> 36
  scale HOG block with hog_scaler; scale spatial block with spatial_scaler then x 6; concat -> 1800
  SVC(C=20.0, gamma='scale', class_weight='balanced'); label_encoder.inverse_transform for names
  confidence: weighted_svm_calibrated.pkl .predict_proba (isotonic, fit on val)
