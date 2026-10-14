# Frozen model provenance

The final age-band analysis configuration explicitly names these two source artifacts:

```text
ASCVD/top_risk_label_classification/top50/Logistic/final.joblib
ASCVD/direct_survival_regression/event_time/RF/final.joblib
```

The first predicts the frozen top-50%-teacher-risk label using logistic regression.
The second predicts event time using random forest regression. Its saved `risk_direction` determines the high-risk direction;
the inference pipeline must not invert or normalize the score independently.
Both inspected source artifacts record `source_training_wave = 1.0` and scikit-learn 1.3.2.
They are not the later risk-wave1/clinical-wave3 five-seed collection.

The released inference artifacts are named `risk.joblib` and `event.joblib` to keep deployment simple.
The model manifest records the exact source hash, sanitized-file hash, frozen identity and generated-input parity result.
Only fitted pipelines, required transforms, feature contracts and non-participant metadata survive the export.
The original source paths, source cohort summaries and training performance payloads are omitted.

To prepare these two files from a trusted server-side model collection, run:

```bash
python -m pipelines.models.final_pair \
  --risk data/private/original_models/ASCVD/top_risk_label_classification/top50/Logistic/final.joblib \
  --event data/private/original_models/ASCVD/direct_survival_regression/event_time/RF/final.joblib \
  --output release
```

The output directory must be empty. Use the original model-loading environment when exporting original joblib files.
For normal inference, use the two already sanitized release files. Do not add the original model directory to Git.
The original model-loading code may require pandas 1.x compatibility; public weights do not retain source pandas training frames.
