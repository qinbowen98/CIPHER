# Public release boundary

The user-authorized release now includes training and inference source code that reads protected local inputs. Code may be public; the inputs and participant-level outputs must remain private. This supersedes the earlier notebook-only boundary.

## Publishable assets

- Python/shell-equivalent pipeline code, example configurations and documentation.
- Clinical predictors rebuilt by the reviewed inference-state allowlist, with generated-input prediction parity checks.
- Model-level or bootstrap-replicate-level aggregate performance tables, with strict schemas and unique keys.
- Model identity, software versions, checksums and aggregate release receipts.

## Never publish

Participant IDs, pseudonymized IDs, cohort membership, split membership, clinical records, outcomes/time/event rows, per-participant predictions, genotypes, protein matrices, original images, ECG, embeddings, attention, per-participant IG, sorted IRN reference observations, raw logs, notebook outputs or caches. Removing an ID column alone does not make a participant-level table publishable.

KNN fitted matrices/labels and SVR support vectors retain participant inputs. Those model artifacts are blocked. Raw lifelines Cox objects can retain training outcomes; public Cox representations contain only coefficients and feature centering. Unreviewed model implementations, including XGB in this exporter, are blocked rather than copied. Model coefficients/trees are learned parameters; this boundary concerns exclusion of participant records, and does not claim a formal privacy guarantee for learned weights.

Training and the preserved historical UKB/MUCH scripts may write participant-level outputs. Keep their output directories private. The standardized prepared-table inference defaults to aggregate metrics. Only the exporter-created release with matching receipts and independently checked model states may be staged.

The legacy notebooks remain source-only; no permitted intermediate dataset has been added to them in this change. Each final analysis must document its model collection, cohort, selection and software version separately. wave1 and risk-wave1/clinical-wave3 artifacts must not be mixed.
