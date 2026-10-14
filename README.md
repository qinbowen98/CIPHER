# CIPHER

Training pipelines and a focused external-validation pipeline using exactly two frozen ASCVD models.
The repository contains English source/documentation and relative example paths only.

## Final model pair

| Role | Algorithm | Training target | Released weight |
|---|---|---|---|
| Risk-based CIPHER | Logistic regression | Top 50% teacher-risk label | `release/public_models/risk.joblib` |
| Event-based CIPHER | Random forest regression | Event time | `release/public_models/event.joblib` |

Both artifacts are clinical wave 1.0 models with the same 13-feature input contract.
The released files have been rebuilt from reviewed inference state, stripped of source paths and participant metadata,
reloaded and checked for score parity on generated inputs. SHA256 hashes bind each file to its manifest.
The pair matches the frozen identities named in the final age-band analysis configuration; server-side hashes
can be compared with `source_sha256` in `release/model_manifest.json` to confirm identical source files.

## Setup

Use Python 3.10+ in an isolated environment. Historical artifacts use scikit-learn 1.3.2.

```bash
python -m pip install -r requirements-clinical.txt
cp config/pipeline.example.json config/pipeline.local.json
```

Edit the local configuration to point to authorized input files. Relative paths are resolved from the configuration file.
The local configuration, private inputs, run outputs and caches are ignored by Git.

## Reproduce UKB and MUCH

```bash
python pipelines/run.py infer --dry-run
python pipelines/run.py ukb
python pipelines/run.py much
# Or run both cohorts:
python pipelines/run.py infer
```

The comparison includes the two CIPHER predictors plus historical Framingham, China-PAR and CKM2S2-BAG formulas.
No model is trained, recalibrated or selected using either external cohort.
The pipeline writes only aggregate metrics, bootstrap confidence intervals and comparison figures.

UKB CIPHER scores use the original clinical-feature table. Clinical comparator scores use the enriched-feature table.
These inputs must not be substituted for one another. The pipeline checks the locked 40-74 and 60-74 cohort counts.
MUCH uses the completed, standardized clinical table from the final analysis; a raw RDS is not interchangeable with it.
Optional private historical prediction files can verify every cohort member and score without producing sample outputs.

See [input and analysis contracts](docs/FINAL_INFERENCE.md) and [model provenance](docs/FINAL_MODELS.md).
Real cohort results still require authorized input data. No external-result numbers have been fabricated or bundled.

## Training stages

```bash
python -m pip install -r requirements-training.txt
python pipelines/run.py single --dry-run
python pipelines/run.py multi --dry-run
python pipelines/run.py single
python pipelines/run.py multi
```

The original five-fold single-modal and multimodal training designs are retained.
All required modality assets must be available in the protected training working directory.
Multimodal initialization is not automatically connected to architecture-dependent single-modal best weights.
Training outputs remain private and are not inputs to the fixed final-pair inference stage.

## Publish aggregate results and push

Run these commands on `main`. The examples below update `main` directly.

```bash
python pipelines/run.py publish-results
python scripts/check_public_release.py --stage
python scripts/check_public_release.py --staged
git diff --cached --stat
git commit -m "Add frozen two-model UKB and MUCH reproduction pipeline"
git push origin main
```

`publish-results` copies only validated model/replicate-level metric tables into `release/public_results/`.
It does not copy private tables, predictions, figures, logs or input configurations.
Skip that step when cohort analyses have not yet run: code and the two vetted model weights can still be published.
The staging script explicitly includes only receipted model weights and aggregate tables.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/check_public_release.py
```

Tests cover frozen model identity/parity, clinical-score formula parity, source-table separation, shared evaluation cohorts,
participant-table rejection and release receipts. Real UKB/MUCH counts and metrics remain subject to the authorized inputs.
Legacy figure notebooks are retained as source-only references with English comments, cleared outputs and relative input placeholders.
