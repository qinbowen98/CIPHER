# CIPHER

CIPHER combines genetic, proteomic, ECG, and fundus information for cardiovascular risk prediction. This repository contains preprocessing and training pipelines, figure notebooks, and two frozen clinical transfer models for UKB and MUCH evaluation.

## Repository

```text
pipelines/
  0_preprocess/                    SNP, protein, fundus, and ECG preparation
  1_single_modality/               Five-fold training and protein ablation
  2_multi_modality/                Multimodal survival-model training
  3_CIPHER_clinical_applications/  Frozen-model scoring and clinical evaluation
  metrics.py                      Event-status AUROC
  run.py                          Pipeline entry point
notebooks/
  main_figures/
  supplementary_figures/
release/
  public_models/
    risk.joblib
    event.joblib
environment.yml
```

## Installation and configuration

```bash
conda env create -f environment.yml
conda activate cipher
python pipelines/run.py init
```

Set input and output locations in `pipeline.local.json`. Relative paths resolve from the local configuration directory. SNP preprocessing also uses PLINK 1.9, available on `PATH` or specified by the `plink` option. GPU training uses PyTorch and a compatible CUDA installation.

The stages below run in order. Use `--config` to select another configuration and `--dry-run` to inspect planned preprocessing or training jobs.

## Source-cohort endpoint definitions

The following definitions apply to the CMCS source cohorts. CHD was classified using these MONICA-informed study rules; an event was recorded when any of the following conditions was met:

1. The primary discharge diagnosis was ICD-10 I21 or I22.
2. Discharge disposition was coded as `5`, and the primary discharge diagnosis was I20, I23, I24 or I25.
3. Discharge disposition was coded as `5`, at least one discharge diagnosis was I46, I50, R96, R98 or R99, and at least one discharge diagnosis was in I20–I25.

These categories cover definite acute myocardial infarction, coronary heart disease death, and sudden death in participants with a history of coronary heart disease. Discharge disposition code `5` refers to the coding used in the source hospital records.

Ischemic stroke (`fistroke`; pipeline endpoint `istroke`) was defined by a primary discharge diagnosis of I63 or I69.3, excluding I63.903.

Ischemic cardiovascular disease (`fICVD`; pipeline endpoint `ICVD`) was the composite of CHD and ischemic stroke: `fICVD = CHD OR fistroke`.

UKB and MUCH use their cohort-specific outcome ascertainment and field mappings; the source-cohort discharge rules above are not applied to those cohorts by default.

## 0. Preprocess modality inputs

```bash
python pipelines/run.py preprocess --target snp
python pipelines/run.py preprocess --target soma
python pipelines/run.py preprocess --target fundus
python pipelines/run.py preprocess --target ecg
```

| Input | Processing | Output |
| --- | --- | --- |
| QC'd PLINK BED/BIM/FAM files, discovery metadata, and endpoint labels | Define the discovery population; run additive logistic association for CHD and ischemic stroke; take the union of variants having P < 5e-5 for CHD or ischemic stroke | Selected genotype files, SNP list, and selection manifest |
| Protein assay tables and optional feature panel | Parse participant and visit identifiers, retain ordered `seq.` features, and align training and validation columns | Prepared protein tables and feature manifest |
| Fundus images | Center-crop to a square and resize to 224 × 224 | Prepared RGB images |
| Twelve-lead ECG XML | Retain up to 1,000 samples per lead, apply the selected amplitude normalization, and zero-pad shorter leads | Fixed-length ECG XML and preprocessing manifest |

ECG normalization options are `zscore`, `maxabs`, and `none`. Normalization is applied to retained samples before padding. Training reads the prepared ECG without repeating normalization. Run preparation once for each input dataset; existing prepared inputs can be configured directly.

Fundus tensor normalization is configured separately for training: `imagenet` uses mean [0.485, 0.456, 0.406] and SD [0.229, 0.224, 0.225]; `half` uses mean and SD 0.5; `none` retains the scaled image tensor. Select the normalization associated with the checkpoint being evaluated.

The example protein inputs are already log1p transformed. The min-max comparator therefore scales those values directly. Rank normalization and min-max parameters are fitted on the training fold and applied unchanged to validation and evaluation data. `soma_input_transform` records whether the supplied values are raw or log1p transformed.

Protein panels are stored in `pipelines/0_preprocess/soma_panels/`:

| Panel | Source | Matching |
| --- | --- | --- |
| `replicate_1` | NCVR panel | Gene symbol |
| `replicate_2` | Deng et al., Cell | Gene symbol |
| `svm` | Union of five-fold SVM-selected features | Complete assay feature name |
| `cardio` | Cardio panel | Gene symbol |

The example training configuration selects `replicate_2`. Matched assay features retain their input-column order; assay features and collapsed gene symbols are counted separately.

## 1. Train single-modality models

```bash
python pipelines/run.py single
```

This stage prepares modality-specific participant sets and uses participant-grouped five-fold cross-validation. SNP, protein, and ECG Transformer models search hidden dimensions 64/128/256, encoder depths 1/2/3, learning rates 1e-4/3e-4/1e-3, and batch sizes 32/64/128/256. ECG BiLSTM searches hidden size/depth, and the CNN searches filter count. Mean validation C-index across all five folds selects the architecture and optimization settings. Mean validation C-index selects the hyperparameter setting; the highest-validation-C-index fold within that setting supplies the saved checkpoint and its fitted preprocessing. Evaluation then uses the held-out external set.

The example runs seeds 42, 66, and 100, learning rates 1e-4/3e-4/1e-3, and batch sizes 32/64/128/256. Models include molecular Transformer, MLP-Cox and PCA-Cox, ECG Transformer/BiLSTM/CNN, and fundus backbones. Outputs include checkpoints, predictions, embeddings, training curves, and aggregate evaluation metrics.

```bash
bash pipelines/1_single_modality/run_soma_ablation.sh
python pipelines/1_single_modality/collect_ablation.py \
  --root private_runs/training/soma_ablation --output figure_data/ablation
```

The protein ablation compares rank normalization with min-max scaling of log1p inputs, then compares the four feature panels using MLP-Cox. Conditions share the configured seeds and tuning budget. Each selected setting exports all five validation-fold scores, their mean, and sample standard deviation.

Single-modality Transformer interpretation uses evaluation samples and saves signed per-sample integrated gradients in batch NPZ files. Each batch retains aligned identifiers, input tensors, follow-up and event labels. A feature-order manifest and mean-absolute attribution summary accompany these files. Compatibility outputs include per-sample `*_IG.csv` (`ig_abs`: absolute attribution), `*_IG_input.csv` (`input_value` and legacy `ig_abs`: model-input values), and per-sample ECG attribution text files. These files remain in the private run directory. Fundus Grad-CAM uses the evaluation dataset.

## 2. Train multimodal models

```bash
python pipelines/run.py multi
```

This stage aligns the configured modalities by participant and visit, constructs modality encoders and the fusion model, and trains with a Cox survival objective. Modality combinations, endpoints, seeds, epochs and batch size are selected in `training.multi`.

Training saves fold checkpoints, validation summaries, held-out risk predictions and embeddings. When `save_ig` is enabled, it also saves per-sample attributions for each included modality. The configured modality inputs are shared with stage 1; encoder construction and training are defined in `2_multi_modality/model.py` and `train.py`.

## 3. Evaluate clinical transfer models

```bash
python pipelines/run.py infer
# Individual cohorts:
python pipelines/run.py ukb
python pipelines/run.py much
```

| Model | File | Source target |
| --- | --- | --- |
| Risk-based Logistic model | `release/public_models/risk.joblib` | Top-50% ASCVD risk label |
| Event-based random forest | `release/public_models/event.joblib` | ASCVD event-time regression |

The model loader validates the model roles and file checksums. Both models use the ordered clinical features `age_feat`, `sex`, `SBP`, `DBP`, `TC`, `TG`, `HDLC`, `LDLC`, `BMI`, `height`, `weight`, `waist`, and `smoking`.

This stage loads the frozen predictors, aligns prepared cohort features and outcomes, calculates model and clinical-comparator scores, and performs 500 paired bootstrap evaluations. UKB ages 40–74 form the common eligibility range of the clinical comparators; ages 60–74 are evaluated as a separate subgroup. MUCH is evaluated separately.

Outputs are `aggregate_metrics.csv`, `aggregate_summary.csv`, a run manifest and comparison figures. C-index uses observed follow-up and event status. `event_auroc` uses observed binary event status and the fixed risk score.

To calculate the same metrics from existing prediction files:

```bash
python -m pipelines.3_CIPHER_clinical_applications.inference.evaluate_predictions \
  --manifest predictions.local.json \
  --output private_runs/event_metrics
```

The manifest is a JSON list. Each item specifies `prediction_file`, `endpoint`, `model`, and `split`; optional fields include `strategy`, `cohort`, and `event_column`/`risk_column`/`time_column`. Prediction scores must increase with risk. The evaluator exports aggregate point estimates and bootstrap results.

### Six-strategy evaluation in UKB and MUCH

```bash
python -m pipelines.3_CIPHER_clinical_applications.inference.six_strategies \
  --cohort UKB --table data/private/ukb_six_strategy_features.csv \
  --models-dir private_runs/clinical_models --output private_runs/six_strategies \
  --id-column eid --event-column event --time-column time --age-range 40 74

python -m pipelines.3_CIPHER_clinical_applications.inference.six_strategies \
  --cohort MUCH --table data/private/much_six_strategy_features.csv \
  --models-dir private_runs/clinical_models --output private_runs/six_strategies \
  --id-column PATIENT_SN --event-column ASCVD_EVENT --time-column FOLLOWUP_YEARS
```

`models-dir` contains frozen `final.joblib` artifacts for risk regression, risk-to-Cox, top-risk-label classification, direct event-time regression, direct observed-event classification, and clinical Cox. The evaluator requires all six strategies and uses their stored preprocessing. It exports aggregate C-index and observed event-status AUROC with paired bootstrap results. Figure 5 reads these tables from `figure_data/fig5_final/six_strategies/<cohort>/aggregate_metrics.csv`.

## Predict in your own cohort

```bash
python -m pipelines.3_CIPHER_clinical_applications.predict \
  --input cohort_features.csv \
  --output private_runs/cohort_predictions.csv \
  --id-column participant_id
```

The input is a CSV or Parquet table containing the 13 model features listed above. Outcomes are not required. Supply numeric clinical variables on the source model's measurement scales and categorical encodings; the fitted imputation and scaling steps are applied by the saved pipelines. The prediction command preserves row order and optionally an identifier column. It writes `risk_label_probability` and `event_risk_score`; larger values indicate higher model-assigned risk. The first predicts the source top-risk label, and the second is the direction-adjusted event-time regression score. Neither is a calibrated fixed-horizon event probability.

## 4. Generate figures

```bash
jupyter lab
```

Open a notebook and run its cells in order. Each analysis cell has an English heading identifying the manuscript figure or supporting analysis. Data loading, calculations, and plotting are written directly in the notebook.

| Notebook | Analysis |
| --- | --- |
| `main_figures/fig1.ipynb` | Cohort modality availability and intersections |
| `main_figures/fig2.ipynb` | Molecular model evaluation, attributions, loci, and enrichment |
| `main_figures/fig3.ipynb` | ECG/fundus evaluation, attribution, and protein associations |
| `main_figures/fig4.ipynb` | Multimodal performance, event enrichment, and representation analyses |
| `main_figures/fig5.ipynb` | Clinical transfer-model evaluation |
| `supplementary_figures/figs0.ipynb` | Training and validation curves |
| `supplementary_figures/figs1.ipynb` | Cohort characteristics and modality availability |
| `supplementary_figures/figs2.ipynb` | Protein ablation and SNP coverage |
| `supplementary_figures/figs3.ipynb` | Unimodal validation and cytoband attribution |
| `supplementary_figures/figs4.ipynb` | Molecular attribution and GO enrichment |
| `supplementary_figures/figs5.ipynb` | Protein associations, enrichment, and ECG examples |
| `supplementary_figures/figs6.ipynb` | Representation and attribution comparisons |

Notebooks read the aggregate figure tables in `figure_data/` by default. Set `CIPHER_ANALYSIS_ROOT` to use an authorized analysis directory with the original relative input structure. Plots and generated tables are written under `figure_outputs/<notebook>/`. Individual example selection uses the local `CIPHER_EXAMPLE_PID` environment variable.

AUROC panels consume recalculated `event_auroc` results. Aggregate source tables and participant-level predictions serve different analyses; configure the inputs referenced by each cell accordingly.

## Data access

Participant inputs, outcomes, predictions, embeddings, individual attributions, local configurations, and run outputs are excluded from Git. The distributed research assets are code, feature-only panel lists, aggregate figure source tables, notebooks with cleared outputs, and the two clinical model files.
