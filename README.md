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

### Figure and table map

`C1`, `C2`, etc. count code cells only, starting at 1 in each notebook. Run C1 to set the input and output directories. The cell groups below locate each analysis; tables and plots use the variables and imports prepared by preceding analysis cells. The risk-averaging calculation in `fig3.ipynb` C16 produces the ICVD metric table consumed by C15.

```bash
jupyter lab
```

Open the linked notebook and run the relevant cell group. Input paths are relative to `INPUT_ROOT`, which defaults to `figure_data/`; outputs are relative to `figure_outputs/<notebook>/`. `{endpoint}`, `{modality}`, `{metric}` and `*` denote the values or file patterns selected in the code. Participant records, predictions, inputs, embeddings and individual attributions are used within authorized analysis environments. Published aggregate tables and reference annotations retain the same relative layout.

For clinical characteristics, C2 of `figs1.ipynb` calculates statistics from the participant table selected by `CIPHER_CLINICAL_PARTICIPANT_TABLE`. C3/C4 export the aggregate clinical group summary. These are separate input routes. Inspection cells display intermediate tables; schematic study diagrams correspond to the pipeline stages described above.

#### Shared inputs and columns

| Input family | File or selector | Columns / array contents |
| --- | --- | --- |
| CV summary | `STAGE1_train_Data/analysis_summary_1201/cv_best_fivefold_per_model_endpoint.csv` | `modality`, `endpoint`, `model_tag`, `n_folds`, `mean_valid_ci`, `std_valid_ci`; `curve_files` identifies source curves |
| Predictions | `code_02_02_save/**/preds/*_pred.csv`; multimodal predictions under `fig3/visit02/` and `fig3/longitudinal/` | `pid` or `ID`, `time`, `event`, `risk` |
| Molecular IG | `code_02_02_save/**/explanations/*_IG.csv` | `pid`, `feature`, `ig_abs` or `ig`; SNP/gene mapping is applied separately |
| Protein inputs | `code_3_3_save/soma_single/*_IG_input.csv` | `pid`, `feature`, `input_value`; historical files use `ig_abs` for the protein input value |
| Variant annotation | `CHD_SNP_variant_gene_with_symbol.csv`; `fig2/SNP/CHD_IG_final.csv` for SNP/gene mapping | Variant/SNP identifier, chromosome, position, gene symbol; cell-specific aliases are resolved by the loaders |
| GWAS and cytobands | Files selected by `GWAS_CATALOG`/`GWAS_FILE` and `CYTOBAND_FILE` | GWAS trait, SNP, chromosome, position, P value and study identifiers; chromosome-band intervals |
| Protein associations and enrichment | Correlation CSVs and GO/GSEA result files selected in each cell | Feature/gene, correlation coefficient, P/FDR; GO `Term`, `Adjusted P-value`, `Genes`; GSEA term, `NES` and FDR |
| Embeddings and interpretation arrays | `sample_embeddings_ICVD_test.npy`, `sample_risk_ICVD_test.npy`, optional `sample_attn_ICVD_test.npy`; sample metadata and per-sample IG/Grad-CAM files | Participant/visit alignment, modality/fused representations, risk, attribution arrays |
| Clinical participant table | `CIPHER_CLINICAL_PARTICIPANT_TABLE` for `figs1.ipynb` C2 | `group`, `ageEntry`, modality indicators, `fCHD`, `fistroke`, `fICVD`, sex/risk-factor indicators and clinical measurements |
| Clinical group summary | `figs1_final/clinical_oncoprint_four_cohorts_baseline_q70_with_modality/baseline_clinical_oncoprint_group_summary.csv` | `group`, `n_clinical_matched`, age summaries, `*_n_present`, `*_rate` |
| Clinical strategy results | `fig5_final/outputs_final_valid_select_more_models_ASCVD_toprisk*/ALL_model_results_long.csv`; `fig5_final/six_strategies/{ukb,much}/aggregate_metrics.csv` | `cohort`, `endpoint`, `strategy`, `split`, bootstrap/replicate identifier, `cindex`, `event_auroc`; the loader also reads `auc_all_event` |

#### Main figures

| Figure / analysis | Code cells | Inputs | Primary tables / plots |
| --- | --- | --- | --- |
| Figure 1c: modality overlap, UpSet | [fig1.ipynb](notebooks/main_figures/fig1.ipynb) C2 | Training prediction participant IDs, grouped by modality | `figs1_final/CHD_trainset_four_modal_sample_upset/*_counts_exact.csv`, `*_upset_bw_wide.pdf` |
| Figure 1c: BD modality overlap | [fig1.ipynb](notebooks/main_figures/fig1.ipynb) C3 | Training predictions (`pid` or `ID`); modality membership | `fig1_final/CMCS_trainset_four_modal_overlap/CMCS_*_counts_exact_observed.csv`, `CMCS_*_venn_observed.pdf` |
| Figure 1c: SG modality overlap | [fig1.ipynb](notebooks/main_figures/fig1.ipynb) C4 | `fig1_final/SingleModal_Run_printonly/*/data/test_pid_pool.csv` (`pid`) | `fig1_final/SingleModal_Run_printonly/four_modal_test_overlap/*_summary_observed.csv`, `*_venn_observed.pdf` |
| Figure 2b: molecular architecture selection and tables | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C2, C3, C4 | CV summary; molecular modalities | `fig2_final/model_performance_by_disease.tex`; CV performance PDFs and displayed summary |
| Figure 2c: molecular evaluation and legend | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C5, C8, C9 | Molecular prediction tables | `fig2_final/SNP_Proteomics_model_performance_summary.csv`, `SNP_Proteomics_Cindex_AUC_combined_overall_gain.pdf`, `legend_models.pdf` |
| Figure 2c: molecular evaluation tables | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C6, C7 | `fig2_final/SNP_Proteomics_model_performance_summary.csv`; evaluation metrics | `fig2_final/SNP_Proteomics_model_performance_tables.tex`; displayed summary |
| Extended Data Figure 4a: gene-level SNP attribution | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C10 | SNP IG and predictions; `fig2/SNP/CHD_IG_final.csv` (`SNP`, `gene`) | `fig2_final/{endpoint}_top_gene_origin_boxplot_with_sig.pdf` |
| Figure 2g: prioritized plasma proteins | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C11, C22 | Soma IG and predictions | `fig2_final/{endpoint}_top_protein_boxplot_with_sig.pdf` |
| Figure 2e: prioritized SNP loci | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C12 | SNP IG, predictions and SNP-to-gene mapping | `fig2_final/{endpoint}_top_loci_origin_boxplot_with_sig.pdf` |
| Figure 2f: rs1327115596 and rs3782805 locus views | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C13, C14, C15, C20, C21 | SNP IG, predictions, variant annotation and GWAS catalog | `fig2_final/locus_plots/*_locus_pretty*.pdf`, `*_selected_GWAS_tracks*.csv`; locus preparation and displayed GWAS associations |
| Figure 2d: cytoband attribution and GWAS annotations | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C18, C19 | SNP IG, predictions, variant annotation, cytobands and GWAS catalog | `fig2_final/chromosome_IG_combined_manual_studies_all_hits_final/`: cytoband/SNP/GWAS CSVs, heatmap and colorbar PDFs |
| Figure 2i: SNP versus protein gene attribution | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C23 | `shared_gene_tables/{CHD,iStroke}_shared_SNP_SOMA_gene_rank.csv` (`gene`, `snp_reverse_rank`, `soma_reverse_rank`) | `fig2_final/Fig_CHD_iStroke_SNP_vs_SOMA_final.pdf` |
| Figure 2j: shared and endpoint-specific molecular features | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C24 | Same shared-gene ranking tables; gene identifiers | `fig2_final/Fig_CHD_iStroke_venn.pdf`, `GO_{name}.csv` |
| Supporting molecular pathway analysis | [fig2.ipynb](notebooks/main_figures/fig2.ipynb) C16, C17 | `fig2_final/GSEA_unbiased/GSEA_{CHD,iStroke}_raw.csv`, corresponding `.rnk` files | `fig2_final/GSEA_unbiased/barstyle_plots_from_existing_results/`: enrichment PDFs |
| Figure 3b: ECG/fundus architecture selection and tables | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C2, C3, C4 | CV summary; ECG and fundus modalities | `fig3_final/ecg_fundus_disease_performance.tex`; CV PDFs and displayed summary |
| Figure 3c: ECG/fundus evaluation | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C5 | ECG/fundus prediction tables | `fig3_final/ECG_Fundus_model_performance_summary.csv`, `{modality}_{metric}_with_redline_overall_gain.pdf`, model legends |
| Figure 3c: evaluation tables | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C6, C7 | Results assembled by C5; evaluation metrics | `fig3_final/ECG_Fundus_test_performance.tex`; displayed summary |
| Figure 3d: cohort fundus Grad-CAM differences | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C8 | Fundus Grad-CAM arrays and predictions | `fundus_global_gradcam_high_minus_low/{endpoint}_high_minus_low_*.pdf` |
| Figure 3e: representative ECG participant lookup | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C9 | ECG predictions (`pid`/`ID`, `risk`); locally selected example | Displayed prediction/risk row |
| Figure 3i: protein association shifts | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C11 | Protein inputs and modality prediction tables | `fig3_final/protein_average_minus_single_modal_score_rankplot_ICVD_only/*_score_table.csv`, `*_rankplot_top50_label10.pdf` |
| Figure 3g: protein overlap and enrichment | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C12, C17, C18, C19 | Protein inputs and modality predictions; derived correlations and GO results | `fig3_final/protein_four_modal_risk_{venn,upset,4set_venn}/`: correlation, intersection and GO tables; Venn/UpSet PDFs and legend |
| Figure 3h: calculate averaged ICVD risk | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C16 | Four unimodal ICVD prediction tables | `code_3_3_save/toy_multimodal_risk_combination/ICVD_toy_combined_metrics.csv` and aligned prediction outputs |
| Figure 3h: performance gains from risk averaging | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C15 | `code_3_3_save/toy_multimodal_risk_combination/{endpoint}_toy_combined_metrics.csv`; metric gains | `fig3_final/improvement/toy_combined_*_improvement_barh_fixed_order.pdf` |
| Figure 3j: directional pathway comparison | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C20, C21 | Protein inputs, modality/averaged-risk predictions and Hallmark gene sets | `code_3_3_save/protein_risk_correlation_gsea_directional/`: correlation/GSEA tables and NES comparison PDFs |
| Figure 3j: full-ranking and selected pathway plots | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C22, C23 | Protein inputs/predictions; `code_3_3_save/protein_risk_correlation_gsea_final/cached_gsea/*_Hallmark_gsea.csv` | Cached GSEA CSVs; `fig3_final/final_selected_pathway_plots/{endpoint}_selected_pathways.pdf`; model-comparison PDFs |
| Supporting ECG/fundus molecular analyses | [fig3.ipynb](notebooks/main_figures/fig3.ipynb) C10, C13, C14 | Protein inputs/predictions; `code_3_3_save/high_vs_low_risk_feature_analysis/`; protein correlations and GO results | Quadrant scatter/GO plots, `fig4_final/*_protein_selected_GO_4quadrants.pdf`, protein heatmaps and enrichment CSVs |
| Figure 4b: modality-combination evaluation | [fig4.ipynb](notebooks/main_figures/fig4.ipynb) C2, C3 | `fig3/visit02/` prediction tables | `fig4_final/fig4_visit02_{cindex,event_auroc}_barplot_by_combo.pdf`; performance summaries |
| Figure 4b: longitudinal model comparison | [fig4.ipynb](notebooks/main_figures/fig4.ipynb) C4, C5 | `fig3/longitudinal/` prediction tables | `fig4_final/fig4_longitudinal_{cindex,event_auroc}_barplot_by_combo.pdf`; displayed model results |
| Figure 4c: integration-strategy comparison | [fig4.ipynb](notebooks/main_figures/fig4.ipynb) C6, C7 | `code_3_3_save/compare_external_mean_fivefold_fullcox_visit02/external_comparison_cindex_auc_summary.csv` (`endpoint`, `method`, `n_analyzed`, `n_event`, `cindex`, `event_auroc`) | `fig4_final/figures_visit02_style_istroke_ascvd*/`: separate and combined strategy PDFs |
| Figure 4d: event enrichment and capture | [fig4.ipynb](notebooks/main_figures/fig4.ipynb) C8, C9 | Multimodal and averaged-risk prediction tables | `fig4_final/ICVD_top_risk_capture_rate_combined_update/`: event/capture CSVs and plots |
| Figure 4d: risk-segment tests | [fig4.ipynb](notebooks/main_figures/fig4.ipynb) C10, C11, C12 | Same paired participant predictions and observed outcomes | `fig4_final/ICVD_top_risk_segment_*` and `ICVD_dual_metric_segment_tests/`: event-rate/capture test CSVs |
| Figure 4e: protein/pathway comparison | [fig4.ipynb](notebooks/main_figures/fig4.ipynb) C13, C14 | `ICVD_protein_multimodal_top5_vs_direct_mean_top5_all_diff.csv`; `ICVD_GSEA_groupwise_NES_change_multimodal_minus_direct.csv` | `fig4_final/ICVD_GO_BP_FDR005_and_selected_GSEA_NES_subplots/`, `ICVD_selected_GSEA_NES_barplots/`: GO CSVs and pathway PDFs |
| Figure 4f: component correlations with risk | [fig4.ipynb](notebooks/main_figures/fig4.ipynb) C18 | Embeddings, participant metadata and risk arrays | `fig4_final/embeddingv4/*_pearson_corr_v02.csv`, `*_smooth_curve_v02.pdf` |
| Supporting representation views | [fig4.ipynb](notebooks/main_figures/fig4.ipynb) C15, C16, C17 | Same embeddings/metadata; optional attention arrays; baseline age/sex table | `fig4_final/{embedding_v2,embedding_v3,embeddingv4}/`: diffusion-map/PCA views and correlation tables |
| Figure 5h: clinical strategy evaluation | [fig5.ipynb](notebooks/main_figures/fig5.ipynb) C2, C3 | Clinical strategy result tables and optional UKB/MUCH six-strategy aggregates | `fig5_final/outputs_final_valid_select_more_models_ASCVD_toprisk/ALL_model_results_long_merged_CHD_istroke_ASCVD.csv`; `boxplot_redraw/{cohort}/{split}_{metric}.pdf`, `summary.pdf` |

#### Extended-data and supporting analyses

| Figure / analysis | Code cells | Inputs | Primary tables / plots |
| --- | --- | --- | --- |
| Figure 2b / Extended Data Figure 3a: molecular training-curve summary | [figs0.ipynb](notebooks/supplementary_figures/figs0.ipynb) C2 | Molecular `*_train_curve.csv` (`epoch`, `valid_ci`; training columns where present) | `figs2_final_validation/best_validation_valid_ci_lr{LR}_bs{BS}.csv`, molecular validation PDFs |
| Figure 3b / Extended Data Figure 5a: imaging/signal training-curve summary | [figs0.ipynb](notebooks/supplementary_figures/figs0.ipynb) C3 | ECG/fundus training curves | `fig3_final_validation_all_lr_bs/ECG_Fundus_best_validation_valid_ci_all_lr_bs.csv`, validation PDFs |
| Supplementary Data Table 1: participant-level clinical characteristics | [figs1.ipynb](notebooks/supplementary_figures/figs1.ipynb) C2 | Participant clinical table selected by `CIPHER_CLINICAL_PARTICIPANT_TABLE` | `extended_table1_latex_final/Extended_Table1.csv`, `Extended_Table1.tex` |
| Clinical group-summary table exports | [figs1.ipynb](notebooks/supplementary_figures/figs1.ipynb) C3, C4 | Clinical group summary; `n_clinical_matched`, age summaries, modality counts and clinical rates | `extended_table1_latex/Extended_Table1_matrix.csv`, `Extended_Table1.tex`; `extended_table1_latex_final/Extended_Table1.csv`, `Extended_Table1.tex` |
| Extended Data Figure 2b: SNP coverage and table | [figs2.ipynb](notebooks/supplementary_figures/figs2.ipynb) C2, C3 | Variant annotation, cytobands and GWAS catalog | `figs2_final/snp_cytoband_coverage_preprocessing_onerow/`: coverage PDF, three source CSVs and `prediction_SNP_cytoband_coverage_table.tex` |
| Extended Data Figure 2c: protein normalization | [figs2.ipynb](notebooks/supplementary_figures/figs2.ipynb) C4, C5 | `figs2_final/IRN_vs_normlog1p_summary.csv` (`method`, `endpoint`, `mean_cindex`, `sd_cindex`) | `IRN_vs_normlog1p_barplot.pdf`, `IRN_vs_normlog1p_summary.csv`; displayed summary |
| Extended Data Figure 2d: protein-panel performance | [figs2.ipynb](notebooks/supplementary_figures/figs2.ipynb) C6 | `figs2_final/FINAL_SELECTION_TABLE.csv` (`selection`, `endpoint`, `mean`, `std`) | `FINAL_SELECTION_BARPLOT.pdf`, `FINAL_SELECTION_BARPLOT_legend.pdf` |
| Supporting protein-panel GO/overlap analysis | [figs2.ipynb](notebooks/supplementary_figures/figs2.ipynb) C7, C8 | Panel-specific Soma feature/IG tables (`feature`); gene symbols derived in the cells | `GO_enrichment_final/`: GO tables/plots; panel-overlap plot |
| Extended Data Figures 3a and 5a: architecture CV | [figs3.ipynb](notebooks/supplementary_figures/figs3.ipynb) C2 | CV summary | Molecular, ECG and fundus cross-validation PDFs |
| Figure 2c: molecular evaluation layout | [figs3.ipynb](notebooks/supplementary_figures/figs3.ipynb) C3 | Molecular prediction tables | `figs2_final/SNP_Proteomics_model_performance_summary.csv`, `SNP_Proteomics_Cindex_AUC_combined_overall_gain.pdf` |
| Extended Data Figure 3b: genome-wide cytoband attribution | [figs3.ipynb](notebooks/supplementary_figures/figs3.ipynb) C4, C5, C6 | SNP IG/predictions, variant annotation, cytobands and GWAS catalog | `fig2_final/chromosome_IG_combined_manual_studies_all_chromosomes_with_chrX_public/` and `figs3_final/chromosome_IG_vertical_2rows_all_chromosomes/`: attribution/GWAS tables, heatmaps and colorbars |
| Extended Data Figure 4b,c: GWAS feature ranks | [figs4.ipynb](notebooks/supplementary_figures/figs4.ipynb) C2 | `highrisk_raw_absIG_tables_CHD/SNP{,_GENE}_raw_absIG_high_top_85.csv`; SNP/gene reverse ranks | `fig2/CHD_GWAS_genes_SNP_reverse_rank_barplot.pdf`, `figs4_final/CHD_GWAS_genes_GENE_reverse_rank_barplot.pdf` |
| Extended Data Figure 4b,c: GWAS attribution contrasts | [figs4.ipynb](notebooks/supplementary_figures/figs4.ipynb) C3, C4 | SNP IG, predictions and SNP-to-gene mapping | Locus- and gene-level high-risk attribution plots |
| Extended Data Figure 4a: gene-level SNP attribution | [figs4.ipynb](notebooks/supplementary_figures/figs4.ipynb) C5 | SNP IG/predictions and SNP-to-gene mapping | `figs4_final/{endpoint}_top_gene_origin_boxplot_with_sig.pdf` |
| Extended Data Figure 4d: PCSK9, TSPAN9, APOC3 and GRID2 neighborhoods | [figs4.ipynb](notebooks/supplementary_figures/figs4.ipynb) C6, C7, C8, C9 | SNP IG/predictions and variant annotation | `figs4_final/{PCSK9_rs662145,TSPAN9_rs3782805,APOC3_rs1327115596,GRID2_rs2089990}_locus_IG.pdf` |
| Extended Data Figure 4e: protein overlap and GO | [figs4.ipynb](notebooks/supplementary_figures/figs4.ipynb) C11 | Soma feature annotations and GO enrichment tables | `figs5_final/FINAL_SOMA_GO_VENN_BAR_BUBBLE_CHD_istroke.pdf`; merged GO table |
| Supporting SNP gene-set GO analysis | [figs4.ipynb](notebooks/supplementary_figures/figs4.ipynb) C10 | CHD/stroke SNP gene sets and GO enrichment tables | `figs5_final/FINAL_VENN_BAR_BUBBLE_CHD_istroke_v1.pdf` |
| Extended Data Figure 5c: representative ECG lookup | [figs5.ipynb](notebooks/supplementary_figures/figs5.ipynb) C2 | ECG predictions; locally selected example | Displayed prediction/risk row |
| Supporting fundus/protein heatmaps and source tables | [figs5.ipynb](notebooks/supplementary_figures/figs5.ipynb) C3, C4, C5 | Fundus/proteomic predictions, protein inputs and correlation tables | `fig3_final/protein_heatmap_Fundus_vs_Proteomics_same_direction_top_bottom/`: heatmaps, computed correlations, GO tables and plots; displayed input summaries |
| Extended Data Figure 6b: SNP/ECG/protein attribution agreement | [figs6.ipynb](notebooks/supplementary_figures/figs6.ipynb) C3 | Paired single/multimodal attribution vectors and participant risk | `figs6_final/fig_single_vs_multi_pearson_pid/{snp,ecg,soma}_PID_level_pearson.pdf` |
| Extended Data Figure 6b: fundus attribution agreement | [figs6.ipynb](notebooks/supplementary_figures/figs6.ipynb) C4 | Paired Grad-CAM text arrays and multimodal risk | `figs6_final/fig_single_vs_multi_pearson_pid/ICVD_fundus_per_sample_single_vs_multimodal_gradcam_pearson.csv`; agreement PDF |
| Supporting single/multimodal attribution magnitudes | [figs6.ipynb](notebooks/supplementary_figures/figs6.ipynb) C2 | Single/multimodal IG summaries, ECG attribution text arrays and fundus Grad-CAM arrays | `figs6_final/IG_compare_{endpoint}.pdf` |

#### Clinical external evaluation

The notebook map above covers clinical strategy boxplots. Frozen-pair and six-strategy cohort evaluation use these pipeline entries:

| Analysis | Code / command | Inputs | Outputs |
| --- | --- | --- | --- |
| UKB/MUCH frozen clinical pair and comparator evaluation | `python pipelines/run.py infer` → [`inference/final_pair.py`](pipelines/3_CIPHER_clinical_applications/inference/final_pair.py) | Configured cohort features, follow-up/event table and frozen clinical models | `aggregate_metrics.csv`, `aggregate_summary.csv`, comparison plots and run manifest under the configured private output directory |
| Six-strategy external evaluation | [`inference/six_strategies.py`](pipelines/3_CIPHER_clinical_applications/inference/six_strategies.py); commands in stage 3 | Frozen strategy artifacts; cohort clinical features, event and time columns | Cohort-level aggregate metrics and summary tables; Figure 5 C2/C3 reads the exported cohort aggregates |
| Event-status metrics from existing predictions | [`inference/evaluate_predictions.py`](pipelines/3_CIPHER_clinical_applications/inference/evaluate_predictions.py); command in stage 3 | Prediction manifest and aligned event/time/risk columns | Aggregate point estimates, bootstrap results and run manifest |

The preprocessing and training stages produce the feature tables, predictions, embeddings and per-sample interpretations used by the analysis cells. The mapped CSV/LaTeX exports identify the numerical tables associated with each plotting route; published table numbering follows the manuscript legends.

## Data access

Participant inputs, outcomes, predictions, embeddings, individual attributions, local configurations, and run outputs are excluded from Git. The distributed research assets are code, feature-only panel lists, aggregate figure source tables, notebooks with cleared outputs, and the two clinical model files.
