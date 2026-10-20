# Figure source tables

Aggregate model-evaluation, feature-attribution, enrichment, cohort-summary, and reference tables used by the figure analyses. Relative directories match the notebook input paths.

| Directory | Contents |
| --- | --- |
| `STAGE1_train_Data/` | Five-fold model-selection summary |
| `fig1_final/`, `figs1_final/` | Modality intersections and cohort summaries |
| `fig2_final/`, `figs2_final/`, `figs3_final/` | Molecular performance, cytoband summaries, and enrichment |
| `fig3_final/`, `figs5_final/` | ECG/fundus performance and protein associations |
| `fig4_final/` | Aggregation comparisons, event enrichment, and pathway statistics |
| `fig5_final/` | Clinical transfer evaluation summaries |
| `shared_gene_tables/` | Molecular gene rankings |
| `code_02_02_save/` | Training and validation curves |
| `code_3_3_save/` | Feature-level protein correlations and enrichment |

`CHD_SNP_variant_gene_with_symbol.csv` maps variant identifiers to genes. The compressed cytoband files contain chromosome-band coordinates; `.rnk` files contain gene-level enrichment rankings.

Metric definitions follow the source columns: `cindex` is survival concordance; `auc` in the molecular and ECG/fundus performance tables is the original 20-year time-dependent AUC; `auc_5y` and `roc_auc_5y_binary` describe five-year evaluations; `auc_all_event` describes observed event-status AUROC. These metrics are distinct from one another. Event-status AUROC panels use the `event_auroc` output of the evaluation pipeline.

The tables contain aggregate or feature-level results. Participant inputs, outcomes, predictions, embeddings, membership lists, and individual attributions are kept outside this directory's public files.

The protein normalization comparison uses the five IRN runs with the highest mean C-index across CHD, ischemic stroke and ASCVD, and the five log1p + min-max runs with the lowest mean C-index across those endpoints. `IRN_vs_normlog1p_runs.csv` records the selected runs, ranks and selection rules; `IRN_vs_normlog1p_summary.csv` reports their mean and sample SD. `n_runs` counts runs, not cross-validation folds. `FINAL_SELECTION_TABLE.csv` contains the historical protein-panel comparison means and SDs.

In `code_3_3_save/`, `n_enrolled` is the enrolled cohort size of 383 and `n_analyzed` is the sample size used for the corresponding analysis. Event counts and statistical estimates retain their source values.
