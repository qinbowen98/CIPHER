# Figure inputs

The lists below come from the retained notebook code. Paths are relative to the analysis project, not this release checkout.
Files have not been inspected on the server. A filename is a lookup hint, not a release approval. Confirm row semantics and column names before sharing.

Only model-, feature-, pathway- or region-level summaries are candidates. Patient inputs, outcomes, predictions, membership, embeddings, attention and per-patient attribution stay private. UKB/MUCH participant data stay private even after ID removal.

Outputs marked `generated` already contain the statistics used downstream. Prefer these aggregate outputs over rerunning private participant-level steps.
Manually adjusted Venn totals and synthetic membership files are not observed cohort source data. Use the original count summaries.

## Input families

| Figure | Aggregate tables to prepare | Private dependencies that must not be shared |
|---|---|---|
| 1 | Original modality totals and exact overlap counts | Prediction CSVs and membership/PID tables |
| 2 | Model/fold metrics, SNP/cytoband and gene ranks, GWAS annotations, GO/GSEA tables | Per-patient predictions and IG matrices |
| 3 | ECG/Fundus metrics, feature correlations, Venn gene counts and enrichment tables | Model scores and sample-specific attribution/Grad-CAM |
| 4 | External model comparison, capture/segment summaries, protein differential and GSEA tables | Aligned predictions, outcomes, age metadata and embeddings |
| 5 | Model-level strategy/bootstrap results | Source cohort clinical records and external cohort records |
| S0 | Validation metrics by model, epoch or fold | Individual predictions |
| S1 | Cohort-level baseline summaries | Baseline participant rows |
| S2 | Cytoband/GWAS coverage, selection metrics and enrichment | Participant IG matrices |
| S3 | Model metrics and cytoband summaries | Participant predictions and IG matrices |
| S4 | SNP/gene rank and enrichment tables | Participant IG matrices |
| S5 | Feature-correlation and GO tables | Participant risk/protein matrices |
| S6 | Aggregate single-versus-multimodal attribution comparisons need a separate export | PID-level correlations, predictions and attribution arrays |

## Candidate table paths

Patterns combine endpoint/modality variants and revised blocks. Not every pattern names a pre-existing file. `read` indicates a direct input; `generated` indicates a notebook output. Standalone relative names retain the original notebook working-directory assumption.

### AAA_final_fig1.ipynb

| Relative table or pattern | Role |
|---|---|
| `fig1_final/SingleModal_Run_printonly/four_modal_test_pid_venn_forced_totals_ordered/original_modality_totals.csv` | generated |
| `fig1_final/SingleModal_Run_printonly/four_modal_test_pid_venn_forced_totals_ordered/test_sample_4set_venn_exact_region_summary_original.csv` | generated |
| `fig1_final/{endpoint}_trainset_four_modal_sample_4set_venn_forced_totals_ordered/exact_region_tables/original/{endpoint}_train_sample_4set_venn_exact_region_summary_original.csv` | generated |
| `fig1_final/{endpoint}_trainset_four_modal_sample_4set_venn_forced_totals_ordered/{endpoint}_original_modality_totals.csv` | generated |
| `fig1_final/{endpoint}_trainset_four_modal_sample_4set_venn_forced_totals_ordered/{endpoint}_train_sample_4set_venn_counts_exact_original.csv` | generated |
| `figs1_final/{endpoint}_trainset_four_modal_sample_upset/{endpoint}_train_modality_total_counts.csv` | generated |
| `figs1_final/{endpoint}_trainset_four_modal_sample_upset/{endpoint}_train_sample_upset_counts_exact.csv` | generated |

### AAA_final_fig2.ipynb

| Relative table or pattern | Role |
|---|---|
| `CHD_SNP_variant_gene_with_symbol.csv` | read |
| `STAGE1_train_Data/analysis_summary_1201/cv_best_fivefold_per_model_endpoint.csv` | read |
| `fig2/SNP/{endpoint}_IG_final.csv` | read |
| `fig2_final/GO_Overlap.csv` | generated |
| `fig2_final/GO_{endpoint}_only.csv` | generated |
| `fig2_final/GO_{set}.csv` | generated |
| `fig2_final/GSEA_unbiased/GSEA_{endpoint}_raw.csv` | read |
| `fig2_final/GSEA_unbiased/barstyle_plots_from_existing_results/{endpoint} {direction}_GO_BP_significant_q{q_cutoff}.csv` | generated |
| `fig2_final/GSEA_unbiased/barstyle_plots_from_existing_results/{endpoint}_GO_BP_significant_q0.05.csv` | generated |
| `fig2_final/GSEA_unbiased/barstyle_plots_from_existing_results/{endpoint}_GO_BP_significant_q{q_cutoff}.csv` | generated |
| `fig2_final/GSEA_unbiased/barstyle_plots_from_existing_results/{endpoint}_GO_BP_{direction}_q{q_cutoff}_tmp.csv` | generated, read |
| `fig2_final/GSEA_unbiased/{endpoint}_GSEA_ranked_list.rnk` | read |
| `fig2_final/SNP_Proteomics_model_performance_summary.csv` | generated, read |
| `fig2_final/chromosome_IG_combined_manual_studies_all_hits_final/hg38_cytoBand.txt.gz` | read |
| `fig2_final/chromosome_IG_combined_manual_studies_all_hits_final/manual_selected_{endpoint}_{endpoint}_GWAS_hits_all_hits_final.csv` | generated |
| `fig2_final/chromosome_IG_combined_manual_studies_all_hits_final/manual_selected_{endpoint}_{endpoint}_GWAS_study_summary_all_hits_final.csv` | generated |
| `fig2_final/chromosome_IG_combined_manual_studies_all_hits_final/{endpoint}_{endpoint}_cytoband_summary_with_OR_manual_studies_all_hits_final.csv` | generated |
| `fig2_final/chromosome_IG_combined_manual_studies_all_hits_final/{endpoint}_{endpoint}_q75_selected_GWAS_studies.csv` | generated |
| `fig2_final/chromosome_IG_combined_manual_studies_all_hits_final/{endpoint}_{endpoint}_q75_selected_GWAS_study_summary.csv` | generated |
| `fig2_final/chromosome_IG_combined_manual_studies_all_hits_final/{endpoint}_{endpoint}_snp_band_ig_summary_manual_studies_all_hits_final.csv` | generated |
| `fig2_final/locus_plots/rs1327115596_{endpoint}_selected_GWAS_tracks_manual_5tracks.csv` | generated |
| `fig2_final/locus_plots/rs3782805_{endpoint}_selected_GWAS_tracks_manual_5tracks.csv` | generated |
| `gwas-catalog-download-associations-v1.0-full.tsv` | read |
| `shared_gene_tables/{endpoint}_shared_SNP_SOMA_gene_rank.csv` | read |

### AAA_final_fig3.ipynb

| Relative table or pattern | Role |
|---|---|
| `STAGE1_train_Data/analysis_summary_1201/cv_best_fivefold_per_model_endpoint.csv` | read |
| `code_3_3_save/high_vs_low_quadrant_scatter_GO_final/GO/{endpoint}/protein/Fundus_high__ECG_high/{endpoint}_protein_Fundus_high__ECG_high_GO.csv` | read |
| `code_3_3_save/high_vs_low_quadrant_scatter_GO_final/GO/{endpoint}/protein/Fundus_high__ECG_low/{endpoint}_protein_Fundus_high__ECG_low_GO.csv` | read |
| `code_3_3_save/high_vs_low_quadrant_scatter_GO_final/GO/{endpoint}/protein/Fundus_low__ECG_high/{endpoint}_protein_Fundus_low__ECG_high_GO.csv` | read |
| `code_3_3_save/high_vs_low_quadrant_scatter_GO_final/GO/{endpoint}/protein/Fundus_low__ECG_low/{endpoint}_protein_Fundus_low__ECG_low_GO.csv` | read |
| `code_3_3_save/high_vs_low_quadrant_scatter_GO_final/GO/{endpoint}/protein/{quadrant}/{endpoint}_protein_{quadrant}_GO.csv` | read |
| `code_3_3_save/protein_risk_correlation_gsea_directional/{endpoint}_{modality}_negative_protein_corr.csv` | generated |
| `code_3_3_save/protein_risk_correlation_gsea_directional/{endpoint}_{modality}_positive_protein_corr.csv` | generated |
| `code_3_3_save/protein_risk_correlation_gsea_final/cached_gsea/{endpoint}_{modality}_Hallmark_gsea.csv` | generated, read |
| `code_3_3_save/protein_risk_correlation_gsea_final/cached_gsea/{endpoint}_{modality}_corr.csv` | generated |
| `code_3_3_save/risk_feature_correlation/{endpoint}_{modality}_vs_SomaProtein_pearson.csv` | read |
| `code_3_3_save/risk_feature_correlation/{endpoint}_{risk_model}_vs_SomaProtein_pearson.csv` | read |
| `fig3_final/ECG_Fundus_model_performance_summary.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/GO_enrichment_intersections/ALL_4set_venn_large_regions_GO_GO_Biological_Process_2023_FDR_0.05.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/GO_enrichment_intersections/ALL_4set_venn_large_regions_GO_GO_Biological_Process_2023_raw.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/GO_enrichment_intersections/{endpoint}/{direction}/{safe_name}/{endpoint}_{direction}_{safe_name}_n{len}_GO_FDR_0.05.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/GO_enrichment_intersections/{endpoint}/{direction}/{safe_name}/{endpoint}_{direction}_{safe_name}_n{len}_GO_raw.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/GO_enrichment_intersections/{endpoint}/{direction}/{safe_name}/{endpoint}_{direction}_{safe_name}_n{len}_input_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/venn_region_genes/ALL_endpoints_4set_venn_exact_region_summary_rho_{abs}_FDR_1.0.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/venn_region_genes/{endpoint}/negative/{endpoint}_negative_4set_venn_exact_region_summary.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/venn_region_genes/{endpoint}/negative/{endpoint}_negative_{safe_name}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/venn_region_genes/{endpoint}/positive/{endpoint}_positive_4set_venn_exact_region_summary.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/venn_region_genes/{endpoint}/positive/{endpoint}_positive_{safe_name}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/venn_region_genes/{endpoint}/{mode}/{endpoint}_{mode}_4set_venn_exact_region_summary.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/venn_region_genes/{endpoint}/{mode}/{endpoint}_{mode}_{safe_name}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_4set_venn/{endpoint}_protein_four_modal_risk_spearman_corr.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/ALL_endpoints_positive_negative_rho_{abs}_FDR_0.2_upset_counts.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/GO_enrichment_intersections/ALL_large_intersections_GO_GO_Biological_Process_2023_FDR_0.05.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/GO_enrichment_intersections/ALL_large_intersections_GO_GO_Biological_Process_2023_raw.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/GO_enrichment_intersections/{endpoint}/{direction}/{safe_name}/{endpoint}_{direction}_{safe_name}_n{len}_GO_FDR_0.05.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/GO_enrichment_intersections/{endpoint}/{direction}/{safe_name}/{endpoint}_{direction}_{safe_name}_n{len}_GO_raw.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/GO_enrichment_intersections/{endpoint}/{direction}/{safe_name}/{endpoint}_{direction}_{safe_name}_n{len}_input_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/{endpoint}_negative_rho_{abs}_FDR_0.2_upset_counts.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/{endpoint}_positive_rho_{abs}_FDR_0.2_upset_counts.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/{endpoint}_protein_four_modal_risk_spearman_corr.csv` | generated |
| `fig3_final/protein_four_modal_risk_upset/{endpoint}_{mode}_rho_{abs}_FDR_0.2_upset_counts.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/ALL_endpoints_{.join}_venn_region_gene_summary.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_negative_venn_region_gene_summary.csv` | generated, read |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_negative_{a}_only_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_negative_{a}_{b}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_negative_{a}_{b}_{c}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_negative_{a}_{c}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_negative_{b}_only_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_negative_{b}_{c}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_negative_{c}_only_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_positive_venn_region_gene_summary.csv` | generated, read |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_positive_{a}_only_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_positive_{a}_{b}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_positive_{a}_{b}_{c}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_positive_{a}_{c}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_positive_{b}_only_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_positive_{b}_{c}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_positive_{c}_only_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_protein_four_modal_risk_spearman_corr.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_{mode}_venn_region_gene_summary.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_{mode}_{a}_only_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_{mode}_{a}_{b}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_{mode}_{a}_{b}_{c}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_{mode}_{a}_{c}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_{mode}_{b}_only_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_{mode}_{b}_{c}_genes.csv` | generated |
| `fig3_final/protein_four_modal_risk_venn/{endpoint}_{mode}_{c}_only_genes.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_ECG_same_direction_top_bottom/GO_enrichment/{endpoint}/both_negative/GO_Biological_Process_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_ECG_same_direction_top_bottom/GO_enrichment/{endpoint}/both_negative/GO_Cellular_Component_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_ECG_same_direction_top_bottom/GO_enrichment/{endpoint}/both_negative/GO_Molecular_Function_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_ECG_same_direction_top_bottom/GO_enrichment/{endpoint}/both_negative_genes.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_ECG_same_direction_top_bottom/GO_enrichment/{endpoint}/both_positive/GO_Biological_Process_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_ECG_same_direction_top_bottom/GO_enrichment/{endpoint}/both_positive/GO_Cellular_Component_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_ECG_same_direction_top_bottom/GO_enrichment/{endpoint}/both_positive/GO_Molecular_Function_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_ECG_same_direction_top_bottom/GO_enrichment/{endpoint}/both_positive_genes.csv` | generated |

### AAA_final_fig4.ipynb

| Relative table or pattern | Role |
|---|---|
| `code_3_3_save/compare_external_mean_fivefold_fullcox_visit02/external_comparison_cindex_auc_summary.csv` | read |
| `fig4_final/embeddingv4/{endpoint}_visit12_PCA_PC1_PC10_risk_corr_pearson_corr_v02.csv` | generated |
| `fig4_final/{endpoint}_GO_BP_FDR005_and_selected_GSEA_NES_subplots/{endpoint}_GO_BP_enrichment_all_directions_fdr_bh_lt_0.05_GOFDR005.csv` | generated |
| `fig4_final/{endpoint}_GO_BP_FDR005_and_selected_GSEA_NES_subplots/{endpoint}_GO_BP_enrichment_higher_in_multimodal_top5_fdr_bh_lt_0.05.csv` | generated |
| `fig4_final/{endpoint}_GO_BP_FDR005_and_selected_GSEA_NES_subplots/{endpoint}_GO_BP_enrichment_lower_in_multimodal_top5_fdr_bh_lt_0.05.csv` | generated |
| `fig4_final/{endpoint}_GO_BP_FDR005_and_selected_GSEA_NES_subplots/{endpoint}_GO_BP_enrichment_{direction}_fdr_bh_lt_0.05.csv` | generated |
| `fig4_final/{endpoint}_GO_BP_FDR005_and_selected_GSEA_NES_subplots/{endpoint}_GO_BP_input_genes_higher_in_multimodal_top5_fdr_bh_lt_0.05.csv` | generated |
| `fig4_final/{endpoint}_GO_BP_FDR005_and_selected_GSEA_NES_subplots/{endpoint}_GO_BP_input_genes_lower_in_multimodal_top5_fdr_bh_lt_0.05.csv` | generated |
| `fig4_final/{endpoint}_GO_BP_FDR005_and_selected_GSEA_NES_subplots/{endpoint}_selected_GSEA_NES_values_one_row.csv` | generated |
| `fig4_final/{endpoint}_dual_metric_segment_tests/{endpoint}_dual_metric_segment_results.csv` | generated |
| `fig4_final/{endpoint}_protein_multimodal_top5_vs_direct_mean_top5_FDR_BH_and_Storey/{endpoint}_protein_multimodal_top5_vs_direct_mean_top5_all_diff.csv` | read |
| `fig4_final/{endpoint}_protein_top5_boxplot_and_groupwise_GSEA/{endpoint}_GSEA_groupwise_NES_change_multimodal_minus_direct.csv` | read |
| `fig4_final/{endpoint}_selected_GSEA_NES_barplots/{endpoint}_selected_GSEA_NES_direct_mean_vs_multimodal_long_values.csv` | generated |
| `fig4_final/{endpoint}_selected_GSEA_NES_barplots/{endpoint}_selected_GSEA_NES_direct_mean_vs_multimodal_values.csv` | generated |
| `fig4_final/{endpoint}_top_risk_capture_rate_combined_update/{endpoint}_observed_event_rate_and_capture_curves_top40_summary.csv` | generated |
| `fig4_final/{endpoint}_top_risk_capture_rate_combined_update/{endpoint}_top_risk_event_capture_rate_summary.csv` | generated |
| `fig4_final/{endpoint}_top_risk_segment_fast_significance/{endpoint}_fast_segment_event_rate_and_capture_significance.csv` | generated |
| `fig4_final/{endpoint}_top_risk_segment_significance_2sided/{endpoint}_segment_significance_two_sided.csv` | generated |

### AAA_final_fig5.ipynb

| Relative table or pattern | Role |
|---|---|
| `fig5_final/outputs_final_valid_select_more_models_{endpoint}_toprisk/ALL_model_results_long.csv` | read |
| `fig5_final/outputs_final_valid_select_more_models_{endpoint}_toprisk/ALL_model_results_long_merged_{endpoint}_{endpoint}_{endpoint}.csv` | generated, read |
| `fig5_final/outputs_final_valid_select_more_models_{endpoint}_toprisk_{endpoint}/ALL_model_results_long.csv` | read |

### AAA_final_figs0.ipynb

| Relative table or pattern | Role |
|---|---|
| `code_02_02_save/SingleModal_Run_update_v5/ecg_20251123_1316/curves/UA_ecg_bilstm_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/ecg_20251123_1316/curves/UA_ecg_cnn1d_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/ecg_20251123_1316/curves/UA_ecg_transformer_h256_l3_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/ecg_20251123_1316/curves/stroke_ecg_bilstm_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/ecg_20251123_1316/curves/stroke_ecg_cnn1d_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/ecg_20251123_1316/curves/stroke_ecg_transformer_h256_l3_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/ecg_20251123_1316/curves/{endpoint}_ecg_bilstm_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/ecg_20251123_1316/curves/{endpoint}_ecg_cnn1d_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/ecg_20251123_1316/curves/{endpoint}_ecg_transformer_h256_l3_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/UA_fundus_resnet_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/UA_fundus_retfound_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/UA_fundus_vitsmallFREEZE_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/UA_fundus_vitsmallFT_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/stroke_fundus_resnet_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/stroke_fundus_retfound_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/stroke_fundus_vitsmallFREEZE_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/stroke_fundus_vitsmallFT_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/{endpoint}_fundus_resnet_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/{endpoint}_fundus_retfound_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/{endpoint}_fundus_vitsmallFREEZE_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/fundus_20251123_1314/curves/{endpoint}_fundus_vitsmallFT_fold*_lr*_bs*_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/snp_20251125_2051/curves/stroke_snp_MLP_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/snp_20251125_2051/curves/stroke_snp_PCA_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/snp_20251125_2051/curves/stroke_snp_TR_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/snp_20251125_2051/curves/{endpoint}_snp_MLP_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/snp_20251125_2051/curves/{endpoint}_snp_PCA_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/snp_20251125_2051/curves/{endpoint}_snp_TR_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/soma_20251123_1305/curves/stroke_soma_PCA_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/soma_20251123_1305/curves/{endpoint}_soma_PCA_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/soma_20251123_1306/curves/stroke_soma_MLP_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/soma_20251123_1306/curves/stroke_soma_TR_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/soma_20251123_1306/curves/{endpoint}_soma_MLP_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `code_02_02_save/SingleModal_Run_update_v5/soma_20251123_1306/curves/{endpoint}_soma_TR_fold*_lr0.0001_bs32_train_curve.csv` | glob |
| `fig3_final_validation_all_lr_bs/ECG_Fundus_best_validation_valid_ci_all_lr_bs.csv` | generated |
| `figs2_final_validation/best_validation_valid_ci_lr0.0001_bs32.csv` | generated |

### AAA_final_figs1.ipynb

| Relative table or pattern | Role |
|---|---|
| `extended_table1_latex/Extended_Table1_matrix.csv` | generated |
| `extended_table1_latex_final/Extended_Table1.csv` | generated |
| `figs1_final/clinical_oncoponct_four_cohorts_baseline_q70_with_modality/baseline_clinical_oncoponct_group_summary.csv` | read |
| `figs1_final/clinical_oncoprint_four_cohorts_baseline_q70_with_modality/baseline_clinical_oncoprint_group_summary.csv` | read |

### AAA_final_figs2.ipynb

| Relative table or pattern | Role |
|---|---|
| `CHD_SNP_variant_gene_with_symbol.csv` | read |
| `FINAL_SELECTION_RAW.csv` | generated |
| `FINAL_SELECTION_STATS.csv` | generated |
| `GO_enrichment_final/NCR_select_GO_full.csv` | generated |
| `GO_enrichment_final/SVM_select_GO_full.csv` | generated |
| `GO_enrichment_final/cell_select_GO_full.csv` | generated |
| `GO_enrichment_final/{set}_GO_full.csv` | generated |
| `IRN_vs_normlog1p_performance/IRN_best5_normlog1p_worst5_{endpoint}_{endpoint}_{endpoint}_performance.csv` | generated |
| `IRN_vs_normlog1p_performance/selected_folders_rank_summary.csv` | generated |
| `STAGE1_train_Data/analysis_summary_1201/cv_best_fivefold_per_model_endpoint.csv` | read |
| `figs2_final/FINAL_SELECTION_TABLE.csv` | generated |
| `figs2_final/snp_cytoband_coverage_preprocessing_onerow/hg38_cytoBand.txt.gz` | read |
| `figs2_final/snp_cytoband_coverage_preprocessing_onerow/prediction_SNP_cytoband_counts.csv` | generated |
| `figs2_final/snp_cytoband_coverage_preprocessing_onerow/prediction_SNP_cytoband_coverage_known_GWAS_loci_summary.csv` | generated |
| `figs2_final/snp_cytoband_coverage_preprocessing_onerow/prediction_SNPs_with_cytoband.csv` | generated |
| `figs2_final/snp_cytoband_coverage_preprocessing_onerow/selected_known_GWAS_hits_with_prediction_SNP_coverage.csv` | generated |
| `gwas-catalog-download-associations-v1.0-full.tsv` | read |

### AAA_final_figs3.ipynb

| Relative table or pattern | Role |
|---|---|
| `CHD_SNP_variant_gene_with_symbol.csv` | read |
| `STAGE1_train_Data/analysis_summary_1201/cv_best_fivefold_per_model_endpoint.csv` | read |
| `fig2_final/chromosome_IG_combined_manual_studies_all_chromosomes_with_chrX_public/hg38_cytoBand.txt.gz` | read |
| `fig2_final/chromosome_IG_combined_manual_studies_all_chromosomes_with_chrX_public/manual_selected_{endpoint}_{endpoint}_GWAS_hits_all_hits_chrX_public.csv` | generated |
| `fig2_final/chromosome_IG_combined_manual_studies_all_chromosomes_with_chrX_public/manual_selected_{endpoint}_{endpoint}_GWAS_study_summary_all_hits_chrX_public.csv` | generated |
| `fig2_final/chromosome_IG_combined_manual_studies_all_chromosomes_with_chrX_public/{endpoint}_{endpoint}_cytoband_summary_with_OR_manual_studies_all_hits_chrX_public.csv` | generated |
| `fig2_final/chromosome_IG_combined_manual_studies_all_chromosomes_with_chrX_public/{endpoint}_{endpoint}_snp_band_ig_summary_manual_studies_all_hits_chrX_public.csv` | generated |
| `figs2_final/SNP_Proteomics_model_performance_summary.csv` | generated |
| `figs3_final/chromosome_IG_vertical_2rows_all_chromosomes/hg38_cytoBand.txt.gz` | read |
| `figs3_final/chromosome_IG_vertical_2rows_all_chromosomes/manual_selected_{endpoint}_{endpoint}_GWAS_hits_all_hits_chrX_public.csv` | generated |
| `figs3_final/chromosome_IG_vertical_2rows_all_chromosomes/manual_selected_{endpoint}_{endpoint}_GWAS_study_summary_all_hits_chrX_public.csv` | generated |
| `figs3_final/chromosome_IG_vertical_2rows_all_chromosomes/{endpoint}_{endpoint}_cytoband_summary_with_OR_manual_studies_all_hits_chrX_public.csv` | generated |
| `figs3_final/chromosome_IG_vertical_2rows_all_chromosomes/{endpoint}_{endpoint}_snp_band_ig_summary_manual_studies_all_hits_chrX_public.csv` | generated |
| `gwas-catalog-download-associations-v1.0-full.tsv` | read |

### AAA_final_figs4.ipynb

| Relative table or pattern | Role |
|---|---|
| `CHD_SNP_variant_gene_with_symbol.csv` | read |
| `fig2/SNP/{endpoint}_IG_final.csv` | read |
| `figs5_final/merged_GO_for_plot_{endpoint}_{endpoint}.csv` | generated |
| `figs5_final/top5%_GO_Biological_Process_2021_{endpoint}_SNP.csv` | read |
| `figs5_final/{endpoint}_GO_full.csv` | read |

### AAA_final_figs5.ipynb

| Relative table or pattern | Role |
|---|---|
| `fig3_final/protein_heatmap_Fundus_vs_Proteomics_same_direction_top_bottom/GO_enrichment/{endpoint}/both_negative/GO_Biological_Process_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_Proteomics_same_direction_top_bottom/GO_enrichment/{endpoint}/both_negative/GO_Cellular_Component_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_Proteomics_same_direction_top_bottom/GO_enrichment/{endpoint}/both_negative/GO_Molecular_Function_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_Proteomics_same_direction_top_bottom/GO_enrichment/{endpoint}/both_negative_genes.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_Proteomics_same_direction_top_bottom/GO_enrichment/{endpoint}/both_positive/GO_Biological_Process_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_Proteomics_same_direction_top_bottom/GO_enrichment/{endpoint}/both_positive/GO_Cellular_Component_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_Proteomics_same_direction_top_bottom/GO_enrichment/{endpoint}/both_positive/GO_Molecular_Function_2023_enrichr_results.csv` | generated |
| `fig3_final/protein_heatmap_Fundus_vs_Proteomics_same_direction_top_bottom/GO_enrichment/{endpoint}/both_positive_genes.csv` | generated |
| `fig3_final/protein_heatmap_{modality}_vs_{modality}_same_direction_top_bottom/computed_proteomics_corr/{endpoint}_{modality}_vs_SomaProtein_pearson_computed.csv` | generated |

### AAA_final_figs6.ipynb

| Relative table or pattern | Role |
|---|---|
| No public-ready aggregate input identified in the retained code; export a group-level summary in the protected environment. | Private analysis |

## UKB/MUCH figures

External figures require aggregate model metrics, confidence intervals and KS/subgroup/protein statistics from the authorized analysis. Never provide cohort rows, completed features, event tables or participant predictions.
The inference pipeline produces aggregate metric and summary tables in its configured private output directory. Any export remains subject to the cohort data-use terms.

## Plots requiring a new aggregate export

Participant-level embedding scatterplots, risk-versus-outcome panels and individual Grad-CAM/ECG panels cannot be reproduced publicly from the current patient-level inputs. A protected analysis must first produce an approved group-level or display-only representation; deleting IDs is insufficient.
No data tables are added by this update.
