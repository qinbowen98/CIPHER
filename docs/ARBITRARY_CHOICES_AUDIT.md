# Audit of potentially arbitrary analytical choices

Scope: static review of the 12 supplied notebooks. A choice is listed because it could affect estimates, selected features, biological interpretation, or perceived visual strength. This is not an allegation that the choice is invalid; it identifies where justification, preregistration, or sensitivity analysis is needed.

## High priority: can change scientific conclusions

1. **Inconsistent risk-group definitions.** High-risk cutoffs include the 50th, 85th, and 95th percentiles; low/high fractions include 10%, 15%, 20%, 40%, 50%, and 89%. Examples occur in main Figures 2-4 and Supplementary Figures 4 and 6. Choose definitions based on a clinical or prespecified rationale and repeat key results over a common grid (for example 80/20, 85/15, 90/10, 95/5).
2. **Inference sometimes effectively disabled.** `P_CUTOFF = 1` and `FDR_CUTOFF = 1.0` occur in Figure 3/Supplementary Figure 5 workflows; one GO plot sets `alpha = 1`. These settings admit all tested features/terms. They must not be described as statistically significant. Use FDR-controlled results for confirmatory claims and label unrestricted results as ranked exploratory displays.
3. **Manual GWAS selection.** Names such as `selected_GWAS_tracks_manual_5tracks.csv`, `manual_selected_*GWAS*`, an `ALLOWED_JOURNALS` list, and `HIGHLIGHT_MODE = "any_hit"` indicate manual inclusion/highlighting. Publish the exact rule before looking at the final plot, include the full candidate table, and show an automated rule-based alternative.
4. **Small/inconsistent bootstrap budgets.** Performance intervals use 200 replicates in several notebooks, 1,000 in one Figure 5 analysis, and only 20 in another Figure 5 cell. Twenty is inadequate for stable percentile confidence intervals; 200 may also be noisy in tail estimates. Use at least 1,000-2,000 (preferably more when feasible), a fixed seed, and participant-level resampling that respects the sampling unit.
5. **Feature selection thresholds drive enrichment.** `TOP_N` values of 10, 15, 50, or 100; top fractions of 5%, 10%, 20%, and 50%; correlation cutoffs of 0, 0.05, and 0.10; and FDR thresholds of 0.05/0.20 feed gene lists and GO/GSEA. Report threshold-free/rank-based analyses where possible and a sensitivity matrix across plausible cutoffs.
6. **Asymmetric group split.** Figure 4 includes `HIGH_RISK_FRAC = 0.1` with `LOW_RISK_FRAC = 0.89`. Explain the missing 1%, confirm that groups cannot overlap, and justify the imbalance. Prefer explicit quantile boundaries with assertions on membership.
7. **Post hoc sample/image selection risk.** Grad-CAM panels select top-risk event cases (`TOP_N = 15` or top 50) and some cells refer to particular PIDs. Define the selection algorithm prospectively, publish the eligible denominator and manifest, and include random or representative panels to avoid cherry-picking.

## Medium priority: affects robustness or interpretation

1. **Image-processing constants.** Retinal vessel extraction uses many fixed values (black-corner threshold 18, percentile 86, scale 0.85, morphology sizes, eccentricity/aspect rules). Validate against annotated masks or report robustness over plausible ranges.
2. **Display clipping/truncation.** Importance colors use 1st/99th percentiles; other cells use 99.5% clipping, `Z_CLIP = 2`, fixed correlation limits, or hand-set axis maxima. These are reasonable for display only if raw statistics are unchanged and clipped observations are disclosed.
3. **Fallback significance thresholds.** Figure 4 uses `FDR_GSEA_FALLBACK = 0.25` and relaxed FDR 0.20. This matches some exploratory GSEA conventions but must be separated from confirmatory FDR 0.05 claims.
4. **Minimum sample sizes are very small.** Several correlation/group comparisons allow `MIN_N` of 5, 8, 10, or 20. Estimates at the lower values will be unstable. Report sample sizes, uncertainty, and sensitivity to stricter minimums.
5. **Model combination described as “toy”.** Figure 3 writes `toy_combined` predictions/metrics. Do not use these as headline multimodal model evidence unless the combination rule is formally specified, trained without test leakage, and evaluated on held-out data.
6. **Potential endpoint naming mismatch.** Both `stroke` and `istroke` appear in filename patterns. Define a canonical endpoint dictionary and assert that files/labels refer to the intended outcome.
7. **Randomness is only partially controlled.** Seeds 42 and 1234 appear, but not every stochastic library/workflow is visibly seeded. Centralize seed control for Python, NumPy, scikit-learn, XGBoost and enrichment routines, and record library versions.

## Reproducibility/engineering issues

1. Absolute paths (`/home/qinbw/CIPHER`, `/data200T/...`) and timestamped run directories prevent execution elsewhere.
2. Notebooks include shell commands (`cat`, `ls`, `head`, `tar`) and network download of UCSC cytobands; these should become explicit setup scripts with checksums.
3. Repeated and revised code cells contain conflicting constants (for example cutoff 0.4 followed by 0.5). A clean run may not reproduce an interactive run unless execution order is enforced.
4. Input tables, trained models, manifests, and a data dictionary are absent. The repository currently documents analysis code, not a complete reproducible research compendium.
5. Dependencies are broad and unpinned. Freeze exact versions after a successful clean run.
6. Some plotting code uses hard-coded labels, genes, PIDs, journals, and axes. Move these to a documented configuration and distinguish analysis parameters from aesthetics.

## Choices that are mainly aesthetic

Figure size, DPI, colors, point alpha, label padding, legend dimensions, and most axis spacing are not intrinsically problematic. They become problematic only when they hide observations, truncate uncertainty, or visually exaggerate effects. Keep them in a plotting-style configuration and do not let them alter statistical input.

## Minimum release checklist

- Freeze one canonical parameter set and explain its origin.
- Run cutoff/quantile/bootstrap sensitivity analyses for every headline result.
- Remove or fully disclose manual feature/study/sample selection.
- Separate exploratory ranking from FDR-controlled inference.
- Verify no train/validation/test leakage in feature attribution, model selection, or combination.
- Run all notebooks from a fresh environment in documented order.
- Publish de-identified aggregate inputs where permitted, plus checksums and schemas.
- Add assertions for endpoint names, unique participant IDs, non-overlapping groups, and expected sample counts.
