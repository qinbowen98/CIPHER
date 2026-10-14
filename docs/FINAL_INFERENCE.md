# UKB and MUCH reproduction contract

## Inputs

The two CIPHER artifacts require this feature order:

```text
age_feat, sex, SBP, DBP, TC, TG, HDLC, LDLC, BMI,
height, weight, waist, smoking
```

Age is in years; blood pressure in mmHg; TC/TG/HDLC/LDLC in mmol/L; height/waist in cm; weight in kg.
Sex uses the training convention male=1/female=0. Smoking must match the saved model's source coding.
Do not silently switch from historical ever-smoking to a replacement smoking variable.
The scores use fitted preprocessing from each artifact, without external refitting, clipping or score rescaling.

### UKB

Supply three private tables, joined by unique `eid`:

1. Original clinical features for CIPHER. Use the historical original table, including its original laboratory and smoking variables.
2. Enriched clinical-score features. Required fields: `age_feat, sex, SBP, TC, HDLC, waist, smoking, diabetes, treated_hypertension, family_history, FBG, eGFR`.
3. Outcomes with `eid`, `event`, `time`; if `endpoint` is present, only `ASCVD` is retained.

The original model features and enriched score fields have separate namespaces. No enriched field overwrites a CIPHER input.
Only valid events (0/1), finite positive follow-up and complete required fields enter the five-model comparison.
The age restriction uses the original model-feature age. The locked counts are:

| Age band | Participants | ASCVD events |
|---|---:|---:|
| 40-74 | 212176 | 21847 |
| 60-74 | 89646 | 13750 |

The second band is a subset of the first common evaluation cohort. A count mismatch stops inference rather than publishing a different cohort.
Changing expected counts constitutes a different analysis and should be documented.

### MUCH

Supply the finalized completed clinical table, with `PATIENT_SN`, `ASCVD_EVENT`, `FOLLOWUP_YEARS`,
all 13 CIPHER features and the clinical-score features listed above.
The finalized preparation/defaults from the historical MUCH analysis must already be applied.
The pipeline does not reconstruct missing clinical-score variables, recode an unprepared RDS or impute external cohort statistics.
Missing CIPHER-only values can be handled by the saved imputer. Only rows with valid outcomes and all five finite scores are evaluated.
Clinical-score age is clipped to 30-74 as in the historical MUCH age-band workflow; CIPHER input age is not clipped.

## Clinical comparators

The formulas reproduce the historical analysis, rather than silently replacing them with newer scores.
China-PAR uses the historical northern/urban constants; Framingham uses its fixed published sex-specific centering;
CKM2S2-BAG uses the historical point-score implementation. These names do not imply external recalibration.
FBG uses mmol/L and eGFR uses mL/min/1.73m2. No additional clinical defaults are introduced by this inference code.

## Bootstrap and outputs

Point metrics are event AUROC and Harrell C-index. They are not interchangeable with a censoring-adjusted time-dependent AUC.
The default analysis uses 300 paired participant bootstrap draws and seed 20260923.
Each replicate uses `default_rng(seed + replicate_index)` and samples participants uniformly with replacement, exactly as the final age-band workflow.
A replicate with only one event class is omitted. All five models use the same draw.
UKB is ordered by normalized ID to match the historical pivoted predictions; the MUCH reconstructed-feature route retains its original row order.
Set `bootstrap_sort_ids` to match the historical input route if using a different MUCH prediction export.
Confidence intervals depend on cohort membership, row order, replicate count, seed and software versions.

Only aggregate files are written:

- `aggregate_metrics.csv`: cohort/age-band/model/replicate metrics and counts.
- `aggregate_summary.csv`: point estimates and percentile confidence intervals.
- `run_manifest.json`: model checksums and analysis settings, without input paths or identifiers.
- comparison PDFs generated from aggregate confidence intervals.

`reference_predictions` may point to a private historical prediction export (wide or label/score long format).
It verifies all participant IDs, outcomes and five scores in memory and fails on differences. It never copies the reference rows.
This is the strongest check of exact historical reproduction; matching counts alone does not prove identical membership.

No real cohort data are included, and the current release does not claim that real UKB/MUCH analyses have been executed locally.
Two CIPHER weights can reproduce the final comparison, but cannot reproduce an all-algorithm six-setting boxplot.
