# Data boundary

The repository distributes code and two inspected frozen model weights. It does not distribute participant data.

Keep these files private:

- Clinical inputs, outcomes and follow-up, including all UKB/MUCH participant records.
- Predictions, IDs, cohort/split membership and aligned risk tables.
- Embeddings, attention, per-participant IG, Grad-CAM and ECG/image arrays.
- Training matrices, sorted normalization references, logs and notebook outputs.

Removing an ID column does not make a participant-level table shareable.

Possible figure inputs are model/fold/epoch metrics, overlap counts, feature/region-level statistics,
correlation coefficients, gene ranks and pathway enrichment results. Each table needs a schema and provenance check.
UKB/MUCH figure summaries additionally remain subject to their data-use terms.

[FIGURE_INPUTS.md](FIGURE_INPUTS.md) lists candidate aggregate files and remaining private dependencies.
It lists filenames, not approved datasets. No table values were copied in the notebook cleanup.

Original KNN/SVR objects and unreviewed model implementations are excluded from model export.
Only the two vetted inference artifacts in the release manifest are included.
