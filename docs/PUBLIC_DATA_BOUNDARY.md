# Public data boundary

The public notebooks intentionally begin with de-identified derived artifacts. They do not include code that reads or processes source cohort data.

## Included intermediate inputs

- model prediction/risk tables;
- integrated gradients (IG), feature-importance and rank tables;
- Grad-CAM arrays without source images;
- learned embeddings and attention weights;
- model-performance and bootstrap-result tables;
- correlation, differential-analysis, GWAS annotation, GSEA and GO result tables;
- aggregate cohort summaries that do not contain participant-level rows.

These files are not automatically safe to publish. Participant identifiers and rare combinations must still be removed or replaced with release IDs before any intermediate file is committed.

## Excluded source inputs and code

- participant-level clinical and baseline tables;
- protein expression/abundance matrices and other model input values;
- raw genotype/dosage matrices;
- original fundus images and image preprocessing;
- raw ECG waveforms/XML and waveform preprocessing;
- participant-level cohort membership and raw event tables.

When an original cell supported either a derived attribution column or a raw expression/input column, the complete cell was excluded. This conservative rule prevents a public notebook from revealing the raw-data processing path.

## Provenance contract

Each retained intermediate dataset should have a separate schema/provenance record containing its generating internal workflow, software version, parameter set, creation date, row/column semantics, privacy review status and checksum. Internal raw-data workflows should be version-controlled in an access-restricted repository rather than in this public repository.
