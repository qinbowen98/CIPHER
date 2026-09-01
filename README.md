# CIPHER figure and analysis code

This repository contains the notebooks used to calculate results and generate the main and supplementary figures for CIPHER.

## Repository layout

```text
notebooks/
  main_figures/            Figure 1-5 notebooks
  supplementary_figures/  Supplementary Figure 0-6 notebooks
config/
  paths.example.yaml       Example local data-path configuration
docs/
  ARBITRARY_CHOICES_AUDIT.md
  PUBLIC_DATA_BOUNDARY.md
scripts/
  audit_notebooks.py       Static reproducibility checks
environment.yml
```

Notebook outputs have been cleared for a smaller, reviewable repository. The source notebooks in `Downloads` were not modified.

## Reproducibility status

The notebooks are a public, derived-data-only release. Code that reads or processes source clinical data, protein expression/input values, raw genotypes, original fundus images, raw ECG waveforms, or participant-level cohort tables has been replaced by disclosure notes. See `docs/PUBLIC_DATA_BOUNDARY.md` for the exact boundary.

The retained workflow is **not portable end to end** until its permitted intermediate artifacts are supplied. Before claiming reproducibility:

1. Place de-identified input data/model artifacts in a stable location.
2. Replace remaining internal paths to permitted intermediate artifacts with values loaded from `config/paths.yaml` (start from `config/paths.example.yaml`).
3. Record the provenance and checksum of every external input.
4. Resolve the high-priority items in `docs/ARBITRARY_CHOICES_AUDIT.md` and report sensitivity analyses.
5. Execute notebooks from a clean environment and save an execution log.

## Environment

```bash
conda env create -f environment.yml
conda activate cipher-figures
jupyter lab
```

Some analyses may require additional system libraries and protected data access. Package versions should be frozen after the notebooks have been rerun successfully in the target environment.

## Notebook map

| Manuscript item | Notebook |
|---|---|
| Main Figure 1 | `notebooks/main_figures/AAA_final_fig1.ipynb` |
| Main Figure 2 | `notebooks/main_figures/AAA_final_fig2.ipynb` |
| Main Figure 3 | `notebooks/main_figures/AAA_final_fig3.ipynb` |
| Main Figure 4 | `notebooks/main_figures/AAA_final_fig4.ipynb` |
| Main Figure 5 | `notebooks/main_figures/AAA_final_fig5.ipynb` |
| Supplementary Figures 0-6 | matching notebook under `notebooks/supplementary_figures/` |
