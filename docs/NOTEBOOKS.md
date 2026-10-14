# Figure notebooks

`main_figures/` contains Figures 1-5. `supplementary_figures/` contains Figures S0-S6.

Restricted-data placeholders, empty cells and the hand-entered GO barplot have been removed.
The internal endpoint name is `ICVD`; displayed ASCVD labels are retained where used in the manuscript.
Outputs and execution counts are cleared. Paths are project-relative.

Run notebooks in the authorized analysis project, with the input tables listed in
[FIGURE_INPUTS.md](FIGURE_INPUTS.md). The cleaned notebooks have not been run against real cohort data.
Some retained analyses need private participant-level intermediate files and cannot be fully reproduced from public summaries.

```bash
python scripts/clean_notebooks.py
```

The cleaner is idempotent and does not execute any notebook or read data tables.
