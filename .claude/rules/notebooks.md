---
name: notebooks
description: Structure and conventions for the analysis notebooks.
paths:
  - "notebooks/**"
---

# Notebooks

## No analysis logic in notebooks

All analysis code lives in `src/`. A notebook imports and calls; it does not
define. If you need a new computation, add it to the appropriate module and
import it.

The same goes for plotting: figures come from `visualization.py`,
`qm_visualization.py` or `qm_transform_viz.py`.

## Structure

Every notebook opens with an Environment Setup cell that fails loudly —
`FileNotFoundError` — if a required cache is missing. Silent fallbacks hide a
stale or absent cache until the figures are already wrong.

Each figure cell is followed immediately by a markdown cell holding its
caption, in the project caption format.

Figures save to `plots/` under the naming already in use: `fig0N_*` for the
exploration notebook, `fig_m0N_*` for the models notebook, `fig_q0N_*` for
quantile mapping. Continue the existing numbering rather than restarting.

## The SKIP_HEAVY flag

`notebooks/01` defines `SKIP_HEAVY` in its setup cell. When `True` it skips the
two figures that run over 28 million rows. Keep it `True` while editing
structure; set it `False` for a full run.

## Reading data

Read the **aggregates**, not the large percentile tables. Reach for a
`pct_*.parquet` only when a single window slice is genuinely needed, and pass
`columns=` to `qm_metrics.load_pct_table` when you do.
