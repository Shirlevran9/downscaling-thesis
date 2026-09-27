---
status: accepted
---

# The older diagnostic pipeline is kept alongside the fitted baseline

`scripts/run_quantile_mapping.py` and its 5 × 4 × 5 predictor grid were
superseded as the project's baseline by the fitted transfer functions, but not
deleted.

It remains the producer of the percentile tables, `static_pixels.parquet` — which
the fitted analysis reuses — and the Streamlit dashboard's inputs. The two
answer different questions: the old grid compares *predictors*, the new one
compares *transfer functions* on a single predictor.

Outputs are kept in separate directories so a `--force` rebuild of one can
never clobber the other.
