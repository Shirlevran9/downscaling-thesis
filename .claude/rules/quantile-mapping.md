---
name: quantile-mapping
description: Conventions and traps for the quantile-mapping layer — season schemes, bias sign, node counts, monotonicity.
paths:
  - "src/qm_*.py"
  - "src/quantile_windows.py"
  - "src/predictors.py"
  - "scripts/run_qm*.py"
  - "scripts/run_quantile_mapping.py"
  - "scripts/sanity_qm_transforms.py"
---

# Quantile mapping — conventions and traps

Every item here was learned by getting it wrong first. See `docs/adr/` for the
decisions behind them.

## Two season conventions exist

`make_windows("quarter")` is **JFM/AMJ/JAS/OND**. `make_windows("season")` is
**DJF/MAM/JJA/SON**, grouped by season-year with December rolled into the
following winter. `data_io.seasonal_split` and `predictors._SEASON_OF_MONTH`
are two further DJF copies.

New code uses only the `"season"` scheme. `SCHEMES` excludes it deliberately so
the older pipeline is untouched; `WINDOW_SCHEMES` includes it.

`make_windows("season")` can return `-1` for days in an incomplete season. No
other scheme does. Handle it.

## Sign convention

`bias = x − y`, predictor minus target. Positive means too warm. This is the
default across the project.

`visualization.compute_regression_metrics` returns the **opposite** sign
(observed minus predicted). Both are live and both are correct for their own
callers. Do not unify them — check which one you are calling.

`qm_metrics.combination_metrics` takes `(y, x)` positionally. **Always call it
with keyword arguments.** Swapping them flips `bias` silently while leaving
`mae`, `rmse` and `r2` unchanged, so no test would catch it.

## Order of operations

The correction is applied to **daily values, and percentiles are taken
afterwards** — never the reverse.

Mapping a percentile directly equals mapping the days only when quantiles are
plain order statistics. Under the interpolating estimator this project uses the
two differ by up to 0.39 °C for QUANT.

## Monotonicity is correctness, not polish

It is what makes percentile-level evaluation well defined.
`predict_percentiles` enforces it; `predict` deliberately does not, because it
is used for daily values where ordering is meaningless. `n_nonmono` records
where the repair fired — read it.

## QUANT node counts stay at 11 or above

A table of *m* nodes reaches only to probability `0.5/m`, so fewer than 11
cannot represent P5 and the boundary rule invents it.

The unanchored version clamped flat below the first node and carried a
**+1.8 °C bias at SON P5** — worse than no correction at all — while scoring a
*better* MAE, because clamping acts as accidental shrinkage toward climatology.
**Always check `bias` alongside `mae` when changing anything here.**

## There are two climatology floors

Compare MAE against `obs_mad` (mean absolute deviation) and RMSE against
`obs_sd`. For a normal sample the first is about 0.8 of the second, so scoring
MAE against the SD understates the error by a fifth and can make a method look
as if it beat an unreachable floor.

## The fitting sample differs by family

`QQTransform.fit_on_raw` is `True` for `normal`, `linear` and `poly`, which use
every training day. It is `False` for `quant`, `rquant` and `ssplin`, which use
99 nodes. This matches the `qmap` package's `qstep` defaults.

`select_hyper` must use the same resolution as the outer fit — it routes on
`fit_on_raw` too.

## Data facts that bite

Latitude runs **descending**, 38 → 24. Never assume ascending.

The cached bilinear field is **already in °C**. Do not call `to_celsius` on it.
`qm_inputs.load_bilinear_predictor` validates the grid and dates before use.

Caches live under `data/cache/qm/` — the transform outputs under
`data/cache/qm/transforms/`.

## After any change

Run `python3.10 scripts/sanity_qm_transforms.py`. There is no test framework in
this repo; that script is the check.
