---
status: accepted
---

# The fitting sample follows the qmap package, per family

`normal`, `linear` and `poly` are fitted on every training day. `quant`,
`rquant` and `ssplin` are fitted on 99 quantile nodes. This matches the `qstep`
defaults of the R `qmap` package, which uses `NULL` for the first group and
`0.01` for exactly the second.

Refitting the node-based methods on raw days changes MAE by at most 0.003 °C
and makes the spline about five times slower — 13 minutes to 100 minutes for a
full run. The split is therefore both faithful to the reference implementation
and cheap.

`select_hyper` routes on the same `fit_on_raw` flag, so hyper-parameter
selection uses the same resolution as the outer fit.
