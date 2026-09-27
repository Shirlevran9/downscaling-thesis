---
status: accepted
---

# bias = predictor − target, and two sign conventions are live

The project default is `bias = x − y`, predictor minus target, so a positive
bias means too warm. `qm_metrics.combination_metrics` computes this.

`visualization.compute_regression_metrics` computes the **opposite** sign,
observed minus predicted. Both are live, both are correct for their own
callers, and neither is being changed. Renaming a sign convention across a
working pipeline mid-project risks silently flipping a published figure, which
is a worse outcome than documenting the inconsistency.

When calling `combination_metrics`, **always pass keyword arguments**. It takes
`(y, x)` positionally, and swapping them flips `bias` while leaving `mae`,
`rmse` and `r2` unchanged — so nothing downstream would reveal the mistake.
