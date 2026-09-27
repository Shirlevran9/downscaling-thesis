---
status: accepted
---

# The correction is applied to daily values, then percentiles are taken

Not the reverse. Mapping a percentile directly, `h(Q_p(x))`, equals mapping the
days and then taking the percentile, `Q_p(h(x))`, only when `h` is monotone
**and** quantiles are plain order statistics.

This project uses numpy's interpolating estimator, under which the two differ
by up to 0.39 °C for QUANT. The daily path is the correct one.

This is also why monotonicity is treated as a correctness property rather than
a nicety: it is what makes percentile-level evaluation well defined at all.
`predict_percentiles` enforces it and `n_nonmono` records where the repair
fired.
