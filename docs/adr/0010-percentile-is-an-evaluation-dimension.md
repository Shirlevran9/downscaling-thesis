---
status: accepted
---

# Percentile is an evaluation dimension, not a model dimension

One transfer function is fitted per (land pixel, season) on pooled daily
values, and all five reported percentiles are read off that same function.

Fitting five independent per-percentile models would allow P25 to be predicted
above P50, and would leave almost no sample: the training data for one
percentile is ten annual values. A single monotone function guarantees
P5 ≤ P25 ≤ P50 ≤ P75 ≤ P90 by construction.

Errors are still *reported* per percentile, which is where the interesting
structure is — the tails are where the methods differ.
