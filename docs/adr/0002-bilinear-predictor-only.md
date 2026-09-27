---
status: accepted
---

# One predictor, bilinear interpolation

The fitted baseline uses the bilinear predictor alone. The earlier diagnostic
grid compared five coarse-to-fine fields (k-NN with k=4 and k=9, bilinear, and
two trilinear variants with a lapse-rate term); none beat bilinear by enough to
justify carrying five predictors through a model comparison.

Holding the predictor fixed makes the transfer-function comparison readable:
differences between methods are differences between methods, not between
interpolation schemes.

The cached bilinear field was validated against a fresh recompute before being
locked in — bit-identical, `max|diff| = 0.000e+00`.
