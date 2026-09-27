---
status: accepted
---

# Hyper-parameters are chosen per season, not per pixel

Each method gets one hyper-parameter value per season, selected on a pooled
sample of pixels, rather than one value per (pixel, season).

A per-pixel value would mean 7,683 different amounts of smoothing across the
domain, which cannot be described in a Methods section and adds estimation
noise on a ten-year record. One value per season is four numbers that can be
reported in a table.

Selection scores on inner folds of the training years only, so the held-out
season-year stays clean.
