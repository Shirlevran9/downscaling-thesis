---
status: accepted
---

# The smoothing parameter is searched, not fitted per pixel

Per-fit generalised cross-validation for the smoothing spline was measured at
roughly 25 hours for a full run. A single value per season, selected by grid
search, is about 27× faster and gives better statistics — a per-pixel value
gives every pixel a different amount of smoothing.

A second reason: `scipy.interpolate.make_smoothing_spline` returns a plain
`BSpline` with no attribute recording the λ it chose, so an early attempt to
read the GCV value back silently fell through to the default on every fit. The
bug was invisible because the fits still succeeded. λ is now an ordinary
searched hyper-parameter with a recorded value.
