# Downscaling CMIP6 temperature to ERA5-Land over the EMME

This project maps coarse global climate model temperature onto a fine
observational grid over the Eastern Mediterranean and Middle East. The
language below is the project's own. Where several words exist for one idea,
one is chosen and the rest are listed under _Avoid_.

Code keeps the CF standard variable names `tas` and `t2m`, because they match
the NetCDF files. Prose uses the terms defined here.

## The two temperature fields

**Predictor**:
The coarse climate model temperature field, after interpolation onto the fine
grid. Gloss it as "the modelled temperature" on first use in a document.
_Avoid_: TAS, T2M as prose, source, input, coarse field, X

**Target**:
The fine-resolution observed temperature field the predictor is mapped onto.
Gloss it as "the observed temperature" on first use.
_Avoid_: truth, ground truth, label, output, Y

**Downscaling**:
Producing a fine-resolution estimate from a coarse-resolution field. It names
the goal, never a particular method.
_Avoid_: super-resolution, refinement, disaggregation

## Space

**Domain**:
The study region, 24–38°N and 30–38°E. Distinct from the padded box used
during interpolation, which extends one coarse cell further on every side.
_Avoid_: study box, study area, region of interest, extent

**Pixel**:
One point of the fine observational grid. "Land pixel" is one whose value is
not masked as sea.
_Avoid_: gridpoint, fine cell, point

**Cell**:
One box of the coarse climate model grid. Always qualify as "coarse cell" on
first use, since "pixel" and "cell" are otherwise easy to swap by accident.
_Avoid_: grid box, coarse pixel, tile

## Time

**Season**:
One of DJF, MAM, JJA or SON. This is the project's only meaning of the word.
_Avoid_: quarter (which is a different, calendar-aligned grouping), trimester

**Season-year**:
The year a season is filed under, with December counted into the following
year's winter. It is the unit held out during cross-validation.
_Avoid_: water year, climate year

**Distribution window**:
The span of days whose values are pooled to form one distribution. In the
current baseline the window is the season.
_Avoid_: block, bin, chunk, period

**Fold**:
One season-year held out from fitting and kept for scoring.
_Avoid_: split, test set, holdout set

## The correction

**Transfer function**:
The fitted mapping from a predictor value to a corrected value. One is fitted
per land pixel and season.
_Avoid_: transform, correction function, mapping, adjustment

**Node**:
One paired point of the empirical quantile–quantile curve that a transfer
function is fitted through.
_Avoid_: knot, breakpoint, anchor, control point

**Percentile**:
A point of a distribution at which the correction is scored. It is a unit of
evaluation, never a unit of modelling — one transfer function serves all
percentiles.
_Avoid_: quantile (reserve that for the method's name), level

**Bias**:
Predictor minus target, so a positive bias means too warm. Where a function
computes the opposite sign, that function says so explicitly.
_Avoid_: error, offset, drift, deviation

**Climatology floor**:
The error a prediction cannot beat, set by how much the observed value itself
moves from year to year. There are two: one for mean absolute error, a
different one for root mean squared error.
_Avoid_: baseline, noise floor, irreducible error
