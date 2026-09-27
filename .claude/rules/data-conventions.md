---
name: data-conventions
description: Units, calendars, masks and grid handling for ERA5-Land and CMIP6 data loading.
paths:
  - "src/data_io.py"
  - "src/spatial_ops.py"
  - "src/interpolation.py"
  - "src/elevation.py"
  - "src/qm_inputs.py"
---

# Data conventions

## Units

Convert Kelvin to Celsius before any analysis: `arr - 273.15`, via
`data_io.to_celsius`. Cached predictor fields are already in °C — converting
twice is silent and produces a 273 °C offset that looks like a masking bug.

## Calendars

ERA5-Land is Gregorian; CMIP6 here is no-leap. Remove the ERA5-Land leap days
1992-02-29 and 1996-02-29, leaving **3,650 shared days** for 1990–1999.
`data_io.align_calendars` does this.

Decode CMIP6 times with `t.strftime('%Y-%m-%d')`. `pd.to_datetime` may fail on
cftime objects.

Leap-day removal shifts `dayofyear`, which is why 14-day blocks are ranked
within each year rather than by `dayofyear`.

## Grids

ERA5-Land dimensions are named `latitude` and `longitude`, not `lat`/`lon`.
CMIP6 uses `lat`/`lon`. Code that mixes them raises at runtime.

**Latitude runs descending**, 38 → 24. Longitude ascends, 30 → 38.

CMIP6 longitude may be stored as [0°, 360°]; pass it through
`spatial_ops.standardize_longitude` first.

When subsetting CMIP6, use `pad_lat=1.0, pad_lon=1.5`. Bilinear interpolation
needs four surrounding coarse cells, so border pixels fail without the pad.
The padded box is **not** the domain: the domain is 24–38°N, 30–38°E and holds
105 coarse cells; the padded subset holds 153.

## Land and sea

32.7% of domain pixels are ocean and masked NaN. The pattern is spatially
fixed. There are 7,683 land pixels.

Always pass `skipna=True` when taking ERA5-Land domain means, or the ocean
turns every average into NaN.

## Elevation

Elevation comes from **ETOPO** via NOAA ERDDAP (`elevation.fetch_etopo_elevation`),
not SRTM.

`Δz` is the sub-grid terrain: fine elevation minus the bilinearly interpolated
CMIP6 cell-mean orography. The CMIP6 file carries no orography variable, so it
is derived from the DEM.

A fitted near-surface lapse rate of about **−3 °C km⁻¹** is expected — roughly
half the free-air rate. It is fitted on a residual, and it is not a bug.

## Interpreter

Use `python3.10`. Plain `python3` resolves to a Homebrew interpreter with no
xarray, so every script fails with `ModuleNotFoundError`. Install packages with
`python3.10 -m pip`, never bare `pip`.

For Streamlit, use `python3.10 -m streamlit`, not the bare `streamlit` command,
which belongs to an anaconda environment without xarray.
