---
name: visualization
description: Figure standards for spatial maps, time series, colorbars and captions in this project.
paths:
  - "src/visualization.py"
  - "src/qm_visualization.py"
  - "src/qm_transform_viz.py"
  - "src/vis_constants.py"
  - "plots/**"
---

# Figure standards

These are non-negotiable for every figure this project produces.

## The code is the source of truth

Never hardcode a font size, colour or alpha. They live in `src/vis_constants.py`
— edit that file to change any aesthetic. Never restate its values in prose;
read the file.

Reuse the helpers rather than reimplementing them:

- `visualization.apply_map_formatting(ax, region)` — degree ticks and aspect
- `visualization.make_spatial_figure(ncols, region)` — correctly sized axes
- `visualization._temp_formatter` — temperature tick labels
- `visualization._overlay_coarse_grid`, `_draw_highlight_box` — private, but
  import them rather than writing a second copy

`qm_visualization.py` and `qm_transform_viz.py` import these on purpose. Adding
a local copy of any of them is a bug.

## Axes

Longitude ticks read `30°E`, `32°E`; latitude ticks read `24°N`, `26°N`. Never
bare decimals.

Set the aspect to `1 / cos(lat_mid)` so the map is not geographically
distorted. For this domain, lat_mid ≈ 31°N and the aspect ≈ 1.167.

Temperature axes use `_temp_formatter`, which writes `20°`, not `20°C` repeated
on every tick. The unit belongs in the axis label, once.

## Colour

| Data | Colormap |
|---|---|
| Absolute temperature | `RdYlBu_r` |
| Anomaly or bias | `RdBu_r`, symmetric — assert `vmin == -vmax` |
| Missing fraction | `YlOrRd` |
| Counts | `YlGnBu` |

Never `jet` or `rainbow`: they are not perceptually uniform and invent features
that are not in the data.

## Colorbars

Always labelled with the variable and its unit — `Mean temperature (°C)`.

Position right of the panel by default (`fraction=0.046, pad=0.04`). The
exception is the multi-panel seasonal comparison, which uses one horizontal bar
across the bottom.

Heatmaps of counts use `fmt='d'`: fill NaN to 0 and cast to int before plotting,
so no count is drawn with a decimal point.

## Multi-panel figures

Seasonal ERA5-vs-CMIP6 comparisons use `plot_seasonal_comparison_maps` and its
**2 rows × 5 columns** layout — DJF and MAM on row 0, JJA and SON on row 1,
column 2 a narrow separator, one horizontal colorbar beneath. Do not revert to
the older 4×2 layout and do not build it from two `plot_seasonal_maps` calls.

Label panels `(a)`, `(b)`, `(c)`, `(d)`, and reference every label in the
caption. Never nest numbering — `Fig1a1` is not a panel label.

Suppress y-axis labels on every panel except the leftmost, so the shared axis is
not drawn four times.

Never split one multi-panel figure across two sections of a document.

## Captions

Captions sit immediately below the figure and read:

> **Fig. N.** Short title. Panel (a): description. Panel (b): description.
> Data: source, period.

A caption must stand alone — a reader who sees only the figure and its caption
should not need the body text. The title says what is shown, not what it means:
"OLS residual distribution by CMIP6 cell sea fraction", not "Residuals worsen
over the sea".

## Numbers in figures

Two decimal places unless there is a reason otherwise. Pair every central
measure with a dispersion measure: `19.2 ± 8.7 °C` (mean ± SD) or
`20.1 [12.7, 26.3] °C` (median [IQR]). A space precedes the unit.
