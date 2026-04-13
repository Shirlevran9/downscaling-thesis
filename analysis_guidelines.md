# Analysis Guidelines: Spatial Temperature Data

Guidelines for working with gridded spatial temperature data in the context of
statistical downscaling. Covers plotting conventions, data preprocessing,
calendar handling, and figure caption standards for publication-quality output.

---

## 1. Spatial Map Standards

### 1.1 Axis tick labels

Longitude (x-axis) and latitude (y-axis) ticks must carry degree symbols and
cardinal direction suffixes. Never use plain decimal numbers.

| Correct | Incorrect |
|---|---|
| `30°E`, `32°E`, `34°E` | `30`, `32`, `34` |
| `24°N`, `26°N`, `28°N` | `24.0`, `26.0`, `28.0` |

Implementation in matplotlib:

```python
import matplotlib.ticker as mticker

def lon_fmt(x, pos): return f"{int(x)}°E" if x >= 0 else f"{int(-x)}°W"
def lat_fmt(y, pos): return f"{int(y)}°N" if y >= 0 else f"{int(-y)}°S"

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_fmt))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_fmt))
```

Use `visualization.apply_map_formatting(ax, region)` for automatic tick
placement and formatting.

### 1.2 Geographic aspect ratio

A rectangular lat/lon plot is geographically distorted unless the display
aspect ratio accounts for the convergence of meridians at higher latitudes.
At latitude φ, one degree of longitude spans `cos(φ)` of the distance of one
degree of latitude.

Set the axes aspect via:

```python
import numpy as np
lat_mid = (region['south'] + region['north']) / 2.0
ax.set_aspect(1.0 / np.cos(np.radians(lat_mid)))
```

For the EMME domain (lat_mid ≈ 31°N): `aspect ≈ 1.167`.

Use `visualization.make_spatial_figure(ncols, region)` which handles this
automatically.

### 1.3 Colormap selection

| Data type | Recommended colormap | Notes |
|---|---|---|
| Absolute temperature (climatology) | `RdYlBu_r` | Blue = cold, red = warm |
| Temperature anomaly | `RdBu_r` | Centred at zero; verify `vmin = -vmax` |
| Missing-value fraction | `YlOrRd` | White = no missing, yellow–red = high missing |
| Count data (pixels per cell) | `YlGnBu` | Sequential, perceptually uniform |
| Categorical (cell assignment) | `tab20b` | Enough colours for ~97 cells |

Avoid `jet`/`rainbow` — they are not perceptually uniform and introduce false
features.

### 1.4 Colorbars

- Always label the colorbar with the variable name and units:
  `"Mean temperature (°C)"`, `"Fraction of missing values"`.
- Position: right of the map panel (`fraction=0.046, pad=0.04` for single
  panel; use `ax=axes` list for shared bar across multiple panels).
- For shared-clim side-by-side plots, use one colorbar anchored to both axes.

### 1.5 Figure size

Figure dimensions should reflect the geographic extent of the domain. For the
EMME domain (24–38°N, 30–38°E):

| Layout | Suggested figsize |
|---|---|
| Single map panel | `(5.5, ~8)` inches |
| Two panels side by side | `(11, ~8)` inches |
| 2×2 seasonal panels | `(10, ~14)` inches |
| 4×2 ERA5 vs CMIP6 seasonal comparison | `(11, ~22)` inches |

Use `visualization.make_spatial_figure()` to compute sizes automatically.

### 1.7 Seasonal comparison maps

When comparing ERA5-Land and CMIP6 seasonal climatologies, use a 4×2 grid
(4 seasons × 2 datasets) rather than two separate 4-panel figures. This
layout enables direct visual comparison within each season row.

Use `visualization.plot_seasonal_comparison_maps(...)`:

```python
fig = viz.plot_seasonal_comparison_maps(
    era5_seasonal_means,   # dict: {DJF, MAM, JJA, SON} → 2-D np.ndarray
    cmip_seasonal_means,   # same structure
    era5_lats, era5_lons,
    cmip_lats, cmip_lons,
    region=REGION,
    save_path=PLOT_DIR / 'fig04_seasonal_comparison.png',
)
```

Key layout details:
- Left column header: "ERA5-Land"; right column header: "CMIP6 CESM2-WACCM"
- Season labels on the left of each row: DJF, MAM, JJA, SON
- One shared colorbar positioned outside the grid on the right
  (`fig.subplots_adjust(right=0.87)` + `fig.add_axes([0.90, 0.10, 0.025, 0.78])`)
- Colorbar ticks formatted with `_temp_formatter` (`20°`, `25°`, etc.)

### 1.6 Panel labelling

Multi-panel figures must label each panel with `(a)`, `(b)`, `(c)`, `(d)`,
placed in the panel title or as a text annotation at the upper-left corner.
The figure caption must reference each panel label explicitly.

---

## 2. Temporal Plot Standards

### 2.1 Date-aware x-axis

Always use `pd.DatetimeIndex` for the x-axis when plotting time series.
Format major ticks by year (`mdates.YearLocator`) and minor ticks by month
if the series spans more than one year.

```python
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
```

### 2.2 Seasonal cycle plots (monthly climatology)

- Use month abbreviations on the x-axis: `Jan, Feb, ..., Dec`.
- Mark the x-axis from 1 to 12 with `ax.set_xticks(range(1, 13))`.
- Include markers at each data point for clarity.

### 2.3 Multi-series styling

- Use `tab10` palette (first colour: blue, second: orange, etc.) for
  consistent cross-figure styling.
- Line width 1.2–1.8 pt for time series; 1.8–2.0 pt for climatology lines.
- Include a legend with `framealpha=0.8`.

### 2.4 Temperature axis tick format

Temperature axes (both x and y) must display values as `20°`, `25°`, `30°`
— degree symbol only, **no repeated `°C` unit label on every tick**. The
unit appears once in the axis label or colorbar label.

Implementation in matplotlib using `_temp_formatter` defined in
`visualization.py`:

```python
import matplotlib.ticker as mticker

def _temp_formatter(x, pos):
    return f"{int(x)}°"

ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
```

Apply to:
- Y-axis of `plot_domain_timeseries` and `plot_monthly_climatology`
- Both axes of `plot_scatter_regression` and `plot_residual_analysis`
- Y-axis of `plot_temperature_distributions` (vertical boxplot)
- Colorbar ticks of all spatial map functions

---

## 3. Calendar and Temporal Alignment

### 3.1 ERA5-Land (Gregorian) vs CMIP6 (no-leap)

ERA5-Land includes 29 February in leap years; most CMIP6 models (including
CESM2-WACCM) use a no-leap calendar. This produces a mismatch of 2 days for
the decade 1990–1999 (leap years 1992 and 1996).

**Required action:** Remove ERA5-Land leap days before computing the shared
time axis.

```python
import pandas as pd, numpy as np

times = pd.to_datetime(era5_ds.time.values)
mask = ~((times.month == 2) & (times.day == 29))
era5_aligned = era5_ds.isel(time=mask)
```

Use `data_io.align_calendars(era5_ds, cmip_ds)` for automatic handling.

### 3.2 CMIP6 cftime decoding

CMIP6 files store time as `cftime.DatetimeNoLeap` objects. Convert to strings
for safe cross-dataset comparison:

```python
date_strs = [t.strftime('%Y-%m-%d') for t in cmip_ds.time.values]
```

`pd.to_datetime` may fail on non-standard cftime objects; always convert via
`strftime` first.

### 3.3 Season definitions

Meteorological seasons (used throughout):

| Season | Months |
|---|---|
| DJF (winter) | December, January, February |
| MAM (spring) | March, April, May |
| JJA (summer) | June, July, August |
| SON (autumn) | September, October, November |

Note: for a year spanning September to August (as used in some analyses),
DJF crosses year boundaries. Use `data_io.seasonal_split()` to handle this
correctly.

---

## 4. Temperature Data Conventions

### 4.1 Kelvin → Celsius conversion

Both ERA5-Land and CMIP6 store temperature in Kelvin. Convert by subtracting
273.15 K. Always check the `units` attribute before applying:

```python
if arr.attrs.get('units', '').lower() in {'k', 'kelvin'} or float(arr.mean()) > 150:
    arr = arr - 273.15
    arr.attrs['units'] = '°C'
```

Use `data_io.to_celsius(arr)` for consistent application.

### 4.2 Domain mean vs pixel statistics

- **Domain mean**: computed with `skipna=True` to exclude ocean pixels from
  the average; otherwise the mean is biased toward the fixed missing fraction.
- **Pixel statistics**: report separately for land pixels only. Include:
  minimum, mean, standard deviation, and maximum.
- **Bias**: report CMIP6 mean minus ERA5-Land mean (positive = warm bias).

### 4.3 Pressure-level variables

For upper-atmosphere variables (T850, Z850, Z250, U850, V850):

- **T850**: temperature at ~850 hPa (~1.5 km above sea level). Units: K.
- **Z850/Z250**: geopotential (J kg⁻¹); divide by g = 9.80665 m s⁻² to
  obtain geopotential height in metres.
- **U, V**: zonal and meridional wind components (m s⁻¹); positive U is
  westerly (eastward), positive V is southerly (northward).

---

## 5. Land–Sea Mask Handling

### 5.1 Identifying the mask

The ERA5-Land land–sea mask is inferred from the NaN pattern:
- Compute the temporal mean; pixels that are NaN in the mean are always NaN.
- Verify that the fraction is constant over time (use
  `era5_temp.isnull().mean(dim=['lat', 'lon']).to_series()`).

### 5.2 Applying the mask in computations

- Spatial mean: use `skipna=True` to average only over land pixels.
- Plotting: `pcolormesh` will display NaN as transparent (no color); ensure
  the background colour is set to white or grey.
- DataFrame operations: use `.dropna(subset=['t2m'])` after flattening.

### 5.3 Reporting

State the missing fraction as a percentage of total grid cells in the domain,
e.g. "Approximately 32.7% of ERA5-Land grid cells within the study domain are
classified as ocean and contain no data."

---

## 6. Nearest-Neighbour Grid Assignment

### 6.1 Method

Each ERA5-Land pixel (0.1°) is assigned to the nearest CMIP6 cell (~1°) by
minimising Euclidean distance in degrees along latitude and longitude
independently:

```python
cmip_lat_idx = np.argmin(np.abs(era5_lats[:, None] - cmip_lats[None, :]), axis=1)
cmip_lon_idx = np.argmin(np.abs(era5_lons[:, None] - cmip_lons[None, :]), axis=1)
```

### 6.2 Statistics to report

Always report pixel-per-cell counts with:
- Mean, standard deviation, minimum, and maximum.
- Narrative interpretation, e.g. "Cells located entirely within the Arabian
  Peninsula receive the greatest number of ERA5-Land pixels (maximum: N),
  while coastal cells straddling the land–sea boundary receive the fewest
  (minimum: N)."

### 6.3 Limitations

- **Boundary artefacts**: pixels near cell boundaries may be assigned to a
  sub-optimal cell. Examine residuals vs. distance to cell centre to quantify
  this effect (Fig. 8 in the notebook).
- **Alternative methods**: bilinear interpolation and area-weighted remapping
  (e.g. using `xesmf`) assign weights to multiple cells and are more accurate
  for pixels near cell boundaries. These are recommended when fine-scale
  boundary effects matter for model accuracy.
- **Grid padding**: always include padding (~0.5–1.5°) when extracting the
  CMIP6 domain to ensure cells straddling the boundary are retained. The
  footprint of a ~1° cell extends ±0.5° from its centre.

---

## 7. Figure Caption Standard

Figure captions must follow this structure:

> **Fig. N.** *Short descriptive title.* Panel (a): description of left/top
> panel. Panel (b): description of right/bottom panel. Data: dataset name,
> variable, period, study region.

**Example:**

> **Fig. 2.** Mean 2 m temperature climatology, 1990–1999. Panel (a):
> ERA5-Land reanalysis at 0.1° resolution. Panel (b): CMIP6 CESM2-WACCM
> historical simulation at ~1° resolution. Both panels share the same colour
> scale. Data: ERA5-Land daily 2 m temperature and CMIP6 CESM2-WACCM TAS,
> study region 24–38°N, 30–38°E.

**Rules:**
- Captions appear *beneath* figures in the document/notebook.
- Do not repeat the axis labels verbatim in the caption; instead describe what
  the figure shows and what can be concluded.
- Mention data source, period, and spatial extent.
- For multi-panel figures, reference each panel label.

---

## 8. Residual Analysis Checklist

After fitting any regression model (linear or otherwise):

1. **Homoscedasticity**: plot residuals vs. predicted values or predictor
   values. A homogeneous horizontal band around zero indicates constant error
   variance. Funnel-shaped spread indicates heteroscedasticity.

2. **Spatial dependency**: plot residuals vs. Euclidean distance from the
   assigned coarse-grid cell centre. A non-zero trend indicates that
   boundary-assignment artefacts inflate errors for pixels far from the cell
   centre.

3. **Temporal autocorrelation**: compute the autocorrelation function (ACF)
   of daily domain-mean residuals. Significant autocorrelation at lag 1
   indicates temporal structure not captured by the predictor.

4. **Seasonal stratification**: compute mean residual by season (DJF/MAM/
   JJA/SON). Season-dependent bias indicates that a single global model
   cannot capture the full seasonal variability; a season-specific or
   seasonally-stratified model may be needed.

5. **Spatial map of residuals**: compute mean residual per ERA5-Land pixel
   and plot as a spatial map. Systematic spatial patterns suggest unresolved
   topographic or land-cover effects.

---

## 9. Scientific Writing Conventions for Climate Papers

- **Voice and tense**: use the passive voice in Methods and Results sections
  ("The data were processed...", "Temperature was converted..."). Use the
  active voice sparingly in Discussion.
- **Numbers**: report statistics to an appropriate number of significant
  figures (e.g. "19.2°C", not "19.23456°C"). Round to one decimal place for
  temperatures; use three significant figures for correlation coefficients.
- **Units**: always include units; use the SI convention: °C for temperature,
  ° (degrees) for lat/lon, km for distance, days for time intervals.
- **Figure references**: always cite figures parenthetically: "(Fig. 2)",
  "(Figs. 3a–b)".
- **Abbreviations**: define on first use: "CMIP6 (Coupled Model
  Intercomparison Project Phase 6)", "ERA5-Land", "GCM (global climate
  model)", "OLS (ordinary least squares)".
- **Correlations**: report as "Pearson *r* = 0.9" (italic *r*).
- **Bias**: defined as model minus observation (positive = overestimate).

---

## 10. Data Sources and References

### Primary datasets

| Dataset | Source | URL |
|---|---|---|
| ERA5-Land daily statistics | Copernicus Climate Data Store (CDS) | cds.climate.copernicus.eu |
| CMIP6 CESM2-WACCM | ESGF (Earth System Grid Federation) | esgf-node.llnl.gov |
| SRTM elevation (if used) | CGIAR-CSI, v4.1 | srtm.csi.cgiar.org |

### Key references

- Maraun, D., & Widmann, M. (2018). *Statistical Downscaling and Bias Correction for Climate Research*. Cambridge University Press. — Comprehensive treatment of statistical downscaling methods.
- Muñoz-Sabater, J., et al. (2021). ERA5-Land: A state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383. — ERA5-Land dataset description.
- Eyring, V., et al. (2016). Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6). *Geoscientific Model Development*, 9, 1937–1958. — CMIP6 experiment design.
- Wood, A. W., et al. (2004). Hydrologic implications of dynamical and statistical approaches to downscaling climate model outputs. *Climatic Change*, 62, 189–216. — BCSD (bias-correction spatial disaggregation) method.
- Bodenheimer, S., Nirel, R., Lensky, I. M., & Dayan, U. (2021). The synoptic skill of aerosol optical depth and angstrom exponent levels over the Mediterranean Basin. *International Journal of Climatology*, 41(3), 1801–1820. — Mediterranean domain definition used in this project.
- Yosef, Y., Aguilar, E., & Alpert, P. (2019). Changes in extreme temperature and precipitation indices using an innovative daily homogenized database in Israel. *International Journal of Climatology*, 39(13), 5022–5045. — Temperature trends in the EMME region.

---

## 11. Python Environment Requirements

The following packages are required:

```
xarray >= 2023.1
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
seaborn >= 0.12
netCDF4
h5netcdf
scipy
```

Install with:

```bash
pip install xarray numpy pandas matplotlib seaborn netCDF4 h5netcdf scipy
```

Optional (for enhanced geographic projections):

```bash
pip install cartopy
```

If `cartopy` is not available, use `visualization.apply_map_formatting()` which
implements geographic aspect correction and degree-formatted tick labels using
standard `matplotlib`.

---

## 8. Figure Design Guidelines

### 8.1 Side-by-side map panels — Y-axis suppression

When two spatial map panels share the same latitude range (e.g., `plot_side_by_side_maps`, `plot_seasonal_comparison_maps`), suppress the Y-axis tick labels and label on the **right panel** to avoid redundant annotation. The latitude axis is fully labelled on the left panel only.

```python
# After the per-panel formatting loop:
axes[1].set_ylabel("")
axes[1].set_yticklabels([])
axes[1].tick_params(axis="y", left=False)
fig.subplots_adjust(right=0.85, wspace=0.08)
```

Apply the same principle to every column > 0 in multi-column grids:

```python
if col > 0:
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.tick_params(axis="y", left=False)
```

### 8.2 Binary land/sea mask map

The land–sea mask is displayed as a two-colour binary map (no continuous colourbar):
- **Land pixels (True):** `#a1d99b` (light green)
- **Ocean / missing pixels (False):** `#6baed6` (light blue)
- Use `matplotlib.patches.Patch` legend items instead of a colourbar.
- Title: `"ERA5-Land coverage: land and ocean pixels"`
- Saved as `fig01b_land_sea_mask.png`.

Use `visualization.plot_land_sea_mask(land_mask_2d, lats, lons, region, save_path)`.

### 8.3 Narrative section order (notebook)

The notebook follows this section structure:

| Section | Title |
|---|---|
| 0 | Environment Setup |
| 1 | Datasets |
| 2 | Main Statistics |
| 3 | Data Alignment & Preprocessing |
| 4 | Exploratory and Descriptive Analysis |
| 5 | Pairing Mechanism |
| 6 | Coarse–Fine Temperature Relationship |
| 7 | Feature Engineering Preview *(WIP)* |
