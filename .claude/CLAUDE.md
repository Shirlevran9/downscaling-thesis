# CLAUDE.md — Project Context for AI Agents

This file provides context for Claude Code agents working in this repository.
Read it before making any changes.

---

## Project Overview

**Goal:** Develop a statistical downscaling framework to map coarse-resolution
CMIP6 global climate model output (~1° grid) to fine-resolution ERA5-Land
reanalysis temperature (0.1° grid) over the Eastern Mediterranean and Middle
East.

**Study region:** 24–38°N, 30–38°E (EMME)
**Baseline analysis period:** 1990–1999
**Target variable:** Daily mean 2 m air temperature (T2M / TAS)

**Future scope (from `Data 8 Mar 2026.docx`):**
- Training period: 1980–2004
- Test periods: 2005–2014 and 2015–2025
- Projection period: 2081–2100
- Full Mediterranean Basin (24–47°N, 11°W–40°E)
- Six CMIP6 GCMs (CESM2-WACCM + 5 others from Andre Klif's set)
- Additional predictors: T850, Z850, Z250, U850, V850, global mean TAS

---

## Repository Structure

```
Thesis/
├── data/                          # Raw NetCDF files (gitignored)
│   ├── t2m_ERA5land_daily_YYYY.nc # ERA5-Land annual files (1990–1999)
│   └── tas_day_CESM2-WACCM_...nc  # CMIP6 historical 1990–1999
│
├── src/                           # Reusable Python modules
│   ├── __init__.py
│   ├── data_io.py                 # Data loading, calendar alignment, unit conversion
│   ├── spatial_ops.py             # Subsetting, nearest-neighbour assignment, land mask
│   ├── interpolation.py           # Bilinear CMIP6 → ERA5 regridding (CDO / scipy)
│   ├── elevation.py               # ETOPO DEM download, regridding, DataFrame merge
│   ├── predictors.py              # The 5 coarse-to-fine predictor fields (QM baseline)
│   ├── quantile_windows.py        # Distribution windows + per-window percentiles
│   ├── qm_inputs.py               # Shared loading + validated bilinear predictor
│   ├── qm_nodes.py                # Season/fold bookkeeping + q-q node reduction
│   ├── qm_transforms.py           # The 8 quantile-mapping transforms, one interface
│   ├── qm_cv.py                   # Leave-one-season-year-out cross-validation
│   ├── qm_eval.py                 # Model metrics + the climatology floor
│   ├── qm_transform_viz.py        # Figures for the fitted-transform analysis
│   ├── qm_metrics.py              # Quantile-mapping skill metrics and aggregates
│   ├── qm_visualization.py        # Quantile-mapping figures
│   ├── vis_constants.py           # Central style constants (font sizes, colours, CI alpha)
│   └── visualization.py          # All plotting functions (scientific formatting)
│
├── notebooks/
│   ├── 01_initial_data_exploration.ipynb   # Descriptive analysis
│   ├── 02_models.ipynb                     # Daily OLS baselines (M1 NN, M2 bilinear, M3 +elev)
│   └── 03_quantile_mapping.ipynb           # Quantile-mapping baseline (distribution level)
│
├── dashboards/
│   └── qm_dashboard.py            # Streamlit app for the quantile-mapping grid
│
├── history/                       # Archived first-version files
│   ├── notebooks/downscaling_data_exploration.ipynb
│   ├── initial_findings_v1.md
│   └── plots/
│
├── plots/                         # Generated figures (fig0N_*.png)
├── initial_findings.md            # Scientific summary of initial analysis
├── guidelines/analysis_guidelines.md         # Guidelines for spatial/temporal analysis and figure standards
├── guidelines/findings-presentation-guidelines.md  # Supervisor-derived rules for report writing and figures
├── requirements.txt               # Python dependencies
└── scripts/
    ├── fetch_remote_climate_data.sh  # SSH fetch from Moriah HPC cluster
    ├── run_quantile_mapping.py       # Driver for the predictor-comparison grid
    ├── run_qm_transforms.py          # Driver for the fitted-transform analysis
    └── sanity_qm_transforms.py       # Correctness checks (no test framework here)
```

---

## Module Architecture

All analysis code lives in `src/`. The notebook imports from these modules.
**Do not write analysis logic directly in notebooks** — add it to the
appropriate module instead.

### `src/data_io.py`
- `open_dataset(path)` — netCDF4 → h5netcdf fallback
- `load_era5_land(files, region, pad_lat, pad_lon)`
- `load_cmip6(file, region, pad_lat, pad_lon, var)`
- `to_celsius(arr)` — K → °C conversion
- `align_calendars(era5_ds, cmip_ds)` → returns aligned datasets + shared dates list
- `build_paired_dataframe(era5_temp, cmip_tas, assignment_df, land_mask_2d, shared_dates)`
- `seasonal_split(obj)` → dict `{DJF, MAM, JJA, SON}`
- `compute_global_daily_mean(cmip_ds, var)` → pd.Series (area-weighted)

### `src/spatial_ops.py`
- `standardize_longitude(ds, lon_name)` — [0,360] → [-180,180]
- `subset_box(ds, lat_name, lon_name, region, pad_lat, pad_lon)`
- `compute_land_mask(era5_temp_da)` → 2D bool array
- `assign_era5_to_cmip_cells(era5_lats, era5_lons, cmip_lats, cmip_lons, land_mask_2d)` → DataFrame
- `pixel_counts_per_cmip_cell(assignment_df)` → DataFrame with n_pixels
- `pixel_count_stats(assignment_df)` → dict (mean, std, min, max)
- `compute_cell_edges(centers)` → boundary edge array
- `compute_distance_to_cell_center(assignment_df)` → pd.Series (degrees)

### `src/interpolation.py`
- `write_domain_grid_file(lats, lons, output_path)` — CDO lonlat grid description
- `remapbil_cdo(input_nc, grid_file, output_nc, cdo_bin)` → bool
- `remapbil_scipy(cmip_da, target_lats, target_lons)` → np.ndarray (accepts 2-D or 3-D)
- `interpolate_cmip_to_era5(cmip_da, target_lats, target_lons, cache_nc, cdo_bin)` → xr.DataArray
  dims `(time, latitude, longitude)`; CDO first, scipy fallback, NetCDF cache

### `src/elevation.py`
- `fetch_etopo_elevation(region, cache_path)` → Path (ERDDAP download)
- `regrid_elevation(raw_nc_path, target_lats, target_lons)` → xr.DataArray
- `get_domain_elevation(region, target_lats, target_lons, cache_dir)` → xr.DataArray
  two-level cache: `etopo_domain.nc` (raw) → `elevation_era5grid.nc` (regridded)
- `add_elevation_to_df(df, elev_da)` → DataFrame with an `elevation` column

### `src/predictors.py`
The five coarse-to-fine predictor fields of the quantile-mapping baseline. All
are on the ERA5-Land grid, dims `(time, latitude, longitude)`, cached as NetCDF
under `data/cache/predictors/`.
- `PREDICTOR_NAMES = ["knn4", "knn9", "bilinear", "trilinear_fit", "trilinear_fixed"]`
- `FIXED_LAPSE_RATE = -0.0065` (°C m⁻¹)
- `knn_indices(era5_lats, era5_lons, cmip_lats, cmip_lons, k)` → (n_pix, k) flat indices;
  uses `scipy.spatial.cKDTree` with cos-latitude longitude scaling.
  **`spatial_ops.assign_era5_to_cmip_cells` cannot be reused here** — it is a
  separable per-axis argmin and supports k=1 only.
- `knn_predictor(cmip_da, era5_lats, era5_lons, k, cache_nc, time_chunk)` → xr.DataArray
- `coarse_orography_on_fine_grid(elev_da, cmip_lats, cmip_lons)` → `(coarse_da, dz_da)`;
  `dz = elev_fine − bilinear(CMIP6 cell-mean orography)`, the sub-grid terrain
- `fit_lapse_rate(era5_temp, bilin_da, dz_da, land_mask_2d, dates)` → dict
  keys `all`, `DJF`, `MAM`, `JJA`, `SON` plus `n_<key>`; Γ in °C m⁻¹
- `trilinear_predictor(bilin_da, dz_da, gamma, dates, name, cache_nc)` → xr.DataArray
- `build_all_predictors(cmip_tas, era5_temp, land_mask_2d, region, dates, cache_dir, force)`
  → `(fields, meta)`; `meta` has `lapse_rates`, `dz`, `elevation`, `coarse_orography`
- `load_cached_predictors(cache_dir, names)` → dict — read-only, refits nothing.
  **Use this from notebooks**, not `build_all_predictors`.

### `src/quantile_windows.py`
- `SCHEMES = ["14d", "month", "quarter", "year"]`, `PERCENTILES = (5, 25, 50, 75, 90)`
- `make_windows(dates, scheme)` → `(windows_df, group_codes)`; non-overlapping and
  calendar aligned. 14-day blocks are ranked **within each year** (not by
  `dayofyear`) because leap-day removal shifts `dayofyear` in 1992 and 1996.
  Window counts: 260 / 120 / 40 / 10; labels `1999-B12`, `1999-06`, `1999-Q2`, `1999`.
- `window_percentiles(field_3d, group_codes, n_windows, percentiles)` →
  (n_windows, n_lat, n_lon, n_q) float32
- `build_percentile_table(era5_temp, predictor_da, land_mask_2d, dates, scheme, percentiles, static_cols)`
  → DataFrame, one row per (window × land pixel), **wide on percentile**:
  `window_id, window_label, window_start, n_days, lat, lon, y_p{q}…, x_p{q}…`
  plus merged static columns. Merge keys `lat`/`lon` are float32.

### `src/qm_metrics.py`
Sign convention: **`bias = x − y`** (predictor minus observation, positive =
too warm). This is the negative of `visualization.compute_regression_metrics`'s
`bias`, which is observed minus predicted.
- `METRIC_FILES` — the five aggregate file names
- `pct_table_path(qm_dir, predictor, scheme)` → Path
- `load_pct_table(qm_dir, predictor, scheme, columns)` → DataFrame
- `slice_percentile(df, q)` → DataFrame with `y`, `x`, `bias` in place of the wide columns
- `combination_metrics(y, x)` → dict: `n, bias, bias_sd, mae, rmse, pearson_r, r2, ols_slope, ols_intercept`
- `metrics_table(qm_dir, predictors, schemes, percentiles)` → one row per combination
- `per_pixel_bias(df, q)`, `per_window_bias(df, q)`,
  `elevation_binned_bias(df, q, n_bins)` (equal-count bins),
  `sea_fraction_binned_bias(df, q, n_bins)` (equal-width bins)
- `build_all_aggregates(qm_dir, predictors, schemes, percentiles, ...)` → dict;
  writes all five parquet aggregates, reading each percentile table once

### `src/qm_inputs.py`
Shared loading for both quantile-mapping drivers, so the region box and the
leap-day handling cannot drift apart between them.
- `REGION` — the EMME domain dict
- `load_aligned_inputs(data_dir, region)` → dict with `era5_temp`, `cmip_tas`
  (both °C, loaded), `shared_dates`, the four coordinate axes, `land_mask_2d`
- `load_bilinear_predictor(cache_dir, cmip_tas, era5_lats, era5_lons, shared_dates)`
  → xr.DataArray. Cache-first, but **validates** the grid and the date axis
  before use and rebuilds on mismatch. Returns °C — do not convert again.

### `src/qm_nodes.py`
Season/fold bookkeeping and the q–q node reduction. The only place that knows the
season convention and the quantile plotting positions.
- `SEASONS`, `MIN_TRAIN_DAYS = 200`, `MIN_DISTINCT_NODES = 8`
- `DailySeason` — frozen dataclass: `x`/`y` `(n_days, n_pix)` float32, `syear`,
  `lat`, `lon`; `.season_years()` lists the folds
- `QQNodes` — frozen dataclass: `x`/`y` `(n_nodes, n_pix)` plus raw-day
  `x_lo/x_hi/y_lo/y_hi/x_mean/x_sd/y_mean/y_sd`, `n_train`, `n_distinct`,
  `valid`; `.subset(cols)` restricts to pixel columns
- `node_probs(n)` — Hazen positions `(j−0.5)/n`
- `select_pixels(land_mask_2d, lats, lons, stride, max_pixels, seed)` → DataFrame
  with `pix_id`, `row`, `col`, `lat`, `lon` (float32). **Canonical pixel order.**
- `build_daily_seasons(era5_temp, predictor_da, dates, pixels, seasons)` → dict
- `fold_nodes(ds, hold_out, n_nodes, min_days)` → QQNodes. Pairwise-complete
  masking, `nanpercentile`, de-tying, validity flags.

### `src/qm_transforms.py`
The eight transforms behind one batched interface. Arrays are `(n, n_pix)` —
batched over pixels, so five of eight methods are a few numpy calls for the whole
domain.
- `QQTransform` — base class. `fit(nodes)`, `predict(x)` (any order),
  `predict_percentiles(xq)` (**enforces non-decreasing output**),
  `diagnostics()`. Subclasses implement `_fit` and `_raw_predict`.
- `IdentityQM`, `NormalQM`, `LinearQM`, `PolyQM`, `QuantQM`, `RQuantQM`,
  `SSplinQM`
- `TRANSFORMS` (name → class), `METHOD_NAMES`, `RAW_METHOD`, `HYPER_GRIDS`
- `SSplinQM` has `is_batched = False` (internal Python loop)

### `src/qm_cv.py`
- `loso_folds(ds)` → season-years to hold out
- `build_fold_nodes(daily, n_nodes)` → nodes for all 39 folds, **shared across
  methods**; saves ~70 s per method
- `season_percentiles(daily_block, percentiles)` — numpy's default estimator,
  matching `window_percentiles`
- `select_hyper(method, ds, ...)` → `(best_hp, scores)`; one value per season
- `run_method(method, daily, pixels, ..., nodes_by_fold=None)` →
  `(pred, diag, hyper)`

### `src/qm_eval.py`
- `melt_predictions(pred, percentiles)` — wide → one row per percentile
- `observed_percentile_sd(long)` → `obs_mad` (**the MAE floor**), `obs_sd` (the
  RMSE floor), `obs_mean`, `obs_median`, `n_years`
- `per_pixel_error(long)`, `method_summary(long, floor, diag)`
- `daily_moments(daily, pixels)` — skewness/kurtosis, the normality check

### `src/qm_transform_viz.py`
- `METHOD_LABELS`, `METHOD_FAMILIES`, `SEASON_LABELS`
- `plot_error_map_panels(...)` — 4 seasons × 5 percentiles of error maps
- `plot_method_heatmap(...)`, `plot_method_bars(...)` — both draw the floor
- `plot_transform_curves(nodes, fitted, pix_col, ...)` — all fitted `h` over one
  pixel's q–q scatter, plus the correction applied
- `plot_moments_maps(...)` — skewness and kurtosis maps
Imports `_grid_from_points`, `_symmetric_limit`, `_save`, `_BIAS_CMAP` from
`qm_visualization` and `apply_map_formatting` from `visualization`. Never
duplicate those.

### `src/qm_visualization.py`
Quantile-mapping figures. Kept separate from `visualization.py` (already ~2,750
lines) but **imports its helpers** (`apply_map_formatting`, `_temp_formatter`,
`_geographic_aspect`, `_overlay_coarse_grid`) so all project figure standards
carry over. Never duplicate those helpers here.
- `PREDICTOR_LABELS`, `SCHEME_LABELS` — display names for all figures and the dashboard
- `plot_quantile_scatter(df, q, title, colour_by, max_points, ax, save_path)` — X vs Y, 1:1 + OLS
- `plot_percentile_panels(df, percentiles, suptitle, ...)` — one panel per percentile
- `plot_bias_map(bias_df, era5_lats, era5_lons, ...)` — per-pixel bias map
- `plot_bias_map_grid(pixel_df, ..., predictors, scheme, percentile, ...)` — shared colour scale
- `plot_metric_heatmap(metrics_df, metric, ...)` — predictor × (scheme, percentile)
- `plot_metric_by_window_length(metrics_df, metric, percentile, ...)` — log x-axis in days
- `plot_bias_vs_elevation(elev_df, scheme, percentile, ...)` — ±1 SD band
- `plot_bias_by_sea_fraction(sea_df, scheme, percentile, ...)`
- `plot_bias_by_window(window_df, scheme, percentile, ...)` — bias over time
- `plot_predictor_climatology_comparison(clim_fields, era5_clim, ...)` — observed + differences

### `src/vis_constants.py`
Central style constants — **always edit this file** to change any plot aesthetic.
Key values: `TITLE_FONT_SIZE=15`, `LABEL_FONT_SIZE=13`, `TICK_FONT_SIZE=11`,
`LEGEND_FONT_SIZE=11`, `CI_ALPHA=0.27`, `DEFAULT_CI_PCT=90`,
`COARSE_GRID_COLOR="#333333"`, `FIG_DPI=150`, `SAVE_DPI=200`.

### `src/visualization.py`
- `apply_map_formatting(ax, region)` — degree tick labels + aspect ratio
- `make_spatial_figure(ncols, region)` → (fig, axes)
- `_overlay_coarse_grid(ax, coarse_lats, coarse_lons)` — private helper; draws CMIP6 cell boundaries
- `_draw_highlight_box(ax, lat_min, lat_max, lon_min, lon_max, label)` — private helper; draws a red bold rectangle to highlight a sub-region
- `plot_temperature_map(..., show_coarse_grid, coarse_lats, coarse_lons)` — single-panel map
- `plot_side_by_side_maps(..., show_coarse_grid, coarse_lats, coarse_lons, highlight_box)` — two-panel comparison; coarse grid on ERA5 (left) panel only; `highlight_box=dict(lat_min, lat_max, lon_min, lon_max, label)` draws a red box on both panels
- `plot_seasonal_comparison_maps(..., show_coarse_grid, highlight_box)` — **2 rows × 5 cols** layout: [ERA5_s1, CMIP_s1, narrow_sep, ERA5_s2, CMIP_s2]; row 0 = DJF+MAM, row 1 = JJA+SON; horizontal colorbar at bottom. `highlight_box` draws on all 8 panels, label on DJF-CMIP panel only
- `plot_seasonal_maps(...)` — 4-panel DJF/MAM/JJA/SON (single dataset)
- `plot_missing_fraction_map(...)`, `plot_missing_fraction_timeseries(...)`
- `plot_land_sea_mask(land_mask_2d, lats, lons, region, save_path)` — binary two-colour land/sea map (fig01b); land=#a1d99b, ocean=#6baed6, legend not colourbar
- `plot_domain_timeseries(..., fill_dict)` — domain-mean time series; `fill_dict` keys match `series_dict`, values are `(lower, upper)` pd.Series tuples for shaded CI bands; legend entry auto-added when `fill_dict` provided
- `plot_monthly_climatology(..., confidence_pct)` — climatology with **inter-annual** 90% CI (t·SE across 10 years per calendar month); legend entry explains the band
- `plot_combined_temperature_distribution(era5_flat, cmip_flat, save_path)` — unified 2-panel figure: (a) KDE density curves, (b) boxplots; replaces the old separate `plot_temperature_percentiles` + `plot_temperature_distributions` calls
- `plot_temperature_distributions(...)` — legacy vertical boxplot (kept but not called from notebook)
- `plot_pixel_assignment_map(...)`, `plot_pixels_per_cell_heatmap(...)`
- `plot_scatter_regression(...)` — hexbin + OLS two panels
- `plot_residual_analysis(...)` — homoscedasticity + distance diagnostics
- `compute_ols_residuals(paired_df)` → DataFrame with `cmip_lat`, `cmip_lon`, `residual` (all 28M pixel×day residuals)
- `plot_residuals_by_sea_fraction(resid_df, cell_stats_df, n_bins, save_path)` — box plots by 5 sea-fraction bins (fig11)
- `plot_quarterly_warming_trend(..., show_trend_band)` — trend lines per quarter; `show_trend_band=False` for fig10 (only 10 annual points, band not meaningful)

**`SKIP_HEAVY` flag** (set in the Environment Setup cell): when `True`, skips fig07 (scatter regression on 28M rows) and fig08 (residual diagnostics). Set to `False` for a full notebook run. Keep `True` during restructuring/editing runs to save time.

---

## Quantile-Mapping Diagnostic Grid (the earlier predictor comparison)

**Superseded as the baseline model** by the fitted transforms described in the
next section, but still live: it produces the percentile tables,
`static_pixels.parquet`, and the dashboard's inputs. Do not delete it.

Ronit's note *On the baseline model* (16 Aug 2026) argues the daily-OLS models
in notebook 02 are the wrong frame: there is no day-to-day correspondence
between a CMIP6 date and an observed date, so a Perfect-Prognosis regression
cannot be fitted on daily GCM output. The MOS route — empirical quantile
mapping — compares **distributions** instead. Notebook 03 builds the diagnostic
layer that EQM needs; it does **not** fit or apply a correction.

### The hyper-parameter grid — 5 x 4 x 5 = 100 combinations

| Axis | Values |
|---|---|
| Predictor | `knn4`, `knn9`, `bilinear`, `trilinear_fit`, `trilinear_fixed` |
| Distribution window | `14d` (260), `month` (120), `quarter` (40), `year` (10) |
| Percentile | 5, 25, 50, 75, 90 |

Windows are non-overlapping and calendar aligned. Land pixels only (~7,700).

### Running it

```bash
python3.10 scripts/run_quantile_mapping.py            # full grid, idempotent
python3.10 scripts/run_quantile_mapping.py --force    # rebuild every cache
python3.10 scripts/run_quantile_mapping.py --aggregates-only
python3.10 -m streamlit run dashboards/qm_dashboard.py  # explore the results
```

First run takes ~23 min; later runs skip any percentile table already on disk.

Use **`python3.10`**, not `python3`. Plain `python3` resolves to
`/opt/homebrew/bin/python3`, which has **no xarray**, so every script fails with
`ModuleNotFoundError`. `python3.10` has xarray 2025.6.1, scipy 1.15.3, joblib and
pandas; it is also what the `python` alias points to.

Use `python3.10 -m streamlit`, not the bare `streamlit` command. The `streamlit`
script on this machine's PATH belongs to an anaconda env that does not have
`xarray` installed, so `streamlit run` fails with `ModuleNotFoundError`.

The dashboard uses a radio selector, not `st.tabs`. Streamlit executes every
`st.tabs` body on every rerun, which built all five plot groups at once and
locked up the browser.

### Cache layout under `data/cache/`

| Path | Content |
|---|---|
| `tas_bilinear_era5grid.nc` | bilinear predictor (shared with notebook 02) |
| `transforms/qq_pred_{method}.parquet` | per (pixel, season, fold): `y_p*`, `x_p*`, `yhat_p*` |
| `transforms/qq_fit_diag_{method}.parquet` | per-fit diagnostics and fitted parameters |
| `transforms/qq_hyper_{method}.parquet` | inner-CV scores per candidate |
| `transforms/qq_per_pixel_error.parquet` | per (method, season, percentile, pixel) |
| `transforms/qq_metrics_summary.parquet` | the method comparison, 160 rows |
| `transforms/qq_obs_floor.parquet` | the two climatology floors |
| `transforms/qq_daily_moments.parquet` | skewness/kurtosis, the normality check |
| `transforms/qq_manifest.json` | run configuration and timings |
| `pct_bilinear_season.parquet` | percentile table on the DJF/MAM/JJA/SON scheme |
| `predictors/knn{4,9}_era5grid.nc` | k-NN predictor fields |
| `predictors/trilinear_{fit,fixed}_era5grid.nc` | trilinear predictor fields |
| `elevation/elevation_era5grid.nc` | ETOPO DEM on the ERA5-Land grid |
| `qm/pct_{predictor}_{scheme}.parquet` | 20 wide percentile tables |
| `qm/metrics_summary.parquet` | one row per combination |
| `qm/per_pixel_bias.parquet`, `per_window_bias.parquet` | spatial and temporal aggregates |
| `qm/elev_bias.parquet`, `sea_bias.parquet` | binned diagnostics |
| `qm/static_pixels.parquet` | per-pixel `elevation`, `dz`, `sea_fraction`, parent cell |
| `qm/lapse_rates.parquet` | fitted Γ per season |

The notebook and the dashboard read the **aggregates**, not the large percentile
tables. Reach for a `pct_*.parquet` only when a single window slice is needed,
and pass `columns=` to `qm_metrics.load_pct_table`.

### Conventions specific to this baseline

- `bias = x − y` (predictor minus observation). Positive = predictor too warm.
- `Δz` is the sub-grid terrain: fine elevation minus the bilinearly
  interpolated CMIP6 cell-mean orography. The CMIP6 file has no orography
  variable, so it is derived from the ETOPO DEM.
- Fitted Γ comes out around **−3 °C km⁻¹**, roughly half the free-air
  environmental rate. This is expected for a near-surface lapse rate fitted on
  a residual, not a bug.
- A 14-day window holds 14 values, so its P5 and P90 are noisy. Always state
  this caveat when reporting short-window results.

---

## Fitted Quantile-Mapping Transforms (the current baseline)

`notebooks/03_quantile_mapping.ipynb` and `scripts/run_qm_transforms.py`. This
**superseded** the diagnostic grid below as the baseline model, but did not
replace it: the older script still produces the percentile tables, the
`static_pixels.parquet` this analysis reuses, and the dashboard's inputs.

### The design, all of it locked

| Axis | Choice |
|---|---|
| Predictor | Bilinear only, from the validated cache |
| Distribution window | **Meteorological season DJF/MAM/JJA/SON** — the `"season"` scheme |
| Model unit | One transform `h` per (land pixel, season), fitted on pooled daily values |
| Percentile | An **evaluation** dimension, never a model dimension |
| Cross-validation | Leave-one-season-year-out (DJF 9 folds, others 10) |
| Hyper-parameters | One value **per season**, pooled over a 400-pixel sample — never per pixel |

The seven transforms are `raw` (no correction), `normal`, `linear`, `poly`,
`quant`, `rquant`, `ssplin`, registered in `qm_transforms.TRANSFORMS`.
Gudmundsson's Eq. 7 was implemented and then removed: it finished last on every
metric and cost 42 min against the linear fit's 29 s.

### Running it

```bash
python3.10 scripts/run_qm_transforms.py                                  # full, ~45 min, idempotent
python3.10 scripts/run_qm_transforms.py --pixel-stride 10 --methods quant linear
python3.10 scripts/run_qm_transforms.py --aggregates-only
python3.10 scripts/sanity_qm_transforms.py                               # after ANY transform change
```

Outputs land in `data/cache/qm/transforms/`, plus
`data/cache/qm/pct_bilinear_season.parquet`. `qq_manifest.json` records the
configuration and the driver refuses to mix a `--pixel-stride` run with a full
one.

### Conventions and traps specific to this layer

- **Two season conventions now exist.** `make_windows("quarter")` is
  **JFM/AMJ/JAS/OND**; `make_windows("season")` is **DJF/MAM/JJA/SON** grouped by
  season-year with December rolled forward. `data_io.seasonal_split` and
  `predictors._SEASON_OF_MONTH` are two further DJF copies. New code uses only
  the `"season"` scheme. `SCHEMES` deliberately excludes it so the old pipeline
  is untouched; `WINDOW_SCHEMES` includes it.
- **`make_windows("season")` can return `-1`** group codes, for days in an
  incomplete season. No other scheme does.
- **Latitude runs descending** (38 → 24). Never assume ascending.
- **The cached bilinear field is already in °C.** Do not call `to_celsius` on it.
  `qm_inputs.load_bilinear_predictor` validates the grid and dates before use.
- **`bias = prediction − observation`** here. Note `qm_metrics.combination_metrics`
  takes `(y, x)` positionally — always call it with **keyword** arguments, since
  swapping them flips `bias` silently while leaving `mae`/`rmse`/`r2` unchanged.
- **The correction is applied to daily values, then percentiles are taken** —
  not the reverse. Mapping a percentile directly is only equal to mapping the
  days when quantiles are plain order statistics; under the interpolating
  estimator this project uses, the two differ by up to **0.39 °C** for QUANT.
- **There are two climatology floors, not one.** Compare MAE against `obs_mad`
  (mean absolute deviation) and RMSE against `obs_sd`. For a normal sample the
  former is ~0.8 of the latter, so using the SD for MAE understates the error by
  a fifth and can make a method look as if it beat an unreachable floor.
- **Monotonicity is a correctness property**, not a nicety: it is what makes the
  percentile-level evaluation well defined. `predict_percentiles` enforces it and
  `n_nonmono` records where the repair fired.
- **QUANT's node grid must stay at 11 or above.** A table of *n* nodes reaches
  only to probability `0.5/n`, so 4 nodes cannot represent P5 and the boundary
  rule invents it. The unanchored version clamped flat there and carried a
  **+1.8 °C bias at SON P5** — worse than no correction — while scoring a
  *better* MAE, because clamping acts as accidental shrinkage toward
  climatology. Check `bias` alongside `mae` when changing this.
- **The fitting sample is set per family, matching the `qmap` package.**
  `QQTransform.fit_on_raw` is `True` for `normal`/`linear`/`poly`, which use
  every training day (`qstep = NULL` in `qmap`), and `False` for
  `quant`/`rquant`/`ssplin`, which use 99 nodes (`qstep = 0.01`, the package
  default for exactly those three). `fold_nodes(..., n_nodes=None)` returns the
  raw sorted pairs. Refitting the node-based methods on raw days changes MAE by
  at most 0.003 °C and makes the spline ~5x slower (13 min → 100 min for the
  full run), so the split is both faithful and cheap. `select_hyper` must use
  the same resolution as the outer fit — it routes on `fit_on_raw` too.

---

## Scientific Formatting Requirements

These are non-negotiable for all figures produced in this project:

1. **Longitude axis:** labels formatted as `30°E`, `32°E`, … (use `visualization.apply_map_formatting`)
2. **Latitude axis:** labels formatted as `24°N`, `26°N`, … (same function)
3. **Geographic aspect ratio:** corrected for latitude convergence at the domain's mean latitude (~31°N); computed as `1 / cos(lat_mid_radians)`. Use `make_spatial_figure()` or `apply_map_formatting()`.
4. **Colorbars:** always labelled with variable name and units; ticks formatted as `20°`, `25°` via `_temp_formatter`.
5. **Integer counts:** heatmaps of pixel counts must use `fmt='d'` (no decimals); fill NaN → 0 and cast to int before `sns.heatmap`.
6. **Colormaps:** `RdYlBu_r` for absolute temperature, `RdBu_r` for anomalies, `YlOrRd` for missing fraction, `YlGnBu` for count data.
7. **Figure captions:** written in Markdown cells immediately below each figure cell in the notebook. Format: "**Fig. N.** Short title. Panel (a): description. Data: source, period."
8. **Temperature axis ticks:** use `_temp_formatter` (defined in `visualization.py`) on all temperature axes — format `20°`, `30°`, never `20°C` repeated per tick. Apply via `ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))`.
9. **Seasonal maps:** use `plot_seasonal_comparison_maps(...)` — **2 rows × 5 cols** layout (DJF+MAM in row 0, JJA+SON in row 1; col 2 is a narrow separator; horizontal colorbar in row 3). Do **not** use two separate `plot_seasonal_maps` calls and do **not** revert to the old 4×2 layout.
10. **Style constants:** all font sizes, CI alpha, and grid colours live in `src/vis_constants.py`. Never hardcode these values in `visualization.py` or the notebook.
11. **Highlight boxes:** the region 32–34°N, 34–36°E is highlighted on Fig 2 and Fig 4 with `highlight_box=dict(lat_min=32, lat_max=34, lon_min=34, lon_max=36, label=...)`. This marks the area of largest ERA5–CMIP6 bias where cold terrain is undersampled by the coarse grid.

See `guidelines/analysis_guidelines.md` for full details.

---

## Key Data Conventions

| Convention | Detail |
|---|---|
| Temperature units | Always convert K → °C before analysis: `arr - 273.15` |
| Calendar alignment | Remove ERA5-Land leap days (1992-02-29, 1996-02-29); 3,650 shared days |
| CMIP6 time decoding | Use `t.strftime('%Y-%m-%d')` — `pd.to_datetime` may fail on cftime objects |
| Land mask | 32.7% of domain pixels are ocean (NaN); pattern is spatially fixed |
| Spatial mean | Always use `skipna=True` for ERA5-Land domain averages |
| CMIP6 longitude | May be stored as [0°, 360°]; use `standardize_longitude()` |
| CMIP6 grid padding | Use `pad_lat=1.0, pad_lon=1.5` when subsetting to ensure border cells are included |
| Pearson r (cell-level) | CMIP6 TAS vs ERA5-Land T2M ≈ 0.9 (not 1.0) |
| Systematic bias | CMIP6 domain mean ~1–2°C warmer than ERA5-Land |

---

## Data Location (Remote HPC)

Raw data is fetched from the Moriah cluster at HUJI via SSH jump host
`bava.cs.huji.ac.il`. Use `scripts/fetch_remote_climate_data.sh`.

Key paths on Moriah:
- ERA5-Land annual files: `/sci/labs/efratmorin/ronit/`
- CMIP6 historical (Anton): `/sci/labs/efratmorin/anton.gelman/work/`
- ERA5 (Andre): `/sci/labs/assafhochman/andre.klif/data/hist/ERA5/`

---

## Allowed Shell Commands (permissions granted)

- `python3.10` (or the `python` alias) — run analysis scripts. **Not** `python3`, which lacks xarray.
- `jupyter nbconvert` / `jupyter execute` — execute notebooks
- `pip install` — install packages
- Standard file operations within the project directory

---

## Contacts

- **Ronit Nirel** — project lead, downloaded ERA5-Land data (March 2026)
- **Dorita Morin** — supervisor; proposed global mean TAS as a predictor
- **Efrat Morin** — research group PI
- **Anton Gelman** — post-doc, CMIP6 data on Moriah, no-leap calendar notes
- **Andre Klif** — provides list of 6 GCMs and ERA5 data paths
- **Chaim** — ensemble member selection guidance
