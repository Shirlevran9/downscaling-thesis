"""
data_io.py — Data loading and preprocessing for the downscaling project.

Provides functions to load ERA5-Land and CMIP6 NetCDF files, harmonise
calendars, convert units, and build the paired pixel × day DataFrame used
by downstream analyses and models.

Usage example
-------------
from src.data_io import load_era5_land, load_cmip6, align_calendars, build_paired_dataframe
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "open_dataset",
    "load_era5_land",
    "load_cmip6",
    "to_celsius",
    "align_calendars",
    "build_paired_dataframe",
    "seasonal_split",
    "compute_global_daily_mean",
]


# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------

def open_dataset(path: str | Path) -> xr.Dataset:
    """Open a NetCDF file, trying the netCDF4 engine before h5netcdf.

    Parameters
    ----------
    path : str or Path
        Path to the NetCDF file.

    Returns
    -------
    xr.Dataset
    """
    path = Path(path)
    try:
        return xr.open_dataset(path, engine="netCDF4")
    except Exception:
        return xr.open_dataset(path, engine="h5netcdf")


def _prepare_era5_land_file(path: Path) -> xr.Dataset:
    """Standardise a single ERA5-Land annual file.

    ERA5-Land post-processed daily statistics files may use ``valid_time``
    instead of ``time``, and carry auxiliary scalar coordinates (``number``,
    ``expver``) that must be dropped before concatenation.

    Parameters
    ----------
    path : Path
        Path to a single ERA5-Land annual NetCDF file.

    Returns
    -------
    xr.Dataset
        Dataset containing only ``t2m`` with a ``time`` dimension.
    """
    ds = open_dataset(path)
    rename_map = {}
    if "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if rename_map:
        ds = ds.rename(rename_map)
    drop_vars = [v for v in ("number", "expver") if v in ds.coords or v in ds.variables]
    if drop_vars:
        ds = ds.drop_vars(drop_vars)
    return ds[["t2m"]]


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_era5_land(
    files: list[Path],
    region: dict | None = None,
    pad_lat: float = 0.0,
    pad_lon: float = 0.0,
) -> xr.Dataset:
    """Load and concatenate ERA5-Land annual files into a single Dataset.

    Parameters
    ----------
    files : list of Path
        Sorted list of annual ERA5-Land NetCDF files
        (e.g. ``t2m_ERA5land_daily_1990.nc``, …, ``t2m_ERA5land_daily_1999.nc``).
    region : dict, optional
        Bounding box ``{"south": ..., "north": ..., "west": ..., "east": ...}``
        in decimal degrees.  If provided, the dataset is spatially subsetted
        after loading.
    pad_lat : float
        Latitude padding (degrees) added symmetrically around the region.
    pad_lon : float
        Longitude padding (degrees) added symmetrically around the region.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``time``, ``latitude``, ``longitude``
        and variable ``t2m`` (still in Kelvin; call :func:`to_celsius`
        afterwards).
    """
    from src.spatial_ops import subset_box  # avoid circular at module level

    parts = [_prepare_era5_land_file(p) for p in sorted(files)]
    ds = xr.concat(parts, dim="time").sortby("time")

    if region is not None:
        ds = subset_box(ds, "latitude", "longitude", region, pad_lat=pad_lat, pad_lon=pad_lon)

    return ds


def load_cmip6(
    file: str | Path,
    region: dict | None = None,
    pad_lat: float = 1.0,
    pad_lon: float = 1.5,
    var: str = "tas",
) -> xr.Dataset:
    """Load a CMIP6 NetCDF file and optionally subset it to a region.

    CMIP6 files may store longitude in the range [0°, 360°]; this function
    normalises coordinates to [-180°, 180°] before subsetting.

    Parameters
    ----------
    file : str or Path
        Path to the CMIP6 NetCDF file.
    region : dict, optional
        Bounding box ``{"south": ..., "north": ..., "west": ..., "east": ...}``.
    pad_lat : float
        Latitude padding in degrees (default 1.0°; ensures boundary cells
        are fully covered).
    pad_lon : float
        Longitude padding in degrees (default 1.5°; accounts for coarse
        CMIP6 grid spacing of ~1.25°).
    var : str
        Variable name to retain (default ``"tas"``).

    Returns
    -------
    xr.Dataset
        Dataset with variables at least containing *var*.
    """
    from src.spatial_ops import standardize_longitude, subset_box

    ds = open_dataset(file)
    ds = standardize_longitude(ds, "lon")

    if region is not None:
        ds = subset_box(ds, "lat", "lon", region, pad_lat=pad_lat, pad_lon=pad_lon)

    return ds


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def to_celsius(arr: xr.DataArray) -> xr.DataArray:
    """Convert a temperature DataArray from Kelvin to degrees Celsius.

    The conversion is applied only when the ``units`` attribute indicates
    Kelvin, or when the domain mean exceeds 150 (a heuristic guard).

    Parameters
    ----------
    arr : xr.DataArray
        Temperature array, potentially in Kelvin.

    Returns
    -------
    xr.DataArray
        Temperature in degrees Celsius, with ``units`` attribute updated.
    """
    units = str(arr.attrs.get("units", "")).lower().strip()
    needs_conversion = units in {"k", "kelvin"} or float(arr.mean().values) > 150
    if needs_conversion:
        out = arr - 273.15
        out.attrs = {**arr.attrs, "units": "°C"}
        return out
    return arr


# ---------------------------------------------------------------------------
# Calendar alignment
# ---------------------------------------------------------------------------

def align_calendars(
    era5_ds: xr.Dataset,
    cmip_ds: xr.Dataset,
    era5_time_name: str = "time",
    cmip_time_name: str = "time",
) -> tuple[xr.Dataset, xr.Dataset, list[str]]:
    """Align ERA5-Land (Gregorian) and CMIP6 (no-leap) calendars.

    ERA5-Land follows the Gregorian calendar and therefore includes 29
    February in leap years.  CMIP6 uses a no-leap calendar (365 days per
    year).  To obtain a common time axis, leap days are removed from the
    ERA5-Land dataset and the intersection of date strings is computed.

    Parameters
    ----------
    era5_ds : xr.Dataset
        ERA5-Land dataset with a Gregorian-calendar time coordinate.
    cmip_ds : xr.Dataset
        CMIP6 dataset with a no-leap time coordinate.
    era5_time_name : str
        Name of the time dimension in *era5_ds*.
    cmip_time_name : str
        Name of the time dimension in *cmip_ds*.

    Returns
    -------
    era5_aligned : xr.Dataset
        ERA5-Land dataset restricted to shared dates (no leap days).
    cmip_aligned : xr.Dataset
        CMIP6 dataset restricted to shared dates.
    shared_dates : list of str
        Sorted list of shared date strings in ``"YYYY-MM-DD"`` format.
    """
    era5_times = pd.DatetimeIndex(
        pd.to_datetime(
            pd.Series(era5_ds[era5_time_name].values).astype(str),
            errors="coerce",
        )
    )
    era5_date_strs = np.array(
        [t.strftime("%Y-%m-%d") if not pd.isnull(t) else "" for t in era5_times]
    )

    # Decode CMIP6 cftime objects to date strings
    cmip_raw_times = cmip_ds[cmip_time_name].values
    try:
        cmip_date_strs = np.array([t.strftime("%Y-%m-%d") for t in cmip_raw_times])
    except AttributeError:
        cmip_date_strs = pd.DatetimeIndex(
            pd.to_datetime(
                pd.Series(cmip_raw_times).astype(str), errors="coerce"
            )
        ).strftime("%Y-%m-%d").values

    shared_dates = sorted(set(era5_date_strs) & set(cmip_date_strs) - {""})

    era5_mask = np.isin(era5_date_strs, shared_dates)
    cmip_mask = np.isin(cmip_date_strs, shared_dates)

    era5_aligned = era5_ds.isel({era5_time_name: era5_mask})
    cmip_aligned = cmip_ds.isel({cmip_time_name: cmip_mask})

    return era5_aligned, cmip_aligned, shared_dates


# ---------------------------------------------------------------------------
# Paired pixel × day DataFrame
# ---------------------------------------------------------------------------

def build_paired_dataframe(
    era5_temp: xr.DataArray,
    cmip_tas: xr.DataArray,
    assignment_df: pd.DataFrame,
    land_mask_2d: np.ndarray,
    shared_dates: list[str],
) -> pd.DataFrame:
    """Build a long-format paired pixel × day DataFrame.

    Each row represents one ERA5-Land land pixel on one day, paired with the
    temperature of the CMIP6 grid cell to which that pixel was assigned via
    nearest-neighbour matching.

    Parameters
    ----------
    era5_temp : xr.DataArray
        ERA5-Land 2 m temperature, already aligned to *shared_dates*,
        dimensions ``(time, latitude, longitude)``, in °C.
    cmip_tas : xr.DataArray
        CMIP6 near-surface air temperature, already aligned to *shared_dates*,
        dimensions ``(time, lat, lon)``, in °C.
    assignment_df : pd.DataFrame
        Output of :func:`src.spatial_ops.assign_era5_to_cmip_cells`, containing
        columns ``era5_lat``, ``era5_lon``, ``cmip_lat``, ``cmip_lon``.
    land_mask_2d : np.ndarray
        2D boolean array (True = land), shape ``(n_lat, n_lon)`` matching the
        ERA5-Land spatial grid.
    shared_dates : list of str
        Sorted list of ``"YYYY-MM-DD"`` date strings.

    Returns
    -------
    pd.DataFrame
        Columns: ``era5_lat``, ``era5_lon``, ``cmip_lat``, ``cmip_lon``,
        ``day`` (datetime), ``t2m`` (ERA5-Land °C), ``tas`` (CMIP6 °C).
        Rows with missing ERA5-Land values (ocean pixels) are dropped.
    """
    era5_lats = era5_temp.latitude.values
    era5_lons = era5_temp.longitude.values
    cmip_lats = cmip_tas.lat.values
    cmip_lons = cmip_tas.lon.values

    # Build index arrays for CMIP6 remapping onto ERA5 grid
    era5_to_cmip_lat_idx = np.searchsorted(cmip_lats, assignment_df.groupby("era5_lat").first()["cmip_lat"])
    # Recompute nearest-neighbour indices via broadcasting
    lat_dist = np.abs(era5_lats[:, None] - cmip_lats[None, :])
    lon_dist = np.abs(era5_lons[:, None] - cmip_lons[None, :])
    cmip_lat_idx = np.argmin(lat_dist, axis=1)  # shape (n_era5_lat,)
    cmip_lon_idx = np.argmin(lon_dist, axis=1)  # shape (n_era5_lon,)

    cmip_lat_idx_2d = np.tile(cmip_lat_idx[:, None], (1, len(era5_lons)))
    cmip_lon_idx_2d = np.tile(cmip_lon_idx[None, :], (len(era5_lats), 1))

    # Remap CMIP6 onto ERA5-Land spatial grid
    cmip_on_era5 = cmip_tas.values[:, cmip_lat_idx_2d, cmip_lon_idx_2d]  # (n_time, n_lat, n_lon)

    # Assigned CMIP6 cell coordinates for each ERA5 pixel
    cmip_lat_grid = cmip_lats[cmip_lat_idx_2d]
    cmip_lon_grid = cmip_lons[cmip_lon_idx_2d]

    # Extract land pixel indices
    land_rows, land_cols = np.where(land_mask_2d)
    n_land = len(land_rows)
    n_time = len(shared_dates)

    era5_vals = era5_temp.values[:, land_rows, land_cols]   # (n_time, n_land)
    cmip_vals = cmip_on_era5[:, land_rows, land_cols]       # (n_time, n_land)

    dates = pd.to_datetime(shared_dates)

    df = pd.DataFrame(
        {
            "era5_lat": np.tile(era5_lats[land_rows], n_time),
            "era5_lon": np.tile(era5_lons[land_cols], n_time),
            "cmip_lat": np.tile(cmip_lat_grid[land_rows, land_cols], n_time),
            "cmip_lon": np.tile(cmip_lon_grid[land_rows, land_cols], n_time),
            "day": np.repeat(dates, n_land),
            "t2m": era5_vals.ravel(order="F"),
            "tas": cmip_vals.ravel(order="F"),
        }
    ).dropna(subset=["t2m"])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

_SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def seasonal_split(
    obj: xr.Dataset | xr.DataArray | pd.DataFrame,
    time_dim: str = "time",
    day_col: str = "day",
) -> dict[str, xr.Dataset | xr.DataArray | pd.DataFrame]:
    """Split a Dataset, DataArray, or DataFrame into the four meteorological seasons.

    Parameters
    ----------
    obj : xr.Dataset, xr.DataArray, or pd.DataFrame
    time_dim : str
        Name of the time coordinate (for xarray objects).
    day_col : str
        Name of the date column (for DataFrames).

    Returns
    -------
    dict
        Keys ``"DJF"``, ``"MAM"``, ``"JJA"``, ``"SON"``; values are subsets
        of *obj* for each season.
    """
    result = {}
    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        months = obj[time_dim].dt.month
        for season, m_list in _SEASON_MONTHS.items():
            result[season] = obj.sel({time_dim: months.isin(m_list)})
    elif isinstance(obj, pd.DataFrame):
        col = day_col if day_col in obj.columns else time_dim
        months = pd.to_datetime(obj[col]).dt.month
        for season, m_list in _SEASON_MONTHS.items():
            result[season] = obj[months.isin(m_list)].copy()
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")
    return result


def compute_global_daily_mean(
    cmip_ds: xr.Dataset,
    var: str = "tas",
    lat_name: str = "lat",
    lon_name: str = "lon",
    time_name: str = "time",
) -> pd.Series:
    """Compute the area-weighted global daily mean temperature from a CMIP6 Dataset.

    This predictor feature was proposed by Dorita Morin as a proxy for the
    large-scale temperature state on each day.  The weights are proportional
    to the cosine of latitude.

    Parameters
    ----------
    cmip_ds : xr.Dataset
        Global (or at least large-domain) CMIP6 dataset.
    var : str
        Variable name (default ``"tas"``).
    lat_name, lon_name, time_name : str
        Coordinate names.

    Returns
    -------
    pd.Series
        Daily global mean temperature in the same units as *var*,
        indexed by date string ``"YYYY-MM-DD"``.
    """
    arr = cmip_ds[var]
    lats = arr[lat_name].values
    weights = np.cos(np.radians(lats))
    weights_da = xr.DataArray(weights, coords={lat_name: lats}, dims=[lat_name])

    weighted = arr.weighted(weights_da)
    global_mean = weighted.mean(dim=[lat_name, lon_name])

    raw_times = cmip_ds[time_name].values
    try:
        date_strs = [t.strftime("%Y-%m-%d") for t in raw_times]
    except AttributeError:
        date_strs = pd.to_datetime(
            pd.Series(raw_times).astype(str), errors="coerce"
        ).dt.strftime("%Y-%m-%d").tolist()

    return pd.Series(global_mean.values, index=date_strs, name="global_mean_tas")
