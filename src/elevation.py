"""
elevation.py — Terrain elevation data for the downscaling project.

Downloads terrain elevation as a single NetCDF subset for the study domain —
no tile counting, no authentication, no external CLI tools required.

Sources tried in order
----------------------
1. NOAA CoastWatch ERDDAP — ETOPO1 Ice Surface, 1 arc-minute  ← primary (confirmed working)
     dataset id: etopo180  (longitude range −180 to 180)
     https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180
     variable: altitude  (metres, positive up)

2. NOAA NCEI ERDDAP — ETOPO2022 Ice Surface, 60 arc-second
     dataset id: etopo2022Ice60s
     https://erddap.ncei.noaa.gov/erddap/griddap/etopo2022Ice60s
     variable: z

3. NOAA CoastWatch ERDDAP — ETOPO2022 Ice Surface, 60 arc-second (alt dataset IDs)
     etopo2022IceSurface60s / etopo2022IceSurfaceZPositive1arcminute

Note: The ETOPO1 dataset ``etopo1Ice`` at NOAA NCEI ERDDAP was retired in 2022.
The equivalent data are served by CoastWatch under the dataset id ``etopo180``.

Resolution
----------
1 arc-minute (~1.8 km at 31°N) — finer than the ERA5-Land 0.1° target grid,
so the regridding step averages ~36 source points per ERA5 cell.

Reference
---------
Amante, C. and B.W. Eakins, 2009. ETOPO1 1 Arc-Minute Global Relief Model.
  NOAA Technical Memorandum NESDIS NGDC-24.
  https://doi.org/10.7289/V5C8276M

Typical usage
-------------
from src.elevation import get_domain_elevation, add_elevation_to_df

elev_da = get_domain_elevation(
    region=REGION,
    target_lats=era5_temp.latitude.values,
    target_lons=era5_temp.longitude.values,
    cache_dir=DATA_DIR / "elevation",
)

paired_df_elev = add_elevation_to_df(paired_df_interp, elev_da)
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "fetch_etopo_elevation",
    "regrid_elevation",
    "get_domain_elevation",
    "add_elevation_to_df",
]

log = logging.getLogger(__name__)

_RAW_NC_FILENAME       = "etopo_domain.nc"
_REGRIDDED_NC_FILENAME = "elevation_era5grid.nc"

# ---------------------------------------------------------------------------
# URL templates — tried in order until one succeeds.
# Each entry: (label, url_template)
# url_template uses .format(south=…, north=…, west=…, east=…) substitution.
# Coordinates formatted as floats so ERDDAP resolves to nearest grid point.
# ---------------------------------------------------------------------------
_DOWNLOAD_SOURCES = [
    # 1. CoastWatch — ETOPO1 Ice Surface, 1 arc-minute  ← CONFIRMED WORKING
    #    variable: altitude, coords: latitude/longitude
    (
        "CoastWatch ERDDAP — etopo180 (ETOPO1 Ice, 1 arc-min)",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc"
        "?altitude[({south:.4f}):1:({north:.4f})][({west:.4f}):1:({east:.4f})]",
    ),
    # 2. NOAA NCEI — ETOPO2022, 60 arc-second  (variable: z)
    (
        "NOAA NCEI ERDDAP — etopo2022Ice60s (ETOPO2022, 1 arc-min)",
        "https://erddap.ncei.noaa.gov/erddap/griddap/etopo2022Ice60s.nc"
        "?z[({south:.4f}):1:({north:.4f})][({west:.4f}):1:({east:.4f})]",
    ),
    # 3. CoastWatch — ETOPO2022, alternate dataset IDs  (variable: z)
    (
        "CoastWatch ERDDAP — etopo2022IceSurface60s",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo2022IceSurface60s.nc"
        "?z[({south:.4f}):1:({north:.4f})][({west:.4f}):1:({east:.4f})]",
    ),
    (
        "CoastWatch ERDDAP — etopo2022IceSurfaceZPositive1arcminute",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
        "etopo2022IceSurfaceZPositive1arcminute.nc"
        "?z[({south:.4f}):1:({north:.4f})][({west:.4f}):1:({east:.4f})]",
    ),
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def fetch_etopo_elevation(
    region: dict,
    cache_path: str | Path,
) -> Path:
    """Download terrain elevation for the study domain.

    Tries NOAA ERDDAP ETOPO2022 endpoints in order, then falls back to the
    GMRT REST API.  Makes a single HTTP request that returns a pre-subsetted
    NetCDF for *region* — no tiles, no authentication.  If *cache_path*
    already exists the download is skipped.

    Parameters
    ----------
    region : dict
        Bounding box ``{"south", "north", "west", "east"}`` in decimal degrees.
    cache_path : str or Path
        Destination NetCDF file path (``*.nc``).

    Returns
    -------
    Path
        Path to the downloaded (or pre-existing) NetCDF.

    Raises
    ------
    RuntimeError
        If all endpoints fail.
    """
    import urllib.request  # stdlib — no extra deps

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        log.info("Elevation cache hit: %s", cache_path)
        return cache_path

    fmt = dict(
        south=float(region["south"]),
        north=float(region["north"]),
        west=float(region["west"]),
        east=float(region["east"]),
    )

    tmp_path: Path | None = None
    errors: list[str] = []

    for label, url_template in _DOWNLOAD_SOURCES:
        url = url_template.format(**fmt)
        log.info("Trying elevation source: %s\n  %s", label, url)
        print(f"Trying {label} …")
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent, suffix=".nc", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
            urllib.request.urlretrieve(url, tmp_path)

            # Verify the file is a valid NetCDF (not an HTML error page)
            try:
                _ds = xr.open_dataset(tmp_path)
                _ds.close()
            except Exception:
                raise ValueError("Downloaded file is not a valid NetCDF")

            tmp_path.rename(cache_path)
            log.info("Elevation download complete → %s  (source: %s)", cache_path, label)
            print(f"  ✓ Saved to {cache_path}  [{label}]")
            return cache_path

        except Exception as exc:  # noqa: BLE001
            err_msg = f"{label}: {exc}"
            log.warning("  ✗ Failed: %s", err_msg)
            print(f"  ✗ Failed: {exc}")
            errors.append(err_msg)
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            tmp_path = None

    raise RuntimeError(
        "All elevation endpoints failed.\n"
        + "\n".join(f"  • {e}" for e in errors)
        + "\n\nManual download option:"
        "\n  ETOPO1 (CoastWatch ERDDAP form):"
        "\n  https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.html"
    )


# ---------------------------------------------------------------------------
# Regrid
# ---------------------------------------------------------------------------

def regrid_elevation(
    raw_nc_path: str | Path,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> xr.DataArray:
    """Load elevation NetCDF and bilinearly resample to the ERA5-Land 0.1° grid.

    Handles coordinate and variable naming conventions from multiple sources:

    * ETOPO2022 (ERDDAP): coordinates ``latitude``/``longitude``, variable ``z``
    * GMRT: coordinates ``lat``/``lon``, variable ``altitude``

    Values are in metres (positive up for ice surface / land).

    Parameters
    ----------
    raw_nc_path : str or Path
        NetCDF file produced by :func:`fetch_etopo_elevation`.
    target_lats : np.ndarray
        1-D ascending array of ERA5-Land latitude centres (degrees north).
    target_lons : np.ndarray
        1-D ascending array of ERA5-Land longitude centres (degrees east).

    Returns
    -------
    xr.DataArray
        2-D DataArray, dims ``(latitude, longitude)``, matching the ERA5-Land
        grid, with ``units`` attribute ``"m a.s.l."``.
    """
    target_lats = np.asarray(target_lats, dtype=float)
    target_lons = np.asarray(target_lons, dtype=float)

    log.info("Loading elevation from %s …", raw_nc_path)
    ds = xr.open_dataset(raw_nc_path)

    # Normalise coordinate names → latitude / longitude
    rename = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    # GMRT uses 'x' / 'y' in some versions
    if "y" in ds.dims and "latitude" not in ds.dims:
        rename["y"] = "latitude"
    if "x" in ds.dims and "longitude" not in ds.dims:
        rename["x"] = "longitude"
    if rename:
        ds = ds.rename(rename)

    # Pick elevation variable: prefer 'z', then 'altitude', then first available
    if "z" in ds.data_vars:
        var = "z"
    elif "altitude" in ds.data_vars:
        var = "altitude"
    else:
        var = list(ds.data_vars)[0]
        log.warning("Unknown variable name in elevation NetCDF; using '%s'", var)

    dem = ds[var].squeeze()

    # Ensure ascending latitude for interp
    if dem.latitude.values[0] > dem.latitude.values[-1]:
        dem = dem.isel(latitude=slice(None, None, -1))

    log.info(
        "Resampling elevation %dx%d → %dx%d …",
        dem.sizes["longitude"],
        dem.sizes["latitude"],
        len(target_lons),
        len(target_lats),
    )
    dem_regridded = dem.interp(
        latitude=target_lats,
        longitude=target_lons,
        method="linear",
    )

    dem_regridded.attrs["units"]     = "m a.s.l."
    dem_regridded.attrs["long_name"] = "Terrain elevation (ETOPO1 Ice Surface, NOAA NCEI)"
    dem_regridded.attrs["source"]    = (
        "ETOPO1 Ice Surface, 1 arc-min, NOAA CoastWatch ERDDAP (etopo180)"
    )
    dem_regridded.name = "elevation"
    return dem_regridded


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def get_domain_elevation(
    region: dict,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    cache_dir: str | Path,
) -> xr.DataArray:
    """Download, regrid, and cache terrain elevation for the study domain.

    Two-level cache:
      1. Raw download → ``<cache_dir>/etopo_domain.nc``
      2. Regridded ERA5-Land grid → ``<cache_dir>/elevation_era5grid.nc``

    On subsequent calls the regridded NetCDF is loaded directly.

    Parameters
    ----------
    region : dict
        Bounding box ``{"south", "north", "west", "east"}``.
    target_lats : np.ndarray
        ERA5-Land latitude centres.
    target_lons : np.ndarray
        ERA5-Land longitude centres.
    cache_dir : str or Path
        Directory for cached files.

    Returns
    -------
    xr.DataArray
        Elevation at each ERA5-Land grid point, dims ``(latitude, longitude)``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    nc_cache = cache_dir / _REGRIDDED_NC_FILENAME

    # Fast path: regridded NetCDF already present
    if nc_cache.exists():
        log.info("Loading regridded elevation from cache: %s", nc_cache)
        ds = xr.open_dataset(nc_cache)
        return ds["elevation"]

    # Download raw elevation subset
    raw_nc = fetch_etopo_elevation(region, cache_dir / _RAW_NC_FILENAME)

    # Regrid to ERA5-Land grid
    elev_da = regrid_elevation(raw_nc, target_lats, target_lons)

    # Persist regridded result
    elev_da.to_dataset(name="elevation").to_netcdf(nc_cache)
    log.info("Regridded elevation saved to %s", nc_cache)
    print(f"  ✓ Regridded elevation saved to {nc_cache}")

    return elev_da


# ---------------------------------------------------------------------------
# Merge into paired DataFrame
# ---------------------------------------------------------------------------

def add_elevation_to_df(
    df: pd.DataFrame,
    elev_da: xr.DataArray,
) -> pd.DataFrame:
    """Add an ``elevation`` column (m a.s.l.) to a paired pixel × day DataFrame.

    Elevation is a static property of each ERA5-Land pixel, so the lookup is
    performed once per unique (lat, lon) pair and broadcast to all rows.

    Parameters
    ----------
    df : pd.DataFrame
        Paired DataFrame with columns ``era5_lat`` and ``era5_lon``.
    elev_da : xr.DataArray
        Regridded elevation, dims ``(latitude, longitude)``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new ``elevation`` column (float, metres).
    """
    elev_vals = elev_da.values        # (n_lat, n_lon)
    e_lats    = elev_da.latitude.values
    e_lons    = elev_da.longitude.values

    lat_idx_map = {round(float(v), 6): i for i, v in enumerate(e_lats)}
    lon_idx_map = {round(float(v), 6): i for i, v in enumerate(e_lons)}

    unique_pixels = df[["era5_lat", "era5_lon"]].drop_duplicates()

    elev_col: dict = {}
    for _, row in unique_pixels.iterrows():
        lat_k = round(float(row["era5_lat"]), 6)
        lon_k = round(float(row["era5_lon"]), 6)
        i = lat_idx_map.get(lat_k)
        j = lon_idx_map.get(lon_k)
        elev_col[(lat_k, lon_k)] = (
            float(elev_vals[i, j]) if (i is not None and j is not None) else np.nan
        )

    df = df.copy()
    keys = list(zip(df["era5_lat"].round(6), df["era5_lon"].round(6)))
    df["elevation"] = [elev_col.get(k, np.nan) for k in keys]
    return df
