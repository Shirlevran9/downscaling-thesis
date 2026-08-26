"""
predictors.py — Coarse-to-fine predictor fields for the quantile-mapping baseline.

Every predictor is returned on the ERA5-Land 0.1° grid with dimensions
``(time, latitude, longitude)``, so the observed field Y and the predictor
field X share a pixel and can be percentiled the same way.

Predictors
----------
``knn4``, ``knn9``
    Mean TAS of the *k* nearest CMIP6 cell centres (unweighted).  ``k=9``
    reproduces the 3x3 regional average used by Dorita and Anton.
``bilinear``
    Bilinear remapping of TAS to the fine grid (delegates to
    :mod:`src.interpolation`).
``trilinear_fit``, ``trilinear_fixed``
    ``bilinear(TAS) + gamma * dz``, where ``dz`` is the sub-grid terrain the
    coarse model cannot see: fine-grid elevation minus the coarse model
    orography interpolated back to the fine grid.  ``trilinear_fit`` uses a
    lapse rate fitted per season, ``trilinear_fixed`` the standard
    -6.5 °C km-1.

Typical usage
-------------
from src.predictors import build_all_predictors

fields, meta = build_all_predictors(
    cmip_tas=cmip_tas, era5_temp=era5_temp,
    land_mask_2d=land_mask_2d, region=REGION,
    cache_dir=DATA_DIR / "cache",
)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import spatial_ops as sops
from .elevation import get_domain_elevation
from .interpolation import interpolate_cmip_to_era5, remapbil_scipy

__all__ = [
    "PREDICTOR_NAMES",
    "FIXED_LAPSE_RATE",
    "knn_indices",
    "knn_predictor",
    "coarse_orography_on_fine_grid",
    "fit_lapse_rate",
    "trilinear_predictor",
    "build_all_predictors",
    "load_cached_predictors",
]

log = logging.getLogger(__name__)

#: Canonical predictor order used by every downstream table and figure.
PREDICTOR_NAMES = ["knn4", "knn9", "bilinear", "trilinear_fit", "trilinear_fixed"]

#: Standard environmental lapse rate, °C per metre (negative = cooling upward).
FIXED_LAPSE_RATE = -0.0065

_SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


# ---------------------------------------------------------------------------
# k-nearest-neighbour predictor
# ---------------------------------------------------------------------------

def knn_indices(
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    cmip_lats: np.ndarray,
    cmip_lons: np.ndarray,
    k: int,
) -> np.ndarray:
    """Find the *k* nearest CMIP6 cell centres for every ERA5-Land pixel.

    Distance is Euclidean in degrees after scaling longitude by
    ``cos(latitude)``, so a degree of longitude is weighted by its true
    ground distance at the domain's mean latitude.  A true 2-D neighbour
    search is required here; ``spatial_ops.assign_era5_to_cmip_cells``
    performs a separable per-axis argmin and only supports ``k = 1``.

    Parameters
    ----------
    era5_lats, era5_lons : np.ndarray
        1-D ascending ERA5-Land coordinate centres.
    cmip_lats, cmip_lons : np.ndarray
        1-D CMIP6 coordinate centres.
    k : int
        Number of neighbours (4 or 9 in this project).

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_era5_lat * n_era5_lon, k)`` holding flat
        indices into the raveled CMIP6 ``(lat, lon)`` grid.  Row order is
        C-order over the ERA5-Land grid.
    """
    from scipy.spatial import cKDTree  # local import

    era5_lats = np.asarray(era5_lats, dtype=float)
    era5_lons = np.asarray(era5_lons, dtype=float)
    cmip_lats = np.asarray(cmip_lats, dtype=float)
    cmip_lons = np.asarray(cmip_lons, dtype=float)

    # Longitude scaling factor at the domain's mean latitude
    lat_mid = 0.5 * (era5_lats.min() + era5_lats.max())
    lon_scale = float(np.cos(np.radians(lat_mid)))

    c_lat_g, c_lon_g = np.meshgrid(cmip_lats, cmip_lons, indexing="ij")
    tree = cKDTree(
        np.column_stack([c_lat_g.ravel(), c_lon_g.ravel() * lon_scale])
    )

    e_lat_g, e_lon_g = np.meshgrid(era5_lats, era5_lons, indexing="ij")
    query = np.column_stack([e_lat_g.ravel(), e_lon_g.ravel() * lon_scale])

    n_cells = c_lat_g.size
    if k > n_cells:
        raise ValueError(f"k={k} exceeds the {n_cells} available CMIP6 cells.")

    _, idx = tree.query(query, k=k)
    return np.atleast_2d(idx.astype(np.int32))


def knn_predictor(
    cmip_da: xr.DataArray,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    k: int,
    cache_nc: str | Path | None = None,
    time_chunk: int = 365,
) -> xr.DataArray:
    """Build the mean-of-*k*-nearest-cells predictor field on the fine grid.

    The value at a fine pixel is the unweighted mean of TAS over the *k*
    CMIP6 cells whose centres are closest to that pixel.  Because the
    neighbour set is fixed in time, the indices are computed once and the
    gather is applied time step by time step in chunks.

    Parameters
    ----------
    cmip_da : xr.DataArray
        CMIP6 TAS in °C, dims ``(time, lat, lon)``, already padded to cover
        the target domain.
    era5_lats, era5_lons : np.ndarray
        ERA5-Land coordinate centres.
    k : int
        Number of neighbours.
    cache_nc : str, Path, or None
        If given, load from here when present and save here otherwise.
    time_chunk : int
        Number of time steps loaded into memory at once.

    Returns
    -------
    xr.DataArray
        Named ``knn{k}``, dims ``(time, latitude, longitude)``, dtype float32.
    """
    name = f"knn{k}"

    if cache_nc is not None:
        cache_nc = Path(cache_nc)
        if cache_nc.exists():
            log.info("Loading cached %s from %s", name, cache_nc)
            return xr.open_dataset(cache_nc)[name]

    lat_name = "lat" if "lat" in cmip_da.dims else "latitude"
    lon_name = "lon" if "lon" in cmip_da.dims else "longitude"
    cmip_lats = cmip_da[lat_name].values
    cmip_lons = cmip_da[lon_name].values

    idx = knn_indices(era5_lats, era5_lons, cmip_lats, cmip_lons, k)  # (n_pix, k)

    n_time = cmip_da.sizes["time"]
    n_lat, n_lon = len(era5_lats), len(era5_lons)
    out = np.empty((n_time, n_lat, n_lon), dtype=np.float32)

    for start in range(0, n_time, time_chunk):
        stop = min(start + time_chunk, n_time)
        block = cmip_da.isel(time=slice(start, stop)).values  # (t, n_clat, n_clon)
        flat = block.reshape(block.shape[0], -1)              # (t, n_cells)
        # Gather the k neighbours for every pixel, then average over them.
        gathered = flat[:, idx]                               # (t, n_pix, k)
        out[start:stop] = (
            gathered.mean(axis=2).reshape(-1, n_lat, n_lon).astype(np.float32)
        )

    da = xr.DataArray(
        out,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": cmip_da.time.values,
            "latitude": era5_lats,
            "longitude": era5_lons,
        },
        name=name,
        attrs={
            "units": cmip_da.attrs.get("units", "degC"),
            "long_name": f"CMIP6 TAS averaged over the {k} nearest coarse cells",
            "k_neighbours": k,
        },
    )

    if cache_nc is not None:
        cache_nc.parent.mkdir(parents=True, exist_ok=True)
        da.to_dataset(name=name).to_netcdf(cache_nc)
        log.info("Saved %s to %s", name, cache_nc)

    return da


# ---------------------------------------------------------------------------
# Sub-grid terrain (dz) for the trilinear predictors
# ---------------------------------------------------------------------------

def coarse_orography_on_fine_grid(
    elev_da: xr.DataArray,
    cmip_lats: np.ndarray,
    cmip_lons: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build the coarse model orography and bring it back to the fine grid.

    The CMIP6 file carries no orography variable, so the coarse model's
    effective terrain height is approximated by averaging the fine ETOPO DEM
    over the pixels of each coarse cell.  All pixels are used, land and sea
    alike, because the coarse cell average is what the model resolves.  The
    coarse field is then bilinearly interpolated back to the fine grid using
    the same routine as the TAS regridding, so ``dz`` is consistent with the
    bilinear predictor.

    Parameters
    ----------
    elev_da : xr.DataArray
        Fine-grid elevation (m a.s.l.), dims ``(latitude, longitude)``.
    cmip_lats, cmip_lons : np.ndarray
        CMIP6 coordinate centres.

    Returns
    -------
    (coarse_da, dz_da) : tuple of xr.DataArray
        ``coarse_da`` is the cell-mean orography on the coarse grid, dims
        ``(lat, lon)``.  ``dz_da`` is ``elev_fine - bilinear(coarse_da)`` on
        the fine grid, dims ``(latitude, longitude)``, in metres.  Positive
        ``dz`` marks terrain the coarse grid smooths away.
    """
    fine_lats = elev_da.latitude.values
    fine_lons = elev_da.longitude.values

    # Every fine pixel takes part, so pass an all-land mask.
    all_pixels = np.ones((len(fine_lats), len(fine_lons)), dtype=bool)
    assign = sops.assign_era5_to_cmip_cells(
        fine_lats, fine_lons, cmip_lats, cmip_lons, all_pixels
    )
    assign["elevation"] = elev_da.values.ravel(order="C")

    cell_mean = assign.groupby(["cmip_lat", "cmip_lon"])["elevation"].mean()

    coarse = np.full((len(cmip_lats), len(cmip_lons)), np.nan, dtype=float)
    lat_pos = {round(float(v), 6): i for i, v in enumerate(cmip_lats)}
    lon_pos = {round(float(v), 6): j for j, v in enumerate(cmip_lons)}
    for (clat, clon), val in cell_mean.items():
        i = lat_pos.get(round(float(clat), 6))
        j = lon_pos.get(round(float(clon), 6))
        if i is not None and j is not None:
            coarse[i, j] = val

    # Coarse cells on the padded border receive no fine pixels.  Fill them
    # from the nearest valid cell so the bilinear step has no holes to
    # propagate into the interior.  pandas fills along both axes without
    # pulling in xarray's optional bottleneck dependency.
    filled = pd.DataFrame(coarse)
    filled = filled.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
    coarse = filled.to_numpy()

    coarse_da = xr.DataArray(
        coarse,
        dims=("lat", "lon"),
        coords={"lat": cmip_lats, "lon": cmip_lons},
        name="orography",
    )
    coarse_da.attrs.update(
        units="m a.s.l.",
        long_name="CMIP6 cell-mean orography from the ETOPO DEM",
    )

    coarse_on_fine = remapbil_scipy(coarse_da, fine_lats, fine_lons)

    dz_da = xr.DataArray(
        (elev_da.values - coarse_on_fine).astype(np.float32),
        dims=("latitude", "longitude"),
        coords={"latitude": fine_lats, "longitude": fine_lons},
        name="dz",
        attrs={
            "units": "m",
            "long_name": (
                "Sub-grid terrain height: fine elevation minus bilinearly "
                "interpolated coarse orography"
            ),
        },
    )
    return coarse_da, dz_da


def fit_lapse_rate(
    era5_temp: xr.DataArray,
    bilin_da: xr.DataArray,
    dz_da: xr.DataArray,
    land_mask_2d: np.ndarray,
    dates: list[str],
) -> dict:
    """Fit the lapse rate that maps sub-grid terrain to a temperature offset.

    The residual of the bilinear predictor, ``t2m - bilinear``, is regressed
    on ``dz`` by ordinary least squares.  One slope is fitted per
    meteorological season plus one over all days, because the near-surface
    lapse rate over this domain is stronger in summer than in winter.

    Only land pixels enter the fit, matching the rest of the analysis.

    Parameters
    ----------
    era5_temp : xr.DataArray
        Observed ERA5-Land T2M in °C, dims ``(time, latitude, longitude)``.
    bilin_da : xr.DataArray
        Bilinear predictor on the same grid and time axis.
    dz_da : xr.DataArray
        Sub-grid terrain, dims ``(latitude, longitude)``.
    land_mask_2d : np.ndarray
        2-D boolean land mask.
    dates : list of str
        ``"YYYY-MM-DD"`` strings, one per time step, in time order.

    Returns
    -------
    dict
        ``{"all": gamma, "DJF": gamma, "MAM": ..., "JJA": ..., "SON": ...}``
        with gamma in °C per metre, plus ``"_n"`` giving the sample size used
        for each key under the ``"n_<key>"`` convention.
    """
    months = pd.to_datetime(pd.Index(dates)).month.to_numpy()
    seasons = np.array([_SEASON_OF_MONTH[m] for m in months])

    obs = era5_temp.values
    pred = bilin_da.values
    dz = dz_da.values[land_mask_2d]                       # (n_land,)

    resid = (obs - pred)[:, land_mask_2d]                 # (n_time, n_land)

    out: dict = {}

    def _slope(mask_time: np.ndarray) -> tuple[float, int]:
        y = resid[mask_time].ravel()
        x = np.tile(dz, int(mask_time.sum()))
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 100:
            return float("nan"), int(good.sum())
        # Fit with dz in km, then rescale: dz in metres spans ~10^3 while the
        # intercept is order 1, which makes the design matrix ill-conditioned.
        # The least-squares solution is unchanged by the rescaling.
        slope_per_km = float(np.polyfit(x[good] / 1000.0, y[good], 1)[0])
        return slope_per_km / 1000.0, int(good.sum())

    gamma, n = _slope(np.ones(len(dates), dtype=bool))
    out["all"], out["n_all"] = gamma, n

    for season in ("DJF", "MAM", "JJA", "SON"):
        gamma, n = _slope(seasons == season)
        out[season], out[f"n_{season}"] = gamma, n

    return out


def trilinear_predictor(
    bilin_da: xr.DataArray,
    dz_da: xr.DataArray,
    gamma: float | dict,
    dates: list[str] | None = None,
    name: str = "trilinear",
    cache_nc: str | Path | None = None,
) -> xr.DataArray:
    """Add a terrain-height correction to the bilinear predictor.

    ``X = bilinear(TAS) + gamma * dz``.  This is the "trilinear" predictor:
    bilinear in longitude and latitude, linear in elevation.

    Parameters
    ----------
    bilin_da : xr.DataArray
        Bilinear predictor, dims ``(time, latitude, longitude)``.
    dz_da : xr.DataArray
        Sub-grid terrain, dims ``(latitude, longitude)``.
    gamma : float or dict
        A single lapse rate in °C m-1, or the dict returned by
        :func:`fit_lapse_rate`.  When a dict is given, the season-specific
        slope is applied to each day and *dates* is required.
    dates : list of str, optional
        ``"YYYY-MM-DD"`` strings, one per time step.  Required when *gamma*
        is a dict.
    name : str
        Variable name for the returned DataArray.
    cache_nc : str, Path, or None
        Optional NetCDF cache path.

    Returns
    -------
    xr.DataArray
        Dims ``(time, latitude, longitude)``, dtype float32.
    """
    if cache_nc is not None:
        cache_nc = Path(cache_nc)
        if cache_nc.exists():
            log.info("Loading cached %s from %s", name, cache_nc)
            return xr.open_dataset(cache_nc)[name]

    dz = dz_da.values.astype(np.float32)

    if isinstance(gamma, dict):
        if dates is None:
            raise ValueError("dates is required when gamma is a per-season dict.")
        months = pd.to_datetime(pd.Index(dates)).month.to_numpy()
        per_day = np.array(
            [gamma[_SEASON_OF_MONTH[m]] for m in months], dtype=np.float32
        )
        offset = per_day[:, None, None] * dz[None, :, :]
        gamma_attr = "per-season fitted"
    else:
        offset = np.float32(gamma) * dz[None, :, :]
        gamma_attr = f"{gamma:.6g} degC/m"

    values = (bilin_da.values.astype(np.float32) + offset).astype(np.float32)

    da = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": bilin_da.time.values,
            "latitude": bilin_da.latitude.values,
            "longitude": bilin_da.longitude.values,
        },
        name=name,
        attrs={
            "units": "degC",
            "long_name": "Bilinear CMIP6 TAS plus sub-grid terrain correction",
            "lapse_rate": gamma_attr,
        },
    )

    if cache_nc is not None:
        cache_nc.parent.mkdir(parents=True, exist_ok=True)
        da.to_dataset(name=name).to_netcdf(cache_nc)
        log.info("Saved %s to %s", name, cache_nc)

    return da


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_all_predictors(
    cmip_tas: xr.DataArray,
    era5_temp: xr.DataArray,
    land_mask_2d: np.ndarray,
    region: dict,
    dates: list[str],
    cache_dir: str | Path,
    force: bool = False,
) -> tuple[dict[str, xr.DataArray], dict]:
    """Build every predictor field listed in :data:`PREDICTOR_NAMES`.

    Each field is cached as a NetCDF file under
    ``<cache_dir>/predictors/``.  The bilinear field reuses the existing
    project cache at ``<cache_dir>/tas_bilinear_era5grid.nc``.

    Parameters
    ----------
    cmip_tas : xr.DataArray
        Calendar-aligned CMIP6 TAS in °C, dims ``(time, lat, lon)``.
    era5_temp : xr.DataArray
        Calendar-aligned ERA5-Land T2M in °C, dims ``(time, latitude, longitude)``.
    land_mask_2d : np.ndarray
        2-D boolean land mask on the ERA5-Land grid.
    region : dict
        Domain box with keys ``south``, ``north``, ``west``, ``east``.
    dates : list of str
        Shared ``"YYYY-MM-DD"`` dates, in time order.
    cache_dir : str or Path
        Root cache directory (``data/cache``).
    force : bool
        Rebuild the k-NN and trilinear caches even if the files exist.

    Returns
    -------
    (fields, meta) : tuple
        ``fields`` maps predictor name to DataArray.  ``meta`` carries
        ``lapse_rates`` (the :func:`fit_lapse_rate` dict), ``dz`` (the
        sub-grid terrain DataArray), ``elevation`` (fine-grid DEM) and
        ``coarse_orography``.
    """
    cache_dir = Path(cache_dir)
    pred_dir = cache_dir / "predictors"
    pred_dir.mkdir(parents=True, exist_ok=True)

    era5_lats = era5_temp.latitude.values
    era5_lons = era5_temp.longitude.values
    lat_name = "lat" if "lat" in cmip_tas.dims else "latitude"
    lon_name = "lon" if "lon" in cmip_tas.dims else "longitude"
    cmip_lats = cmip_tas[lat_name].values
    cmip_lons = cmip_tas[lon_name].values

    if force:
        for path in pred_dir.glob("*.nc"):
            path.unlink()

    fields: dict[str, xr.DataArray] = {}

    # --- k-nearest-neighbour fields -----------------------------------------
    for k in (4, 9):
        print(f"  · building knn{k} …", flush=True)
        fields[f"knn{k}"] = knn_predictor(
            cmip_tas, era5_lats, era5_lons, k,
            cache_nc=pred_dir / f"knn{k}_era5grid.nc",
        )

    # --- bilinear (reuses the existing project cache) -----------------------
    print("  · building bilinear …", flush=True)
    bilin = interpolate_cmip_to_era5(
        cmip_tas, era5_lats, era5_lons,
        cache_nc=cache_dir / "tas_bilinear_era5grid.nc",
    )
    bilin = bilin.rename("bilinear")
    fields["bilinear"] = bilin

    # --- sub-grid terrain ---------------------------------------------------
    print("  · building sub-grid terrain (dz) …", flush=True)
    elev_da = get_domain_elevation(
        region=region,
        target_lats=era5_lats,
        target_lons=era5_lons,
        cache_dir=cache_dir / "elevation",
    )
    coarse_oro, dz_da = coarse_orography_on_fine_grid(
        elev_da, cmip_lats, cmip_lons
    )

    print("  · fitting lapse rates …", flush=True)
    lapse = fit_lapse_rate(era5_temp, bilin, dz_da, land_mask_2d, dates)
    for key in ("all", "DJF", "MAM", "JJA", "SON"):
        print(f"      gamma[{key}] = {1000 * lapse[key]:+.2f} degC/km", flush=True)

    # --- trilinear fields ---------------------------------------------------
    print("  · building trilinear_fit …", flush=True)
    fields["trilinear_fit"] = trilinear_predictor(
        bilin, dz_da, lapse, dates=dates, name="trilinear_fit",
        cache_nc=pred_dir / "trilinear_fit_era5grid.nc",
    )
    print("  · building trilinear_fixed …", flush=True)
    fields["trilinear_fixed"] = trilinear_predictor(
        bilin, dz_da, FIXED_LAPSE_RATE, name="trilinear_fixed",
        cache_nc=pred_dir / "trilinear_fixed_era5grid.nc",
    )

    meta = {
        "lapse_rates": lapse,
        "dz": dz_da,
        "elevation": elev_da,
        "coarse_orography": coarse_oro,
    }
    return {n: fields[n] for n in PREDICTOR_NAMES}, meta


def load_cached_predictors(
    cache_dir: str | Path,
    names: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """Load predictor fields straight from the NetCDF caches.

    Read-only counterpart to :func:`build_all_predictors`, for the notebook
    and any other consumer that only needs the fields and must not refit the
    lapse rate or rebuild a cache.

    Parameters
    ----------
    cache_dir : str or Path
        Root cache directory (``data/cache``).
    names : list of str, optional
        Subset of :data:`PREDICTOR_NAMES`; defaults to all five.

    Returns
    -------
    dict of str to xr.DataArray
        Only the predictors whose cache file exists are returned.

    Raises
    ------
    FileNotFoundError
        If none of the requested caches exist.
    """
    cache_dir = Path(cache_dir)
    pred_dir = cache_dir / "predictors"
    names = list(names or PREDICTOR_NAMES)

    paths = {
        "knn4": pred_dir / "knn4_era5grid.nc",
        "knn9": pred_dir / "knn9_era5grid.nc",
        "bilinear": cache_dir / "tas_bilinear_era5grid.nc",
        "trilinear_fit": pred_dir / "trilinear_fit_era5grid.nc",
        "trilinear_fixed": pred_dir / "trilinear_fixed_era5grid.nc",
    }

    fields: dict[str, xr.DataArray] = {}
    for name in names:
        path = paths[name]
        if not path.exists():
            log.warning("No cache for %s at %s — skipped", name, path)
            continue
        ds = xr.open_dataset(path)
        var = list(ds.data_vars)[0]
        fields[name] = ds[var].rename(name)

    if not fields:
        raise FileNotFoundError(
            f"No predictor caches found under {pred_dir}. "
            "Run scripts/run_quantile_mapping.py first."
        )
    return fields
