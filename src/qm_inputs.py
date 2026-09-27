"""
qm_inputs.py — Shared loading of the aligned datasets and the bilinear predictor.

Both quantile-mapping drivers need the same starting point: ERA5-Land and CMIP6
loaded over the same region box, aligned onto a common no-leap calendar,
converted to °C, and a land mask derived.  Holding that in one place means the
region box and the leap-day handling cannot drift apart between the two
scripts.

Extracted from ``scripts/run_quantile_mapping.py::load_inputs`` without
behaviour change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from . import data_io as dio
from . import spatial_ops as sops
from .interpolation import interpolate_cmip_to_era5

__all__ = ["REGION", "load_aligned_inputs", "load_bilinear_predictor",
           "BILINEAR_CACHE_NAME"]

log = logging.getLogger(__name__)

#: The EMME study domain.
REGION = dict(south=24, north=38, west=30, east=38)

#: Filename of the cached bilinear field under ``data/cache/``.
BILINEAR_CACHE_NAME = "tas_bilinear_era5grid.nc"


def load_aligned_inputs(data_dir: str | Path, region: dict | None = None) -> dict:
    """Load, align and convert both datasets, and derive the land mask.

    Returns
    -------
    dict
        Keys ``era5_temp`` and ``cmip_tas`` (both °C, loaded into memory),
        ``shared_dates`` (list of ``"YYYY-MM-DD"``), ``era5_lats``/``era5_lons``,
        ``cmip_lats``/``cmip_lons``, and ``land_mask_2d``.

    Notes
    -----
    ``era5_lats`` runs **descending** (38 -> 24) on this grid, and
    ``land_mask_2d`` follows the same order.  Nothing downstream may assume
    ascending latitude.
    """
    data_dir = Path(data_dir)
    region = region or REGION

    era5_files = sorted(data_dir.glob("t2m_ERA5land_daily_*.nc"))
    if not era5_files:
        raise FileNotFoundError(f"No ERA5-Land files found in {data_dir}")
    try:
        cmip_file = next(iter(sorted(data_dir.glob("tas_day_*.nc"))))
    except StopIteration:
        raise FileNotFoundError(f"No CMIP6 tas_day_*.nc file found in {data_dir}") from None

    log.info("Loading %d ERA5-Land file(s) …", len(era5_files))
    era5_ds = dio.load_era5_land(era5_files, region=region)
    log.info("Loading CMIP6 %s …", cmip_file.name)
    cmip_ds = dio.load_cmip6(cmip_file, region=region, pad_lat=1.0, pad_lon=1.5)

    era5_ds, cmip_ds, shared_dates = dio.align_calendars(era5_ds, cmip_ds)
    era5_temp = dio.to_celsius(era5_ds["t2m"]).load()
    cmip_tas = dio.to_celsius(cmip_ds["tas"]).load()
    log.info(
        "Aligned on %d shared day(s) (%s … %s).",
        len(shared_dates), shared_dates[0], shared_dates[-1],
    )

    lat_name = "lat" if "lat" in cmip_tas.dims else "latitude"
    lon_name = "lon" if "lon" in cmip_tas.dims else "longitude"
    land_mask_2d = sops.compute_land_mask(era5_temp)
    n_land = int(land_mask_2d.sum())
    log.info(
        "Grid %dx%d; %d land pixels (%.1f%%).",
        land_mask_2d.shape[0], land_mask_2d.shape[1], n_land,
        100 * n_land / land_mask_2d.size,
    )

    return dict(
        era5_temp=era5_temp, cmip_tas=cmip_tas, shared_dates=shared_dates,
        era5_lats=era5_temp.latitude.values, era5_lons=era5_temp.longitude.values,
        cmip_lats=cmip_tas[lat_name].values, cmip_lons=cmip_tas[lon_name].values,
        land_mask_2d=land_mask_2d,
    )


def load_bilinear_predictor(
    cache_dir: str | Path,
    cmip_tas: xr.DataArray,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    shared_dates: list[str],
) -> xr.DataArray:
    """The bilinearly regridded CMIP6 field, read from cache when possible.

    The cache is validated before use, not merely opened.  A stale file whose
    time axis is offset by a day would misalign every q–q pair with no symptom
    beyond slightly worse skill, so the grid and the dates are both checked
    against the freshly aligned inputs and a mismatch rebuilds the field.

    The cached values are **already in °C**; do not pass the result through
    :func:`src.data_io.to_celsius`.
    """
    cache_nc = Path(cache_dir) / BILINEAR_CACHE_NAME

    if cache_nc.exists():
        ds = xr.open_dataset(cache_nc)
        da = ds[list(ds.data_vars)[0]]
        problems = []
        if not np.allclose(da.latitude.values, era5_lats):
            problems.append("latitude axis")
        if not np.allclose(da.longitude.values, era5_lons):
            problems.append("longitude axis")
        cached_dates = [str(t)[:10] for t in da.time.values]
        if cached_dates != list(shared_dates):
            problems.append(
                f"time axis ({len(cached_dates)} vs {len(shared_dates)} days)"
            )
        if not problems:
            log.info("Using cached bilinear predictor %s.", cache_nc.name)
            return da.load()
        log.warning(
            "Cached bilinear predictor %s does not match the inputs (%s); "
            "rebuilding.", cache_nc.name, ", ".join(problems),
        )
        ds.close()

    log.info("Regridding CMIP6 onto the ERA5-Land grid …")
    return interpolate_cmip_to_era5(
        cmip_tas, era5_lats, era5_lons, cache_nc=cache_nc
    ).load()
