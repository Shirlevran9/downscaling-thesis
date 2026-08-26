"""
interpolation.py — Bilinear spatial interpolation for CMIP6 → ERA5-Land regridding.

Provides CDO-based bilinear remapping (primary) with a pure-Python scipy fallback
so the pipeline runs even without a CDO installation.

Typical usage
-------------
from src.interpolation import interpolate_cmip_to_era5

tas_interp = interpolate_cmip_to_era5(
    cmip_tas,
    target_lats=era5_temp.latitude.values,
    target_lons=era5_temp.longitude.values,
    cache_nc=DATA_DIR / "tas_bilinear_era5grid.nc",
)
"""

import logging
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

__all__ = [
    "write_domain_grid_file",
    "remapbil_cdo",
    "remapbil_scipy",
    "interpolate_cmip_to_era5",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CDO grid-description helper
# ---------------------------------------------------------------------------

def write_domain_grid_file(
    lats: np.ndarray,
    lons: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Write a CDO-format lonlat grid description file.

    The grid parameters are inferred from the supplied coordinate arrays so
    that the output file exactly matches the ERA5-Land domain used in the
    project.  Do *not* hard-code coordinates here; pass the actual DataArray
    coordinate values.

    Parameters
    ----------
    lats : np.ndarray
        1-D array of latitude centres (ascending, degrees north).
    lons : np.ndarray
        1-D array of longitude centres (ascending, degrees east).
    output_path : str or Path
        Destination file path (will be overwritten if it exists).

    Returns
    -------
    Path
        Absolute path to the written grid file.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    ysize = len(lats)
    xsize = len(lons)
    yfirst = float(lats[0])
    xfirst = float(lons[0])

    # Infer grid spacing (round to 6 dp to avoid floating-point noise)
    yinc = round(float(np.median(np.diff(lats))), 6) if ysize > 1 else 0.1
    xinc = round(float(np.median(np.diff(lons))), 6) if xsize > 1 else 0.1

    content = (
        f"gridtype  = lonlat\n"
        f"xsize     = {xsize}\n"
        f"ysize     = {ysize}\n"
        f"xname     = lon\n"
        f"xunits    = degrees_east\n"
        f"yname     = lat\n"
        f"yunits    = degrees_north\n"
        f"xfirst    = {xfirst}\n"
        f"xinc      = {xinc}\n"
        f"yfirst    = {yfirst}\n"
        f"yinc      = {yinc}\n"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    log.debug("CDO grid file written to %s (%dx%d)", output_path, xsize, ysize)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# CDO wrapper
# ---------------------------------------------------------------------------

def remapbil_cdo(
    input_nc: str | Path,
    grid_file: str | Path,
    output_nc: str | Path,
    cdo_bin: str = "cdo",
) -> bool:
    """Run CDO bilinear remapping from *input_nc* onto the grid in *grid_file*.

    Uses ``cdo remapbil,<grid_file> <input_nc> <output_nc>``.

    Parameters
    ----------
    input_nc : str or Path
        Source NetCDF file (CMIP6 TAS on the coarse grid).
    grid_file : str or Path
        CDO grid description file produced by :func:`write_domain_grid_file`.
    output_nc : str or Path
        Destination path for the regridded NetCDF.
    cdo_bin : str
        Name or full path of the CDO executable (default ``"cdo"``).

    Returns
    -------
    bool
        ``True`` if CDO completed successfully, ``False`` otherwise (binary
        not found, non-zero exit code, or any other exception).
    """
    # Check CDO is available before attempting the call
    if shutil.which(cdo_bin) is None:
        log.warning("CDO binary '%s' not found on PATH; falling back to scipy.", cdo_bin)
        return False

    cmd = [
        cdo_bin,
        f"remapbil,{grid_file}",
        str(input_nc),
        str(output_nc),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("CDO subprocess failed (%s); falling back to scipy.", exc)
        return False

    if result.returncode != 0:
        log.warning(
            "CDO returned exit code %d.\nstderr: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return False

    log.info("CDO remapbil completed → %s", output_nc)
    return True


# ---------------------------------------------------------------------------
# Pure-Python scipy fallback
# ---------------------------------------------------------------------------

def remapbil_scipy(
    cmip_da: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation using ``scipy.interpolate.RegularGridInterpolator``.

    This function replicates the behaviour of ``cdo remapbil`` without
    requiring CDO to be installed.  It performs bilinear (linear in both lat
    and lon) interpolation from the CMIP6 coarse grid to the ERA5-Land fine
    grid, one time step at a time to keep peak memory low.

    Parameters
    ----------
    cmip_da : xr.DataArray
        CMIP6 temperature DataArray with dimensions ``(time, lat, lon)``
        (or ``(lat, lon)`` for a single time step).
    target_lats : np.ndarray
        1-D ascending array of ERA5-Land latitude centres.
    target_lons : np.ndarray
        1-D ascending array of ERA5-Land longitude centres.

    Returns
    -------
    np.ndarray
        Interpolated values, shape ``(n_time, n_lat_target, n_lon_target)``
        (or ``(n_lat_target, n_lon_target)`` for a single time step).
    """
    from scipy.interpolate import RegularGridInterpolator  # local import

    target_lats = np.asarray(target_lats)
    target_lons = np.asarray(target_lons)

    # Determine source coordinate names
    lat_name = "lat" if "lat" in cmip_da.dims else "latitude"
    lon_name = "lon" if "lon" in cmip_da.dims else "longitude"

    src_lats = cmip_da[lat_name].values
    src_lons = cmip_da[lon_name].values

    # Ensure ascending latitude for RegularGridInterpolator
    lat_flip = src_lats[0] > src_lats[-1]
    if lat_flip:
        cmip_da = cmip_da.isel({lat_name: slice(None, None, -1)})
        src_lats = src_lats[::-1]

    # Build output mesh once
    mesh_lons, mesh_lats = np.meshgrid(target_lons, target_lats)  # (n_lat, n_lon)
    pts = np.stack([mesh_lats.ravel(), mesh_lons.ravel()], axis=1)

    single_step = cmip_da.ndim == 2

    if single_step:
        data = cmip_da.values
        interp = RegularGridInterpolator(
            (src_lats, src_lons),
            data,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate at boundary if needed
        )
        return interp(pts).reshape(len(target_lats), len(target_lons))

    n_time = cmip_da.sizes["time"]
    result = np.empty(
        (n_time, len(target_lats), len(target_lons)),
        dtype=cmip_da.dtype,
    )
    for i in range(n_time):
        data = cmip_da.isel(time=i).values
        interp = RegularGridInterpolator(
            (src_lats, src_lons),
            data,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        result[i] = interp(pts).reshape(len(target_lats), len(target_lons))

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def interpolate_cmip_to_era5(
    cmip_da: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    cache_nc: str | Path | None = None,
    cdo_bin: str = "cdo",
) -> xr.DataArray:
    """Bilinearly interpolate CMIP6 TAS from the coarse grid to the ERA5-Land grid.

    Execution order
    ---------------
    1. If *cache_nc* exists, load and return it immediately (no recomputation).
    2. Attempt CDO bilinear remapping via :func:`remapbil_cdo`.
    3. If CDO is unavailable or fails, fall back to :func:`remapbil_scipy`.
    4. Wrap the result in a properly labelled ``xr.DataArray`` and, if
       *cache_nc* is provided, persist it to disk.

    The returned DataArray uses coordinate names ``latitude`` and ``longitude``
    (matching the ERA5-Land convention) so it is drop-in compatible with
    ``build_interpolated_paired_df``.

    Parameters
    ----------
    cmip_da : xr.DataArray
        CMIP6 temperature DataArray with dimensions ``(time, lat, lon)``
        (or ``(lat, lon)``).  Must already be in °C and have the source
        grid padded to cover the full target domain.
    target_lats : np.ndarray
        1-D ascending array of ERA5-Land latitude centres.
    target_lons : np.ndarray
        1-D ascending array of ERA5-Land longitude centres.
    cache_nc : str, Path, or None
        If provided, save the result here as NetCDF the first time and
        reload it on subsequent calls.
    cdo_bin : str
        CDO executable name (default ``"cdo"``; override if installed
        in a non-standard location).

    Returns
    -------
    xr.DataArray
        Bilinearly interpolated TAS, dimensions:
        - ``(time, latitude, longitude)`` — multi-step input
        - ``(latitude, longitude)`` — single-step input
        Same dtype and units attribute as *cmip_da*.
    """
    target_lats = np.asarray(target_lats)
    target_lons = np.asarray(target_lons)

    # --- 1. Cache hit --------------------------------------------------------
    if cache_nc is not None:
        cache_nc = Path(cache_nc)
        if cache_nc.exists():
            log.info("Loading cached interpolation from %s", cache_nc)
            ds = xr.open_dataset(cache_nc)
            var = list(ds.data_vars)[0]
            return ds[var]

    # --- 2. Try CDO ----------------------------------------------------------
    used_cdo = False
    if shutil.which(cdo_bin) is not None:
        # Write CMIP6 DataArray to a temporary NetCDF so CDO can read it
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_in = Path(tmpdir) / "cmip_input.nc"
            tmp_grid = Path(tmpdir) / "era5_grid.txt"
            tmp_out = Path(tmpdir) / "cmip_remapped.nc"

            # Save CMIP6 data; use a simple dataset wrapper
            cmip_da.to_dataset(name="tas").to_netcdf(tmp_in)
            write_domain_grid_file(target_lats, target_lons, tmp_grid)

            success = remapbil_cdo(tmp_in, tmp_grid, tmp_out, cdo_bin=cdo_bin)
            if success and tmp_out.exists():
                ds_out = xr.open_dataset(tmp_out)
                var = list(ds_out.data_vars)[0]
                arr = ds_out[var].load()

                # Rename coords to ERA5 convention
                rename = {}
                for cname in arr.dims:
                    if cname in {"lat", "rlat"}:
                        rename[cname] = "latitude"
                    elif cname in {"lon", "rlon"}:
                        rename[cname] = "longitude"
                if rename:
                    arr = arr.rename(rename)

                # Assign target coordinate arrays (CDO may produce slightly
                # different float values due to rounding)
                coord_map = {}
                if "latitude" in arr.dims:
                    coord_map["latitude"] = target_lats
                if "longitude" in arr.dims:
                    coord_map["longitude"] = target_lons
                if coord_map:
                    arr = arr.assign_coords(coord_map)

                arr.attrs = cmip_da.attrs
                used_cdo = True
                # Fall through to save + return

    # --- 3. Scipy fallback ---------------------------------------------------
    if not used_cdo:
        log.info("Using scipy bilinear interpolation (CDO not available).")
        values = remapbil_scipy(cmip_da, target_lats, target_lons)

        # Build coordinate dict
        single_step = cmip_da.ndim == 2
        if single_step:
            coords = {"latitude": target_lats, "longitude": target_lons}
            dims = ["latitude", "longitude"]
        else:
            time_coord = cmip_da.time
            coords = {
                "time": time_coord,
                "latitude": target_lats,
                "longitude": target_lons,
            }
            dims = ["time", "latitude", "longitude"]

        arr = xr.DataArray(values, coords=coords, dims=dims, attrs=cmip_da.attrs)

    # --- 4. Persist cache ----------------------------------------------------
    if cache_nc is not None:
        cache_nc.parent.mkdir(parents=True, exist_ok=True)
        arr.to_dataset(name="tas_interp").to_netcdf(cache_nc)
        log.info("Cached interpolation saved to %s", cache_nc)

    return arr
