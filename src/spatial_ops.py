"""
spatial_ops.py — Spatial operations for the downscaling project.

Provides functions for coordinate standardisation, domain subsetting,
nearest-neighbour grid assignment, land-mask derivation, and related
geometric computations used throughout the analysis pipeline.

Usage example
-------------
from src.spatial_ops import subset_box, assign_era5_to_cmip_cells, compute_land_mask
"""

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "standardize_longitude",
    "subset_box",
    "compute_land_mask",
    "assign_era5_to_cmip_cells",
    "pixel_counts_per_cmip_cell",
    "compute_cell_edges",
    "compute_distance_to_cell_center",
    "compute_cell_sea_fraction",
]


# ---------------------------------------------------------------------------
# Coordinate standardisation
# ---------------------------------------------------------------------------

def standardize_longitude(ds: xr.Dataset, lon_name: str = "lon") -> xr.Dataset:
    """Convert longitude coordinates from [0°, 360°] to [-180°, 180°].

    If the coordinate is already in [-180°, 180°] the dataset is returned
    unchanged.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    lon_name : str
        Name of the longitude coordinate.

    Returns
    -------
    xr.Dataset
        Dataset with longitude in [-180°, 180°], sorted ascending.
    """
    lon = ds[lon_name]
    if float(lon.max()) > 180:
        ds = ds.assign_coords({lon_name: (((lon + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)
    return ds


# ---------------------------------------------------------------------------
# Domain subsetting
# ---------------------------------------------------------------------------

def subset_box(
    ds: xr.Dataset,
    lat_name: str,
    lon_name: str,
    region: dict,
    pad_lat: float = 0.0,
    pad_lon: float = 0.0,
) -> xr.Dataset:
    """Subset a Dataset to a geographic bounding box.

    Padding is applied symmetrically around the specified region boundary to
    ensure that grid cells whose centres lie outside the target area but whose
    footprints overlap it are retained.  This is particularly important for
    coarse-resolution CMIP6 cells (~1°).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    lat_name, lon_name : str
        Names of latitude and longitude coordinates.
    region : dict
        Bounding box with keys ``"south"``, ``"north"``, ``"west"``, ``"east"``
        in decimal degrees.
    pad_lat : float
        Latitude padding in degrees (applied to both north and south).
    pad_lon : float
        Longitude padding in degrees (applied to both east and west).

    Returns
    -------
    xr.Dataset
        Spatially subsetted dataset.
    """
    lat_vals = ds[lat_name].values
    south = region["south"] - pad_lat
    north = region["north"] + pad_lat
    west = region["west"] - pad_lon
    east = region["east"] + pad_lon

    # Handle both ascending and descending latitude orderings
    if lat_vals[0] <= lat_vals[-1]:
        lat_slice = slice(south, north)
    else:
        lat_slice = slice(north, south)

    lon_slice = slice(west, east)
    return ds.sel({lat_name: lat_slice, lon_name: lon_slice})


# ---------------------------------------------------------------------------
# Land mask
# ---------------------------------------------------------------------------

def compute_land_mask(era5_temp_da: xr.DataArray) -> np.ndarray:
    """Derive a static land-sea mask from the ERA5-Land missing value pattern.

    ERA5-Land provides data exclusively over land pixels; ocean grid cells
    are filled with NaN.  Because the missing-value pattern is spatially
    fixed (independent of time), the temporal mean is NaN precisely where
    no land data exist.

    Parameters
    ----------
    era5_temp_da : xr.DataArray
        ERA5-Land 2 m temperature with dimensions ``(time, latitude, longitude)``.

    Returns
    -------
    np.ndarray
        2D boolean array of shape ``(n_lat, n_lon)``.  ``True`` indicates a
        land pixel.
    """
    return ~np.isnan(era5_temp_da.mean(dim="time").values)


# ---------------------------------------------------------------------------
# Nearest-neighbour grid assignment
# ---------------------------------------------------------------------------

def assign_era5_to_cmip_cells(
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    cmip_lats: np.ndarray,
    cmip_lons: np.ndarray,
    land_mask_2d: np.ndarray,
) -> pd.DataFrame:
    """Assign each ERA5-Land land pixel to its nearest CMIP6 grid cell.

    Nearest-neighbour matching is performed independently along the latitude
    and longitude axes using Euclidean distance in degrees.  This is an
    approximation; a geodetic distance metric would be marginally more
    accurate but is unnecessary at these spatial scales.

    The limitation of this approach is that it assigns each fine pixel to
    exactly one coarse cell, ignoring the partial overlap of fine-resolution
    pixels with adjacent coarse cells.  Alternative methods (bilinear
    interpolation, area-weighted remapping) are discussed in
    ``analysis_guidelines.md``.

    Parameters
    ----------
    era5_lats : np.ndarray
        1-D array of ERA5-Land latitude centres (ascending).
    era5_lons : np.ndarray
        1-D array of ERA5-Land longitude centres (ascending).
    cmip_lats : np.ndarray
        1-D array of CMIP6 latitude centres.
    cmip_lons : np.ndarray
        1-D array of CMIP6 longitude centres.
    land_mask_2d : np.ndarray
        2D boolean land mask, shape ``(n_era5_lat, n_era5_lon)``.

    Returns
    -------
    pd.DataFrame
        One row per ERA5-Land **land** pixel with columns:
        ``era5_lat``, ``era5_lon``, ``cmip_lat``, ``cmip_lon``.
    """
    lat_dist = np.abs(era5_lats[:, None] - cmip_lats[None, :])
    lon_dist = np.abs(era5_lons[:, None] - cmip_lons[None, :])

    cmip_lat_idx = np.argmin(lat_dist, axis=1)  # (n_era5_lat,)
    cmip_lon_idx = np.argmin(lon_dist, axis=1)  # (n_era5_lon,)

    # Broadcast to 2D grids
    cmip_lat_grid = cmip_lats[np.tile(cmip_lat_idx[:, None], (1, len(era5_lons)))]
    cmip_lon_grid = cmip_lons[np.tile(cmip_lon_idx[None, :], (len(era5_lats), 1))]

    land_rows, land_cols = np.where(land_mask_2d)

    return pd.DataFrame(
        {
            "era5_lat": era5_lats[land_rows],
            "era5_lon": era5_lons[land_cols],
            "cmip_lat": cmip_lat_grid[land_rows, land_cols],
            "cmip_lon": cmip_lon_grid[land_rows, land_cols],
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assignment diagnostics
# ---------------------------------------------------------------------------

def pixel_counts_per_cmip_cell(assignment_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics of ERA5-Land pixel counts per CMIP6 cell.

    Parameters
    ----------
    assignment_df : pd.DataFrame
        Output of :func:`assign_era5_to_cmip_cells`.

    Returns
    -------
    pd.DataFrame
        One row per unique CMIP6 cell with columns:
        ``cmip_lat``, ``cmip_lon``, ``n_pixels``, ``mean`` (same as n_pixels
        per cell — included for consistency), and summary statistics computed
        across all cells: ``std``, ``min``, ``max`` are scalar values appended
        as the last row labelled ``"DOMAIN"``.

        In practice, returns a clean per-cell table with columns:
        ``cmip_lat``, ``cmip_lon``, ``n_pixels``.
    """
    counts = (
        assignment_df.groupby(["cmip_lat", "cmip_lon"])
        .size()
        .reset_index(name="n_pixels")
    )
    return counts


def pixel_count_stats(assignment_df: pd.DataFrame) -> dict:
    """Return domain-wide statistics of pixels-per-CMIP6-cell counts.

    Parameters
    ----------
    assignment_df : pd.DataFrame
        Output of :func:`assign_era5_to_cmip_cells`.

    Returns
    -------
    dict
        Keys: ``n_cells``, ``n_land_pixels``, ``mean``, ``median``,
        ``std``, ``min``, ``max``.
    """
    counts = pixel_counts_per_cmip_cell(assignment_df)["n_pixels"]
    return {
        "n_cells": len(counts),
        "n_land_pixels": int(counts.sum()),
        "mean": float(counts.mean()),
        "std": float(counts.std()),
        "min": int(counts.min()),
        "p25": float(counts.quantile(0.25)),
        "median": float(counts.median()),
        "p75": float(counts.quantile(0.75)),
        "p90": float(counts.quantile(0.90)),
        "max": int(counts.max()),
    }


# ---------------------------------------------------------------------------
# Grid geometry helpers
# ---------------------------------------------------------------------------

def compute_cell_edges(centers: np.ndarray) -> np.ndarray:
    """Compute grid-cell boundary edges from an array of cell centres.

    Assumes uniform or near-uniform spacing.  The outer edges are extrapolated
    by half the grid spacing from the first and last centres, respectively.

    Parameters
    ----------
    centers : np.ndarray
        1-D sorted array of cell-centre coordinates.

    Returns
    -------
    np.ndarray
        Array of length ``len(centers) + 1`` containing cell boundary edges.
    """
    mid = (centers[:-1] + centers[1:]) / 2.0
    lo = centers[0] - (centers[1] - centers[0]) / 2.0
    hi = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return np.concatenate([[lo], mid, [hi]])


def compute_distance_to_cell_center(assignment_df: pd.DataFrame) -> pd.Series:
    """Compute the Euclidean distance (in degrees) from each ERA5-Land pixel
    to its assigned CMIP6 cell centre.

    This metric is used to investigate whether downscaling residuals are
    systematically larger for pixels located far from their assigned coarse
    cell centre, which would indicate that the nearest-neighbour assumption
    introduces spatial bias.

    Parameters
    ----------
    assignment_df : pd.DataFrame
        Output of :func:`assign_era5_to_cmip_cells`, containing
        ``era5_lat``, ``era5_lon``, ``cmip_lat``, ``cmip_lon``.

    Returns
    -------
    pd.Series
        Distance in degrees (Euclidean), same index as *assignment_df*.
    """
    dlat = assignment_df["era5_lat"].values - assignment_df["cmip_lat"].values
    dlon = assignment_df["era5_lon"].values - assignment_df["cmip_lon"].values
    # Apply longitude correction for geographic accuracy
    lat_mid = np.radians(
        (assignment_df["era5_lat"].values + assignment_df["cmip_lat"].values) / 2
    )
    dlon_corrected = dlon * np.cos(lat_mid)
    return pd.Series(
        np.sqrt(dlat**2 + dlon_corrected**2),
        index=assignment_df.index,
        name="dist_deg",
    )


def compute_cell_sea_fraction(
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    cmip_lats: np.ndarray,
    cmip_lons: np.ndarray,
    land_mask_2d: np.ndarray,
) -> pd.DataFrame:
    """Compute the sea fraction and pixel counts for every CMIP6 cell.

    Unlike :func:`assign_era5_to_cmip_cells`, this function assigns **all**
    ERA5 grid points (land and ocean) to their nearest CMIP6 cell and reports
    the fraction that are ocean (null in ERA5-Land).  The land–sea mask is
    static (the same pixels are null on every day), so this is a geometric
    property of the grid and needs to be computed only once.

    Parameters
    ----------
    era5_lats : np.ndarray, shape (n_era5_lat,)
        ERA5-Land latitude coordinate array (ascending).
    era5_lons : np.ndarray, shape (n_era5_lon,)
        ERA5-Land longitude coordinate array (ascending).
    cmip_lats : np.ndarray, shape (n_cmip_lat,)
        CMIP6 cell-centre latitude array.
    cmip_lons : np.ndarray, shape (n_cmip_lon,)
        CMIP6 cell-centre longitude array.
    land_mask_2d : np.ndarray of bool, shape (n_era5_lat, n_era5_lon)
        True = land pixel, False = ocean pixel.

    Returns
    -------
    pd.DataFrame
        One row per CMIP6 cell that contains at least one ERA5 grid point.
        Columns: ``cmip_lat``, ``cmip_lon``, ``n_total``, ``n_land``,
        ``n_sea``, ``sea_fraction``.
    """
    # Nearest CMIP6 index for each ERA5 latitude and each ERA5 longitude
    lat_dist = np.abs(era5_lats[:, None] - cmip_lats[None, :])
    lon_dist = np.abs(era5_lons[:, None] - cmip_lons[None, :])
    cmip_lat_idx = np.argmin(lat_dist, axis=1)   # (n_era5_lat,)
    cmip_lon_idx = np.argmin(lon_dist, axis=1)   # (n_era5_lon,)

    # Flat index over every ERA5 grid point
    n_lat, n_lon = len(era5_lats), len(era5_lons)
    row_idx, col_idx = np.mgrid[0:n_lat, 0:n_lon]
    row_idx = row_idx.ravel()
    col_idx = col_idx.ravel()

    flat = pd.DataFrame({
        "cmip_lat": cmip_lats[cmip_lat_idx[row_idx]],
        "cmip_lon": cmip_lons[cmip_lon_idx[col_idx]],
        "is_land":  land_mask_2d[row_idx, col_idx].astype(np.int8),
    })

    agg = (
        flat.groupby(["cmip_lat", "cmip_lon"])["is_land"]
        .agg(n_total="count", n_land="sum")
        .reset_index()
    )
    agg["n_sea"] = agg["n_total"] - agg["n_land"]
    agg["sea_fraction"] = agg["n_sea"] / agg["n_total"]
    return agg
