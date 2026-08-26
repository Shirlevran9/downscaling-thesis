"""
quantile_windows.py — Distribution windows and per-window percentiles.

The quantile-mapping baseline compares distributions, not days.  A
*distribution window* is a block of consecutive days over which a temperature
distribution is estimated at a single fine-grid pixel.  Four window lengths are
used, all non-overlapping and calendar aligned:

===========  ==============================================  ==========
scheme       definition                                      n windows
===========  ==============================================  ==========
``14d``      26 fixed blocks per year from day-of-year 1;     260
             the last block absorbs the leftover day
``month``    calendar month                                   120
``quarter``  JFM / AMJ / JAS / OND                             40
``year``     calendar year                                      10
===========  ==============================================  ==========

Caveat: a 14-day window holds only 14 values, so its 5th and 90th percentiles
are noisy estimates.  Window length trades sampling noise against temporal
resolution.

Typical usage
-------------
from src.quantile_windows import make_windows, build_percentile_table

windows_df, codes = make_windows(shared_dates, "quarter")
table = build_percentile_table(era5_temp, pred_da, land_mask_2d,
                               shared_dates, "quarter", (5, 25, 50, 75, 90))
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "SCHEMES",
    "PERCENTILES",
    "BLOCK_DAYS",
    "make_windows",
    "window_percentiles",
    "build_percentile_table",
]

log = logging.getLogger(__name__)

#: Window schemes in increasing length order.
SCHEMES = ["14d", "month", "quarter", "year"]

#: Percentiles of interest.
PERCENTILES = (5, 25, 50, 75, 90)

#: Nominal length of a ``14d`` block.
BLOCK_DAYS = 14


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def _block_index(idx: pd.DatetimeIndex) -> np.ndarray:
    """Map each date to a 0-based 14-day block index within its year.

    The rank of a day *within its own year* is used, not the calendar
    day-of-year.  After ``data_io.align_calendars`` removes the ERA5-Land leap
    days, every year holds exactly 365 rows, but 29 February is missing from
    the calendar in 1992 and 1996, so ``dayofyear`` would be off by one from
    1 March onward and the blocks would drift between years.  Ranking within
    the year keeps all 10 years on identical block boundaries.

    365 = 26*14 + 1, so the final block of each year is given 15 days rather
    than creating a 27th block holding a single day.
    """
    rank = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.year).cumcount()
    return np.minimum(rank.to_numpy() // BLOCK_DAYS, 25)


def make_windows(
    dates: list[str] | pd.DatetimeIndex,
    scheme: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Cut a date axis into non-overlapping, calendar-aligned windows.

    Parameters
    ----------
    dates : list of str or pd.DatetimeIndex
        ``"YYYY-MM-DD"`` strings (or a DatetimeIndex) in time order, one entry
        per time step of the data.
    scheme : {"14d", "month", "quarter", "year"}
        Window length.

    Returns
    -------
    (windows_df, group_codes) : tuple
        ``windows_df`` has one row per window with columns ``window_id``
        (0-based, in chronological order), ``label``, ``start``, ``end`` and
        ``n_days``.  ``group_codes`` is an integer array of length
        ``len(dates)`` giving the ``window_id`` of each time step.

    Raises
    ------
    ValueError
        If *scheme* is unknown, or if the date axis is not sorted.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}; expected one of {SCHEMES}.")

    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    if not idx.is_monotonic_increasing:
        raise ValueError("dates must be sorted in ascending order.")

    year = idx.year.to_numpy()

    if scheme == "14d":
        block = _block_index(idx)
        labels = np.array(
            [f"{y}-B{b + 1:02d}" for y, b in zip(year, block)]
        )
    elif scheme == "month":
        labels = np.array([f"{y}-{m:02d}" for y, m in zip(year, idx.month)])
    elif scheme == "quarter":
        labels = np.array([f"{y}-Q{q}" for y, q in zip(year, idx.quarter)])
    else:  # "year"
        labels = np.array([str(y) for y in year])

    # Labels are already chronological because the date axis is sorted, so
    # first-appearance order gives the window ids.
    uniq, first_pos = np.unique(labels, return_index=True)
    order = np.argsort(first_pos)
    ordered_labels = uniq[order]
    label_to_id = {lab: i for i, lab in enumerate(ordered_labels)}

    group_codes = np.array([label_to_id[lab] for lab in labels], dtype=np.int32)

    rows = []
    for lab in ordered_labels:
        member = idx[labels == lab]
        rows.append(
            {
                "window_id": label_to_id[lab],
                "label": lab,
                "start": member[0],
                "end": member[-1],
                "n_days": len(member),
            }
        )
    windows_df = pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)

    return windows_df, group_codes


# ---------------------------------------------------------------------------
# Per-window percentiles
# ---------------------------------------------------------------------------

def window_percentiles(
    field_3d: np.ndarray,
    group_codes: np.ndarray,
    n_windows: int,
    percentiles: tuple | list = PERCENTILES,
) -> np.ndarray:
    """Compute percentiles of a gridded field within each window.

    Parameters
    ----------
    field_3d : np.ndarray
        Data of shape ``(n_time, n_lat, n_lon)``.  NaNs are ignored.
    group_codes : np.ndarray
        Window id of each time step, as returned by :func:`make_windows`.
    n_windows : int
        Total number of windows.
    percentiles : sequence of int
        Percentiles to compute, in the range 0-100.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(n_windows, n_lat, n_lon, n_percentiles)``.
    """
    n_lat, n_lon = field_3d.shape[1], field_3d.shape[2]
    qs = np.asarray(percentiles, dtype=float)
    out = np.empty((n_windows, n_lat, n_lon, len(qs)), dtype=np.float32)

    # Ocean pixels are NaN for the whole record, so every window hits an
    # all-NaN column.  Those pixels are dropped later by the land mask; the
    # warning carries no information here.
    import warnings

    # Time steps of a window are contiguous, so slice rather than mask.
    for wid in range(n_windows):
        pos = np.flatnonzero(group_codes == wid)
        block = field_3d[pos[0]: pos[-1] + 1]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            # (n_q, n_lat, n_lon) -> move the quantile axis last
            pct = np.nanpercentile(block, qs, axis=0)
        out[wid] = np.moveaxis(pct, 0, -1).astype(np.float32)

    return out


def build_percentile_table(
    era5_temp: xr.DataArray,
    predictor_da: xr.DataArray,
    land_mask_2d: np.ndarray,
    dates: list[str],
    scheme: str,
    percentiles: tuple | list = PERCENTILES,
    static_cols: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the wide percentile table for one predictor and one window scheme.

    One row per (window, land pixel).  Y columns hold the observed ERA5-Land
    percentiles, X columns the predictor percentiles over the same days at the
    same pixel.

    Parameters
    ----------
    era5_temp : xr.DataArray
        Observed T2M in °C, dims ``(time, latitude, longitude)``.
    predictor_da : xr.DataArray
        Predictor field on the same grid and time axis.
    land_mask_2d : np.ndarray
        2-D boolean land mask; only ``True`` pixels are kept.
    dates : list of str
        Shared ``"YYYY-MM-DD"`` dates in time order.
    scheme : str
        One of :data:`SCHEMES`.
    percentiles : sequence of int
        Percentiles to compute.
    static_cols : pd.DataFrame, optional
        Per-pixel static attributes to merge in, indexed by the columns
        ``lat`` and ``lon`` (for example ``elevation`` and ``sea_fraction``).

    Returns
    -------
    pd.DataFrame
        Columns: ``window_id``, ``window_label``, ``window_start``,
        ``n_days``, ``lat``, ``lon``, ``y_p{q}`` and ``x_p{q}`` for each *q*,
        plus any columns supplied via *static_cols*.
    """
    windows_df, codes = make_windows(dates, scheme)
    n_windows = len(windows_df)

    y_pct = window_percentiles(
        era5_temp.values, codes, n_windows, percentiles
    )
    x_pct = window_percentiles(
        predictor_da.values, codes, n_windows, percentiles
    )

    rows, cols = np.where(land_mask_2d)
    lats = era5_temp.latitude.values[rows].astype(np.float32)
    lons = era5_temp.longitude.values[cols].astype(np.float32)
    n_pix = len(rows)

    frame = {
        "window_id": np.repeat(
            windows_df["window_id"].to_numpy(np.int32), n_pix
        ),
        "window_label": np.repeat(windows_df["label"].to_numpy(), n_pix),
        "window_start": np.repeat(windows_df["start"].to_numpy(), n_pix),
        "n_days": np.repeat(windows_df["n_days"].to_numpy(np.int16), n_pix),
        "lat": np.tile(lats, n_windows),
        "lon": np.tile(lons, n_windows),
    }
    for qi, q in enumerate(percentiles):
        frame[f"y_p{q}"] = y_pct[:, rows, cols, qi].ravel()
        frame[f"x_p{q}"] = x_pct[:, rows, cols, qi].ravel()

    df = pd.DataFrame(frame)
    df["window_label"] = df["window_label"].astype("category")

    if static_cols is not None:
        df = df.merge(static_cols, on=["lat", "lon"], how="left")

    return df
