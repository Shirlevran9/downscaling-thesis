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
    "SEASON_SCHEME",
    "WINDOW_SCHEMES",
    "SEASONS",
    "PERCENTILES",
    "BLOCK_DAYS",
    "make_windows",
    "season_year",
    "window_percentiles",
    "build_percentile_table",
]

log = logging.getLogger(__name__)

#: Window schemes of the original predictor-comparison grid, in increasing
#: length order.  Left unchanged so ``scripts/run_quantile_mapping.py`` and the
#: dashboard keep building exactly the tables they built before.
SCHEMES = ["14d", "month", "quarter", "year"]

#: Meteorological-season scheme, used by the fitted quantile-mapping baseline.
#: Deliberately **not** a member of :data:`SCHEMES`; see above.
SEASON_SCHEME = "season"

#: Every scheme :func:`make_windows` accepts.
WINDOW_SCHEMES = SCHEMES + [SEASON_SCHEME]

#: Meteorological seasons, in the order they occur within a season-year.
SEASONS = ("DJF", "MAM", "JJA", "SON")

#: Month number -> meteorological season.
_SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

#: The three calendar months each season must contain to count as complete.
_MONTHS_OF_SEASON = {
    "DJF": (12, 1, 2), "MAM": (3, 4, 5),
    "JJA": (6, 7, 8), "SON": (9, 10, 11),
}

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


def season_year(
    idx: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each date to its meteorological season and season-year.

    December belongs to the **following** year's winter, so 1 December 1990,
    15 January 1991 and 20 February 1991 all carry season ``"DJF"`` and
    season-year ``1991``.  Grouping this way means holding out one season-year
    also holds out its December, so leave-one-season-year-out cross-validation
    cannot leak across the calendar-year boundary.

    Parameters
    ----------
    idx : pd.DatetimeIndex
        Dates in time order.

    Returns
    -------
    (season, syear) : tuple of np.ndarray
        ``season`` holds ``"DJF"``/``"MAM"``/``"JJA"``/``"SON"`` strings and
        ``syear`` the int16 season-year, both of length ``len(idx)``.
    """
    month = idx.month.to_numpy()
    season = np.array([_SEASON_OF_MONTH[m] for m in month])
    syear = np.where(month == 12, idx.year.to_numpy() + 1, idx.year.to_numpy())
    return season, syear.astype(np.int16)


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
    scheme : {"14d", "month", "quarter", "year", "season"}
        Window length.  ``"quarter"`` means the **calendar** quarters
        JFM/AMJ/JAS/OND; ``"season"`` means the **meteorological** seasons
        DJF/MAM/JJA/SON grouped by season-year.  The two are different
        groupings and their labels are deliberately distinguishable
        (``"1999-Q2"`` versus ``"1999-MAM"``).

    Returns
    -------
    (windows_df, group_codes) : tuple
        ``windows_df`` has one row per window with columns ``window_id``
        (0-based, in chronological order), ``label``, ``start``, ``end`` and
        ``n_days``.  ``group_codes`` is an integer array of length
        ``len(dates)`` giving the ``window_id`` of each time step, or ``-1``
        for a time step that belongs to no window.

    Notes
    -----
    Only the ``"season"`` scheme can emit ``-1``.  A meteorological season is
    kept only when all three of its calendar months are present, so a record
    starting on 1 January 1990 drops January-February 1990 (no preceding
    December) and a record ending on 31 December 1999 drops that December (no
    following January).  Over 1990-1999 that leaves **9** DJF windows against
    10 for each other season.  The alternative — keeping a 31-day stub as a
    window — would produce a P5 and P90 estimated from one or two days.

    Raises
    ------
    ValueError
        If *scheme* is unknown, or if the date axis is not sorted.
    """
    if scheme not in WINDOW_SCHEMES:
        raise ValueError(
            f"Unknown scheme {scheme!r}; expected one of {WINDOW_SCHEMES}."
        )

    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    if not idx.is_monotonic_increasing:
        raise ValueError("dates must be sorted in ascending order.")

    year = idx.year.to_numpy()

    if scheme == SEASON_SCHEME:
        season, syear = season_year(idx)
        labels = np.array([f"{y}-{s}" for y, s in zip(syear, season)])

        # Keep a season only when all three of its months are present.
        drop = np.zeros(len(idx), dtype=bool)
        month = idx.month.to_numpy()
        for lab in np.unique(labels):
            member = labels == lab
            wanted = set(_MONTHS_OF_SEASON[lab.split("-", 1)[1]])
            if set(month[member].tolist()) != wanted:
                drop |= member
                log.info(
                    "Dropping incomplete season %s (%d days, months %s).",
                    lab, int(member.sum()), sorted(set(month[member].tolist())),
                )
        labels = np.where(drop, "", labels)

        # Guard the convention: DJF must only ever hold Dec, Jan, Feb.
        is_djf = np.char.endswith(labels.astype(str), "DJF")
        assert set(month[is_djf].tolist()) <= {12, 1, 2}, "DJF holds a wrong month"

    elif scheme == "14d":
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
    # first-appearance order gives the window ids.  The empty label marks a
    # time step that belongs to no window (only the "season" scheme emits it)
    # and is mapped to -1 rather than being given an id.
    uniq, first_pos = np.unique(labels, return_index=True)
    order = np.argsort(first_pos)
    ordered_labels = [lab for lab in uniq[order] if lab != ""]
    label_to_id = {lab: i for i, lab in enumerate(ordered_labels)}
    label_to_id[""] = -1

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
