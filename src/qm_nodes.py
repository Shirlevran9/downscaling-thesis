"""
qm_nodes.py — Season/fold bookkeeping and the q–q node reduction.

This module turns the two daily gridded fields into the only two things the
transform methods need:

1. :class:`DailySeason` — for one meteorological season, the predictor and
   observation daily values at every selected land pixel, plus the season-year
   of each day.  Built once and shared by every method.
2. :class:`QQNodes` — for one cross-validation fold, the empirical q–q curve of
   that season's **training** days at every pixel, reduced to a fixed number of
   nodes.

Why nodes rather than the raw sorted pairs
------------------------------------------
Every transform in :mod:`src.qm_transforms` is a functional of the empirical
q–q curve alone.  Order statistics already sit at (near) equally spaced
plotting positions, so replacing ~820 sorted pairs with 99 quantiles is a
uniform thinning **in probability**, not a reweighting: both are quadrature
estimates of the same integral.  Measured effect on cost: a smoothing-spline
fit drops from 305 ms to 28 ms, and a four-parameter ``curve_fit`` from 1.4 ms
to 0.46 ms.  The information given up lies beyond roughly the 0.5th and 99.5th
probability, which is exactly where 820 days give an unreliable estimate.

Two things are deliberately *not* reduced.  ``QuantQM``'s node count is its own
hyper-parameter, so it re-derives nodes from the raw days.  ``NormalQM`` fits
its mean and standard deviation on the raw days too, because a node-based
standard deviation is slightly under-dispersed — linear interpolation between
quantile nodes shortens the tails.  Both read the raw-day statistics that
:class:`QQNodes` carries alongside the nodes.

Season convention
-----------------
Seasons are **meteorological** — DJF/MAM/JJA/SON — grouped by *season-year*,
with December rolled forward into the following year's winter.  This is the
``"season"`` scheme of :mod:`src.quantile_windows`, and it is not the same as
that module's ``"quarter"`` scheme (calendar quarters JFM/AMJ/JAS/OND).  The
grouping is imported from there rather than redefined here, so the project has
one definition; do not reach for :func:`src.data_io.seasonal_split`, which is a
third, incompatible copy.

Typical usage
-------------
from src.qm_nodes import select_pixels, build_daily_seasons, fold_nodes

pixels = select_pixels(land_mask_2d, era5_lats, era5_lons)
daily = build_daily_seasons(era5_temp, bilinear_da, dates, pixels)
nodes = fold_nodes(daily["JJA"], hold_out=1995, n_nodes=99)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import quantile_windows as qw

__all__ = [
    "SEASONS",
    "DailySeason",
    "QQNodes",
    "node_probs",
    "select_pixels",
    "build_daily_seasons",
    "fold_nodes",
]

log = logging.getLogger(__name__)

#: Meteorological seasons, re-exported so callers need only this module.
SEASONS = qw.SEASONS

#: A pixel needs at least this many pairwise-complete training days to be fitted.
MIN_TRAIN_DAYS = 200

#: A pixel needs at least this many distinct x nodes to be fitted.
MIN_DISTINCT_NODES = 8


@dataclasses.dataclass(frozen=True)
class DailySeason:
    """Daily values for one season, at every selected land pixel.

    Attributes
    ----------
    season : str
        One of :data:`SEASONS`.
    x, y : np.ndarray
        Float32, shape ``(n_days, n_pix)``.  ``x`` is the predictor (bilinearly
        regridded CMIP6 TAS), ``y`` the observation (ERA5-Land T2M), both in °C.
    syear : np.ndarray
        Int16, shape ``(n_days,)``.  Season-year of each day.
    lat, lon : np.ndarray
        Float32, shape ``(n_pix,)``.  Pixel coordinates, in column order of
        ``x`` and ``y``.
    """

    season: str
    x: np.ndarray
    y: np.ndarray
    syear: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    @property
    def n_pix(self) -> int:
        return self.x.shape[1]

    def season_years(self) -> list[int]:
        """The season-years present, in ascending order."""
        return sorted(int(v) for v in np.unique(self.syear))


@dataclasses.dataclass(frozen=True)
class QQNodes:
    """One fold's empirical q–q curves, batched over pixels.

    Attributes
    ----------
    x, y : np.ndarray
        Float32, shape ``(n_nodes, n_pix)``.  Strictly increasing along axis 0
        for ``x``, non-decreasing for ``y``.  Column *p* is pixel *p*'s q–q
        curve: ``(x[j, p], y[j, p])`` is the pair at the *j*-th node
        probability.
    x_lo, x_hi, y_lo, y_hi : np.ndarray
        Float32, shape ``(n_pix,)``.  **Raw** training minima and maxima, not
        the node extremes (which sit about ``1/(2 n_nodes)`` inside them).
        Extrapolation is anchored on these.
    x_mean, x_sd, y_mean, y_sd : np.ndarray
        Float32, shape ``(n_pix,)``.  Raw-day moments, used by ``NormalQM``.
    n_train : np.ndarray
        Int16, shape ``(n_pix,)``.  Pairwise-complete training days.
    n_distinct : np.ndarray
        Int16, shape ``(n_pix,)``.  Distinct ``x`` node values before de-tying.
    valid : np.ndarray
        Bool, shape ``(n_pix,)``.  ``False`` where the pixel must not be fitted.
    season : str
    hold_out : int | None
        The season-year excluded from the fit, or ``None`` when every year was
        pooled.
    """

    x: np.ndarray
    y: np.ndarray
    x_lo: np.ndarray
    x_hi: np.ndarray
    y_lo: np.ndarray
    y_hi: np.ndarray
    x_mean: np.ndarray
    x_sd: np.ndarray
    y_mean: np.ndarray
    y_sd: np.ndarray
    n_train: np.ndarray
    n_distinct: np.ndarray
    valid: np.ndarray
    season: str
    hold_out: int | None

    @property
    def n_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def n_pix(self) -> int:
        return self.x.shape[1]

    def subset(self, cols: np.ndarray) -> "QQNodes":
        """Return the same fold restricted to a subset of pixel columns."""
        return QQNodes(
            x=self.x[:, cols], y=self.y[:, cols],
            x_lo=self.x_lo[cols], x_hi=self.x_hi[cols],
            y_lo=self.y_lo[cols], y_hi=self.y_hi[cols],
            x_mean=self.x_mean[cols], x_sd=self.x_sd[cols],
            y_mean=self.y_mean[cols], y_sd=self.y_sd[cols],
            n_train=self.n_train[cols], n_distinct=self.n_distinct[cols],
            valid=self.valid[cols], season=self.season, hold_out=self.hold_out,
        )


# ---------------------------------------------------------------------------
# Node probabilities
# ---------------------------------------------------------------------------

def node_probs(n_nodes: int) -> np.ndarray:
    """Percentile positions of the q–q nodes, in the range 0-100.

    Uses the Hazen positions ``(j - 0.5) / n`` for ``j = 1..n``, which keeps
    every node strictly inside the sample and so avoids making the first and
    last node simply the minimum and maximum.  The raw extremes are carried
    separately on :class:`QQNodes` as ``x_lo``/``x_hi``.

    The *estimator* is left as ``numpy``'s default (``method="linear"``, i.e.
    linear interpolation between order statistics), matching
    :func:`src.quantile_windows.window_percentiles`, which produces the
    evaluation percentiles that are later fed to ``predict``.  Keeping one
    estimator on both sides means ``h``'s domain and the values it is evaluated
    at come from the same construction.  That consistency matters more here
    than any particular textbook plotting position.
    """
    if n_nodes < 4:
        raise ValueError(f"n_nodes must be at least 4, got {n_nodes}.")
    j = np.arange(1, n_nodes + 1, dtype=float)
    return (j - 0.5) / n_nodes * 100.0


# ---------------------------------------------------------------------------
# Pixel selection
# ---------------------------------------------------------------------------

def select_pixels(
    land_mask_2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    stride: int = 1,
    max_pixels: int = 0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Choose the land pixels to fit, and fix their order once.

    Every later artefact joins on ``pix_id`` or on ``(lat, lon)``, so this
    ordering is canonical for the whole run.

    Parameters
    ----------
    land_mask_2d : np.ndarray
        2-D boolean mask, shape ``(n_lat, n_lon)``, ``True`` = land.
    lats, lons : np.ndarray
        Coordinate axes of the mask.  Note that ``lats`` runs **descending**
        (38 -> 24) on this project's ERA5-Land grid; nothing here assumes
        otherwise.
    stride : int
        Keep every *stride*-th land pixel.  ``stride=10`` gives a spatially
        spread tenth of the domain, for fast iteration.
    max_pixels : int
        Truncate to at most this many pixels after striding.  ``0`` means no
        limit.  Combined with *seed* the selection is shuffled first, so a
        sample is not confined to one corner of the domain.
    seed : int, optional
        Shuffle seed, used only when *max_pixels* is set.

    Returns
    -------
    pd.DataFrame
        Columns ``pix_id`` (int32), ``row``/``col`` (int16, indices into the
        mask), ``lat``/``lon`` (float32).  ``lat``/``lon`` are float32 to join
        cleanly with ``static_pixels.parquet`` and to survive the 4-decimal
        rounding in :func:`src.qm_visualization._grid_from_points`.
    """
    rows, cols = np.where(land_mask_2d)
    if stride > 1:
        rows, cols = rows[::stride], cols[::stride]
    if max_pixels and len(rows) > max_pixels:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(rows), size=max_pixels, replace=False))
        rows, cols = rows[take], cols[take]

    out = pd.DataFrame({
        "pix_id": np.arange(len(rows), dtype=np.int32),
        "row": rows.astype(np.int16),
        "col": cols.astype(np.int16),
        "lat": np.asarray(lats)[rows].astype(np.float32),
        "lon": np.asarray(lons)[cols].astype(np.float32),
    })
    log.info(
        "Selected %d of %d land pixels (stride=%d, max_pixels=%s).",
        len(out), int(land_mask_2d.sum()), stride, max_pixels or "all",
    )
    return out


# ---------------------------------------------------------------------------
# Daily per-season matrices
# ---------------------------------------------------------------------------

def build_daily_seasons(
    era5_temp: xr.DataArray,
    predictor_da: xr.DataArray,
    dates: Sequence[str],
    pixels: pd.DataFrame,
    seasons: Sequence[str] = SEASONS,
) -> dict[str, DailySeason]:
    """Split the two daily fields into one :class:`DailySeason` per season.

    Days belonging to an incomplete season are dropped, exactly as
    :func:`src.quantile_windows.make_windows` drops them for the ``"season"``
    scheme — over 1990-1999 that is January-February 1990 and December 1999.

    Parameters
    ----------
    era5_temp : xr.DataArray
        Observed T2M in °C, dims ``(time, latitude, longitude)``.
    predictor_da : xr.DataArray
        Predictor on the same grid and time axis, already in °C.  The cached
        bilinear field is stored in °C, so do **not** pass it through
        :func:`src.data_io.to_celsius` first.
    dates : sequence of str
        Shared ``"YYYY-MM-DD"`` dates in time order.
    pixels : pd.DataFrame
        Output of :func:`select_pixels`.
    seasons : sequence of str
        Subset of :data:`SEASONS` to build.

    Returns
    -------
    dict of str to DailySeason
    """
    dates = list(dates)
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    season_of_day, syear_of_day = qw.season_year(idx)

    # Reuse the window builder's completeness rule rather than re-deriving it,
    # so the daily matrices and the percentile table drop the same days.
    _, codes = qw.make_windows(dates, qw.SEASON_SCHEME)
    complete = codes >= 0
    if not complete.all():
        log.info("Dropping %d day(s) in incomplete seasons.", int((~complete).sum()))

    rows = pixels["row"].to_numpy()
    cols = pixels["col"].to_numpy()
    lat = pixels["lat"].to_numpy(np.float32)
    lon = pixels["lon"].to_numpy(np.float32)

    y_all = np.asarray(era5_temp.values)
    x_all = np.asarray(predictor_da.values)
    if x_all.shape != y_all.shape:
        raise ValueError(
            f"predictor shape {x_all.shape} != observation shape {y_all.shape}."
        )

    out: dict[str, DailySeason] = {}
    for s in seasons:
        sel = complete & (season_of_day == s)
        if not sel.any():
            raise ValueError(f"No complete days found for season {s!r}.")
        out[s] = DailySeason(
            season=s,
            x=x_all[sel][:, rows, cols].astype(np.float32),
            y=y_all[sel][:, rows, cols].astype(np.float32),
            syear=syear_of_day[sel].astype(np.int16),
            lat=lat,
            lon=lon,
        )
        log.info(
            "Season %s: %d days over %d season-years, %d pixels.",
            s, int(sel.sum()), len(out[s].season_years()), out[s].n_pix,
        )
    return out


# ---------------------------------------------------------------------------
# Per-fold q–q nodes
# ---------------------------------------------------------------------------

def fold_nodes(
    ds: DailySeason,
    hold_out: int | None,
    n_nodes: int | None = 99,
    min_days: int = MIN_TRAIN_DAYS,
) -> QQNodes:
    """Build the training q–q nodes for one cross-validation fold.

    Parameters
    ----------
    ds : DailySeason
        One season's daily values.
    hold_out : int or None
        Season-year to exclude from the fit.  ``None`` pools every year, which
        is used for the in-sample identity check and for hyper-parameter
        selection shells.
    n_nodes : int or None
        Number of q–q nodes.  ``None`` uses **every** training day, i.e. the
        raw sorted pairs, which is the ``qstep = NULL`` behaviour of the
        ``qmap`` package.  A node count is ``qstep = 1 / n_nodes``.
    min_days : int
        Minimum pairwise-complete training days for a pixel to be fitted.

    Returns
    -------
    QQNodes

    Notes
    -----
    **Pairwise-complete masking.** A day is used only when it is finite in both
    ``x`` and ``y``.  Masking the two sides independently is superficially
    defensible — quantile mapping compares marginals — but it would leave the
    two node vectors summarising different sets of days, so ``(x[j], y[j])``
    would no longer be a q–q pair over a common period.

    **NaN.** ``np.percentile`` sorts NaN to the end, so a single NaN day would
    silently corrupt every node in that column.  ``np.nanpercentile`` is used
    throughout, and the pairwise mask is applied to both arrays first.  In
    practice the risk is one-sided: the bilinear predictor is gap-free, so only
    ``y`` can have missing days.

    **Ties.** ``scipy.interpolate.make_smoothing_spline`` raises
    ``ValueError: 'x' should be an ascending array`` on any duplicate, while
    ``np.interp`` tolerates ties — so an untreated tie surfaces on one method
    only and looks like a method bug.  It is therefore resolved here, once, for
    every method.  The textbook fix (average ``y`` over each tied run of ``x``)
    would make the node arrays ragged and break the batched-over-pixels
    interface, so instead ``x`` is forced strictly increasing by a cumulative
    nudge of ``1e-6`` of the pixel's training range.  Spread over at most
    ``n_nodes`` nodes that is under ``1e-4`` of the range, far below the
    sampling error of any quantile. ``n_distinct`` records the pre-nudge count
    so the tie rate stays reportable.
    """
    if hold_out is None:
        train = np.ones(len(ds.syear), dtype=bool)
    else:
        train = ds.syear != hold_out
        if not train.any():
            raise ValueError(f"Holding out {hold_out} leaves no training days.")

    xt = ds.x[train].astype(np.float64, copy=True)
    yt = ds.y[train].astype(np.float64, copy=True)

    # Pairwise-complete: blank the same day on both sides.
    bad = ~(np.isfinite(xt) & np.isfinite(yt))
    if bad.any():
        xt[bad] = np.nan
        yt[bad] = np.nan
        n_cols_hit = int((bad.any(axis=0)).sum())
        log.info(
            "Season %s fold %s: %d day-pixel value(s) blanked across %d pixel(s).",
            ds.season, hold_out, int(bad.sum()), n_cols_hit,
        )
    n_train = (~bad).sum(axis=0).astype(np.int16)

    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
            x_lo = np.nanmin(xt, axis=0)
            x_hi = np.nanmax(xt, axis=0)
            y_lo = np.nanmin(yt, axis=0)
            y_hi = np.nanmax(yt, axis=0)
            x_mean = np.nanmean(xt, axis=0)
            y_mean = np.nanmean(yt, axis=0)
            x_sd = np.nanstd(xt, axis=0, ddof=1)
            y_sd = np.nanstd(yt, axis=0, ddof=1)

            if n_nodes is None:
                # Every training day: sort each column and drop the rows that
                # are NaN in every column.  Sorting is cheaper than asking
                # nanpercentile for n_train probabilities and returns the same
                # order statistics.  NaNs sort to the end, so trailing rows are
                # dropped down to the smallest pairwise-complete count.
                xn = np.sort(xt, axis=0)
                yn = np.sort(yt, axis=0)
                keep = int(n_train.min())
                xn, yn = xn[:keep], yn[:keep]
            else:
                probs = node_probs(n_nodes)
                xn = np.nanpercentile(xt, probs, axis=0)
                yn = np.nanpercentile(yt, probs, axis=0)

    # Defensive: the node axis should already be sorted.
    xn = np.maximum.accumulate(xn, axis=0)
    yn = np.maximum.accumulate(yn, axis=0)

    # The node axis is sorted, so distinct values are just the strict
    # increases plus one.  An all-NaN column gives 1 and so fails the
    # MIN_DISTINCT_NODES check below.
    n_distinct = (1 + (np.diff(xn, axis=0) > 0).sum(axis=0)).astype(np.int16)

    # De-tie x so every method sees a strictly increasing abscissa.  The nudge
    # must survive the float32 cast at the end of this function: float32 holds
    # about 7 significant digits, so at 25 °C the representable spacing is
    # ~3e-6 and a 1e-6 nudge would round straight back to a tie.  Sizing it in
    # float32 ULPs at the column's own magnitude keeps the invariant true after
    # the cast, and 99 nodes of it still moves x by under 1e-2 °C.
    span = np.where(np.isfinite(x_hi - x_lo) & (x_hi > x_lo), x_hi - x_lo, 1.0)
    scale = np.maximum(np.maximum(np.abs(x_lo), np.abs(x_hi)), 1.0)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    eps = np.maximum(1e-6 * span, 8.0 * scale * np.finfo(np.float32).eps)
    for j in range(1, xn.shape[0]):
        xn[j] = np.maximum(xn[j], xn[j - 1] + eps)

    valid = (
        (n_train >= min_days)
        & (n_distinct >= MIN_DISTINCT_NODES)
        & np.isfinite(x_sd) & (x_sd > 0)
        & np.isfinite(y_sd) & (y_sd > 0)
        & np.isfinite(xn).all(axis=0)
        & np.isfinite(yn).all(axis=0)
    )
    if not valid.all():
        log.info(
            "Season %s fold %s: %d of %d pixel(s) not fittable.",
            ds.season, hold_out, int((~valid).sum()), len(valid),
        )

    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
    return QQNodes(
        x=f32(xn), y=f32(yn),
        x_lo=f32(x_lo), x_hi=f32(x_hi), y_lo=f32(y_lo), y_hi=f32(y_hi),
        x_mean=f32(x_mean), x_sd=f32(x_sd),
        y_mean=f32(y_mean), y_sd=f32(y_sd),
        n_train=n_train, n_distinct=n_distinct, valid=valid,
        season=ds.season, hold_out=hold_out,
    )
