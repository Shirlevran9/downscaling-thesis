"""
qm_cv.py — Leave-one-season-year-out cross-validation for the transforms.

For one season and one held-out season-year the procedure is:

1. Build the training q–q nodes from the other years (:func:`fold_nodes`).
2. Fit a transform on them.
3. Apply it to the **daily** predictor values of the held-out year.
4. Take percentiles of the corrected daily series, and of the observed daily
   series, with the same estimator.

Why step 3 works on daily values rather than on the predictor's percentiles
---------------------------------------------------------------------------
Mapping the percentile directly, ``h(Q_q(x))``, is far cheaper — five numbers
per pixel instead of ninety-odd — and it is exactly equal to
``Q_q(h(x))`` when quantiles are defined as plain order statistics, because a
monotone transform commutes with an order statistic.

It is *not* equal under numpy's default quantile estimator, which interpolates
between neighbouring order statistics: ``h((a + b) / 2)`` differs from
``(h(a) + h(b)) / 2`` for any ``h`` that is not affine.  Measured on synthetic
data with 92 test days, the gap reaches **0.39 °C for QUANT** and 0.20 °C for
RQUANT — the same size as the errors being measured, so it cannot be treated as
rounding.  ``scripts/sanity_qm_transforms.py`` checks both routes and prints the
gap for every method.

Two ways out: switch every percentile in the project to the order-statistic
estimator, or map the daily values.  This module maps the daily values, because
that is also what the literature does when it applies quantile mapping
(Cannon et al. 2015; Lange 2019), and because it keeps one quantile estimator —
numpy's default, the same one :func:`src.quantile_windows.window_percentiles`
already uses — for the prediction and the target alike.  The cost is a few
seconds over the whole run.

Hyper-parameter selection
-------------------------
One value per (method, season), chosen by an inner leave-one-season-year-out
pass pooled over a sample of 400 pixels.  Never per pixel: a hyper-parameter
fitted on ~820 points at one pixel is its own overfitting route, and 7,683
separate values could not be described in a methods section.

The selection is made once per season, not once per outer fold, so the chosen
value has seen every year.  That is a mild leak — one integer per season from a
pixel sample — and it was measured to be inconsequential: re-selecting inside
every one of the 39 outer folds returned the identical value each time, at ten
times the cost.  The selection margin is recorded so a reader can see how close
the runner-up was.

The smoothing spline's ``lam`` goes through the same path, on a log-spaced grid.
It is deliberately *not* chosen by generalised cross-validation: scipy's
``make_smoothing_spline(..., lam=None)`` costs 27x more per fit, returns a plain
``BSpline`` that does not carry the value it picked — so it could not be
reported — and GCV assumes independent residuals, which daily temperature
violates.  See :class:`src.qm_transforms.SSplinQM`.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Sequence

import numpy as np
import pandas as pd

from . import qm_nodes as qn
from .qm_nodes import DailySeason, fold_nodes
from .quantile_windows import PERCENTILES
from .qm_transforms import TRANSFORMS

__all__ = [
    "loso_folds",
    "build_fold_nodes",
    "season_percentiles",
    "select_hyper",
    "run_method",
]

log = logging.getLogger(__name__)

#: Pixels sampled when choosing a hyper-parameter or a smoothing lambda.
SAMPLE_PIXELS = 400


# ---------------------------------------------------------------------------
# Folds and percentiles
# ---------------------------------------------------------------------------

def loso_folds(ds: DailySeason) -> list[int]:
    """Season-years to hold out, one per fold.

    DJF yields 9 over 1990-1999 and the other seasons 10, because a winter
    needs a December from the preceding calendar year.  See
    :func:`src.quantile_windows.make_windows`.
    """
    return ds.season_years()


def season_percentiles(
    daily: np.ndarray,
    percentiles: Sequence[int] = PERCENTILES,
) -> np.ndarray:
    """Column-wise percentiles of a ``(n_days, n_pix)`` block.

    Uses numpy's default estimator, matching
    :func:`src.quantile_windows.window_percentiles` so that predictions and
    targets are measured the same way.
    """
    import warnings

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanpercentile(
            daily, np.asarray(percentiles, dtype=float), axis=0
        )


def _subset(ds: DailySeason, days: np.ndarray, cols: np.ndarray | None) -> DailySeason:
    """A DailySeason restricted to a day mask and optionally a pixel subset."""
    if cols is None:
        return dataclasses.replace(
            ds, x=ds.x[days], y=ds.y[days], syear=ds.syear[days]
        )
    return DailySeason(
        season=ds.season,
        x=ds.x[days][:, cols], y=ds.y[days][:, cols],
        syear=ds.syear[days], lat=ds.lat[cols], lon=ds.lon[cols],
    )


def _sample_cols(n_pix: int, n_sample: int, seed: int) -> np.ndarray:
    if n_sample <= 0 or n_sample >= n_pix:
        return np.arange(n_pix)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_pix, size=n_sample, replace=False))


def build_fold_nodes(
    daily: dict[str, DailySeason],
    n_nodes: int = 99,
) -> dict[tuple[str, int], qn.QQNodes]:
    """Pre-build the training nodes for every (season, held-out year).

    The nodes depend only on the season, the fold and the node count — never on
    the method — so building them once and sharing them across all eight methods
    removes what is otherwise the largest repeated cost in the run.  Measured at
    ~1.8 s per fold over 7,683 pixels, that is ~70 s per method saved, or about
    eight minutes over a full run.

    Memory is modest: 39 folds x 2 arrays x 99 x 7,683 float32 is ~240 MB.
    """
    out: dict[tuple[str, int], qn.QQNodes] = {}
    t0 = time.time()
    for season, ds in daily.items():
        for hold in loso_folds(ds):
            out[(season, hold)] = fold_nodes(ds, hold_out=hold, n_nodes=n_nodes)
    log.info("Built %d fold node set(s) in %.1f s.", len(out), time.time() - t0)
    return out


def _get_nodes(
    ds: DailySeason,
    hold: int,
    n_nodes: int | None,
    cache: dict[tuple[str, int], qn.QQNodes] | None,
) -> qn.QQNodes:
    """Nodes for one fold, from the shared cache when the resolution matches.

    The cache holds one resolution.  A method that fits on every training day
    (``fit_on_raw``) asks for ``n_nodes=None`` and is built on demand instead:
    the full-resolution arrays are ~50 MB per fold, so caching all 39 would cost
    ~2 GB for no gain, and the three raw-fitting methods are batched and cheap.
    """
    if n_nodes is not None and cache is not None:
        try:
            return cache[(ds.season, hold)]
        except KeyError:
            pass
    return fold_nodes(ds, hold_out=hold, n_nodes=n_nodes)


# ---------------------------------------------------------------------------
# Hyper-parameter selection
# ---------------------------------------------------------------------------

def _hp_candidates(method: str) -> list[dict]:
    """The grid for *method*, as a list of keyword dicts."""
    grid = TRANSFORMS[method].hp_grid
    if not grid:
        return [{}]
    keys = list(grid)
    out: list[dict] = [{}]
    for k in keys:
        out = [dict(base, **{k: v}) for base in out for v in grid[k]]
    return out


def select_hyper(
    method: str,
    ds: DailySeason,
    n_nodes: int = 99,
    percentiles: Sequence[int] = PERCENTILES,
    extrap: str = "boe",
    sample_pixels: int = SAMPLE_PIXELS,
    seed: int = 0,
    fixed: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Pick one hyper-parameter setting for a season.

    An inner leave-one-season-year-out pass over the season's years, scored as
    the mean absolute error across the requested percentiles and a sample of
    pixels.  One value per (method, season), which is what the design calls for:
    a value per pixel would be its own overfitting route and could not be
    described in a methods section.

    The selection is made once per season rather than once per outer fold.  That
    means the chosen value has seen every year, so it is a mild leak — one
    integer per season, informed by a 400-pixel sample.  The alternative,
    re-running the inner pass inside all 39 outer folds, costs ten times as much
    and was measured to return the identical value in every fold, so the leak
    buys nothing to remove.  State the choice in Methods and report the
    selection margin from *scores*.

    Parameters
    ----------
    fixed : dict, optional
        Hyper-parameters not to search over, e.g. ``{"lam": 0.37}`` for the
        spline, whose value comes from :func:`select_lambda` instead.

    Returns
    -------
    (best, scores) : tuple
        *best* is the winning keyword dict; *scores* has one row per candidate
        with its inner MAE, best first.
    """
    fixed = dict(fixed or {})
    cands = _hp_candidates(method)
    if len(cands) == 1 and not cands[0]:
        return fixed, pd.DataFrame(
            [{"method": method, "season": ds.season, "hp": "{}", "inner_mae": np.nan}]
        )

    cols = _sample_cols(ds.n_pix, sample_pixels, seed)
    sub = _subset(ds, np.ones(len(ds.syear), dtype=bool), cols)
    inner_years = sub.season_years()
    # The inner search must use the same sample the outer fit will use.
    fit_nodes = None if TRANSFORMS[method].fit_on_raw else n_nodes

    rows = []
    for hp in cands:
        use = {**hp, **fixed}
        errs = []
        for hold in inner_years:
            nodes = fold_nodes(sub, hold_out=hold, n_nodes=fit_nodes)
            t = TRANSFORMS[method](extrap=extrap, **use).fit(nodes)
            test = sub.syear == hold
            yhat_q = season_percentiles(t.predict(sub.x[test].astype(np.float64)), percentiles)
            y_q = season_percentiles(sub.y[test].astype(np.float64), percentiles)
            errs.append(np.nanmean(np.abs(yhat_q - y_q)))
        rows.append({
            "method": method, "season": ds.season,
            "hp": str(hp), "inner_mae": float(np.mean(errs)),
            **{f"hp_{k}": v for k, v in hp.items()},
        })

    scores = pd.DataFrame(rows).sort_values("inner_mae").reset_index(drop=True)
    best_hp = cands[int(np.argmin([r["inner_mae"] for r in rows]))]

    # A winner sitting at the edge of the grid means the grid is too narrow and
    # the comparison has a silent ceiling.  Say so rather than hide it.
    grid = TRANSFORMS[method].hp_grid
    for k, v in best_hp.items():
        if v in (grid[k][0], grid[k][-1]) and len(grid[k]) > 1:
            log.warning(
                "%s season %s: chose %s=%s at the edge of grid %s; "
                "consider widening it.",
                method, ds.season, k, v, grid[k],
            )
    log.info(
        "%s season %s: chose %s (inner MAE %.4f; runner-up %.4f).",
        method, ds.season, best_hp, scores.loc[0, "inner_mae"],
        scores.loc[1, "inner_mae"] if len(scores) > 1 else np.nan,
    )
    return {**best_hp, **fixed}, scores


# ---------------------------------------------------------------------------
# The outer loop
# ---------------------------------------------------------------------------

def run_method(
    method: str,
    daily: dict[str, DailySeason],
    pixels: pd.DataFrame,
    n_nodes: int = 99,
    extrap: str = "boe",
    percentiles: Sequence[int] = PERCENTILES,
    sample_pixels: int = SAMPLE_PIXELS,
    seed: int = 0,
    search_hyper: bool = True,
    nodes_by_fold: dict[tuple[str, int], qn.QQNodes] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cross-validate one method over every season.

    Parameters
    ----------
    method : str
        Key of :data:`src.qm_transforms.TRANSFORMS`.
    daily : dict of str to DailySeason
        Output of :func:`src.qm_nodes.build_daily_seasons`.
    pixels : pd.DataFrame
        Output of :func:`src.qm_nodes.select_pixels`; supplies ``pix_id``,
        ``lat`` and ``lon`` for the output tables.
    search_hyper : bool
        When ``False``, use each transform's default hyper-parameters instead of
        searching.  Useful for a fast pass.
    nodes_by_fold : dict, optional
        Pre-built nodes from :func:`build_fold_nodes`, shared across methods.
        Built on demand when omitted.

    Returns
    -------
    (pred, diag, hyper) : tuple of pd.DataFrame
        *pred* has one row per (pixel, season, held-out year) with the observed,
        uncorrected and corrected percentiles side by side.  *diag* carries the
        per-fit diagnostics.  *hyper* records the inner-CV scores.
    """
    if method not in TRANSFORMS:
        raise ValueError(f"Unknown method {method!r}; expected one of {list(TRANSFORMS)}.")

    pix_id = pixels["pix_id"].to_numpy(np.int32)
    lat = pixels["lat"].to_numpy(np.float32)
    lon = pixels["lon"].to_numpy(np.float32)
    qs = list(percentiles)

    pred_rows, diag_rows, hyper_rows = [], [], []
    t_start = time.time()

    for season, ds in daily.items():
        if len(lat) != ds.n_pix:
            raise ValueError(
                f"pixels has {len(lat)} rows but season {season} has {ds.n_pix} columns."
            )
        years = loso_folds(ds)

        # One hyper-parameter setting per (method, season), chosen before the
        # fold loop.  Every method with a grid goes through the same path,
        # including the spline's lambda.
        if search_hyper and TRANSFORMS[method].hp_grid:
            hp, scores = select_hyper(
                method, ds, n_nodes, qs, extrap, sample_pixels, seed,
            )
            hyper_rows.append(scores)
        else:
            hp = {}

        # Match the qmap package per family: its parametric and
        # distribution-derived fitters use every data point, its non-parametric
        # fitters a percentile table.  See qm_transforms.QQTransform.fit_on_raw.
        fit_nodes = None if TRANSFORMS[method].fit_on_raw else n_nodes

        for hold in years:
            nodes = _get_nodes(ds, hold, fit_nodes, nodes_by_fold)
            t = TRANSFORMS[method](extrap=extrap, **hp).fit(nodes)

            test = ds.syear == hold
            x_days = ds.x[test].astype(np.float64)
            y_days = ds.y[test].astype(np.float64)
            # Map the daily values, then take percentiles -- see the module
            # docstring for why this is not done at percentile level.
            yhat_q = season_percentiles(t.predict(x_days), qs)
            x_q = season_percentiles(x_days, qs)
            y_q = season_percentiles(y_days, qs)

            frame = {
                "pix_id": pix_id, "lat": lat, "lon": lon,
                "season": season, "syear": np.int16(hold),
                "n_test_days": np.int16(int(test.sum())),
            }
            for qi, q in enumerate(qs):
                frame[f"y_p{q}"] = y_q[qi].astype(np.float32)
                frame[f"x_p{q}"] = x_q[qi].astype(np.float32)
                frame[f"yhat_p{q}"] = yhat_q[qi].astype(np.float32)
            pred_rows.append(pd.DataFrame(frame))

            d = t.diagnostics()
            diag_rows.append(pd.DataFrame({
                "pix_id": pix_id, "season": season, "syear": np.int16(hold),
                "n_nodes": np.int16(nodes.n_nodes),
                **{k: v for k, v in d.items()},
            }))

    pred = pd.concat(pred_rows, ignore_index=True)
    diag = pd.concat(diag_rows, ignore_index=True)
    hyper = (pd.concat(hyper_rows, ignore_index=True) if hyper_rows
             else pd.DataFrame(columns=["method", "season", "hp", "inner_mae"]))

    pred["method"] = method
    diag["method"] = method
    for df in (pred, diag):
        df["season"] = pd.Categorical(df["season"], categories=list(qn.SEASONS))
        df["method"] = df["method"].astype("category")

    log.info(
        "%s: %d prediction row(s) over %d season-fold(s) in %.1f s.",
        method, len(pred), len(diag) // max(len(lat), 1), time.time() - t_start,
    )
    return pred, diag, hyper
