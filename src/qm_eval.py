"""
qm_eval.py — Metrics for the cross-validated quantile-mapping transforms.

Three things are computed here:

1. **Per-pixel error**, one row per (method, season, percentile, pixel), which
   is what the lat/lon error maps read.
2. **The climatology floor** — the year-to-year spread of the *observed*
   percentile.  A free-running GCM draws a different year's weather than
   reality did, so part of the error is irreducible.  Without the floor beside
   it, an MAE of 0.5 °C cannot be told apart from a perfect result.  Two floors
   are returned, one for MAE and one for RMSE, because comparing an MAE against
   a standard deviation understates it by about a fifth.
3. **The method summary**, one row per (method, season, percentile), which is
   the comparison table.

Sign convention
---------------
``bias = yhat - y`` — prediction minus observation, so a positive bias means
the corrected predictor is too warm.  This matches the project convention
documented in ``guidelines/analysis_guidelines.md`` §9 and used by
:func:`src.qm_metrics.combination_metrics`.

Note the trap this module has to navigate.  ``combination_metrics`` returns
``bias = x - y``, while :func:`src.visualization.compute_regression_metrics`
returns observation minus prediction — the opposite.  ``combination_metrics``
is called here **with keyword arguments** (``y=..., x=...``) for that reason: it
takes ``(y, x)`` positionally, so passing them the other way round silently
flips the sign of ``bias`` and ``ols_slope`` while leaving ``mae``, ``rmse`` and
``r2`` untouched — a mistake no assertion on those three would catch.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from .qm_metrics import combination_metrics
from .qm_nodes import SEASONS
from .qm_transforms import RAW_METHOD
from .quantile_windows import PERCENTILES

__all__ = [
    "melt_predictions",
    "observed_percentile_sd",
    "per_pixel_error",
    "method_summary",
    "daily_moments",
]

log = logging.getLogger(__name__)


def melt_predictions(
    pred: pd.DataFrame,
    percentiles: Sequence[int] = PERCENTILES,
) -> pd.DataFrame:
    """Turn the wide prediction table into one row per percentile.

    Input carries ``y_p5 … y_p90``, ``x_p5 … x_p90`` and ``yhat_p5 … yhat_p90``;
    output carries a ``percentile`` column and flat ``y``, ``x``, ``yhat``.
    """
    keep = [c for c in ("pix_id", "lat", "lon", "method", "season", "syear")
            if c in pred.columns]
    parts = []
    for q in percentiles:
        part = pred[keep].copy()
        part["percentile"] = np.int8(q)
        part["y"] = pred[f"y_p{q}"].to_numpy(np.float32)
        part["x"] = pred[f"x_p{q}"].to_numpy(np.float32)
        part["yhat"] = pred[f"yhat_p{q}"].to_numpy(np.float32)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["bias"] = (out["yhat"] - out["y"]).astype(np.float32)
    return out


def observed_percentile_sd(
    long: pd.DataFrame,
) -> pd.DataFrame:
    """Year-to-year spread of the observed percentile — the climatology floor.

    A free-running GCM simulates *a* plausible year, not the year that
    happened, so its internal variability is unrelated to the observed
    variability of any particular year.  The best any bias correction can do is
    therefore predict the **climatological** percentile.  The error that
    remains is the year-to-year spread of the observation, and no method can
    remove it.

    Two floors are returned, because the right one depends on the metric:

    ``obs_mad``
        Mean absolute deviation of the observed percentile from its across-year
        median — the MAE that a climatology-only prediction achieves.  The
        median is used because it is the value that minimises absolute error.
        **This is the number to compare with MAE.**
    ``obs_sd``
        Standard deviation about the mean — the RMSE that a climatology-only
        prediction achieves, since the mean minimises squared error.
        **Compare this with RMSE, not with MAE.**

    Keeping them apart matters.  For a normal sample the mean absolute deviation
    is about 0.8 standard deviations, so comparing an MAE against a standard
    deviation understates the error by roughly a fifth and can make a method
    look as though it beat a floor it never reached.

    Parameters
    ----------
    long : pd.DataFrame
        Output of :func:`melt_predictions`.  Only the observation column is
        used, so any single method's rows suffice; passing several gives the
        same answer because ``y`` does not depend on the method.

    Returns
    -------
    pd.DataFrame
        One row per (pixel, season, percentile): ``obs_mad``, ``obs_sd``,
        ``obs_mean``, ``obs_median``, ``n_years``.

    Notes
    -----
    Both floors are estimated from 10 years (9 for DJF) and use every year
    including the one being predicted, so they are mildly optimistic — a
    leave-one-out version would be a little larger.  They are a reference
    scale, not a precise quantity.
    """
    one = long[long["method"] == long["method"].iloc[0]] if "method" in long else long
    keys = ["pix_id", "lat", "lon", "season", "percentile"]
    work = one[keys + ["y"]].copy()

    # The mean absolute deviation needs each group's median subtracted first.
    # Broadcasting it back with transform keeps everything vectorised; a lambda
    # per group would run 150,000 times in Python.
    med = work.groupby(keys, observed=True)["y"].transform("median")
    work["abs_dev"] = (work["y"] - med).abs()

    out = work.groupby(keys, observed=True).agg(
        obs_mad=("abs_dev", "mean"),
        obs_sd=("y", "std"),              # pandas std is ddof=1
        obs_mean=("y", "mean"),
        obs_median=("y", "median"),
        n_years=("y", "size"),
    ).reset_index()
    for c in ("obs_mad", "obs_sd", "obs_mean", "obs_median"):
        out[c] = out[c].astype(np.float32)
    out["n_years"] = out["n_years"].astype(np.int8)
    return out


def per_pixel_error(long: pd.DataFrame) -> pd.DataFrame:
    """Error aggregated over folds, one row per (method, season, percentile, pixel).

    The absolute and squared bias are materialised as columns first so every
    aggregation is a built-in (``mean``, ``std``, ``size``) rather than a lambda.
    With eight methods this groups about 12 million rows into 1.2 million
    groups; pandas runs a lambda once per group in Python, which took minutes,
    while the built-ins are cythonised and take seconds.
    """
    work = long[["method", "season", "percentile", "pix_id", "lat", "lon", "bias"]].copy()
    work["abs_bias"] = work["bias"].abs()
    work["sq_bias"] = work["bias"] * work["bias"]

    out = work.groupby(
        ["method", "season", "percentile", "pix_id", "lat", "lon"], observed=True
    ).agg(
        mean_bias=("bias", "mean"),
        bias_sd=("bias", "std"),          # pandas std is ddof=1
        mae=("abs_bias", "mean"),
        _msq=("sq_bias", "mean"),
        n=("bias", "size"),
    ).reset_index()
    out["rmse"] = np.sqrt(out.pop("_msq"))

    for c in ("mean_bias", "bias_sd", "mae", "rmse"):
        out[c] = out[c].astype(np.float32)
    out["n"] = out["n"].astype(np.int8)
    return out[["method", "season", "percentile", "pix_id", "lat", "lon",
                "mean_bias", "bias_sd", "mae", "rmse", "n"]]


def method_summary(
    long: pd.DataFrame,
    floor: pd.DataFrame,
    diag: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The comparison table: one row per (method, season, percentile).

    ``mae``/``rmse``/``bias`` are pooled over every (pixel, fold) pair.
    ``pearson_r`` and ``r2`` are computed **across pixels**, pooled over folds,
    which is what :func:`src.qm_metrics.metrics_table` already does — so the
    numbers are comparable with the older predictor-comparison grid.
    ``pearson_r_foldmean`` repeats the calculation fold by fold and averages, so
    a reader can see how much the pooling inflates it.

    ``mae_vs_raw`` and ``mae_over_floor`` are the two numbers worth reading
    first: whether the correction beat doing nothing, and how much of what
    remains is the irreducible year-to-year spread.  ``mae_floor`` is the MAE a
    climatology-only prediction achieves and ``rmse_floor`` the RMSE, so each
    ratio compares like with like — see :func:`observed_percentile_sd`.
    """
    rows = []
    for (method, season, q), grp in long.groupby(
        ["method", "season", "percentile"], observed=True
    ):
        ok = grp[np.isfinite(grp["y"]) & np.isfinite(grp["yhat"])]
        if len(ok) < 3:
            continue
        m = combination_metrics(y=ok["y"].to_numpy(), x=ok["yhat"].to_numpy())

        per_fold = []
        for _, f in ok.groupby("syear", observed=True):
            if len(f) >= 3:
                per_fold.append(combination_metrics(
                    y=f["y"].to_numpy(), x=f["yhat"].to_numpy()
                ))
        rows.append({
            "method": method, "season": season, "percentile": int(q),
            "n": m["n"],
            "mae": m["mae"], "rmse": m["rmse"],
            "bias": m["bias"], "bias_sd": m["bias_sd"],
            "pearson_r": m["pearson_r"], "r2": m["r2"],
            "ols_slope": m["ols_slope"],
            "pearson_r_foldmean": float(np.mean([f["pearson_r"] for f in per_fold]))
            if per_fold else np.nan,
            "r2_foldmean": float(np.mean([f["r2"] for f in per_fold]))
            if per_fold else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    fl = (floor.groupby(["season", "percentile"], observed=True)
          .agg(mae_floor=("obs_mad", "mean"), rmse_floor=("obs_sd", "mean"))
          .reset_index())
    out = out.merge(fl, on=["season", "percentile"], how="left")
    # Compare like with like: MAE against the mean absolute deviation a
    # climatology-only prediction achieves, RMSE against the standard
    # deviation.  Using the SD for both would understate MAE by about a fifth.
    out["mae_over_floor"] = out["mae"] / out["mae_floor"]
    out["rmse_over_floor"] = out["rmse"] / out["rmse_floor"]
    out["mae_excess"] = out["mae"] - out["mae_floor"]

    raw = (out[out["method"] == RAW_METHOD]
           .set_index(["season", "percentile"])["mae"])
    if len(raw):
        key = list(zip(out["season"], out["percentile"]))
        base = np.array([raw.get(k, np.nan) for k in key], dtype=float)
        out["mae_raw"] = base
        out["mae_vs_raw"] = out["mae"] - base
        out["skill_vs_raw"] = out["mae"] / base
    else:
        log.warning(
            "No %r rows found; skill relative to the uncorrected predictor "
            "cannot be computed.", RAW_METHOD,
        )

    if diag is not None and len(diag):
        d = diag.groupby(["method", "season"], observed=True).agg(
            frac_nonmono=("n_nonmono", lambda s: float((s > 0).mean())),
            frac_failed=("converged", lambda s: float((~s.astype(bool)).mean())),
            frac_extrap=(
                "n_extrap_lo",
                lambda s: np.nan,   # replaced below; needs both columns
            ),
        ).reset_index()
        ex = diag.assign(
            _any=(diag["n_extrap_lo"] + diag["n_extrap_hi"]) > 0
        ).groupby(["method", "season"], observed=True)["_any"].mean()
        d["frac_extrap"] = d.set_index(["method", "season"]).index.map(ex)
        out = out.merge(d, on=["method", "season"], how="left")

    out["season"] = pd.Categorical(out["season"], categories=list(SEASONS), ordered=True)
    return out.sort_values(["method", "season", "percentile"]).reset_index(drop=True)


def daily_moments(daily: dict, pixels: pd.DataFrame) -> pd.DataFrame:
    """Skewness and excess kurtosis of the observed daily values per pixel and season.

    This turns the normality assumption behind ``NormalQM`` into a measurement.
    The literature says daily surface temperature is close to Gaussian in the
    body of the distribution but measurably non-Gaussian in the tails, with the
    departure depending on season (Ruff & Neelin 2012; Perron & Sura 2013), so
    it is worth checking on this domain rather than assuming.

    It also bears on the choice of a seasonal window: pooled over a whole year
    the daily distribution is broad and bimodal because of the annual cycle, so
    nowhere near normal.  Within a season it is unimodal.
    """
    from scipy import stats

    rows = []
    for season, ds in daily.items():
        y = ds.y.astype(np.float64)
        rows.append(pd.DataFrame({
            "pix_id": pixels["pix_id"].to_numpy(np.int32),
            "lat": pixels["lat"].to_numpy(np.float32),
            "lon": pixels["lon"].to_numpy(np.float32),
            "season": season,
            "skew": stats.skew(y, axis=0, nan_policy="omit").astype(np.float32),
            "kurtosis": stats.kurtosis(y, axis=0, nan_policy="omit").astype(np.float32),
            "sd": np.nanstd(y, axis=0, ddof=1).astype(np.float32),
            "n_days": np.int16(y.shape[0]),
        }))
    out = pd.concat(rows, ignore_index=True)
    out["season"] = pd.Categorical(out["season"], categories=list(SEASONS), ordered=True)
    return out
