"""
qm_metrics.py — Skill metrics and aggregates for the quantile-mapping baseline.

Consumes the wide percentile tables written by
:func:`src.quantile_windows.build_percentile_table` (one parquet per
predictor x window scheme) and produces four compact aggregate tables that the
notebook and the dashboard read instead of the large per-pixel files:

============================  ==========================================
``metrics_summary.parquet``   one row per (predictor, scheme, percentile)
``per_pixel_bias.parquet``    one row per (predictor, scheme, pct, pixel)
``per_window_bias.parquet``   one row per (predictor, scheme, pct, window)
``elev_bias.parquet``         one row per (predictor, scheme, pct, elev bin)
============================  ==========================================

Sign convention
---------------
``bias = mean(x - y)`` — predictor minus observation, so a positive bias means
the predictor is too warm.  This follows the project convention (see
``guidelines/analysis_guidelines.md`` §9) and is the negative of the ``bias``
key returned by :func:`src.visualization.compute_regression_metrics`, which is
observed minus predicted.

Typical usage
-------------
from src import qm_metrics as qmm

qmm.build_all_aggregates(qm_dir=DATA_DIR / "cache" / "qm")
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .quantile_windows import PERCENTILES, SCHEMES
from .visualization import compute_regression_metrics

__all__ = [
    "METRIC_FILES",
    "pct_table_path",
    "load_pct_table",
    "slice_percentile",
    "combination_metrics",
    "metrics_table",
    "per_pixel_bias",
    "per_window_bias",
    "elevation_binned_bias",
    "sea_fraction_binned_bias",
    "build_all_aggregates",
]

log = logging.getLogger(__name__)

#: Aggregate file names written by :func:`build_all_aggregates`.
METRIC_FILES = {
    "metrics": "metrics_summary.parquet",
    "pixel": "per_pixel_bias.parquet",
    "window": "per_window_bias.parquet",
    "elev": "elev_bias.parquet",
    "sea": "sea_bias.parquet",
}


# ---------------------------------------------------------------------------
# Table access
# ---------------------------------------------------------------------------

def pct_table_path(qm_dir: str | Path, predictor: str, scheme: str) -> Path:
    """Return the parquet path for one (predictor, scheme) percentile table."""
    return Path(qm_dir) / f"pct_{predictor}_{scheme}.parquet"


def load_pct_table(
    qm_dir: str | Path,
    predictor: str,
    scheme: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read one percentile table, optionally only a subset of columns.

    Parameters
    ----------
    qm_dir : str or Path
        Directory holding the ``pct_*.parquet`` files.
    predictor : str
        Predictor name, e.g. ``"bilinear"``.
    scheme : str
        Window scheme, e.g. ``"quarter"``.
    columns : list of str, optional
        Column subset to read.  Restricting columns is the main way to keep
        the 14-day tables (~2 M rows) cheap to load.

    Returns
    -------
    pd.DataFrame
    """
    path = pct_table_path(qm_dir, predictor, scheme)
    if not path.exists():
        raise FileNotFoundError(
            f"Percentile table not found: {path}. "
            "Run scripts/run_quantile_mapping.py first."
        )
    return pd.read_parquet(path, columns=columns)


def slice_percentile(df: pd.DataFrame, q: int) -> pd.DataFrame:
    """Reduce a wide percentile table to one percentile in long form.

    Parameters
    ----------
    df : pd.DataFrame
        Wide table with ``y_p{q}`` / ``x_p{q}`` columns.
    q : int
        Percentile to extract.

    Returns
    -------
    pd.DataFrame
        The table's identifying columns plus ``y``, ``x`` and
        ``bias`` (= ``x - y``).  The ``y_p*`` / ``x_p*`` columns are dropped.
    """
    keep = [c for c in df.columns if not c.startswith(("y_p", "x_p"))]
    out = df[keep].copy()
    out["y"] = df[f"y_p{q}"].to_numpy()
    out["x"] = df[f"x_p{q}"].to_numpy()
    out["bias"] = out["x"] - out["y"]
    return out


# ---------------------------------------------------------------------------
# Metrics for one (predictor, scheme, percentile) combination
# ---------------------------------------------------------------------------

def combination_metrics(y: np.ndarray, x: np.ndarray) -> dict:
    """Skill of predictor percentiles *x* against observed percentiles *y*.

    Parameters
    ----------
    y : np.ndarray
        Observed ERA5-Land percentile values (°C).
    x : np.ndarray
        Predictor percentile values (°C) at the same pixel and window.

    Returns
    -------
    dict
        ``n``, ``bias`` (mean x − y), ``bias_sd`` (SD of x − y), ``mae``,
        ``rmse``, ``pearson_r``, ``r2``, ``ols_slope``, ``ols_intercept``.
        ``r2`` and ``rmse`` describe the predictor used *as is* (no fitting);
        ``ols_slope`` / ``ols_intercept`` come from regressing y on x and say
        how a linear correction would have to be shaped.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    good = np.isfinite(y) & np.isfinite(x)
    y, x = y[good], x[good]

    if len(y) < 3:
        return dict(
            n=int(len(y)), bias=np.nan, bias_sd=np.nan, mae=np.nan,
            rmse=np.nan, pearson_r=np.nan, r2=np.nan,
            ols_slope=np.nan, ols_intercept=np.nan,
        )

    base = compute_regression_metrics(y, x)
    diff = x - y
    slope, intercept = np.polyfit(x, y, 1)

    return dict(
        n=int(len(y)),
        bias=float(diff.mean()),          # predictor minus observation
        bias_sd=float(diff.std(ddof=1)),
        mae=float(base["mae"]),
        rmse=float(base["rmse"]),
        pearson_r=float(base["pearson_r"]),
        r2=float(base["r2"]),
        ols_slope=float(slope),
        ols_intercept=float(intercept),
    )


def metrics_table(
    qm_dir: str | Path,
    predictors: list[str],
    schemes: list[str] = None,
    percentiles: tuple | list = PERCENTILES,
) -> pd.DataFrame:
    """Compute metrics for every (predictor, scheme, percentile) combination.

    Parameters
    ----------
    qm_dir : str or Path
        Directory holding the percentile tables.
    predictors : list of str
        Predictor names to include.
    schemes : list of str, optional
        Window schemes; defaults to :data:`src.quantile_windows.SCHEMES`.
    percentiles : sequence of int
        Percentiles to score.

    Returns
    -------
    pd.DataFrame
        One row per combination, columns ``predictor``, ``scheme``,
        ``percentile`` plus the keys of :func:`combination_metrics`.
    """
    schemes = list(schemes or SCHEMES)
    rows = []
    for predictor in predictors:
        for scheme in schemes:
            cols = ["n_days"] + [f"{p}_p{q}" for q in percentiles for p in ("y", "x")]
            df = load_pct_table(qm_dir, predictor, scheme, columns=cols)
            for q in percentiles:
                m = combination_metrics(df[f"y_p{q}"].to_numpy(),
                                        df[f"x_p{q}"].to_numpy())
                rows.append(
                    dict(predictor=predictor, scheme=scheme, percentile=q,
                         mean_window_days=float(df["n_days"].mean()), **m)
                )
            del df
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregates by pixel, window, elevation and sea fraction
# ---------------------------------------------------------------------------

def per_pixel_bias(df: pd.DataFrame, q: int) -> pd.DataFrame:
    """Mean and absolute bias per pixel, averaged over all windows.

    Parameters
    ----------
    df : pd.DataFrame
        Wide percentile table for one (predictor, scheme).
    q : int
        Percentile.

    Returns
    -------
    pd.DataFrame
        Columns ``lat``, ``lon``, ``mean_bias``, ``mae``, ``bias_sd``, ``n``.
    """
    diff = df[f"x_p{q}"].to_numpy() - df[f"y_p{q}"].to_numpy()
    work = pd.DataFrame(
        {"lat": df["lat"].to_numpy(), "lon": df["lon"].to_numpy(), "bias": diff}
    )
    out = work.groupby(["lat", "lon"], observed=True)["bias"].agg(
        mean_bias="mean", bias_sd="std", n="size"
    )
    out["mae"] = (
        work.assign(abs_bias=work["bias"].abs())
        .groupby(["lat", "lon"], observed=True)["abs_bias"].mean()
    )
    return out.reset_index()


def per_window_bias(df: pd.DataFrame, q: int) -> pd.DataFrame:
    """Domain-mean bias per window, for looking at seasonality and drift.

    Returns
    -------
    pd.DataFrame
        Columns ``window_id``, ``window_label``, ``window_start``,
        ``mean_bias``, ``mae``, ``bias_sd``, ``n``.
    """
    diff = df[f"x_p{q}"].to_numpy() - df[f"y_p{q}"].to_numpy()
    work = pd.DataFrame(
        {
            "window_id": df["window_id"].to_numpy(),
            "window_label": df["window_label"].astype(str).to_numpy(),
            "window_start": df["window_start"].to_numpy(),
            "bias": diff,
        }
    )
    grouped = work.groupby(["window_id", "window_label", "window_start"],
                           observed=True)
    out = grouped["bias"].agg(mean_bias="mean", bias_sd="std", n="size")
    out["mae"] = grouped["bias"].apply(lambda s: s.abs().mean())
    return out.reset_index().sort_values("window_id").reset_index(drop=True)


def _binned_bias(
    df: pd.DataFrame,
    q: int,
    by: str,
    n_bins: int,
    quantile_bins: bool,
    per_window: bool = False,
) -> pd.DataFrame:
    """Mean and spread of the bias inside bins of a static per-pixel column.

    Bin edges are always computed from every row of *df*, because *by* is a
    static property of a pixel (elevation, sub-grid terrain, sea fraction) and
    so takes the same set of values in every window.  Fixing the edges this way
    keeps the bins identical across windows, which is what makes per-window
    results comparable with each other.

    Parameters
    ----------
    df : pd.DataFrame
        Wide percentile table for one (predictor, scheme).
    q : int
        Percentile.
    by : str
        Static per-pixel column to bin on.
    n_bins : int
        Number of bins.
    quantile_bins : bool
        Equal-count bins when True, equal-width when False.
    per_window : bool
        When False, every window is pooled into one set of bins, so
        ``bias_sd`` mixes pixel-to-pixel with window-to-window variation.
        When True, each window is binned separately, so ``bias_sd`` is the
        spatial spread across the pixels of that window alone.

    Returns
    -------
    pd.DataFrame
        Columns ``bin_low``, ``bin_high``, ``bin_mid``, ``mean_bias``,
        ``bias_sd``, ``mae``, ``n``, plus ``window_label``.  Pooled rows carry
        the sentinel label ``"ALL"``.
    """
    cols = ["window_label", "bin_low", "bin_high", "bin_mid",
            "mean_bias", "bias_sd", "mae", "n"]
    if by not in df.columns:
        return pd.DataFrame(columns=cols)

    vals = df[by].to_numpy(dtype=float)
    diff = df[f"x_p{q}"].to_numpy(dtype=float) - df[f"y_p{q}"].to_numpy(dtype=float)
    good = np.isfinite(vals) & np.isfinite(diff)
    if not good.any():
        return pd.DataFrame(columns=cols)

    # --- bin edges, from all rows ------------------------------------------
    v_all = vals[good]
    if quantile_bins:
        edges = np.unique(np.quantile(v_all, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(v_all.min(), v_all.max(), n_bins + 1)
    if len(edges) < 2:
        edges = np.array([v_all.min(), v_all.min() + 1.0])

    which = np.clip(np.digitize(vals, edges[1:-1]), 0, len(edges) - 2)

    work = pd.DataFrame({"bin": which[good], "bias": diff[good]})
    keys = ["bin"]
    if per_window:
        work["window_id"] = df["window_id"].to_numpy()[good]
        work["window_label"] = df["window_label"].astype(str).to_numpy()[good]
        keys = ["window_id", "window_label", "bin"]

    grouped = work.groupby(keys, observed=True)["bias"]
    out = grouped.agg(mean_bias="mean", bias_sd="std", n="size").reset_index()
    out["mae"] = grouped.apply(lambda t: t.abs().mean()).to_numpy()

    out["bin_low"] = edges[out["bin"].to_numpy()]
    out["bin_high"] = edges[out["bin"].to_numpy() + 1]
    out["bin_mid"] = 0.5 * (out["bin_low"] + out["bin_high"])
    if not per_window:
        out["window_label"] = "ALL"
    out = out.drop(columns=["bin"] + (["window_id"] if per_window else []))
    return out[cols]


def elevation_binned_bias(
    df: pd.DataFrame,
    q: int,
    n_bins: int = 10,
    per_window: bool = False,
) -> pd.DataFrame:
    """Mean bias inside equal-count elevation bins.

    Equal-count (quantile) bins are used because the domain's elevation
    distribution is strongly skewed toward low ground.  Set *per_window* to
    get one set of bins per window, whose ``bias_sd`` is then purely the
    spatial spread across that window's pixels.
    """
    return _binned_bias(df, q, "elevation", n_bins,
                        quantile_bins=True, per_window=per_window)


def sea_fraction_binned_bias(
    df: pd.DataFrame,
    q: int,
    n_bins: int = 5,
    per_window: bool = False,
) -> pd.DataFrame:
    """Mean bias inside equal-width sea-fraction bins of the parent coarse cell."""
    return _binned_bias(df, q, "sea_fraction", n_bins,
                        quantile_bins=False, per_window=per_window)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_all_aggregates(
    qm_dir: str | Path,
    predictors: list[str],
    schemes: list[str] = None,
    percentiles: tuple | list = PERCENTILES,
    n_elev_bins: int = 10,
    n_sea_bins: int = 5,
) -> dict[str, pd.DataFrame]:
    """Compute and write every aggregate table under *qm_dir*.

    Each percentile table is read once and all aggregates for it are derived
    in that pass, so the large 14-day tables are never held twice.

    Parameters
    ----------
    qm_dir : str or Path
        Directory holding ``pct_*.parquet``; the aggregates are written here.
    predictors : list of str
        Predictor names.
    schemes : list of str, optional
        Window schemes; defaults to :data:`src.quantile_windows.SCHEMES`.
    percentiles : sequence of int
        Percentiles to score.
    n_elev_bins, n_sea_bins : int
        Bin counts for the elevation and sea-fraction diagnostics.

    Returns
    -------
    dict of str to pd.DataFrame
        Keyed as in :data:`METRIC_FILES`.
    """
    qm_dir = Path(qm_dir)
    schemes = list(schemes or SCHEMES)

    metrics_rows, pixel_parts, window_parts = [], [], []
    elev_parts, sea_parts = [], []

    for predictor in predictors:
        for scheme in schemes:
            df = load_pct_table(qm_dir, predictor, scheme)
            print(f"  · aggregating {predictor} / {scheme} "
                  f"({len(df):,} rows)", flush=True)
            tag = dict(predictor=predictor, scheme=scheme)

            for q in percentiles:
                metrics_rows.append(
                    dict(
                        **tag, percentile=q,
                        mean_window_days=float(df["n_days"].mean()),
                        **combination_metrics(df[f"y_p{q}"].to_numpy(),
                                              df[f"x_p{q}"].to_numpy()),
                    )
                )
                pixel_parts.append(
                    per_pixel_bias(df, q).assign(**tag, percentile=q)
                )
                window_parts.append(
                    per_window_bias(df, q).assign(**tag, percentile=q)
                )
                # Both the pooled view (window_label "ALL") and one set of
                # bins per window. The per-window rows are what a single-window
                # figure needs, because their bias_sd is the spatial spread
                # across that window's pixels rather than a mix of spatial and
                # seasonal variation.
                for per_window in (False, True):
                    elev_parts.append(
                        elevation_binned_bias(df, q, n_elev_bins, per_window)
                        .assign(**tag, percentile=q)
                    )
                    sea_parts.append(
                        sea_fraction_binned_bias(df, q, n_sea_bins, per_window)
                        .assign(**tag, percentile=q)
                    )
            del df

    out = {
        "metrics": pd.DataFrame(metrics_rows),
        "pixel": pd.concat(pixel_parts, ignore_index=True),
        "window": pd.concat(window_parts, ignore_index=True),
        "elev": pd.concat(elev_parts, ignore_index=True),
        "sea": pd.concat(sea_parts, ignore_index=True),
    }

    for key, frame in out.items():
        path = qm_dir / METRIC_FILES[key]
        frame.to_parquet(path, index=False)
        print(f"  ✓ {path.name}  ({len(frame):,} rows)", flush=True)

    return out
