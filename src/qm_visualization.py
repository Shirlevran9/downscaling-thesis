"""
qm_visualization.py — Figures for the quantile-mapping baseline.

Kept separate from :mod:`src.visualization`, which is already large.  All
project figure standards are inherited by importing that module's helpers
rather than re-implementing them: degree-formatted map ticks, geographic aspect
correction, ``20°`` temperature ticks, and every font size from
:mod:`src.vis_constants`.

Sign convention: ``bias = x - y`` (predictor minus observation), so positive
means the predictor's percentile is too warm.

Typical usage
-------------
from src import qm_visualization as qviz

qviz.plot_quantile_scatter(slice_df, q=50,
                           title="bilinear, 1999-Q2, p50",
                           save_path=PLOT_DIR / "fig_q03_scatter.png")
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from . import vis_constants as VC
from .visualization import (
    _geographic_aspect,
    _overlay_coarse_grid,
    _temp_formatter,
    apply_map_formatting,
)

__all__ = [
    "PREDICTOR_LABELS",
    "SCHEME_LABELS",
    "SCHEME_ADJECTIVES",
    "plot_quantile_scatter",
    "plot_percentile_panels",
    "plot_bias_map",
    "plot_bias_map_grid",
    "plot_metric_heatmap",
    "plot_metric_by_window_length",
    "plot_bias_vs_elevation",
    "plot_bias_by_sea_fraction",
    "plot_bias_by_window",
    "plot_predictor_climatology_comparison",
]

log = logging.getLogger(__name__)

#: Human-readable predictor names for axis labels and legends.
PREDICTOR_LABELS = {
    "knn4": "k-NN, k=4",
    "knn9": "k-NN, k=9",
    "bilinear": "Bilinear",
    "trilinear_fit": "Trilinear (fitted Γ)",
    "trilinear_fixed": "Trilinear (−6.5 K/km)",
}

#: Human-readable window-scheme names, for naming a single window length.
SCHEME_LABELS = {
    "14d": "14 days",
    "month": "1 month",
    "quarter": "1 quarter",
    "year": "1 year",
}

#: Adjectival form, for titles describing a pool of many windows. "1 quarter
#: windows" reads as though a single quarter were plotted; "quarterly windows"
#: does not.
SCHEME_ADJECTIVES = {
    "14d": "14-day",
    "month": "monthly",
    "quarter": "quarterly",
    "year": "annual",
}

_BIAS_CMAP = "RdBu_r"
_TEMP_CMAP = "RdYlBu_r"
_DEFAULT_REGION = {"south": 24, "north": 38, "west": 30, "east": 38}


def _save(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    """Write *fig* to disk at the project save DPI, if a path is given."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=VC.SAVE_DPI, bbox_inches="tight")
        log.info("Saved figure to %s", save_path)
    return fig


def _symmetric_limit(values: np.ndarray, pct: float = 98.0) -> float:
    """Robust symmetric colour limit for a diverging (anomaly) field."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(finite), pct)), 1e-6)


def _grid_from_points(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    """Scatter per-pixel values back onto a full 2-D grid, NaN elsewhere."""
    grid = np.full((len(target_lats), len(target_lons)), np.nan, dtype=float)
    lat_pos = {round(float(v), 4): i for i, v in enumerate(target_lats)}
    lon_pos = {round(float(v), 4): j for j, v in enumerate(target_lons)}
    ii = np.array([lat_pos.get(round(float(v), 4), -1) for v in lats])
    jj = np.array([lon_pos.get(round(float(v), 4), -1) for v in lons])
    ok = (ii >= 0) & (jj >= 0)
    grid[ii[ok], jj[ok]] = np.asarray(values, dtype=float)[ok]
    return grid


# ---------------------------------------------------------------------------
# Percentile scatter: X vs Y
# ---------------------------------------------------------------------------

def plot_quantile_scatter(
    df: pd.DataFrame,
    q: int,
    title: str = "",
    colour_by: str | None = "elevation",
    max_points: int = 40_000,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter the predictor percentile against the observed percentile.

    Each point is one (window, pixel) pair.  The 1:1 line marks a perfect
    match; the fitted line is the OLS regression of observed on predictor,
    which is the linear correction the data would ask for.

    Parameters
    ----------
    df : pd.DataFrame
        Wide percentile table (needs ``y_p{q}`` and ``x_p{q}``) or a long
        slice from :func:`src.qm_metrics.slice_percentile` (needs ``y``, ``x``).
    q : int
        Percentile being shown; used for the axis labels.
    title : str
        Panel title.
    colour_by : str or None
        Column used to colour the points (``"elevation"`` by default).  Pass
        ``None`` for a single-colour scatter.
    max_points : int
        Random subsample cap, so a 14-day table stays quick to render.
    ax : plt.Axes, optional
        Draw into an existing axes instead of creating a figure.
    save_path : str or Path, optional
        Where to write the figure.

    Returns
    -------
    plt.Figure
    """
    y = df["y"].to_numpy() if "y" in df.columns else df[f"y_p{q}"].to_numpy()
    x = df["x"].to_numpy() if "x" in df.columns else df[f"x_p{q}"].to_numpy()
    c = (
        df[colour_by].to_numpy()
        if colour_by is not None and colour_by in df.columns
        else None
    )

    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if c is not None:
        c = c[good]

    n_total = len(x)
    if n_total > max_points:
        rng = np.random.default_rng(42)
        pick = rng.choice(n_total, max_points, replace=False)
        xs, ys = x[pick], y[pick]
        cs = c[pick] if c is not None else None
    else:
        xs, ys, cs = x, y, c

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=VC.FIG_DPI)
    else:
        fig = ax.figure

    if cs is not None:
        # A perceptually ordered map, not "terrain": the points must be read
        # as a low-to-high ranking, and terrain's hard break at sea level
        # splits the Jordan Valley pixels off from the rest of the lowland.
        sc = ax.scatter(xs, ys, c=cs, s=3, alpha=0.35, cmap="viridis",
                        linewidths=0, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Elevation (m a.s.l.)", fontsize=VC.CBAR_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=VC.TICK_FONT_SIZE)
    else:
        ax.scatter(xs, ys, s=3, alpha=0.35, color="tab:blue",
                   linewidths=0, rasterized=True)

    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    pad = 0.03 * (hi - lo)
    span = np.array([lo - pad, hi + pad])

    ax.plot(span, span, color="black", lw=1.2, ls="--", label="1:1")

    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(span, slope * span + intercept, color="crimson", lw=1.6,
            label=f"OLS: y = {slope:.2f}x {intercept:+.2f}")

    bias = float(np.mean(x - y))
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    r = float(np.corrcoef(x, y)[0, 1])
    ax.text(
        0.03, 0.97,
        f"n = {n_total:,}\nbias = {bias:+.2f}°C\nRMSE = {rmse:.2f}°C\nr = {r:.3f}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=VC.LEGEND_FONT_SIZE,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.85),
    )

    ax.set_xlim(span)
    ax.set_ylim(span)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Predictor P{q} (°C)", fontsize=VC.LABEL_FONT_SIZE)
    ax.set_ylabel(f"ERA5-Land P{q} (°C)", fontsize=VC.LABEL_FONT_SIZE)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    ax.tick_params(labelsize=VC.TICK_FONT_SIZE)
    ax.legend(loc="lower right", fontsize=VC.LEGEND_FONT_SIZE, framealpha=0.8)
    if title:
        ax.set_title(title, fontsize=VC.TITLE_FONT_SIZE)

    if created:
        fig.tight_layout()
    return _save(fig, save_path)


def plot_percentile_panels(
    df: pd.DataFrame,
    percentiles: tuple | list = (5, 25, 50, 75, 90),
    suptitle: str = "",
    colour_by: str | None = "elevation",
    max_points: int = 20_000,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """One X-vs-Y scatter panel per percentile, on a shared layout.

    Parameters
    ----------
    df : pd.DataFrame
        Wide percentile table restricted to the window of interest.
    percentiles : sequence of int
        Percentiles to draw, one panel each.
    suptitle : str
        Figure-level title.
    colour_by : str or None
        Column used to colour points.
    max_points : int
        Subsample cap per panel.
    save_path : str or Path, optional

    Returns
    -------
    plt.Figure
    """
    n = len(percentiles)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.8), dpi=VC.FIG_DPI)
    axes = np.atleast_1d(axes)

    for k, (ax, q) in enumerate(zip(axes, percentiles)):
        plot_quantile_scatter(
            df, q, title=f"({chr(97 + k)}) P{q}",
            colour_by=colour_by if k == n - 1 else None,
            max_points=max_points, ax=ax,
        )
        if k > 0:
            ax.set_ylabel("")

    if suptitle:
        fig.suptitle(suptitle, fontsize=VC.TITLE_FONT_SIZE, y=1.02)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Spatial bias maps
# ---------------------------------------------------------------------------

def plot_bias_map(
    bias_df: pd.DataFrame,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    region: dict | None = None,
    value_col: str = "mean_bias",
    title: str = "",
    cbar_label: str = "Percentile bias, predictor − ERA5-Land (°C)",
    vmax: float | None = None,
    coarse_lats: np.ndarray | None = None,
    coarse_lons: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Map the mean percentile bias per fine-grid pixel.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Per-pixel table with ``lat``, ``lon`` and *value_col*, as returned by
        :func:`src.qm_metrics.per_pixel_bias`.
    era5_lats, era5_lons : np.ndarray
        Full ERA5-Land coordinate centres, used to rebuild the 2-D grid.
    region : dict, optional
        Domain box; defaults to the EMME domain.
    value_col : str
        Column to map.
    title : str
        Panel title.
    cbar_label : str
        Colourbar label including units.
    vmax : float, optional
        Symmetric colour limit; inferred from the 98th percentile of the
        absolute values when omitted.
    coarse_lats, coarse_lons : np.ndarray, optional
        CMIP6 cell centres; when given, the coarse-grid boundaries are drawn.
    ax : plt.Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    plt.Figure
    """
    region = region or _DEFAULT_REGION
    grid = _grid_from_points(
        bias_df["lat"].to_numpy(), bias_df["lon"].to_numpy(),
        bias_df[value_col].to_numpy(), era5_lats, era5_lons,
    )

    diverging = value_col in {"mean_bias", "bias", "diff"}
    if vmax is None:
        vmax = _symmetric_limit(grid)

    created = ax is None
    if created:
        aspect = _geographic_aspect(region)
        width = 5.5
        height = width * aspect * (region["north"] - region["south"]) / (
            region["east"] - region["west"]
        )
        fig, ax = plt.subplots(figsize=(width, min(height, 9.5)), dpi=VC.FIG_DPI)
    else:
        fig = ax.figure

    if diverging:
        mesh = ax.pcolormesh(era5_lons, era5_lats, grid, cmap=_BIAS_CMAP,
                             vmin=-vmax, vmax=vmax, shading="auto")
    else:
        mesh = ax.pcolormesh(era5_lons, era5_lats, grid, cmap="YlOrRd",
                             vmin=0, vmax=vmax, shading="auto")

    if coarse_lats is not None and coarse_lons is not None:
        _overlay_coarse_grid(ax, coarse_lats, coarse_lons)

    apply_map_formatting(ax, region)
    if title:
        ax.set_title(title, fontsize=VC.TITLE_FONT_SIZE)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=VC.CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=VC.TICK_FONT_SIZE)

    if created:
        fig.tight_layout()
    return _save(fig, save_path)


def plot_bias_map_grid(
    pixel_df: pd.DataFrame,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    predictors: list[str],
    scheme: str,
    percentile: int,
    region: dict | None = None,
    ncols: int = 3,
    suptitle: str = "",
    coarse_lats: np.ndarray | None = None,
    coarse_lons: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bias maps for several predictors on one shared colour scale.

    Parameters
    ----------
    pixel_df : pd.DataFrame
        The full ``per_pixel_bias`` aggregate, with ``predictor``, ``scheme``
        and ``percentile`` columns.
    era5_lats, era5_lons : np.ndarray
        ERA5-Land coordinate centres.
    predictors : list of str
        Predictors to draw, in panel order.
    scheme : str
        Window scheme to select.
    percentile : int
        Percentile to select.
    region : dict, optional
    ncols : int
        Panels per row.
    suptitle : str
    coarse_lats, coarse_lons : np.ndarray, optional
    save_path : str or Path, optional

    Returns
    -------
    plt.Figure
    """
    region = region or _DEFAULT_REGION
    sel = pixel_df[
        (pixel_df["scheme"] == scheme) & (pixel_df["percentile"] == percentile)
    ]
    vmax = _symmetric_limit(
        sel[sel["predictor"].isin(predictors)]["mean_bias"].to_numpy()
    )

    nrows = int(np.ceil(len(predictors) / ncols))
    aspect = _geographic_aspect(region)
    panel_h = 3.0 * aspect * (region["north"] - region["south"]) / (
        region["east"] - region["west"]
    )
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.9 * ncols, min(panel_h, 6.0) * nrows),
        dpi=VC.FIG_DPI, squeeze=False,
    )

    mesh = None
    for k, predictor in enumerate(predictors):
        ax = axes[k // ncols][k % ncols]
        part = sel[sel["predictor"] == predictor]
        grid = _grid_from_points(
            part["lat"].to_numpy(), part["lon"].to_numpy(),
            part["mean_bias"].to_numpy(), era5_lats, era5_lons,
        )
        mesh = ax.pcolormesh(era5_lons, era5_lats, grid, cmap=_BIAS_CMAP,
                             vmin=-vmax, vmax=vmax, shading="auto")
        if coarse_lats is not None and coarse_lons is not None:
            _overlay_coarse_grid(ax, coarse_lats, coarse_lons)
        apply_map_formatting(ax, region)
        mean_b = float(np.nanmean(part["mean_bias"].to_numpy()))
        ax.set_title(
            f"({chr(97 + k)}) {PREDICTOR_LABELS.get(predictor, predictor)}\n"
            f"domain mean {mean_b:+.2f}°C",
            fontsize=VC.TITLE_FONT_SIZE - 2,
        )
        if k % ncols > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

    for k in range(len(predictors), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.subplots_adjust(right=0.88, wspace=0.10, hspace=0.28)
    cax = fig.add_axes([0.90, 0.15, 0.018, 0.70])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Percentile bias, predictor − ERA5-Land (°C)",
                   fontsize=VC.CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=VC.TICK_FONT_SIZE)

    if suptitle:
        fig.suptitle(suptitle, fontsize=VC.TITLE_FONT_SIZE, y=0.98)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Metric summaries
# ---------------------------------------------------------------------------

def plot_metric_heatmap(
    metrics_df: pd.DataFrame,
    metric: str = "bias",
    schemes: list[str] | None = None,
    predictors: list[str] | None = None,
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Heatmap of one metric over predictors (rows) x scheme-percentile (cols).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The ``metrics_summary`` aggregate.
    metric : str
        Column to display (``"bias"``, ``"mae"``, ``"rmse"``, ``"pearson_r"``,
        ``"r2"``, ``"ols_slope"``).
    schemes, predictors : list of str, optional
        Restrict and order the axes.
    title : str
    save_path : str or Path, optional

    Returns
    -------
    plt.Figure
    """
    import seaborn as sns  # local import; seaborn styling is global

    df = metrics_df.copy()
    if schemes:
        df = df[df["scheme"].isin(schemes)]
    if predictors:
        df = df[df["predictor"].isin(predictors)]

    scheme_order = schemes or [s for s in SCHEME_LABELS if s in set(df["scheme"])]
    pred_order = predictors or [
        p for p in PREDICTOR_LABELS if p in set(df["predictor"])
    ]

    df["col"] = [
        f"{SCHEME_LABELS.get(s, s)}\nP{q}"
        for s, q in zip(df["scheme"], df["percentile"])
    ]
    col_order = [
        f"{SCHEME_LABELS.get(s, s)}\nP{q}"
        for s in scheme_order
        for q in sorted(df["percentile"].unique())
    ]

    mat = (
        df.pivot_table(index="predictor", columns="col", values=metric)
        .reindex(index=pred_order)
        .reindex(columns=[c for c in col_order if c in df["col"].values])
    )
    mat.index = [PREDICTOR_LABELS.get(i, i) for i in mat.index]

    diverging = metric in {"bias", "ols_intercept"}
    centre = 0.0 if diverging else None
    cmap = _BIAS_CMAP if diverging else ("YlOrRd" if metric in
                                         {"mae", "rmse", "bias_sd"} else "YlGnBu")

    # 20 columns (4 windows x 5 percentiles) is wide by necessity; keep the
    # per-cell width just large enough for the 8 pt annotations to stay legible.
    fig, ax = plt.subplots(
        figsize=(0.80 * mat.shape[1] + 3.0, 0.70 * mat.shape[0] + 2.4),
        dpi=VC.FIG_DPI,
    )
    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap=cmap, center=centre, ax=ax,
        annot_kws={"size": VC.ANNOT_FONT_SIZE},
        cbar_kws={"label": metric},
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("Distribution window and percentile", fontsize=VC.LABEL_FONT_SIZE)
    ax.set_ylabel("Predictor", fontsize=VC.LABEL_FONT_SIZE)
    ax.tick_params(labelsize=VC.TICK_FONT_SIZE)
    ax.set_title(title or f"{metric} by predictor, window and percentile",
                 fontsize=VC.TITLE_FONT_SIZE)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_metric_by_window_length(
    metrics_df: pd.DataFrame,
    metric: str = "mae",
    percentile: int = 50,
    predictors: list[str] | None = None,
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """One line per predictor showing a metric against window length.

    The x-axis is the mean number of days in a window, on a log scale, so the
    four schemes are spaced by their true length.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The ``metrics_summary`` aggregate.
    metric : str
        Metric column to plot.
    percentile : int
        Percentile to hold fixed.
    predictors : list of str, optional
    title : str
    save_path : str or Path, optional

    Returns
    -------
    plt.Figure
    """
    df = metrics_df[metrics_df["percentile"] == percentile]
    pred_order = predictors or [
        p for p in PREDICTOR_LABELS if p in set(df["predictor"])
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=VC.FIG_DPI)
    colours = plt.get_cmap("tab10")

    for i, predictor in enumerate(pred_order):
        part = df[df["predictor"] == predictor].sort_values("mean_window_days")
        ax.plot(part["mean_window_days"], part[metric], marker="o", lw=1.8,
                color=colours(i), label=PREDICTOR_LABELS.get(predictor, predictor))

    if metric == "bias":
        ax.axhline(0.0, color="black", lw=1.0, ls="--")

    ticks = sorted(df["mean_window_days"].unique())
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks], fontsize=VC.TICK_FONT_SIZE)
    ax.minorticks_off()
    ax.set_xlabel("Distribution window length (days)", fontsize=VC.LABEL_FONT_SIZE)
    ax.set_ylabel(f"{metric} (°C)" if metric in {"bias", "mae", "rmse", "bias_sd"}
                  else metric, fontsize=VC.LABEL_FONT_SIZE)
    ax.tick_params(labelsize=VC.TICK_FONT_SIZE)
    ax.legend(fontsize=VC.LEGEND_FONT_SIZE, framealpha=0.8)
    ax.set_title(title or f"{metric} vs window length at P{percentile}",
                 fontsize=VC.TITLE_FONT_SIZE)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Diagnostics: elevation, sea fraction, time
# ---------------------------------------------------------------------------

def plot_bias_vs_elevation(
    elev_df: pd.DataFrame,
    scheme: str,
    percentile: int,
    predictors: list[str] | None = None,
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Mean bias against elevation, one line per predictor.

    Bins hold equal pixel counts, so the spacing on the x-axis reflects the
    domain's skew toward low ground.  The shaded band is ±1 SD of the bias
    within each bin.

    Parameters
    ----------
    elev_df : pd.DataFrame
        The ``elev_bias`` aggregate.
    scheme : str
    percentile : int
    predictors : list of str, optional
    title : str
    save_path : str or Path, optional

    Returns
    -------
    plt.Figure
    """
    df = elev_df[
        (elev_df["scheme"] == scheme) & (elev_df["percentile"] == percentile)
    ]
    pred_order = predictors or [
        p for p in PREDICTOR_LABELS if p in set(df["predictor"])
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=VC.FIG_DPI)
    colours = plt.get_cmap("tab10")

    for i, predictor in enumerate(pred_order):
        part = df[df["predictor"] == predictor].sort_values("bin_mid")
        ax.plot(part["bin_mid"], part["mean_bias"], marker="o", lw=1.8,
                color=colours(i),
                label=PREDICTOR_LABELS.get(predictor, predictor))
        ax.fill_between(
            part["bin_mid"],
            part["mean_bias"] - part["bias_sd"],
            part["mean_bias"] + part["bias_sd"],
            color=colours(i), alpha=VC.CI_ALPHA, linewidth=0,
        )

    ax.axhline(0.0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("Elevation (m a.s.l.), equal-count bins",
                  fontsize=VC.LABEL_FONT_SIZE)
    ax.set_ylabel("Percentile bias (°C)", fontsize=VC.LABEL_FONT_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    ax.tick_params(labelsize=VC.TICK_FONT_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="0.5",
                                 alpha=min(VC.CI_ALPHA * 1.8, 0.7)))
    labels.append("Shaded: ±1 SD of bias within each bin")
    ax.legend(handles, labels, fontsize=VC.LEGEND_FONT_SIZE, framealpha=0.8)
    ax.set_title(
        title or f"Percentile bias by elevation — P{percentile}, pooled over "
                 f"all {SCHEME_ADJECTIVES.get(scheme, scheme)} windows",
        fontsize=VC.TITLE_FONT_SIZE,
    )
    fig.tight_layout()
    return _save(fig, save_path)


def plot_bias_by_sea_fraction(
    sea_df: pd.DataFrame,
    scheme: str,
    percentile: int,
    predictors: list[str] | None = None,
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Mean bias against the sea fraction of the parent CMIP6 cell.

    Parameters
    ----------
    sea_df : pd.DataFrame
        The ``sea_bias`` aggregate.
    scheme : str
    percentile : int
    predictors : list of str, optional
    title : str
    save_path : str or Path, optional

    Returns
    -------
    plt.Figure
    """
    df = sea_df[
        (sea_df["scheme"] == scheme) & (sea_df["percentile"] == percentile)
    ]
    pred_order = predictors or [
        p for p in PREDICTOR_LABELS if p in set(df["predictor"])
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=VC.FIG_DPI)
    colours = plt.get_cmap("tab10")
    width = 0.8 / max(len(pred_order), 1)

    bins = sorted(df["bin_mid"].unique())
    positions = np.arange(len(bins))

    for i, predictor in enumerate(pred_order):
        part = df[df["predictor"] == predictor].set_index("bin_mid").reindex(bins)
        ax.bar(positions + i * width - 0.4 + width / 2,
               part["mean_bias"].to_numpy(), width=width,
               color=colours(i), edgecolor="white", linewidth=0.4,
               label=PREDICTOR_LABELS.get(predictor, predictor))

    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{b:.2f}" for b in bins], fontsize=VC.TICK_FONT_SIZE)
    ax.set_xlabel("Sea fraction of the parent CMIP6 cell (bin centre)",
                  fontsize=VC.LABEL_FONT_SIZE)
    ax.set_ylabel("Percentile bias (°C)", fontsize=VC.LABEL_FONT_SIZE)
    ax.tick_params(labelsize=VC.TICK_FONT_SIZE)
    ax.legend(fontsize=VC.LEGEND_FONT_SIZE, framealpha=0.8, ncol=2)
    ax.set_title(
        title or f"Percentile bias by coarse-cell sea fraction — P{percentile}, "
                 f"pooled over all {SCHEME_ADJECTIVES.get(scheme, scheme)} windows",
        fontsize=VC.TITLE_FONT_SIZE,
    )
    fig.tight_layout()
    return _save(fig, save_path)


def plot_bias_by_window(
    window_df: pd.DataFrame,
    scheme: str,
    percentile: int,
    predictors: list[str] | None = None,
    title: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Domain-mean bias as a time series over windows.

    Parameters
    ----------
    window_df : pd.DataFrame
        The ``per_window_bias`` aggregate.
    scheme : str
    percentile : int
    predictors : list of str, optional
    title : str
    save_path : str or Path, optional

    Returns
    -------
    plt.Figure
    """
    import matplotlib.dates as mdates

    df = window_df[
        (window_df["scheme"] == scheme) & (window_df["percentile"] == percentile)
    ]
    pred_order = predictors or [
        p for p in PREDICTOR_LABELS if p in set(df["predictor"])
    ]

    fig, ax = plt.subplots(figsize=(11.0, 4.4), dpi=VC.FIG_DPI)
    colours = plt.get_cmap("tab10")

    for i, predictor in enumerate(pred_order):
        part = df[df["predictor"] == predictor].sort_values("window_start")
        ax.plot(pd.to_datetime(part["window_start"]), part["mean_bias"],
                lw=1.4, color=colours(i),
                label=PREDICTOR_LABELS.get(predictor, predictor))

    ax.axhline(0.0, color="black", lw=1.0, ls="--")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Window start", fontsize=VC.LABEL_FONT_SIZE)
    ax.set_ylabel("Domain-mean bias (°C)", fontsize=VC.LABEL_FONT_SIZE)
    ax.tick_params(labelsize=VC.TICK_FONT_SIZE)
    ax.legend(fontsize=VC.LEGEND_FONT_SIZE, framealpha=0.8, ncol=3)
    ax.set_title(
        title or f"Domain-mean percentile bias per window — "
                 f"{SCHEME_LABELS.get(scheme, scheme)}, P{percentile}",
        fontsize=VC.TITLE_FONT_SIZE,
    )
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Predictor fields
# ---------------------------------------------------------------------------

def plot_predictor_climatology_comparison(
    clim_fields: dict,
    era5_clim: np.ndarray,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    region: dict | None = None,
    coarse_lats: np.ndarray | None = None,
    coarse_lons: np.ndarray | None = None,
    suptitle: str = "Mean temperature climatology, 1990–1999",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Observed climatology plus each predictor's difference from it.

    Panel (a) shows the ERA5-Land mean field on an absolute temperature scale.
    The remaining panels show ``predictor − ERA5-Land`` on a shared diverging
    scale, so the predictors can be compared directly.

    Parameters
    ----------
    clim_fields : dict of str to np.ndarray
        Predictor name to 2-D time-mean field on the ERA5-Land grid.
    era5_clim : np.ndarray
        2-D ERA5-Land time-mean field (ocean pixels NaN).
    era5_lats, era5_lons : np.ndarray
        ERA5-Land coordinate centres.
    region : dict, optional
    coarse_lats, coarse_lons : np.ndarray, optional
    suptitle : str
    save_path : str or Path, optional

    Returns
    -------
    plt.Figure
    """
    region = region or _DEFAULT_REGION
    names = list(clim_fields)
    n_panels = len(names) + 1
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    aspect = _geographic_aspect(region)
    panel_h = 3.0 * aspect * (region["north"] - region["south"]) / (
        region["east"] - region["west"]
    )
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.9 * ncols, min(panel_h, 6.0) * nrows),
        dpi=VC.FIG_DPI, squeeze=False,
    )

    diffs = {n: clim_fields[n] - era5_clim for n in names}
    vmax = _symmetric_limit(np.concatenate([d.ravel() for d in diffs.values()]))

    ax0 = axes[0][0]
    m0 = ax0.pcolormesh(era5_lons, era5_lats, era5_clim, cmap=_TEMP_CMAP,
                        shading="auto")
    if coarse_lats is not None and coarse_lons is not None:
        _overlay_coarse_grid(ax0, coarse_lats, coarse_lons)
    apply_map_formatting(ax0, region)
    ax0.set_title("(a) ERA5-Land observed", fontsize=VC.TITLE_FONT_SIZE - 2)
    cb0 = fig.colorbar(m0, ax=ax0, fraction=0.046, pad=0.04)
    cb0.set_label("Mean T2M (°C)", fontsize=VC.CBAR_LABEL_SIZE)
    cb0.ax.set_yticklabels(
        [_temp_formatter(t, None) for t in cb0.get_ticks()],
        fontsize=VC.TICK_FONT_SIZE,
    )

    mesh = None
    for k, name in enumerate(names, start=1):
        ax = axes[k // ncols][k % ncols]
        mesh = ax.pcolormesh(era5_lons, era5_lats, diffs[name], cmap=_BIAS_CMAP,
                             vmin=-vmax, vmax=vmax, shading="auto")
        apply_map_formatting(ax, region)
        mean_d = float(np.nanmean(diffs[name]))
        ax.set_title(
            f"({chr(97 + k)}) {PREDICTOR_LABELS.get(name, name)} − ERA5-Land\n"
            f"domain mean {mean_d:+.2f}°C",
            fontsize=VC.TITLE_FONT_SIZE - 2,
        )
        if k % ncols > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

    for k in range(n_panels, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.subplots_adjust(right=0.88, wspace=0.16, hspace=0.30)
    cax = fig.add_axes([0.90, 0.15, 0.018, 0.70])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Difference from ERA5-Land (°C)", fontsize=VC.CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=VC.TICK_FONT_SIZE)

    fig.suptitle(suptitle, fontsize=VC.TITLE_FONT_SIZE, y=0.98)
    return _save(fig, save_path)
