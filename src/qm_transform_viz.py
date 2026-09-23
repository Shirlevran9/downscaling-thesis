"""
qm_transform_viz.py — Figures for the cross-validated quantile-mapping transforms.

Companion to :mod:`src.qm_visualization`, which serves the older
predictor-comparison grid.  A separate module rather than new functions there
because that module's figures are keyed on ``predictor`` and ``scheme`` columns
and labelled from ``PREDICTOR_LABELS``/``SCHEME_LABELS``; reusing them by
aliasing method to predictor and season to scheme would mislabel every axis.

All project figure standards are inherited by importing the shared helpers
rather than re-implementing them: degree-formatted map ticks, geographic aspect
correction, ``20°`` temperature ticks, the diverging colour map for anomalies,
and every font size from :mod:`src.vis_constants`.

Sign convention: ``bias = yhat - y`` (prediction minus observation), so a
positive bias means the corrected predictor is too warm.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from . import vis_constants as VC
from .qm_nodes import SEASONS
from .qm_visualization import (
    _BIAS_CMAP,
    _DEFAULT_REGION,
    _grid_from_points,
    _save,
    _symmetric_limit,
)
from .visualization import _geographic_aspect, _temp_formatter, apply_map_formatting

__all__ = [
    "METHOD_LABELS",
    "METHOD_FAMILIES",
    "SEASON_LABELS",
    "plot_error_map_panels",
    "plot_method_heatmap",
    "plot_method_bars",
    "plot_transform_curves",
    "plot_moments_maps",
]

log = logging.getLogger(__name__)

#: Display names, in the order results should be reported.
METHOD_LABELS = {
    "raw": "No correction",
    "normal": "Normal",
    "linear": "Linear",
    "poly": "Polynomial",
    "quant": "QUANT",
    "rquant": "RQUANT",
    "ssplin": "SSPLIN",
}

#: The family each method belongs to, after Gudmundsson et al. (2012).
METHOD_FAMILIES = {
    "raw": "none",
    "normal": "distribution derived",
    "linear": "parametric",
    "poly": "parametric",
    "quant": "non-parametric",
    "rquant": "non-parametric",
    "ssplin": "non-parametric",
}

#: Month spans, so a reader need not decode the acronym.
SEASON_LABELS = {
    "DJF": "Dec–Feb",
    "MAM": "Mar–May",
    "JJA": "Jun–Aug",
    "SON": "Sep–Nov",
}

#: Colour map for a strictly positive magnitude, per the project standards.
_MAG_CMAP = "YlOrRd"

#: Metrics that are signed and so need a diverging scale centred on zero.
_DIVERGING = {"mean_bias", "bias", "diff"}


def _metric_label(metric: str) -> str:
    return {
        "mae": "MAE (°C)",
        "rmse": "RMSE (°C)",
        "mean_bias": "Bias, prediction − observation (°C)",
        "bias": "Bias, prediction − observation (°C)",
        "bias_sd": "SD of bias (°C)",
        "obs_sd": "Year-to-year SD of the observation (°C)",
        "obs_mad": "Year-to-year mean absolute deviation (°C)",
        "mae_floor": "Climatology-only MAE (°C)",
        "mae_over_floor": "MAE / climatology-only MAE",
        "mae_excess": "MAE above the climatology floor (°C)",
        "skew": "Skewness",
        "kurtosis": "Excess kurtosis",
    }.get(metric, metric)


# ---------------------------------------------------------------------------
# Error maps: season x percentile
# ---------------------------------------------------------------------------

def plot_error_map_panels(
    pixel_df: pd.DataFrame,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    method: str,
    metric: str = "mean_bias",
    seasons: Sequence[str] = SEASONS,
    percentiles: Sequence[int] = (5, 25, 50, 75, 90),
    region: dict | None = None,
    shared_scale: bool = True,
    coarse_lats: np.ndarray | None = None,
    coarse_lons: np.ndarray | None = None,
    suptitle: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Per-pixel error maps, one row per season and one column per percentile.

    Parameters
    ----------
    pixel_df : pd.DataFrame
        ``qq_per_pixel_error.parquet``: columns ``method``, ``season``,
        ``percentile``, ``lat``, ``lon`` and the metric.
    method : str
        Which method to show; one key of :data:`METHOD_LABELS`.
    metric : str
        Column to map.  Signed metrics get a diverging scale centred on zero,
        magnitudes get a sequential one starting at zero.
    shared_scale : bool
        One colour scale across all panels, so the seasons are comparable.  Set
        ``False`` to give each panel its own range, which shows spatial pattern
        at the cost of comparability.
    """
    region = region or _DEFAULT_REGION
    sub = pixel_df[(pixel_df["method"] == method)
                   & pixel_df["season"].isin(list(seasons))
                   & pixel_df["percentile"].isin(list(percentiles))]
    if sub.empty:
        raise ValueError(f"No rows for method={method!r} with that metric.")

    diverging = metric in _DIVERGING
    if shared_scale:
        if diverging:
            vmax = _symmetric_limit(sub[metric].to_numpy())
            vmin = -vmax
        else:
            vmin = 0.0
            vmax = float(np.nanpercentile(sub[metric].to_numpy(), 98))

    nrows, ncols = len(seasons), len(percentiles)
    aspect = _geographic_aspect(region)
    lon_span = region["east"] - region["west"]
    lat_span = region["north"] - region["south"]
    panel_w = 2.6
    panel_h = panel_w * (lat_span / lon_span) * aspect
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_w * ncols + 1.8, panel_h * nrows + 1.2),
        squeeze=False,
    )

    mesh = None
    for i, season in enumerate(seasons):
        for j, q in enumerate(percentiles):
            ax = axes[i][j]
            cell = sub[(sub["season"] == season) & (sub["percentile"] == q)]
            grid = _grid_from_points(
                cell["lat"].to_numpy(), cell["lon"].to_numpy(),
                cell[metric].to_numpy(), era5_lats, era5_lons,
            )
            if not shared_scale:
                if diverging:
                    vmax = _symmetric_limit(grid)
                    vmin = -vmax
                else:
                    vmin, vmax = 0.0, float(np.nanpercentile(grid, 98))
            mesh = ax.pcolormesh(
                era5_lons, era5_lats, grid, shading="nearest",
                cmap=_BIAS_CMAP if diverging else _MAG_CMAP,
                vmin=vmin, vmax=vmax,
            )
            apply_map_formatting(ax, region)
            if coarse_lats is not None and coarse_lons is not None:
                from .visualization import _overlay_coarse_grid
                _overlay_coarse_grid(ax, coarse_lats, coarse_lons)

            if i == 0:
                ax.set_title(f"P{q}", fontsize=VC.TITLE_FONT_SIZE - 2)
            if j == 0:
                ax.set_ylabel(
                    f"{season}\n{SEASON_LABELS.get(season, '')}",
                    fontsize=VC.LABEL_FONT_SIZE,
                )
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if i < nrows - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])

    if suptitle is None:
        suptitle = (f"{METHOD_LABELS.get(method, method)}: "
                    f"{_metric_label(metric).split(' (')[0].lower()} by season and percentile")
    fig.suptitle(suptitle, fontsize=VC.TITLE_FONT_SIZE, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.9, 0.99))

    # Centre the bar over the middle of the grid rather than its full height:
    # on a 4-row figure of tall geographic panels a full-height bar is metres
    # long and reads as a page border.
    cax = fig.add_axes([0.915, 0.34, 0.014, 0.32])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(_metric_label(metric), fontsize=VC.CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=VC.TICK_FONT_SIZE)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Method comparison
# ---------------------------------------------------------------------------

def plot_method_heatmap(
    summary_df: pd.DataFrame,
    metric: str = "mae",
    methods: Sequence[str] | None = None,
    seasons: Sequence[str] = SEASONS,
    percentiles: Sequence[int] = (5, 25, 50, 75, 90),
    show_floor: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Methods down the rows, (season, percentile) across the columns.

    With *show_floor* the between-year standard deviation of the observation is
    appended as a final row.  That row is the irreducible error: a method cannot
    be expected to beat it, and reading the table without it invites treating a
    number near the floor as a poor result.
    """
    methods = list(methods) if methods else [
        m for m in METHOD_LABELS if m in set(summary_df["method"])
    ]
    sub = summary_df[summary_df["method"].isin(methods)
                     & summary_df["season"].isin(list(seasons))
                     & summary_df["percentile"].isin(list(percentiles))]

    piv = sub.pivot_table(index="method", columns=["season", "percentile"],
                          values=metric, observed=True)
    piv = piv.reindex(index=methods)
    piv = piv.reindex(columns=pd.MultiIndex.from_product(
        [list(seasons), list(percentiles)], names=["season", "percentile"]
    ))
    rows = [METHOD_LABELS.get(m, m) for m in piv.index]
    data = piv.to_numpy(dtype=float)

    floor_col = "mae_floor" if metric == "mae" else (
        "rmse_floor" if metric == "rmse" else None)
    if show_floor and floor_col and floor_col in sub.columns:
        fl = (sub.groupby(["season", "percentile"], observed=True)[floor_col]
              .mean())
        floor_row = np.array([fl.get((s, q), np.nan)
                              for s in seasons for q in percentiles])
        data = np.vstack([data, floor_row])
        rows = rows + ["Climatology floor"]

    # Cells sized so the annotation text is legible at the saved DPI; the
    # heatmap is the densest figure in the set, so it gets the most room.
    fig, ax = plt.subplots(figsize=(1.02 * data.shape[1] + 3.6,
                                    0.72 * data.shape[0] + 2.8))
    diverging = metric in _DIVERGING
    if diverging:
        vmax = _symmetric_limit(data)
        im = ax.imshow(data, cmap=_BIAS_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        # A robust upper limit, so one unusually hard column does not compress
        # the colour scale for every other cell.  Values above it still print.
        finite = data[np.isfinite(data)]
        im = ax.imshow(data, cmap=_MAG_CMAP, aspect="auto",
                       vmin=0.0, vmax=float(np.percentile(finite, 95)))

    # seaborn's whitegrid theme would otherwise draw gridlines over the cells.
    ax.grid(False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=VC.TICK_FONT_SIZE + 2)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(
        [f"P{q}" for _ in seasons for q in percentiles],
        fontsize=VC.TICK_FONT_SIZE + 1,
    )

    n_q = len(percentiles)
    for k, s in enumerate(seasons):
        # Season names go ABOVE the grid, in axes coordinates, so they cannot
        # land on top of the percentile tick labels below it.
        ax.text((k * n_q + (n_q - 1) / 2 + 0.5) / data.shape[1], 1.015,
                f"{s} ({SEASON_LABELS.get(s, '')})",
                ha="center", va="bottom", fontsize=VC.LABEL_FONT_SIZE + 2,
                transform=ax.transAxes)
        if k:
            ax.axvline(k * n_q - 0.5, color="white", lw=2.5)
    if show_floor and len(rows) > 1:
        ax.axhline(len(rows) - 1.5, color="white", lw=2.5)

    # Cell labels switch to white on dark fills, otherwise the values in the
    # hardest cells are unreadable dark-on-dark.
    norm, cmap = im.norm, im.cmap
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            r, g, b, _ = cmap(norm(v))
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=VC.ANNOT_FONT_SIZE + 4,
                    color="white" if luma < 0.5 else "#141a1e")

    ax.set_title(title or f"Cross-validated {_metric_label(metric)}",
                 fontsize=VC.TITLE_FONT_SIZE + 3, pad=34)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, extend="max")
    cbar.set_label(_metric_label(metric), fontsize=VC.CBAR_LABEL_SIZE + 2)
    cbar.ax.tick_params(labelsize=VC.TICK_FONT_SIZE + 1)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_method_bars(
    summary_df: pd.DataFrame,
    metric: str = "mae",
    methods: Sequence[str] | None = None,
    seasons: Sequence[str] = SEASONS,
    show_floor: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """One panel per season: the metric per method, averaged over percentiles.

    The floor is drawn as a horizontal line rather than another bar, because it
    is a property of the observations and not a competing method.
    """
    methods = list(methods) if methods else [
        m for m in METHOD_LABELS if m in set(summary_df["method"])
    ]
    fig, axes = plt.subplots(1, len(seasons), figsize=(3.3 * len(seasons), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, season in zip(axes, seasons):
        sub = summary_df[(summary_df["season"] == season)
                         & summary_df["method"].isin(methods)]
        vals = (sub.groupby("method", observed=True)[metric].mean()
                .reindex(methods))
        colours = ["#9e9e9e" if m == "raw" else "#2d6e8e" for m in methods]
        ax.bar(range(len(methods)), vals.to_numpy(), color=colours)
        floor_col = "mae_floor" if metric == "mae" else (
            "rmse_floor" if metric == "rmse" else None)
        if show_floor and floor_col and floor_col in sub.columns and len(sub):
            floor = float(sub[floor_col].mean())
            ax.axhline(floor, color="#b0532a", ls="--", lw=1.6,
                       label=f"Climatology floor ({floor:.2f} °C)")
            # Headroom first, so the legend does not sit on top of the tallest
            # bar.  The y axis is shared, so the widest panel sets it for all.
            top = float(np.nanmax(vals.to_numpy())) * 1.30
            ax.set_ylim(0.0, max(top, ax.get_ylim()[1]))
            ax.legend(fontsize=VC.LEGEND_FONT_SIZE - 1, loc="upper right",
                      framealpha=0.9)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=45, ha="right", fontsize=VC.TICK_FONT_SIZE - 1)
        ax.set_title(f"{season} ({SEASON_LABELS.get(season, '')})",
                     fontsize=VC.TITLE_FONT_SIZE - 2)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(_metric_label(metric), fontsize=VC.LABEL_FONT_SIZE)
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.suptitle(title or f"Cross-validated {_metric_label(metric).split(' (')[0]}"
                          " by method and season", fontsize=VC.TITLE_FONT_SIZE)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# The fitted curves themselves
# ---------------------------------------------------------------------------

def plot_transform_curves(
    nodes,
    transforms: dict,
    pix_col: int = 0,
    season: str = "",
    methods: Sequence[str] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """One panel per method: its fitted ``h`` over that pixel's q-q scatter.

    A single overlaid panel is unreadable, because on the absolute scale every
    transform sits within a few tenths of a degree of the others and the curves
    cover one another.  One column per method separates them, and each column
    carries its own modelled-value axis so the panels are self-contained.

    Each panel is annotated with the root-mean-square departure of ``h`` from
    the q-q nodes, which is the quantity the eye cannot judge at this scale and
    is what actually distinguishes the methods.

    Parameters
    ----------
    nodes : QQNodes
        The fold whose training curve is plotted.
    transforms : dict of str to QQTransform
        Already fitted on *nodes*.
    pix_col : int
        Which pixel column to draw.
    """
    methods = [m for m in (methods or METHOD_LABELS) if m in transforms and m != "raw"]
    x = nodes.x[:, pix_col].astype(float)
    y = nodes.y[:, pix_col].astype(float)
    grid = np.linspace(x.min(), x.max(), 300)

    ncols = len(methods)
    fig, axes = plt.subplots(1, ncols, figsize=(3.05 * ncols + 0.8, 4.3),
                             sharey=True, squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("tab10")
    lo = min(x.min(), y.min()) - 0.4
    hi = max(x.max(), y.max()) + 0.4

    for k, (ax, m) in enumerate(zip(axes, methods)):
        t = transforms[m]
        col = np.full((len(grid), t.n_pix_), np.nan)
        col[:, pix_col] = grid
        curve = t.predict(col)[:, pix_col]

        # RMS departure of h from the nodes it was fitted on.
        at_nodes = np.full((len(x), t.n_pix_), np.nan)
        at_nodes[:, pix_col] = x
        rms = float(np.sqrt(np.nanmean(
            (t.predict(at_nodes)[:, pix_col] - y) ** 2)))

        ax.plot([lo, hi], [lo, hi], "--", color="#9aa5ab", lw=1.0, zorder=1)
        ax.plot(x, y, "o", ms=3.2, color="#5a6a73", alpha=0.5, zorder=2,
                label="q–q nodes" if k == 0 else None)
        ax.plot(grid, curve, lw=2.2, color=cmap(k % 10), zorder=3)

        ax.set_title(METHOD_LABELS.get(m, m), fontsize=VC.TITLE_FONT_SIZE - 1)
        ax.set_xlabel("Modelled (°C)", fontsize=VC.LABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
        ax.tick_params(labelsize=VC.TICK_FONT_SIZE)
        ax.grid(alpha=0.3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.04, 0.955, f"RMS departure\n{rms:.3f} °C",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=VC.TICK_FONT_SIZE,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#d5dbde", alpha=0.9))

    axes[0].set_ylabel("Observed (°C)", fontsize=VC.LABEL_FONT_SIZE)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    axes[0].legend(fontsize=VC.LEGEND_FONT_SIZE - 1, loc="lower right")

    fig.suptitle(
        title or ("Fitted transforms at one pixel"
                  + (f", {season}" if season else "")),
        fontsize=VC.TITLE_FONT_SIZE + 1,
    )
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# The normality check
# ---------------------------------------------------------------------------

def plot_moments_maps(
    moments_df: pd.DataFrame,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    metrics: Sequence[str] = ("skew", "kurtosis"),
    seasons: Sequence[str] = SEASONS,
    region: dict | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Skewness and excess kurtosis of the observed daily values.

    Both are zero for a normal distribution, so these maps say how far the
    assumption behind the distribution-derived method is from holding, and
    where.  Drawn on a diverging scale centred on zero for that reason.
    """
    region = region or _DEFAULT_REGION
    aspect = _geographic_aspect(region)
    lon_span = region["east"] - region["west"]
    lat_span = region["north"] - region["south"]
    panel_w = 2.6
    panel_h = panel_w * (lat_span / lon_span) * aspect

    fig, axes = plt.subplots(
        len(metrics), len(seasons),
        figsize=(panel_w * len(seasons) + 2.0, panel_h * len(metrics) + 1.2),
        squeeze=False,
    )
    for i, metric in enumerate(metrics):
        vmax = _symmetric_limit(moments_df[metric].to_numpy())
        mesh = None
        for j, season in enumerate(seasons):
            ax = axes[i][j]
            cell = moments_df[moments_df["season"] == season]
            grid = _grid_from_points(
                cell["lat"].to_numpy(), cell["lon"].to_numpy(),
                cell[metric].to_numpy(), era5_lats, era5_lons,
            )
            mesh = ax.pcolormesh(era5_lons, era5_lats, grid, shading="nearest",
                                 cmap=_BIAS_CMAP, vmin=-vmax, vmax=vmax)
            apply_map_formatting(ax, region)
            if i == 0:
                ax.set_title(f"{season} ({SEASON_LABELS.get(season, '')})",
                             fontsize=VC.TITLE_FONT_SIZE - 3)
            if j == 0:
                ax.set_ylabel(_metric_label(metric), fontsize=VC.LABEL_FONT_SIZE)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if i < len(metrics) - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
        cbar = fig.colorbar(mesh, ax=axes[i].tolist(), fraction=0.02, pad=0.02)
        cbar.set_label(_metric_label(metric), fontsize=VC.CBAR_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=VC.TICK_FONT_SIZE)

    fig.suptitle(
        "Shape of the observed daily temperature distribution "
        "(both zero for a normal distribution)",
        fontsize=VC.TITLE_FONT_SIZE,
    )
    return _save(fig, save_path)
