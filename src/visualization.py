"""
visualization.py — Publication-quality figure functions for the downscaling project.

All spatial figures adhere to the following formatting standards:
  - Longitude tick labels: ``30°E``, ``32°E``, … (degree symbol + cardinal direction)
  - Latitude tick labels:  ``24°N``, ``26°N``, … (degree symbol + cardinal direction)
  - Aspect ratio:          Corrected for the latitude of the domain so that
                           one degree of latitude and one degree of longitude
                           span equal physical distances on screen.
  - Colorbars:             Labelled with variable name and units.
  - Figure captions:       Written as Markdown cells immediately after each
                           figure cell in the notebook.

Usage example
-------------
from src.visualization import plot_temperature_map, plot_side_by_side_maps
"""

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

__all__ = [
    "apply_map_formatting",
    "make_spatial_figure",
    "plot_temperature_map",
    "plot_side_by_side_maps",
    "plot_seasonal_maps",
    "plot_seasonal_comparison_maps",
    "plot_land_sea_mask",
    "plot_missing_fraction_map",
    "plot_missing_fraction_timeseries",
    "plot_domain_timeseries",
    "plot_domain_range_timeseries",
    "plot_monthly_climatology",
    "plot_temperature_distributions",
    "plot_temperature_percentiles",
    "plot_pixel_assignment_map",
    "plot_pixels_per_cell_heatmap",
    "plot_scatter_regression",
    "plot_residual_analysis",
    "compute_ols_residuals",
    "plot_residuals_by_sea_fraction",
    "plot_residuals_by_pixel_count",
    "plot_quarterly_warming_trend",
]

# ---------------------------------------------------------------------------
# Global style defaults
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
    }
)

# Default study region
_DEFAULT_REGION = {"south": 24, "north": 38, "west": 30, "east": 38}

# Temperature colormap (diverging for anomalies, sequential for climatology)
_TEMP_CMAP = "RdYlBu_r"
_MISSING_CMAP = "YlOrRd"


# ---------------------------------------------------------------------------
# Axis formatting helpers
# ---------------------------------------------------------------------------

def _lon_formatter(x, pos):
    """Format a longitude value as ``30°E`` or ``10°W``."""
    x = int(round(x))
    if x >= 0:
        return f"{x}°E"
    else:
        return f"{-x}°W"


def _lat_formatter(y, pos):
    """Format a latitude value as ``24°N`` or ``5°S``."""
    y = int(round(y))
    if y >= 0:
        return f"{y}°N"
    else:
        return f"{-y}°S"


def _temp_formatter(x, pos):
    """Format a temperature tick value as ``20°``, ``30°``, etc.

    The degree symbol is appended without a repeated unit label — the axis
    label (e.g. "Temperature (°C)") carries the full unit information.
    """
    return f"{int(x)}°"


def _geographic_aspect(region: dict) -> float:
    """Compute the display aspect ratio for a lat/lon map.

    At the centre latitude of the domain, one degree of longitude spans
    cos(lat_mid) of the physical distance of one degree of latitude.  To
    display the map without distortion the axes height/width ratio must
    account for this.

    Returns the value to pass to ``ax.set_aspect()``.
    """
    lat_mid = np.radians((region["south"] + region["north"]) / 2.0)
    return 1.0 / np.cos(lat_mid)


def apply_map_formatting(
    ax: plt.Axes,
    region: dict,
    lon_step: int = 2,
    lat_step: int = 2,
) -> None:
    """Apply standard geographic formatting to a map axes.

    Sets degree-formatted tick labels, tick spacing, axis labels, and
    corrects the display aspect ratio for the study region.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format.
    region : dict
        Bounding box ``{"south", "north", "west", "east"}``.
    lon_step : int
        Longitude tick interval in degrees (default 2°).
    lat_step : int
        Latitude tick interval in degrees (default 2°).
    """
    west, east = region["west"], region["east"]
    south, north = region["south"], region["north"]

    lon_ticks = np.arange(
        int(np.ceil(west / lon_step)) * lon_step,
        int(np.floor(east / lon_step)) * lon_step + lon_step,
        lon_step,
    )
    lat_ticks = np.arange(
        int(np.ceil(south / lat_step)) * lat_step,
        int(np.floor(north / lat_step)) * lat_step + lat_step,
        lat_step,
    )

    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_lon_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_lat_formatter))
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(_geographic_aspect(region))


def make_spatial_figure(
    ncols: int = 1,
    region: dict = None,
    col_width: float = 5.5,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a figure pre-configured for geographic map panels.

    Parameters
    ----------
    ncols : int
        Number of side-by-side map panels.
    region : dict, optional
        Study region (default: 24–38°N, 30–38°E).
    col_width : float
        Width of each panel in inches.

    Returns
    -------
    fig, axes : (Figure, list of Axes)
    """
    region = region or _DEFAULT_REGION
    asp = _geographic_aspect(region)
    lat_span = region["north"] - region["south"]
    lon_span = region["east"] - region["west"]
    h = col_width * (lat_span * asp) / lon_span
    figsize = (col_width * ncols + 0.5 * (ncols - 1), h)
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]
    return fig, axes


# ---------------------------------------------------------------------------
# Spatial map functions
# ---------------------------------------------------------------------------

def plot_temperature_map(
    data_2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    cbar_label: str = "Temperature (°C)",
    cmap: str = _TEMP_CMAP,
    region: dict = None,
    vmin: float = None,
    vmax: float = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot a single-panel 2-D temperature map.

    Parameters
    ----------
    data_2d : np.ndarray
        2-D array of shape ``(n_lat, n_lon)``.
    lats, lons : np.ndarray
        1-D coordinate arrays.
    title : str
        Panel title.
    cbar_label : str
        Colorbar label (include units).
    cmap : str
        Matplotlib colormap name.
    region : dict, optional
        Bounding box for axis limits and tick formatting.
    vmin, vmax : float, optional
        Color scale limits.
    save_path : str or Path, optional
        If provided, the figure is saved at this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    region = region or _DEFAULT_REGION
    fig, axes = make_spatial_figure(ncols=1, region=region)
    ax = axes[0]

    im = ax.pcolormesh(lons, lats, data_2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))

    ax.set_title(title, fontsize=13, pad=8)
    apply_map_formatting(ax, region)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_side_by_side_maps(
    data1: np.ndarray,
    data2: np.ndarray,
    lats1: np.ndarray,
    lons1: np.ndarray,
    lats2: np.ndarray,
    lons2: np.ndarray,
    titles: tuple[str, str],
    suptitle: str = "",
    panel_labels: tuple[str, str] = ("(a)", "(b)"),
    cbar_label: str = "Temperature (°C)",
    cmap: str = _TEMP_CMAP,
    region: dict = None,
    shared_clim: bool = True,
    vmin: float = None,
    vmax: float = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot two spatial maps side by side for direct comparison.

    Optionally uses a shared colour scale so that the two panels are
    directly comparable (``shared_clim=True``).

    Parameters
    ----------
    data1, data2 : np.ndarray
        2-D arrays for the left and right panels.
    lats1, lons1, lats2, lons2 : np.ndarray
        Coordinate arrays for each panel.
    titles : tuple of str
        Sub-titles for each panel (e.g. ``("ERA5-Land", "CMIP6 CESM2-WACCM")``).
    suptitle : str
        Figure-level super-title.
    panel_labels : tuple of str
        Labels prepended to each title, e.g. ``("(a)", "(b)")``.
    cbar_label : str
        Shared colorbar label.
    shared_clim : bool
        If True, compute vmin/vmax across both datasets.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    region = region or _DEFAULT_REGION

    if shared_clim:
        all_vals = np.concatenate(
            [data1[~np.isnan(data1)].ravel(), data2[~np.isnan(data2)].ravel()]
        )
        vmin = vmin if vmin is not None else float(np.nanpercentile(all_vals, 2))
        vmax = vmax if vmax is not None else float(np.nanpercentile(all_vals, 98))

    fig, axes = make_spatial_figure(ncols=2, region=region, col_width=5.5)

    for ax, data, lats, lons, title, label in zip(
        axes, [data1, data2], [lats1, lats2], [lons1, lons2], titles, panel_labels
    ):
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"{label} {title}", fontsize=12, pad=6)
        apply_map_formatting(ax, region)

    # Suppress Y-axis on the right panel — latitude is labelled on the left only
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis="y", left=False)

    fig.subplots_adjust(right=0.85, wspace=0.08)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.70])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_seasonal_maps(
    seasonal_means: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    dataset_label: str = "",
    cbar_label: str = "Mean temperature (°C)",
    cmap: str = _TEMP_CMAP,
    region: dict = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot four-panel seasonal mean temperature maps (DJF, MAM, JJA, SON).

    Parameters
    ----------
    seasonal_means : dict
        Keys ``"DJF"``, ``"MAM"``, ``"JJA"``, ``"SON"``; values are 2-D
        arrays of shape ``(n_lat, n_lon)``.
    lats, lons : np.ndarray
        Coordinate arrays.
    dataset_label : str
        Dataset name appended to the super-title (e.g. ``"ERA5-Land"``).
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    region = region or _DEFAULT_REGION
    seasons = ["DJF", "MAM", "JJA", "SON"]
    season_names = {
        "DJF": "December–February (DJF)",
        "MAM": "March–May (MAM)",
        "JJA": "June–August (JJA)",
        "SON": "September–November (SON)",
    }
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    all_vals = np.concatenate(
        [seasonal_means[s][~np.isnan(seasonal_means[s])].ravel() for s in seasons]
    )
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    asp = _geographic_aspect(region)
    lat_span = region["north"] - region["south"]
    lon_span = region["east"] - region["west"]
    col_w = 5.0
    h = col_w * (lat_span * asp) / lon_span
    fig, axes = plt.subplots(2, 2, figsize=(col_w * 2 + 0.5, h * 2 + 0.5))
    axes = axes.ravel()

    for ax, season, label in zip(axes, seasons, panel_labels):
        im = ax.pcolormesh(
            lons, lats, seasonal_means[season],
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )
        ax.set_title(f"{label} {season_names[season]}", fontsize=11, pad=5)
        apply_map_formatting(ax, region)

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.018, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))

    suptitle = "Seasonal mean 2 m temperature"
    if dataset_label:
        suptitle += f" — {dataset_label}"
    fig.suptitle(suptitle, fontsize=14, y=1.01)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_seasonal_comparison_maps(
    era5_seasonal_means: dict,
    cmip_seasonal_means: dict,
    era5_lats: np.ndarray,
    era5_lons: np.ndarray,
    cmip_lats: np.ndarray,
    cmip_lons: np.ndarray,
    cbar_label: str = "Mean temperature (°C)",
    cmap: str = _TEMP_CMAP,
    region: dict = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Four-season side-by-side comparison: ERA5-Land (left) vs CMIP6 (right).

    Produces a 4-row × 2-column figure where each row corresponds to one
    meteorological season (DJF, MAM, JJA, SON) and the two columns show
    ERA5-Land and CMIP6 CESM2-WACCM on a shared colour scale.

    Parameters
    ----------
    era5_seasonal_means : dict
        Keys ``"DJF"``, ``"MAM"``, ``"JJA"``, ``"SON"``; values are 2-D
        arrays ``(n_era5_lat, n_era5_lon)``.
    cmip_seasonal_means : dict
        Same keys; values are 2-D arrays ``(n_cmip_lat, n_cmip_lon)``.
    era5_lats, era5_lons : np.ndarray
        ERA5-Land coordinate arrays.
    cmip_lats, cmip_lons : np.ndarray
        CMIP6 coordinate arrays.
    cbar_label : str
        Shared colorbar label.
    cmap : str
        Matplotlib colormap name.
    region : dict, optional
        Bounding box for axis limits and tick formatting.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    region = region or _DEFAULT_REGION
    seasons = ["DJF", "MAM", "JJA", "SON"]
    season_names = {
        "DJF": "Dec–Feb (DJF)",
        "MAM": "Mar–May (MAM)",
        "JJA": "Jun–Aug (JJA)",
        "SON": "Sep–Nov (SON)",
    }

    # Shared colour limits across all 8 panels
    all_vals = np.concatenate([
        v[~np.isnan(v)].ravel()
        for d in (era5_seasonal_means, cmip_seasonal_means)
        for v in d.values()
    ])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    asp = _geographic_aspect(region)
    lat_span = region["north"] - region["south"]
    lon_span = region["east"] - region["west"]
    col_w = 4.5
    row_h = col_w * (lat_span * asp) / lon_span

    fig, axes = plt.subplots(
        4, 2,
        figsize=(col_w * 2 + 1.5, row_h * 4 + 0.5),
    )

    for row, season in enumerate(seasons):
        for col, (data, lats, lons) in enumerate([
            (era5_seasonal_means[season], era5_lats, era5_lons),
            (cmip_seasonal_means[season], cmip_lats, cmip_lons),
        ]):
            ax = axes[row, col]
            im = ax.pcolormesh(
                lons, lats, data,
                cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
            )
            # Row label (season) on left column only
            season_str = season_names[season]
            dataset_str = "ERA5-Land (0.1°)" if col == 0 else "CMIP6 (~1°)"
            ax.set_title(f"{season_str} — {dataset_str}", fontsize=10, pad=4)
            apply_map_formatting(ax, region)
            # Suppress Y-axis on right column — latitude is labelled on the left only
            if col == 1:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)

    # Column headers
    fig.text(0.28, 0.955, "ERA5-Land (0.1°)", ha="center", va="bottom",
             fontsize=12, fontweight="bold")
    fig.text(0.72, 0.955, "CMIP6 CESM2-WACCM (~1°)", ha="center", va="bottom",
             fontsize=12, fontweight="bold")

    # Shared horizontal colorbar at the bottom
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.05, right=0.97,
                        hspace=0.18, wspace=0.10)
    cbar_ax = fig.add_axes([0.12, 0.025, 0.76, 0.016])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))

    fig.suptitle(
        "Seasonal mean 2 m temperature — ERA5-Land vs CMIP6 CESM2-WACCM, 1990–1999",
        fontsize=13, y=0.97,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_land_sea_mask(
    land_mask_2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    region: dict = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot a binary land/ocean map derived from the ERA5-Land land–sea mask.

    Land pixels (True) are shown in light green; ocean/missing pixels (False)
    are shown in light blue.  No percentage colourbar is displayed — this is
    a clean binary classification map for use in final results.

    Parameters
    ----------
    land_mask_2d : np.ndarray of bool
        2-D boolean array, shape ``(n_lat, n_lon)``.  ``True`` = land pixel.
    lats, lons : np.ndarray
        1-D coordinate arrays.
    region : dict, optional
        Bounding box for axis limits and tick formatting.
    save_path : str or Path, optional
        If provided, saves the figure at 200 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    region = region or _DEFAULT_REGION
    fig, axes = make_spatial_figure(ncols=1, region=region)
    ax = axes[0]

    cmap_binary = ListedColormap(["#6baed6", "#a1d99b"])  # ocean=blue, land=green
    ax.pcolormesh(
        lons, lats, land_mask_2d.astype(float),
        cmap=cmap_binary, vmin=0, vmax=1, shading="auto",
    )

    legend_elements = [
        Patch(facecolor="#a1d99b", edgecolor="grey", label="Land"),
        Patch(facecolor="#6baed6", edgecolor="grey", label="Ocean / no data"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9)

    ax.set_title("ERA5-Land coverage: land and ocean pixels\n(1990–1999)", fontsize=12)
    apply_map_formatting(ax, region)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    return fig


def plot_missing_fraction_map(
    missing_fraction_2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    region: dict = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot the fraction of missing values per grid pixel.

    Parameters
    ----------
    missing_fraction_2d : np.ndarray
        2-D array in [0, 1]; value of 1.0 indicates always missing (ocean).
    lats, lons : np.ndarray
        Coordinate arrays.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    region = region or _DEFAULT_REGION
    fig, axes = make_spatial_figure(ncols=1, region=region)
    ax = axes[0]

    im = ax.pcolormesh(
        lons, lats, missing_fraction_2d,
        cmap=_MISSING_CMAP, vmin=0, vmax=1, shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of missing values", fontsize=10)
    ax.set_title("Fraction of missing values per pixel\n(ERA5-Land, 1990–1999)", fontsize=12)
    apply_map_formatting(ax, region)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Temporal plots
# ---------------------------------------------------------------------------

def plot_missing_fraction_timeseries(
    missing_per_day: pd.Series,
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot the daily fraction of missing ERA5-Land pixels over time.

    Parameters
    ----------
    missing_per_day : pd.Series
        Daily missing fraction with a DatetimeIndex.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(missing_per_day.index, missing_per_day.values, lw=0.8, color="steelblue")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Fraction of missing values", fontsize=10)
    ax.set_title(
        "Daily fraction of missing ERA5-Land pixels, 1990–1999\n"
        "(constant value confirms static land–sea mask)",
        fontsize=12,
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_domain_timeseries(
    series_dict: dict[str, pd.Series],
    title: str = "Domain-averaged daily 2 m temperature",
    ylabel: str = "Temperature (°C)",
    colors: dict | None = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot domain-averaged daily temperature time series for one or more datasets.

    Parameters
    ----------
    series_dict : dict
        Keys are dataset labels (e.g. ``"ERA5-Land"``, ``"CMIP6 CESM2-WACCM"``);
        values are pd.Series with a DatetimeIndex.
    title : str
        Figure title.
    ylabel : str
        Y-axis label.
    colors : dict, optional
        Map from label to colour string.  Defaults to a categorical palette.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    palette = sns.color_palette("tab10", n_colors=len(series_dict))
    colors = colors or {label: palette[i] for i, label in enumerate(series_dict)}

    fig, ax = plt.subplots(figsize=(12, 4))
    for label, series in series_dict.items():
        idx = pd.to_datetime(series.index) if not isinstance(series.index, pd.DatetimeIndex) else series.index
        ax.plot(idx, series.values, label=label, lw=1.2, color=colors[label])

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.legend(framealpha=0.8)
    # Year-level x-axis ticks for multi-year series
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    fig.autofmt_xdate(rotation=0, ha="center")
    # Temperature tick formatting on y-axis
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_domain_range_timeseries(
    mean_dict: dict[str, pd.Series],
    min_dict: dict[str, pd.Series],
    max_dict: dict[str, pd.Series],
    title: str = "Daily domain-mean ± within-cell min/max temperature",
    ylabel: str = "Temperature (°C)",
    colors: dict | None = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot domain-mean timeseries with shaded min–max envelope per dataset.

    Parameters
    ----------
    mean_dict : dict
        Keys are dataset labels; values are pd.Series (domain mean per day).
    min_dict : dict
        Same keys; values are pd.Series (daily average of within-cell min).
    max_dict : dict
        Same keys; values are pd.Series (daily average of within-cell max).
    title : str
        Figure title.
    ylabel : str
        Y-axis label.
    colors : dict, optional
        Map from label to colour string. Defaults to tab10 palette.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    palette = sns.color_palette("tab10", n_colors=len(mean_dict))
    colors = colors or {label: palette[i] for i, label in enumerate(mean_dict)}

    fig, ax = plt.subplots(figsize=(12, 4))
    for label, mean_s in mean_dict.items():
        color = colors[label]
        idx = (pd.to_datetime(mean_s.index)
               if not isinstance(mean_s.index, pd.DatetimeIndex)
               else mean_s.index)
        min_s = min_dict[label]
        max_s = max_dict[label]
        min_idx = (pd.to_datetime(min_s.index)
                   if not isinstance(min_s.index, pd.DatetimeIndex)
                   else min_s.index)
        ax.fill_between(min_idx, min_s.values, max_s.values,
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(idx, mean_s.values, label=label, lw=1.2, color=color)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.legend(framealpha=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_quarterly_warming_trend(
    era5_ts: pd.Series,
    cmip_ts: pd.Series,
    era5_label: str = "ERA5-Land pixel",
    cmip_label: str = "CMIP6 cell",
    quarter_dates: list[str] | None = None,
    quarter_labels: list[str] | None = None,
    title: str = "Quarterly temperature at selected location",
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot temperature at a single location for four quarterly dates across all years.

    For each of the four quarter-start dates, the value from each time series is
    extracted for every available year and plotted with an OLS linear trend line.
    This removes the seasonal cycle entirely, so any slope reflects year-over-year
    warming or cooling.

    Parameters
    ----------
    era5_ts : pd.Series
        Daily ERA5-Land temperature at one pixel, with DatetimeIndex.
    cmip_ts : pd.Series
        Daily CMIP6 temperature at one cell, with DatetimeIndex.
    era5_label : str
        Legend label for the ERA5-Land series.
    cmip_label : str
        Legend label for the CMIP6 series.
    quarter_dates : list of str, optional
        Four date strings in ``'MM-DD'`` format.
        Defaults to ``['01-01', '04-01', '07-01', '10-01']``.
    quarter_labels : list of str, optional
        Display titles for each panel.
        Defaults to ``['1 January', '1 April', '1 July', '1 October']``.
    title : str
        Figure suptitle.
    save_path : str or Path, optional
        If provided, saves the figure at 200 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if quarter_dates is None:
        quarter_dates = ['01-01', '04-01', '07-01', '10-01']
    if quarter_labels is None:
        quarter_labels = ['1 January', '1 April', '1 July', '1 October']

    palette = sns.color_palette("tab10", n_colors=2)
    c_era5, c_cmip = palette[0], palette[1]
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    # Ensure DatetimeIndex
    era5_ts = era5_ts.copy()
    cmip_ts = cmip_ts.copy()
    if not isinstance(era5_ts.index, pd.DatetimeIndex):
        era5_ts.index = pd.to_datetime(era5_ts.index)
    if not isinstance(cmip_ts.index, pd.DatetimeIndex):
        cmip_ts.index = pd.to_datetime(cmip_ts.index)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    axes = axes.ravel()

    for i, (date_str, qlabel) in enumerate(zip(quarter_dates, quarter_labels)):
        ax = axes[i]
        era5_sub = era5_ts[era5_ts.index.strftime('%m-%d') == date_str]
        cmip_sub = cmip_ts[cmip_ts.index.strftime('%m-%d') == date_str]

        years_e = era5_sub.index.year
        years_c = cmip_sub.index.year

        ax.plot(years_e, era5_sub.values, color=c_era5, marker='o', lw=1.4,
                ms=6, label=era5_label)
        ax.plot(years_c, cmip_sub.values, color=c_cmip, marker='s', lw=1.4,
                ms=6, label=cmip_label)

        # OLS trend lines
        for years_arr, vals, col in [
            (years_e, era5_sub.values, c_era5),
            (years_c, cmip_sub.values, c_cmip),
        ]:
            if len(years_arr) >= 2:
                x_num = np.arange(len(years_arr))
                slope, intercept = np.polyfit(x_num, vals, 1)
                ax.plot(years_arr, slope * x_num + intercept,
                        color=col, lw=0.9, linestyle=':', alpha=0.7)

        all_years = sorted(set(years_e) | set(years_c))
        ax.set_xticks(all_years)
        ax.set_xticklabels(all_years, fontsize=8, rotation=45, ha='right')
        ax.set_title(qlabel, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
        ax.text(0.03, 0.96, panel_labels[i], transform=ax.transAxes,
                fontsize=10, va='top', fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        if i % 2 == 0:
            ax.set_ylabel('Temperature (°C)', fontsize=9)

    # Shared legend below the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2,
               framealpha=0.8, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
    return fig


def plot_monthly_climatology(
    series_dict: dict[str, pd.Series],
    title: str = "Monthly mean 2 m temperature climatology, 1990–1999",
    ylabel: str = "Temperature (°C)",
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot monthly mean temperature climatology for multiple datasets.

    Parameters
    ----------
    series_dict : dict
        Keys are dataset labels; values are pd.Series with a DatetimeIndex
        spanning the full study period.
    title : str
        Figure title.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    palette = sns.color_palette("tab10", n_colors=len(series_dict))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (label, series) in enumerate(series_dict.items()):
        idx = pd.to_datetime(series.index) if not isinstance(series.index, pd.DatetimeIndex) else series.index
        monthly = series.copy()
        monthly.index = idx
        climatology = monthly.groupby(monthly.index.month).mean()
        ax.plot(climatology.index, climatology.values, marker="o", lw=1.8,
                label=label, color=palette[i])

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_temperature_distributions(
    data_dict: dict[str, np.ndarray | pd.Series],
    ylabel: str = "Temperature (°C)",
    title: str = "Temperature distribution by dataset",
    save_path: str | Path = None,
) -> plt.Figure:
    """Vertical boxplot comparing temperature distributions across datasets or GCMs.

    Parameters
    ----------
    data_dict : dict
        Keys are dataset labels; values are 1-D arrays or pd.Series of
        temperature values.
    ylabel : str
        Vertical axis label (temperature).
    title : str
        Figure title.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(data_dict.keys())
    values = [np.asarray(v).ravel() for v in data_dict.values()]
    # Subsample for plotting efficiency if arrays are large
    max_pts = 200_000
    values_plot = [v[np.random.choice(len(v), min(max_pts, len(v)), replace=False)] for v in values]

    fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(labels)), 6))
    bp = ax.boxplot(
        values_plot,
        vert=True,
        patch_artist=True,
        labels=labels,
        notch=False,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    palette = sns.color_palette("tab10", n_colors=len(labels))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Percentile distribution plot (Main Statistics)
# ---------------------------------------------------------------------------

def plot_temperature_percentiles(
    era5_flat: np.ndarray,
    cmip_flat: np.ndarray,
    percentiles: tuple | list = (25, 50, 75, 90),
    save_path: str | Path = None,
) -> plt.Figure:
    """Density histogram comparing ERA5-Land and CMIP6 temperature distributions
    with explicit percentile markers.

    Uses numpy histograms for efficiency with large (>10 M) arrays.
    Percentile lines are annotated with labels positioned at two distinct
    y-fractions so ERA5-Land and CMIP6 labels do not overlap.

    Parameters
    ----------
    era5_flat : np.ndarray
        1-D array of ERA5-Land land-pixel temperatures in °C.
    cmip_flat : np.ndarray
        1-D array of CMIP6 cell temperatures in °C.
    percentiles : tuple or list
        Percentile values to mark (e.g. (25, 50, 75, 90)).
    save_path : str or Path, optional
        Output file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _c_era5 = "#1f77b4"   # tab10 blue
    _c_cmip  = "#ff7f0e"  # tab10 orange

    datasets = [
        ("ERA5-Land (0.1°)", era5_flat, _c_era5),
        ("CMIP6 (~1°)",      cmip_flat,  _c_cmip),
    ]

    # Shared bin edges over the combined temperature range
    combined = np.concatenate([
        era5_flat[np.isfinite(era5_flat)],
        cmip_flat[np.isfinite(cmip_flat)],
    ])
    bins = np.linspace(np.nanmin(combined), np.nanmax(combined), 300)

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, vals, color in datasets:
        clean = vals[np.isfinite(vals)]
        counts, edges = np.histogram(clean, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, lw=1.8, color=color, label=label)
        ax.fill_between(centers, counts, alpha=0.15, color=color)

    # Percentile vertical lines — ERA5 labels near top, CMIP6 slightly lower
    # transform=get_xaxis_transform() → x in data coords, y in axes fraction
    label_y = {0: 0.96, 1: 0.83}   # y-fraction per dataset index
    for i_ds, (label, vals, color) in enumerate(datasets):
        clean = vals[np.isfinite(vals)]
        pct_vals = np.percentile(clean, percentiles)
        y_frac = label_y[i_ds]
        for p, pval in zip(percentiles, pct_vals):
            ax.axvline(pval, color=color, lw=0.9, linestyle="--", alpha=0.8)
            ax.text(
                pval, y_frac,
                f"P{p}",
                ha="center", va="top",
                fontsize=7.5, color=color,
                transform=ax.get_xaxis_transform(),
            )

    ax.set_xlabel("2 m temperature (°C)", fontsize=10)
    ax.set_ylabel("Probability density", fontsize=10)
    ax.set_title(
        "Daily 2 m temperature distribution, 1990–1999 — with key percentiles",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    ax.legend(framealpha=0.85)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    return fig


# ---------------------------------------------------------------------------
# Grid assignment plots
# ---------------------------------------------------------------------------

def plot_pixel_assignment_map(
    assignment_df: pd.DataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
    land_mask_2d: np.ndarray,
    region: dict = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Plot ERA5-Land pixels coloured by their assigned CMIP6 grid cell.

    Parameters
    ----------
    assignment_df : pd.DataFrame
        Output of :func:`src.spatial_ops.assign_era5_to_cmip_cells`.
    lats, lons : np.ndarray
        ERA5-Land coordinate arrays.
    land_mask_2d : np.ndarray
        Boolean land mask.
    region : dict, optional
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    region = region or _DEFAULT_REGION

    # Build 2-D integer array mapping each pixel to a unique CMIP6 cell ID
    cell_ids = assignment_df.groupby(["cmip_lat", "cmip_lon"]).ngroup()
    cell_id_map = dict(zip(
        zip(assignment_df["cmip_lat"], assignment_df["cmip_lon"]),
        cell_ids,
    ))

    land_rows, land_cols = np.where(land_mask_2d)
    cell_grid = np.full(land_mask_2d.shape, np.nan)
    for row, col in zip(land_rows, land_cols):
        key = (assignment_df.loc[(assignment_df["era5_lat"] == lats[row]) &
                                 (assignment_df["era5_lon"] == lons[col]),
                                 ["cmip_lat", "cmip_lon"]].values)
        if len(key):
            cell_grid[row, col] = cell_id_map.get(tuple(key[0]), np.nan)

    fig, axes = make_spatial_figure(ncols=1, region=region)
    ax = axes[0]
    im = ax.pcolormesh(lons, lats, cell_grid, cmap="tab20b", shading="auto")
    ax.set_title("ERA5-Land pixels coloured by assigned CMIP6 grid cell", fontsize=12)
    apply_map_formatting(ax, region)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_pixels_per_cell_heatmap(
    pixel_counts_df: pd.DataFrame,
    cmip_lats: np.ndarray,
    cmip_lons: np.ndarray,
    region: dict = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Heatmap of ERA5-Land pixel counts per CMIP6 grid cell.

    Parameters
    ----------
    pixel_counts_df : pd.DataFrame
        Output of :func:`src.spatial_ops.pixel_counts_per_cmip_cell`.
    cmip_lats, cmip_lons : np.ndarray
        CMIP6 coordinate arrays (for axis labels).
    region : dict, optional
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    region = region or _DEFAULT_REGION

    # Pivot to 2-D grid for display; fill cells with no land pixels as 0
    pivot = pixel_counts_df.pivot(index="cmip_lat", columns="cmip_lon", values="n_pixels")
    pivot = pivot.sort_index(ascending=False)  # north at top
    pivot_int = pivot.fillna(0).astype(int)    # convert to int so fmt="d" works

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot_int,
        ax=ax,
        cmap="YlGnBu",
        annot=True,
        fmt="d",          # integer format — no decimal point
        annot_kws={"size": 7},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Number of ERA5-Land pixels"},
    )
    # Clean tick labels: round to nearest integer degree
    ax.set_xticklabels(
        [f"{round(float(c))}°E" for c in pivot.columns], rotation=45, ha="right", fontsize=8
    )
    ax.set_yticklabels(
        [f"{round(float(r))}°N" for r in pivot.index], rotation=0, fontsize=8
    )
    ax.set_xlabel("CMIP6 longitude", fontsize=10)
    ax.set_ylabel("CMIP6 latitude", fontsize=10)
    ax.set_title(
        "Number of ERA5-Land (0.1°) pixels assigned to each CMIP6 (~1°) grid cell",
        fontsize=12,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Coarse–fine relationship
# ---------------------------------------------------------------------------

def plot_scatter_regression(
    paired_df: pd.DataFrame,
    x_col: str = "tas",
    y_col: str = "t2m",
    title: str = "CMIP6 vs ERA5-Land temperature",
    xlabel: str = "CMIP6 near-surface air temperature, TAS (°C)",
    ylabel: str = "ERA5-Land 2 m temperature, T2M (°C)",
    sample_frac: float = 0.005,
    region: dict = None,
    save_path: str | Path = None,
) -> plt.Figure:
    """Two-panel scatter plot: hexbin density and aggregated cell-mean OLS.

    Panel (a): Hexbin density plot of all pixel × day pairs (subsampled for
    display efficiency).
    Panel (b): Scatter of CMIP6 cell daily mean vs ERA5-Land cell mean with
    ordinary least-squares regression line.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Full pixel × day paired DataFrame (from :func:`src.data_io.build_paired_dataframe`).
    x_col, y_col : str
        Column names for predictor (CMIP6) and target (ERA5-Land).
    sample_frac : float
        Fraction of rows used in the hexbin panel (default 0.5%).
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # — Panel (a): hexbin density —
    ax = axes[0]
    sample = paired_df.sample(frac=sample_frac, random_state=42)
    hb = ax.hexbin(
        sample[x_col], sample[y_col],
        gridsize=60, cmap="YlOrRd", mincnt=1,
    )
    fig.colorbar(hb, ax=ax, label="Count")
    lims = [
        min(sample[x_col].min(), sample[y_col].min()),
        max(sample[x_col].max(), sample[y_col].max()),
    ]
    ax.plot(lims, lims, "k--", lw=1, label="1:1 line")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title("(a) Pixel × day density", fontsize=12)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))

    # — Panel (b): cell-mean OLS —
    ax = axes[1]
    agg = (
        paired_df.groupby(["cmip_lat", "cmip_lon", "day", x_col], observed=True)[y_col]
        .mean()
        .reset_index()
        .rename(columns={y_col: f"{y_col}_mean"})
    )
    x_agg = agg[x_col].values
    y_agg = agg[f"{y_col}_mean"].values
    mask = np.isfinite(x_agg) & np.isfinite(y_agg)
    coeffs = np.polyfit(x_agg[mask], y_agg[mask], 1)
    slope, intercept = coeffs
    r = np.corrcoef(x_agg[mask], y_agg[mask])[0, 1]

    ax.scatter(x_agg, y_agg, s=1, alpha=0.3, color="steelblue", rasterized=True)
    x_line = np.linspace(x_agg[mask].min(), x_agg[mask].max(), 200)
    ax.plot(x_line, np.polyval(coeffs, x_line), "r-", lw=1.8,
            label=f"OLS: y = {slope:.3f}x + {intercept:.2f}")
    ax.plot(x_line, x_line, "k--", lw=1, label="1:1 line")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(f"Cell-mean {ylabel}", fontsize=10)
    ax.set_title(f"(b) CMIP6 cell mean vs ERA5-Land cell mean\nPearson r = {r:.3f}", fontsize=12)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_residual_analysis(
    paired_df: pd.DataFrame,
    x_col: str = "tas",
    y_col: str = "t2m",
    dist_col: str = "dist_deg",
    sample_n: int = 100_000,
    save_path: str | Path = None,
) -> plt.Figure:
    """Two-panel residual diagnostic plot.

    Fits a simple OLS model (t2m ~ tas) per CMIP6 cell and examines the
    residuals to assess:

    Panel (a) — Homoscedasticity: residuals plotted against CMIP6 TAS values.
    A horizontal spread with no trend indicates constant variance.

    Panel (b) — Spatial dependency: residuals plotted against distance (°)
    from the assigned CMIP6 cell centre.  A non-zero trend would indicate
    that the nearest-neighbour assignment introduces systematic bias for
    pixels near cell boundaries.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Paired pixel × day DataFrame; must contain columns *x_col*, *y_col*,
        and optionally *dist_col* (distance to cell centre in degrees,
        computed by :func:`src.spatial_ops.compute_distance_to_cell_center`).
    sample_n : int
        Maximum number of points to display (random subsample for speed).
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = paired_df[[x_col, y_col] + ([dist_col] if dist_col in paired_df.columns else [])].dropna()

    # Fit OLS globally: residual = t2m - (a * tas + b)
    mask = np.isfinite(df[x_col].values) & np.isfinite(df[y_col].values)
    coeffs = np.polyfit(df.loc[mask, x_col], df.loc[mask, y_col], 1)
    df = df[mask].copy()
    df["residual"] = df[y_col] - np.polyval(coeffs, df[x_col])

    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    ncols = 2 if dist_col in df.columns else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # Panel (a) — residuals vs predictor
    ax = axes[0]
    ax.scatter(df[x_col], df["residual"], s=1, alpha=0.15, color="steelblue", rasterized=True)
    ax.axhline(0, color="red", lw=1.2, linestyle="--")
    ax.set_xlabel("CMIP6 TAS (°C)", fontsize=10)
    ax.set_ylabel("Residual T2M − Ŷ (°C)", fontsize=10)
    ax.set_title("(a) Residuals vs predictor\n(homoscedasticity check)", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))

    # Panel (b) — residuals vs distance from cell centre
    if dist_col in df.columns:
        ax2 = axes[1]
        ax2.scatter(df[dist_col], df["residual"], s=1, alpha=0.15, color="darkorange", rasterized=True)
        ax2.axhline(0, color="red", lw=1.2, linestyle="--")
        # Smooth trend via binned means
        bins = np.linspace(df[dist_col].min(), df[dist_col].max(), 20)
        bin_labels = (bins[:-1] + bins[1:]) / 2
        binned = df.groupby(pd.cut(df[dist_col], bins=bins))["residual"].mean()
        ax2.plot(bin_labels, binned.values, "k-", lw=1.8, label="Bin mean")
        ax2.set_xlabel("Distance to CMIP6 cell centre (°)", fontsize=10)
        ax2.set_ylabel("Residual T2M − Ŷ (°C)", fontsize=10)
        ax2.set_title("(b) Residuals vs distance from cell centre\n(spatial bias check)", fontsize=12)
        ax2.legend(fontsize=9)

    fig.suptitle("Residual diagnostics — OLS (T2M ~ TAS)", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def compute_ols_residuals(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Fit a global OLS (T2M ~ TAS) and return per-pixel×day residuals.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Output of ``build_paired_dataframe``.  Must contain ``tas``,
        ``t2m``, ``cmip_lat``, ``cmip_lon``.

    Returns
    -------
    pd.DataFrame
        Lightweight DataFrame with columns ``cmip_lat``, ``cmip_lon``,
        ``residual`` (one row per valid pixel × day pair).
    """
    mask = np.isfinite(paired_df["tas"].values) & np.isfinite(paired_df["t2m"].values)
    coeffs = np.polyfit(paired_df.loc[mask, "tas"], paired_df.loc[mask, "t2m"], 1)
    resid = (paired_df.loc[mask, "t2m"].values
             - np.polyval(coeffs, paired_df.loc[mask, "tas"].values))
    return pd.DataFrame({
        "cmip_lat": paired_df.loc[mask, "cmip_lat"].values,
        "cmip_lon": paired_df.loc[mask, "cmip_lon"].values,
        "residual": resid,
    })


def _build_bin_groups(
    resid_df: pd.DataFrame,
    cell_stats_df: pd.DataFrame,
    stat_col: str,
    bin_edges: np.ndarray,
) -> tuple:
    """Group pixel×day residuals into bins defined by a cell-level stat.

    Iterates over CMIP6 cells (fast, O(n_cells)) rather than merging the
    full 28 M-row residual frame, keeping memory overhead low.

    Returns
    -------
    groups : list of np.ndarray
        One array of residual values per bin.
    n_cells : list of int
        Number of CMIP6 cells that fell into each bin.
    """
    sf_map = dict(zip(
        zip(cell_stats_df["cmip_lat"], cell_stats_df["cmip_lon"]),
        cell_stats_df[stat_col],
    ))
    n_bins = len(bin_edges) - 1
    bins_resid = [[] for _ in range(n_bins)]
    bins_cells = [0] * n_bins

    for (clat, clon), grp in resid_df.groupby(["cmip_lat", "cmip_lon"]):
        val = sf_map.get((clat, clon), np.nan)
        if not np.isfinite(val):
            continue
        idx = min(int(np.searchsorted(bin_edges[1:], val, side="left")), n_bins - 1)
        bins_resid[idx].append(grp["residual"].values)
        bins_cells[idx] += 1

    groups = [np.concatenate(g) if g else np.array([]) for g in bins_resid]
    return groups, bins_cells


def _draw_residual_boxplot(
    ax: plt.Axes,
    groups: list,
    tick_labels: list,
    title: str,
    xlabel: str,
) -> None:
    """Draw a styled box plot of OLS residuals on *ax*."""
    ax.axhline(0, lw=0.8, linestyle="--", color="grey")
    bp = ax.boxplot(
        groups,
        labels=tick_labels,
        patch_artist=True,
        showfliers=False,  # omit outlier dots for large arrays
        medianprops=dict(color="firebrick", lw=2),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.45)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("OLS residual (°C)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_temp_formatter))


def plot_residuals_by_sea_fraction(
    resid_df: pd.DataFrame,
    cell_stats_df: pd.DataFrame,
    n_bins: int = 5,
    title: str = "OLS residual distribution by CMIP6 cell sea fraction, 1990–1999",
    save_path=None,
) -> plt.Figure:
    """Box plots of pixel×day OLS residuals grouped by CMIP6 cell sea fraction.

    Each ERA5 land pixel × day residual is placed in a sea-fraction bin
    determined by the proportion of ocean ERA5 grid points within its
    assigned CMIP6 cell.  Box plots reveal whether coastal cells (high sea
    fraction) — whose CMIP TAS averages over a land–sea temperature mix —
    produce wider residual spreads than purely inland cells.

    Parameters
    ----------
    resid_df : pd.DataFrame
        Output of :func:`compute_ols_residuals`.  Columns:
        ``cmip_lat``, ``cmip_lon``, ``residual``.
    cell_stats_df : pd.DataFrame
        One row per CMIP6 cell; must contain ``cmip_lat``, ``cmip_lon``,
        ``sea_fraction``.
    n_bins : int
        Number of equal-width sea-fraction bins. Default: 5.
    title : str
        Figure title.
    save_path : str or Path, optional
        If provided, saves the figure at 200 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [
        f"{int(bin_edges[i] * 100)}–{int(bin_edges[i + 1] * 100)}%"
        for i in range(n_bins)
    ]

    groups, n_cells = _build_bin_groups(resid_df, cell_stats_df, "sea_fraction", bin_edges)
    tick_labels = [
        f"{lbl}\n({nc} cells,\nn={len(g):,})"
        for lbl, nc, g in zip(bin_labels, n_cells, groups)
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_residual_boxplot(
        ax, groups, tick_labels,
        title=title,
        xlabel="Sea fraction of assigned CMIP6 cell",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    return fig


def plot_residuals_by_pixel_count(
    resid_df: pd.DataFrame,
    cell_stats_df: pd.DataFrame,
    n_bins: int = 5,
    title: str = "OLS residual distribution by CMIP6 cell land-pixel count, 1990–1999",
    save_path=None,
) -> plt.Figure:
    """Box plots of pixel×day OLS residuals grouped by CMIP6 cell land-pixel count.

    Each ERA5 land pixel × day residual is placed in a pixel-count bin
    determined by the number of ERA5 land pixels assigned to its CMIP6 cell.
    Box plots reveal whether data-sparse cells (few pixels, typically coastal
    or domain-edge) produce different residual distributions from data-rich
    inland cells.

    Parameters
    ----------
    resid_df : pd.DataFrame
        Output of :func:`compute_ols_residuals`.  Columns:
        ``cmip_lat``, ``cmip_lon``, ``residual``.
    cell_stats_df : pd.DataFrame
        One row per CMIP6 cell; must contain ``cmip_lat``, ``cmip_lon``,
        ``n_land_pixels``.
    n_bins : int
        Number of quantile-based bins so each bin contains roughly equal
        numbers of CMIP6 cells. Default: 5.
    title : str
        Figure title.
    save_path : str or Path, optional
        If provided, saves the figure at 200 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    counts = cell_stats_df["n_land_pixels"].dropna().values
    bin_edges = np.quantile(counts, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1  # include the minimum value

    groups, n_cells = _build_bin_groups(resid_df, cell_stats_df, "n_land_pixels", bin_edges)
    tick_labels = [
        f"{int(bin_edges[i] + 1)}–{int(bin_edges[i + 1])}\n({nc} cells,\nn={len(g):,})"
        for i, (nc, g) in enumerate(zip(n_cells, groups))
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_residual_boxplot(
        ax, groups, tick_labels,
        title=title,
        xlabel="Land-pixel count of assigned CMIP6 cell",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    return fig
