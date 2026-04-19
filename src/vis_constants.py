"""
vis_constants.py — Centralised style parameters for all project figures.

Import this module in visualization.py (and any other plotting modules) to
control font sizes, line weights, and colour choices from a single location.
"""

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
TITLE_FONT_SIZE  = 15   # ax.set_title / fig.suptitle
LABEL_FONT_SIZE  = 13   # axis labels (xlabel / ylabel)
TICK_FONT_SIZE   = 11   # tick label size (xtick / ytick)
LEGEND_FONT_SIZE = 11   # legend text
PANEL_LABEL_SIZE = 13   # in-axes panel labels "(a)", "(b)", …
ANNOT_FONT_SIZE  = 8    # seaborn heatmap cell annotations
CBAR_LABEL_SIZE  = 12   # colourbar axis label

# ---------------------------------------------------------------------------
# Figure output settings
# ---------------------------------------------------------------------------
FIG_DPI  = 150   # screen / notebook DPI
SAVE_DPI = 200   # saved-to-disk DPI

# ---------------------------------------------------------------------------
# Coarse-grid overlay (CMIP6 cell boundaries drawn on fine-resolution maps)
# ---------------------------------------------------------------------------
COARSE_GRID_COLOR = "#333333"
COARSE_GRID_LW    = 0.5
COARSE_GRID_ALPHA = 0.6

# ---------------------------------------------------------------------------
# Confidence / credible bands on time-series plots
# ---------------------------------------------------------------------------
CI_ALPHA      = 0.27   # fill_between transparency
DEFAULT_CI_PCT = 90    # default confidence level (%)
