"""
qm_dashboard.py — Interactive comparison of the quantile-mapping baseline.

Launch with::

    python3 -m streamlit run dashboards/qm_dashboard.py

Use ``python3 -m streamlit``, not the bare ``streamlit`` command: the
``streamlit`` script on PATH may belong to a different interpreter that does not
have this project's dependencies installed.

Reads the cached tables written by ``scripts/run_quantile_mapping.py`` from
``data/cache/qm/``.  The four aggregate tables are small and loaded once; the
large per-pixel percentile tables are read only for the current selection, one
column pair at a time, and cached by Streamlit.

Sign convention throughout: ``bias = predictor − ERA5-Land``, so a positive
bias means the predictor's percentile is too warm.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import qm_metrics as qmm            # noqa: E402
from src import quantile_windows as qw       # noqa: E402
from src.predictors import PREDICTOR_NAMES   # noqa: E402
from src.qm_visualization import PREDICTOR_LABELS, SCHEME_LABELS  # noqa: E402

def _resolve_data_dir() -> Path:
    """Pick the data directory: the full local cache, else the committed subset.

    ``data/cache/qm`` holds the full ~700 MB output of the pipeline and is
    gitignored, so it is absent on a deployment. ``data/deploy`` holds the
    committed subset built by ``scripts/export_dashboard_data.py``. The full
    cache wins when present so local runs always see every combination.
    """
    full = ROOT / "data" / "cache" / "qm"
    slim = ROOT / "data" / "deploy"
    if (full / qmm.METRIC_FILES["metrics"]).exists():
        return full
    return slim


QM_DIR = _resolve_data_dir()
IS_SLIM = QM_DIR.name == "deploy"
REGION = dict(south=24, north=38, west=30, east=38)

st.set_page_config(
    page_title="Quantile-mapping baseline",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_aggregate(key: str) -> pd.DataFrame:
    """Load one small aggregate table by its :data:`qm_metrics.METRIC_FILES` key."""
    path = QM_DIR / qmm.METRIC_FILES[key]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_lapse_rates() -> pd.DataFrame:
    path = QM_DIR / "lapse_rates.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner="Reading percentile table …")
def load_slice(predictor: str, scheme: str, q: int, window_label: str) -> pd.DataFrame:
    """Read one (predictor, scheme) table and reduce it to one percentile.

    Only the columns needed for the current view are read from parquet.
    ``window_label`` of ``"All windows"`` keeps every window.
    """
    cols = [
        "window_id", "window_label", "window_start", "n_days",
        "lat", "lon", "elevation", "dz", "sea_fraction",
        f"y_p{q}", f"x_p{q}",
    ]
    df = qmm.load_pct_table(QM_DIR, predictor, scheme, columns=cols)
    if window_label != "All windows":
        df = df[df["window_label"].astype(str) == window_label]
    return qmm.slice_percentile(df, q)


@st.cache_data(show_spinner=False)
def available_predictors() -> list[str]:
    """Predictors that actually have percentile tables on disk."""
    metrics = load_aggregate("metrics")
    if not metrics.empty:
        scored = set(metrics["predictor"])
        return [n for n in PREDICTOR_NAMES if n in scored]
    # No aggregates: fall back to whatever percentile tables are on disk.
    return [n for n in PREDICTOR_NAMES
            if any(qmm.pct_table_path(QM_DIR, n, s).exists() for s in qw.SCHEMES)]


@st.cache_data(show_spinner=False)
def window_labels(scheme: str) -> list[str]:
    """Chronological window labels for one scheme.

    Read from the small per-window aggregate rather than a percentile table.
    The aggregate covers all four schemes even when only some percentile
    tables shipped, and it avoids a multi-million-row read at startup.
    """
    agg = load_aggregate("window")
    if agg.empty:
        return []
    part = agg[agg["scheme"] == scheme]
    return (
        part.drop_duplicates("window_id")
        .sort_values("window_id")["window_label"].astype(str).tolist()
    )


@st.cache_data(show_spinner=False)
def schemes_with_points(predictor: str) -> list[str]:
    """Schemes whose per-pixel percentile table is available for *predictor*.

    Only these can back the point-level views. The aggregate-driven views work
    for every scheme regardless.
    """
    return [s for s in qw.SCHEMES
            if qmm.pct_table_path(QM_DIR, predictor, s).exists()]


def points_available(predictors: list[str], scheme: str) -> bool:
    """True when every selected predictor has point-level data for *scheme*."""
    return all(scheme in schemes_with_points(p) for p in predictors)


def _no_points_notice(scheme: str, predictors: list[str]) -> None:
    """Explain that this view needs a percentile table that did not ship."""
    have = sorted(
        {s for p in predictors for s in schemes_with_points(p)},
        key=qw.SCHEMES.index,
    )
    names = ", ".join(SCHEME_LABELS.get(s, s) for s in have) or "none"
    st.info(
        f"This view plots one point per pixel, which needs the full percentile "
        f"table for **{SCHEME_LABELS.get(scheme, scheme)}** windows. "
        f"That table is not part of this deployment.\n\n"
        f"Available here: **{names}**. The Leaderboard and Diagnostics views "
        f"cover all four window lengths.\n\n"
        f"For every combination, run the pipeline locally: "
        f"`python3 scripts/run_quantile_mapping.py`."
    )


def label_of(predictor: str) -> str:
    return PREDICTOR_LABELS.get(predictor, predictor)


# Streamlit's theme overrides plotly's default qualitative colorway with a
# sequence of blues, which makes two selected predictors almost impossible to
# tell apart. Pin one distinct colour per predictor instead, keyed by display
# label so the same predictor keeps its colour across every view.
PREDICTOR_COLORS = {
    PREDICTOR_LABELS["knn4"]: "#4C78A8",            # blue
    PREDICTOR_LABELS["knn9"]: "#72B7B2",            # teal
    PREDICTOR_LABELS["bilinear"]: "#F58518",        # orange
    PREDICTOR_LABELS["trilinear_fit"]: "#E45756",   # red
    PREDICTOR_LABELS["trilinear_fixed"]: "#B279A2", # purple
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if not QM_DIR.exists() or not available_predictors():
    st.error(
        "No results found in `data/cache/qm/` or `data/deploy/`.\n\n"
        "Run the pipeline first:\n\n"
        "```bash\npython3 scripts/run_quantile_mapping.py\n```"
    )
    st.stop()

preds = available_predictors()

st.sidebar.title("Quantile-mapping baseline")
st.sidebar.caption(
    "ERA5-Land 0.1° vs CMIP6 CESM2-WACCM, 1990–1999, EMME domain. "
    "Percentiles are computed per pixel within each distribution window."
)

sel_predictors = st.sidebar.multiselect(
    "Predictor(s)",
    options=preds,
    default=[p for p in ("bilinear", "trilinear_fit") if p in preds][:2] or preds[:1],
    format_func=label_of,
    help="Pick one for a single view, or two to compare side by side.",
)
if not sel_predictors:
    st.sidebar.warning("Select at least one predictor.")
    st.stop()

sel_scheme = st.sidebar.selectbox(
    "Distribution window",
    options=qw.SCHEMES,
    index=qw.SCHEMES.index("quarter"),
    format_func=lambda s: SCHEME_LABELS.get(s, s),
)

sel_q = st.sidebar.selectbox(
    "Percentile",
    options=list(qw.PERCENTILES),
    index=list(qw.PERCENTILES).index(50),
    format_func=lambda q: f"P{q}",
)

labels = window_labels(sel_scheme)
default_window = "1999-Q2" if "1999-Q2" in labels else labels[-1]
sel_window = st.sidebar.selectbox(
    "Window instance",
    options=["All windows"] + labels,
    index=(["All windows"] + labels).index(default_window),
)

st.sidebar.divider()
st.sidebar.subheader("Pixel filters")

static = load_aggregate("pixel")
elev_lo, elev_hi = st.sidebar.slider(
    "Elevation range (m a.s.l.)",
    min_value=-500, max_value=3500, value=(-500, 3500), step=100,
)
sea_max = st.sidebar.slider(
    "Max sea fraction of parent coarse cell",
    min_value=0.0, max_value=1.0, value=1.0, step=0.05,
    help="Lower this to keep inland pixels only.",
)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the sidebar elevation and sea-fraction filters."""
    out = df
    if "elevation" in out.columns:
        out = out[out["elevation"].between(elev_lo, elev_hi)]
    if "sea_fraction" in out.columns:
        out = out[out["sea_fraction"] <= sea_max]
    return out


def _map_axes() -> dict:
    """Axis settings shared by every map panel.

    The latitude axis is anchored to the longitude axis at 1/cos(31°N) so the
    domain is not geographically distorted, matching
    ``visualization.apply_map_formatting``. Both ranges are pinned to the study
    region; without an explicit longitude range the aspect lock widens the x
    axis well past the data and leaves the map floating in empty space.
    """
    return dict(
        xaxis=dict(range=[REGION["west"], REGION["east"]], constrain="domain"),
        yaxis=dict(
            range=[REGION["south"], REGION["north"]],
            scaleanchor="x", scaleratio=1 / np.cos(np.radians(31.0)),
            constrain="domain",
        ),
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Quantile-mapping baseline: predictor comparison")
st.caption(
    f"Window: **{SCHEME_LABELS[sel_scheme]}** · Percentile: **P{sel_q}** · "
    f"Instance: **{sel_window}** · "
    f"Predictors: **{', '.join(label_of(p) for p in sel_predictors)}**"
)

# A single view is rendered at a time. st.tabs would execute every tab body on
# every rerun, so all five plot groups would be built and shipped to the browser
# at once — enough plotly figures to lock the page up. A radio renders one.
VIEWS = ["Scatter", "Maps", "Leaderboard", "Diagnostics", "About"]
view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Tab 1 — scatter
# ---------------------------------------------------------------------------

if view == "Scatter":
    st.subheader(f"Predictor P{sel_q} vs observed P{sel_q}")
    st.caption(
        "One point per (window, pixel). The dashed line is 1:1; the red line "
        "is the OLS fit of observed on predictor, which is the linear "
        "correction the data asks for."
    )

    if not points_available(sel_predictors, sel_scheme):
        _no_points_notice(sel_scheme, sel_predictors)
        st.stop()

    max_points = st.slider("Max points drawn per panel", 2_000, 60_000,
                           20_000, step=2_000)

    cols = st.columns(len(sel_predictors))
    for col, predictor in zip(cols, sel_predictors):
        df = apply_filters(load_slice(predictor, sel_scheme, sel_q, sel_window))
        df = df.dropna(subset=["x", "y"])
        if df.empty:
            col.warning("No pixels match the current filters.")
            continue

        n_total = len(df)
        show = df.sample(min(max_points, n_total), random_state=42)

        lo = float(min(df["x"].min(), df["y"].min()))
        hi = float(max(df["x"].max(), df["y"].max()))
        slope, intercept = np.polyfit(df["x"], df["y"], 1)

        fig = px.scatter(
            show, x="x", y="y", color="elevation",
            color_continuous_scale="Turbo", opacity=0.45,
            labels={
                "x": f"Predictor P{sel_q} (°C)",
                "y": f"ERA5-Land P{sel_q} (°C)",
                "elevation": "Elev (m)",
            },
            hover_data={"lat": ":.2f", "lon": ":.2f", "window_label": True,
                        "bias": ":.2f"},
        )
        fig.update_traces(marker=dict(size=4))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
            line=dict(color="black", dash="dash", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[slope * lo + intercept, slope * hi + intercept],
            mode="lines", name=f"OLS y={slope:.2f}x{intercept:+.2f}",
            line=dict(color="crimson", width=2),
        ))
        # Both axes must span the same range, or the 1:1 line no longer reads
        # as the diagonal and the whole comparison becomes misleading.
        pad = 0.03 * (hi - lo)
        fig.update_layout(
            title=label_of(predictor), height=560,
            xaxis=dict(range=[lo - pad, hi + pad], constrain="domain"),
            yaxis=dict(range=[lo - pad, hi + pad], scaleanchor="x",
                       scaleratio=1, constrain="domain"),
            legend=dict(orientation="h", y=-0.18),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        col.plotly_chart(fig, use_container_width=True)

        bias = float((df["x"] - df["y"]).mean())
        rmse = float(np.sqrt(((df["x"] - df["y"]) ** 2).mean()))
        r = float(np.corrcoef(df["x"], df["y"])[0, 1])
        m1, m2, m3, m4 = col.columns(4)
        m1.metric("Bias (°C)", f"{bias:+.2f}")
        m2.metric("RMSE (°C)", f"{rmse:.2f}")
        m3.metric("Pearson r", f"{r:.3f}")
        m4.metric("n points", f"{n_total:,}")


# ---------------------------------------------------------------------------
# Tab 2 — maps
# ---------------------------------------------------------------------------

if view == "Maps":
    st.subheader("Where the percentile bias sits")
    has_points = points_available(sel_predictors, sel_scheme)
    if has_points:
        scope = st.radio(
            "Averaging scope", ["Selected window", "All windows"],
            horizontal=True,
            help="'All windows' uses the precomputed per-pixel aggregate.",
        )
    else:
        # Only the aggregate is available, so a single window cannot be shown.
        scope = "All windows"
        st.caption(
            f"Averaged over all {SCHEME_LABELS.get(sel_scheme, sel_scheme)} "
            f"windows. Single-window maps need the full percentile table, "
            f"which is not part of this deployment."
        )

    def bias_grid(predictor: str) -> pd.DataFrame:
        """Per-pixel mean bias for one predictor under the current scope."""
        if scope == "All windows":
            agg = load_aggregate("pixel")
            part = agg[
                (agg["predictor"] == predictor)
                & (agg["scheme"] == sel_scheme)
                & (agg["percentile"] == sel_q)
            ]
            return part.rename(columns={"mean_bias": "bias"})[
                ["lat", "lon", "bias"]
            ]
        df = load_slice(predictor, sel_scheme, sel_q, sel_window)
        return (
            df.groupby(["lat", "lon"], observed=True)["bias"]
            .mean().reset_index()
        )

    frames = {p: apply_filters(bias_grid(p)) for p in sel_predictors}
    all_vals = np.concatenate(
        [f["bias"].to_numpy() for f in frames.values() if len(f)]
    ) if any(len(f) for f in frames.values()) else np.array([0.0])
    vmax = float(np.nanpercentile(np.abs(all_vals), 98)) or 1.0

    cols = st.columns(len(frames))
    for col, (predictor, f) in zip(cols, frames.items()):
        if f.empty:
            col.warning("No pixels match the current filters.")
            continue
        fig = px.scatter(
            f, x="lon", y="lat", color="bias",
            color_continuous_scale="RdBu_r", range_color=(-vmax, vmax),
            labels={"lon": "Longitude (°E)", "lat": "Latitude (°N)",
                    "bias": "Bias (°C)"},
        )
        fig.update_traces(marker=dict(size=3.2, symbol="square"))
        fig.update_layout(
            title=f"{label_of(predictor)} — mean {f['bias'].mean():+.2f}°C",
            height=620, margin=dict(l=10, r=10, t=50, b=10),
            **_map_axes(),
        )
        col.plotly_chart(fig, use_container_width=True)

    if len(sel_predictors) == 2:
        a, b = sel_predictors
        merged = frames[a].merge(frames[b], on=["lat", "lon"],
                                 suffixes=("_a", "_b"))
        if not merged.empty:
            merged["improvement"] = merged["bias_a"].abs() - merged["bias_b"].abs()
            lim = float(np.nanpercentile(merged["improvement"].abs(), 98)) or 1.0
            st.subheader("Which predictor wins where")
            # "RdBu" runs red at the low end to blue at the high end, and
            # improvement is positive where the second predictor is closer.
            st.caption(
                f"Blue = |bias| smaller for {label_of(b)}; "
                f"red = smaller for {label_of(a)}."
            )
            fig = px.scatter(
                merged, x="lon", y="lat", color="improvement",
                color_continuous_scale="RdBu", range_color=(-lim, lim),
                labels={"improvement": "|bias| reduction (°C)",
                        "lon": "Longitude (°E)", "lat": "Latitude (°N)"},
            )
            fig.update_traces(marker=dict(size=3.2, symbol="square"))
            fig.update_layout(
                height=620, margin=dict(l=10, r=10, t=30, b=10),
                **_map_axes(),
            )
            st.plotly_chart(fig, use_container_width=True)
            frac = float((merged["improvement"] > 0).mean())
            st.metric(f"Pixels where {label_of(b)} is closer",
                      f"{100 * frac:.1f}%")


# ---------------------------------------------------------------------------
# Tab 3 — leaderboard
# ---------------------------------------------------------------------------

if view == "Leaderboard":
    metrics = load_aggregate("metrics")
    if metrics.empty:
        st.warning("No metrics table found.")
    else:
        st.subheader("All 100 combinations")
        metric_name = st.selectbox(
            "Metric",
            ["mae", "rmse", "bias", "bias_sd", "pearson_r", "r2",
             "ols_slope", "ols_intercept"],
            index=0,
        )
        scope_all = st.checkbox("Show every window and percentile", value=True)

        view = metrics.copy()
        if not scope_all:
            view = view[
                (view["scheme"] == sel_scheme) & (view["percentile"] == sel_q)
            ]

        pivot = view.pivot_table(
            index="predictor", columns=["scheme", "percentile"],
            values=metric_name,
        )
        pivot = pivot.reindex(
            index=[p for p in PREDICTOR_NAMES if p in pivot.index]
        )
        pivot = pivot.reindex(
            columns=sorted(
                pivot.columns, key=lambda c: (qw.SCHEMES.index(c[0]), c[1])
            )
        )
        pivot.index = [label_of(i) for i in pivot.index]

        lower_is_better = metric_name in {"mae", "rmse", "bias_sd"}
        cmap = "RdYlGn_r" if lower_is_better else (
            "RdBu_r" if metric_name in {"bias", "ols_intercept"} else "RdYlGn"
        )
        st.dataframe(
            pivot.style.format("{:.3f}").background_gradient(cmap=cmap, axis=None),
            use_container_width=True,
        )

        st.subheader("Ranking at the selected window and percentile")
        rank = (
            metrics[
                (metrics["scheme"] == sel_scheme)
                & (metrics["percentile"] == sel_q)
            ]
            .assign(predictor=lambda d: d["predictor"].map(label_of))
            .sort_values("mae")
            [["predictor", "bias", "bias_sd", "mae", "rmse",
              "pearson_r", "r2", "ols_slope", "ols_intercept", "n"]]
            .reset_index(drop=True)
        )
        st.dataframe(rank.style.format({
            c: "{:.3f}" for c in rank.columns if c != "predictor" and c != "n"
        }), use_container_width=True)

        fig = px.bar(
            metrics[metrics["percentile"] == sel_q]
            .assign(predictor=lambda d: d["predictor"].map(label_of),
                    window=lambda d: d["scheme"].map(SCHEME_LABELS)),
            x="window", y=metric_name, color="predictor", barmode="group",
            color_discrete_map=PREDICTOR_COLORS,
            category_orders={"window": [SCHEME_LABELS[s] for s in qw.SCHEMES]},
            labels={"window": "Distribution window", metric_name: metric_name},
            title=f"{metric_name} by window length at P{sel_q}",
        )
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — diagnostics
# ---------------------------------------------------------------------------

if view == "Diagnostics":
    elev = load_aggregate("elev")
    sea = load_aggregate("sea")
    win = load_aggregate("window")

    st.subheader("Bias against elevation")
    st.caption("Equal-count elevation bins. Band is ±1 SD within the bin.")
    diag_window = "ALL" if sel_window == "All windows" else sel_window
    st.caption(
        "Pooled over every window; the band mixes spatial with seasonal variation."
        if diag_window == "ALL" else
        f"Window {diag_window} only; the band is the spatial spread across pixels."
    )
    part = elev[
        (elev["scheme"] == sel_scheme) & (elev["percentile"] == sel_q)
        & (elev["window_label"].astype(str) == diag_window)
        & (elev["predictor"].isin(sel_predictors))
    ].assign(predictor=lambda d: d["predictor"].map(label_of))
    if part.empty:
        st.info("No elevation aggregate for this selection.")
    else:
        fig = px.line(part, x="bin_mid", y="mean_bias", color="predictor",
                      markers=True, error_y="bias_sd",
                      color_discrete_map=PREDICTOR_COLORS,
                      labels={"bin_mid": "Elevation (m a.s.l.)",
                              "mean_bias": "Bias (°C)"})
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bias against coarse-cell sea fraction")
    part = sea[
        (sea["scheme"] == sel_scheme) & (sea["percentile"] == sel_q)
        & (sea["window_label"].astype(str) == diag_window)
        & (sea["predictor"].isin(sel_predictors))
    ].assign(predictor=lambda d: d["predictor"].map(label_of))
    if part.empty:
        st.info("No sea-fraction aggregate for this selection.")
    else:
        fig = px.bar(part, x="bin_mid", y="mean_bias", color="predictor",
                     barmode="group", color_discrete_map=PREDICTOR_COLORS,
                     labels={"bin_mid": "Sea fraction of parent CMIP6 cell",
                             "mean_bias": "Bias (°C)"})
        fig.add_hline(y=0, line_color="black")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bias over time, one point per window")
    part = win[
        (win["scheme"] == sel_scheme) & (win["percentile"] == sel_q)
        & (win["predictor"].isin(sel_predictors))
    ].assign(predictor=lambda d: d["predictor"].map(label_of))
    if part.empty:
        st.info("No per-window aggregate for this selection.")
    else:
        fig = px.line(
            part.sort_values("window_start"),
            x="window_start", y="mean_bias", color="predictor",
            hover_data=["window_label"], color_discrete_map=PREDICTOR_COLORS,
            labels={"window_start": "Window start",
                    "mean_bias": "Domain-mean bias (°C)"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 — about
# ---------------------------------------------------------------------------

if view == "About":
    st.subheader("What this dashboard shows")
    st.markdown(
        """
The quantile-mapping (MOS) baseline compares **distributions**, not days.
There is no real day-to-day correspondence between a CMIP6 date and an observed
date, so the comparison is made between percentiles estimated over a window of
consecutive days at a single fine-grid pixel.

**Y** is the observed ERA5-Land percentile at that pixel and window.
**X** is the same percentile of a coarse-derived predictor at the same pixel and
window. `bias = X − Y`, so a positive bias means the predictor is too warm.

No correction is fitted or applied here. This is the diagnostic step that
comes before empirical quantile mapping.
        """
    )

    st.subheader("Predictors")
    st.table(pd.DataFrame([
        {"Name": PREDICTOR_LABELS["knn4"],
         "Definition": "Mean TAS of the 4 nearest CMIP6 cell centres"},
        {"Name": PREDICTOR_LABELS["knn9"],
         "Definition": "Mean TAS of the 9 nearest cell centres (the 3x3 block)"},
        {"Name": PREDICTOR_LABELS["bilinear"],
         "Definition": "Bilinear remapping of TAS to the 0.1° grid"},
        {"Name": PREDICTOR_LABELS["trilinear_fit"],
         "Definition": "Bilinear + Γ·Δz, Γ fitted per season on 1990–1999"},
        {"Name": PREDICTOR_LABELS["trilinear_fixed"],
         "Definition": "Bilinear + Γ·Δz, Γ = −6.5 °C/km"},
    ]))

    lapse = load_lapse_rates()
    if not lapse.empty:
        st.subheader("Fitted lapse rates")
        st.caption(
            "Γ from OLS of (observed − bilinear) on Δz, the sub-grid terrain "
            "height. Δz is the fine elevation minus the bilinearly "
            "interpolated CMIP6 cell-mean orography."
        )
        st.dataframe(
            lapse[["season", "gamma_per_km", "n"]]
            .rename(columns={"gamma_per_km": "Γ (°C/km)", "season": "Season"})
            .style.format({"Γ (°C/km)": "{:+.2f}", "n": "{:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

    if IS_SLIM:
        st.subheader("About this deployment")
        avail = sorted(
            {sc for p in PREDICTOR_NAMES for sc in schemes_with_points(p)},
            key=qw.SCHEMES.index,
        )
        names = ", ".join(SCHEME_LABELS.get(sc, sc) for sc in avail) or "none"
        st.markdown(
            f"""
The full pipeline output is about 700 MB, almost all of it in the per-pixel
percentile tables, so it cannot be committed to the repository. This
deployment carries the five aggregate tables plus the percentile tables for:
**{names}**.

The Leaderboard and Diagnostics views cover all four window lengths, because
the aggregates were computed over the whole grid. The Scatter view, and the
single-window option in Maps, are limited to the window lengths listed above.

To explore all 100 combinations, clone the repository and run
`python3 scripts/run_quantile_mapping.py`.
            """
        )

    st.subheader("Distribution windows")
    st.table(pd.DataFrame([
        {"Scheme": SCHEME_LABELS["14d"], "Windows": 260,
         "Note": "26 blocks per year; the last holds 15 days"},
        {"Scheme": SCHEME_LABELS["month"], "Windows": 120, "Note": "calendar month"},
        {"Scheme": SCHEME_LABELS["quarter"], "Windows": 40,
         "Note": "JFM / AMJ / JAS / OND"},
        {"Scheme": SCHEME_LABELS["year"], "Windows": 10, "Note": "calendar year"},
    ]))
    st.info(
        "A 14-day window holds only 14 values, so its P5 and P90 are noisy "
        "estimates. Read the short-window results with that in mind."
    )
