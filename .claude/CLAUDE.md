# CLAUDE.md — Project context

Read `CONTEXT.md` for the project's vocabulary and `docs/adr/` for the
decisions behind the design. Detailed guidance loads automatically from
`.claude/rules/` when you touch the files it governs — you do not need to open
those files yourself.

---

## What this project is

A statistical downscaling framework mapping coarse CMIP6 global climate model
temperature (~1° grid) onto fine ERA5-Land reanalysis temperature (0.1° grid)
over the Eastern Mediterranean and Middle East.

- **Domain:** 24–38°N, 30–38°E
- **Baseline period:** 1990–1999
- **Variable:** daily mean 2 m air temperature
- **Current baseline:** fitted, cross-validated quantile-mapping transfer
  functions — one per (land pixel, season). See `docs/adr/`.

**Planned scope:** training 1980–2004, testing 2005–2014 and 2015–2025,
projection 2081–2100, the full Mediterranean Basin (24–47°N, 11°W–40°E), six
CMIP6 GCMs, and additional predictors (T850, Z850, Z250, U850, V850, global
mean temperature).

---

## Layout

```
Thesis/
├── CONTEXT.md          # glossary — the project's vocabulary
├── docs/adr/           # why the design is the way it is
├── .claude/rules/      # guidance, loaded per file path
├── data/               # raw NetCDF (gitignored)
├── src/                # all analysis code
├── notebooks/          # 01 exploration, 02 daily OLS, 03 quantile mapping
├── scripts/            # drivers and data fetch
├── dashboards/         # Streamlit app
├── summaries/          # findings documents
├── plots/              # generated figures
├── papers/             # paper summaries + the reading app (PDFs gitignored)
└── history/            # superseded work, kept for the record
```

### What each module is for

Read the module itself for its API — these files change, and a hand-maintained
signature list in this file would drift.

| Module | Concern |
|---|---|
| `data_io.py` | Loading, calendar alignment, unit conversion, seasonal split |
| `spatial_ops.py` | Subsetting, land mask, nearest-neighbour assignment |
| `interpolation.py` | Bilinear CMIP6 → ERA5 regridding (CDO, scipy fallback) |
| `elevation.py` | ETOPO DEM download, regridding, merge |
| `predictors.py` | The five coarse-to-fine predictor fields |
| `quantile_windows.py` | Distribution windows and per-window percentiles |
| `qm_inputs.py` | Shared loading for both quantile-mapping drivers |
| `qm_nodes.py` | Season/fold bookkeeping, q–q node reduction |
| `qm_transforms.py` | The transfer functions, one interface |
| `qm_cv.py` | Leave-one-season-year-out cross-validation |
| `qm_eval.py` | Metrics and the climatology floors |
| `qm_metrics.py` | Skill metrics and aggregates for the diagnostic grid |
| `visualization.py` | All plotting; the figure standards live here |
| `qm_visualization.py`, `qm_transform_viz.py` | Quantile-mapping figures |
| `vis_constants.py` | Every style constant — edit here, never inline |

---

## Invariants that apply everywhere

**Interpreter.** Use `python3.10`, never plain `python3` — the latter has no
xarray and every script fails with `ModuleNotFoundError`. Install with
`python3.10 -m pip`; run Streamlit with `python3.10 -m streamlit`.

**Units.** Convert Kelvin to Celsius before analysis. Cached predictor fields
are already in °C — do not convert twice.

**Calendars.** ERA5-Land leap days are removed, leaving 3,650 shared days for
1990–1999.

**Latitude runs descending**, 38 → 24.

**Two season conventions exist.** `make_windows("quarter")` is JFM/AMJ/JAS/OND;
`make_windows("season")` is DJF/MAM/JJA/SON. New code uses only `"season"`.
See ADR-0001.

**Bias is predictor minus target** by default. One visualization function uses
the opposite sign on purpose. See ADR-0013.

**No analysis logic in notebooks.** It goes in `src/`.

**No test framework.** After changing the transfer functions, run
`python3.10 scripts/sanity_qm_transforms.py`.

---

## Running things

```bash
python3.10 scripts/run_qm_transforms.py          # fitted baseline, ~45 min
python3.10 scripts/run_quantile_mapping.py       # diagnostic grid, ~23 min
python3.10 scripts/sanity_qm_transforms.py       # correctness checks
python3.10 -m streamlit run dashboards/qm_dashboard.py
```

Both drivers are idempotent and skip work already cached under `data/cache/`.

---

## Data location

Raw data comes from the Moriah cluster at HUJI via the jump host
`bava.cs.huji.ac.il`. Use `scripts/fetch_remote_climate_data.sh`.

---

## People

- **Ronit Nirel** — project lead
- **Dorita Morin** — supervisor
- **Efrat Morin** — group PI
- **Anton Gelman** — CMIP6 data on Moriah
- **Andre Klif** — GCM list and ERA5 paths
- **Chaim** — ensemble member selection
