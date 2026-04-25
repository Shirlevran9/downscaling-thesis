# Findings Presentation Guidelines

Guidelines for turning the initial analysis notebook into a well-structured
scientific presentation or report.  Distilled from supervisor feedback
(Ronit Nirel, 17 April 2026) and project-wide writing conventions.

---

## 1. Document Structure

Organise the written report according to the IMRaD framework.  The current
notebook mixes presentation order with execution order; the report should
reorder content by topic.

| Report section | What goes here |
|---|---|
| **Introduction** | Study region and motivation; climate context; research question |
| **Methods** | ERA5-Land and CMIP6 data sources; calendar alignment; unit conversion; domain and land–sea mask definition; CMIP6–ERA5 spatial pairing (padding, nearest-neighbour assignment); sea-fraction computation; global daily mean computation; regression model specification |
| **Results** | Descriptive statistics; spatial climatology comparison; temporal variability; pairing statistics; regression performance; residual diagnostics |
| **Discussion** | Interpretation of biases and residual patterns; comparison with literature; model limitations; outlook for improvement |

Items to move from the current "Results" section to **Methods**:
- The CMIP6–ERA5 spatial linking procedure (currently in Section 5 of the notebook).
- The definition of sea fraction per CMIP6 cell.
- The computation of the global daily area-weighted mean TAS.

---

## 2. Writing Style

- **Dry and precise.** Scientific writing in this field avoids superlatives.
  Replace "near perfect agreement" with "Pearson r = 0.90" and let the
  number convey the quality.
- **No dramatic adjectives.** Do not use "excellent", "remarkable",
  "surprisingly", or similar intensifiers.
- **Hedged explanations.** When proposing a mechanism, write "this may be
  explained by …" or "one possible factor is …". Do not claim completeness.
- **Results refer to variables.** Write "TAS exhibits a warm bias of ~1.5°C"
  not "CMIP6 exhibits a warm bias".
- **No bold in running sentences.** Bold is for headings and table/figure
  labels only.
- **Trust the heading.** Do not open a section with a sentence that paraphrases
  the heading.  Begin directly with content.
- **Monotonicity.** If a pattern is not strictly monotonic, describe it
  accurately rather than collapsing it to a single directional statement.
- **Paragraph structure** (preferred over deeply nested subsections): Use
  an italic lead phrase at the start of a paragraph to signal its topic,
  rather than creating a dedicated sub-subsection for two sentences.

  Example:
  > *Calendar processing.* ERA5-Land uses a Gregorian calendar, while
  > CESM2-WACCM uses a no-leap calendar …

---

## 3. Figure Organisation Rules

### Numbering and panels

- Figures are numbered **Fig. 1, Fig. 2, …** Sub-panels are labelled **(a),
  (b), (c), (d)** in the figure itself and referenced in the caption.
- **No double hierarchy** such as "Fig. 1a1" or "Fig. 3.2b".
- Do not split a single multi-panel figure across different sections.

### Titles

- The figure title describes **what is displayed**, not what it means.
  - ✓ "OLS residual distribution by CMIP6 cell sea fraction, 1990–1999"
  - ✗ "Coastal cells produce larger residuals due to mixed land–sea signal"
- Avoid findings, conclusions, or superlatives in titles.

### Captions

- Captions must be **self-contained**: a reader should understand the
  figure without looking at the text.
- Include: dataset name, variable, period, study region, and a brief
  description of each panel.

### Font sizes

All font sizes are controlled by `src/vis_constants.py`:
- Panel titles: 15 pt
- Axis labels: 13 pt
- Tick labels: 11 pt
- Legend: 11 pt

These must remain legible after reduction to a single column width (~8 cm).

### Maps

- Add a coarse-grid overlay (CMIP6 ~1° cell boundaries) on any
  fine-resolution map panel to visualise the resolution mismatch.
  Pass `show_coarse_grid=True, coarse_lats=..., coarse_lons=...` to the
  spatial plot functions.
- Always correct the geographic aspect ratio for the domain's mean latitude
  (use `visualization.make_spatial_figure()` or `apply_map_formatting()`).
- Use `highlight_box=dict(lat_min=..., lat_max=..., lon_min=..., lon_max=..., label=...)`
  to mark a sub-region of interest with a bold red rectangle. Currently applied
  to the 32–34°N, 34–36°E area (largest ERA5–CMIP6 bias) in Fig 2 and Fig 4.

### Time series

- **Preferred order** (most aggregated first):
  1. Long-term trend with confidence band
  2. Seasonal / monthly climatology with confidence band
  3. Daily time series
- **Fig 3b (monthly climatology):** 90% t-CI on the inter-annual mean per
  calendar month (n = 10 yr); auto-computed by `plot_monthly_climatology`.
  Legend entry: "Shaded: 90% CI on inter-annual mean (±t·SE, n=10 yr)".
- **Fig 3 / Fig 3c (monthly & annual aggregated):** ±1 SD envelope of daily
  values; passed via `fill_dict` to `plot_domain_timeseries`.
  Legend entry: "Shaded: ±1 SD of daily values within each period".
- **Fig 10:** no CI band — `show_trend_band=False` (single point per date).
- When the seasonal cycle has already been presented, consider showing the
  ERA5–CMIP6 *difference* series rather than overlapping the two raw series.

---

## 4. Specific Figure Changes (from Supervisor Review)

| Figure | Current state | Recommended change |
|---|---|---|
| **Fig. 2a + Fig. 5** | Separate KDE (Fig. 2a) and boxplot (Fig. 5) | **Done** — replaced by `plot_combined_temperature_distribution`: panel (a) KDE, panel (b) boxplots, no percentile lines |
| **Fig. 4** | Old 4-row × 2-col layout | **Done** — restructured to 2 rows × 5 cols (DJF+MAM / JJA+SON side by side with narrow separator); highlight box on 32–34°N, 34–36°E |
| **Fig. 7** | Two panels: hexbin + cell-mean scatter | Keep only the hexbin panel (a); remove the blue cell-mean scatter panel (b) as it adds little beyond the hexbin |
| **Fig. 11** | Residuals by sea fraction | Keep; this is the primary spatial-mixing diagnostic |
| **Fig. 12** | Residuals by pixel count | Remove — largely redundant with Fig. 11 (sea fraction and pixel count are collinear for coastal cells) |
| **New fig.** | — | Add a two-panel map: (a) mean T2M spatial field, (b) mean OLS residual spatial field. Seeing both together is informative. |
| **Fig. 3 (daily)** | Two overlapping time series | Once seasonal cycle is described, optionally replace with a single ERA5–CMIP6 difference time series |
| **Fig. 10** (quarterly trend) | OLS trend lines only | `show_trend_band=False` — only 10 annual points, one value per date; a trend band is not statistically meaningful here |

---

## 5. Statistical Reporting Rules

### Always pair central and dispersion measures

- Mean: report as "19.2 ± 7.8°C" (mean ± SD).
- Median: report as "19.8 [10.2–28.4]°C" (median [IQR]).
- Never report a central measure alone without a dispersion measure.

### Residual diagnostics

Before interpreting regression results, verify and report:

1. **Homoscedasticity** — residuals vs. predictor scatter.
2. **Linearity** — residuals vs. predictor (same plot, check for curvature).
3. **Temporal independence** — ACF/PACF of daily domain-mean residuals;
   report Durbin–Watson statistic.
4. **Spatial independence** — Moran's I on mean-residual map.
5. **Normality** — histogram of standardised residuals with Gaussian overlay.

All diagnostics use **standardised residuals** (divide raw residual by
residual SD), not raw residuals.

---

## 6. Table Formatting Rules

- **Uniform decimal places** per variable (e.g. one decimal for temperatures).
- **Units in the header**, not in every cell.
- **Layout**: variables in rows, statistics in columns.
- **Column order**: Mean → Std Dev → Median → IQR → Min → Max.
- When the study extends to additional variables (T850, Z850, etc.), add
  them as new rows; do not create new tables.

---

## 7. Proposed Model Extensions (Supervisor Notes)

The following predictors and effects are suggested for the next modelling
phase:

| Addition | Rationale |
|---|---|
| Sea fraction as main effect + interaction with TAS | Coastal CMIP6 cells average over mixed land–sea temperatures; the bias likely varies with fraction |
| Global daily mean TAS | Large-scale thermodynamic signal; removes global warming trend from the residuals |
| Lagged TAS (t−1, t+1 days) | Temporal autocorrelation in residuals suggests unresolved heat storage or advection effects |
| TAS at neighbouring CMIP6 cells | Spatial autocorrelation; captures mesoscale gradients not resolved by coarse grid |
| Random intercept per coarse CMIP6 cell | Accounts for clustering of fine pixels within the same coarse cell; standard multilevel model term |

**Open questions for the next meeting:**
- How to detrend residuals before analysing seasonal/temporal patterns?
- How to neutralise the seasonal cycle in the residuals when fitting the
  extended model?
