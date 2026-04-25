# Downscaling Investigation: Initial Findings

*Study domain:* 24–38°N, 30–38°E (Eastern Mediterranean and Middle East)
*Study period:* 1990–1999

---

## 1. Introduction

Global climate models (GCMs) operate at spatial resolutions of approximately
1°, insufficient for regional impact assessments that require fine-scale
temperature information. Statistical downscaling addresses this gap by
learning the empirical relationship between coarse GCM output and
observed fine-resolution data, then applying that relationship to
future projections.

This report documents the initial exploratory analysis for a statistical
downscaling framework applied to the Eastern Mediterranean and Middle East
(EMME) region. The predictor variable is daily near-surface air temperature
(TAS) from the CMIP6 model CESM2-WACCM at approximately 1° resolution; the
predictand is daily 2 m temperature (T2M) from the ERA5-Land reanalysis at
0.1° resolution. The analysis characterises the relationship between these
two variables, documents the spatial pairing mechanism, and evaluates an
initial ordinary least-squares (OLS) regression model.

---

## 2. Methods

### 2.1 Data Sources

*ERA5-Land T2M.* The target variable is daily mean 2 m temperature from the
ERA5-Land reanalysis (Muñoz-Sabater et al., 2021) at 0.1° × 0.1° resolution
(approximately 11 km). The dataset covers land pixels only within the study
domain and spans the period 1990–1999.

*CMIP6 CESM2-WACCM TAS.* The predictor variable is daily near-surface air
temperature from the CESM2-WACCM model (Eyring et al., 2016) at approximately
0.94° × 1.25° resolution (approximately 100 km). The dataset covers both land
and ocean grid cells within the study domain over the same period.

### 2.2 Preprocessing

*Calendar alignment.* ERA5-Land uses a Gregorian calendar and includes leap
days; CESM2-WACCM uses a no-leap calendar. Leap days (1992-02-29 and
1996-02-29) were removed from the ERA5-Land record, yielding 3,650 shared
days across both datasets.

*Unit conversion.* Both datasets are stored in Kelvin. All values were
converted to Celsius by subtracting 273.15 K.

*Land–sea mask.* ERA5-Land provides values for land pixels only; ocean
pixels are represented as missing values and were excluded from spatial
averages and model fitting. Approximately 32.7% of ERA5-Land grid cells
within the study domain are classified as ocean.

### 2.3 Spatial Pairing (CMIP6 → ERA5-Land)

Linking each ERA5-Land fine-resolution pixel to a CMIP6 coarse-grid predictor
requires a spatial pairing strategy. Multiple approaches were considered; the
two described below represent the current implementation and a planned
extension respectively.

*Nearest-neighbour assignment.* Each ERA5-Land pixel was assigned to the
nearest CMIP6 cell by minimising Euclidean distance in degrees along latitude
and longitude independently. For each paired CMIP6 cell the number of
assigned fine-resolution pixels was recorded; both descriptive statistics and
a spatial heatmap of pixel counts per cell were produced to characterise the
pairing.

*Bilinear interpolation *(planned).** Rather than assigning each pixel to a
single coarse cell, bilinear interpolation constructs the predictor value at
each ERA5-Land pixel location as a weighted average of the four surrounding
CMIP6 cell centres, with weights inversely proportional to distance. This
approach produces a spatially smoother predictor field and avoids the hard
boundaries introduced by nearest-neighbour assignment, but requires all four
neighbouring cells to lie within the extraction window and is not yet
implemented.

*Domain padding.* To ensure that coarse cells straddling the domain boundary
were retained under both approaches, the CMIP6 extraction used a padding of
±1.0° in latitude and ±1.5° in longitude beyond the nominal domain extent.

*Sea-fraction definition.* For each CMIP6 cell, the sea fraction was
computed as the proportion of the cell area classified as ocean (NaN) in
the ERA5-Land land–sea mask. Cells with a high sea fraction straddle the
coastline or lie predominantly over water.

### 2.4 Global Daily Mean TAS

An area-weighted global daily mean TAS was computed from the full CMIP6
global field by weighting each cell by the cosine of its latitude. This
quantity provides a large-scale thermodynamic signal for use as an
additional predictor in later model phases.

### 2.5 Spatial Analysis

*Spatial climatology.* The 10-year temporal mean was computed independently
at each grid cell to produce mean temperature maps for T2M and TAS, allowing
direct visual comparison of the fine-resolution ERA5-Land field against the
coarse CMIP6 field at their native resolutions.

*Seasonal means.* Daily values were grouped into the four meteorological
seasons — DJF (December–February), MAM (March–May), JJA (June–August),
and SON (September–November) — and averaged to produce seasonal climatology
maps for both variables side by side.

### 2.6 Time Series Analysis

*Domain-mean time series.* A single value was obtained for each variable on
each day by averaging all grid cells within the domain. For T2M, ocean pixels
are absent (NaN) in the ERA5-Land dataset, so the mean is effectively a
land-only average. For TAS, the CMIP6 field covers both land and ocean cells,
so the domain mean includes ocean grid cells. The two series are therefore
averages over different effective domains, a distinction that may contribute
to differences in their mean values. The daily domain-mean series were
subsequently aggregated to monthly and annual means. For all three
aggregations (daily, monthly, annual), a shaded band spanning ±1 SD of the
values within each plotted period was added to each line.

*Annual cycle.* For each calendar month (January through December), the
climatological mean was computed as the average of the corresponding monthly
means across all ten years (1990–1999). A shaded band spanning ±1 SD of the
ten annual means is drawn around each monthly value to represent year-to-year
spread.

*Warming trend analysis.* Detecting a decade-scale warming trend in
domain-mean temperature is complicated by strong year-to-year variability.
Three complementary approaches were used. First, the domain-wide spatial
minimum and maximum were tracked at the daily, monthly, and annual
timescales as an exploratory screen. Second, to remove the confounding
effect of the seasonal cycle, values were extracted on a fixed calendar
date — 1 August — for each year from 1990 to 1999, both domain-wide
(spatial min and max) and at a single pixel/cell nearest to 35°E, 32°N
(central Levant). An OLS linear trend was fitted to the ten extracted
values at the single location. Third, this fixed-date extraction was
extended to four quarterly dates (1 January, 1 April, 1 July, 1 October)
at the same location to assess whether any trend signal varies by season.
All trend analyses were performed independently for T2M and TAS.

### 2.7 Statistical Model

The initial model is an ordinary least-squares (OLS) regression of T2M
on TAS at the pixel-day level:

> T2M = α + β · TAS + ε

where α is an intercept, β is the regression slope, and ε is the residual.
Model performance was assessed using Pearson *r*. Residual diagnostics
examined homoscedasticity, spatial structure, and seasonal patterns, and
were computed on standardised residuals (raw residuals divided by residual
standard deviation).

### 2.8 Planned Predictor Extensions *(work in progress)*

The following predictors are under evaluation for inclusion in subsequent
model phases:

- Global daily mean TAS as a large-scale thermodynamic covariate (§2.4)
- Sea fraction as a main effect and interaction term with TAS
- TAS at neighbouring CMIP6 cells (spatial context)
- Lagged TAS values (t−1, t+1 days) to capture heat storage effects
- A random intercept per CMIP6 cell to account for the clustering of
  fine-resolution pixels within each coarse cell

---

## 3. Results

### 3.1 Descriptive Statistics

Summary statistics for T2M and TAS over all land pixels and shared days are
given in Table 1. TAS is systematically higher than T2M by approximately
1.6 °C in the domain mean, with a somewhat smaller standard deviation.

**Table 1.** Descriptive statistics for daily 2 m temperature, 1990–1999,
land pixels within 24–38°N, 30–38°E. Units: °C.

| Variable | Mean (°C) | Std Dev (°C) | Median (°C) | IQR (°C) | Min (°C) | Max (°C) |
|---|---|---|---|---|---|---|
| T2M (ERA5-Land, 0.1°) | 19.2 | 8.7 | 20.1 | 13.6 | −21.3 | 40.6 |
| TAS (CMIP6 CESM2-WACCM, ~1°) | 20.8 | 7.9 | 21.2 | 12.0 | −14.7 | 41.5 |

The marginal distributions of T2M and TAS are shown in Fig. 2. Both
variables display an approximately unimodal distribution with a pronounced
left tail, reflecting the relatively rare but extreme cold spells that occur
in the northern elevated parts of the domain in winter. The warm bias of TAS
relative to T2M is visible as a rightward shift of the TAS distribution; the
difference is consistent across the full range of temperatures.

[Fig. 2 — temperature distribution: KDE density curves and boxplots, T2M and TAS, 1990–1999]

### 3.2 Temporal Variability

[Fig. 3 — domain-averaged T2M and TAS: (a) daily, (b) monthly, (c) annual, 1990–1999]

*Monthly climatology.* The domain-mean monthly climatology shows a
pronounced seasonal cycle in both T2M and TAS, peaking in July–August and
reaching a minimum in January [Fig. 2]. TAS is higher than T2M throughout
the year by approximately 1–2 °C. The shaded band around each climatological
mean spans ±1 SD of the 10 annual means for that calendar month, representing
year-to-year spread; the width is similar for both variables.

*Monthly and annual trend.* The three panels of Fig. 3 show the domain mean
at daily (a), monthly (b), and annual (c) resolution. In panels (b) and (c)
the shaded band spans ±1 SD of daily domain-mean values within each aggregation
period. Neither the monthly nor the annual aggregation shows a statistically
conclusive warming trend at the domain scale across the decade.

*Warming signal at 35°E, 32°N.* To remove the confounding effect of the
seasonal cycle, August (1 August) values were extracted for each year from
1990 to 1999. The domain-wide spatial range of T2M and TAS on that date
is shown in Fig. 4a. At a single land pixel nearest to 35°E, 32°N (central
Levant), an OLS trend fitted to the ten 1-August values yields approximately
+0.22 °C per year for T2M and +0.12 °C per year for TAS [Fig. 4b]. These
estimates rest on ten data points and should be interpreted with caution.
The same fixed-date extraction applied to four quarterly dates (1 January,
1 April, 1 July, 1 October) indicates that the trend signal, where present,
varies by season [Fig. 5].

### 3.3 Spatial Climatology

[Fig. 5 — mean T2M and TAS maps, 1990–1999]

The 10-year mean T2M map shows a marked spatial gradient, with the coldest
values over the elevated terrain of northern Turkey and the warmest values
over the southern Arabian Peninsula and Jordan Valley [Fig. 5a]. The
corresponding TAS map reproduces the large-scale pattern but lacks the
fine-scale spatial structure visible in T2M, particularly over mountain
ranges. TAS is higher than T2M across most of the domain, most notably over
elevated terrain, which may reflect the coarse model's limited resolution of
orographic effects.

[Fig. 6 — seasonal comparison maps, ERA5-Land vs CMIP6]

The seasonal climatology comparison shows that both variables exhibit a
similar seasonal progression across the domain [Fig. 6]. The warm bias in
TAS relative to T2M is present in all four seasons, though its magnitude
varies spatially and seasonally. The region 32–34°N, 34–36°E (highlighted
in Fig. 6) shows some of the largest discrepancies between TAS and T2M,
possibly because the coarse grid undersamples cold mountainous terrain in
that area.

### 3.4 Pairing Statistics

The nearest-neighbour assignment linked 7,683 ERA5-Land land pixels to
97 CMIP6 cells. The number of fine pixels per coarse cell has a mean of
79.2 (SD: —; range: 1–130). The high variability is largely attributable
to coastline and boundary effects: cells that straddle the land–sea boundary
receive far fewer ERA5-Land pixels than cells that lie entirely over land.

[Fig. 7 — pixel-per-cell heatmap]

### 3.5 OLS Model Performance

[Fig. 8 — hexbin scatter plot, T2M vs TAS, pixel-day level]

At the pixel-day level (all land pixels, all 3,650 days), T2M and TAS are
strongly correlated (Pearson *r* ≈ 0.90). The OLS regression yields a
non-zero intercept, indicating a systematic offset that cannot be removed
by scaling alone. The scatter around the regression line represents the
sub-grid spatial heterogeneity in T2M that TAS does not capture — elevation
differences, land cover variation, and proximity to the coastline.

### 3.6 Residual Diagnostics

[Fig. 9 — residual diagnostics panel]

*Distance from cell centre.* Standardised residuals show no systematic
dependence on the distance of each ERA5-Land pixel from its assigned CMIP6
cell centroid. The distribution of residuals is approximately uniform across
distances from 0° to 0.7°, with a marginal increase in spread in the
0.3–0.5° range.

*Sea fraction.* CMIP6 cells were grouped into five sea-fraction bins. The
two bins with the highest sea fractions (predominantly coastal cells) show
somewhat elevated median standardised residuals relative to the three
lower-fraction bins. The pattern is not monotonic across the full range of
sea fractions [Fig. 9b].

*Seasonal and monthly patterns.* The distribution of standardised residuals
varies across meteorological seasons (DJF, MAM, JJA, SON) and calendar
months [Fig. 9c–d]. A systematic seasonal pattern would indicate that a
single global model cannot fully capture seasonal variability and that
season-specific terms may be warranted in later model phases.

---

## 4. Discussion

TAS provides a strong large-scale predictor for T2M, with Pearson *r* ≈ 0.90
at the pixel-day level. However, a systematic warm bias of ~1–2 °C and a
non-zero regression intercept indicate that a simple linear rescaling is
insufficient, and that additional predictors are needed to capture the
sub-grid spatial heterogeneity visible in the residuals.

The residual analysis suggests that proximity to the coastline may be a
relevant factor: cells with high sea fractions show somewhat larger residuals,
which may reflect the coarse model's averaging of land and ocean temperatures
within a single grid cell. The absence of a relationship between residual
magnitude and distance from the cell centre suggests that the nearest-neighbour
assignment does not introduce a strong spatial bias, at least at the scales
examined here.

Completing the residual diagnostic checklist — including temporal
autocorrelation (ACF/Durbin–Watson) and spatial autocorrelation (Moran's I) —
is a priority for the next phase, as unresolved autocorrelation would affect
the reliability of inferential statistics. Planned model extensions are
described in §2.8.
