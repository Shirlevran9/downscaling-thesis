# Quantile mapping cannot downscale: the inflation problem

## Title

**Bias Correction, Quantile Mapping, and Downscaling: Revisiting the Inflation
Issue**

Douglas Maraun (GEOMAR Helmholtz Centre for Ocean Research Kiel, Germany)

*Journal of Climate* **26**, 2137–2143, 15 March 2013.
doi:10.1175/JCLI-D-12-00821.1 · received 21 November 2012, final form
7 January 2013.

PDF: `Maraun2013_QM_InflationIssue_JClimate.pdf`

## Abstract

Quantile mapping is used routinely to correct the biases of regional climate
model output against observations. The paper separates two uses of it. When the
observations have about the same resolution as the model, the method only
corrects a bias, and the author accepts it as reasonable: "If the observations
are of similar resolution as the regional climate model, quantile mapping is
a feasible approach" (Abstract). When the observations are much finer, the same
method is also being asked to bridge a scale gap. The paper's claim is that this
second use is not valid, and that it repeats a mistake already identified for
variance inflation in perfect prognosis downscaling.

The argument is made with one worked example: daily precipitation from a 25-km
regional climate model grid box mapped onto 20 rain gauges inside that grid box
in the Harz Mountains, Germany. Four consequences are reported. The spatial and
temporal structure of the corrected series stays that of the grid box, not of
the gauges. The drizzle correction produces too many completely dry days over
the area. Area-mean extremes are overestimated by roughly 30 %. Trends in
seasonal totals and seasonal maxima are changed, and strong trends are
amplified. The author concludes that a deterministic correction cannot do this
job and that stochastic bias correction is needed instead.

## Terms and notation

### Terms

**Inflation** is the practice of rescaling a predicted time series so that its
variance matches the observed variance. The paper takes the term and the
objection from von Storch (1999): inflation "does not add any unexplained
variability and therefore wrongly assumes that all local variance is indeed
completely explained by the chosen large-scale predictors" (Sect. 1). Its direct
effect is an increase in the root-mean-squared error.

**Randomization** is the alternative von Storch (1999) recommends: adding random
small-scale variability instead of rescaling.

**Perfect prognosis (PP)** downscaling assumes a relationship between
large-scale predictors and local-scale predictands, fitted on observations.
**Model output statistics (MOS)** instead post-processes model output directly
against observations. The paper places bias correction, variance correction and
quantile mapping inside the MOS family, and this is the point of the paper: the
inflation problem known for PP also appears in MOS.

**Bias correction** here means a deterministic post-processing of the marginal
distribution of the model data. "A specific simulated value will always yield
the same corrected value, and the spatiotemporal dependence is not explicitly
altered" (Sect. 1). The author names the implicit assumption this carries: that
all local-scale spatial and temporal variability is already determined by the
simulated grid-box variability, apart from an adjustment of the marginal
distribution.

The **marginal distribution** is the time-independent distribution of the
variable at one location, taken without regard to the spatial or temporal
dependence. The paper notes in a footnote that its empirical equivalent is the
histogram of the observations.

**Quantile mapping** is the method under test: it removes quantile-dependent
biases by mapping simulated quantiles onto observed quantiles. The **variance
correction** is treated throughout as a special case of it.

The **drizzle effect** is the tendency of a climate model to produce too many
days with very small precipitation. It is corrected by a wet-day threshold.

A **perfect boundary condition** setting means the regional model is driven by
reanalysis rather than by a global climate model, so simulated and observed
weather run roughly in step.

### Notation

The trend models of the appendix, Equations (A1) to (A4):

| Symbol | Meaning |
|---|---|
| $y_i$ | Seasonal total, or seasonal maximum, precipitation in year $t_i$ |
| $t_i$ | Year index, $i = 1 \dots N$ |
| $\mu_i$ | Time-dependent location (mean) parameter |
| $\sigma$ | Constant width of the normal model |
| $\sigma_i$ | Time-dependent scale parameter of the GEV model |
| $\xi$ | Constant shape parameter of the GEV model |
| $a$, $b$ | Intercept and slope of the linear time dependence |
| $E_i$ | Expected seasonal maximum in year $t_i$ |
| $\Gamma(\cdot)$ | The gamma function |

An **absolute trend** is given in millimetres per decade. A **relative trend** is
given in percent per decade, taken relative to the expected value for the year
1985.

## Previous work

The paper builds directly on von Storch (1999), which showed for perfect
prognosis downscaling that inflation is not meaningful, and recommended
randomization instead. Maraun et al. (2010) is cited as the review of the
stochastic downscaling approaches that followed. The paper's contribution is to
carry the same objection across into model output statistics.

On the bias correction side, Christensen et al. (2008) is cited as the source of
the wish to post-process model output to match the observed climate, and also,
together with Maraun (2012), as evidence of a separate known problem: biases are
not stationary in time. Hay and Clark (2003) and Piani et al. (2010) are cited
for the wet-day threshold approach to the drizzle effect, which the paper then
uses.

The author identifies one specific position in the literature as too optimistic.
"Eden et al. (2012) argue that model errors caused by parameterization and
orography can reasonably be corrected by bias correction" (Sect. 4). The reply
is a distinction between two things: a model error, which can be corrected, and
the mismatch caused by unresolved small-scale variability, which the author
calls "an additional discrepancy—not error—between model and observations"
(Sect. 4). The paper's case is that quantile mapping cannot remove the second
one.

Two side routes are noted rather than tested. Covariate-dependent quantile
mapping "may be applied with randomization (Kallache et al. 2011)" (footnote 3).
Distributed hydrological models (Xu 1999; Das et al. 2008) are given as the
motivating application that creates the demand for kilometre-scale fields.

## Problem definition

### Problem

The target is not to improve quantile mapping but to test whether it is valid
at all when the observations are much finer than the model. The author sets out
the condition clearly. Correcting a model against a gridded dataset of the same
resolution "might in principle be a valid assumption for a pure bias
correction". But if the correction is against station data or very-high-
resolution gridded data, then "deterministic variance correction and quantile
mapping approaches are not feasible" (Sect. 1).

The reason given is structural. Grid-box variability is much smoother than
local variability. Quantile mapping only corrects the marginals and generates no
new small-scale variability, so the temporal dependence, and the spatial
dependence between locations in different grid boxes, remain those of the grid
box. Inside one grid box the situation is stronger still: because the mapping is
deterministic, "within a grid box the spatial dependence between locations is
fully deterministic" (Sect. 1). The method is therefore rescaling the simulated
series in an attempt to explain variability it has not explained — which is
inflation.

Note the status of this claim. The argument that a deterministic mapping cannot
add variability is a logical one, stated rather than tested. What the paper
tests is the size of the resulting problems in one case.

### Models

Two separate pieces of modelling appear.

**The correction.** A simple empirical quantile mapping. Because the observed
and simulated series were of equal length, and no separate validation period was
used, "the simulated quantiles could be directly mapped onto the observed
quantiles and no interpolation had to be carried out" (Sect. 2). Where the
observations had missing values, the matching simulated values were dropped so
that the two series stayed the same length. The drizzle effect was corrected
with a wet-day threshold of 1 mm day⁻¹ on the observations. The mapping was
carried out separately for winter and summer. No equation is given for the
mapping itself. By construction, the corrected model reproduces the observed
marginal distribution exactly.

**The trend models**, in the appendix. Seasonal totals are modelled by linear
regression, Equation (A1):

$$y_i \sim \mathcal{N}(\mu_i, \sigma) \quad \text{with} \quad \mu_i = a + b\,t_i \qquad (A1)$$

Seasonal maxima are modelled by a generalized extreme value distribution
(Coles 2001), Equation (A2):

$$y_i \sim \mathrm{GEV}(\mu_i, \sigma_i, \xi) \qquad (A2)$$

with location and scale depending on time and the shape parameter held constant.
The time dependence is linear, Equation (A3):

$$\mu_i = a_\mu + b_\mu t_i \quad \text{and} \quad \sigma_i = a_\sigma + b_\sigma t_i \qquad (A3)$$

The expected seasonal maximum then follows from Equation (A4):

$$E_i = \mu_i - \frac{\sigma_i}{\xi} + \frac{\sigma_i}{\xi}\,\Gamma(1 - \xi) \qquad (A4)$$

### Data

The regional model is REMO from the Max Planck Institute of Meteorology
(Jacob 2001), on a 25-km rotated grid, taken from the ENSEMBLES project. It is
driven by ERA-40 boundary conditions for 1961–2000. The author states the reason
for this choice and its limits: a perfect boundary setting avoids problems caused
by global model biases and roughly synchronizes simulated and observed
precipitation, but because quantile mapping sees only the marginal
distributions, "this temporal agreement is irrelevant for the analysis. The
conclusions would be the same for an RCM driven by GCM boundary conditions"
(Sect. 2). That last sentence is an assertion; no GCM-driven run is shown.

One grid box is studied, chosen mainly because it contains more than 20 rain
gauges. The paper prints its centre as "11.00°N, 51.64°E" (Sect. 2), which does
not match the Harz Mountains in central northern Germany or the gauge
coordinates in Table 1; the latitude and longitude appear to have been swapped
in the text, and the intended centre is near 51.64°N, 11.00°E. Twenty gauges
with long enough records were selected. Table 1 lists them with coordinates,
elevations between 140 m and 523 m, and record periods that mostly start in 1961
or 1969 and end in 2000. Three gauges belong to the catchment of the Helme and
the rest to the Bode.

Deliberately, there is no validation period: "The effect to be demonstrated
occurs already in the calibration period; therefore, I deliberately do not
choose a separate validation period" (Sect. 2).

### Evaluation metrics

The paper uses no skill score and computes no error statistic such as MAE or
RMSE. The evidence is of three kinds:

- **Quantile–quantile plots**, first per gauge (Fig. 2) and then for the
  area mean of all 20 gauges against the area mean of the 20 corrections
  (Fig. 4).
- **Visual inspection of time series** across the 20 gauges for example
  winters and summers (Fig. 3), used to judge spatial variability.
- **Trend estimates** from the models of Equations (A1) to (A4), reported as
  absolute trends in mm decade⁻¹ and relative trends in % decade⁻¹.

Because there is no skill score, the paper cannot and does not rank quantile
mapping against any competing correction method. No alternative method is
implemented.

## Main results

**The correction works on the marginals, by construction.** In winter the
uncorrected model heavily underestimates observed precipitation at Thale (Harz)
and produces too many drizzle days; in summer the effect is similar, although
the model does produce some high rainfall events matching observed heavy
precipitation. After correction the marginal distribution of the observations is
reproduced exactly. The Q–Q values behind this are shown only in Figure 2 and
are not tabulated.

**The spatial structure is wrong.** In the observations, rain at some gauges can
coincide with dry conditions at others, and extreme events are spatially
localized, more so in summer than in winter. The corrected model cannot do this:
"because quantile mapping is deterministic, a high (modest) RCM gridbox
precipitation value is always transformed into a high (modest) local value"
(Sect. 3). Two consequences follow. Extreme events always cover the whole grid
box, so their spatial extent is exaggerated. When the model simulates drizzle,
the correction "in most cases leads to complete dryness across all gauges", so
the drizzle effect is overcorrected.

**The ranking of gauges is frozen.** For a given quantile the order of
precipitation across gauges can never change, and in most cases it is the same
at all quantiles. The only exception the author allows is where one gauge's
transfer function crosses another's, which can change the order between high and
low quantiles. The worked case: the gauge at Stiege sits on a hill and is on
average wetter than Thale (Harz) in a valley. In reality the valley is wetter on
some days; in the corrected series "this will never occur in the deterministic
quantile mapping case" (Sect. 3).

**Area means are distorted in both tails.** Averaging the 20 corrected series
and comparing with the average of the 20 observed series, the corrected model
"simulates too many area-mean dry days" and "strongly overestimates area-mean
extreme events by roughly 30%" (Sect. 3). The 30 % figure is the only number the
text gives for this result; the rest is in Figure 4.

**Trends are changed, and the change depends on the strength of the trend.**
All the numbers below are for Thale (Harz), and describe how quantile mapping
changes the trend of the corrected model series. The signs need care: the trends
are negative, and the paper reports the increase in their magnitude.

| Quantity | Change in absolute trend | Change in relative trend | Author's reading |
|---|---|---|---|
| DJF seasonal total | negative trend increased by 0.8 mm decade⁻¹ | +27.7 % | inflation and deflation evenly spread in time; trend "only marginally increased" |
| JJA seasonal total | negative trend increased by 3.9 mm decade⁻¹ | +11.7 % | the strong negative trend puts inflation mainly early in the series |
| DJF seasonal maximum | negative trend increased by 0.9 mm decade⁻¹ | +85.6 % | asymmetric amplification |
| JJA seasonal maximum | 0.28 mm decade⁻¹ | 30.1 % | trend weak, no time-dependent inflation, effect "negligible" |

For the two rows where the trend was weak, DJF totals and JJA maxima, the author
explicitly dismisses the large relative numbers as "not relevant" (Sect. 3). The
mechanism is clearest in the winter maxima: the highest simulated values, which
fall early in the record, are amplified by about 80 %, while the lower values
later in the record are amplified by only about 30 %.

**The result generalizes across the gauges, by assertion.** "The same analysis
has been carried out for all rain gauges with similar results, suggesting that
already strong trends (relative to the interannual variability) tend to get
amplified by quantile mapping, for both precipitation totals and heavy
precipitation" (Sect. 3). Those per-gauge results are not shown.

**Two further generalizations are asserted without evidence shown.** Equivalent
analyses "for other regions showed that these problems also occur in flat
terrain" (Sect. 4), and a repeat with the RACMO2 regional model "yields similar
results" (footnote 5).

## Discussion

Quantile mapping used to downscale from grid box to local scale is inflation,
and inflation is not valid: the problems "arise from the attempt to explain local
variability by gridbox variability" (Sect. 4).

For impact modelling, three consequences are named. The temporal structure
remains that of the grid box, so applications that depend on it will be
misspecified. Flood risk from distributed hydrological models may be heavily
overestimated, in particular for small, fast-responding catchments. Future
changes in mean and extreme precipitation, and impacts derived from them, are
likely to be misrepresented because trends are affected.

Averaging neighbouring grid boxes to raise the signal-to-noise ratio makes
matters worse when the target is sub-grid scale, because it widens the scale gap.

These problems are additional to the known non-stationarity of model biases
(Christensen et al. 2008; Maraun 2012).

The author expects the problem to be weaker for temperature: "The effect might
be less important for temperature, as this variable has a much higher spatial
coherence and small-scale variations mostly stem from—correctable—orographic
effects" (Sect. 4). This is stated as an expectation. No temperature data is
analysed in the paper.

Three alternatives are recommended. If day-to-day variability is not needed,
correct only the mean, which avoids the effect on trends. If a catchment total
is needed for a lumped model, correct that total directly instead of downscaling
to points and averaging back. If local day-to-day variability is needed, follow
the perfect prognosis solution of von Storch (1999): a regression between
modelled and observed values whose deterministic part removes the systematic
error, plus a noise model whose realizations supply the missing small-scale
variability. The paper closes on this point: it "clearly demonstrates the need
for stochastic bias correction" (Sect. 4).

Limits the author acknowledges or accepts: there is no validation period, by
choice; one grid box and one variable are analysed; the claims about other
regions, other models, and GCM-driven boundary conditions are stated without the
supporting analysis being shown.

## Relevance to this project

*Everything below is my own reading, not the paper's content.* In this project's
terms (`CONTEXT.md`), the paper's "gridbox" is our coarse cell, its "local scale"
or "station" is our pixel, and its "quantile mapping transformation" is our
transfer function.

- **This is the paper to cite for the limit of our method, and we must cite it.**
  Our baseline fits one transfer function per (land pixel, season), which is
  exactly the deterministic per-location mapping Maraun argues cannot downscale.
  We should state this limit ourselves rather than wait for a reviewer to state
  it.

- **But our setting sits closer to the case he calls feasible than to the case
  he attacks.** He contrasts correcting against "a gridded dataset of the same
  resolution", which he accepts, with correcting against station or
  very-high-resolution gridded data, which he rejects. Our target is ERA5-Land
  at 0.1°, an area-mean gridded product, not a point gauge. Our scale gap is
  about 1° to 0.1°, roughly a factor of ten, not 25 km to a point. We are between
  his two cases, and the write-up should say where and why rather than claim the
  paper does not apply.

- **His own temperature caveat is our strongest defence, and it is only a
  caveat.** He expects the effect to be weaker for temperature because of higher
  spatial coherence and because small-scale variation is mostly orographic and
  correctable. Cite it as his expectation, not as a finding — he analysed no
  temperature. It also tells us what to check: if our residual structure is
  mostly elevation-driven, his argument predicts a deterministic correction can
  handle it.

- **It sets a concrete diagnostic we do not currently run.** He detects the
  problem through the spatial dependence between corrected locations, not through
  a per-pixel error score. Our evaluation is per-pixel MAE, RMSE and percentile
  bias, which is blind to exactly this failure. Worth adding: compare the
  observed and corrected inter-pixel correlation within one coarse cell, and
  check whether the rank order of pixels inside a cell is frozen across days, as
  it must be if the transfer functions do not cross.

- **It predicts a specific artefact in our corrected fields.** Because one coarse
  cell drives every pixel inside it deterministically, our corrected field should
  be too smooth within a cell and should show discontinuities at cell edges. This
  is a testable prediction on output we already have.

- **It is a warning about the projection stage, not the baseline stage.** The
  trend result says quantile mapping amplifies trends that are strong relative to
  interannual variability. Our planned 2081–2100 projection carries a large
  warming trend, so this matters there in a way it does not for 1990–1999. The
  size of the effect for temperature is unknown from this paper.

- **Do not use it to argue against quantile mapping outright.** It is one grid
  box, one variable, precipitation, no validation period, and no competing method
  is tested. Its force is in the argument, not in the sample size.

- **The stochastic alternative is out of our current scope, and naming it is
  cheap.** His recommendation is a deterministic regression plus a noise model.
  Listing that as future work is honest and costs nothing; adopting it would
  change the project.
