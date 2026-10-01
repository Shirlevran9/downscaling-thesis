# Quantile-based bias adjustment for climate change scenarios

## Title

**Evaluating skills and issues of quantile-based bias adjustment for climate
change scenarios**

Fabian Lehner, Imran Nadeem, Herbert Formayer

*Advances in Statistics, Climatology, Meteorology and Oceanography* **9**,
29–44, 2023.
doi:10.5194/ascmo-9-29-2023 · CC BY 4.0 · received 5 July 2022, accepted
30 March 2023, published 24 April 2023.

PDF: `Lehner2023_QM_BiasAdjustment_ASCMO.pdf`

## Abstract

Climate models produce daily temperature and precipitation with large
systematic errors, and impact studies in hydrology or agriculture cannot use
that output directly. Many bias-adjustment methods exist. The paper reviews
them, sorts them, and then tests four: quantile mapping (QM), scaled
distribution mapping (SDM), quantile delta mapping (QDM) and an empirical
version of PresRAT that the authors build and name PresRATe.

The authors judge the methods against three demands they state themselves:
(1) after adjustment the model should match the observed climatological mean in
the historical period; (2) the long-term trend of the mean in the raw model —
the climate change signal, CCS — should survive the adjustment unchanged; and
(3) a model with too few wet days should still be corrected, so that the wet-day
frequency is right afterwards. The tests use real observations for Austria and
artificial data built from them. The conclusion is that "QDM and PresRATe
combined fulfill all three demands" (Abstract). QM fails demands (2) and (3).
SDM fails demand (1) and meets demand (3) only partly.

## Terms and notation

### Terms

**Bias adjustment** is the general name the paper uses for correcting a climate
model's output against gridded observations. It covers the whole family of
methods here.

The **climate change signal (CCS)** is the change over time in the arithmetic
mean of a variable in the raw model — "the change of the arithmetic mean of a
meteorological variable over time" (Sect. 1). For temperature it is a
difference; for precipitation it is a ratio between the future and the
historical mean.

A method is **trend preserving** if the CCS of the raw model survives the
adjustment, and **trend altering** if it does not. The paper ties this to where
the bias is assumed to be fixed. A trend-preserving method fixes the bias at a
**quantile**: the quantile of a future value is taken from the future
distribution. A trend-altering method fixes the bias at an **absolute value**:
the quantile of a future value is taken from the calibration distribution. That
difference is the whole distinction between QDM and QM in this paper.

**Time invariance**, also called stationarity, is the assumption that the bias
does not change over time. The authors warn that the term is used
inconsistently: Switanek et al. (2017) and Maraun and Widmann (2018) contradict
each other, and the authors put this down to "different definitions" (Sect. 1).
Stationarity can mean a time-independent mean bias, or a time-independent bias
at a given absolute value of the variable.

A method is **parametric** if a statistical function, such as a gamma or normal
distribution, is fitted to the CDF, and **non-parametric** or **empirical** if
the empirical CDF is used directly.

The methods tested are: **QM**, traditional empirical quantile mapping;
**QDM**, quantile delta mapping from Cannon et al. (2015); **PresRATe**, the
authors' empirical version of PresRAT (Pierce et al., 2015), which is QDM plus
an extra step that forces the mean CCS of precipitation back to the raw model's
value; and **SDM**, scaled distribution mapping (Switanek et al., 2017), which
is parametric. **QDMd** is QDM run without the wet-day algorithm, used as a
control.

**Singularity stochastic removal (SSR)** is the wet-day trick, named by Vrac et
al. (2016). Zero-precipitation days are replaced by a trace amount below
0.05 mm before the correction values are computed, and values below the trace
amount are set back to zero afterwards. It lets dry days take part in a
multiplicative adjustment without dividing by zero, and it allows wet days to be
added.

A **wet day** here is a day with precipitation of at least 0.1 mm.

Four versions of SDM are used. **SDM(raw)** is the method as published, with no
wet-day correction. **SDM(0)** adds an interpolation of corrected wet days up
to the expected number, an algorithm supplied by the authors of Switanek et al.
(2017). **SDM(1)** adds the shape parameter as a starting guess for the gamma
fit. **SDM(2)** adds both the shape and the scale parameter as starting
guesses.

### Notation

The variables of the correction itself:

| Symbol | Meaning |
|---|---|
| $x_{m\text{-}f}$ | The raw model time series to be corrected |
| $x_{m\text{-}fr}$ | The same series, sorted by rank |
| $x_{\mathrm{corr}}$ | The corrected series |
| $F$ | An empirical CDF; subscripts $o$ or $m$ for observations or model, $c$ or $f$ for the calibration or the future period |
| $F^{-1}$ | The inverse CDF, or quantile function |
| $F_{100}$ | The 100 equidistant percentiles from 0.5 % to 99.5 % |
| $\mathrm{CV}$ | Correction value: the model bias at one percentile |
| $\mathrm{CV}_i$ | The correction values interpolated to the length of $x_{m\text{-}fr}$ |

The quantities of the precipitation CCS correction, Equations (7) to (10):

| Symbol | Meaning |
|---|---|
| $R_{m\text{-}c}$, $R_{m\text{-}f}$ | Mean precipitation of the raw model in the calibration and the future period |
| $R_{\mathrm{corr}\text{-}c}$, $R_{\mathrm{corr}\text{-}f}$ | The same, after adjustment |
| $\mathrm{CCS}_m$ | The raw model's CCS, as a ratio |
| $\mathrm{CCS}_{\mathrm{corr}}$ | The adjusted data's CCS, as a ratio |
| $E$ | Error of $\mathrm{CCS}_{\mathrm{corr}}$ against $\mathrm{CCS}_m$, in per cent |

The gamma-fit starting guesses, Equations (13) and (14): $\theta$ is the scale
parameter, $k$ the shape parameter, $X$ the data and $\bar{X}$ its mean.

**ME** is mean error and **MAE** is mean absolute error, both used on maps of
annual precipitation in millimetres.

## Previous work

The paper is partly a review, so its account of the literature is long. It
covers four threads.

**Simple methods came first, and are still used.** Methods that correct only
the mean or the variance (Maraun, 2016; Lafon et al., 2013; Widmann et al.,
2003) remain in use "due to their simplicity" (Sect. 1). The objection is that
models "may have different biases for extremes than for average values" (Di Luca
et al., 2020a, b), so distribution-wide methods were introduced. The authors
report that distribution-based methods "usually outperform other simpler methods
like mean bias adjustment, as shown by Lafon et al. (2013) or Themeßl et al.
(2011)" (Sect. 1).

**The same method has many names.** The paper lists variable correction method
(Déqué, 2007), distribution-based scaling (Yang et al., 2010), distribution
mapping (Teutschbein and Seibert, 2012), statistical bias-correction (Piani et
al., 2010), statistical transformation (Gudmundsson et al., 2012),
quantile–quantile mapping and quantile mapping.

**Parametric against non-parametric is unsettled.** The paper says the question
"is still in scientific discussion (Teng et al., 2015), but the non-parametric
approach is more common" (Sect. 1). On the non-parametric side: Lafon et al.
(2013) compared the two and "found that the empirical approach was the most
accurate"; Cannon et al. (2015) and Gudmundsson et al. (2012) also prefer
empirical QM; Themeßl et al. (2012) warn that parametric QM "can introduce new
biases, because the distribution of a meteorological variable is not fully
known and also depends on the region and season". On the parametric side:
Switanek et al. (2017) argue the correction of extremes is more robust with a
fitted function, "as the return level of the most extreme event is somewhat
random". The paper also reports, without giving a source for it in that
sentence, that "non-parametric QM depends more on the calibration period than
parametric QM" (Sect. 1).

**Trend preservation is the main flaw the authors pick out in QM.** Traditional
QM may alter the raw CCS (Hagemann et al., 2011; Maurer and Pierce, 2014;
Maraun, 2013, 2016). The paper is careful that this is not always bad: the
model's own CCS may itself be biased (Boberg and Christensen, 2012; Gobiet et
al., 2015), and trend-changing adjustment "may even improve implausible trends"
(Maraun et al., 2017). If the model has large errors in circulation patterns,
CCS-preserving adjustment "may amplify the bias" (Maraun et al., 2021). The
fixes proposed before this paper: detrend before QM and add the trend back
afterwards, DQM (Bürger et al., 2013; Hempel et al., 2013); QDM (Cannon et al.,
2015); SDM, a parametric variant of QDM (Switanek et al., 2017); EDCDFm (Li et
al., 2010), improved by Pierce et al. (2015). Cannon et al. (2015) prove in
their appendix that EDCDFm and QDM are equivalent, however different in
concept. The multiplicative form, PresRAT (Pierce et al., 2015), preserves the
CCS at every quantile but not in the mean.

**Adding wet days is barely covered in the literature.** Most methods correct a
wet-day bias only when the model has too many wet days. Themeßl et al. (2012)
is named as one of the few to address the opposite case, with a linear
interpolation that fills the gap in the precipitation CDF; the authors object
that this "does not necessarily conserve precipitation sums, because the CDF of
precipitation does not follow a linear curve" (Sect. 1). The alternative is to
modify the dry days first (Cannon et al., 2015; Cannon, 2018; Mehrotra et al.,
2018; Vrac et al., 2016), which Vrac et al. named SSR.

The authors state that "there is no single best bias-adjustment method that fits
all needs" (Sect. 1) and point to Maraun and Widmann (2018) and Doblas-Reyes et
al. (2021) for the full review.

## Problem definition

### Problem

Find a quantile-based bias-adjustment method suitable for climate impact studies
that are sensitive to changes in means and to threshold effects. The authors
restrict the field to quantile-based methods from the start, "because they
usually outperform simpler methods" (Sect. 1), and then require three things of
a method:

1. The adjusted data match the observations in the historical period in
   arithmetic mean.
2. The CCS is not altered. The mean change from the historical to the future
   period in the raw model is preserved, as a ratio where the adjustment is
   multiplicative.
3. Models with too few wet days are corrected reasonably, which means wet days
   must be added.

Demand (2) rests on a choice, and the paper says so: "Preserving the CCS is a
choice of the researcher and not an a priori given" (Sect. 3.1).

The authors also give their reason for preferring a bias fixed at the quantile.
They "postulate that the bias of a climate model is correlated to the modeled
weather pattern" (Sect. 3.1) — a model can predict the rank of a day but not its
value (Déqué, 2007), so a quantile stands in for a weather situation. If the
frequency of weather patterns changes little, a bias attached to a quantile
carries forward and a bias attached to a temperature does not. This is argued,
not tested. It is supported by a citation to Maraun and Widmann (2018), who
state that biases depend "not only on the actual values but more generally on
the state of the climate system".

### Models

All methods are applied per grid cell and per month: "The daily data of each
grid cell of the model are adjusted separately with the observations on a
monthly basis" (Sect. 3). This monthly stratification is used throughout and is
never tested against an alternative.

**QDM and PresRATe** (Sect. 3.2). The starting point is the correction term of
Equation (2) in Li et al. (2010):

$$x_{\mathrm{corr}} = x_{m\text{-}f} + \underbrace{F^{-1}_{o\text{-}c}\left(F_{m\text{-}f}(x_{m\text{-}f})\right) - F^{-1}_{m\text{-}c}\left(F_{m\text{-}f}(x_{m\text{-}f})\right)}_{\text{correction term}} \qquad (1)$$

The bracketed term $F_{m\text{-}f}(x_{m\text{-}f})$ is then replaced by a fixed
grid of 100 percentiles:

$$F_{100} = \left(0.995,\ 0.985,\ \dots,\ 0.015,\ 0.005\right)^{\top} \qquad (2)$$

The correction values are the difference of the two inverse CDFs at those
percentiles, for temperature and dew point:

$$\mathrm{CV} = F^{-1}_{o\text{-}c}(F_{100}) - F^{-1}_{m\text{-}c}(F_{100}) \qquad (3)$$

and their ratio for variables bounded below by zero, such as precipitation, wind
speed and global radiation:

$$\mathrm{CV} = \frac{F^{-1}_{o\text{-}c}(F_{100})}{F^{-1}_{m\text{-}c}(F_{100})} \qquad (4)$$

If a denominator in Equation (4) is exactly zero, that CV is set to 0 by hand.
The choice of 100 nodes is justified, not tested: "The number of 100 points
seems to be a reasonable compromise. A higher number would be less robust to
extremes, as especially the CVs of extremes would depend even more on single
extreme events. A lower number would provide less detail about the
distributional shape of the model bias" (Sect. 3.2).

The correction is then applied to the ranked model data:

$$x_{\mathrm{corr}} = x_{m\text{-}fr} + \mathrm{CV}_i \qquad (5)$$
$$x_{\mathrm{corr}} = x_{m\text{-}fr} \cdot \mathrm{CV}_i \qquad (6)$$

Between nodes the correction values are linearly interpolated. Outside the
range they are held constant: every value below the 0.5 % percentile takes the
CV of the 0.5 % percentile, and every value above the 99.5 % percentile takes
the CV of the 99.5 % percentile. The corrected values come out ranked and must
be put back into their original time order.

(A small inconsistency in the text: Sect. 3.2 refers to "Eq. (4) for temperature
and dew point" and "Eq. (5) for precipitation" when describing this step, but
the equations that add and multiply the correction values are numbered (5) and
(6). The maths is unambiguous; only the cross-reference is wrong.)

**Conserving the mean CCS for precipitation** (Sect. 3.3). Multiplicative
adjustment preserves the CCS at every quantile but not for the mean. The raw
model's CCS and the adjusted data's CCS are

$$\mathrm{CCS}_m = \frac{R_{m\text{-}f}}{R_{m\text{-}c}} \qquad (7)$$
$$\mathrm{CCS}_{\mathrm{corr}} = \frac{R_{\mathrm{corr}\text{-}f}}{R_{\mathrm{corr}\text{-}c}} \qquad (8)$$

and the error of one against the other, in per cent, is

$$E = \frac{\mathrm{CCS}_{\mathrm{corr}}}{\mathrm{CCS}_m} \cdot 100 - 100 \qquad (9)$$

where 0 is a perfect adjustment. Each adjusted daily value is then rescaled:

$$R_{\mathrm{corr\ CCS},t} = R_{\mathrm{corr}\text{-}f,t} \cdot \frac{\mathrm{CCS}_m}{\mathrm{CCS}_{\mathrm{corr}}} \qquad (10)$$

Equation (10) is the only difference between QDM and PresRATe. It can be
applied monthly, seasonally or annually, but not to all at once: "every CCS
cannot be exactly conserved at the same time, because the second CCS (e.g., the
annual one) alters the data from the first CCS correction (e.g., monthly)"
(Sect. 3.3).

**QM** (Sect. 3.5), in its original empirical form:

$$x_{\mathrm{corr}} = F^{-1}_{o\text{-}c}\left(F_{m\text{-}c}(x_{m\text{-}f})\right) \qquad (11)$$

This cannot produce values outside the observed range, so the version actually
used adds constant extrapolation following Boé et al. (2007):

$$x_{\mathrm{corr}} = F^{-1}_{o\text{-}c}\left(F_{m\text{-}c}(x_{m\text{-}f})\right) + \underbrace{x_{m\text{-}f} - F^{-1}_{m\text{-}c}\left(F_{m\text{-}c}(x_{m\text{-}f})\right)}_{\text{extrapolation term}} \qquad (12)$$

The extrapolation term is zero inside the historical model range. This is the
implementation in the Python module pyCAT. Comparing Equation (12) with
Equation (1) shows the structural difference: QM takes the model CDF from the
historical period, $F_{m\text{-}c}$, while QDM and PresRATe take it from the
period being corrected, $F_{m\text{-}f}$.

**SDM** (Sect. 3.6), also from pyCAT, is parametric; for precipitation a gamma
distribution can be chosen, fitted iteratively by maximum likelihood. The
authors report two practical problems. It is slow: they "observed the SDM
script to be more than one order of magnitude slower than the other empiric
bias-adjustment methods". And the fit is unreliable: "Tests showed that the
fitting is sometimes defective and results in errors when the corrected model
data are compared with the observations". Their fix is to give the fit starting
guesses from the method of moments (Thom, 1958; Wiens et al., 2003):

$$\theta = \frac{\mathrm{Var}(X)}{\bar{X}} \qquad (13)$$
$$k = \frac{\bar{X}^2}{\mathrm{Var}(X)} \qquad (14)$$

**Wet days** (Sect. 3.4). All tested methods remove surplus wet days by
multiplying the lower part of the model CDF by 0, but no quantile-based method
can add wet days that the model does not have. SSR is used inside QDM and
PresRATe for this. QDM without SSR is called QDMd.

### Data

The area is Austria, described as "representative of a mountainous area in the
middle latitudes", with elevation from 114 m to 3798 m.

- **OBS**: the SPARTACUS gridded observational data set, daily, 1 km, 1961–2019,
  for minimum temperature, maximum temperature and precipitation, used
  unchanged.
- **Artificial model**: SPARTACUS smoothed with a 12 km running mean, described
  as "a typical spatial resolution of RCMs". The only difference from OBS is the
  spatial resolution.
- **Artificial dry model**: the smoothed data with each day's precipitation
  multiplied by a uniform random number between 0 and 1, plus a drying trend
  made by "successively canceling more and more wet days going from 1961 to
  2019".
- **Artificial temperature data** for the CCS test: normally distributed, with a
  historical period of 1981–2010 and a future period of 2071–2100. Observations
  have mean 10 °C and standard deviation 3.1 °C; the raw historical model has
  mean 8 °C and standard deviation 1.8 °C, so it is too cold and too narrow; the
  raw future model has mean 12.4 °C and the same standard deviation of 1.8 °C.
- **Real adjusted data**, used only to show that the problem exists: 35 RCM data
  sets from the Austrian projects ÖKS15 and STARC-Impact, all adjusted with SDM
  at 1 km, compared against the GPARD1 observations over the reference period
  1971–2000.

The calibration period is discussed in one sentence and never varied: "For the
calibration data, a time period of 30 years is typical, since the statistical
distribution of data of a shorter time period can be very noisy and a longer
time period usually has pronounced climatological trends" (Sect. 3).

### Evaluation metrics

There is no single skill score. Each demand has its own diagnostic.

- Demand (1): maps of mean annual precipitation, adjusted model minus OBS, in
  millimetres, summarised by ME and MAE.
- Demand (2), temperature: the linear trend in mean annual temperature, in
  °C per decade over 1981–2100, compared with the raw model's trend.
- Demand (2), precipitation: the CCS error $E$ of Equation (9), in per cent,
  summarised by its mean absolute value over the domain.
- Demand (3): the area-mean annual precipitation bias in millimetres, and the
  area-mean bias in the number of wet days per year.

Everything is scored inside the period used for fitting. The paper reports no
cross-validation and no independent test period; the tests are about whether a
method is internally consistent, not about how well it generalises to unseen
years.

## Main results

### Demand (1): matching the observed historical mean

Mean annual precipitation of the adjusted model minus OBS, over Austria
(Sect. 4.1, Fig. 4).

| Method | Kind | MAE of annual precipitation |
|---|---|---|
| SDM(raw) | parametric | 51.3 mm |
| SDM(0) | parametric | "considerably smaller" than SDM(raw); value not in the text |
| SDM(1) | parametric | "considerably smaller" than SDM(raw); value not in the text |
| SDM(2) | parametric | 3.7 mm yr⁻¹ |
| QM | empirical | "close to zero"; value not in the text |
| QDM / PresRATe | empirical | "close to zero"; value not in the text |

The MAE values for SDM(0), SDM(1), QM and QDM/PresRATe are printed only as
labels on the panels of Figure 4 and are not tabulated, so they cannot be quoted
from the paper. SDM(raw) exceeds 100 mm difference in parts of East Tyrol and
Carinthia; with SDM(2) the error is 10 mm or less over most of Austria.

The reason given for the empirical methods succeeding here: they "calculate
empirical CDFs of both model and OBS which produces very accurate results in the
reference period" (Sect. 4.1).

The motivating evidence from the real project data (Fig. 2) is that the domain
mean annual precipitation bias of the 35 already-adjusted models runs from about
−6 % for the driest model to +2 % for the wettest. Per grid cell the bias
exceeds 5 % for "the wettest 0.1 percentile" and reaches about −25 % for the
driest cells. The median bias across all models is +0.5 %, "which we consider
as quite good" (Sect. 2). The largest errors were found in very dry models with
a clear negative wet-day bias, which is why the rest of the paper concentrates
on those.

### Demand (2): preserving the climate change signal

**Temperature**, linear trend over 1981–2100 in the artificial data (Sect. 4.2,
Fig. 5):

| Series | Trend |
|---|---|
| Raw model | 0.41 °C per decade |
| QDM | 0.41 °C per decade (the same trend, exactly conserved) |
| SDM | 0.41 °C per decade (the same trend, exactly conserved) |
| QM | 0.72 °C per decade |

QM inflates the warming trend by 0.31 °C per decade, which is about 76 % of the
raw signal. The authors also tested non-linear trends, where "QM tends to
inflate or deflate the CCS (not shown), while SDM and QDM keep the CCS
unchanged" — that result is asserted and no figure supports it.

The mechanism is explained rather than measured. In this artificial case the
raw historical model is too narrow, so the largest biases sit at the top of the
CDF. As the climate warms, high temperatures occur more often, so QM reaches for
correction values from the upper part of the historical CDF more often, and the
adjusted model comes out too warm (Sect. 3.2, Fig. 3).

**Precipitation**, mean absolute CCS error $E$ of Equation (9) (Sect. 4.2,
Fig. 6):

| Method | Mean absolute CCS error | Sign |
|---|---|---|
| SDM | 4.3 % | underestimates the CCS |
| QM | 3.3 % | overestimates the CCS |
| QDM, without the CCS correction | 1.9 % | overestimates the CCS |
| PresRATe | "almost 0 %" | forced to match by Equation (10) |

### Demand (3): wet days in a dry model

Artificial dry model, historical period (Sect. 4.3, Figs. 7 and 8):

| Method | Area-mean annual precipitation bias | Area-mean wet-day bias |
|---|---|---|
| Raw model | large dry bias (not given as a single number) | −69.9 d yr⁻¹ |
| SDM(2) | +6 mm (slight wet bias; over 40 mm in some cells) | +15.5 d yr⁻¹ |
| QM | −48.5 mm | −69.9 d yr⁻¹ (unchanged from raw) |
| QDMd | "a similar pattern" to QM; value not in the text | −69.9 d yr⁻¹ (unchanged from raw) |
| QDM / PresRATe | +8.6 mm | −5 d yr⁻¹ |

QDM/PresRATe performs best on wet days, with "only very few grid cells exceed[ing]
a wet day bias of +10 or −10 d". QDMd is reported to be "almost identical to QM
in the historical period, with the only difference that we used 100 discrete
percentiles for QDMd and all values for QM for the CDFs" (Sect. 4.3) — which
shows that the 100-node discretisation on its own has little effect here. The
positive wet-day bias of SDM(2) is unexplained: "The reason for this positive
wet bias is still in discussion. We suspect that it might be caused by the
fitting of gamma functions to the CDFs which introduces new errors"
(Sect. 5).

### Summary across the three demands

Table 2 of the paper, reproduced:

| Method | Kind | Trend | (1) historical mean | (2) CCS | (3) wet days |
|---|---|---|---|---|---|
| SDM | parametric | preserving | no | yes additively; only at quantiles multiplicatively | to some extent, yes |
| QDM / PresRATe | empirical | preserving | yes | yes additively; yes for PresRATe multiplicatively | yes |
| Empirical QM | empirical | altering | yes | no | no |
| (parametric, trend altering) | — | — | not tested | not tested | not tested |

### On the calibration period

The paper does **not** test how the length of the calibration period affects any
method. It makes two statements about it, both without supporting evidence in
this paper.

- The assumption behind their own setup: "For the calibration data, a time
  period of 30 years is typical, since the statistical distribution of data of a
  shorter time period can be very noisy and a longer time period usually has
  pronounced climatological trends" (Sect. 3). This is a justification for the
  30-year choice, not a result. No shorter or longer calibration period is run.
- A claim carried from the literature review: "non-parametric QM depends more on
  the calibration period than parametric QM" (Sect. 1). No reference is attached
  to that sentence, and the paper does not test it.

All results above use the full available period, and nothing in the paper
quantifies the cost of a shorter one.

## Discussion

QDM and PresRATe together meet all three demands. QM fails demands (2) and
(3); SDM fails demand (1) and meets demand (3) only in part.

Fitting functions is the cause of SDM's failure on demand (1): "The fitting of
functions (SDM) will always produce errors which can be minimized with a good
fitting algorithm" (Sect. 5). Parametric approaches also "require knowledge
about the statistical distribution of a meteorological variable in order to
choose a suitable distribution function".

The authors generalise from SDM to parametric methods as a class, and say that
they are doing so: "We assume that SDM can be seen as a representative of
parametric methods in general because the errors introduced with SDM are mainly
due to the fitting of functions" (Sect. 5). This is an assumption, not a
demonstration — only one parametric method was tested.

For multiplicative adjustment, the CCS at quantiles and the CCS of the mean
cannot both be conserved: "in general, the relative CCSs of monthly and annual
means differ from the ratios at quantiles. Depending on the application, a
decision has to be made" (Sect. 5).

The SSR wet-day algorithm is not tied to QDM: "As a supplementary method, this
algorithm can be applied after any bias-adjustment method and could therefore
also be applied with QM" (Sect. 5).

Accuracy in the historical period matters even when only the CCS is of interest,
because impact models are calibrated on the adjusted historical data: "If an
impact model is calibrated with inaccurate meteorological data in the historical
period, the impact of climate change can lead to wrong conclusions even if the
CCS is accurate" (Sect. 5).

Limits the authors acknowledge:

- Bias adjustment "cannot fully remove all model errors" (Sect. 5).
- QDM is univariate — each grid cell is corrected on its own. Long-term mean
  spatial patterns match the observations, but for shorter timescales the
  literature disagrees about whether univariate methods help, and the paper does
  not settle it.
- Temporal and spatial correlation need other methods (nesting approaches,
  multivariate methods, or the two-step approach of Volosciuk et al., 2017), and
  these "suffer from disadvantages such as very high computational demands or a
  limited measure of the full multivariate dependence of structure".
- For the two-step approach, "the added skill is different from case to case and
  may even increase the bias at times".
- Whether to preserve the CCS at all remains the researcher's choice, and
  depends on trusting the model's own trend.

## Relevance to this project

*Everything below is my own reading, not the paper's content.* In the paper's
terms, its "correction values" are our **transfer function** evaluated at 100
**nodes**; its "bias adjustment" is our correction step; its grid cell at 1 km
is our **pixel**.

- **The calibration-length question is not answered here — cite it as an
  assumption, not evidence.** The paper asserts that 30 years is typical and
  that shorter periods are noisy, and it never tests this. Our current 10-year
  baseline and our planned 25-year training period (1980–2004) cannot be
  defended or attacked with this paper. If we need evidence on calibration
  length, we need a different source or our own experiment. What we can cite is
  the assumption itself, as a statement of common practice that puts our planned
  25 years inside the normal range and our present 10 years below it.
- **A 25-year training window brings the paper's second warning into play.** The
  same sentence says a longer period "usually has pronounced climatological
  trends". Our planned 1980–2004 window contains a warming trend that our
  1990–1999 baseline barely shows. A single pooled distribution over 25 years
  will therefore be wider than any single decade's, which inflates the spread of
  the fitted transfer function. This is a concrete reason to check, when we move
  to 1980–2004, whether the fitted nodes drift relative to the 1990–1999 ones.
- **Our transfer functions are QM in this paper's sense, and QM alters the
  trend.** Equation (12) is what we implement. The paper's Figure 5 result —
  0.41 °C per decade in the raw model becoming 0.72 °C per decade after QM — is
  the number to quote when we get to the 2081–2100 projection. Our baseline as
  it stands will not carry the GCM's warming signal forward unchanged. That is
  acceptable while we are only evaluating over 1990–1999, but it is a blocking
  issue for the projection stage of the planned scope.
- **QDM is the upgrade path, and it is a small change to our code.** For
  temperature the whole method is Equations (1) to (3) and (5): the same
  node-based correction we already fit, applied at the quantile the value holds
  in the *future* distribution instead of the calibration one. Nothing about our
  per-(pixel, season) structure has to change. Worth an ADR when we add the
  projection period.
- **The 100-node grid is an independent precedent for our node count.** They use
  100 equidistant percentiles from 0.5 % to 99.5 %, with linear interpolation
  between and constant extrapolation outside, and they justify it as a trade-off
  between tail robustness and distributional detail. Our QUANT grid starts at 11
  nodes and is anchored. Their reasoning supports the direction we chose — more
  nodes, not fewer — but their number is asserted rather than tested, so it is
  not evidence against our anchoring.
- **Their constant extrapolation rule is the same as ours, and they state it
  plainly.** Every value beyond the outermost node takes that node's correction.
  Useful as a citable statement of the convention when we document how our
  transfer functions behave outside the fitted range.
- **The monthly stratification is another assertion we can list, not lean on.**
  They correct per grid cell on a monthly basis and never compare it with an
  annual or seasonal grouping. Add this to the set of papers that assume
  sub-annual stratification helps without testing it — the same category as the
  ones Reiter et al. (2018) actually tests.
- **Everything here is scored in-sample, so it is not comparable to our
  cross-validated numbers.** No test period, no cross-validation. Their
  "close to zero" historical error for empirical QM is what any empirical method
  gives on its own fitting data. Do not put their figures beside our
  leave-one-season-year-out results as though they measured the same thing.
- **Nothing in this paper transfers from precipitation to our variable.**
  Demand (3), SSR, PresRATe and the gamma fitting all exist because
  precipitation has a mass of zeros. For daily mean temperature only demands (1)
  and (2) apply, and the method reduces to additive QDM.

## Note on this summary

Two departures from the template. *Main results* is split into `###`
subsections, one per demand, because the paper has no single skill score and its
three demands are measured in different units — a flat list of numbers would not
say which table answers which question. And the last of those subsections,
"On the calibration period", records a question the paper does **not** answer;
it is there because calibration length is a live question for this project and a
reader would otherwise assume the paper had tested it.
