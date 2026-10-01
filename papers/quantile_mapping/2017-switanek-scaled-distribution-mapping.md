# Scaled distribution mapping: preserving the raw projected change

## Title

**Scaled distribution mapping: a bias correction method that preserves raw
climate model projected changes**

Matthew B. Switanek, Peter A. Troch, Christopher L. Castro, Armin Leuprecht,
Hsin-I Chang, Rajarshi Mukherjee, Eleonora M. C. Demaria

*Hydrology and Earth System Sciences* **21**, 2649–2666, 2017.
doi:10.5194/hess-21-2649-2017 · CC BY 3.0 · received 25 August 2016, published
6 June 2017.

PDF: `Switanek2017_ScaledDistributionMapping_HESS.pdf`

## Abstract

Quantile mapping (QM) corrects a climate model by learning, in a calibration
period, how much to add to or subtract from the modelled value at each
quantile. Applying that same correction to a different period assumes the
correction values do not change with time. The paper attacks this assumption
head on: "Commonly used bias correction methods such as quantile mapping (QM)
assume the function of error correction values between modeled and observed
distributions are stationary or time invariant. This article finds that this
function of the error correction values cannot be assumed to be stationary. As
a result, QM lacks justification to inflate/deflate various moments of the
climate change signal" (Abstract).

The paper then makes a second argument, about how such methods are judged: a
split-sample or cross-validation test cannot separate the skill of the
correction method from the skill of the raw model. From these two arguments the
authors propose a new method, scaled distribution mapping (SDM). SDM is
parametric, it scales the observed distribution by the modelled change in
magnitude and in event likelihood, and for precipitation it also scales the
rain-day frequency. Judged by how well the raw projected change is preserved,
SDM "is found to outperform QM, QDM, and detrended QM in its ability to better
preserve raw climate model projected changes to meteorological variables such
as temperature and precipitation" (Abstract).

## Terms and notation

### Terms

**Bias correction** here means adjusting model output so that its distribution
matches an observed distribution. The paper separates this from **downscaling**
and studies only bias correction: "we focus specifically on the performance of
bias correction and not issues related to downscaling/upscaling" (Sect. 2).

An **error correction value** is the amount QM adds to (or, for precipitation,
multiplies) a modelled value at a given quantile. The set of these values over
all quantiles is the "function of error correction values".

The **stationarity assumption**, also called the time-invariant assumption, is
that the error correction values found in a calibration period can be applied
to any other period.

The **climate change signal** is the change the raw model projects between two
periods. **Inflation** and **deflation** mean a correction method making that
change larger or smaller than the raw model gave it.

**Model performance error** is defined in this paper as the difference between
the raw modelled change in a statistic and the observed change in the same
statistic, over the same two periods. It involves no bias correction at all. The
paper computes it for the mean, the standard deviation and the skewness.

Four methods are compared. **QM** is standard non-parametric quantile mapping.
**DETQM** is detrended QM, which removes the modelled trend first; the paper
credits it to Hempel et al. (2013) and Bürger et al. (2013). **QDM** is quantile
delta mapping of Cannon et al. (2015), which is non-parametric and multiplies
the observed values by the ratio of modelled values (period of interest divided
by calibration period) at the same quantiles. **SDM** is the paper's own method.

A **recurrence interval** (return period) is the reciprocal of the exceedance
probability of an event, in days here.

The **calibration period** is the historical period from which the correction is
learnt. The **period of interest**, also called the future period, is the period
being corrected; the paper notes it "can be the same period as the raw
historical model period" (Sect. 4.1.2).

### Notation

The three data sets, used as subscripts throughout:

| Symbol | Meaning |
|---|---|
| OBS | Observations in the calibration period |
| MODH | Raw model in the historical (calibration) period |
| MODF | Raw model in the period being corrected |

The quantities of the SDM equations:

| Symbol | Meaning |
|---|---|
| $\mathrm{CDF}$ | Cumulative distribution function value of an event, from the fitted distribution |
| $\mathrm{ICDF}$ | Inverse CDF, or percent point function, of a fitted distribution |
| $\mathrm{RI}$ | Recurrence interval, in days |
| $\mathrm{RI}^{I}$ | A recurrence-interval array after linear interpolation to another length |
| $\mathrm{RI}_{\mathrm{SCALED}}$ | The recurrence interval after scaling by the modelled change in likelihood |
| $\mathrm{CDF}_{\mathrm{SCALED}}$ | The CDF value corresponding to $\mathrm{RI}_{\mathrm{SCALED}}$ |
| $\mathrm{SF}_{A}$ | Array of absolute scaling factors, used for temperature |
| $\mathrm{SF}_{R}$ | Array of relative (multiplicative) scaling factors, used for precipitation |
| $\mathrm{BC}_{\mathrm{INITIAL}}$ | The bias corrected values before the final steps |
| $\sigma_{\mathrm{OBS}}$, $\sigma_{\mathrm{MODH}}$ | Standard deviation of the observed and the raw historical distributions |
| $\mathrm{RD}$, $\mathrm{TD}$ | Number of rain days, and total number of days, in a series |
| $\mathrm{RD}_{\mathrm{BC}}$ | Expected number of rain days after bias correction |
| $k$, $\theta$, $\Gamma(k)$ | Shape and scale parameters of the gamma distribution, and the gamma function |

$\mathrm{MAE}$ is the mean absolute error. The paper uses it in two different
roles, which must not be confused: in Section 3.3 it is the error between
observed and bias corrected quantiles, and from Section 4 onwards it is the
spatial mean absolute error between the raw modelled change in a statistic and
the bias corrected change in the same statistic.

## Previous work

The paper's account of the field starts from agreement. Bias correction is
widely used for impact studies (Berg et al., 2003; Ines and Hansen, 2006;
Muerth et al., 2013; Teng et al., 2015), and several review papers had already
concluded that QM is among the best of the available methods (Gudmundsson et
al., 2012; Teutschbein and Seibert, 2012, 2013; Chen et al., 2013).

Against that, the authors line up work reporting that QM damages the projected
change. The alteration of the raw model's climate change signal is reported by
Hagemann et al. (2011), Themeßl et al. (2012), Brekke et al. (2013), Maurer et
al. (2013), Pierce et al. (2013) and Maurer and Pierce (2014). The paper's own
diagnosis is that this "exists as an artifact of the stationarity assumption"
(Sect. 3).

Later work had tried to fix this. Michelangeli et al. (2009), Olsson et al.
(2009), Willems and Vrac (2011), Wang and Chen (2014), Sunyer et al. (2015) and
Cannon et al. (2015) all adapt QM to preserve the modelled change better. Two
are singled out. Hempel et al. (2013) and Bürger et al. (2013) use detrended QM,
which "better preserved monthly trends, but the daily values still are subject
to the stationarity assumption, which can ultimately result in altering the raw
modeled projected change" (Sect. 3). Cannon et al. (2015) introduced QDM, which
"is a break from other typical QM methods insofar as that it is not constrained
by the stationarity assumption" (Sect. 3).

QDM is therefore the closest relative, and the authors state the three
differences from their own method (Sect. 3): "(1) SDM uses a parametric model
instead of a non-parametric one, (2) SDM and QDM handle days with zero rainfall
very differently, and (3) SDM more accurately accounts for the differences in
the modeled variances, for temperature, between the period of interest and the
calibration period."

On evaluation, the paper takes issue with the standard practice of split-sample
and cross-validation testing (Klemeš, 1986; Piani et al., 2010; Maurer and
Pierce, 2014; Wang and Chen, 2014), and with Maraun et al. (2010) on
pseudo-realities. It accepts Maraun et al. (2015) on the need to check that
modelled and observed distributions agree in the calibration period, but argues
that this is not enough. It also adopts Bárdossy and Pegram (2012) on spatial
re-correlation, and advises applying it before SDM.

## Problem definition

### Problem

Two things are to be improved.

First, a correction method should not change the raw model's projected change
without justification. The paper argues the justification does not exist:
"until some bias correction method provides proper justification to manipulate
and alter the raw model projected climate change signal, a better performing
method should strive to preserve the original projected changes" (Sect. 3.1).

Second, the method should compare events that are equally *probable*, not
events that share the same empirical quantile. Non-parametric methods contain
"an implicit assumption that each respective quantile is equally probable"
(Sect. 3.2), and the paper tests that assumption.

The assumption SDM keeps is that the observed distribution, scaled by the
modelled change, is the right target. It also assumes a distribution family can
be fitted: gamma for precipitation, normal for temperature. The authors treat
the family as the user's choice, and warn that gamma "will likely not be
suitable for all studies involving precipitation, especially with respect to
extremes" (Sect. 4.1.1). They also warn that pre-screening models is still
needed, since bias correction "will not, and should not, be expected to fix
serious model deficiencies (garbage in – garbage out)" (Sect. 4.1).

### Models

**Three general properties of SDM.** It makes no stationarity assumption. It
scales the observed distribution by the modelled changes in magnitude, in
rain-day frequency (precipitation only), and in event likelihood. The scaling
depends on which period is being corrected.

**Temperature** (Sect. 4.1.2). This is the variant relevant here, and it differs
from the precipitation variant in three ways stated in Section 4.1: the scaling
is absolute rather than multiplicative, all values are used rather than only
those above a threshold, and the series is detrended before correction with the
trend added back afterwards, "as a result, the variance is not inflated by
temporal trends".

Step 1: detrend the raw modelled and observed series, "in order to get a more
accurate measure of the natural variability". A linear trend is used here, but
"any trend line could be used". All later steps work on detrended series.

Step 2: fit a normal distribution to the detrended OBS, MODH and MODF series.
For a normal distribution "the fitted parameters are simply the empirical mean
and standard deviation". Find the CDF value of every event under its own fitted
distribution, and clip those values away from 0 and 1 (for example to
$[0.0001, 0.9999]$) so the inverse CDF stays finite.

Step 3: compute the absolute scaling factors between the fitted future and
historical model distributions, at the probabilities of the future events:

$$\mathrm{SF}_{A} = \left[\mathrm{ICDF}_{\mathrm{MODF}}(\mathrm{CDF}_{\mathrm{MODF}}) - \mathrm{ICDF}_{\mathrm{MODH}}(\mathrm{CDF}_{\mathrm{MODF}})\right] \times \left(\frac{\sigma_{\mathrm{OBS}}}{\sigma_{\mathrm{MODH}}}\right) \qquad (8)$$

The bracket is the modelled change in magnitude at that probability. The ratio
$\sigma_{\mathrm{OBS}} / \sigma_{\mathrm{MODH}}$ is the term that QDM does not
have, and the authors attribute SDM's advantage on the higher moments to it.

Step 4: compute recurrence intervals for the three sorted temperature arrays:

$$\mathrm{RI} = \frac{1}{0.5 - \left|\mathrm{CDF} - 0.5\right|} \qquad (9)$$

This differs from the precipitation form, Equation (4), "to reflect the
two-tailed nature of a normal distribution" — a cold extreme and a warm extreme
are both rare.

Step 5: scale the recurrence interval, using the same equation as for
precipitation:

$$\mathrm{RI}_{\mathrm{SCALED}} = \max\left(1,\; \mathrm{RI}^{I}_{\mathrm{OBS}} \times \frac{\mathrm{RI}_{\mathrm{MODF}}}{\mathrm{RI}^{I}_{\mathrm{MODH}}}\right) \qquad (5)$$

The floor of 1 is "necessary (especially for temperature) to ensure that the
values of $\mathrm{CDF}_{\mathrm{SCALED}}$ ... are between 0 and 1". If the
historical and future series have the same length, no interpolation is needed,
so $\mathrm{RI}^{I}$ is just $\mathrm{RI}$. Then convert back to a CDF value:

$$\mathrm{CDF}_{\mathrm{SCALED}} = 0.5 + \operatorname{sgn}(\mathrm{CDF}_{\mathrm{OBS}} - 0.5) \times \left|0.5 - \frac{1}{\mathrm{RI}_{\mathrm{SCALED}}}\right| \qquad (10)$$

The sign term puts the value back on the correct side of the median. Note that
the paper writes the sign function on $\mathrm{CDF}_{\mathrm{OBS}}$, while the
recurrence interval being converted is the scaled one; the text does not explain
this pairing, so it is unclear whether it is intended or a typographical slip.

Step 6: build the bias corrected values, adding rather than multiplying:

$$\mathrm{BC}_{\mathrm{INITIAL}} = \mathrm{ICDF}_{\mathrm{OBS}}(\mathrm{CDF}_{\mathrm{SCALED}}) + \mathrm{SF}_{A} \qquad (11)$$

Step 7: put the corrected values back at the days of the raw future model they
came from, then add the raw future model's trend back in. If the calibration
period and the period of interest are the same, the corrected temperature
distribution equals the observed one, "except the trend of the bias corrected
data will be that of the modeled trend and not of observations".

**Precipitation** (Sect. 4.1.1), in outline, for contrast. A threshold (0.1 mm
here) sets which days count as rain days. The expected corrected rain-day count
is

$$\mathrm{RD}_{\mathrm{BC}} = \mathrm{RD}_{\mathrm{MODF}} \times \frac{\mathrm{RD}_{\mathrm{OBS}} / \mathrm{TD}_{\mathrm{OBS}}}{\mathrm{RD}_{\mathrm{MODH}} / \mathrm{TD}_{\mathrm{MODH}}} \qquad (1)$$

A gamma distribution, Equation (2), with shape $k$ and scale $\theta$, is fitted
to the positive values. The scaling factors are a ratio, not a difference:

$$\mathrm{SF}_{R} = \frac{\mathrm{ICDF}_{\mathrm{MODF}}(\mathrm{CDF}_{\mathrm{MODF}})}{\mathrm{ICDF}_{\mathrm{MODH}}(\mathrm{CDF}_{\mathrm{MODF}})} \qquad (3)$$

Recurrence intervals are one-tailed, $\mathrm{RI} = 1/(1 - \mathrm{CDF})$,
Equation (4), and the conversion back is
$\mathrm{CDF}_{\mathrm{SCALED}} = 1 - 1/\mathrm{RI}_{\mathrm{SCALED}}$,
Equation (6). The corrected values are
$\mathrm{BC}_{\mathrm{INITIAL}} = \mathrm{ICDF}_{\mathrm{OBS}}(\mathrm{CDF}_{\mathrm{SCALED}}) \times \mathrm{SF}_{R}$,
Equation (7), and are then linearly interpolated from the modelled rain-day
count down to $\mathrm{RD}_{\mathrm{BC}}$. The method does not adjust the
frequency when the model has too few rain days.

**Time-sliced application** (Sect. 4.1.3). Applied to one long block, SDM
preserves the change between that block and the calibration period, but not
necessarily the change for shorter periods inside it. The authors therefore run
it in 30-year windows sliding by 10 years, keeping only the middle 10 years of
each. For example, 2011–2020 is corrected using 2001–2030 as the period of
interest against the 1971–2000 calibration period, and 2091–2100 is corrected
using 2081–2100.

### Data

One model chain: the KNMI-RACMO22E regional climate model, forced by the
ICHEC-EC-EARTH GCM, from EURO-CORDEX. Runs are historical for 1951–2005 and
RCP 8.5 for 2006–2100, ensemble member r1i1p1. Daily mean temperature and
precipitation. Observations are the E-OBS data set (Haylock et al., 2008). The
model data "were upscaled from its original 0.11° resolution to the 0.5° E-OBS
resolution" (Sect. 2). The domain is Europe. Correction is applied separately
for each calendar month and each grid cell. The main calibration period is
1971–2000 and the main future period is 2071–2100; the stationarity test uses
1951–1980 and 1976–2005 as two alternative calibration periods.

Synthetic data are used in Section 3.2: values drawn from a gamma distribution
with shape 0.8 and scale 12.0, in samples of 100, 200 and 10 000.

A reference implementation is available in pyCAT
(https://github.com/wegener-center/pyCAT); the SDM Python code is available
from the corresponding author on request.

### Evaluation metrics

The paper's central methodological claim is that the usual metric is wrong, so
it defines its own.

The rejected metric is the error between bias corrected and observed values in a
validation period. Section 3.3 argues that this "cannot distinguish between bias
correction methodological performance and the performance of the underlying raw
model". The conclusion drawn is a rule: "validation (or evaluation) should
measure how well the raw modeled projected changes to the entire distribution
are captured or preserved by the bias correction method between any two
periods" (Sect. 3.3).

The metric used is therefore the spatial MAE between the raw modelled change in
a statistic and the bias corrected change in the same statistic. The statistics
are the mean, the standard deviation, the skewness and the trend. Lower is
better, and zero means the method left the projected change untouched. Moments
are used rather than individual quantiles "because of our findings concerning
the sampling noise associated with extreme values in the distributions"
(Sect. 4). Results are reported by calendar month and by outlook period
(2011–2040, 2021–2050, and so on). Differences between methods are tested with
a $t$ test.

## Main results

**The stationarity assumption fails.** The same future data (2071–2100) were
corrected by QM twice, using calibration periods 1951–1980 and 1976–2005. The
two answers differ: "There are instances where this sensitivity to the
calibration period is nearly as large as the raw model projected mean changes"
(Sect. 3.1). The authors state the consequence plainly: "The calibration period
largely influences the error correction values and, as a result, the
stationarity assumption is invalid." They report that a parametric
implementation of QM was "equally sensitive to the chosen calibration period".
The maps are Figure 2 and the values are not tabulated, so only the statements
above can be quoted. The illustrative example in the same section — a modelled
50 mm corrected to 35 mm under one calibration period and 55 mm under another —
is presented as an example, not as a measured case.

**Equal quantiles are not equally probable.** In the synthetic example of
Section 3.2, two samples of 200 values from the *same* gamma distribution have
largest values of 32.8 and 57.3, an empirical gap of 24.5. Fitting gamma
distributions puts those two values at fitted CDF 0.993 and 0.998. Comparing at
the same expected probability of 0.993 gives a modelled value of 44.8, so the
expected gap falls from 24.5 to 12.0. The authors are careful about what one
case shows: "With this one example case, it is impossible to know if we are
truly gaining information by accounting for differences in event likelihood."
The repetition over 1000 draws at sample sizes 100 and 10 000 is Figure 4; those
counts are not tabulated. From it the authors conclude that the parametric
approach "reduce[s] the error associated with sampling noise over that of the
non-parametric method", and that its advantage grows with sample size.

**Split-sample validation measures the model, not the method.** June temperature
for 1981–2010 was corrected by QM calibrated on 1951–1980. For one example grid
cell the MAE between corrected and observed quantiles is 1.5 °C, and the raw
model performance error of the mean at that cell is 1.4 °C. Across the domain,
"[m]ost of the variability pertaining to methodological performance (in a
split-sample test) can be explained by the model performance error of the mean".
Removing the mean error leaves a relation with the model error of the standard
deviation; removing both leaves "a statistically significant relationship
(p < 0.01) between method performance and the model performance error of the
skewness". The correlation coefficients themselves appear only in the scatter
panels of Figure 5 and are not tabulated.

**Method comparison.** All comparisons use the spatial MAE between the raw and
the bias corrected change. The paper reports these as colour panels (Figs. 7 to
13) and gives no table of numbers, so the values for each method are not
tabulated and cannot be quoted. What the paper states about each method:

| Method | Change to the mean | Change to the standard deviation | Change to the skewness | Significance |
|---|---|---|---|---|
| SDM | Best, or tied best. "SDM has minimal inflation to the raw model projected mean change" (Sect. 4.2). Largest margin over QDM and DETQM for precipitation | Best. "SDM better minimizes MAE for standard deviation and skewness" than QDM | Best, by the same sentence | Average MAE over all months and outlook periods significantly smaller for SDM than for QM, QDM and DETQM, $p < 0.01$, for all three leading moments |
| QDM | Equal to SDM. "Both SDM and QDM perform equally well for preserving changes to the mean" for temperature | Worse than SDM; QDM "does not properly scale the higher moments of the distribution" | Worse than SDM | Same test as above |
| DETQM | Worse than SDM and QDM. "DETQM removes the mean modeled trend, but still performs poorly because the detrended error correction values are still assumed to be stationary" | Worse than SDM | Worse than SDM | Same test as above |
| QM | Worst. Inflation "greater than 1 °C and 10 %, for temperature and precipitation, respectively, across large regions of Europe" (Sect. 4.2) | Clearly worse. "SDM much better preserves the raw model projected changes to the standard deviation" | Worse | Same test as above |

The one numeric statement that survives without a figure is the QM inflation
above: more than 1 °C on the mean temperature change, and more than 10 % on the
relative precipitation change, over large parts of Europe, between 2071–2100 and
1971–2000.

**QM degrades with distance from the calibration period; SDM does not.** "What
is most noticeable is how the QM's alteration (inflation/deflation) of the
climate change signal increases as a function of the projected time period. When
the projected period is furthest from the calibration period (2071–2100), the
alteration to the leading three moments are the greatest. In contrast, ... the
performance of SDM does not degrade as a function of the projected time period"
(Sect. 4.2). This is a temperature result (Fig. 9); Figure 10 reports the same
pattern for precipitation.

**Precipitation-specific findings.** All methods do worse for precipitation in
summer, which the authors attribute to regions such as Spain having too few rain
days to fit well, parametrically or not. SDM's advantage on the precipitation
mean comes from how it handles zero-rain days: QDM fills them with small random
values, and the paper gives a worked four-value example in which that makes a
raw mean change of 1.56 become a corrected mean change of 2.79. Raising the
threshold from 0.1 mm to 1.0 mm improves SDM while leaving the other methods
"approximately the same". In the observations, values between 0.1 and 1.0 mm are
21.6 % of positive precipitation days but only 2.4 % of total precipitation; in
the model, 41.3 % of days and 5.4 % of total precipitation.

## Discussion

The stationarity assumption behind QM is invalid, so QM has no justification for
altering the projected climate change signal.

Split-sample and cross-validation tests conflate method performance with raw
model performance, and so cannot be used to rank bias correction methods. A
method should instead be judged on how well it preserves the raw projected
change across the whole distribution.

The authors recommend SDM, or any method that scales the observed distribution
by the simulated changes across the modelled distribution.

Limits the authors acknowledge:

- Bias correction cannot repair a poor model; pre-screening of GCMs and RCMs is
  still advised.
- Longer calibration records reduce, but do not remove, the instability of QM's
  error correction values: "one can never be sure that these error correction
  values have converged to be completely independent of time" (Sect. 3.1).
- The gamma distribution may be a poor choice for precipitation extremes; the
  user should choose the distribution.
- Applied to one long period, SDM does not preserve the change for shorter
  sub-periods inside it, which is why the sliding-window scheme is needed.
- SDM does not correct the rain-day frequency when the model has too few rain
  days.
- Bias corrected data can still be biased at larger spatial scales; the authors
  advocate spatial re-correlation (Bárdossy and Pegram, 2012) before SDM, but do
  not test it here.
- Four questions are listed as open, including whether correcting variables
  separately breaks their physical consistency, whether corrected values stay
  physically realistic, whether models with large biases can be trusted for
  projections, and how to avoid treating a real model deficiency as bias. On
  these, "more reflection and investigation is required".

Also worth carrying forward: pseudo-realities (Maraun et al., 2010) are named as
a partial alternative for validation, but the authors say the method's
performance "still cannot be separated from how well individual models simulate
raw projected changes relative to the other models" (Sect. 3.3).

## Relevance to this project

*Everything below is my own reading, not the paper's content.* In this section I
use the project's vocabulary from `CONTEXT.md`: the paper's "modeled" data is our
**predictor**, its "observations" are our **target**, and its "bias correction
method" is our **transfer function**.

- **This is the paper to cite for the limit of our current design.** Our transfer
  functions are fitted once per (land pixel, season) on 1990–1999 and then
  applied unchanged. That is exactly the stationarity assumption this paper
  argues is invalid. The claim is demonstrated, not asserted: two calibration
  periods give different corrections for the same future data (Fig. 2). Cite it
  wherever we state the assumption our baseline rests on.
- **It changes what we must do before the 2081–2100 projection.** Applying the
  fitted transfer functions straight to 2081–2100 would inflate or deflate the
  GCM's own projected warming, and the paper shows that the damage grows with
  distance from the calibration period. Section 4.1.3's sliding-window scheme
  (30-year period of interest, 10-year step, keep the middle 10 years) is a
  concrete recipe we can copy if we adopt a change-preserving method.
- **The temperature variant is directly implementable for us.** It needs only a
  normal fit per (pixel, season) — mean and standard deviation — plus detrending,
  Equations (8) to (11). It is much cheaper than our current node-based fitting
  and would slot in beside the existing transforms in `qm_transforms.py` as one
  more member of the same interface. The floor in Equation (5) and the CDF
  clipping in step 2 are the two implementation details most likely to bite.
- **A caution on the normal fit.** Our seasonal temperature distributions are
  not always Gaussian, which is the whole reason we fit nodes rather than assume
  a shape. Before adopting SDM we should check the normal fit per (pixel,
  season); the paper's own advice is that the user chooses the distribution.
- **It is an argument against our validation design, and we should answer it,
  not ignore it.** Our leave-one-season-year-out cross-validation scores corrected
  values against the target. Section 3.3 says such a score is contaminated by how
  well the GCM happens to reproduce the observed change. The force of that
  argument is weaker for us within the 1990–1999 baseline, where we are measuring
  a correction and not a projected change, but it applies fully the moment we
  score anything across periods (2005–2014, 2015–2025). Plan a
  change-preservation metric alongside MAE for those tests.
- **The evaluation metric is worth borrowing as a second score.** The spatial MAE
  between the raw predictor's projected change and the corrected change, for the
  mean, standard deviation and skewness, is easy to add to `qm_metrics.py` and
  would answer a question our current metrics cannot: how much does our transfer
  function move the GCM's signal?
- **QDM is the alternative to compare against, not just SDM.** The paper finds
  QDM equal to SDM on the mean for temperature and worse only on the higher
  moments, and the Cannon et al. (2015) PDF is already in this folder. If we
  implement one change-preserving method, QDM is the cheaper non-parametric
  option; the single term $\sigma_{\mathrm{OBS}}/\sigma_{\mathrm{MODH}}$ in
  Equation (8) is the only reason SDM beats it on variance.
- **Most of the precipitation content does not transfer.** Rain-day frequency,
  the gamma fit, thresholds and Equations (1) to (4), (6) and (7) exist because
  precipitation has a mass of zeros. Skip them; only the two-tailed recurrence
  interval, Equation (9), matters as the temperature counterpart.
- **One scope difference limits the transfer.** The paper corrects an RCM at
  0.5° against gridded E-OBS observations at the same resolution, with no
  downscaling. We map a coarse GCM onto a 0.1° target, so our transfer function
  carries a resolution change as well as a bias. The stationarity argument still
  holds, but the paper's error magnitudes are not comparable with ours.

## Note on this summary

*Main results* was required to carry a table with a metric value for each
method. The paper reports every comparison as colour panels (Figs. 7 to 13) and
tabulates no numbers, so the table gives the authors' own statement about each
method instead of a value, and says so.
