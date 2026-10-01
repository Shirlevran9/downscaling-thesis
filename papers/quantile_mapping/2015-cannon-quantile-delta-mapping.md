# Quantile delta mapping and the preservation of projected trends

## Title

**Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes?**

Alex J. Cannon, Stephen R. Sobie, Trevor Q. Murdock

*Journal of Climate* **28**, 6938–6959, 1 September 2015.
doi:10.1175/JCLI-D-14-00754.1 · received 30 October 2014, final form 18 June 2015.

PDF: `Cannon2015_QDM_JClimate.pdf`

## Abstract

Quantile mapping removes systematic distributional bias from climate model
precipitation. It works well on the historical period, but earlier work had
shown that it can also change the model's projected trend. Those earlier
studies looked mostly at the mean. This paper asks what quantile mapping does
to trends in the extremes, which is where impact studies and engineering design
take their numbers from.

The paper presents quantile delta mapping (QDM), a bias correction that
corrects historical distributional bias and, at the same time, keeps the
model's projected relative change in every quantile. QDM is compared with
detrended quantile mapping (DQM), which keeps the projected change in the mean,
and with standard quantile mapping (QM). The comparison uses synthetic gamma
data and then daily CMIP5 precipitation from three GCMs over Canada. The main
finding is that QM inflates relative trends in precipitation extremes, often by
a lot, while DQM and QDM do not. For 20-yr return values by the 2080s,
"relative changes in excess of +500% with respect to historical conditions are
noted at some locations", against "maximum changes by DQM and QDM nearing
+240% and +140%, respectively, whereas raw GCM changes are never projected to
exceed +120%" (Abstract).

## Terms and notation

### Terms

**Bias correction** here means correcting a model field against observations at
a comparable spatial resolution. The authors separate this from **downscaling**,
which goes from a coarse model scale to a finer observed scale, and they state
that the paper treats only the first case: "In this study, we are only concerned
with quantile mapping as a bias correction algorithm, that is, when the observed
and modeled data have comparable spatial resolutions or have been appropriately
regridded to the same resolution" (Sect. 1).

**Quantile mapping (QM)** is the standard method. It builds one transfer
function from the historical period alone and applies it to the future. Future
model information is not used.

**Detrended quantile mapping (DQM)** first removes the modelled trend in the
long-term mean, applies quantile mapping to the detrended series, then puts the
mean trend back. It therefore uses one piece of future model information: the
projected mean.

**Quantile delta mapping (QDM)** is the method the paper introduces. It removes
the modelled trend quantile by quantile, applies quantile mapping to the
detrended series, then puts the per-quantile changes back on top. It therefore
uses the projected change in every quantile.

A **ratio variable** is a quantitative variable with an absolute zero, such as
precipitation; the paper preserves *relative* changes for these. An **interval
variable**, such as temperature in °C, has no absolute zero; for these the
authors state that absolute changes can be preserved by applying the same
equations additively rather than multiplicatively (Sect. 1 and appendix A).

**Trend preservation** is the goal that the bias-corrected series should keep
the climate model's own projected change, "so that the climate sensitivity of
the model is not affected by bias correction" (Sect. 1, attributing the argument
to Hempel et al. 2013).

The **ETCCDI indices** are 11 standard annual precipitation extremes indices
from the WMO Expert Team on Climate Change Detection and Indices, used here to
score the methods. The **GEV** distribution is the generalized extreme value
distribution, fitted to annual maxima to estimate return magnitudes. A **20-yr
return magnitude** is the daily precipitation amount expected to be exceeded on
average once in 20 years.

### Notation

The variables of the transfer functions, Equations (1) to (6):

| Symbol | Meaning |
|---|---|
| $x_{o,h}$ | Observed value in the historical period |
| $x_{m,h}$ | Modelled value in the historical period |
| $x_{m,p}(t)$ | Modelled value at time $t$ in the projected period |
| $\hat{x}_{m,p}(t)$ | The bias-corrected value at time $t$ |
| $F_{o,h}$, $F_{m,h}$ | CDFs of the observed and modelled historical data |
| $F_{o,h}^{-1}$, $F_{m,h}^{-1}$ | The matching inverse CDFs (quantile functions) |
| $F^{(t)}_{m,p}$ | The time-dependent CDF of the projected modelled series at $t$ |
| $\tau_{m,p}(t)$ | Non-exceedance probability of $x_{m,p}(t)$ under $F^{(t)}_{m,p}$ |
| $\Delta_m(t)$ | The modelled change in that quantile, relative in Eq. (4), absolute in Eq. (A2) |
| $\hat{x}_{o:m,h:p}(t)$ | The detrended, bias-corrected intermediate value, Eq. (5) |
| $\bar{x}_{m,h}$, $\bar{x}_{m,p}(t)$ | Long-term modelled means, historical and at time $t$ |

The synthetic example, Equation (7), uses the gamma distribution with shape $k$
and scale $\theta$, so that the mean is $\mu = k\theta$ and the standard
deviation is $\sigma = \sqrt{k}\,\theta$.

The GEV distribution, Equations (B1) to (B3), is $\mathrm{GEV}(\xi, \alpha,
\kappa)$ with location $\xi$, scale $\alpha > 0$ and shape $\kappa$.

Score names: $D$ is the Kolmogorov–Smirnov test statistic, "the maximum
difference between cumulative distribution functions" (Sect. 5b). RMSE and bias
are taken with respect to the raw GCM's projected relative change, not with
respect to observations. An **anomaly correlation** is the spatial correlation
between two fields of relative change.

## Previous work

The paper builds on a settled starting point: GCM and RCM precipitation carries
large systematic bias (Mearns et al. 2012; Sillmann et al. 2013), quantile
mapping beats simpler corrections of the mean or of the mean and variance
(Teutschbein and Seibert 2012; Gudmundsson et al. 2012; Chen et al. 2013), and
within quantile mapping the nonparametric estimators outperform the parametric
ones (Gudmundsson et al. 2012).

Eden et al. (2012) is used to bound what bias correction can do. They sort GCM
precipitation error into three sources: unrealistic large-scale variability or
response to forcing, internal variability that differs from observations, and
errors in convective parameterizations and unresolved subgrid orography. The
authors state that a univariate, grid-cell bias correction "can, in principle,
only correct the third source of error" (Sect. 1).

The gap the paper attacks is trend corruption. Several studies had shown that
quantile mapping changes long-term trends (Hagemann et al. 2011; Themeßl et al.
2012; Maraun 2013; Maurer and Pierce 2014). Two are given in detail. Maurer and
Pierce (2014) found that modifications of projected seasonal mean precipitation
trends were in some places as large as the original GCM change, and noted, in a
synthetic example only, that trends in extreme quantiles can be affected
differently from trends in the mean. Maraun (2013) used a nonstationary GEV
analysis and found that an RCM which underestimated observed variability had its
trends in extremes strongly amplified by quantile mapping.

The authors identify what existing trend-preserving methods do not cover.
Hempel et al. (2013) preserve relative trends in *monthly mean* precipitation
but quantile-map the daily anomalies, so "quantile mapping of the anomalies
could still modify trends in the daily extremes" (Sect. 1). Bürger et al. (2013)
detrend the long-term mean, which "will tend to maintain the modeled long-term
trend in the mean, but does not guarantee that trends in precipitation extremes
... are preserved". The quantile delta change and quantile perturbation methods
(Olsson et al. 2009; Willems and Vrac 2011; Sunyer et al. 2014) do preserve
modelled trends in all quantiles, but they apply those changes on top of the
observed series and so "[do] not explicitly bias-correct the daily time series
from the climate model". The stated gap is that the trade-offs between
preserving the mean trend and preserving the quantile trends "have yet to be
explored in a systematic manner" (Sect. 1).

The paper also inherits its tools: QM and DQM are run with the `fitQmapQUANT`
code of Gudmundsson (2014), extrapolation follows the constant correction of
Boé et al. (2007), and the GEV fitting follows Martins and Stedinger (2000) and
Cannon (2010) with the shape prior of Papalexiou and Koutsoyiannis (2013).

## Problem definition

### Problem

Correct the systematic distributional bias of a climate model against historical
observations, and, subject to that correction, keep the model's own projected
change. For precipitation that means keeping *relative* change, because relative
change is what links precipitation to projected warming through Clausius–
Clapeyron scaling (about 7 % more column water vapour per K; O'Gorman and
Muller 2010).

Two assumptions are stated and not tested here. First, stationarity of the
biases: QM "relies strongly on an assumption that the climate model biases to be
corrected are stationary", and "it is beyond the scope of this paper to address
this assumption" (Sect. 2). Second, that the model's trends are worth keeping at
all. The authors do not claim they are physically real: "we do not claim that
the GCM-projected trends are physically realistic, only that the QM can alter
these trends considerably as an artifact of its application" (Sect. 5d). They
also note the case against preservation — if a model resolves processes the
driving model does not, a good correction *should* change the trend, but a
univariate quantile mapping has no extra information with which to do so
(Sect. 1).

### Models

**Quantile mapping.** Equation (1) is the standard transfer function, built from
the historical period only:

$$\hat{x}_{m,p}(t) = F_{o,h}^{-1}\left\{F_{m,h}\left[x_{m,p}(t)\right]\right\} \qquad (1)$$

With empirical CDFs this is a lookup table read off the quantile–quantile plot,
and it is only defined over the historical range of the modelled data, so values
outside that range need extrapolation.

**Detrended quantile mapping.** Equation (2) scales the projected value by the
ratio of the historical to the projected modelled mean, maps it, then rescales:

$$\hat{x}_{m,p}(t) = F_{o,h}^{-1}\left\{F_{m,h}\left[\frac{\bar{x}_{m,h}\,x_{m,p}(t)}{\bar{x}_{m,p}(t)}\right]\right\}\frac{\bar{x}_{m,p}(t)}{\bar{x}_{m,h}} \qquad (2)$$

The authors note in parentheses that "for an interval variable, trend
removal/reimposition would be performed additively rather than
multiplicatively" (Sect. 2). Detrending also has a side benefit: it shifts the
future distribution back inside the support of the historical one, so less
extrapolation is needed.

**Quantile delta mapping.** Four steps. First, find where the projected value
sits in the projected distribution:

$$\tau_{m,p}(t) = F^{(t)}_{m,p}\left[x_{m,p}(t)\right], \quad \tau_{m,p}(t) \in [0,1] \qquad (3)$$

Second, form the modelled relative change at that same quantile, between the
historical period and time $t$:

$$\Delta_m(t) = \frac{F^{(t)-1}_{m,p}\left[\tau_{m,p}(t)\right]}{F_{m,h}^{-1}\left[\tau_{m,p}(t)\right]} = \frac{x_{m,p}(t)}{F_{m,h}^{-1}\left[\tau_{m,p}(t)\right]} \qquad (4)$$

Third, bias-correct that quantile against the historical observations:

$$\hat{x}_{o:m,h:p}(t) = F_{o,h}^{-1}\left[\tau_{m,p}(t)\right] \qquad (5)$$

Fourth, put the change back on:

$$\hat{x}_{m,p}(t) = \hat{x}_{o:m,h:p}(t)\,\Delta_m(t) \qquad (6)$$

The worked example in Figure 1 is $x_{m,p}(2065) = 36.5$ at
$\tau_{m,p} = 0.99$, giving $F_{m,h}^{-1}(0.99) = 28.5$, so
$\Delta_m(2065) = 36.5/28.5 = 1.28$; with $\hat{x}_{o:m,h:p}(2065) = 22.6$ the
corrected value is $22.6 \times 1.28 = 28.9$.

Three properties are stated. Equations (4) and (6) applied additively instead of
multiplicatively preserve absolute rather than relative changes. QDM "reduces to
standard QM if the modeled distribution does not change between the historical
and projected periods". And when the historical and projected samples have equal
length, values outside the historical range are handled by the algorithm itself
when Equation (6) reintroduces the change signal, so no separate extrapolation
rule is needed.

Appendix A derives the additive form, Equations (A1) to (A4), and shows by
rearrangement that it is identical to the equidistant CDF matching method of Li
et al. (2010, their Eq. 2):

$$\hat{x}_{m,p}(t) = x_{m,p}(t) + F_{o,h}^{-1}\left\{F^{(t)}_{m,p}\left[x_{m,p}(t)\right]\right\} - F_{m,h}^{-1}\left\{F^{(t)}_{m,p}\left[x_{m,p}(t)\right]\right\}$$

The authors add that this equivalence "may not have been clear in the original
exposition of equidistant CDF matching (and which is more suited to variables
like temperature rather than precipitation)" (appendix A). The equiratio form
raised in passing by Li et al. (2010) and rediscovered by Wang and Chen (2014) is
the multiplicative version. The conclusion drawn is that quantile delta change,
quantile perturbation, QDM and equidistant/equiratio CDF matching "are all
fundamentally similar".

### Data

**Synthetic.** Gamma distributions following Maurer and Pierce (2014), with the
density given as Equation (7). Observed historical $\sim \mathrm{gamma}(4, 7.5)$
so $\mu = 30$, $\sigma = 15$; modelled historical $\sim \mathrm{gamma}(8.15,
3.68)$ so $\mu = 30$, $\sigma = 10.5$; modelled projected $\sim
\mathrm{gamma}(16, 2.63)$ so $\mu = 42$, $\sigma = 10.5$. The model therefore
has the right mean but underestimates the observed standard deviation by 30 %,
and projects a 40 % increase in the mean with no change in the standard
deviation. A second synthetic experiment sweeps the modelled standard deviation
across ±60 % of the observed one.

**Real.** Daily precipitation from three CMIP5 GCMs — MIROC5 (r3i1p1), CanESM2
(r1i1p1) and CCSM4 (r1i1p1) — over the Canadian landmass, historical 1950–2005
and RCP8.5 2006–2100, regridded to a common 1.4° grid. Observations are the
Natural Resources Canada 1/12° gridded daily product, aggregated up to the GCM
grid; the authors "treat these observations as truth, but note that significant
uncertainty has been found for historical precipitation datasets". Calibration
is 1971–2000. Validation is out of calibration, 1950–70 plus 2001–05. Future
slices are the 2020s (2011–40), 2050s (2041–70) and 2080s (2071–2100).

Two implementation details matter. Dry days are censored below a trace of
0.05 mm day⁻¹: zeros are replaced with small random values before correction and
reset to zero after, which corrects wet-day frequency bias. The seasonal cycle is
handled by pooling days in sliding 3-month windows centred on the month of
interest, so December is corrected using November to January. Time-dependent
means and empirical quantiles of the projected data use 30-yr sliding windows
centred on the year of interest.

### Evaluation metrics

Three tests, in order.

1. **Historical distributional skill.** For each of the 11 ETCCDI indices at each
   grid cell, a two-sample Kolmogorov–Smirnov test at the 1 % level against the
   observed distribution. A cell passes when the null hypothesis of a common
   distribution is not rejected. Reported as the proportion of cells passing and
   as the median $D$ statistic, over the calibration period and the out-of-sample
   validation years.

2. **Preservation of relative change in moderate extremes.** Five of the 11
   indices are used — PRCPTOT, R95pTOT, R99pTOT, Rx1day and Rx5day — chosen
   because they do not depend on a large fixed-magnitude threshold. RMSE and bias
   of the bias-corrected 2080s-versus-1980s relative change, taken with respect
   to the raw GCM's relative change, over the Canada-wide domain.

3. **Preservation of relative change in rare extremes.** A stationary GEV is
   fitted to annual Rx1day series by generalized maximum likelihood, separately
   for each period, and 20-yr return magnitudes are compared. Scored by the
   distribution of grid-cell relative changes and by the spatial anomaly
   correlation, slope and intercept of the bias-corrected change field against
   the raw GCM change field.

Note the direction of tests 2 and 3: the reference is the climate model, not the
observations. The authors justify using extremes as the test because bias
correction is calibrated on daily series and is "not explicitly tuned to
replicate distributions of annual extremes, so this provides a stringent and
relatively independent test" (Sect. 1, following Bürger et al. 2012).

## Main results

### Synthetic data

With the model underestimating the observed standard deviation by 30 %, QM
turns the GCM's prescribed +40 % change in the mean into +58.6 %. "For these
distributions, neither DQM nor QDM leads to similar inflation of the trend
magnitude for the mean" (Sect. 4); the DQM and QDM values themselves are shown
in Figure 2 and are not tabulated.

| Method | Change in the mean, model prescribes +40 % |
|---|---|
| Raw GCM | +40 % (by construction) |
| QM | +58.6 % |
| DQM | closely reproduces +40 %; exact value not tabulated |
| QDM | closely reproduces +40 %; exact value not tabulated |

In the sweep over ±60 % variance bias, relative changes in the $\tau = 0.25$,
median and $\tau = 0.99$ quantiles are "reproduced perfectly" by QDM, which
follows from its construction. DQM's deviation is small for the median but larger
for $\tau = 0.25$ and $\tau = 0.99$. For QM, overestimated historical variance
inflates the corrected trends and underestimated variance suppresses them. The
numeric deviations are only in Figure 3 and are not tabulated.

### Historical skill, the 11 ETCCDI indices

DQM and QDM reduce to QM inside the calibration sample, so the three are
identical there by construction.

| Period | Statistic | Raw GCM | QM | DQM | QDM |
|---|---|---|---|---|---|
| Calibration 1971–2000 | median % of cells passing K-S | 40–60 % | >99.5 % | >99.5 % | >99.5 % |
| Calibration 1971–2000 | median $D$ | 0.4–0.5 | ~0.17 | ~0.17 | ~0.17 |
| Validation 1950–70, 2001–05 | median % of cells passing K-S | similar to calibration | 91 % | 91 % | 91 % |
| Validation 1950–70, 2001–05 | median $D$ | not given | 0.29 | 0.29 | 0.29 |

The reason the three tie in validation is given: "Because GCM-projected changes
due to external forcings over the historical period are small relative to natural
variability, use of additional information about trends in the mean or quantiles
by DQM and QDM does not lead to improvements in historical skill relative to QM"
(Sect. 5b). The authors also flag that the rejection rate is lower than a nominal
1 % level would imply, and keep the test as a diagnostic for consistency with
Bürger et al. (2012).

### Relative change in the five ETCCDI indices, 2080s

Ordered by RMSE against the raw GCM change, aggregated over five indices and
three GCMs: QDM best, DQM close behind, QM far worse. The RMSE values themselves
appear only as stacked bars in Figure 7 and are not tabulated, so only the
ordering can be quoted.

| Method | RMSE rank vs GCM change | Note from the text |
|---|---|---|
| QDM | 1 | "reproduces GCM-projected changes marginally better than DQM overall" |
| DQM | 2 | "QDM and DQM outperform QM by a large margin" |
| QM | 3 | biases "of comparable magnitude to the underlying GCM change signals themselves" |

The ranking holds for every index except PRCPTOT, the index closest to the mean
of the distribution, where QDM is worst for MIROC5 and CanESM2 and best for
CCSM4. The authors treat this as expected, since QDM constrains the quantiles
and not the mean. Bias distributions for DQM and QDM are of similar size,
"although QDM has a large impact in a small number of cases, particularly for
R99pTOT", while QM shows "heavily right-skewed distributions, characterized by
large positive outliers".

### Relative change in 20-yr return values

Median relative change in the 20-yr return magnitude, by time slice, across the
three GCMs:

| Period | Raw GCM | QM |
|---|---|---|
| 2020s | +5 % (CCSM4) to +8 % (MIROC5) | +11 % to +15 % |
| 2050s | +10 % to +18 % | +19 % to +28 % |
| 2080s | +18 % to +37 % | +29 % to +53 % |

DQM and QDM medians are not given as percentages; they are given as
amplification factors relative to the raw GCM median:

| Method | Amplification of the median relative change | Largest projected change, 2080s | Inflation of the maximum vs raw GCM |
|---|---|---|---|
| Raw GCM | 1 (reference) | +117 % (MIROC5); never exceeds +120 % | 1 |
| QM | 1.45–2.44 | +573 % (MIROC5) | over 4× |
| DQM | 1.15 (2080s MIROC5) – 2.07 (2020s CCSM4) | +241 % (MIROC5) | 2.0× |
| QDM | 1.1 (2080s MIROC5) – 1.7 (2020s CCSM4) | +138 % (MIROC5) | 1.2× |

The 2080s spatial field of relative change, scored against the raw GCM field:

| Method | Spatial anomaly correlation with the GCM change field |
|---|---|
| QM | 0.52–0.62 |
| DQM | 0.73–0.82 |
| QDM | 0.82–0.86 |

QM also "suffers from the largest conditional and unconditional bias" in the
slope and intercept of the fit against the GCM field; the slope and intercept
values are shown on Figures 11 to 13 and are not tabulated in the text.

The mechanism behind the QM outliers is demonstrated, not just asserted:
underestimation of the historical observed GEV variance is found at 68 % of grid
cells where QM's relative change exceeds the GCM's by more than 200 percentage
points, and at *all* cells where the difference exceeds 300 percentage points.

Historical 20-yr return magnitudes are corrected well by all three methods. In
the 1980s calibration period the bias-corrected maps are "visually
indistinguishable from observations", with anomaly correlations above 0.99,
slopes 0.98–1.01 and intercepts 0.06–0.44 mm day⁻¹. In the out-of-calibration
years the ranges are 0.91–0.93, 1.03–1.07 and 0.92–1.73 mm day⁻¹. These ranges
span the three bias-corrected GCMs; the paper does not separate them by method.
For reference, the raw GCM 1980s return magnitudes have anomaly correlations of
0.77–0.87 against observations.

## Discussion

Standard QM should not be used when the model's projected trend in extremes
matters: "If the goal is to reproduce relative trends in precipitation extremes
as originally simulated by a climate model, then the use of standard QM for bias
correction cannot be recommended" (Sect. 6). Some form that accounts for future
trends — at minimum in the mean, better in all quantiles — is preferred.

The authors do not claim the trends are right, only that QM changes them: "we do
not claim that the GCM-projected or bias-corrected trends are physically
realistic, only that the QM can alter the underlying GCM trends considerably as
an artifact of its application" (Sect. 6). They note that QM's projected changes
above +300 % exceed what Clausius–Clapeyron or observed super
Clausius–Clapeyron scaling would anticipate in the extratropics.

Trade-off between the two trend-preserving methods: DQM guarantees the mean,
QDM guarantees the quantiles, and neither guarantees the other. In the synthetic
example, constraining all quantiles gave "a reasonably tight constraint on the
mean", but this is an observation about that example, not a proof.

Limits the authors acknowledge:

- Stationarity of the biases is assumed and not assessed.
- Only the simplest quantile estimator was tested — empirical CDFs over 30-yr
  sliding windows. Asynchronous regression, time-dependent quantile regression
  or parametric conditional density estimators "may be more robust" or "may also
  perform better".
- Corrections were not applied separately on different time scales, as Hempel
  et al. (2013) recommend.
- Univariate, grid-cell corrections can only address one of the three error
  sources, and can distort spatial variability and the model's modes of
  variability; this "points to the need for a multivariate bias correction
  algorithm".
- A physical test of whether a method's trends are realistic is possible — a
  nonstationary GEV analysis with temperature as covariate — but was not done.
- The general warning: "Care must be taken when applying quantile mapping
  algorithms, as is the case with any postprocessing technique for climate model
  outputs, to ensure their fitness for the purpose at hand" (Sect. 6).

## Relevance to this project

*Everything below is my own reading, not the paper's content.* Mapping of terms:
the paper's "transfer function" is our transfer function, its "quantile mapping"
is our fitted per-(pixel, season) mapping, and its "grid cell" is our pixel.

- **This is the paper to cite for the projection step, not for the baseline.**
  Our current baseline fits and scores inside one period, where QM, DQM and QDM
  are identical by construction — the paper demonstrates that tie itself
  (validation median 91 % of cells for all three). The difference only appears
  once we push to 2081–2100. Plan for it now rather than discovering it then.

- **Use the additive form, and know what it is.** We work on temperature in °C,
  an interval variable in the paper's terms. Equations (4) and (6) applied
  additively preserve absolute changes, and appendix A proves that additive QDM
  is algebraically the same as Li et al. (2010) equidistant CDF matching. So a
  literature search under either name returns the same method, and we should say
  so once in our own write-up instead of treating them as two options.

- **Two caveats before we import the headline result.** First, the paper is
  about precipitation, and it says so: it excludes temperature deliberately
  because natural variability relative to the forced trend is much larger for
  precipitation. The +500 % inflation numbers do not transfer to our variable.
  Second, and more directly relevant, the paper explicitly restricts itself to
  bias correction between fields at comparable resolution and warns that using
  quantile mapping *for* downscaling distorts finescale variability (Maraun
  2013; Gutmann et al. 2014). Our framework does exactly what the paper excludes.
  That is a limitation to state in our own text, citing this sentence, not a
  reason to stop.

- **The diagnostic we should run is cheap and concrete.** The mechanism is a
  variance bias: QM inflates trends where the model underestimates observed
  variance and suppresses them where it overestimates. We already compute
  per-(pixel, season) statistics. Adding a map of modelled-minus-observed
  standard deviation tells us in advance which pixels and seasons would have
  their projected trend distorted by plain QM. The paper's 68 % and 100 %
  figures make this a demonstrated mechanism, not a guess.

- **Their seasonal handling differs from ours and is worth a line in ADR terms.**
  They pool sliding 3-month windows centred on each month, which is a smoother
  distribution window than our four fixed seasons. Neither paper tests which is
  better — Cannon et al. assert the choice rather than compare it — so this is
  not evidence against our season convention, only a noted alternative.

- **Their sample sizes set an expectation we cannot meet yet.** Calibration is
  30 years and the projected quantiles use 30-yr sliding windows, and QDM's clean
  handling of out-of-range future values depends on historical and projected
  samples being of equal length. Our 1990–1999 baseline is 10 years. When we move
  to the planned 1980–2004 training period we get 25, which is close; until then,
  any QDM result we produce is outside the regime this paper tested.

- **QDM removes our extrapolation problem, which is a practical gain.** Under
  strong warming, a large share of future daily values will fall above the
  historical range, and our current constant-correction-style handling of the
  top node is exactly the weak point. QDM handles those values inside the
  algorithm when Equation (6) reapplies the change signal.

- **Expect QDM to be slightly worse on the mean.** PRCPTOT is the index closest
  to the mean, and QDM lost on it for two of three GCMs. If we score projected
  change with a mean-based metric, DQM may look better than QDM even when QDM is
  the right choice for the tails. Pick the metric before picking the method.
