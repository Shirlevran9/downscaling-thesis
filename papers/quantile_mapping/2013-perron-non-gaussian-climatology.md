# A global climatology of skewness and kurtosis in daily atmospheric data

## Title

**Climatology of Non-Gaussian Atmospheric Statistics**

Maxime Perron, Philip Sura

*Journal of Climate* **26**, 1063–1083, 1 February 2013.
doi:10.1175/JCLI-D-11-00504.1 · American Meteorological Society ·
manuscript received 8 September 2011, final form 6 June 2012.

PDF: `Perron2013_NonGaussian_JClimate.pdf`

## Abstract

Earth scientists often assume that a variable is Gaussian in time. The paper
starts from the position that this assumption is usually wrong: "A common
assumption in the earth sciences is the Gaussianity of data over time. However,
several independent studies in the past few decades have shown this assumption
to be mostly false" (Abstract). Before non-Gaussian climate statistics can be
studied, the authors argue, someone has to compile the higher moments
systematically. That compilation is the paper.

The authors take 62 years of daily data from the NCEP–NCAR Reanalysis I project
(1948–2009) and compute skewness and kurtosis of the daily anomalies at every
grid point, for nine atmospheric variables. For each variable they show global
maps of skewness and kurtosis at one chosen pressure level, for December–February
(DJF) and June–August (JJA), plus zonally averaged vertical cross sections. A
Monte Carlo test against fitted Gaussian red noise gives the significance. The
main conclusion is stated plainly: "it is evident that Gaussianity is actually
quite rare in the atmosphere. In fact, for daily observations, non-Gaussian
statistics are the norm and not an exception" (Sect. 4). The authors offer the
result as "a one-stop benchmark for future model validations" (Sect. 4).

## Terms and notation

### Terms

A **probability distribution function (PDF)** here is the distribution of daily
anomalies at one grid point. The paper's whole subject is the shape of that PDF
beyond its mean and variance.

**Skewness** is a measure of the asymmetry of the PDF. It is zero for a Gaussian.
"A positive number denotes a heavier tail in the positive anomalies, and a
negative number denotes a heavier tail in the negative anomalies" (Sect. 2b).

**Kurtosis** is a measure of the peakedness of the PDF. A Gaussian has kurtosis
three. Above three the distribution is more peaked, which also means more data in
the tails; below three it is flatter, with fewer data in the tails and near the
mean. **Excess kurtosis** is the kurtosis minus three, so a Gaussian has skewness
and excess kurtosis both zero. The paper plots excess kurtosis. Note that the
figure captions and most of the text still say "kurtosis" where they mean excess
kurtosis; the paper states the convention once, in Section 2b, and then relies on
it.

An **anomaly** is the value at one day minus the 62-year time mean for that
calendar day at that grid point. Every moment in the paper is computed from
anomalies, not from the raw field, so the yearly cycle is removed first.

**Reanalysis** is described as "a cluster of observations from different sources
that uses models to fill in the gaps" (Sect. 2a), citing Kalnay et al. (1996).

**Red noise** in the significance test means a first-order autoregressive, AR(1),
process. It is Gaussian by construction, and it is the null hypothesis: the test
asks whether an observed skewness or kurtosis value could have come from Gaussian
data with the same variance and the same decorrelation time.

**Quasigeostrophic potential vorticity (QGPV)** is used because it "has the
advantage of requiring only information about the geopotential height $\Phi$ and
temperature $T$ fields" (Sect. 2a).

### Notation

The moments and their errors:

| Symbol | Meaning |
|---|---|
| $g_1$ | Skewness, the third central moment, Equation (3) |
| $g_2$ | Kurtosis, the fourth central moment, Equation (4) |
| $X$ | The variable |
| $\mu$ | The mean of the variable, $E[X]$ |
| $E[\,\cdot\,]$ | Expected value |
| $\varepsilon_{g_1}$, $\varepsilon_{g_2}$ | Standard errors of skewness and kurtosis, Equations (5) and (6) |
| $N_i$ | Number of independent data points in the time series at one location |

The nine variables, with the pressure level each is mapped at (Table 1):

| Symbol | Variable | Derived from | Level plotted |
|---|---|---|---|
| $\Phi$ | Geopotential height | base variable | 500 hPa |
| $\zeta$ | Relative vorticity | $u$, $v$ | 300 hPa |
| QGPV | Quasigeostrophic potential vorticity | $\Phi$, $T$ | 300 hPa |
| $u$ | Zonal wind | base variable | 925 hPa |
| $v$ | Meridional wind | base variable | 925 hPa |
| $|\mathbf{u}|$ | Horizontal wind speed | $u$, $v$ | 925 hPa |
| $\omega$ | Vertical velocity in pressure coordinates | base variable | 500 hPa |
| $T$ | Air temperature | base variable | 925 hPa |
| $q$ | Specific humidity | base variable | 925 hPa |

The symbols of the null-hypothesis model, Equations (A1) to (A4):

| Symbol | Meaning |
|---|---|
| $\lambda$ | Positive damping constant of the variable $x$ |
| $\eta(t)$ | Gaussian white noise, amplitude $\sigma$ |
| $\sigma_\eta$ | Standard deviation of the discrete white noise |
| $\sigma_x^2$ | Variance of $x$ |
| $1/\lambda$ | Autocorrelation time scale |
| $R$ | Residual of the AR(1) fit, Equation (A4) |

## Previous work

The paper places itself after two separate lines of work.

**Regimes and bimodality.** Charney and DeVore (1979), Wiin-Nielsen (1979) and
Hart (1979) showed that multiple equilibrium states, blocked and zonal, could
exist in the atmosphere. A long list of later studies looked for such regimes in
observed midlatitude flows (Hansen and Sutera 1986; Mo and Ghil 1988; Molteni
et al. 1990; Kimoto and Ghil 1993; Cheng and Wallace 1993; Corti et al. 1999;
Smyth et al. 1999; Monahan et al. 2001; Christiansen 2005).

**Empirical studies of local higher moments.** White (1980), Trenberth and Mo
(1985), Nakamura and Wallace (1991), Holzer (1996), Monahan (2006b) and
Petoukhov et al. (2008) all measured deviations from Gaussianity in key
variables. The authors say the theoretical reasons came only later, in Holzer
(1996), Sura and Sardeshmukh (2008), Sardeshmukh and Sura (2009) and Sura and
Perron (2010).

The gap the authors claim is one of coverage and consistency, not of correctness:
"To date, no paper has summarized the non-Gaussianity of the atmosphere in a
comprehensive and elegant manner" (Sect. 1, *Objectives*). They list the specific
limits of each predecessor. White (1980) covered only the Northern Hemisphere and
only 11 years of midtropospheric data. Trenberth and Mo (1985) analysed the
Southern Hemisphere with even fewer data. Nakamura and Wallace (1991) filtered
out low-frequency variability. Holzer (1996) gave only zonally averaged cross
sections of geopotential height. Monahan (2006b) covered horizontal wind speed
alone, though across four datasets. Petoukhov et al. (2008) covered more
variables, including temperature, but in ERA-40. The authors' conclusion is that
"there is no study that presents the non-Gaussianity of key variables (from one
dataset) together in one place".

On data quality they acknowledge one known problem and answer it with earlier
work: some regions, "e.g. the Southern Hemisphere before the 1970s; see Hines
et al. (2000)", carry slight biases, but Sura and Perron (2010) showed that "the
general large-scale patterns of higher statistical moments remain stable over the
reanalysis period" (Sect. 2a).

## Problem definition

### Problem

Produce one consistent, global climatology of the third and fourth central
moments of daily atmospheric anomalies, for nine variables, from a single
dataset, split by season and shown both horizontally and in the vertical. The
aim is descriptive. The paper does not propose a method, and **it compares no
methods** — there is no competing technique in it, so there is no *Models*
subsection here.

Two assumptions are worth naming because the authors name them. First, the
significance test assumes the null process is AR(1) Gaussian red noise; the
authors check this assumption in the appendix rather than assert it. Second, the
standard-error formulas in Equations (5) and (6) "are derived assuming a Gaussian
distribution and are, therefore, approximately valid for weakly non-Gaussian
data" (Sect. 2b2). The authors state this as a reason not to rely on those
formulas for the plots.

The moments themselves are the standard ones. Skewness is Equation (3):

$$g_1 = \frac{E\left[(X-\mu)^3\right]}{E\left[(X-\mu)^2\right]^{3/2}} \qquad (3)$$

and kurtosis is Equation (4):

$$g_2 = \frac{E\left[(X-\mu)^4\right]}{E\left[(X-\mu)^2\right]^{2}} \qquad (4)$$

Two variables are derived rather than read from the dataset. Relative vorticity
is Equation (1), in spherical coordinates:

$$\zeta = \frac{1}{a\cos\phi}\left[\frac{\partial v}{\partial \lambda} - \frac{\partial (u\cos\phi)}{\partial \phi}\right] \qquad (1)$$

with $a$ the distance from the centre of the earth, $\phi$ latitude and $\lambda$
longitude. QGPV is Equation (2), built from $\Phi$ and $T$ with a static
stability parameter $\sigma_p$, the Coriolis parameter $f = 2\Omega\sin\phi$, the
gas constant for dry air $R$, the specific heat capacity $C_p$, pressure $p$ and
reference surface pressure $p_s$, following Evans and Black (2003). The full
expression is long and the PDF text extraction of it is partly garbled, so it is
not reproduced here; see Equation (2) in the paper.

### Data

Daily-averaged data from the NCEP–NCAR Reanalysis I project, covering 62 years,
1948–2009, globally. At each grid point the daily anomaly is the full field on
that day minus the 62-year time mean for that day, so the yearly cycle is removed
before any moment is computed.

Moments are computed over the whole record and also for the DJF and JJA subsets
separately. The nine variables and their plotted levels are in the notation table
above. Levels were chosen for physical reasons: 500 hPa for $\Phi$ and $\omega$
as the level of nondivergence, 300 hPa for the vorticity variables "where
adiabatic and friction effects are minimized and where the jet stream is found",
and 925 hPa for the winds, $T$ and $q$ because most solar radiation is used to
warm the surface. Specific humidity is not recorded above 300 hPa in this
dataset, so the 300–100 hPa range cannot be analysed for $q$.

### Evaluation metrics

There is no skill score in this paper, because nothing is being predicted. What
stands in its place is a significance threshold.

The authors first give the textbook standard errors, Equations (5) and (6):

$$\varepsilon_{g_1} = \sqrt{6/N_i} \quad (5) \qquad\text{and}\qquad \varepsilon_{g_2} = \sqrt{24/N_i} = 2\,\varepsilon_{g_1} \quad (6)$$

from Brooks and Carruthers (1953), where $N_i$ is the number of *independent*
points. They then reject these as the basis for the plots, because the
distribution shape is not known in advance and the formulas assume Gaussianity.
They note that resampling or subsampling would be the proper answer (Gluhovsky
and Agee 2009; Gluhovsky 2011) but choose a cheaper route.

The route chosen is a Monte Carlo test. At every grid point they fit a Gaussian
red-noise AR(1) process, Equations (A1) to (A3), matching the observed variance
and decorrelation time scale. They generate surrogate series of the same length
as the original, run 200 simulations per grid point, and remove the five lowest
and five highest values to get a 95 % confidence interval around zero for
skewness.

The result of that test is one fixed contour: "we get a 95 % skewness confidence
interval that is smaller than $\pm 0.12$ for eight of the nine variables (QGPV
being the only outlier) outside of the tropics. Thus, for consistency, we chose
$\pm 0.12$ as the nonshaded contour interval for every global plot" (Sect. 2b2).
The authors say this is an overestimate in most areas. Because the kurtosis
standard error is twice the skewness one, the same $\pm 0.12$ band gives
"approximately 95 % confidence in skewness and 68 % confidence in kurtosis"
(Appendix).

They also check the AR(1) assumption itself. The residual $R$ of the AR(1) fit,
Equation (A4), has an autocorrelation time scale "smaller than one day for every
variable of interest", so they rule out low-frequency variability affecting the
significance. Had it been otherwise, they say, surrogate data preserving the full
spectral structure would have been needed (Christiansen 2005, 2009).

## Main results

**Almost every number in this paper is a colour on a map.** The results are 72
panels across nine figures, plus the confidence-interval maps in Figures A2 and
A3. Skewness and kurtosis values are not tabulated anywhere in the paper, and
the only numbers given in the text are the significance threshold and the
worked example below. The findings can therefore be reported only as patterns
and signs, in the authors' own words. Reading values off those maps is not
possible from the text.

The numbers the paper does state:

| Quantity | Value | Where |
|---|---|---|
| Nonshaded contour, all global plots | $\pm 0.12$ in skewness | Sect. 2b2 |
| Confidence at that contour | $\approx 95\,\%$ skewness, $\approx 68\,\%$ kurtosis | Appendix |
| Monte Carlo simulations per grid point | 200, with the 5 lowest and 5 highest removed | Appendix |
| Worked example: days in a midlatitude season-record | $91 \times 62 = 5642$ | Sect. 2b2 |
| Assumed decorrelation, midlatitude Rossby wave | about 7 days | Sect. 2b2 |
| Resulting independent points $N_i$ | $5642/7 = 806$ | Sect. 2b2 |
| Resulting $\varepsilon_{g_1}$ from Equation (5) | $0.086$ | Sect. 2b2 |

That worked example is a check, not a threshold: the authors use it to argue
that Equations (5) and (6) "are good approximations even for non-Gaussian data",
while stressing that "the nonshaded contour interval in the plots was not based
on the standard error formula, but rather on the Monte Carlo method".

The overall finding, and the one the paper rests on, is that daily atmospheric
anomalies are usually not Gaussian: "for daily observations, non-Gaussian
statistics are the norm and not an exception. Only if we calculate averages in
space and/or time does the central limit theorem kick in and the statistics
become more Gaussian" (Sect. 4).

Per variable, the summarised patterns (Sect. 4, with detail from Sect. 3):

| Variable | Skewness | Kurtosis |
|---|---|---|
| $\Phi$, 500 hPa | Negative bands at midlatitudes, centred near 30° in both hemispheres in DJF; positive in the tropics and near the poles, and the positive part largely disappears in JJA | Positive near 30°, bounded by negative values to the north and south; the negative areas grow in JJA |
| $\zeta$, 300 hPa | Mainly a function of latitude: four bands separated by the equator and the mean storm tracks. Positive equatorward of the NH storm track and negative poleward; reversed in the SH | Positive globally, except strongly negative bands along the mean storm tracks |
| QGPV, 300 hPa | Similar to $\zeta$, but neutral poleward of the storm track. Sign is almost entirely a function of hemisphere: positive NH, negative SH | Almost uniformly negative poleward of the storm tracks; mixed in the tropics, turning more negative in JJA |
| $u$, 925 hPa | Positive near the equator, negative poleward of 30°, weaker in the NH in boreal summer; stronger over the oceans. Zonally averaged, the signal almost disappears | Hard to analyse; mostly insignificant in the troposphere apart from narrow negative regions in the subtropics |
| $v$, 925 hPa | Mostly positive in the NH and negative in the SH, with pockets of opposite sign almost exclusively over land. Near zero once zonally averaged | Positive in the tropics and over Antarctica, negative elsewhere. Only the kurtosis survives the zonal average |
| $\|\mathbf{u}\|$, 925 hPa | Positive almost everywhere; a neutral band near 50°S and mixed values in the tropics. Large decrease near the Indian Ocean between DJF and JJA | Mostly negative, except zonal bands near the poles and isolated tropical and subtropical pockets |
| $\omega$, 500 hPa | Neutral or negative almost everywhere; positive only over parts of Antarctica and small areas where trade wind inversions are common | Positive almost everywhere, with thin neutral bands at the equator and along the seasonally active storm track |
| $T$, 925 hPa | Depends strongly on local geography and on the climatological highs and lows. Generally negative over continents | Noisy, with many alternating pockets; a general trend of negative kurtosis at mid to high latitudes over the ocean basins |
| $q$, 925 hPa | Positive in deserts and polar regions, negative over the tropical oceans | Similar in shape to the skewness field but with sharper gradients, negative at mid to high latitudes |

The temperature result is worth stating in full, since it is the variable with the
most structure. "In the Northern Hemisphere we observe positive skewness to the
west and negative skewness to the east of climatological lows (such as the
Icelandic and Aleutian lows). Conversely, we observe negative skewness to the west
and positive skewness to the east of climatological highs (such as the Bermuda and
Siberian highs)" (Sect. 3h). The gradient in skewness is largest in the winter
season of each hemisphere. Over the oceans, the seasonal change is mainly in polar
waters: positive over the Southern Ocean in DJF turning negative in JJA, and
roughly neutral over the Arctic in DJF turning strongly positive in summer. Because
temperature non-Gaussianity follows the land–sea distribution so closely, the
zonally averaged plots "do not show interesting patterns"; the one persistent
feature there is negative skewness at midlatitudes between 700 and 400 hPa, near
the jet stream. The authors note that Petoukhov et al. (2008) computed the same
quantities and that their values "differ slightly because of the different
reanalysis data used" — no numbers are given for that difference.

Two mechanisms are offered where the sign of the skewness has an obvious physical
cause, and both are asserted rather than tested. Horizontal wind speed is
positively skewed because "the wind speed at any given point cannot be negative"
and usually stays close to zero, so extremes can only fall on the positive side.
Vertical velocity is negatively skewed because "it is easier for a parcel of air
to rise quickly than it is for it to sink quickly" — in pressure coordinates
upward motion is negative $\omega$.

**Significance.** For most variables the 95 % skewness confidence interval is
below the 0.12 shading threshold. Geopotential height and air temperature show "a
band of higher error values in the tropics, but at the mid and high latitudes the
patterns are fully significant" (Appendix). QGPV is the exception and fails:
"The local QGPV significance testing … is the only one that failed out of the
nine variables" (Sect. 3c). The authors keep its patterns anyway, arguing they
are "consistent spatially" while the Monte Carlo test only judges significance
locally in time. This is an argument, not a test.

**Two apparent inconsistencies in the text.** In the summary bullet for
meridional wind, the kurtosis is said to be "positive in the tropics and
Antarctica and positive elsewhere" (Sect. 4), while Section 3e says positive in
the tropics and over Antarctica and negative elsewhere; the summary bullet looks
like a typographical error. For vertical velocity, Section 3g says the kurtosis
"is decidedly positive almost everywhere" and then closes the same paragraph with
"resulting in a PDF with negative kurtosis". These two statements contradict each
other and the paper does not resolve which is meant.

## Discussion

Mean and variance are not enough for a climatology of a non-Gaussian variable:
"the variability of a non-Gaussian variable is not sufficiently described by its
mean and variance; higher-order moments are needed to describe the
characteristics of non-Gaussian phenomena" (Sect. 4).

The authors name two uses for this. Weather risk management needs the detailed
non-Gaussian statistics "to make accurate statements about the probability of
extreme events". And higher-moment statistics are "necessary to validate the
ability of numerical models to reproduce extreme events"; the climatology is
offered as "a one-stop benchmark for future model validations".

Gaussianity is the exception for daily data, and averaging in space or time
restores it through the central limit theorem.

Beyond the broad zonal patterns, the fields become complex, and the authors
attribute the small-scale features to "land–sea contrasts, local sea surface
temperatures, and topography" — offered as an interpretation, not a
demonstration.

The limits they acknowledge:

- **No theory.** "we only have a very rudimentary physical understanding of why
  we see the global patterns presented here. In fact, we do not have a fully
  developed theory of non-Gaussian statistics in the atmosphere" (Sect. 4). The
  stochastic multiplicative-noise framework "only predicts the general form of
  the stochastic equation, but not the values of the parameters going into it".
- **The significance test is weaker than it should be.** A resampling or
  subsampling method is preferred where linearity assumptions are not met; AR(1)
  Gaussian red noise was used for cost.
- **QGPV did not pass that test**, and is retained on a spatial-consistency
  argument.
- **The record is one reanalysis over one period.** Changes over longer past and
  future ranges are named as future work, using twentieth-century reanalysis and
  IPCC data.

## Relevance to this project

*Everything in this section is my own reading. Sections 1 to 7 above contain only
what the paper says.*

Mapping of terms: the paper's "air temperature at 925 hPa" is not our target, and
its "grid point" is our **pixel** only by analogy — its grid is the coarse NCEP
reanalysis grid, far coarser than ERA5-Land at 0.1°.

- **This is the citation for why a Gaussian transfer function is the wrong
  default.** Our distribution-derived member assumes normality on both sides.
  Perron and Sura give a direct, general statement that daily temperature
  anomalies are skewed, which is the cleanest justification we have for
  preferring empirical or spline transfer functions over a normal fit. Cite
  Section 4 for "non-Gaussian statistics are the norm and not an exception".
- **It supports removing the seasonal cycle, which is what our per-season
  stratification does.** They compute moments from anomalies about the 62-year
  daily mean, and separately for DJF and JJA. That is the same reasoning behind
  one transfer function per (land pixel, season): pooling across the year mixes
  distributions with different shapes.
- **It warns that our domain is a hard case for temperature.** Temperature
  skewness is shown to depend on land–sea contrast, topography and the position
  of climatological highs and lows. Our domain sits between the Mediterranean and
  a large desert land mass, so we should expect the shape of the distribution to
  vary from pixel to pixel rather than smoothly — an argument for fitting per
  pixel, not per region.
- **It gives no numbers we can use as a target.** Everything is mapped, nothing
  tabulated, and the maps are at 925 hPa on a coarse global grid. Do not quote a
  skewness value for the Eastern Mediterranean from this paper. If we want one,
  we must compute it from our own ERA5-Land record.
- **Their significance threshold is a usable template for ours.** If we report
  per-pixel skewness of the target, the pattern to copy is: fit an AR(1) process
  to the pixel's own series, simulate, and treat anything inside the simulated
  band as not different from Gaussian. Our record is 10 years of daily data, not
  62, so our band will be much wider than their $\pm 0.12$ — worth stating
  explicitly if we make that claim.
- **Their independence correction applies directly to us.** Daily temperature is
  autocorrelated, so the effective sample size is far below the number of days.
  Their worked example divides 5642 days by a 7-day decorrelation time to get 806
  independent points. Any significance statement we make about a per-season
  per-pixel statistic should carry the same division, since one season-year gives
  about 90 days and therefore roughly a dozen independent points.
- **It does not address bias correction at all.** There is no transfer function,
  no downscaling, and no method comparison here. Use it only for the claim about
  distribution shape; for anything about how to correct a distribution, the
  quantile-mapping papers in this folder are the sources.
