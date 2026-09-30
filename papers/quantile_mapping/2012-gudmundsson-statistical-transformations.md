# Downscaling RCM precipitation using statistical transformations

## Title

**Technical Note: Downscaling RCM precipitation to the station scale using statistical transformations – a comparison of methods**

L. Gudmundsson, J. B. Bremnes, J. E. Haugen, T. Engen-Skaugen

*Hydrology and Earth System Sciences* **16**, 3383–3390, 2012.
doi:10.5194/hess-16-3383-2012 · CC BY 3.0 · received 10 April 2012, published 21 September 2012.

PDF: `Gudmundsson2012_QM_transformations_HESS.pdf`

## Abstract

Regional climate models (RCMs) get precipitation wrong in systematic ways, so
their output has to be corrected before it can be used for local impact studies.
A popular family of corrections works on the distribution: find a function that
reshapes the modelled distribution until it looks like the observed one. Many
such functions have been proposed, and the paper's starting point is that nobody
had compared them on a common footing. As the authors put it, "the diversity of
suggested methods renders the selection of optimal techniques difficult and
therefore there is a need for clarification" (Abstract).

The paper does two things. First it sorts the existing methods into three
families — distribution derived, parametric, and nonparametric — separated by
what each assumes about the data. Then it tests them all on daily precipitation
from 82 stations in Norway, scoring them with cross-validation. The headline
result is that "nonparametric transformations have the highest skill in
systematically reducing biases in RCM precipitation" (Abstract). The two
nonparametric methods, QUANT and SSPLIN, rank first and second overall, and they
keep that advantage in the extreme upper tail where several other methods fail.
The methods are released as the R package `qmap`.

## Glossary

| Symbol or term | Meaning in this paper |
|---|---|
| $P_o$ | Observed precipitation |
| $P_m$ | Modelled precipitation, from the RCM |
| $\hat{P_o}$ | The method's best estimate of $P_o$ |
| $h$ | The transformation: the function that maps a modelled value to a corrected one |
| $F_m$ | Cumulative distribution function (CDF) of $P_m$ |
| $F_o^{-1}$ | Inverse CDF, or quantile function, of $P_o$ |
| $a, b, c, x, \tau$ | Free parameters of the parametric transformations, fitted to data |
| RCM | Regional climate model; here HIRHAM at 25 km, driven by the ERA40 reanalysis |
| Statistical transformation | The authors' deliberately neutral name for this whole method family (Appendix A) |
| Distribution derived | Family 1: assume a theoretical distribution for each side and solve for $h$ |
| Parametric | Family 2: fit a chosen functional form directly to the q–q relation |
| Nonparametric | Family 3: no assumed form; read $h$ off the data |
| QUANT | Empirical quantile mapping, using tables of empirical percentiles |
| SSPLIN | Cubic smoothing spline fitted to the q–q relation |
| Bernoulli-X | A mixture: Bernoulli for whether it rains, distribution X for how much |
| MAE | Mean absolute error between the observed and the corrected empirical CDF |
| $\mathrm{MAE}_{0.1} \dots \mathrm{MAE}_{1.0}$ | The same error, computed separately in ten probability bands each 0.1 wide |
| Relative error | A method's MAE divided by the MAE of the uncorrected model output |

## Previous work

The paper opens from an established position: RCM precipitation is biased, for
reasons including "limited process understanding or insufficient spatial
resolution", and therefore needs post-processing before use. It cites a long
list of studies that had already tried to do this, among them Ines and Hansen
(2006), Engen-Skaugen (2007), Schmidli et al. (2007), Themeßl et al. (2011) and
Teutschbein and Seibert (2012).

The authors identify two specific gaps.

**No agreement on which method to use.** "There is no general agreement on the
optimal technique to solve this task and the approaches employed differ at times
substantially. Therefore, there is an urgent need for clarifying the relation
among different approaches as well as for an objective assessment of their
performance" (Sect. 1). Methods had been published and used, but not placed
against each other.

**The existing scores cannot be combined.** Earlier studies had scored these
methods with the root mean square error (Piani et al., 2010b), the
Kolmogorov–Smirnov statistic (Dosio and Paruolo, 2011), or individual moments
such as the mean, standard deviation and skewness. The paper's objection is
practical: "One limitation of the scores above is that they can often not be
summarised into one overall measure, e.g. due to different physical units or lack
of normalisation. This renders a global evaluation, combining the advantages and
drawbacks of different methods, difficult" (Sect. 4.1). So the paper proposes its
own score set.

Individual methods are inherited rather than invented. The parametric forms in
Equations (4) to (7) are all taken from Piani et al. (2010b). QUANT follows the
procedure of Boé et al. (2007), including how it handles values above the
training range. The Bernoulli mixtures come from Thom (1968), Mooley (1973) and
Cannon (2008, 2012).

## Problem definition

The goal is to find a function $h$ that maps the modelled variable so that its
new distribution equals the distribution of the observed variable. Following
Piani et al. (2010b), Equation (1) states it in general form:

$$P_o = h(P_m) \qquad (1)$$

This is an application of the probability integral transform. When the
distribution is known, Equation (2) makes $h$ explicit:

$$P_o = F_o^{-1}\left(F_m(P_m)\right) \qquad (2)$$

In words: take a modelled value, ask what fraction of modelled values fall below
it, then return the observed value with that same fraction below it. The practical
difficulty, and the whole subject of the paper, is that $F_o$ and $F_m$ are not
known and $h$ has to be approximated from data.

The three families differ in how they approximate it.

**Distribution derived** (Sect. 2.1) assumes a theoretical distribution on each
side and solves Equation (2). For precipitation the usual choice is a mixture of
Bernoulli — for whether a day is wet at all — with a Gamma for the intensity.
The paper also tests Bernoulli-Weibull, Bernoulli-Log-normal and
Bernoulli-Exponential. Parameters are estimated by maximum likelihood "for both
$P_o$ and $P_m$ independently" — a detail that turns out to explain the results.

**Parametric** (Sect. 2.2) fits a chosen shape straight onto the q–q relation:

$$\hat{P_o} = b P_m \qquad (3)$$
$$\hat{P_o} = a + b P_m \qquad (4)$$
$$\hat{P_o} = b P_m^{\,c} \qquad (5)$$
$$\hat{P_o} = b (P_m - x)^c \qquad (6)$$
$$\hat{P_o} = (a + b P_m)\left(1 - e^{-(P_m - x)/\tau}\right) \qquad (7)$$

Equation (3) is simple scaling, one free parameter. Equation (7) has four. All
are fitted by minimising the residual sum of squares, over only the part of the
CDF corresponding to observed wet days.

**Nonparametric** (Sect. 2.3) assumes no form. QUANT stores the empirical CDFs as
"tables of empirical percentiles", with linear interpolation between them; for a
new value above the training range, "the correction found for the highest quantile
of the training period is used". SSPLIN fits a cubic smoothing spline to the q–q
relation, with the smoothing parameter chosen by generalised cross-validation.

**Data and scoring.** Daily precipitation from 82 Norwegian stations, all covering
1960–2000, against HIRHAM at 25 km driven by ERA40. Overall skill is the MAE
between the observed and the corrected empirical CDF. To see *where* in the
distribution a method works, that error is also computed in ten probability bands:
$\mathrm{MAE}_{0.1}$ covers the driest tenth and so reflects how many wet days the
method gets right; $\mathrm{MAE}_{1.0}$ covers the wettest tenth and so reflects
the extremes. The total MAE is the mean of the ten, which the authors note
"illustrates the consistency of these measures".

Everything is scored out of sample, by **10-fold cross-validation** over
continuous time intervals — deliberately, because "highly adaptable methods, such
as the nonparametric techniques used in this study, are prone to over fitting the
data". Methods are then ranked by relative error: each method's score divided by
the uncorrected model's score, averaged across the eleven measures. Below one is
an improvement; above one makes things worse.

## Main results

The ranking, best to worst, by mean relative error (Fig. 4):

| Rank | Method | Family |
|---|---|---|
| 1 | QUANT | nonparametric |
| 2 | SSPLIN | nonparametric |
| 3 | $\hat{P_o} = (a + b P_m)(1 - e^{-(P_m - x)/\tau})$, Eq. (7) | parametric, 4 parameters |
| 4 | $\hat{P_o} = b(P_m - x)^c$, Eq. (6) | parametric, 3 parameters |
| 5 | $\hat{P_o} = a + b P_m$, Eq. (4) | parametric, 2 parameters |
| 6 | Bernoulli-Weibull | distribution derived |
| 7 | $\hat{P_o} = b P_m^{\,c}$, Eq. (5) | parametric, 2 parameters |
| 8 | Bernoulli-Gamma | distribution derived |
| 9 | Bernoulli-Exponential | distribution derived |
| 10 | $\hat{P_o} = b P_m$, Eq. (3) | parametric, 1 parameter |
| 11 | Bernoulli-Log-normal | distribution derived |

Two caveats on that table. The numeric relative errors appear only as symbols in
Figures 3 and 4 and are tabulated nowhere in the text, so the values themselves
cannot be quoted from the paper — only the order, read off the figure axis, and
the statements the authors make about it. And the text never separates QUANT from
SSPLIN: it names them together as the two best, so which of the two is first
rests on the axis ordering rather than on anything written. Every other position
in the table is confirmed by a sentence in Section 5.

What the authors state about that ranking:

**The two nonparametric methods win, including in the tail.** "The two
nonparametric methods SSPLINE and QUANT have on average the best skill in
reducing systematic errors, also for very high (extreme) percentiles" (Sect. 5).
They attribute this to flexibility: the methods "do not rely on any predetermined
function", which "allows good fits to any quantile–quantile relation".

**Among parametric forms, more parameters did better.** "Parametric
transformations with three or more free parameters (Eqs. 6 and 7) are almost as
efficient as their nonparametric counterparts. Transformations with less
flexibility, in particular the simple scaling function (Eq. 3), do have worse
performance" (Sect. 5). Equation (3), with one parameter, is second worst of all
eleven methods.

**The distribution derived family ranked lowest on average**, with
Bernoulli-Weibull the best of them and Bernoulli-Log-normal the worst method
overall. Two of them do active harm at the top of the distribution: the
Bernoulli-Exponential and Bernoulli-Log-normal transformations "increase the error
for the most extreme values" (Sect. 5).

**The authors explain why, and the reason is structural rather than about the
choice of distribution.** Their result "may seem somewhat surprising, given the
theoretical elegance of this approach. This is likely related to the fact that the
parameters of the distributions are identified for $P_o$ and $P_m$ separately,
which enables good approximations of the distributions of $P_o$ and $P_m$ but does
not necessarily optimise the statistical transformation as defined in Eq. (1)"
(Sect. 5). Fitting each side well is not the same as fitting the mapping between
them well.

**Spatially**, the uncorrected error is largest along the west coast, "where the
model cannot resolve the orographic effect on precipitation with sufficient
detail". Most methods reduce the error and flatten some of its spatial variation;
the largest improvements are parametric and nonparametric, again on the west
coast. Bernoulli-Log-normal "does not lead to any visible improvements".

**By probability band**, improvements are largest in the upper half of the CDF
($p \geq 0.5$). In the lower part they are smaller, "owing to the small (often
zero) precipitation rates" — there is little error to remove where the values are
zero.

## Discussion

The authors' conclusion is that the nonparametric methods should be the default:
they have the best skill "through the entire range of the distribution" and have
"the additional advantage that they can be applied without specific assumptions
about the distribution of the data", so they are "recommended for most
applications of statistical bias correction" (Sect. 7).

They pair that with a warning against applying any of it blindly. Most methods
did remove bias, but "it was also demonstrated that the performance of the methods
differ substantially. Therefore, we stress that these techniques should not be
applied without checking their suitability for the data under consideration"
(Sect. 7).

Three limitations are acknowledged.

**Overfitting, conditional on sample size.** Because all scores are
cross-validated, "this suggests that over fitting is no major problem if there are
sufficient data. Nevertheless, over fitting may be an issue if the nonparametric
transformations are calibrated using small data samples, i.e. time series that
cover only a short period" (Sect. 5). Their own record is 41 years of daily data
at each of 82 stations.

**Stationarity.** These methods assume the model-minus-observation difference
stays the same in a future climate. The authors are explicit that this "cannot be
fully assessed, as the variable of interest may exceed the observed range in a
changing climate" (Sect. 6), and that "it cannot be ruled out that the methods
perform badly if the projected climatic conditions differ substantially from the
calibration period" (Sect. 5). They offer cross-validated stability, and a finding
from Chen et al. (2011a) that the calibration period matters less than the choice
of model and scenario, as partial reassurance rather than proof.

**Side effects on properties nobody asked the method to change.** Citing Themeßl
et al. (2011), the effect on projected changes in the mean is "comparably small"
but the methods "may systematically alter changes in nonlinearly derived measures,
including characteristics of extreme events". Other reported side effects are
changes in the amplitude of low-frequency variability (Haerter et al., 2011) and
in measures of temporal persistence (Johnson and Sharma, 2011, 2012). Whether such
an effect is good, bad or irrelevant "depends on particular applications and has to
be evaluated on a case to case basis" (Sect. 6).

**Appendix A is about naming**, and worth knowing about. The authors list the terms
in circulation — "quantile mapping", "quantile matching", "CDF matching",
"quantile–quantile transformation", "histogram equalisation or matching",
"probability mapping", "distribution mapping", "statistical bias correction",
"direct error correction methods", "model output statistics (MOS)" — and note that
several are used inconsistently across the three families, "causing some ambiguity
regarding the proper nomenclature". They adopt "statistical transformation" for
this paper to "emphasise the common objective of the presented techniques without
interfering with previously used terminology".

## Relevance to this project

*Everything below is my own reading, not the paper's content.*

**This paper is the source of our method taxonomy and of our code.** Our three
families — distribution derived, parametric, non-parametric — are this paper's
classification, and `src/qm_transforms.py` implements its methods. QUANT, RQUANT
and SSPLIN reach us through the `qmap` R package released with this paper.
RQUANT is worth flagging: it is *not* in this paper, only in the later package,
so it should be cited to the package rather than here.

**Two differences in scope matter when borrowing its conclusions.** The paper
corrects **precipitation** from an **RCM driven by reanalysis**; we correct
**temperature** from a **free-running GCM**. The Bernoulli mixtures exist only
because precipitation has a mass of zeros, which is why our distribution-derived
member is a plain normal instead. More consequentially, an RCM driven by ERA40
follows the observed day-to-day sequence, while a free-running GCM does not — so
their setting has a correspondence between modelled and observed days that ours
lacks entirely.

**Their headline result does not reproduce in our setting, and their own caveat
predicts that.** They find the flexible nonparametric methods best; we find the
choice barely matters and the rigid methods slightly ahead. Their sentence about
overfitting "if the nonparametric transformations are calibrated using small data
samples, i.e. time series that cover only a short period" is the direct
explanation: they calibrate on 41 years, we calibrate on 9 or 10. This is the
single most useful sentence in the paper for us, and it should be cited wherever
our findings report that complexity did not buy accuracy.

**Their per-band scoring is the ancestor of our per-percentile evaluation.** Our
reporting of MAE at P5, P25, P50, P75 and P90 does what their
$\mathrm{MAE}_{0.1} \dots \mathrm{MAE}_{1.0}$ does — refusing to collapse the
distribution into one number. Their warning that two distribution-derived methods
make the extremes *worse* is the same class of finding as our unanchored QUANT
carrying a +1.8 °C bias at SON P5 (ADR-0004): a method can improve the average
while damaging the tail.

**Equation (7) is the one we implemented and removed.** The paper ranks it third
of eleven, which is why it looked worth trying. It is built for precipitation,
where $P_m \geq 0$; on temperature in °C the sign of $(P_m - x)$ can flip inside
the data range. Our reasons for dropping it are in ADR-0007.

**On terminology**, their Appendix A is the best short account of why this field's
vocabulary is a mess, and it is a fair precedent for our own decision in
`CONTEXT.md` to fix one term. They chose "statistical transformation"; we use
"transfer function". Worth citing if that choice is ever questioned.
