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

## Terms and notation

### Terms

A **statistical transformation** is the authors' deliberately neutral name for
this whole family of methods. Appendix A explains the choice: the literature
calls them quantile mapping, CDF matching, probability mapping and several other
things, often inconsistently, so the paper picks a term that does not take sides.

The methods are sorted into three families by what each one assumes.
**Distribution derived** methods assume a theoretical distribution on each side
and solve for the mapping between them. **Parametric** methods assume nothing
about the distributions but fit a chosen functional form straight onto the
quantile–quantile relation. **Nonparametric** methods assume neither: they read
the mapping off the data itself.

Two nonparametric methods are tested. **QUANT** is empirical quantile mapping,
storing the relation as a table of empirical percentiles and interpolating
between them. **SSPLIN** fits a cubic smoothing spline to the same relation.

A **Bernoulli-X** mixture is how the distribution-derived family copes with
precipitation: a Bernoulli distribution for whether a day is wet at all, and
some distribution X — Gamma, Weibull, Log-normal or Exponential — for how much
fell on the wet days.

The **RCM** is a regional climate model; here HIRHAM at 25 km resolution, driven
by the ERA40 reanalysis. **Relative error** is a method's error divided by the
error of the uncorrected model output, so below one is an improvement and above
one makes things worse.

### Notation

The variables of the transformation itself:

| Symbol | Meaning |
|---|---|
| $P_o$ | Observed precipitation |
| $P_m$ | Modelled precipitation, from the RCM |
| $\hat{P_o}$ | The method's best estimate of $P_o$ |
| $h$ | The transformation: maps a modelled value to a corrected one |
| $F_m$ | Cumulative distribution function (CDF) of $P_m$ |
| $F_o^{-1}$ | Inverse CDF, or quantile function, of $P_o$ |

The free parameters of the parametric forms, Equations (3) to (7), all fitted to
data: $a$, $b$, $c$, $x$ and $\tau$.

The error scores form one family. **MAE** is the mean absolute error between the
observed and the corrected empirical CDF. The ten band scores split that same
error by where in the distribution it falls:

| Score | Covers |
|---|---|
| $\mathrm{MAE}_{0.1}$ | the driest tenth — so it reflects how many wet days the method gets right |
| $\mathrm{MAE}_{0.2} \dots \mathrm{MAE}_{0.9}$ | each successive tenth of the distribution |
| $\mathrm{MAE}_{1.0}$ | the wettest tenth — so it reflects the extremes |

The total MAE is the mean of the ten, which the authors note "illustrates the
consistency of these measures".

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

### Problem

Find a function $h$ that maps the modelled variable so that its new
distribution equals the distribution of the observed variable. Following Piani
et al. (2010b), Equation (1) states it in general form:

$$P_o = h(P_m) \qquad (1)$$

This is an application of the probability integral transform. When the
distribution is known, Equation (2) makes $h$ explicit:

$$P_o = F_o^{-1}\left(F_m(P_m)\right) \qquad (2)$$

In words: take a modelled value, ask what fraction of modelled values fall below
it, then return the observed value with that same fraction below it. The
practical difficulty, and the whole subject of the paper, is that $F_o$ and
$F_m$ are not known, so $h$ has to be approximated from data.

### Models

The three families differ in how they approximate $h$.

**Distribution derived** (Sect. 2.1) assumes a theoretical distribution on each
side and solves Equation (2). For precipitation the usual choice is Bernoulli
for whether a day is wet, with a Gamma for the intensity; the paper also tests
Bernoulli-Weibull, Bernoulli-Log-normal and Bernoulli-Exponential. Parameters
are estimated by maximum likelihood "for both $P_o$ and $P_m$ independently" — a
detail that turns out to explain the results.

**Parametric** (Sect. 2.2) fits a chosen shape straight onto the q–q relation:

$$\hat{P_o} = b P_m \qquad (3)$$
$$\hat{P_o} = a + b P_m \qquad (4)$$
$$\hat{P_o} = b P_m^{\,c} \qquad (5)$$
$$\hat{P_o} = b (P_m - x)^c \qquad (6)$$
$$\hat{P_o} = (a + b P_m)\left(1 - e^{-(P_m - x)/\tau}\right) \qquad (7)$$

Equation (3) is simple scaling, with one free parameter; Equation (7) has four.
All are fitted by minimising the residual sum of squares, over only the part of
the CDF corresponding to observed wet days.

**Nonparametric** (Sect. 2.3) assumes no form. QUANT stores the empirical CDFs
as "tables of empirical percentiles", with linear interpolation between them;
for a new value above the training range, "the correction found for the highest
quantile of the training period is used". SSPLIN fits a cubic smoothing spline
to the q–q relation, with the smoothing parameter chosen by generalised
cross-validation.

### Data

Daily precipitation from 82 stations in Norway, all covering 1960–2000, against
the HIRHAM RCM at 25 km resolution driven by the ERA40 reanalysis. The
observations are recorded to 0.1 mm day⁻¹, which sets the wet-day threshold.
The methods are released as the R package `qmap`.

### Evaluation metrics

Overall skill is the MAE between the observed and the corrected empirical CDF,
with the ten band scores showing where in the distribution a method works — see
*Terms and notation*.

Everything is scored out of sample, by **10-fold cross-validation** over
continuous time intervals. This is deliberate, because "highly adaptable
methods, such as the nonparametric techniques used in this study, are prone to
over fitting the data" (Sect. 4.1).

Methods are then ranked by relative error, each method's score divided by the
uncorrected model's score, averaged across the eleven measures.

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

The nonparametric methods are recommended as the default. They have the best
skill "through the entire range of the distribution" and "can be applied without
specific assumptions about the distribution of the data" (Sect. 7).

That comes with a warning against applying any of it blindly: "we stress that
these techniques should not be applied without checking their suitability for
the data under consideration" (Sect. 7).

Three limits are acknowledged.

- **Overfitting, conditional on sample size.** Cross-validation suggests "over
  fitting is no major problem if there are sufficient data. Nevertheless, over
  fitting may be an issue if the nonparametric transformations are calibrated
  using small data samples, i.e. time series that cover only a short period"
  (Sect. 5). Their own record is 41 years.
- **Stationarity.** The assumption that the model-minus-observation difference
  holds in a future climate "cannot be fully assessed, as the variable of
  interest may exceed the observed range in a changing climate" (Sect. 6).
- **Side effects.** The methods "may systematically alter changes in nonlinearly
  derived measures, including characteristics of extreme events" (Sect. 6),
  along with low-frequency variability and temporal persistence. Whether that
  matters "has to be evaluated on a case to case basis".

## Relevance to this project

*Everything below is my own reading, not the paper's content.*

- **This is the source of our taxonomy and our code.** Our three families are
  this paper's classification, and QUANT, RQUANT and SSPLIN reach us through the
  `qmap` package released with it. RQUANT is **not** in this paper — it is only
  in the package, so cite it to the package.
- **Two scope differences limit what we can borrow.** The paper corrects
  precipitation from an RCM driven by reanalysis; we correct temperature from a
  free-running GCM. The Bernoulli mixtures exist only because precipitation has
  a mass of zeros, which is why our distribution-derived member is a plain
  normal. More importantly, an RCM driven by ERA40 follows the observed day-to-day
  sequence and a free-running GCM does not, so their setting has a
  correspondence between modelled and observed days that ours lacks.
- **Their headline result does not reproduce for us, and their own caveat says
  why.** They find the flexible methods best; we find the choice barely matters.
  They calibrate on 41 years, we calibrate on 9 or 10. Cite their overfitting
  sentence wherever our findings report that complexity did not buy accuracy.
- **Their band scoring is the ancestor of our per-percentile evaluation**, and
  their finding that two distribution-derived methods make the extremes worse is
  the same class of result as our unanchored QUANT carrying a +1.8 °C bias at
  SON P5 (ADR-0004).
- **Equation (7) is the one we implemented and removed.** It ranks third here,
  which is why it looked worth trying; it is built for $P_m \geq 0$ and breaks on
  temperature in °C. See ADR-0007.
- **Appendix A is a precedent for fixing one term.** They chose "statistical
  transformation"; we use "transfer function" in `CONTEXT.md`.
