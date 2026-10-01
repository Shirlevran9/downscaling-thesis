# Long tails in regional surface temperature distributions

## Title

**Long tails in regional surface temperature probability distributions with
implications for extremes under global warming**

Tyler W. Ruff, J. David Neelin

*Geophysical Research Letters* **39**, L04704, 2012.
doi:10.1029/2011GL050610 · received 9 December 2011, published 25 February 2012.

PDF: `Ruff2012_LongTails_GRL.pdf`

## Abstract

Work on other atmospheric fields had already shown that probability
distributions of column water vapour and of several passive tropospheric
chemical tracers have tails that fall off more slowly than a Gaussian, and that
are roughly exponential. Simple models of tracer advection across a maintained
gradient explain why. Near-surface air temperature is also moved around by
advection across a large-scale temperature gradient, so the authors ask whether
daily surface temperature distributions show the same behaviour. The question
matters because the usual qualitative story about extremes under global warming
treats the distribution as Gaussian and shifts its mean.

The paper examines daily maximum, daily average and daily minimum temperature
at 29 stations with long records, in JJA and in DJF, from the NCDC Global
Surface Summary of Day archive. Each distribution is compared with a Gaussian
fitted to its core, defined as the part of the distribution above 30 % of the
maximum. The core is usually Gaussian, but "approximately half these
distributions exhibit substantial departure from Gaussian in the tails on at
least one side" (Sect. 5), and most of these are asymmetric, with one long tail
only. The authors then show, for one station, that tail shape controls how much
the probability of crossing a fixed temperature threshold changes when the
distribution is shifted. A long tail means a much smaller change than a Gaussian
tail does. They conclude that models used for regional assessments of extremes
should be checked for whether they reproduce the observed tail shape.

## Terms and notation

### Terms

The **core** of a distribution is the central part that holds most of the data.
Here it is defined operationally, as every histogram point whose value exceeds
30 % of the distribution maximum. The **tails** are what lies outside that core.
The whole method rests on this split: a Gaussian is fitted to the core only, and
the tails are then judged against that fit rather than against a Gaussian fitted
to everything.

A **long tail** is a tail that falls off more slowly than the Gaussian fitted to
the core. In this paper long tails are described as "approximately exponential".
The opposite case also appears: a **shorter-than-Gaussian tail**, which falls off
faster than the core fit predicts. The authors report those but do not analyse
them, because they have no matching prototype from tracer-advection theory.

An **asymmetric** distribution here means one with a long tail on one side only.

A **tracer-advection prototype** is a simple mathematical model in which a
passive tracer is carried by a flow across a maintained gradient. Such models
produce long, roughly exponential tails, and they produce asymmetry when the
gradient or the flow differs between the two directions. The paper uses them as
the reason to expect long tails in temperature, not as a model it fits.

An **exceedance** is a day on which the value crosses a chosen threshold: above
it on the high side, below it on the low side. The **exceedance ratio** is the
probability of exceedance after the distribution is shifted, divided by the
probability before the shift. A ratio of 30 means exceedances become thirty
times as likely.

**GSOD** is the Global Surface Summary of Day product of the National Climatic
Data Center. An **AR(1) process** is a first-order autoregressive process; here
it is a Gaussian one, used to generate synthetic series for a significance test.

### Notation

The three temperature variables and the quantities of the exceedance
calculation:

| Symbol | Meaning |
|---|---|
| $T_{\max}$ | Daily maximum surface air temperature |
| $T_{\mathrm{avg}}$ | Daily average surface air temperature |
| $T_{\min}$ | Daily minimum surface air temperature |
| $\sigma$ | Standard deviation of the Gaussian fitted to the core |
| $T_H$ | High-side threshold temperature, set here to $3\sigma$ above the mean |
| $T_L$ | Low-side threshold temperature, set here to $-3\sigma$ |
| $\Delta T$ | The amount by which the whole distribution is shifted |

All distributions are of daily anomalies: the daily climatology and a linear
trend have been removed first.

## Previous work

The paper builds on two separate lines of work.

**Long tails in other fields.** Neelin et al. (2010) found non-Gaussian tails,
most of them approximately exponential, in probability distributions of passive
tropospheric chemical tracers and water vapour. Those tails are "consistent with
simple mathematical prototypes for passive tracer advection problems with a
forcing that maintains a gradient" (Sect. 1), citing Bourlioux and Majda (2002),
Pierrehumbert (2000), Ngan and Pierrehumbert (2000) and Majda and Gershgorin
(2010). Neelin et al. (2010) also discussed asymmetry in the observed tracers.
The present paper takes this as the reason to look for the same shape in
temperature, while noting that "air temperature is not a passive tracer and
potentially has complications due to effects of soil moisture ... and clouds"
(Sect. 1). Other routes to long tails in geophysical data are noted, including
multiplicative noise such as stochastic damping (Sura and Sardeshmukh, 2008;
Sura and Perron, 2010).

**Observed and projected changes in temperature extremes.** Alexander et al.
(2006) and Caesar et al. (2006) are cited for evidence that hot extremes have
become more frequent and larger and cold extremes weaker, with the caveat that
regions and variables differ (Easterling et al., 2000; Walsh et al., 2001).
Diffenbaugh et al. (2005) is cited for the importance of fine-scale processes,
which is the paper's reason to work at local to regional scale.

The gap the authors identify is in the standard explanation of how extremes
change. The "commonly stated qualitative explanation" (Sect. 1), attributed to
Meehl et al. (2000, 2007), Trenberth et al. (2007) and Walker and Diffenbaugh
(2008), is that a shift in the mean raises the frequency of events above a
threshold. The authors do not reject that picture but argue it is incomplete:
"the presence of non-Gaussian tails implies that a change in such distributions
can potentially have more complex behavior than that of a pure Gaussian, which
can be characterized simply by the mean and standard deviation" (Sect. 1).

One alternative approach is named and set aside: Kharin and Zwiers (2005) fit
distributions of extreme occurrences that apply asymptotically for large
samples. The authors instead work directly from the observed distributions.
Simolo et al. (2010) is cited as evidence that a simple shift is useful for
describing observed changes.

## Problem definition

### Problem

Two questions, in order. First, do daily surface temperature distributions have
tails that depart from a Gaussian fit to their core, and does this depend on
region and on which temperature variable is used? Second, if they do, how much
does that change the projected frequency of threshold crossings under warming?

The second question is answered only under one assumption, stated plainly: the
warming shifts the distribution and changes nothing else. For advection-driven
tails, this "corresponds to assuming that flow statistics and temperature
gradient remain constant while the large-scale background temperature
increases" (Sect. 4). The authors do not claim this is what happens. They argue
it is the simplest case, and that "if tail characteristics are consequential in
this simple case, they appear likely to be at least as important in more complex
cases" (Sect. 4).

### Model

The paper compares no competing methods or models. It applies one diagnostic
procedure, and it numbers no equations.

**Removing the trend and the cycle.** Each station series is linearly detrended,
"to remove potential biases owing to multi-decadal warming or cooling". The
seasonal cycle is removed by subtracting the daily climatology, computed as the
mean of each calendar day over all years. Leap days are removed "for
simplicity". The authors state that the climatology so computed "proved
sufficiently smooth due to the long time series considered here" (Sect. 2).

**Binning.** Distributions are histograms with a bin width of $5/9$ °C, chosen
"to prevent artifacts from the conversion from Fahrenheit to Celsius" (Sect. 2).

**Fitting the core.** A Gaussian is fitted by polynomial regression to all points
exceeding 30 % of the distribution maximum. The authors explain why they do not
use the standard deviation of the whole distribution: "this would be affected by
the presence of non-Gaussian tails" (Sect. 2). They also justify the 30 %
threshold by its effect on the fitted width. At that threshold, the core
$\sigma$ is typically 5–15 % smaller than the $\sigma$ of the whole
distribution, with a few strongly tailed stations as exceptions. "Lower
thresholds decrease this value by a few percent but the fit becomes overly
skewed by tails, while higher thresholds rapidly increase the $\sigma$
difference and tends to inflate the appearance of tails" (Sect. 2). This is a
reasoned choice, not a tested optimum.

**Testing significance.** An error envelope is built by simulating from a
Gaussian AR(1) process matched to the core: it is given the standard deviation
and the 1-day autocorrelation time of that station and season. The authors state
that this process "fits the observed autocorrelation at lags to at least several
days to a week" (Sect. 2). For each distribution, 1,000 artificial series of the
same length as the station record are generated, each is binned the same way,
and the envelope runs from the 5th to the 95th percentile of the spread in each
bin. A tail is called non-Gaussian when it lies outside that envelope.

**The exceedance calculation.** For one station the observed distribution is
shifted by $\Delta T$ and the probability of crossing a fixed threshold is
recomputed. The high-side probability is integrated from $T_H$ up to the highest
non-empty bin; the low-side probability from the lowest non-empty bin down to
$T_L$. The result is expressed as the ratio to the unshifted probability. Three
versions of the curve are compared: the observed tail, an exponential fit to it,
and the Gaussian core fit continued outwards over the same integration interval.

### Data

GSOD version 7 from the NCDC, which holds 18 daily surface variables. Its inputs
are synoptic and hourly observations from U.S. Air Force DATSAV3 Surface data
and Federal Climate Complex Integrated Surface Data (Lott et al., 2008). Most
stations used are inside the United States, chosen because their records are
long, "typically spanning from around 1950 through 2009". Stations were selected
to have few missing values and no other major quality problems. As a check, the
authors compared GSOD with GHCN version 1 daily data at several stations and
report "no significant differences".

The summary statistics in Section 5 cover 29 stations, each with three variables
($T_{\max}$, $T_{\mathrm{avg}}$, $T_{\min}$) and two seasons (JJA and DJF).

Eight stations are shown in the figures. For JJA: LAX Airport and Long Beach in
coastal California, Phoenix in a subtropical arid regime, and Houston in a
subtropical humid regime. For DJF: Grand Junction, Seattle, Chicago and Prague.
San Juan, Puerto Rico is used for the exceedance calculation, chosen because it
has long tails on both sides, a clearly defined core, and tails that are close
to exponential.

### Evaluation metrics

There is no skill score, because no methods are being compared. Two things are
measured.

A tail is judged **non-Gaussian** if it falls outside the 5th-to-95th percentile
AR(1) envelope described above. The station-level results of this test are
reported as counts and fractions of stations and variables.

The effect on extremes is measured by the **exceedance ratio**: the probability
of crossing $T_H$ (or falling below $T_L$) after a shift of $\Delta T$, divided
by the probability before the shift. It is reported as a curve against
$\Delta T$, for the observed tail, its exponential fit and the continued
Gaussian.

## Main results

**Non-Gaussian tails are common.** Over the 29 stations and three variables,
"approximately half these distributions exhibit substantial departure from
Gaussian in the tails on at least one side" (Sect. 5). The paper gives these
summary figures:

| Summary statistic | Value |
|---|---|
| Distributions with substantial departure from Gaussian in at least one tail | about half |
| Difference between summer and winter in that fraction | "similar in summer and winter" |
| Stations with at least one non-Gaussian variable in each of JJA and DJF | three quarters |
| Stations Gaussian within the error bars for all three variables in both seasons | 7 % |

The per-station detail is in Table S1 and Figure S1 of the auxiliary material,
which is not part of the PDF, so the individual station results cannot be quoted
here.

**Most departures are one-sided.** "Asymmetric distributions are very common
among those that depart significantly from Gaussian, with the vast majority
having a long tail only on one side (high side or low side depending on
location)" (Sect. 5). In the stations shown, where one tail is long, the other
either stays close to Gaussian or falls off faster than the core fit.

**The core is usually Gaussian, but not always.** For most locations the clear
departures are confined to the tails. A few stations depart even within the core;
Chicago $T_{\min}$ is named as an example, and a "substantial asymmetry" in the
cores of Chicago and Prague is reported. The authors note that a skewed
non-Gaussian core is found at many other stations too, "with the majority of such
occurring in DJF and not JJA" (Sect. 3).

**Shorter-than-Gaussian tails also occur**, in a number of strongly asymmetric
cases, named for some variables at Long Beach, Seattle and Chicago. The authors
set these aside because there is no clear tracer-advection prototype for them,
and they note the consequences would be the opposite of the long-tailed case.

**Tail behaviour differs between the three variables at the same station.** At
LAX and Long Beach, two coastal Mediterranean stations within 20 km of each
other, $T_{\max}$ and $T_{\mathrm{avg}}$ both have long positive tails in summer
while $T_{\min}$ does not. The two stations agree with each other closely, which
the authors use to argue for consistency within a climate regime. Their proposed
explanation is physical and is offered as a suggestion, not tested: the warm tail
comes from occasional advection of hot air from inland, while the minimum
temperature is set by the sea breeze and ocean air temperature. The same
high-side tails in $T_{\max}$ and $T_{\mathrm{avg}}$ are present at these
stations in DJF, but that figure is not shown. At Grand Junction,
$T_{\mathrm{avg}}$ follows $T_{\min}$ in the low-side tail; at Prague,
$T_{\mathrm{avg}}$ has a longer tail relative to its core than either
$T_{\min}$ or $T_{\max}$.

**Phoenix and Houston behave differently from the coastal stations.** Both sit
near a local maximum of climatological temperature with little gradient, so, the
authors reason, there is no neighbouring region from which a much warmer air mass
could be advected. Their high-side tails are Gaussian or shorter. Both also show
cold-side tails in all three variables in summer. This is presented as a
consistent story, not as a tested mechanism.

**Tail shape strongly changes the exceedance ratio.** At San Juan, with
$T_H = 3\sigma \approx 2.5$ °C and $\Delta T = 1$ °C (about $0.8\sigma$ there):

| Case | Ratio of exceedance probability after the shift to before it |
|---|---|
| High side, Gaussian core fit continued, $\Delta T = 1$ °C | about 30 |
| High side, observed tail, $\Delta T = 1$ °C | "less than a quarter of this amount" |
| High side, Gaussian core fit continued, $\Delta T = 1.5$ °C | about 100 |
| High side, observed tail, $\Delta T = 1.5$ °C | about 10 |
| Low side, Gaussian core fit continued, $\Delta T = 1$ °C | 0.85 % |
| Low side, observed tail (exponential), $\Delta T = 1$ °C | 30 % |

In words: for a warming of 1 °C, a Gaussian tail makes hot threshold crossings
about thirty times more likely, while the real long tail at this station makes
them under about seven times more likely — the paper states only the "less than
a quarter" bound for that case, not a number. The gap widens with more warming.
On the cold side the effect runs the other way: with a long tail, 30 % of the
cold threshold crossings survive a 1 °C warming, against 0.85 % if the tail were
Gaussian. The observed tail and its exponential fit lie close together in both
cases.

One point of the figure description is unclear. Paragraph [14] refers to the
increased high-side exceedance being shown in Figure 2b, while the Figure 2
caption assigns the low-side ratio curve to panel b and the high-side curves to
panels c and d; paragraph [15] then reads the high-side curves off Figure 2c.
The numbers above come from the text, not from reading the plots.

## Discussion

Non-Gaussian tails are common in observed daily surface temperature, across
climate zones, and the great majority of them are one-sided.

The tails "appear qualitatively consistent with tracer advection prototypes", but
the differences between $T_{\max}$, $T_{\min}$ and $T_{\mathrm{avg}}$ and between
regions "suggest the step from qualitative understanding to quantitative
simulation may be significant" (Sect. 5).

Under a pure shift of the mean, a region with a Gaussian high-side tail faces a
much larger relative increase in hot threshold crossings than a region with a
long high-side tail. The authors add a consequence they do not test: a long-tailed
region "is more likely to have experienced such extremes in the past and thus to
have infrastructure which is adapted to such occurrences" (Sect. 5).

Only the shift case is examined. The authors note that tail properties may
themselves change, citing Majda and Gershgorin (2012) for prototypes and
Diffenbaugh et al. (2007) for regional climate models, and they treat the shift
as a lower bound on the importance of tail shape rather than as a realistic
projection.

Shorter-than-Gaussian tails are left unexplained, and the authors say the
mechanism "is important to substantiate in future work" (Sect. 5). They note
those cases would work in the opposite direction, "in some cases enhancing risk
of strong changes in threshold exceedances".

The main recommendation is a validation requirement. It is "essential to verify
whether high-resolution models accurately reproduce observed tail characteristics
for any region for which an assessment of extreme events is being conducted",
because a model that "erroneously produces a Gaussian rather than a long tail
under current climate for a particular region, will likely have serious errors in
quantitatively predicting the increase in exceedances under future climate"
(Sect. 5).

## Relevance to this project

*Everything below is my own reading, not the paper's content.* In this section
the paper's "surface temperature distribution at a station" maps onto our
distribution window at a land pixel, and its "tail" onto the extreme percentiles
at which we score the correction.

- **This is the citation for why our transfer functions should not assume
  normality.** The paper is direct observational evidence that daily temperature
  distributions are often non-Gaussian in the tails, with about half of its cases
  departing on at least one side. Cite it wherever we justify choosing empirical
  or spline transfer functions over a distribution-derived one built on a normal.
- **It supports fitting per season, and it does so with evidence.** The paper
  treats JJA and DJF separately and reports that the fraction of skewed cores is
  higher in DJF than JJA. That is a measured seasonal difference in distribution
  shape, not an assertion, so it is usable support for one transfer function per
  (land pixel, season). Note the limit: the paper never tests whether pooling
  seasons would be worse. For a direct test, Reiter et al. (2018) is still the
  reference.
- **It changes how we should report our percentile scores.** Their core-versus-tail
  split is the same idea as our split between overall error and P5/P95 error. Their
  result implies that a method scoring well on the bulk can still be wrong in the
  tail, which is exactly what our per-percentile evaluation is for. Use this to
  argue that mean absolute error alone is not enough to accept a transfer function.
- **Their warming result raises the stakes for our projection runs, not our
  baseline.** They show that under a shifted distribution, tail shape controls how
  much the frequency of threshold crossings changes — a factor of 30 against under
  7 for 1 °C at their example station. When we extend to 2081–2100, a transfer
  function that flattens or clips the tail will bias projected extreme frequency
  far more than its baseline error suggests. This is a reason to look hard at what
  each transfer function does above the highest fitted node.
- **Their conclusion (ii) gives us a diagnostic we do not currently run.** They
  argue models must be checked for tail shape in the region of interest. We could
  apply their exact procedure — fit a Gaussian to the core above 30 % of the
  maximum, build an AR(1) envelope, ask whether the tail escapes it — to the
  predictor and to the target at each land pixel, before correction. That would
  tell us whether the CMIP6 tail shape is wrong as well as whether its mean is.
- **One scope difference to state whenever we cite it.** Their data are point
  station observations; our target is ERA5-Land at 0.1°, which is a model-based
  reanalysis field, and our predictor is an interpolated coarse field. Spatial and
  model smoothing can shorten tails. The paper tests nothing gridded, so we cannot
  claim its frequencies carry over to our pixels.
- **Do not claim their coastal Mediterranean result covers our domain.** LAX and
  Long Beach are a Mediterranean climate regime with long warm-side tails in
  $T_{\max}$ and $T_{\mathrm{avg}}$, which is suggestive for our coastline, but the
  paper has no station in the Eastern Mediterranean. Cite it as a regime analogue
  at most.
- **Their preprocessing differs from ours, which limits a direct comparison.**
  They work on detrended anomalies with the daily climatology removed, so their
  tails are tails of within-season residuals. We fit transfer functions on raw
  seasonal values. Any tail diagnostic we borrow from them has to be run on
  anomalies, or the seasonal cycle will be read as a fat tail.

## Note on this summary

Two departures. The paper compares no competing methods and numbers no
equations, so the `### Model` subsection describes its single diagnostic
procedure instead of a set of models with equations. The per-station results
live in auxiliary material that is not in the PDF, so *Main results* reports the
aggregate figures the paper states in its text and says where the detail is
missing.
