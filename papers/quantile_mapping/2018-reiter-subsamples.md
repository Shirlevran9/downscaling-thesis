# Does quantile mapping to subsamples improve the bias correction?

## Title

**Does applying quantile mapping to subsamples improve the bias correction of
daily precipitation?**

Philipp Reiter, Oliver Gutjahr, Lukas Schefczyk, Günther Heinemann, Markus
Casper

*International Journal of Climatology* **38**, 1623–1633, 2018. Published
online 10 September 2017.
doi:10.1002/joc.5283 · received 23 December 2016, accepted 10 August 2017.

PDF: `Reiter2018_QM_Subsamples_IJC.pdf`

## Abstract

Quantile mapping (QM) corrects the whole distribution of a modelled variable,
but it "does not correct for errors in the annual cycle" (Abstract). The common
fix is to fit QM separately on temporal subsamples, for example on each calendar
month. That fix has a cost: each subsample holds fewer days, so each transfer
function is fitted on less data. The paper asks whether the smaller calibration
sample cancels the gain. It is, in this library, the paper that **tests**
sub-annual stratification instead of assuming it.

Four QM methods were applied to 40 years of daily precipitation from 10 regional
climate model (RCM) hindcasts, with no subsampling and with three subsampling
timescales (semi-annual, seasonal, monthly), scored by cross-validation. The
answer has two parts. On the calibration data, finer is always better: the
monthly timescale was optimal for all four methods. On independent validation
data the answer depends on the method: the flexible methods lose accuracy when
the subsamples get small, and "the optimal subsampling timescale for the
correction of independent data depended on the chosen QM method and ranged
between semi-annual and monthly" (Abstract). Subsampling at some sub-annual
timescale always beat no subsampling when performance was judged sub-annually.

## Terms and notation

### Terms

**Bias correction (BC)** is the post-processing of model output so that its
statistics match observations. **Quantile mapping (QM)** is the bias-correction
family that matches distributions. The fitted mapping is called the **transfer
function (TF)**: the paper defines it as the right-hand side of Equation (1),
which "transfers the uncorrected $x$ to corrected values $y$" (Sect. 3.1).

**Subsampling** is the paper's name for splitting the calibration record into
temporal groups and fitting a separate TF for each group. The paper uses
**discrete** subsamples, not moving windows; it notes that moving windows are
the alternative it did not test (Sect. 3.2). Four **subsampling timescales** are
compared:

| Timescale | Definition | Number of subsamples |
|---|---|---|
| Complete | no subsampling | 1 |
| Semi-annual | winter NDJFMA, summer MJJASO | 2 |
| Seasonal | each meteorological season | 4 |
| Monthly | each calendar month | 12 |

A separate idea, kept apart from the above, is the **assessment timescale**: the
grouping used when scoring, not when fitting. The scores were computed either on
the whole record ("annual") or on semi-annually, seasonally or monthly grouped
data. Fitting timescale and scoring timescale vary independently in this study.

Four QM methods are compared, of three types. **eQM** is empirical
(nonparametric) QM, after Boé et al. (2007). **gQM** is parametric, based on the
gamma distribution, after Piani et al. (2010a). **GQM** is parametric with a
mixed distribution, after Gutjahr and Heinemann (2013): a gamma distribution for
the lower 95 % and a General Pareto distribution for the upper 5 %. **PTF**, the
"Piani-Transfer-Function", is semi-parametric and fits Equation (2) directly.
The paper groups them by **complexity**, meaning the number of free parameters,
and **flexibility**, meaning how far a method can adapt to the calibration data:
"The more complex methods eQM and GQM allow a better adaptation to the observed
CDF than the less complex methods gQM and PTF" (Sect. 3.1).

A **dry-day correction** is needed before fitting the parametric and
semi-parametric methods, because RCMs "typically simulate too few dry days due
to the 'drizzle effect'" (Sect. 3.1).

### Notation

The transformation and its parameters:

| Symbol | Meaning |
|---|---|
| $x$ | Uncorrected RCM daily precipitation |
| $y$ | Corrected value |
| $F_{\mathrm{RCM}}$ | CDF of the RCM data |
| $F_{\mathrm{obs}}$ | CDF of the observations |
| $F_{\mathrm{obs}}^{-1}$ | Inverse CDF (quantile function) of the observations |
| $a$ | Additive correction factor in Equation (2) |
| $b$ | Multiplicative correction factor in Equation (2) |
| $x_0$ | Dry-day correction factor, $x_0 = -a/b$; modelled values below it are set to zero |
| $\tau$ | Rate at which the asymptote in Equation (2) is reached |

The three skill scores:

| Symbol | Meaning | Range |
|---|---|---|
| $\mathrm{MAE}$ | Mean absolute error between two CDFs | 0 (best) to $\infty$ (worst) |
| $\mathrm{PSS}$ | Perkins skill score, comparing probability distribution functions with a bin width of 1 mm | 0 (worst) to 1 (best) |
| $\mathrm{Ext}_{95}$ | Mean pairwise difference of the values above the 95th percentile of two distributions | 0 (best) to $\infty$ (worst) |

In the figures, each fitted variant carries one of three labels: **b** is the
best bias-correction performance within that QM method, **w** is statistically
worse than that best, and **−** is no significant difference.

## Previous work

The paper starts from an accepted position: RCM output carries systematic
biases, so bias correction is "a standard procedure in climate change impact
studies" (Ehret et al., 2012, quoted in Sect. 1). QM is named as both well
performing (Themeßl et al., 2011; Gudmundsson et al., 2012; Teutschbein and
Seibert, 2012; Räty et al., 2014; Fang et al., 2015; Teng et al., 2015) and
widely used.

The gap the authors identify is specific, and it is about practice rather than
theory. QM "is often calibrated with the entire precipitation time series as a
whole" (Sect. 1), with seven such studies cited. At the same time a second group
of studies (Maraun et al., 2010; Piani et al., 2010b; Yang et al., 2010; Themeßl
et al., 2011; Berg et al., 2012; Wilcke et al., 2013; Ruffault et al., 2014;
Gennaretti et al., 2015; Teng et al., 2015) works from the idea that the annual
cycle "is, however, supposed to improve when QM is calibrated with subsamples of
the time series" (Sect. 1). The verb "supposed" is the authors' own: they treat
this as an expectation in the literature, not as a tested result.

The resulting problem is that no one agrees on which subsampling timescale to
use. "Yet, there is no commonly agreed-on, preferable subsampling timescale.
This is the reason why many different subsampling timescales are used in the
literature" (Sect. 1). Table 1 of the paper lists the practice: semi-annual
(1 study), seasonal (5), monthly (13, "probably the most common"), monthly from
a 3-month or 2-month moving window, and moving windows of 31, 41, 61 and 91
days. Räty et al. (2014) and Rajczak et al. (2016) had recommended a seasonal
timescale.

The counterweight is also cited from prior work: several studies had already
found that "small calibration sample sizes were found to result in a less robust
QM calibration because the sampling uncertainty increases" (Boé et al., 2007;
Berg et al., 2012; Räisänen and Räty, 2013; Räty et al., 2014; Rajczak et al.,
2016; Reiter et al., 2016). A second source of sampling uncertainty, the choice
of which years form the calibration set, is credited to Li et al. (2010) and
Lafon et al. (2013).

## Problem definition

### Problem

Find a subsampling timescale that corrects the distribution of daily
precipitation on sub-annual timescales while still leaving enough data for a
robust parameter estimate. The two effects pull in opposite directions, and the
paper's whole purpose is to find where the balance lies. It also asks whether
that balance is the same for every QM method.

### Models

QM matches the CDF of the RCM data to the CDF of the observations. Equation (1):

$$y = F_{\mathrm{obs}}^{-1}\left(F_{\mathrm{RCM}}(x)\right) \qquad (1)$$

The four methods differ only in how $F_{\mathrm{RCM}}$ and $F_{\mathrm{obs}}$
are defined.

- **eQM** — nonparametric, empirical CDFs, after Boé et al. (2007).
- **gQM** — parametric, gamma distribution, after Piani et al. (2010a).
- **GQM** — parametric mixed distribution, after Gutjahr and Heinemann (2013):
  gamma for the lower 95 % of the distribution, General Pareto for the upper
  5 %.
- **PTF** — semi-parametric, approximating the transfer function directly with
  "an exponential tendency towards an asymptote", Equation (2):

$$y = (a + bx)\left(1 - e^{-(x - x_0)/\tau}\right) \qquad (2)$$

The paper gives no further equations. It refers the reader to Reiter et al.
(2016) for a fuller description of the four methods.

### Data

Daily precipitation from 10 RCMs of the EU ENSEMBLES project, for 1961–2000.
All are hindcast runs driven by the ERA-40 reanalysis, on a common grid at
0.22° (about 25 km). Observations are the E-OBS gridded data set, available on
the same grid. The area is Germany and its bordering areas.

Each TF was fitted separately for every RCM grid box.

### Evaluation metrics

Three skill scores were used: $\mathrm{MAE}$ between two CDFs (from Gudmundsson
et al., 2012), the Perkins skill score $\mathrm{PSS}$ with a 1 mm bin width
(Perkins et al., 2007), and $\mathrm{Ext}_{95}$ (Reiter et al., 2016) for the
extremes above the 95th percentile.

Validation is by cross-validation over decades. Each of the four decades in
1961–2000 was held out in turn as validation data, with the other three decades
(30 years) used for calibration. Four calibration/validation splits, four QM
methods, 10 RCMs and four subsampling timescales give 640 bias-corrected data
sets. Each TF was applied to both the calibration data and the held-out decade,
and both were scored.

Each score was applied per grid box, then averaged over the area, and the scores
were computed at all four assessment timescales. Results are shown as boxplots
of these area mean values: 10 values per box at the annual assessment
(10 RCMs), 20 at semi-annual, 40 at seasonal, 120 at monthly.

The comparison between subsampling timescales uses one-sided Wilcoxon rank-sum
tests at $\alpha = 5\%$. The decision rule is stated explicitly: "The optimal QM
subsampling timescale was defined as the coarsest timescale that did not perform
significantly worse than any finer timescale" (Sect. 3.3). This rule prefers the
coarser option when two are statistically tied.

A "joint analysis" was also run, in which the scores of all four
calibration/validation splits were first averaged and then compared. The
individual splits "do not differ substantially (not shown)" from the joint
analysis, for all three scores.

## Main results

**The numeric skill score values are not tabulated anywhere in the paper.** They
appear only as boxplots in Figures 2, A1 and A2, so no MAE, PSS or
$\mathrm{Ext}_{95}$ value can be quoted for a given scheme. What the figures do
print, as text labels above each box, is the outcome of the Wilcoxon tests. The
tables below give those labels exactly as printed: **b** = best within that QM
method, **w** = statistically worse than the best, **−** = no significant
difference. The four entries of each cell are in the order Complete /
Semi-annual / Seasonal / Monthly.

**MAE, joint analysis (Figure 2), calibration data:**

| Assessment timescale | eQM | GQM | gQM | PTF |
|---|---|---|---|---|
| Annual | b − − − | − b − w | − b − − | w w w b |
| Semi-annual | w b w − | w b w − | w b w − | w b w − |
| Seasonal | w w b − | w w b − | w w b − | w w b − |
| Monthly | w w w b | w w w b | w w w b | w w w b |

**MAE, joint analysis (Figure 2), independent validation data:**

| Assessment timescale | eQM | GQM | gQM | PTF |
|---|---|---|---|---|
| Annual | b − − w | b − w w | b − − − | b − − − |
| Semi-annual | w b − − | w b w w | w b − − | w b − − |
| Seasonal | w b − − | w b − w | w b − − | w b − − |
| Monthly | w b − − | w b − w | w b − − | w b − − |

**$\mathrm{Ext}_{95}$, joint analysis (Figure A1), calibration data:**

| Assessment timescale | eQM | GQM | gQM | PTF |
|---|---|---|---|---|
| Annual | b − − − | b − − − | − b − − | w b − − |
| Semi-annual | w b w − | w b w − | w b − − | w b w − |
| Seasonal | w w b − | w w b − | w w b − | w w b − |
| Monthly | w w w b | w w w b | w w w b | w w w b |

**$\mathrm{Ext}_{95}$, joint analysis (Figure A1), independent validation data:**

| Assessment timescale | eQM | GQM | gQM | PTF |
|---|---|---|---|---|
| Annual | b − − w | b − − w | b − − − | b − − − |
| Semi-annual | w b − w | w b − w | w b − − | w b − − |
| Seasonal | w b − w | w b − w | − b − − | w b − − |
| Monthly | w b − w | b − − w | b − − − | w b − − |

**PSS, joint analysis (Figure A2), calibration data:**

| Assessment timescale | eQM | GQM | gQM | PTF |
|---|---|---|---|---|
| Annual | b − − − | b − − − | b − − − | b − − − |
| Semi-annual | w b w − | w b − − | w b − − | w b − − |
| Seasonal | w w b − | w w b − | w w b − | w w b − |
| Monthly | w w w b | w w w b | w w w b | w w w b |

**PSS, joint analysis (Figure A2), independent validation data:**

| Assessment timescale | eQM | GQM | gQM | PTF |
|---|---|---|---|---|
| Annual | b − − − | b − − − | b − − − | b − − − |
| Semi-annual | w b − − | w b − − | w b − − | w b − − |
| Seasonal | w b − − | w b − − | w b − − | w b − − |
| Monthly | w b − w | w b − − | w b − − | w b − − |

The paper's own recommended subsampling timescale for independent data, stated
in the Discussion and Conclusions:

| QM method | Complexity | Recommended subsampling timescale |
|---|---|---|
| eQM | more complex | Seasonal |
| GQM | more complex | Semi-annual |
| gQM | less complex | Monthly |
| PTF | less complex | Monthly |

For the calibration data the answer is the same for all four: monthly.

What the authors state in words:

**Bias correction itself always helped.** "Primarily, the application of a BC
leads to a significant improvement of the RCM data for all combinations of
calibration data, QM method, and subsampling timescale" (Sect. 4), for every
skill score, and more for calibration data than for validation data.

**On the calibration data, the fitting timescale must match the assessment
timescale.** "The correction of sub-annual cycles benefits from the use of
subsamples in case of all QM methods, as the optimal QM subsampling timescale is
always the same as the timescale of the skill score assessment. The QM
variations without subsampling ('complete') failed to correct the sub-annual
cycles" (Sect. 4). This is visible in the tables above as the diagonal of **b**
labels.

**On the annual assessment timescale, subsampling brings no consistent gain.**
"For the calibration data, there is no consistent benefit from subsampling at
the annual assessment timescale" (Sect. 4). In other words, the case for
subsampling rests entirely on sub-annual performance.

**On the independent data, the diagonal disappears and semi-annual wins almost
everywhere.** "There is no such clear relationship as for the calibration data.
Instead, for all QM methods the semi-annual subsampling performed best for all
sub-annual skill score assessments. Finer QM subsampling timescales did not
improve the results any further; in case of eQM and GQM the results even
worsened (substantially for the monthly GQM)" (Sect. 4).

**Subsampling still beats no subsampling out of sample.** "The QM constructed
without subsampling ('complete') always performed worse than the semi-annual QM
if the skill score was assessed at sub-annual timescales" (Sect. 4). This is the
direct test of sub-annual stratification, and it is passed — but the margin is
between no subsampling and the *coarsest* subsampling, not the finest.

**One exception in the extremes.** For $\mathrm{Ext}_{95}$, "for GQM as well as
gQM the variation without subsampling instead of the semi-annual variation
performed best for the correction of the validation data in case of the monthly
assessment timescale" (Sect. 4).

**PSS separates the schemes least.** The PSS findings agree with the other two,
"although, the differences in the skill score values are lower than for the
other two skill scores, especially for the independent validation data"
(Sect. 4).

**Choice of calibration years did not matter.** "Sampling uncertainties due to
the selection of the calibration data were not found to affect the results"
(Sect. 5). The four splits gave very similar patterns.

**The annual cycle of the bias.** Figure 3 shows monthly means of
$\mathrm{Ext}_{95}$. "A clear annual cycle of the bias is present in all data
sets, with larger biases in the summer months." For calibration data, "the finer
the QM subsampling timescale the better was the correction of the annual cycle".
For validation data the opposite happened with the flexible methods: "for eQM
and especially GQM, the bias in the independent validation data even increased
for the finer QM subsampling timescales, in particular in the summer months"
(Sect. 4). The monthly means are plotted, not tabulated.

**A worked example of the failure mode.** Figure 4 shows one grid cell of the
RCA model for July, calibrated on 1961–1990 and validated on 1991–2000. In the
calibration data the RCM underestimated the extremes: they "needed to be
corrected from values of up to 40 mm day⁻¹ to values of more than 80 mm day⁻¹"
(Sect. 5). eQM and GQM were flexible enough to do this; gQM and PTF were not. In
the validation decade the highest raw RCM value is 45 mm day⁻¹, and the TFs
pushed it above 80 mm day⁻¹ while "the observed maximum in the validation data
is about 60 mm day⁻¹". For monthly GQM, the 45 mm day⁻¹ value was corrected to
**226.5 mm day⁻¹**. "This issue is less severe if coarser subsampling timescales
are used. The less complex methods do not show this issue at all" (Sect. 5).
These are the only numeric values the paper gives in its text.

**Method ranking, separate from the subsampling question.** "The QM method eQM
clearly outperformed the (semi-) parametric alternatives for the calibration
data (which is owed to the high degrees of freedom of the empirical distribution
function) but not for the independent validation data" (Sect. 4).

## Discussion

Sub-annual subsampling gives a clear benefit, and the authors demonstrate it
rather than assume it. The benefit is larger for the calibration data than for
independent data.

For the calibration data the rule is simple: finer is better, which "could be
expected to some extent, since the adaptation to sub-annual cycles in the data
increases with finer subsamples" (Sect. 5).

For independent data there is no further gain below the semi-annual timescale.
The authors attribute this to the smaller calibration sample and the weaker
parameter estimates that follow, and they interpret it as overfitting whose
severity depends on the method's complexity: "the larger the number of free
parameters the better the ability to adapt to the calibration data… In contrast,
the more complex a method is the more prone it is to over-fitting" (Sect. 5).
This interpretation is an explanation the authors offer, not something the study
tests directly; the phrase "a possible reason" is theirs.

The remaining biases in the validation data were larger for the less robust eQM
and GQM than for gQM and PTF.

The authors state that the outcome is hard to turn into advice: "it remains
difficult to draw satisfying conclusions. Even more difficult is to recommend a
specific subsampling strategy" (Sect. 5).

Their recommendation, once given, is per method: monthly for gQM and PTF,
seasonal for eQM, semi-annual for GQM, for the correction of independent data.
"As long as only the calibration data is to be corrected, the monthly QM is to
prefer for all methods" (Sect. 5).

Acknowledged limits and untested choices:

- **Discrete subsamples only.** Moving windows were named as the alternative
  approach and were not tested (Sect. 3.2), even though Table 1 shows that
  window widths of 31 to 91 days are common in the literature.
- **One variable, one region, one driver.** Daily precipitation over Germany,
  from RCMs driven by reanalysis. No other variable is examined.
- **Fixed calibration length.** Always 30 years. The effect of overall record
  length is not varied here; it is referred to Reiter et al. (2016).
- **The extremes above the 95th percentile** are the part of the distribution
  where the flexible methods fail out of sample, and the failure gets worse as
  subsamples get smaller.
- **Larger biases in summer** are attributed to "unresolved convective
  precipitation in RCMs", consistent with Räty et al. (2014).

## Relevance to this project

*Everything below is my own reading, not the paper's content.* The paper's
"transfer function (TF)" is our **transfer function**; its "subsampling
timescale" is our **distribution window**; its "grid box" is our **pixel**; its
"calibration"/"validation" split is our **fold** structure.

- **This is the citation for our seasonal window, and the only one that tests
  it.** Other papers in the library assert that sub-annual stratification helps.
  Reiter et al. test it, out of sample, and find it holds: the unstratified fit
  is always worse than the coarsest sub-annual fit whenever performance is
  judged sub-annually. Cite this wherever ADR-0001 or a findings document
  justifies one transfer function per (pixel, season).
- **But cite it precisely: the test supports *some* stratification, not *fine*
  stratification.** Out of sample, semi-annual beat seasonal and monthly for
  every method under MAE. Our DJF/MAM/JJA/SON choice sits one step finer than
  their best out-of-sample timescale. Do not claim this paper endorses seasonal
  windows in general — for the empirical method eQM, which is closest to our
  QUANT, seasonal is exactly what they recommend, and that is the claim to make.
- **Their eQM recommendation is the one that transfers to us.** eQM is empirical
  quantile mapping after Boé et al. (2007), the same lineage as our QUANT. Their
  recommended timescale for it on independent data is seasonal. That is direct
  support for the current baseline and worth stating in one line in the ADR.
- **Their failure mode is our anchoring problem, in a different variable.**
  Monthly GQM turned a 45 mm day⁻¹ input into 226.5 mm day⁻¹ because the fitted
  TF extrapolated past the training range. This is the same class of failure as
  our unanchored QUANT at the tails (ADR-0004): a flexible transfer function
  fitted on few nodes behaves badly just outside the range it saw. Use it as the
  precedent when explaining why our node table is anchored.
- **The sample-size argument applies to us with more force, not less.** They
  calibrate on 30 years and still see the flexible methods degrade at the
  monthly timescale. Our baseline is 1990–1999, so a season holds about 900
  days across 10 season-years. Going finer than the season is therefore not
  worth trying on the current record; this paper is the reason to say so rather
  than to test it ourselves. Revisit only when the planned 1980–2004 training
  period is in place.
- **Their assessment-timescale design is worth copying.** They score at a
  timescale chosen independently of the fitting timescale, and that is what
  exposes the whole result — the diagonal on calibration data, its collapse out
  of sample. Our diagnostic grid scores per window; adding a scoring split that
  does not match the fitting window would show whether our seasonal transfer
  functions are actually fixing the annual cycle or only fitting it.
- **Caveat to carry with every citation: this is precipitation, not
  temperature.** Precipitation is bounded at zero, skewed, and has a dry-day
  mass that forces the drizzle correction and the Bernoulli-style machinery.
  Temperature is roughly symmetric and has a far stronger and smoother annual
  cycle. The annual-cycle argument for stratification is, if anything, stronger
  for our variable; the overfitting-in-the-tail argument may be weaker, since
  temperature extremes are not as heavy-tailed. Neither direction is tested
  here, so state both as expectations, not as findings.
- **Do not cite this for moving windows.** They tested only discrete subsamples.
  If we ever compare our season window against a 31- or 91-day moving window,
  this paper supplies the literature list (their Table 1) but no result.

## Note on this summary

*Main results* departs from the template in one way. The template asks for a
table giving the metric value for each scheme, but the paper tabulates no skill
score values at all — they appear only as boxplots. The tables therefore give
the significance labels (b / w / −) that the figures print as text, for all
three scores and both data sets, plus a separate table of the recommended
timescale per method. That is more tables than usual, but it is the only exact
record of the comparison the paper provides.
