# Trend-preserving bias adjustment and statistical downscaling with ISIMIP3BASD

## Title

**Trend-preserving bias adjustment and statistical downscaling with ISIMIP3BASD (v1.0)**

Stefan Lange (Potsdam Institute for Climate Impact Research)

*Geoscientific Model Development* **12**, 3055–3070, 2019.
doi:10.5194/gmd-12-3055-2019 · CC BY 4.0 · received 7 February 2019, published
17 July 2019.

PDF: `Lange2019_ISIMIP3BASD_GMD.pdf`

## Abstract

Climate simulation data are biased, and they are usually coarser than the
observation data used to correct them. The paper's starting point is that these
are two different problems: "(i) the actual bias adjustment at the spatial
resolution of the simulation data and (ii) a statistical downscaling to the
spatial resolution of the observation data" (Sect. 1). Earlier ISIMIP phases
solved (ii) by simply interpolating the simulations onto the fine grid first and
then adjusting each fine cell on its own. That is cheap, but it keeps the
spatial coherence of the interpolated field and inflates temporal variability at
the original resolution (Maraun, 2013).

The paper presents the methods for phase 3 of ISIMIP. They keep the two problems
apart: first a bias adjustment at the coarse resolution against spatially
aggregated observations, then a separate stochastic downscaling step to the fine
grid. The bias adjustment is one parametric quantile-mapping framework used for
all ten climate variables, is approximately trend-preserving in every quantile,
and includes a modified version of the event-likelihood adjustment of Switanek
et al. (2017). The downscaling method, called MBCnSD, is the MBCn algorithm of
Cannon (2017) with an extra step that preserves the weighted sum inside each
coarse cell. Both are tested against their ISIMIP2b predecessors in a
cross-validation framework. The new methods adjust biases and preserve trends
better in most variables, months and grid cells, and adjust spatial variability
better "in the vast majority of cases" (Sect. 4.2).

## Terms and notation

### Terms

**Bias adjustment** is defined here as "the adjustment of statistics of climate
simulation data for the purpose of making them more similar to climate
observation data" (Sect. 1). **Statistical downscaling** is the separate task of
raising the spatial resolution of the simulation data to that of the observation
data. The paper's central claim of design is that the two should be done in that
order, at their own resolutions, not merged into one step.

The **training period** is the historical period on which the method is fitted;
the **application period** is the period the fitted method is applied to. They
may be the same period.

**Pseudo future observations** are the key device of the method. They are
historical observations to which the simulated climate change signal has been
added, quantile by quantile. The application-period simulation is then quantile-
mapped onto them instead of onto the raw historical observations. This is what
makes the method trend-preserving in all quantiles.

Four kinds of **trend preservation** are defined, one per variable class.
*Additive* transfers the change signal as a difference (used for psl, rlds, tas).
*Multiplicative* transfers it as a ratio. *Mixed* blends the two, so that a
variable with a large negative historical bias does not get an unrealistically
large signal. *Bounded* transfers the signal in a way that respects a lower and
an upper physical bound.

**Bounds and thresholds.** A bound is a physical limit of a variable. A
threshold sits slightly inside the bound and is used to adjust the frequency of
values close to the bound — for precipitation, the lower threshold of
0.1 mm d⁻¹ defines the dry-day frequency. **Randomization** replaces values
beyond a threshold with random numbers between the threshold and the bound, so
that quantile mapping can later move them across the threshold.

The **event likelihood adjustment** is inherited from the scaled distribution
mapping method of Switanek et al. (2017). Instead of mapping a value straight to
the quantile of the target distribution, it also transfers the simulated change
in the likelihood of the event, so that changes in odds are multiplicatively
preserved. It is used for every variable except tas.

**MBCn** is the multivariate quantile-mapping method of Cannon (2017): repeated
random rotations of the data, each followed by univariate quantile mappings
along the rotated axes. **MBCnSD** is this paper's modification, used for
downscaling rather than for multiple variables — the $K$ axes are the $K$ fine
cells inside one coarse cell, and an extra projection step preserves their
weighted sum.

**LI+BA** and **BA+SD** name the two orderings compared in Sect. 4.2. LI+BA is
the ISIMIP2b order: bilinear interpolation to the fine grid, then bias
adjustment there. BA+SD is the ISIMIP3 order: bias adjustment at the coarse
resolution, then statistical downscaling.

### Notation

The time series that the bias adjustment works on, all for one variable, one
grid cell, and one calendar month:

| Symbol | Meaning |
|---|---|
| $x^{\mathrm{obs}}_{\mathrm{hist}}$ | Historical observations (training) |
| $x^{\mathrm{sim}}_{\mathrm{hist}}$ | Historical simulations (training) |
| $x^{\mathrm{sim}}_{\mathrm{fut}}$ | Simulations for the application period |
| $x^{\mathrm{obs}}_{\mathrm{fut}}$ | Pseudo future observations, built in step 5 |
| $y^{\mathrm{sim}}_{\mathrm{fut}}$ | The bias-adjusted output |

The distribution functions used in Equations (1)–(14):

| Symbol | Meaning |
|---|---|
| $F^{\mathrm{obs}}_{\mathrm{hist}}, F^{\mathrm{sim}}_{\mathrm{hist}}, F^{\mathrm{sim}}_{\mathrm{fut}}$ | Empirical CDFs of the corresponding time series |
| $Q^{\mathrm{obs}}_{\mathrm{hist}}, Q^{\mathrm{sim}}_{\mathrm{hist}}, Q^{\mathrm{sim}}_{\mathrm{fut}}$ | The matching quantile functions |
| $\hat{F}^{\mathrm{obs}}_{\mathrm{hist}}, \hat{F}^{\mathrm{obs}}_{\mathrm{fut}}, \hat{F}^{\mathrm{sim}}_{\mathrm{hist}}, \hat{F}^{\mathrm{sim}}_{\mathrm{fut}}$ | CDFs of the *fitted* parametric distributions |
| $p$ | A cumulative probability |
| $a$, $b$ | Lower and upper physical bound of a variable |
| $\alpha$, $\beta$ | Lower and upper threshold, just inside the bounds |
| $\gamma(p)$ | Weight blending multiplicative and additive transfer, Eq. (7) |
| $P^{\mathrm{obs}}_{\mathrm{hist}}, P^{\mathrm{sim}}_{\mathrm{hist}}, P^{\mathrm{sim}}_{\mathrm{fut}}, P^{\mathrm{obs}}_{\mathrm{fut}}$ | Relative frequency of values beyond a threshold |
| $L^{\mathrm{obs}}_{\mathrm{hist}}, L^{\mathrm{sim}}_{\mathrm{hist}}, L^{\mathrm{sim}}_{\mathrm{fut}}$ | Logit of the fitted CDF at the corresponding value, Eqs. (12)–(14) |

The downscaling algorithm, for one coarse cell:

| Symbol | Meaning |
|---|---|
| $K$ | Number of fine cells inside one coarse cell |
| $X$ | Vector of the bias-adjusted coarse time series |
| $Y$ | Matrix of the simulated fine-grid time series, $K$ columns |
| $Z$ | Matrix of the observed fine-grid time series, $K$ columns |
| $w_{jk}$ | Area weight of fine cell $k$ in coarse cell $j$ |
| $W$ | Vector of normalised weights, $W_k = w_{jk}/\tilde{w}$ |
| $O$ | A $K\times K$ orthogonal rotation matrix |

For evaluation: $e$ is an absolute error, Equations (20) and (21); RMSD is the
root-mean-square deviation of the sixteen 0.5° time series in a 2° cell from
their spatial average, Equation (19).

## Previous work

The paper builds on its own predecessors and on three outside method papers.

**The ISIMIP2b bias adjustment** (Hempel et al., 2013; Frieler et al., 2017;
Lange, 2018) is the baseline it replaces. It used structurally different methods
for different variables. For pr, psl, rlds, sfcWind and tas it removed the bias
in the multiyear monthly mean, then adjusted day-to-day variability around that
mean with transfer functions derived per calendar month. tasmax and tasmin were
adjusted indirectly. rsds and hurs used parametric quantile mapping with beta
distributions. The author names two flaws: trend preservation applied only to
the mean for most variables, and extreme values had to be held in a plausible
range by cap values.

**Interpolate-then-adjust is the common practice and it has a known cost.** The
same order was used in the ISIMIP Fast Track, in ISIMIP2b, and for NEX-GDDP
(Thrasher et al., 2012). When the same univariate method is applied
independently in every fine cell, "the bias adjustment then retains the spatial
coherence of the interpolated simulation data and inflates temporal variability
at their original spatial resolution (Maraun, 2013)" (Sect. 1). Maraun (2013) is
also the source of the prescription the paper follows: the downscaling method
should be stochastic, "given the multivalued nature of statistical downscaling
(there are infinitely many high-resolution fields compatible with the same
low-resolution field)" (Sect. 1).

**Switanek et al. (2017)** supply two things: the event-likelihood adjustment,
which the paper modifies to work on log-odds rather than return intervals, and
the argument that parametric quantile mapping "promises a more robust adjustment
of biases in extreme quantiles than nonparametric quantile mapping" (Sect. 1).
That argument is cited, not retested here. The odd/even-year cross-validation
design also comes from them.

**Cannon (2017)** supplies MBCn. **Cannon et al. (2015)** supply the
randomization of values beyond thresholds, which this paper changes: random
numbers are drawn with a power-law probability increasing towards the bound
rather than uniformly, because that "is found to alleviate kinks in the
distribution of wet-day precipitation after bias adjustment" (Sect. 3.2.1).

**Piani et al. (2010)** are cited for the finding that adjusting tas, tasmax and
tasmin independently can produce large relative errors in the daily temperature
range and in the skewness of the daily cycle, and that adjusting tas, tasrange
and tasskew directly minimises those errors. **Vandal et al. (2018)** showed
that neural-network downscaling works better in several small steps than in one
large one; the paper checks whether the same holds for MBCnSD.

## Problem definition

### Problem

Given coarse, biased climate simulation data and finer observation data, produce
simulation data on the fine grid whose distribution per variable, grid cell and
calendar month matches the observed one, while keeping the trends the climate
model simulated. The paper adds a design constraint of its own: the two tasks
must be solved separately, bias adjustment at the coarse resolution first, then
downscaling.

The method is "independently applied to every variable, grid cell, and calendar
month" (Sect. 3.2.1). The paper describes no running window around each calendar
month: the twelve calendar months are treated as separate, discrete samples. The
only running window in the paper is unrelated — a 31-day window used to estimate
the annual cycle of upper bounds for rsds.

Assumptions that are stated rather than tested: that parametric quantile mapping
is more robust in the tails than nonparametric (attributed to Switanek et al.,
2017); that after coarse-resolution bias adjustment the data "can be considered
to have unbiased distributions of daily values per climate variable, grid cell,
and calendar month" (Sect. 3.2.2), which is what justifies a downscaling step
that preserves the coarse values.

### Models

**The bias adjustment** runs in eight steps. Steps 1–2 and 8 handle two special
variables (rsds scaling, prsnratio missing values). Steps 3 and 7 detrend and
re-trend. Step 4 randomizes values beyond thresholds. Steps 5 and 6 are the
method proper.

*Detrending (steps 3 and 7), applied to psl, rlds and tas.* Linear trend lines
are fitted to annual mean values and shifted so that $\sum_i t_i = 0$. Then
$x_{ij} \mapsto x_{ij} - t_i$ before quantile mapping, and
$y_{ij} \mapsto y_{ij} + t_i$ after it, using the trend line of
$x^{\mathrm{sim}}_{\mathrm{fut}}$. The reason given is "to prevent a confusion of
these trends with interannual variability during quantile mapping" (Sect. 3.2.1).

*Step 5, pseudo future observations.* For a value $x$ of the historical
observations with $p = F^{\mathrm{obs}}_{\mathrm{hist}}(x)$, the pseudo future
observation $y$ is built by transferring the simulated change signal at that
same cumulative probability. Additive:

$$y = x + \Delta_{\mathrm{additive}}(p) \qquad (1)$$
$$\Delta_{\mathrm{additive}}(p) = Q^{\mathrm{sim}}_{\mathrm{fut}}(p) - Q^{\mathrm{sim}}_{\mathrm{hist}}(p) \qquad (2)$$

Multiplicative:

$$y = x\,\Delta_{\mathrm{multiplicative}}(p) \qquad (3)$$
$$\Delta_{\mathrm{multiplicative}}(p) = \max\left(0.01, \min\left(100, \Delta^{*}_{\mathrm{multiplicative}}(p)\right)\right) \qquad (4)$$
$$\Delta^{*}_{\mathrm{multiplicative}}(p) = \begin{cases} 1 & \text{if } Q^{\mathrm{sim}}_{\mathrm{hist}}(p) = 0 \\ Q^{\mathrm{sim}}_{\mathrm{fut}}(p)/Q^{\mathrm{sim}}_{\mathrm{hist}}(p) & \text{otherwise} \end{cases} \qquad (5)$$

Mixed, used for pr, sfcWind and tasrange, because a purely multiplicative
transfer applied where the historical bias is strongly negative "can result in
unrealistically large $y$ values" (Sect. 3.2.1):

$$y = \gamma(p)\,x\,\Delta_{\mathrm{multiplicative}}(p) + (1-\gamma(p))\left(x + \Delta_{\mathrm{additive}}(p)\right) \qquad (6)$$

$$\gamma(p) = \begin{cases} 1 & \text{if } Q^{\mathrm{sim}}_{\mathrm{hist}}(p) \ge Q^{\mathrm{obs}}_{\mathrm{hist}}(p) \\[2pt] 0.5\left(1 + \cos\left(\left(Q^{\mathrm{obs}}_{\mathrm{hist}}(p)/Q^{\mathrm{sim}}_{\mathrm{hist}}(p) - 1\right)\pi/8\right)\right) & \text{if } Q^{\mathrm{sim}}_{\mathrm{hist}}(p) < Q^{\mathrm{obs}}_{\mathrm{hist}}(p) < 9\,Q^{\mathrm{sim}}_{\mathrm{hist}}(p) \\[2pt] 0 & \text{otherwise} \end{cases} \qquad (7)$$

Bounded, for hurs, prsnratio, scaled rsds and tasskew, Equation (8): the signal
is transferred as a ratio of the distance to the lower bound $a$ when the
simulated quantile falls, as a ratio of the distance to the upper bound $b$ when
it rises, and $y = x$ when it does not change.

*Step 6, the quantile mapping itself.* For bounded variables the frequency of
values beyond a threshold is adjusted first. The observed frequency is carried
forward with the simulated change, Equation (9):

$$P^{\mathrm{obs}}_{\mathrm{fut}} = \begin{cases} P^{\mathrm{obs}}_{\mathrm{hist}} P^{\mathrm{sim}}_{\mathrm{fut}}/P^{\mathrm{sim}}_{\mathrm{hist}} & \text{if } P^{\mathrm{sim}}_{\mathrm{hist}} > P^{\mathrm{sim}}_{\mathrm{fut}} \\ P^{\mathrm{obs}}_{\mathrm{hist}} & \text{if } P^{\mathrm{sim}}_{\mathrm{hist}} = P^{\mathrm{sim}}_{\mathrm{fut}} \\ 1 - \left(1-P^{\mathrm{obs}}_{\mathrm{hist}}\right)\left(1-P^{\mathrm{sim}}_{\mathrm{fut}}\right)/\left(1-P^{\mathrm{sim}}_{\mathrm{hist}}\right) & \text{otherwise} \end{cases} \qquad (9)$$

The $n P^{\mathrm{obs}}_{\mathrm{fut}}$ lowest values are then set to the bound.
All remaining values are mapped parametrically. Distributions are beta for
bounded variables, gamma for pr, normal for the unbounded variables psl, rlds
and tas, Weibull for sfcWind, Rice for tasrange.

The mapping is by rank. For values of equal rank in the four fitted
distributions:

$$\hat{x}^{\mathrm{sim}}_{\mathrm{fut}} \mapsto \left(\hat{F}^{\mathrm{obs}}_{\mathrm{fut}}\right)^{-1}\left(\mathrm{logit}^{-1}\left(L^{\mathrm{obs}}_{\mathrm{hist}} + \Delta_{\mathrm{log-odds}}\right)\right) \qquad (10)$$
$$\Delta_{\mathrm{log-odds}} = \max\left(-\log 10, \min\left(\log 10,\; L^{\mathrm{sim}}_{\mathrm{fut}} - L^{\mathrm{sim}}_{\mathrm{hist}}\right)\right) \qquad (11)$$

with $L^{\mathrm{obs}}_{\mathrm{hist}}$, $L^{\mathrm{sim}}_{\mathrm{hist}}$ and
$L^{\mathrm{sim}}_{\mathrm{fut}}$ the logits of the corresponding fitted CDFs at
the corresponding values, Equations (12)–(14). Equation (15) is the logit
identity that shows the construction preserves changes in odds
multiplicatively. If the training and application periods are identical then
$\Delta_{\mathrm{log-odds}} = 0$ and the distributions match exactly. The paper
notes that Equations (10)–(14) as written assume equal sample sizes in the four
distributions, and that "additional interpolations need to be introduced … to
make them work in the general case of unequal sample sizes, as explained by
Switanek et al. (2017)" (Sect. 3.2.1) — those interpolations are not given here.

**tas is the one exception.** Its event likelihood is not adjusted. Equations
(10)–(14) are replaced by the plain mapping

$$\hat{x}^{\mathrm{sim}}_{\mathrm{fut}} \mapsto \left(\hat{F}^{\mathrm{obs}}_{\mathrm{fut}}\right)^{-1}\left(\hat{F}^{\mathrm{sim}}_{\mathrm{fut}}\left(\hat{x}^{\mathrm{sim}}_{\mathrm{fut}}\right)\right)$$

"The reason for this exception is that the event likelihood adjustment can
produce artifacts if large nonlinear trends are present within the training or
application period. Examples of such cases have (only) been found for tas"
(Sect. 3.2.1). So for temperature the method reduces to trend-preserving
parametric quantile mapping with a normal distribution, applied to detrended
data.

**The downscaling method, MBCnSD**, runs in four steps: bilinear interpolation
of the coarse bias-adjusted data to the fine grid; randomization beyond
thresholds for bounded variables; the core algorithm, applied independently per
coarse cell; de-randomization. It requires every fine cell to lie entirely
inside one coarse cell.

Bilinear rather than conservative interpolation is used for the broadcast in
step 1, for two stated reasons: it "already generates some of the spatial
variability within each coarse grid cell that statistical downscaling has to
add", and it "transfers spatial gradients between coarse grid cells to the fine
grid" (Sect. 3.2.2). Conservative interpolation here would simply copy the
coarse value into every fine cell.

The core, step 3, has three sub-steps. In **3a** a rotation matrix $O$ whose
first column equals $W$ is applied to $Y$, $Z$ and $W$ (Equations 16–18), the
first rotated column of $Y$ is set to $X_i w/\tilde{w}$ to restore the
aggregated values, and the first rotated column of $Z$ is quantile-mapped to the
same target to carry the simulated change signal onto the observed fine-grid
data. In **3b** three operations repeat: draw a random orthogonal $K\times K$
matrix (Mezzadri, 2007) and rotate everything; quantile-map each rotated column
of $Y$ to the matching column of $Z$; then project $Y$ back onto the weighted
sum-preserving hyperplane of its pre-mapping value $\tilde{Y}$ by subtracting
$\left((Y-\tilde{Y})W\right)\otimes W$. That projection is the whole difference
from MBCn. In **3c** everything is rotated back and mapped once more so that no
value is out of bounds. Twenty iterations of 3b are used, "as this was deemed
sufficient for the MBCn algorithm by Cannon (2017)".

All quantile mappings inside MBCnSD are nonparametric: empirical quantiles at
$p \in \{0\,\%, 2\,\%, 4\,\%, \dots, 100\,\%\}$, with the sample minimum and
maximum as end points, linearly interpolated between the q–q pairs.

### Data

Simulations are CMIP5 daily output from four models — GFDL-ESM2M, HadGEM2-ES,
IPSL-CM5A-LR and MIROC5 — for ten variables (Table 1: hurs, pr, prsn, psl, rlds,
rsds, sfcWind, tas, tasmax, tasmin). They are conservatively interpolated (Jones,
1999) to a global 2°×2° grid. The historical experiment is concatenated with
rcp85 to give 1980–2015; rcp85 alone gives 2064–2099.

Observations are EWEMBI (Lange, 2019a), global, 0.5°, daily, 1979–2016. EWEMBI is
conservatively aggregated to 1°×1° and 2°×2° for the downscaling and the bias
adjustment respectively.

### Evaluation metrics

Three things are scored: bias adjustment, trend preservation, and spatial
variability adjustment. The distribution is represented by the 5th, 50th and
95th percentile of daily values — the lower tail, the centre and the upper tail.
For pr and prsn the 5th percentile is replaced by the dry-day frequency, and the
50th and 95th percentiles are taken over wet days only.

Spatial variability is the RMSD of the sixteen 0.5° time series inside a 2° cell
from their spatial average, Equation (19). It is computed both on the regular
2° grid and on a grid staggered by 1° in each direction. The staggered grid is
the harder test, since those cells "contain time series whose statistical
dependence is not adjusted by the ISIMIP3 statistical downscaling method"
(Sect. 3.3).

For bias adjustment and spatial variability the absolute error is

$$e = \left| y^{\mathrm{sim}}_{\mathrm{hist}} - x^{\mathrm{obs}}_{\mathrm{hist}} \right| \qquad (20)$$

For trend preservation it is the error in the simulated change:

$$e = \left| \left(y^{\mathrm{sim}}_{\mathrm{fut}} - y^{\mathrm{sim}}_{\mathrm{hist}}\right) - \left(x^{\mathrm{sim}}_{\mathrm{fut}} - x^{\mathrm{sim}}_{\mathrm{hist}}\right) \right| \qquad (21)$$

Errors are aggregated over all calendar months and grid cells by the grid-cell
area-weighted median. For prsn only latitudes above 60° are aggregated.

Bias and spatial-variability results are cross-validated: train on odd years of
1980–2015 and apply to even years, then swap, then merge. This split is used
instead of two consecutive blocks because it "reduces the influence of climate
trends on cross-validation results (Switanek et al., 2017)". The author states
the cost of that choice: "the resulting validation and training data sets are
rather similar in terms of decadal climate variability" (Sect. 3.3). Trend
preservation is not cross-validated — it is trained on the full 1980–2015 and
applied to 1980–2015 and 2064–2099.

## Main results

**All results are ratios, and none are tabulated.** Figures 7 to 10 plot the
ratio of the aggregated absolute error of one method to that of the other, one
symbol per (variable, model). A value above 1 favours the older method, below 1
the newer one. The text gives only the direction and rough size of each
comparison; no numeric ratio appears anywhere in the text or in a table, so the
values cannot be quoted from the paper.

**ISIMIP3 against ISIMIP2b bias adjustment, both at 2°** (Fig. 7). Biases are
better adjusted by ISIMIP3 for all ten variables in most months and grid cells.
The paper groups the size of the gain:

| Gain in bias adjustment | Variables |
|---|---|
| Greatest | hurs, rlds, rsds, sfcWind |
| Intermediate | pr, prsn, psl |
| Least, but "still considerable" | tas, tasmax, tasmin |

Trend preservation is more mixed:

| Trend preservation | Variables |
|---|---|
| ISIMIP3 considerably better | psl, rlds, tas |
| ISIMIP3 mostly better, with a few exceptions | hurs, rsds, sfcWind, tasmax, tasmin |
| Split | pr — ISIMIP3 much better for dry-day frequency, the two similar for the 50th percentile of wet-day precipitation, ISIMIP2b "a bit better" for the 95th |
| ISIMIP2b better | prsn |

The prsn result is explained rather than measured: ISIMIP2b leaves the prsn/pr
ratio unchanged, while ISIMIP3 adjusts it. The paper marks this as an inference
("presumably").

**BA+SD against LI+BA, downscaling from 2° to 0.5°.**

| Comparison | Outcome |
|---|---|
| Bias adjustment judged at 2° (Fig. 8) | BA+SD better, largest gain for pr; LI+BA slightly better only for the centre of the tasmax and tasmin distributions |
| Trend preservation at 2° (Fig. 8) | BA+SD better, "by considerable margins in particular for sfcWind and tas"; LI+BA slightly better in some cases for pr, prsn and psl |
| Bias adjustment judged at 0.5° (Fig. 9) | BA+SD slightly better for every variable except pr and prsn; LI+BA better for dry-day frequency and for the 50th percentile of wet-day precipitation, BA+SD better for the 95th |
| Spatial variability, regular 2° cells (Fig. 10) | BA+SD better in every case but one (prsn from HadGEM2-ES) |
| Spatial variability, staggered 2° cells (Fig. 10) | BA+SD better for hurs, tas, tasmax, tasmin; LI+BA better for prsn and psl; mixed for pr, rlds, rsds, sfcWind |

Across scales, "BA+SD adjusts biases better than LI+BA in the vast majority of
cases" (Sect. 4.2), and the same is said of spatial variability.

Two of these results are expected by construction and the paper says so. BA+SD
adjusts biases at 2° and then preserves the 2° values, so it should win at 2°;
and spatial variability inside regular 2° cells is exactly what MBCnSD adjusts,
so it should win there "by design". The staggered-grid comparison and the 0.5°
bias comparison are the tests that were not decided in advance.

**Two design choices tested inside the method.** Downscaling in two small steps
(2°→1°→0.5°) versus one big step gives "similar results", but the two-step
version gives slightly smoother fields (Fig. 5) and is preferred. Broadcasting
with bilinear rather than conservative interpolation also gives smoother fields
(Fig. 6). Both comparisons are shown as single-day maps for precipitation over
Europe, with no score attached, so they are demonstrations rather than
measurements.

**The corner case for MBCnSD** is Fig. 4: with certain rotation sequences the
original MBCn algorithm "almost reverses the ranks of values along one axis,
which results in strongly changed aggregated values". The projection step
prevents this. This is shown on artificial bivariate normal data, not on climate
data.

## Discussion

The author concludes that the new bias adjustment "preserves trends and adjusts
biases in distribution quantiles more accurately than the ISIMIP2b bias
adjustment method", and that the new stochastic downscaling "prevents the
variability inflation caused by spatial interpolation in ISIMIP2b" (Sect. 5).

"A major fraction of the bias adjustment gains can be attributed to the newly
introduced adjustment of the likelihood of individual events. This new feature
effectively corrects for the imperfections of the distribution fits that are the
basis of parametric quantile mapping" (Sect. 5). This attribution is asserted;
no ablation experiment isolating the event-likelihood adjustment is reported.
It also removes the need for the cap values ISIMIP2b used to keep extremes
plausible.

Trend preservation improves for two separate reasons: it is applied to all
quantiles rather than to the mean, and bias adjustment now happens at the
resolution at which the trends were simulated.

The results are offered as "a proof of concept of the new paradigm of a clear
separation of bias adjustment and statistical downscaling" (Sect. 5), not as a
general validation.

Limits acknowledged in the paper:

- The cross-validation split by odd and even years leaves training and
  validation sets "rather similar in terms of decadal climate variability".
- Trend preservation is not cross-validated at all.
- The event-likelihood adjustment can produce artifacts under large nonlinear
  trends, which is why tas is excluded from it.
- Equations (10)–(14) are written for equal sample sizes; the general case needs
  interpolations taken from Switanek et al. (2017).
- Version 1.0 is univariate across variables. v2.0, already under development,
  will insert an MBCn adjustment of the inter-variable copula between steps 4
  and 5; everything else, including the downscaling method, stays the same.

## Relevance to this project

*Everything below is my own reading, not the paper's content.* In this project's
terms, the paper's $x^{\mathrm{sim}}$ is our **predictor**, its
$x^{\mathrm{obs}}$ is our **target**, and its quantile mapping produces what we
call a **transfer function**.

- **This is the strongest argument we have against our current order of
  operations.** We interpolate CMIP6 to the ERA5-Land grid and then fit one
  transfer function per land pixel — that is exactly the LI+BA arrangement this
  paper replaces. Cite Sect. 1 and Maraun (2013) when we state the limitation,
  and cite Fig. 10 for the claim that a downscaling step recovers sub-cell
  spatial variability that LI+BA cannot.
- **MBCnSD is not a drop-in option for us.** It needs every fine pixel to sit
  entirely inside one coarse cell. Our ~1° CMIP6 grid and 0.1° ERA5-Land grid
  are not nested, so adopting it would mean first interpolating CMIP6 onto a
  compatible coarse grid, as the paper itself prescribes. Worth noting as future
  work, not as a near-term change.
- **Per calendar month, not per season — and the paper does not justify it.**
  Lange stratifies into twelve calendar months with no running window and no
  test of the choice; it is inherited from ISIMIP2b. So this paper is a
  precedent for monthly stratification but not evidence for it. Keep Reiter et
  al. (2018) as the evidence and cite Lange only for common practice. It does
  tell us something practical though: twelve monthly samples from 25 training
  years give about 775 days each, whereas our 9–10 years of seasons give about
  900. If we go monthly under the planned 1980–2004 training period, sample size
  is not the blocker.
- **The detrend–map–retrend sandwich is directly reusable for our 2081–2100
  projection.** Steps 3 and 7 remove a linear trend from each of the three time
  series before quantile mapping and add the application-period trend back
  after. For tas this is the whole trend-handling mechanism, and it is simple to
  implement. It matters for us precisely because our projection period has a
  strong within-period warming trend that a plain transfer function fitted on
  1980–2004 would partly absorb as variance.
- **Pseudo future observations are the cleanest way to make our transfer
  functions trend-preserving.** Equations (1)–(2) are the additive case, which
  is the case for temperature. The change is small: instead of mapping the
  future predictor onto the historical target distribution, shift each quantile
  of the historical target by the model's own simulated change at that quantile
  first. This is the concrete alternative to our current stationary mapping, and
  it should be evaluated before we produce any 2081–2100 field.
- **For temperature the method collapses to something close to what we already
  do.** tas uses a normal distribution, additive trend preservation, and no
  event-likelihood adjustment. That makes ISIMIP3BASD's temperature branch a
  realistic comparison method for us: parametric normal QM on detrended data,
  fitted per (pixel, month). It is a fair rival to our empirical-node transfer
  functions and needs no new data.
- **The tas exception is a warning about the tails.** The event-likelihood
  adjustment was dropped for temperature because it produced artifacts under
  large nonlinear trends. If we ever try a Switanek-style likelihood correction
  on our P5 and P95 scores, expect the same trouble, and check the projection
  period first.
- **The result we cannot borrow is the headline comparison.** Everything is
  reported as ratios of area-weighted median absolute errors against a
  predecessor method, with no absolute numbers. We cannot use any figure from
  this paper as a reference error level for our own MAE or our climatology
  floors.
