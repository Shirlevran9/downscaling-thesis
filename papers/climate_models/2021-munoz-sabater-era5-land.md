# ERA5-Land: the global land reanalysis behind our target field

## Title

**ERA5-Land: a state-of-the-art global reanalysis dataset for land applications**

Joaquín Muñoz-Sabater, Emanuel Dutra, Anna Agustí-Panareda, Clément Albergel,
Gabriele Arduini, Gianpaolo Balsamo, Souhail Boussetta, Margarita Choulga,
Shaun Harrigan, Hans Hersbach, Brecht Martens, Diego G. Miralles, María Piles,
Nemesio J. Rodríguez-Fernández, Ervin Zsoter, Carlo Buontempo, Jean-Noël Thépaut

*Earth System Science Data* **13**, 4349–4383, 2021.
doi:10.5194/essd-13-4349-2021 · CC BY 4.0 · received 9 March 2021, published
7 September 2021.

PDF: `MunozSabater2021_ERA5Land_ESSD.pdf`

## Abstract

This is a dataset description paper. It documents ERA5-Land, the land component
of the fifth generation of European ReAnalysis, produced by ECMWF inside the
Copernicus Climate Change Service (C3S). ERA5-Land is not a new analysis of
observations. It is an offline run of the ECMWF land surface model, driven by
meteorological forcing taken from the ERA5 atmospheric reanalysis and
interpolated from about 31 km to about 9 km, "including an elevation correction
for the thermodynamic near-surface state" (Abstract). It produces 50 land
variables, hourly, globally, at about 9 km. The period covered will run from
1950 to the present once production finishes; at the time of writing the paper,
1981 onward was public and the 1950–1980 segment was still in production.

The paper's second half is an evaluation. ERA5-Land is compared against in situ
networks (soil moisture sensors, snow depth stations, lakes, river gauges,
eddy-covariance towers) and against gridded references (MODIS land surface
temperature, GLEAM), with ERA5 and ERA-Interim included for comparison. The
main claim is that the water cycle improves over ERA5, while the energy cycle
is about the same: soil moisture (especially the root zone) and river discharge
improve, lake surface water temperature improves slightly, snow depth "present[s]
a mixed performance when compared to those of ERA5, depending on geographical
location and altitude" (Abstract), and skin temperature RMSE against MODIS is
reduced mostly through coastal points. The authors state the energy-flux result
is inconclusive because only 65 towers were used.

## Terms and notation

### Terms

A **reanalysis** here is a physically consistent, gridded reconstruction of past
conditions. **ERA-Interim**, **ERA5** and **ERA5-Land** are three ECMWF
products compared throughout the paper; Table 1 lists their differences.

An **offline land surface reanalysis** is the key idea. The land surface model
is run on its own, forced by meteorology from a completed atmospheric
reanalysis, with no feedback to the atmosphere. The paper states plainly:
"ERA5-Land does not assimilate observations directly. The observations influence
the land surface evolution via the atmospheric forcing" (Sect. 2). ERA5, by
contrast, has a **land data assimilation system (LDAS)** that analyses 2 m
temperature, 2 m relative humidity, snow and soil moisture against observations.

**CHTESSEL** is the Carbon Hydrology-Tiled ECMWF Scheme for Surface Exchanges
over Land, the land surface model at the core of ERA5-Land, run at model cycle
Cy45r1 (2018). ERA5 uses the same scheme at Cy41r2 (2016) and at 31 km.

The **atmospheric forcing** is the set of ERA5 fields that drive the run: air
temperature, specific humidity, wind speed and surface pressure from ERA5 model
level 137 (10 m above the surface), plus downward shortwave and longwave
radiation and liquid and solid total precipitation.

The **environmental lapse rate (ELR)** is the vertical temperature gradient used
to correct forcing air temperature for the height difference between the 31 km
ERA5 orography and the 9 km ERA5-Land orography. It is a daily field derived
from ERA5 lower-troposphere temperature profiles, following Dutra et al. (2020),
not a fixed constant.

A **stream** is one of the three independent production segments (from 2001,
from 1981, from 1950). **Spin-up** is the period run before the first archived
year so slow variables, mainly deep soil moisture, reach equilibrium.

**TCo1279** is the ECMWF triangular–cubic–octahedral operational grid, about
9 km, which is ERA5-Land's native grid.

**Skin temperature**, called **LST** in the paper, is "the theoretical
temperature of the Earth's surface that is required to satisfy the surface
energy balance" (Sect. 3.8). It is not the 2 m air temperature. **LSWT** is lake
surface water temperature, produced by the **FLake** lake model inside CHTESSEL.

### Notation

The validation scores, applied to different variables in different sections:

| Symbol | Meaning |
|---|---|
| $\mathrm{STDD}$ | Standard deviation of the difference between estimate and observation, equal to the unbiased RMSD |
| $\mathrm{bias}$ | Estimate minus observation (stated explicitly for snow, Sect. 3.4); for lakes the paper plots observation minus estimate (Figs. 11, 12) |
| $R$ | Pearson correlation between estimate and observation |
| $R_{\mathrm{AN}}$ | Pearson correlation computed on anomaly time series |
| $\mathrm{MAE}$ | Mean absolute error |
| $\mathrm{RMSE}$ | Root mean square error |

Soil moisture anomalies, Equation (1), use a $\pm 17$ d moving window:

$$\mathrm{SM_{AN}}(t) = \frac{\mathrm{SM}(t) - \overline{\mathrm{SM}}}{\sigma_{\mathrm{SM}}} \qquad (1)$$

Snow depth $\mathrm{SD}$ is reconstructed from snow water equivalent
$\mathrm{SWE}$ and snow density $\rho_{\mathrm{snow}}$, Equation (2), with
$\rho_{\mathrm{water}} = 1000$ kg m⁻³:

$$\mathrm{SD} = \rho_{\mathrm{water}} \frac{\mathrm{SWE}}{\rho_{\mathrm{snow}}} \qquad (2)$$

River discharge uses the modified Kling–Gupta efficiency, Equations (3) and (4):

$$\mathrm{KGE}' = 1 - \sqrt{(R-1)^2 - (\beta-1)^2 - (\gamma-1)^2} \qquad (3)$$

$$\beta = \frac{\mu_s}{\mu_o}, \qquad \gamma = \frac{\sigma_s/\mu_s}{\sigma_o/\mu_o} \qquad (4)$$

with $s$ the simulation, $o$ the observation, $\mu$ the mean discharge and
$\sigma$ its standard deviation. Note that Equation (3) as printed has minus
signs before the $\beta$ and $\gamma$ terms; the standard form of $\mathrm{KGE}'$
sums three squared terms, so this is very likely a typesetting error in the
paper. The optimum is 1 for $\mathrm{KGE}'$, $\beta$ and $\gamma$.

The skill score, Equation (5), takes GloFAS-ERA5 as the benchmark and
$\mathrm{KGE}'_{\mathrm{perf}} = 1$:

$$\mathrm{KGE_{SS}} = \frac{\mathrm{KGE}'_{\text{GloFAS-ERA5-Land}} - \mathrm{KGE}'_{\text{GloFAS-ERA5}}}{\mathrm{KGE}'_{\mathrm{perf}} - \mathrm{KGE}'_{\text{GloFAS-ERA5}}} \qquad (5)$$

$\mathrm{KGE_{SS}} > 0$ means ERA5-Land forcing beats ERA5 forcing;
$\mathrm{KGE_{SS}} = 0$ means no skill over the benchmark.

Energy-flux symbols: $H$ is the surface sensible heat flux, $\lambda \rho E$ the
surface latent heat flux, and $\beta$ the Bowen ratio, their ratio. (The paper
reuses $\beta$ for both the Bowen ratio and the KGE bias ratio.)

Monthly means are post-processed in two forms. Equation (6) is the monthly mean
of daily means, Equation (7) the monthly mean of synoptic means at a fixed hour,
and Equation (8) the flux version of Equation (6), which uses only the 24 h
accumulation step. Here $N_d$ is the number of days in the month, $N_s$ the
number of forecast steps in a day, $M$ the forecast model, $d$ the day and $s$
the forecast step from 00:00 UTC.

## Previous work

The paper's starting position is that offline land surface simulations are a
proven and cheap way to get consistent land fields. It cites early offline
intercomparisons driven by in situ data (Henderson-Sellers et al., 1995;
Etchevers et al., 2004) or by reanalyses (Dirmeyer et al., 1999), and later
water-resource and climate-modelling intercomparisons (Harding et al., 2011;
Schellekens et al., 2017; van den Hurk et al., 2016; Krinner et al., 2018).

The named predecessors are GOLD (Dirmeyer and Tan, 2001), MERRA-Land (Reichle
et al., 2011) and ERA-Interim/Land (Balsamo et al., 2015). The paper's objection
to ERA-Interim/Land is about status rather than method: it "was produced as a
one-off single-simulation research dataset covering the period of 1979–2010",
whereas ERA5-Land "is now an integral and operational component" of C3S with
guaranteed updates (Sect. 1). ERA-Interim/Land was therefore excluded from the
comparison.

Two weaknesses of coupled atmospheric reanalyses are identified. First,
systematic bias, "in particular in precipitation, which has led to the
development of bias correction methodologies (Weedon et al., 2011; Reichle
et al., 2017)". Second, land data assimilation can "mitigate model errors" but
"can also result in temporal and spatial inconsistencies (e.g. due to changing
observations' availability) as well as limitations in the closure of the surface
water budget (Zsoter et al., 2019)" (Sect. 1). This is the argument for running
land offline and assimilating nothing.

One acknowledged weakness of the offline approach is stated in the same section:
compared to high-resolution Earth observation data, there is a "lack of
small-scale heterogeneity found in offline model-only based estimates".

The lapse-rate correction is inherited, not invented here. It comes from Dutra
et al. (2020), who compared a constant ELR with daily ELR fields from ERA5.

## Problem definition

### Problem

The goal is to produce a land surface record that is finer, hourly, globally
complete and temporally consistent, without the discontinuities that data
assimilation and stream changes can introduce. The paper is explicit about where
the gain is expected to come from: "Differences between ERA5 and ERA5-Land are
not so obvious. They both share quite similar parameterizations of land
processes; the main improvement of ERA5-Land is due to the non-linear dynamical
downscaling with corrected thermodynamic input" (Sect. 1). In other words, the
physics is nearly the same as ERA5; the resolution and the elevation correction
are the new parts.

The paper compares no alternative methods for building such a dataset. It
compares the resulting product against two earlier reanalyses and against
observations.

### Production chain

**Grid and output.** ERA5-Land produces 50 variables describing the water and
energy cycles over land, globally, hourly, at 9 km on the TCo1279 grid. The full
field list is Table A2. Important for any user: "Note that in the CDS, and for
user convenience, the data have been interpolated to a regular lat–long grid of
0.1° resolution" (Sect. 5). So the 0.1° grid people download is an interpolation
of the native TCo1279 output, not the native grid itself.

**Streams and initialization.** Production runs in three streams: from 2001
(stream-1), from 1981 (stream-2), from 1950 (stream-3). Stream-1 is initialized
from the last year of a long prior ERA5 stream with a 3-year spin-up; stream-2
from an ERA5-Land 1 January climatology of 2001–2018 with a 3-year spin-up;
stream-3 from a 1981–2010 climatology with only 1 spin-up year, limited by the
availability of forcing data. The authors warn that even so, "discontinuities are
still possible at areas with very low variability of soil moisture (deserts and
polar regions)" (Sect. 2.1). Figure 3 shows the problem they were avoiding: ERA5
initialized its streams from ERA-Interim soil moisture, whose climatology is
different, and a 1-year spin-up was "not sufficient for deep soil moisture to
reach equilibrium".

**Static fields.** Land–sea mask, lake cover and depth, soil and vegetation type
and vegetation cover are time-invariant. Albedo and leaf area index are
prescribed as monthly climatologies. Sources are listed in Table A1: the land–sea
mask from GLOBCOVER 2006, orography from SRTM30, land cover from GLCCv1.2, leaf
area index and albedo from MODIS, soil type from the FAO Digital Soil Map of the
World, lake depth from the Global Lake Database.

**Forcing and the elevation correction.** The ERA5 fields listed under *Terms*
are interpolated from 31 km to 9 km "via a linear interpolation method based on a
triangular mesh" (Sect. 2.3). Precipitation is **not** bias-corrected, for two
stated reasons: the improved quality of ERA5 precipitation, and "reduced
dependencies on external data that would limit the near-real-time data
availability".

Air temperature, humidity and pressure **are** corrected for the height
difference between the two orographies, in four steps (Sect. 2.3):

1. relative humidity is computed from the interpolated, uncorrected fields;
2. air temperature is adjusted for the altitude difference using a daily ELR
   field derived from ERA5 lower-troposphere temperature profiles;
3. surface pressure is corrected for the altitude difference and for the
   temperature correction;
4. specific humidity is recomputed from the corrected temperature and pressure,
   assuming relative humidity does not change.

The evidence offered for this correction is from Dutra et al. (2020), not from
new work in this paper: the method "was shown to reduce the mean absolute error
(MAE) of daily maximum temperature by 10 % and by 4 % for daily minimum
temperature with respect to ERA5 when compared with 2941 stations over the
western US" (Sect. 2.3). Figure 4 shows the Alpine orography that ERA5 misses
and the colder ERA5-Land surface temperature over the high peaks, but gives no
numbers in the text.

**Land surface model.** CHTESSEL at cycle Cy45r1, integrated in 24 h cycles.
Relative to ERA-Interim's TESSEL it adds revised soil hydrology, a revised snow
scheme, climatological vegetation seasonality, a new bare-soil evaporation
scheme, the FLake lake model, and carbon fluxes. Relative to ERA5's CHTESSEL the
differences are "mostly technical", except an updated soil thermal conductivity
following Peters-Lidard et al. (1998), a soil water balance conservation fix, and
correct handling of rain over snow. Glaciers have no independent treatment: grid
points above 50 % ice cover are assigned a fixed snow water equivalent of 10 m.

**How 2 m air temperature is produced is not described in this paper.** Table A2
lists "2 m temperature" and "2 m dew-point temperature" among the ERA5-Land
generated fields, but the text nowhere states the diagnostic used to get from the
model's surface layer to 2 m. The paper refers the reader to chapter IV of the
IFS documentation for the model description. Any statement about that diagnostic
must come from the IFS documentation, not from here.

### Data

Evaluation data, mostly for 2001–2018:

| Variable | Reference | Period | Size |
|---|---|---|---|
| Soil moisture | In situ sensors from 14 networks via the International Soil Moisture Network (Table 2) | 2010–2018 | > 800 sensors at 5, 20, 50 cm |
| Snow (SWE) | ESM-SnowMIP reference sites (Table 3) | site-dependent | 10 sites |
| Snow depth | GHCN-daily v3.24 | Jul 2010 – Jun 2018 | > 6000 stations after filtering |
| Lake surface water temperature | Alqueva reservoir (Portugal), 27 Finnish lakes (SYKE), a global in situ/satellite inventory | 2017–2018; 2000–2016; 1995–2009 | 1; 27; 272 usable lakes |
| River discharge | GloFAS observation database, ~75 % GRDC | 2001–2018 | 1285 stations after filtering |
| Energy fluxes | FLUXNET 2015 eddy-covariance towers | 2001–2014 | 65 sites |
| Skin temperature | MODIS MYD11C3/MOD11C3 v6 average ensemble, 0.05° monthly | Jan 2003 – Dec 2018 | global |

There is **no evaluation of 2 m air temperature against station data in this
paper.** The temperature variables that are validated are lake surface water
temperature and skin temperature. The only 2 m temperature numbers in the paper
are the Dutra et al. (2020) figures quoted above for the ELR correction.

The MODIS reference has its own error: Chen et al. (2017) report the average
ensemble validated against 156 flux towers with "RMSE = 2.65, mean bias < ±1 K"
(Sect. 3.8). MODIS was upscaled to 0.1° for the ERA5-Land comparison and to
0.25° for ERA5 and ERA-Interim.

### Evaluation metrics

Soil moisture uses STDD, bias, $R$ and $R_{\mathrm{AN}}$, grouped by continent
and shown as box plots; anomalies follow Equation (1). Correlation differences
between ERA5 and ERA5-Land are called significant only when the confidence
intervals do not overlap.

Snow uses mean bias (reanalysis minus observation) and RMSE, over December to
June of 2010–2018, with snow depth reconstructed by Equation (2). For the
ESM-SnowMIP sites the metrics are normalized, and bars show the spread over the
four nearest grid points.

Lakes use MAE against in situ LSWT during ice-free periods, with significance
tested by the Kruskal–Wallis test by ranks, plus bias distributions.

River discharge uses $\mathrm{KGE}'$, its three components, and the skill score
$\mathrm{KGE_{SS}}$ of Equations (3) to (5).

Energy fluxes use bias, standardized MAE and $R_{\mathrm{AN}}$ against
eddy-covariance measurements at hourly, 3-hourly and daily resolution. The
reference fluxes "were not corrected for energy balance closure", which the
authors note leads to a known tendency to underestimate the latent heat flux.

Skin temperature uses bias, $R$, RMSE on the full monthly time series, and
$R_{\mathrm{AN}}$ on departures from the monthly climatology.

## Main results

### Temperature

All temperature numbers reported in the paper, in one place:

| Quantity | Reference | ERA-Interim | ERA5 | ERA5-Land |
|---|---|---|---|---|
| Skin temperature bias (K), global mean, 2003–2018 (Table 5) | MODIS ensemble | 3.65 | 1.64 | 1.36 |
| Skin temperature RMSE (K), same (Table 5) | MODIS ensemble | 5.87 | 3.96 | 3.78 |
| Skin temperature $R$ (Table 5) | MODIS ensemble | 0.91 | 0.94 | 0.94 |
| Skin temperature $R_{\mathrm{AN}}$ (Table 5) | MODIS ensemble | 0.49 | 0.75 | 0.75 |
| LSWT MAE (°C), hourly, Alqueva 2017–2018 (Table 4) | in situ | — | 3.29 | 3.22 |
| LSWT MAE (°C), daily, 28 lakes (Table 4) | in situ | — | 2.71 | 2.68 |
| LSWT MAE (°C), daily, 6 lakes whose depth is more realistic at 9 km (Table 4) | in situ | — | 3.56 | 2.71 |
| LSWT MAE (°C), daily, 10 lakes with unchanged depth (Table 4) | in situ | — | 2.59 | 2.34 |
| LSWT MAE (°C), summer means, 272 lakes (Table 4) | in situ | — | 3.04 | 3.07 |
| LSWT MAE (°C), summer means, 246 non-exceptional lakes (Table 4) | in situ | — | 2.32 | 2.32 |
| Summer LSWT cold bias (°C), average (Sect. 6c) | in situ | — | 2.2 | 1.3 |
| 2 m daily maximum temperature MAE, 2941 western US stations (Dutra et al., 2020, quoted in Sect. 2.3) | stations | — | reference | 10 % lower |
| 2 m daily minimum temperature MAE, same | stations | — | reference | 4 % lower |

Read the last two rows with care. They are the result of Dutra et al. (2020) on
the lapse-rate correction method, quoted by this paper; they are not a validation
of the released ERA5-Land dataset carried out here, and they cover one region of
one country.

Two further temperature statements have no numbers attached in the text. The
skin temperature improvement of ERA5-Land over ERA5 is "modest" and comes "mainly
due to the contribution of coastal points where spatial resolution is important"
(Abstract, Sect. 6f); over complex terrain the differences "do not seem to favour
any particular reanalysis" (Sect. 6f). Figure 20 gives $\Delta$RMSE and $\Delta R$
at three individual pixels (coastal Norway, the Alps, Iceland), but those values
are printed in the figure and not tabulated, so they are not quoted here.

The exceptional lakes — glacier-fed, saline and warm lakes, 26 of them — show
LSWT MAE "even beyond 10 °C", and the errors "are very similar for ERA5"
(Sect. 4.3).

### Soil moisture

Against North American sensors, ERA5-Land and ERA5 are close at 5 cm, and
ERA5-Land is better at depth. Where the correlation difference between the two is
significant, the share of sensors favouring ERA5-Land rises with depth: 64 % of
382 sensors at 5 cm, 72 % of 417 at 20 cm, 88 % of 479 at 50 cm (Sect. 4.1). In
Europe, Africa and Australia, ERA5-Land is slightly better overall, but the
authors state these top-layer results "are not conclusive, partly because of the
insufficient number of available stations".

### Snow

Mixed, and the split is by altitude. ERA5-Land has lower RMSE at moderate
mountain altitudes, roughly 1300–2500 m, and at Arctic and boreal forest sites.
Above about 3300 m ERA5 is better, which the authors attribute partly to
compensating errors, noting the error spread over the four nearest grid points is
much larger for ERA5. From the GHCN stations binned by height (Fig. 10): below
about 1500 m ERA5 is slightly better, between about 1500 and 3000 m ERA5-Land is
better, above 3300 m ERA5 is better again. Regionally, ERA5-Land wins over the
US Rockies; ERA5 wins over Scandinavia, where it assimilates a dense SYNOP snow
network. All reanalyses have a negative snow bias at mountain sites, "very likely
due to the smoothing of the orography at the resolution of the reanalysis"
(Sect. 4.2).

### River discharge

The clearest improvement. Across 1285 stations the global median $\mathrm{KGE}'$
rises from 0.26 (interquartile range −0.04 to 0.49) with ERA5 forcing to 0.37
(0.08 to 0.57) with ERA5-Land forcing. Median correlation rises from 0.60 to
0.64, median bias ratio from 0.73 to 0.89, "which is equivalent to a 16 %
reduction in overall bias" (Sect. 4.4). Variability errors barely change.
$\mathrm{KGE_{SS}}$ is positive at 65 % of stations, with a global median of 0.08;
11 % of stations are substantially worse ($\mathrm{KGE_{SS}} < -0.2$), mainly in
the western US and South America.

### Energy fluxes

ERA5-Land beats ERA-Interim on every score. Against ERA5 the picture is flat:
biases "only marginally better in ERA5-Land", MAE typically lower for
$\lambda \rho E$ but higher for $H$, correlations better only for
$\lambda \rho E$ at daily resolution and for the Bowen ratio (Sect. 4.5.1). The
largest disagreements are at high-altitude sites. ERA5-Land beats
GLEAM + ERA5-Land except in bias. GLEAM + ERA5 and GLEAM + ERA5-Land are very
close, which the authors read as showing "that the near-surface air temperature
and net radiation are quite similar in both reanalyses for the locations of the
eddy-covariance sites" (Sect. 4.5.2). That is an indirect argument, over
65 towers, not a temperature validation.

## Discussion

The overall conclusion: the water cycle improves in ERA5-Land relative to ERA5,
the energy cycle performs about the same, and both are substantially better than
ERA-Interim. "One can conclude that the horizontal resolution matters and is a
very important aspect in the accurate simulation of the spatial and temporal
evolution of the hydrological cycle" (Sect. 6). The authors recommend ERA5-Land
over ERA5 "for all types of land applications", while telling users to weigh data
volume, area of application and temporal consistency.

Limits the authors acknowledge:

- The energy-flux evidence is weak. "This paper could only provide evidence of a
  modest improvement of the surface fluxes of ERA5-Land compared to ERA5. The
  latter conclusion is based on an evaluation with respect to a small number of
  available samples" (Sect. 6), namely 65 towers.
- An exhaustive evaluation of all 50 variables "is not feasible in a single
  paper"; the community is invited to evaluate individual components.
- No dataset-specific uncertainty estimate exists. ERA5-Land uncertainties are
  simply those of ERA5, and first tests of an offline ensemble gave
  "unrealistically low" spread, because the surface model physics is not
  perturbed (Sect. 7).
- ERA5 precipitation forcing still has large biases, especially in the tropics,
  and is not corrected (Sect. 7).
- Auxiliary data are static: fixed land cover, with cities treated as
  non-existent, and albedo and leaf area index as monthly climatologies, so
  interannual vegetation anomalies are not represented (Sect. 7).
- Snow above about 3300 m is worse than ERA5; snow transport and sublimation are
  absent from the scheme.
- The lake scheme can jump: on 3 August 2018 at Alqueva the mixed-layer depth
  jumped 5.3 m in one hour, raising LSWT by 17.6 °C to 40.7 °C.
- Carbon fluxes are computed but withheld "because of persistent biases".
- Potential evaporation "can give unrealistic results due to too-strong
  evaporation forced by dry air", so it needs caution in arid conditions
  (Sect. 2.4).
- Third-party caveats are cited: soil temperature in permafrost regions (Cao
  et al., 2020) and diurnal-cycle land surface temperature errors over the
  Iberian Peninsula linked to the land cover database and the 1 h time step
  (Johannsen et al., 2019; Nogueira et al., 2020).
- Stream discontinuities remain possible in deserts and polar regions, and the
  1950 stream had only one spin-up year.

## Relevance to this project

*Everything below is my own reading, not the paper's content.* In this project's
terms (`CONTEXT.md`), ERA5-Land supplies the **target** — the observed
temperature field our **transfer functions** map onto — on the fine 0.1° grid
of **pixels** over the **domain**.

- **Our target is a model output, not an observation.** ERA5-Land assimilates
  nothing. Our daily mean 2 m temperature target is CHTESSEL driven by ERA5
  forcing, and the observational content reaches it only through ERA5's own
  assimilation. Write it that way in the thesis: "reanalysis" not "observations",
  and cite Sect. 2 for the sentence that ERA5-Land does not assimilate
  observations directly. It also means our **bias** is predictor minus a modelled
  target, so a residual bias may be ERA5-Land's error rather than the GCM's.

- **This paper cannot tell us how accurate our target's 2 m temperature is.**
  It validates skin temperature and lake temperature, not 2 m air temperature
  against stations. Anyone who asks for the accuracy of our target must be sent
  to a different reference. Do not quote Table 5 (skin temperature, bias 1.36 K,
  RMSE 3.78 K against MODIS) as if it were the 2 m error — skin temperature is a
  different variable with a satellite reference that itself has RMSE 2.65 K.

- **The one 2 m number here bounds our error floor loosely, and it is
  second-hand.** The Dutra et al. (2020) result — 10 % lower MAE for daily
  maximum and 4 % for daily minimum against 2941 western US stations — is a
  relative gain of the lapse-rate correction, not an absolute error, and it is
  for a different region. If we need an absolute bound on ERA5-Land 2 m
  temperature error over the Eastern Mediterranean, we have to find or compute
  one. Until then, state honestly that our **climatology floor** is the floor of
  reproducing ERA5-Land, not of reproducing reality.

- **Practical consequence for the error floor.** Our reported MAE and RMSE
  measure agreement with ERA5-Land. Any systematic ERA5-Land error over our
  domain sits inside every score we publish and inside the fitted transfer
  functions themselves, because quantile mapping will absorb it. This is worth
  one paragraph in the thesis limitations, not more, but it must be there.

- **Two resolution facts to state correctly in the data chapter.** The native
  grid is TCo1279, about 9 km; the 0.1° regular grid we download from the CDS is
  an interpolation of it (Sect. 5). And ERA5-Land's extra detail relative to
  ERA5 comes from the finer orography plus the daily ELR correction, not from
  new observations — so fine-scale structure in our target over the Levantine
  mountains, the Jordan Rift and Sinai is a lapse-rate downscaling of a 31 km
  field, not measured relief-driven variation.

- **Complex terrain is where the target is weakest, and our domain has some.**
  The paper's own evidence for terrain trouble is in snow and fluxes, not
  temperature, so do not overclaim. But the mechanism it names — smoothed
  orography, larger forcing errors at altitude — applies to temperature too.
  This is a reason to look at whether our per-pixel skill degrades with
  elevation, using the ETOPO DEM we already merge in `elevation.py`, and a
  candidate explanation if it does.

- **Temporal coverage constrains our planned scope, not our current baseline.**
  Our 1990–1999 baseline sits inside the 1981-onward segment that was public in
  2021. The planned 1980–2004 training period starts before it: 1980 was in the
  1950–1980 back extension, still in production when this paper was written, and
  initialized from a 1981–2010 climatology with only one spin-up year. Check the
  CDS release status before training on 1980, and prefer 1981 as the start if
  there is any doubt.

- **Do not mix ERA5-Land hourly data across the 2001 and 1981 stream boundary
  without checking.** The paper warns that discontinuities remain possible where
  soil moisture variability is low, which includes deserts — a large part of our
  domain is arid. For daily mean 2 m temperature the risk is probably small, but
  a simple check of the 2000/2001 transition in our target is cheap.

- **Citation use.** Cite this paper for what ERA5-Land is, how it is produced,
  its resolution and coverage, and the fact that it assimilates nothing. Do not
  cite it for ERA5-Land 2 m temperature accuracy.

## Note on this summary

This is a dataset description paper, not a method paper, so *Problem definition*
has no `### Models` subsection; the production chain, forcing, lapse-rate
correction and land surface model are gathered under a `### Production chain`
subsection instead, and one line in `### Problem` records that the paper compares
no competing methods. *Main results* is split by variable with `###` subheadings,
because the paper evaluates six unrelated variables and a single prose block
would be unusable; the temperature table is placed first since it is what this
project needs.
