---
name: writing-summaries
description: How to write findings summaries, reports and thesis chapters — structure, headings, tone, statistical reporting.
paths:
  - "summaries/**"
---

# Writing findings and reports

Distilled from supervisor feedback (Ronit Nirel, 17 April 2026) and the style
of Nirel & Adar (2018). Terminology is fixed in `CONTEXT.md` — read it.

## Structure

IMRaD. The order the analysis ran in does not decide the order it is presented
in.

| Section | Holds |
|---|---|
| **Introduction** | Motivation, region and period, the question, one line on approach. No data description, no results. |
| **Methods** | Data sources, preprocessing, spatial pairing, variable definitions, models, evaluation plan. |
| **Results** | What was found, with figures and tables. Every figure referenced in the text. **No interpretation.** |
| **Discussion** | What it means, comparison with prior work, limitations, next steps. |

Three things belong in Methods that tend to drift into Results: the spatial
linking procedure, the definition of sea fraction, and the computation of the
global daily area-weighted mean.

## Headings

A heading names what the section covers. It does not state the conclusion, and
it does not try to interest the reader.

- ✗ "What's modelled and what's not?"
- ✗ "A climatology benchmark beats every method"
- ✓ "Evaluation metrics"
- ✓ "Comparison against the climatology floor"

Rhetorical questions, teasers and verdict-headings all read as machine-written.
**Report, do not impress.**

## Paragraphs

Do not open a section by paraphrasing its heading. The heading did that. Start
with content.

Do not build a sub-section for two sentences. Use an italic lead phrase inside
a paragraph instead:

> *Calendar alignment.* ERA5-Land uses a Gregorian calendar, while CESM2-WACCM
> uses a no-leap calendar. Leap days were therefore removed …

If a section has six two-sentence sub-sections, collapse them.

## Language

**No dramatic adjectives.** Not "near-perfect", "excellent", "remarkably",
"surprisingly", "striking". The numbers carry it.

- ✗ near-perfect linear correlation (Pearson r ≈ 0.9)
- ✓ Pearson *r* = 0.90

**No bold in running text.** Bold is for headings and figure or table labels
only.

**Refer to the variables, not the datasets**, once the sources are introduced.
Use the terms in `CONTEXT.md`: the predictor and the target.

- ✗ "ERA5-Land shows a cold bias"
- ✓ "The predictor is about 1.6 °C warmer than the target"

**Hedge proposed mechanisms.** "This may be explained by …", not "This is
because …". Do not claim to have enumerated causes that were not tested.

**Describe the pattern that is there**, not the tidy version.

- ✗ "Residuals increase with sea fraction."
- ✓ "The two cells with the highest sea fractions show somewhat elevated median
  residuals; across the full range the relationship is not monotonic [Fig. N]."

**Two short sentences beat one compound sentence.** Do not say the same thing
twice in a row. Do not announce what a section will contain.

## Describing results

In Results, say what the number is. Do not say why.

- ✗ "MAE rises at P5 because the tails are poorly sampled, which reflects the
  short record."
- ✓ "MAE is highest at P5 (1.61 °C) and lowest at P50 (1.18 °C)."

The explanation belongs in Discussion. Keeping them apart is what makes the
Discussion worth reading.

## Numbers

Two decimal places unless there is a reason otherwise. Pearson *r* in italic,
two decimals: *r* = 0.90.

Pair every central measure with a dispersion measure, and put a space before
the unit:

- Mean ± SD — `19.2 ± 8.7 °C`
- Median [IQR] — `20.1 [12.7, 26.3] °C`

Bias is predictor minus target; positive means too warm.

## Figures and tables

Number everything and reference every item in the text. Captions follow the
project format and stand alone — see the visualization rule.

Tables: uniform decimal places down a column, units in the header rather than
repeated in cells, variables in rows and statistics in columns, ordered Mean →
SD → Median → IQR → Min → Max.

## Residual diagnostics

Report all seven, on **standardised** residuals:

1. Homoscedasticity — residuals against fitted values
2. Linearity — residuals against each predictor
3. Autocorrelation — ACF/PACF plus Durbin–Watson
4. Spatial autocorrelation — Moran's I
5. Normality — Q–Q plot
6. Seasonal stratification
7. A spatial map of residuals
