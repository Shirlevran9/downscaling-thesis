# Writing Guidelines

Rules for structuring and writing the statistical downscaling report and any
accompanying scientific documents. Distilled from supervisor feedback (Ronit
Nirel, 17 April 2026) and the writing conventions of Nirel & Adar (2018) as
a style model.

See also:
- `guidelines/analysis_guidelines.md` — technical figure and code standards
- `guidelines/findings-presentation-guidelines.md` — figure-specific changes and model extension notes

---

## 1. Document Structure (IMRaD)

Organise all reports and thesis chapters according to the IMRaD framework.
The order of analysis execution does not determine the order of presentation.

| Section | Content |
|---|---|
| **Introduction** | Scientific motivation; study region and period; research question; brief statement of approach. Do not include data descriptions or results. |
| **Methods** | Data sources and their definitions; preprocessing steps; spatial pairing method; sea-fraction computation; global daily mean computation; variable definitions; statistical models and diagnostic plan. |
| **Results** | Concise description of findings, supported by figures and tables. Reference every figure and table in the text. No causal interpretation or speculation. |
| **Discussion** | Significance of findings; comparison with prior work; model limitations; conclusions and planned extensions. |

Items currently in the notebook Results section that belong in **Methods**:
- The CMIP6–ERA5 spatial linking procedure (nearest-neighbour assignment, padding)
- The definition of sea fraction per CMIP6 cell
- The computation of the global daily area-weighted mean TAS

---

## 2. Section and Paragraph Organisation

- Do not open a section with a sentence that paraphrases the heading. The
  heading does that job. Begin immediately with content.
- Avoid creating a sub-section for one or two sentences. Use an italic lead
  phrase to introduce a topic within a paragraph instead:

  > *Calendar alignment.* ERA5-Land uses a Gregorian calendar, while
  > CESM2-WACCM uses a no-leap calendar. Leap days were therefore removed …

- Reserve numbered sub-sections for substantively distinct topics that warrant
  their own heading.
- Avoid "over-branching": if a section has six sub-sections of two sentences
  each, collapse them into paragraphs with italic lead phrases.

---

## 3. Language Rules

### 3.1 Adjectives and tone

- **No dramatic adjectives.** Avoid "near-perfect", "excellent",
  "remarkably", "surprisingly", "striking". Let the numbers make the case.
  - ✗ "near-perfect linear correlation (Pearson r ≈ 0.9)"
  - ✓ "Pearson *r* ≈ 0.90"
- Scientific writing is dry. Extreme descriptors should be used extremely
  sparingly, if at all.

### 3.2 Formatting within sentences

- **No bold font in running text.** Bold is reserved for headings and
  table/figure labels. Never use it to emphasise words mid-sentence.

### 3.3 Variables, not datasets

- After data sources have been introduced in Methods, refer to the variables,
  not the datasets.
  - ✗ "ERA5-Land shows a cold bias" or "CMIP6 is warmer"
  - ✓ "T2M is lower than TAS by ~1.6 °C on average"
  - ✓ "TAS exhibits a systematic warm bias of ~1–2 °C relative to T2M"
- Results describe what the variables do; the dataset name is background context.

### 3.4 Hedged mechanisms

- When proposing an explanation for a pattern, hedge appropriately:
  - ✓ "This may be explained by the coarse model's inability to resolve
    orographic features."
  - ✗ "This is because the coarse model ignores orographic effects."
- Do not claim to enumerate all causes unless they have been tested.

### 3.5 Monotonicity and pattern accuracy

- Describe findings as they are, not as they would be tidiest.
- If a relationship is not strictly monotonic, describe the actual pattern.
  - ✗ "Residuals increase with sea fraction."
  - ✓ "The two CMIP6 cells with the highest sea fractions show somewhat
    elevated median residuals; the relationship across the full range is
    not monotonic [Fig. N]."

### 3.6 Sentence structure

- Two short sentences are better than one complex compound sentence.
- Do not repeat the same message in consecutive sentences.
- Do not begin a section or paragraph by explaining what will follow.

---

## 4. Statistical Reporting

### 4.1 Central and dispersion measures

Always pair a central measure with a dispersion measure; never report one alone.

| Preferred form | Example |
|---|---|
| Mean ± standard deviation | 19.2 ± 8.7 °C |
| Median [IQR] | 19.8 [10.2–28.4] °C |

### 4.2 Correlation

Report Pearson *r* in italic: "Pearson *r* = 0.90". Round to two decimal places.

### 4.3 Residual diagnostics

All residual diagnostics must use **standardised residuals** (raw residual
divided by residual standard deviation). The full diagnostic checklist:

1. Homoscedasticity — standardised residuals vs predictor values
2. Linearity — standardised residuals vs each predictor (curvature = missing term)
3. Temporal independence — ACF/PACF; Durbin–Watson statistic
4. Spatial independence — Moran's I; spatial map of mean residuals
5. Normality — histogram of standardised residuals with Gaussian overlay; Shapiro–Wilk or K–S test

---

## 5. Table Formatting

- **Uniform decimal places.** All values for the same variable use the same
  number of decimal places (e.g., one decimal for temperatures).
- **Units in the header, not in cells.** Write "Mean (°C)" in the column
  header, not "19.2 °C" in each cell.
- **Layout:** variables in rows, statistics in columns.
- **Column order (moments before quantiles):**
  Mean → Std Dev → Median → IQR → Min → Max
- When additional variables are added in later phases (T850, Z850, etc.),
  add them as rows; do not create separate tables.

Example layout for descriptive statistics:

| Variable | Mean (°C) | Std Dev (°C) | Median (°C) | IQR (°C) | Min (°C) | Max (°C) |
|---|---|---|---|---|---|---|
| T2M (ERA5-Land) | 19.2 | 8.7 | — | — | −21.3 | 40.6 |
| TAS (CMIP6) | 20.8 | 7.9 | — | — | −14.7 | 41.5 |

---

## 6. Figure and Caption Rules

For full technical details see `guidelines/analysis_guidelines.md` (§7, §8,
§12). Key rules:

- **Title describes what is shown, not what it means.**
  - ✓ "OLS residual distribution by CMIP6 cell sea fraction, 1990–1999"
  - ✗ "Coastal cells produce larger errors due to mixed land–sea signal"
- **Self-contained captions.** Include dataset, variable, period, and a
  brief description of each panel. A reader must understand the figure
  without consulting the running text.
- **No double-hierarchy numbering.** Use Fig. 1, Fig. 2 with sub-panels
  (a), (b). "Fig1a1" is not permitted.
- **Reference every figure in the text**, even briefly.
- **Font sizes** are set in `src/vis_constants.py`; never hardcode values.
- **Maps** must include the CMIP6 coarse-grid overlay and a geographically
  correct aspect ratio.
