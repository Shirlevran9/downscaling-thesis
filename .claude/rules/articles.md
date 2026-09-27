---
name: articles
description: How to read and summarise papers held under articles/ and refrences/.
paths:
  - "articles/**"
  - "refrences/**"
---

# Summarising papers

`articles/` is gitignored — the PDFs are not redistributable. Summaries written
from them are, so keep the summary self-contained enough to be useful without
the PDF.

## What a summary contains

- **Full citation** — authors, year, journal, volume, pages, DOI.
- **The question** the paper asks, in one or two sentences.
- **Method**, including the equations this project might reuse. Give each
  equation its number from the paper, so a later reader can find it.
- **What was found**, with the actual numbers.
- **What it means for this project** — kept clearly separate from what the
  paper says. Mark it as your own reading.
- **Direct quotes** where the exact wording matters, in quotation marks with a
  page or section reference. Keep them short and few.

## Accuracy

Never state a number, a method detail or a conclusion that is not in the paper.
If something is unclear, write that it is unclear. An invented detail in a
summary is worse than a gap, because it will be cited later as if checked.

Distinguish what the authors demonstrated from what they asserted or assumed.
Several of the quantile-mapping papers assert that sub-annual stratification
helps without testing it; only Reiter et al. (2018) tests it directly. That
distinction matters and belongs in the summary.

## Terminology

Papers use their own vocabulary — "transformation", "correction function",
"bias adjustment". When writing about them in project documents, translate into
the project's terms from `CONTEXT.md`, and note the original term once so the
mapping is visible.
