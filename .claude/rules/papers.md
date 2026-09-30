---
name: papers
description: How to read papers under papers/ and write their summaries — the section template, register, maths and quoting rules.
paths:
  - "papers/**"
  - "refrences/**"
---

# Summarising papers

`papers/<topic>/` holds a paper's PDF and its summary side by side. The PDFs are
gitignored, the summaries are not, so a summary must stand on its own — a reader
without the PDF should still get the paper's argument and its numbers.

## The template

Eight `##` sections, in this order, every time. A heading names its section; it
never states a verdict and never asks a rhetorical question.

**1. `## Title`** — the citation: full title, all authors, year, venue with
volume and pages, DOI, and the PDF filename.

**2. `## Abstract`** — the problem the paper addresses or the goal it sets, and
its main results. Two short paragraphs is usually right.

**3. `## Glossary`** — a table of the terms, acronyms and notation the paper
uses. One row per item: symbol or term, and what it means in this paper. Include
every symbol that appears later in the summary.

**4. `## Previous work`** — what earlier work the paper builds on, what those
researchers found, and what flaws the authors identify in it. This is the
paper's own account of the literature, not yours.

**5. `## Problem definition`** — the target in more detail than the abstract
carries: what exactly is being improved, under what assumptions, measured how.

**6. `## Main results`** — what the paper found, with the paper's own numbers.
Where it compares methods, techniques or models, **give a table with the metric
value for each one**; a comparison described in prose is not usable later.
Every quantity appears twice: in the paper's notation, and in plain words.

**7. `## Discussion`** — what the authors conclude from those results, including
the limits they acknowledge.

**8. `## Relevance to this project`** — your own reading. **This is the only
section that may contain anything not in the paper.** Open it with a sentence
that says so plainly, so the boundary is unmistakable to anyone skimming.

A summary of this shape runs about two pages. Sections 3 and 4 are the ones most
often skipped; they are the reason the summary is still useful a year later.

## Register

Write plain, clear English. The reader is not a native speaker. Short sentences,
ordinary words, no idioms. Do not simplify to the point of vagueness — a
technical idea stated precisely in easy words is the target, not a watered-down
version of it.

## Maths

Use LaTeX in markdown: `$\sigma\sqrt{2}$` inline, `$$...$$` for a display
equation. **Never put maths in backticks** — it renders as unreadable code.

Give each equation the number the paper gives it, so a later reader can find it:

> Equation (7) maps the modelled value $x$ to a corrected value.

## Accuracy and quoting

Base everything in sections 1 to 7 on the paper. Nothing else. Not your own
knowledge of the field, not what a related paper says, not what the method
"obviously" implies.

**Quote wherever the exact wording matters** — a definition, a claim, a warning,
a caveat. Use quotation marks and give a page or section reference. Quote rather
than paraphrase when the authors' own words are sharper than yours would be.

If something in the paper is unclear, write that it is unclear. A gap is honest;
an invented detail gets cited later as though it had been checked.

Distinguish what the authors **demonstrated** from what they **asserted or
assumed**. Several quantile-mapping papers assert that sub-annual stratification
helps without testing it; only Reiter et al. (2018) tests it directly. That
distinction belongs in the summary.

## Terminology

Papers use their own vocabulary — "transformation", "correction function", "bias
adjustment". In sections 1 to 7 keep the paper's term, since that is what the
paper says. In section 8, translate to this project's terms from `CONTEXT.md`
and note the mapping once.

## Filenames and the index

A summary is `papers/<topic>/YYYY-firstauthor-slug.md`, lower case, hyphenated —
for example `2012-gudmundsson-statistical-transformations.md`. The year first
means a directory listing is already in chronological order.

**Every new summary must be added to `papers/topics.json`**, in the topic's
`papers` array, ordered by year then first author. The app builds its navigation
from that file and cannot discover a summary that is not listed. Fill every
field, including the one-line description that appears in the nav.

## Working on a request

**A specific paper.** Work out which topic it belongs to from the paper itself.
If that is not obvious, ask rather than guess — a wrong topic is worse than a
question. File the PDF in `papers/<topic>/`, write the summary beside it, add it
to `topics.json`.

**A topic, with no papers named.** Summarise the PDFs already in that topic
folder, oldest first. Do not reach outside what is on disk. If the topic folder
is empty or thin, propose a reading list and wait for approval before fetching
anything.

**A batch.** Use one subagent per paper, in parallel. Give each the PDF path and
this template, and have it return the finished markdown. Do not have one agent
read five papers in sequence — it runs out of room and the later summaries get
thin. Update `topics.json` yourself once they are all back, so the ordering is
decided in one place.

**Acquiring a PDF.** Open-access papers can be fetched from arXiv, Copernicus
(HESS, GMD), AMS or ASCMO. Anything paywalled needs the user's own access — ask
for the PDF rather than guessing at bibliographic details from an abstract page.
