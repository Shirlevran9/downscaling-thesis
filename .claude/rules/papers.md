---
name: papers
description: How to read papers under papers/, write their summaries, and act on the notes left on them.
paths:
  - "papers/**"
  - "refrences/**"
---

# Summarising papers

`papers/<topic>/` holds a paper's PDF and its summary side by side. The PDFs are
gitignored, the summaries are not, so a summary must stand on its own — a reader
without the PDF should still get the paper's argument and its numbers.

**These are guidelines, not a rigid form.** Where a paper does not fit the shape
below, depart from it — and say so, both when delivering the summary and in the
summary itself (see *When a paper does not fit*). A summary bent to fit a
template it does not suit is worse than one that explains why it differs.

## The sections

Eight `##` sections, in this order. A heading names its section; it never states
a verdict and never asks a rhetorical question.

**1. `## Title`** — the citation: full title, all authors, year, venue with
volume and pages, DOI, and the PDF filename.

**2. `## Abstract`** — the problem the paper addresses or the goal it sets, and
its main results. Two short paragraphs is usually right.

**3. `## Terms and notation`** — what the reader needs in order to follow the
rest. Write the **terms as prose**: a term is a concept and deserves a sentence,
not a table cell. Group the section internally, normally *Terms* then *Notation*.

Reserve **tables for symbols**, and only where a cluster of symbols belongs to
one piece of maths — the free parameters of a set of equations, the members of a
metric family. A table with one symbol per row covering the whole paper is a
lookup list nobody reads; a small table beside the equations it serves is
useful.

Every symbol that appears later in the summary must be defined here.

**4. `## Previous work`** — what earlier work the paper builds on, what those
researchers found, and what flaws the authors identify in it. This is the
paper's own account of the literature, not yours.

**5. `## Problem definition`** — the target in more detail than the abstract
carries. Where the paper supports it, divide this into `###` subsections:

- `### Problem` — what exactly is being improved, and under what assumptions
- `### Model` or `### Models` — the methods, with their equations
- `### Data` — what was used, over what period and area
- `### Evaluation metrics` — how performance was measured

Include only the subsections the paper actually has. A climatology paper has
data and diagnostics but no competing methods, so it gets no *Models*
subsection — and the summary says in one line that the paper compares no
methods, so the absence reads as a fact about the paper rather than a gap in the
summary. When a subsection is included it takes a `###` heading of its own; do
not fold two of them together.

**6. `## Main results`** — what the paper found, with the paper's own numbers.
Where it compares methods, techniques or models, **give a table with the metric
value for each one**; a comparison described only in prose is not usable later.
Every quantity appears twice: in the paper's notation, and in plain words.

**7. `## Discussion`** — what the authors conclude, and the limits they
acknowledge. **Keep this purposeful.** Every sentence must be a conclusion or a
limitation that a later reader actually needs. Name it and move on: do not
restate the results, do not explain why the conclusion follows, do not add
commentary. There is no word limit — there is a test, and a sentence that fails
it is cut.

**8. `## Relevance to this project`** — your own reading. **This is the only
section that may contain anything not in the paper.** Open it with a sentence
saying so, so the boundary is unmistakable to anyone skimming.

**Keep this practical.** Every point must change something about what we do:
what to cite, what to reuse, what to avoid, or what one of our own results means
in this paper's light. A point that is merely interesting is cut. Name the thing
and its consequence in a line or two.

Sections 3 and 4 are the ones a free-form summary always drops, and they are two
of the reasons a summary is still worth having a year later. Sections 7 and 8
are where length creeps in, and they are where to cut first.

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
an invented detail gets cited later as though it had been checked. The same
applies to a result the paper gives only in a figure: say the values are not
tabulated rather than reading numbers off a plot.

Distinguish what the authors **demonstrated** from what they **asserted or
assumed**. Several quantile-mapping papers assert that sub-annual stratification
helps without testing it; only Reiter et al. (2018) tests it directly. That
distinction belongs in the summary.

## Terminology

Papers use their own vocabulary — "transformation", "correction function", "bias
adjustment". In sections 1 to 7 keep the paper's term, since that is what the
paper says. In section 8, translate to this project's terms from `CONTEXT.md`
and note the mapping once.

## When a paper does not fit

Depart from the template when the paper requires it: a subsection that does not
apply, an extra one it needs, a longer *Main results* because it compares twelve
methods.

Say so twice. **In the message delivering the summary**, name what differed and
why. **And in the file**, under a final `## Note on this summary` heading, in one
or two lines. Add that heading only when something actually departed from the
template — its absence means the summary followed it.

## Filenames and the index

A summary is `papers/<topic>/YYYY-firstauthor-slug.md`, lower case, hyphenated —
for example `2012-gudmundsson-statistical-transformations.md`. The year first
means a directory listing is already in chronological order.

**Every new summary must be added to `papers/topics.json`**, in the topic's
`papers` array, ordered by year then first author. The app builds its navigation
from that file and cannot discover a summary that is not listed. Fill every
field, including the one-line description that appears in the nav.

## Notes left on a summary

Notes live in `papers/comments/<paper-stem>.json`, written through the reading
app. Each records the section it was made in, the text it quotes, its own text,
and a status of `open` or `resolved`.

**Read the open notes before working on a paper's summary**, and read them
whenever asked to act on comments. Then, for each one:

- **A note asking for a change, or reporting something wrong** — make the fix,
  then set its status to `resolved`. Resolve only after the change is actually
  made, never to tidy the list.
- **A note that is the reader's own reminder, or a question for them** — leave it
  `open` and say so in your reply, naming what is still waiting on them. Do not
  answer a question addressed to the reader by editing the summary.
- **A note you disagree with** — say why in your reply and leave it open. Do not
  resolve a note by overruling it.

Several notes often share one cause. Fix the cause, resolve all of them, and say
in your reply that they were one problem rather than listing them as separate
wins. A note about how something *renders* is usually a bug in the app or in the
markdown, not a request to reword the summary — check that first.

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
(HESS, GMD), AMS or ASCMO. Anything paywalled needs the reader's own access —
ask for the PDF rather than guessing at bibliographic details from an abstract
page.
