---
status: accepted
---

# Paper summaries use a fixed eight-section template

Every literature summary under `papers/` follows the same eight sections, in
order: Title, Abstract, Glossary, Previous work, Problem definition, Main
results, Discussion, Relevance to this project. Sections 1 to 7 may contain only
what the paper says; section 8 is the one place the reader's own judgement is
allowed, and it opens with a sentence saying so.

The fixed order is the point. A summary written to no template is readable once
and then useless, because a later reader cannot find the thing they came for.
Glossary and Previous work are the two sections a free-form summary always drops,
and they are the two that make a summary still worth having a year later.

The earlier summaries, in a single gitignored HTML file, had roughly a third of a
page per paper and no Glossary or Previous work at all. They are preserved at
`history/summaries_html_v1.html`.

## Considered options

Keeping the summaries as authored HTML was rejected: markdown diffs show prose
changes rather than markup noise, render on GitHub, and let the app stay a thin
viewer over the real artefact.

Paper summaries deliberately use a **different register** from `summaries/`. The
findings documents forbid verdict headings; a paper summary's headings are fixed
section names, so the question does not arise, but the two rules should not be
merged.
