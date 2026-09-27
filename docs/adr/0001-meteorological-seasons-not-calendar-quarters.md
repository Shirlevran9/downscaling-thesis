---
status: accepted
---

# Meteorological seasons, not calendar quarters

The distribution window is DJF/MAM/JJA/SON, grouped by season-year with
December counted into the following year's winter. Calendar quarters
(JFM/AMJ/JAS/OND) split winter across two windows, which mixes the coldest and
warmest parts of the cold season into different distributions.

Grouping by season-year matters for cross-validation: holding out a season-year
holds out its December too, so a winter fold leaks nothing into its own
training set.

The cost is an asymmetry. DJF has 9 complete season-years over 1990–1999 while
the other three have 10, because January–February 1990 has no preceding
December and December 1999 has no following January. Both stubs are dropped.
Every table carries `n` for this reason.

## Considered options

Calendar quarters were the original implementation and are still what
`make_windows("quarter")` returns. That scheme was kept rather than changed,
so the older diagnostic pipeline is untouched — see ADR-0012.
