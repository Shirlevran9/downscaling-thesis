---
status: accepted
---

# Cross-validation holds out whole season-years

Folds are season-years, not random days. Daily temperature has a day-to-day
autocorrelation around 0.8, so a random-day split puts neighbouring, nearly
identical days on both sides of the split and reports an error far below the
truth.

DJF has 9 folds and the other seasons 10 — see ADR-0001.

Because a season-year rolls December forward, holding out a winter holds out
its December too, so no winter fold trains on part of its own test season.

Every reported number comes from held-out folds. In-sample, quantile mapping
reproduces the observed percentile exactly, so an in-sample figure would read
as near-zero error and mean nothing.
