---
status: accepted
---

# QUANT's node grid starts at 11, and the table is anchored

A lookup table of *m* nodes reaches only to probability `0.5/m`. With fewer
than 11 nodes the lowest node sits above the 5th percentile, so P5 cannot be
represented and the boundary rule invents it.

This was found the hard way. An earlier unanchored version clamped flat below
its first node and carried a **+1.8 °C bias at SON P5** — worse than applying
no correction at all. It scored a *better* MAE than the fixed version, because
clamping acts as accidental shrinkage toward climatology.

**Anyone tuning this must read `bias` alongside `mae`.** Optimising MAE alone
leads straight back to the broken version, and the metric will say it is an
improvement.

## Considered options

Keeping a 4-node grid was briefly attractive precisely because of that better
MAE. It was rejected once the bias was examined per season and percentile
rather than pooled.
