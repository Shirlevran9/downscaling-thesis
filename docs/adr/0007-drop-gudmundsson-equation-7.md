---
status: accepted
---

# Gudmundsson's Eq. 7 was implemented and then removed

The four-parameter form `y = (a + b·x)(1 − exp(−(x − c)/τ))` comes from
precipitation, where `x ≥ 0`. On temperature in °C it has two failure modes: if
`c` lands inside the data range the fitted curve reverses direction mid-range,
and as `τ → 0` the model degenerates to a linear fit with `c` and `τ`
unidentified.

Even with `τ > 0` and `c ≤ x_lo − ε` bounded to keep it monotone, it finished
last on every metric and cost 42 minutes against the linear fit's 29 seconds.

It is removed rather than kept as a losing option, because a method that is
both worse and 90× more expensive adds nothing to the comparison.
