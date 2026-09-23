"""Sanity checks for src/qm_transforms.py and src/qm_nodes.py.

Synthetic data only, so it runs in a second and needs no NetCDF files.  This
repository has no test framework; this script is the stand-in.  Run it after
any change to the transforms or the node builder.

Run:  python3.10 scripts/sanity_qm_transforms.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import qm_nodes as qn                      # noqa: E402
from src.qm_transforms import TRANSFORMS            # noqa: E402

PCT = (5, 25, 50, 75, 90)
FAILS = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def synth(n_days=828, n_pix=40, seed=0, identity=False):
    """A DailySeason with a known monotone relation."""
    rng = np.random.default_rng(seed)
    base = rng.normal(28, 4, (n_days, n_pix))
    x = base.astype(np.float32)
    if identity:
        y = x.copy()
    else:
        # y = 0.85 x - 2 + noise, i.e. a real bias to correct
        y = (0.85 * base - 2.0 + rng.normal(0, 1.0, base.shape)).astype(np.float32)
    syear = np.repeat(np.arange(1990, 1990 + 9, dtype=np.int16), n_days // 9)
    syear = np.resize(syear, n_days).astype(np.int16)
    return qn.DailySeason(
        season="JJA", x=x, y=y, syear=syear,
        lat=np.linspace(30, 32, n_pix, dtype=np.float32),
        lon=np.linspace(34, 36, n_pix, dtype=np.float32),
    )


def pct_of(a, q):
    """Column-wise percentiles, the same estimator the pipeline uses."""
    return np.nanpercentile(a, np.asarray(q, float), axis=0)


print("\n=== 1. round trip: fit with x == y, predict(x) should return x ===")
ds_id = synth(identity=True)
nodes_id = qn.fold_nodes(ds_id, hold_out=None, n_nodes=99)
xq_id = pct_of(ds_id.x, PCT)
for name, cls in TRANSFORMS.items():
    t = cls().fit(nodes_id)
    err = np.nanmax(np.abs(t.predict_percentiles(xq_id) - xq_id))
    tol = 0.05
    check(f"{name:7s} max|predict(x) - x|", err < tol, f"{err:.2e}  (tol {tol})")

print("\n=== 2. monotonicity of predict() over the percentile axis ===")
ds = synth()
nodes = qn.fold_nodes(ds, hold_out=1995, n_nodes=99)
xq = pct_of(ds.x[ds.syear == 1995], PCT)
preds = {}
for name, cls in TRANSFORMS.items():
    t = cls().fit(nodes)
    p = t.predict_percentiles(xq)
    preds[name] = p
    d = np.diff(p, axis=0)
    check(f"{name:7s} non-decreasing", bool((d[np.isfinite(d)] >= -1e-9).all()))

print("\n=== 3. monotonicity audit of the raw fitted h (before repair) ===")
for name, cls in TRANSFORMS.items():
    t = cls().fit(nodes)
    n_bad = int((t.n_nonmono_ > 0).sum())
    print(f"  {name:7s} pixels with a non-monotone h: {n_bad:3d} / {nodes.n_pix}"
          f"   max decreasing steps: {int(t.n_nonmono_.max())}")

print("\n=== 4. COMMUTING PROPERTY: Q_p(h(x_days)) vs h(Q_p(x_days)) ===")
print("      Exact for the order-statistic quantile (method='inverted_cdf').")
print("      NOT exact under numpy's default interpolating estimator, because")
print("      h((a+b)/2) != (h(a)+h(b))/2 unless h is affine.  The pipeline")
print("      therefore maps DAILY values and takes percentiles afterwards.")
x_test_days = ds.x[ds.syear == 1995].astype(np.float64)


def order_stat(a, q):
    return np.percentile(a, np.asarray(q, float), axis=0, method="inverted_cdf")


for name, cls in TRANSFORMS.items():
    t = cls().fit(nodes)
    exact = np.nanmax(np.abs(
        t.predict(order_stat(x_test_days, PCT)) - order_stat(t.predict(x_test_days), PCT)
    ))
    interp = np.nanmax(np.abs(
        t.predict(pct_of(x_test_days, PCT)) - pct_of(t.predict(x_test_days), PCT)
    ))
    check(f"{name:7s} exact under order statistic", exact < 1e-9,
          f"{exact:.2e}   (interpolated gap would be {interp:.3f} C)")

print("\n=== 5. identity trap: in-sample percentile error of QUANT ~ 0 ===")
nodes_all = qn.fold_nodes(ds, hold_out=None, n_nodes=99)
t = TRANSFORMS["quant"]().fit(nodes_all)
yq_all = pct_of(ds.y, PCT)
xq_all = pct_of(ds.x, PCT)
in_mae = np.nanmean(np.abs(t.predict_percentiles(xq_all) - yq_all))
out_mae = np.nanmean(np.abs(preds["quant"] - pct_of(ds.y[ds.syear == 1995], PCT)))
check("in-sample MAE is near zero", in_mae < 0.35, f"{in_mae:.4f} C")
check("out-of-sample MAE is larger", out_mae > in_mae, f"out {out_mae:.4f} > in {in_mae:.4f}")

print("\n=== 6. extrapolation: values outside the training range are handled ===")
t = TRANSFORMS["quant"]().fit(nodes)
far = np.stack([nodes.x_lo - 10.0, nodes.x_hi + 10.0]).astype(np.float64)
out = t.predict_percentiles(far)
check("finite outside the fitted range", bool(np.isfinite(out).all()))
check("extrapolation counted", int(t.n_extrap_lo_.sum()) > 0 and int(t.n_extrap_hi_.sum()) > 0,
      f"lo={int(t.n_extrap_lo_.sum())} hi={int(t.n_extrap_hi_.sum())}")

print("\n=== 7. ties are survivable (the spline crashes without de-tying) ===")
ds_tie = synth(seed=1)
ds_tie.x[:, 0] = 25.0          # one pixel entirely constant
ds_tie.x[:100, 1] = 25.0       # one pixel partly constant
nodes_tie = qn.fold_nodes(ds_tie, hold_out=1995, n_nodes=99)
check("constant pixel marked invalid", not bool(nodes_tie.valid[0]),
      f"n_distinct={int(nodes_tie.n_distinct[0])}")
check("x still strictly increasing", bool((np.diff(nodes_tie.x, axis=0) > 0).all()))
for name in ("ssplin", "quant", "poly"):
    try:
        p = TRANSFORMS[name]().fit(nodes_tie).predict_percentiles(pct_of(ds_tie.x, PCT))
        ok = bool(np.isnan(p[:, 0]).all())     # invalid pixel must be NaN
        check(f"{name:7s} survives ties, invalid pixel NaN", ok)
    except Exception as e:                     # noqa: BLE001
        check(f"{name:7s} survives ties", False, f"{type(e).__name__}: {e}")

print("\n=== 8. hyper-parameters are validated ===")
try:
    TRANSFORMS["poly"](degree=3)
    check("poly accepts degree=3", True)
except Exception as e:                         # noqa: BLE001
    check("poly accepts degree=3", False, str(e))
try:
    TRANSFORMS["linear"](degree=3)
    check("linear rejects an unknown hyper-parameter", False)
except ValueError:
    check("linear rejects an unknown hyper-parameter", True)

print("\n" + "=" * 62)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): " + "; ".join(FAILS))
    sys.exit(1)
print("All sanity checks passed.")
