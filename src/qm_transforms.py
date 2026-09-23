"""
qm_transforms.py — The quantile-mapping transforms, behind one interface.

Each class estimates a monotone transform ``h`` from an empirical q–q curve and
applies it to new predictor values.  The families follow Gudmundsson et al.
(2012), who sort every published variant into three groups:

===========================  ==========================================
distribution derived         ``NormalQM``
parametric                   ``LinearQM``, ``PolyQM``
non-parametric               ``QuantQM``, ``RQuantQM``, ``SSplinQM``
===========================  ==========================================

``IdentityQM`` is added as a seventh member.  It applies no correction at all,
so it carries the uncorrected bilinear predictor through the same metrics as
everything else.  Without it the comparison could not answer whether quantile
mapping helped.

Batched over pixels
-------------------
``fit`` takes a whole :class:`~src.qm_nodes.QQNodes` — that is, every pixel's
q–q curve at once — and ``predict`` takes and returns ``(n_q, n_pix)`` arrays.
Six of the seven transforms then reduce to a handful of numpy calls for the
whole domain.  A per-pixel interface would instead impose a ~300,000-iteration
Python loop on every method, including the ones whose arithmetic is free.  Only
``SSplinQM`` loops internally, and it says so through ``is_batched = False``.

Monotonicity
------------
A corrected percentile can be produced two ways: map the predictor's percentile,
``h(Q_x(q))``, or map every day and then take the percentile, ``Q_{h(x)}(q)``.
These agree **if and only if h is monotone increasing**.  That equivalence is
what lets this pipeline evaluate five numbers per pixel-season-fold instead of
~91 daily values, so monotonicity is a correctness requirement and not a matter
of taste.  It is handled in three layers:

1. :meth:`QQTransform.predict_percentiles` runs ``np.maximum.accumulate`` over
   the percentile axis.  No stored artefact can contain P25 above P50.  Plain
   :meth:`QQTransform.predict` imposes no ordering, so it stays usable for
   daily series.
2. ``RQuantQM`` and ``SSplinQM`` additionally re-sort their own node values,
   because smoothing a monotone sequence does not preserve monotonicity.
3. :meth:`QQTransform.fit` audits ``h`` on a dense grid and records
   ``n_nonmono`` per pixel, so a method that needs frequent repair is visible
   in the results rather than silently patched.

Sample used for fitting
-----------------------
The ``qmap`` package sets its reduction per family: ``fitQmapDIST`` and
``fitQmapPTF`` (distribution derived and parametric) default to
``qstep = NULL``, meaning every data point, while ``fitQmapQUANT``,
``fitQmapRQUANT`` and ``fitQmapSSPLIN`` default to ``qstep = 0.01``, meaning a
101-point percentile table.  ``fit_on_raw`` follows that split: ``NormalQM``,
``LinearQM`` and ``PolyQM`` are fitted on every training day, the three
non-parametric transforms on 99 nodes.

Measured on 641 pixels across all four seasons, refitting the node-based
methods on every training day instead changes MAE by at most 0.003 °C, and for
``RQuantQM`` and ``SSplinQM`` the fuller sample is very slightly *worse* — more
nodes give a smoother more noise to follow.  The cost is not symmetric: the
spline is ~5x slower on raw days, taking the full run from about 13 minutes to
100.

Extrapolation
-------------
The default for **every** method is the constant correction of Boé et al.
(2007): outside the raw training range, carry the offset found at the nearest
end, ``h(x) = x + (h(x_hi) - x_hi)``.  It is applied uniformly so the methods
are compared on equal terms, and it matters most for the parametric families —
a degree-5 polynomial diverges sharply just outside the range it was fitted on.  With nine training years, a held-out
year's P5 and P90 land outside that range often enough that the choice shows up
in the results, so ``n_extrap_lo``/``n_extrap_hi`` are recorded and reported.

Typical usage
-------------
from src.qm_transforms import TRANSFORMS
t = TRANSFORMS["quant"]().fit(nodes)
yhat = t.predict_percentiles(x_percentiles)   # (n_q, n_pix), order guaranteed
hd  = t.predict(x_daily)                      # (n_days, n_pix), order untouched
"""

from __future__ import annotations

import abc
import logging
from typing import Any, ClassVar

import numpy as np
from scipy.interpolate import make_smoothing_spline

from .qm_nodes import QQNodes, node_probs

__all__ = [
    "QQTransform",
    "IdentityQM",
    "NormalQM",
    "LinearQM",
    "PolyQM",
    "QuantQM",
    "RQuantQM",
    "SSplinQM",
    "TRANSFORMS",
    "METHOD_NAMES",
    "RAW_METHOD",
    "HYPER_GRIDS",
]

log = logging.getLogger(__name__)

#: Points on the audit grid used to check monotonicity of a fitted h.
_AUDIT_POINTS = 201

#: Fraction of the training range added either side of the audit grid.
#: Zero, because under the default ``"boe"`` and ``"clip"`` rules the fitted
#: curve is never evaluated outside ``[x_lo, x_hi]`` — the input is clipped
#: first and the offset carried.  Auditing beyond the range would report
#: non-monotonicity in extrapolated cubic tails that no prediction ever uses,
#: which is noise rather than a finding.
_AUDIT_MARGIN = 0.0

#: The uncorrected baseline's name.
RAW_METHOD = "raw"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class QQTransform(abc.ABC):
    """A monotone transform estimated from a batch of per-pixel q–q curves.

    Subclasses implement :meth:`_fit` and :meth:`_raw_predict`.  Everything
    that must hold for *every* method — extrapolation, the monotonicity
    guarantee, the monotonicity audit, invalid-pixel masking — lives here, so
    it cannot be forgotten in one subclass and not another.
    """

    name: ClassVar[str] = ""
    is_batched: ClassVar[bool] = True
    hp_grid: ClassVar[dict[str, list]] = {}
    #: Value used when a hyper-parameter is not supplied and not searched.
    #: Kept separate from ``hp_grid`` deliberately: the grid is ordered for
    #: reading, so its first entry is its smallest value, which is not
    #: necessarily a sensible default.  Taking ``hp_grid[k][0]`` gave
    #: ``QuantQM()`` a 4-node lookup table once the grid was widened downward.
    hp_default: ClassVar[dict[str, object]] = {}
    #: Fit on every training day rather than on a reduced node table.
    #: Set per family to match the ``qmap`` package: its distribution-derived
    #: and parametric fitters default to ``qstep = NULL`` (all data), while its
    #: three non-parametric fitters default to ``qstep = 0.01`` (101 nodes).
    fit_on_raw: ClassVar[bool] = False
    n_params: ClassVar[int] = 0

    def __init__(self, extrap: str = "boe", **hp: Any) -> None:
        if extrap not in ("boe", "clip", "native"):
            raise ValueError(f"Unknown extrap {extrap!r}.")
        unknown = set(hp) - set(self.hp_grid)
        if unknown:
            raise ValueError(
                f"{type(self).__name__} got unknown hyper-parameter(s) {sorted(unknown)}; "
                f"expected a subset of {sorted(self.hp_grid)}."
            )
        self.extrap = extrap
        self.hp = {
            k: hp.get(k, self.hp_default.get(k, v[-1]))
            for k, v in self.hp_grid.items()
        }
        self._fitted = False

    # -- public ------------------------------------------------------------

    def fit(self, nodes: QQNodes) -> "QQTransform":
        """Estimate ``h`` for every pixel in *nodes*."""
        self.x_lo_ = nodes.x_lo.astype(np.float64)
        self.x_hi_ = nodes.x_hi.astype(np.float64)
        self.valid_ = nodes.valid.copy()
        self.n_train_ = nodes.n_train.copy()
        self.n_pix_ = nodes.n_pix
        self.converged_ = np.ones(nodes.n_pix, dtype=bool)
        self.n_extrap_lo_ = np.zeros(nodes.n_pix, dtype=np.int16)
        self.n_extrap_hi_ = np.zeros(nodes.n_pix, dtype=np.int16)

        self._fit(nodes)
        self._fitted = True
        self.n_nonmono_ = self._audit_monotonicity()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Apply ``h`` to arbitrary predictor values.

        Use this for daily values, or anywhere the rows of *x* are not in
        ascending order.  No ordering is assumed and none is imposed, so the
        output is ``h`` exactly as fitted (plus the extrapolation rule).

        Parameters
        ----------
        x : np.ndarray
            Shape ``(n, n_pix)``.  Rows may be in any order.

        Returns
        -------
        np.ndarray
            Same shape, float64, NaN at invalid pixels.
        """
        if not self._fitted:
            raise RuntimeError("predict() called before fit().")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_pix_:
            raise ValueError(f"x must be (n, {self.n_pix_}); got {x.shape}.")
        out = self._apply_extrapolation(x)
        out[:, ~self.valid_] = np.nan
        return out

    def predict_percentiles(self, xq: np.ndarray) -> np.ndarray:
        """Map predictor percentiles to corrected percentiles.

        This is what the pipeline calls.  It adds the one guarantee that only
        makes sense for percentile inputs: the result is forced non-decreasing,
        so no stored artefact can hold a corrected P25 above a corrected P50.

        For a monotone ``h`` that repair is a no-op — which is the point.  It
        fires only where a fit misbehaved, and
        :attr:`n_nonmono_` records where that happened rather than letting the
        repair hide it.

        Parameters
        ----------
        xq : np.ndarray
            Shape ``(n_q, n_pix)``, non-decreasing down axis 0.  Percentiles of
            one sample satisfy this by construction.

        Returns
        -------
        np.ndarray
            Same shape, float64, non-decreasing down axis 0, NaN at invalid
            pixels.
        """
        xq = np.asarray(xq, dtype=np.float64)
        d = np.diff(xq, axis=0)
        if not (d[np.isfinite(d)] >= -1e-6).all():
            raise ValueError(
                "predict_percentiles() needs xq non-decreasing down axis 0; "
                "use predict() for unordered values such as daily series."
            )
        out = self.predict(xq)
        return np.maximum.accumulate(out, axis=0)

    def diagnostics(self) -> dict[str, np.ndarray]:
        """Per-pixel fit diagnostics, for the ``qq_fit_diag_*`` artefact."""
        d = {
            "n_train": self.n_train_,
            "converged": self.converged_,
            "n_nonmono": self.n_nonmono_,
            "n_extrap_lo": self.n_extrap_lo_,
            "n_extrap_hi": self.n_extrap_hi_,
            "x_lo": self.x_lo_.astype(np.float32),
            "x_hi": self.x_hi_.astype(np.float32),
            "valid": self.valid_,
        }
        d.update(self._param_diagnostics())
        return d

    # -- subclass contract -------------------------------------------------

    @abc.abstractmethod
    def _fit(self, nodes: QQNodes) -> None:
        """Estimate and store the per-pixel parameters."""

    @abc.abstractmethod
    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        """Apply ``h`` with no extrapolation rule and no monotone repair."""

    def _param_diagnostics(self) -> dict[str, np.ndarray]:
        """Fitted parameters to carry into the diagnostics table."""
        return {}

    # -- shared machinery --------------------------------------------------

    def _apply_extrapolation(self, x: np.ndarray) -> np.ndarray:
        """Handle values outside the raw training range."""
        below = x < self.x_lo_[None, :]
        above = x > self.x_hi_[None, :]
        self.n_extrap_lo_ = below.sum(axis=0).astype(np.int16)
        self.n_extrap_hi_ = above.sum(axis=0).astype(np.int16)

        if self.extrap == "native":
            return self._raw_predict(x)

        x_clipped = np.clip(x, self.x_lo_[None, :], self.x_hi_[None, :])
        out = self._raw_predict(x_clipped)
        if self.extrap == "clip":
            return out

        # "boe": carry the offset found at the nearest fitted end.
        ends = self._raw_predict(np.stack([self.x_lo_, self.x_hi_]))
        off_lo = ends[0] - self.x_lo_
        off_hi = ends[1] - self.x_hi_
        out = np.where(below, x + off_lo[None, :], out)
        out = np.where(above, x + off_hi[None, :], out)
        return out

    def _audit_monotonicity(self) -> np.ndarray:
        """Count decreasing steps of ``h`` on a dense grid, per pixel.

        One batched evaluation, so the cost is negligible.  This is the only
        way to know whether the monotone repair in :meth:`predict` is masking a
        badly behaved fit.
        """
        span = np.where(self.x_hi_ > self.x_lo_, self.x_hi_ - self.x_lo_, 1.0)
        lo = self.x_lo_ - _AUDIT_MARGIN * span
        hi = self.x_hi_ + _AUDIT_MARGIN * span
        t = np.linspace(0.0, 1.0, _AUDIT_POINTS)[:, None]
        grid = lo[None, :] + t * (hi - lo)[None, :]
        with np.errstate(all="ignore"):
            curve = self._raw_predict(grid)
        d = np.diff(curve, axis=0)
        bad = np.isfinite(d) & (d < -1e-9)
        return bad.sum(axis=0).astype(np.int16)


# ---------------------------------------------------------------------------
# Helpers shared by several transforms
# ---------------------------------------------------------------------------

def _interp_columns(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Column-wise :func:`np.interp`.

    ``x`` is ``(n_q, n_pix)``; ``xp``/``fp`` are ``(n_nodes, n_pix)``.  numpy has
    no batched ``interp``, so this loops over pixels — but it runs once per
    ``predict`` call rather than once per fit, and measures ~20 ms for the whole
    domain, which is why the lookup methods still count as batched.
    """
    out = np.empty_like(x, dtype=np.float64)
    for p in range(x.shape[1]):
        out[:, p] = np.interp(x[:, p], xp[:, p], fp[:, p])
    return out


def _anchor_table(
    xn: np.ndarray, yn: np.ndarray,
    x_lo: np.ndarray, x_hi: np.ndarray,
    y_lo: np.ndarray, y_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the training extremes as the outermost rows of a lookup table.

    ``np.interp`` returns the end value for anything beyond its abscissa, so an
    unanchored percentile table maps everything below its first node to a single
    value.  With a coarse table that flat region covers a large share of the
    distribution and biases the tails badly.  Anchoring makes the outer segments
    interpolate instead.

    The anchors are nudged outward where a raw extreme has collapsed onto the
    first node, so the abscissa stays strictly increasing for the interpolator.
    """
    eps = np.maximum(1e-6 * np.abs(x_hi - x_lo), 1e-6)
    lo = np.minimum(x_lo, xn[0] - eps)
    hi = np.maximum(x_hi, xn[-1] + eps)
    xa = np.vstack([lo, xn, hi])
    ya = np.vstack([np.minimum(y_lo, yn[0]), yn, np.maximum(y_hi, yn[-1])])
    return xa, np.maximum.accumulate(ya, axis=0)


def _ols_columns(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column simple OLS slope and intercept, fully vectorised."""
    n = x.shape[0]
    sx = x.sum(axis=0)
    sy = y.sum(axis=0)
    sxx = (x * x).sum(axis=0)
    sxy = (x * y).sum(axis=0)
    denom = n * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(denom != 0, (n * sxy - sx * sy) / denom, 0.0)
    a = (sy - b * sx) / n
    return a, b


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class IdentityQM(QQTransform):
    """No correction: ``h(x) = x``.  The uncorrected-predictor baseline."""

    name = "raw"
    is_batched = True

    def _fit(self, nodes: QQNodes) -> None:
        pass

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        return x.copy()

    def _apply_extrapolation(self, x: np.ndarray) -> np.ndarray:
        # There is no fitted range to fall outside of.
        self.n_extrap_lo_ = np.zeros(self.n_pix_, dtype=np.int16)
        self.n_extrap_hi_ = np.zeros(self.n_pix_, dtype=np.int16)
        return x.copy()


class NormalQM(QQTransform):
    """Distribution derived, assuming both sides are normal.

    ``h(x) = F_y^-1(F_x(x))`` has the closed form
    ``mu_y + (sigma_y / sigma_x) (x - mu_x)`` when both distributions are
    normal, so no numerical CDF inversion is needed.

    The moments come from the **raw** training days, not the nodes: linear
    interpolation between quantile nodes shortens the tails a little, which
    would bias ``sigma`` low.

    Note this makes ``h`` affine, so results will sit close to
    :class:`LinearQM`.  That is expected for near-Gaussian daily temperature,
    not a sign of duplicated code — the two differ in what they optimise, one
    matching moments and the other minimising squared error on the q–q nodes.
    """

    name = "normal"
    fit_on_raw = True
    is_batched = True
    n_params = 2

    def _fit(self, nodes: QQNodes) -> None:
        sx = nodes.x_sd.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.b_ = np.where(sx > 0, nodes.y_sd.astype(np.float64) / sx, 1.0)
        self.a_ = nodes.y_mean.astype(np.float64) - self.b_ * nodes.x_mean.astype(np.float64)

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        return self.a_[None, :] + self.b_[None, :] * x

    def _param_diagnostics(self) -> dict[str, np.ndarray]:
        return {"p_a": self.a_.astype(np.float32), "p_b": self.b_.astype(np.float32)}


class LinearQM(QQTransform):
    """Parametric, ``y = a + b x`` fitted by least squares on the q–q nodes."""

    name = "linear"
    fit_on_raw = True
    is_batched = True
    n_params = 2

    def _fit(self, nodes: QQNodes) -> None:
        self.a_, self.b_ = _ols_columns(
            nodes.x.astype(np.float64), nodes.y.astype(np.float64)
        )

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        return self.a_[None, :] + self.b_[None, :] * x

    def _param_diagnostics(self) -> dict[str, np.ndarray]:
        return {"p_a": self.a_.astype(np.float32), "p_b": self.b_.astype(np.float32)}


class PolyQM(QQTransform):
    """Parametric polynomial of a chosen degree, fitted on the q–q nodes.

    The abscissa is centred and scaled before the design matrix is built.  Raw
    °C sits near 20, so a degree-5 term reaches ~3e6 and the Vandermonde
    condition number ~1e9; centring removes that entirely.  A tiny ridge keeps
    the batched solve from failing on a near-singular pixel.

    The degree grid starts at 2 because degree 1 *is* :class:`LinearQM`, and
    reporting the same estimator twice under two names would be misleading.
    """

    name = "poly"
    fit_on_raw = True
    is_batched = True
    hp_grid = {"degree": [2, 3, 4, 5]}
    hp_default = {"degree": 3}

    _RIDGE = 1e-10

    def _fit(self, nodes: QQNodes) -> None:
        deg = int(self.hp["degree"])
        x = nodes.x.astype(np.float64)
        y = nodes.y.astype(np.float64)

        self.x_c_ = x.mean(axis=0)
        sd = x.std(axis=0)
        self.x_s_ = np.where(sd > 0, sd, 1.0)
        z = (x - self.x_c_[None, :]) / self.x_s_[None, :]

        # Design matrix per pixel: (n_pix, n_nodes, deg+1)
        V = np.stack([z ** k for k in range(deg + 1)], axis=-1)
        V = np.moveaxis(V, 1, 0)
        yv = np.moveaxis(y, 1, 0)[:, :, None]

        gram = V.transpose(0, 2, 1) @ V
        rhs = V.transpose(0, 2, 1) @ yv
        gram += self._RIDGE * np.eye(deg + 1)[None, :, :]
        try:
            coef = np.linalg.solve(gram, rhs)[:, :, 0]
        except np.linalg.LinAlgError:
            # One singular pixel would otherwise abort the whole batch.
            coef = np.empty((x.shape[1], deg + 1))
            for p in range(x.shape[1]):
                coef[p], *_ = np.linalg.lstsq(V[p], yv[p], rcond=None)
                self.converged_[p] = True
            log.warning("PolyQM: fell back to per-pixel lstsq for a singular batch.")
        self.coef_ = coef.T  # (deg+1, n_pix), ascending power

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.x_c_[None, :]) / self.x_s_[None, :]
        out = np.zeros_like(z)
        for k in range(self.coef_.shape[0] - 1, -1, -1):
            out = out * z + self.coef_[k][None, :]
        return out

    def _param_diagnostics(self) -> dict[str, np.ndarray]:
        d = {f"p_c{k}": self.coef_[k].astype(np.float32)
             for k in range(self.coef_.shape[0])}
        d["p_degree"] = np.full(self.n_pix_, int(self.hp["degree"]), dtype=np.int16)
        return d


class QuantQM(QQTransform):
    """Empirical quantile mapping — the QUANT method.

    The q–q relation is stored as a table of empirical percentiles and read
    with linear interpolation between them, following Boé et al. (2007).  This
    is the classical ``F_obs^-1(F_mod(x))`` with both CDFs empirical, and it is
    monotone by construction.

    ``n_nodes`` is this method's own resolution dial, so the table is rebuilt
    here at the requested resolution from the nodes supplied, rather than
    reusing them as given.

    The table is **anchored at the training extremes**.  Without that anchor
    the bottom and top segments clamp flat, because ``np.interp`` returns the
    end value for anything outside the node range: with 5 nodes the lowest sits
    at the 10th percentile, so the coldest tenth of the distribution would all
    map to a single value.  Measured on this domain that produced a **+1.8 °C**
    bias at SON P5 — worse than applying no correction at all — since P5 fell
    inside the clamped region.  Adding ``(x_lo, y_lo)`` and ``(x_hi, y_hi)``
    makes the outer segments proper interpolations, and matches how empirical
    quantile mapping is normally implemented.

    The node grid starts at 11 rather than lower.  A table of *n* nodes places
    its outermost at probabilities ``0.5/n`` and ``1 - 0.5/n``, so it can only
    resolve a percentile that falls inside that span: 4 nodes reach no further
    than the 12.5th percentile and cannot represent P5 at all, leaving the
    boundary rule to invent it.  Requiring ``n >= 10`` keeps every evaluated
    percentile interior to the table.

    A note on what this cost, because the direction is surprising.  Before the
    anchor was added the unanchored table clamped flat below its first node, and
    that scored a *lower* MAE — 1.50 against 1.62 °C on the two hardest seasons.
    The reason is that clamping pulls the cold tail towards the middle of the
    training distribution, which is an accidental shrinkage estimator, and where
    year-to-year variability is as large as it is at the autumn cold tail
    shrinking towards climatology genuinely beats tracking the model. It also
    carried a **+1.8 °C** systematic bias at SON P5 — worse than applying no
    correction — which is not acceptable in a method whose purpose is removing
    bias, and which would compound in a projection. Correctness was taken over
    the 0.12 °C of MAE.
    """

    name = "quant"
    is_batched = True
    hp_grid = {"n_nodes": [11, 19, 49, 99]}
    hp_default = {"n_nodes": 99}

    def _fit(self, nodes: QQNodes) -> None:
        want = int(self.hp["n_nodes"])
        have = nodes.n_nodes
        if want == have:
            self.xn_ = nodes.x.astype(np.float64)
            self.yn_ = nodes.y.astype(np.float64)
        else:
            # Resample the supplied curve onto the requested node count.  The
            # supplied nodes are themselves quantiles, so interpolating them at
            # the coarser probabilities is the same operation one step removed.
            src_p = node_probs(have)
            dst_p = node_probs(want)
            self.xn_ = np.empty((want, nodes.n_pix))
            self.yn_ = np.empty((want, nodes.n_pix))
            for p in range(nodes.n_pix):
                self.xn_[:, p] = np.interp(dst_p, src_p, nodes.x[:, p])
                self.yn_[:, p] = np.interp(dst_p, src_p, nodes.y[:, p])
            self.xn_ = np.maximum.accumulate(self.xn_, axis=0)
            self.yn_ = np.maximum.accumulate(self.yn_, axis=0)
        self._n_nodes_used = self.xn_.shape[0]
        self.xn_, self.yn_ = _anchor_table(
            self.xn_, self.yn_,
            nodes.x_lo.astype(np.float64), nodes.x_hi.astype(np.float64),
            nodes.y_lo.astype(np.float64), nodes.y_hi.astype(np.float64),
        )

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        return _interp_columns(x, self.xn_, self.yn_)

    def _param_diagnostics(self) -> dict[str, np.ndarray]:
        return {"p_n_nodes": np.full(self.n_pix_, self._n_nodes_used, dtype=np.int16)}


class RQuantQM(QQTransform):
    """Robust empirical quantiles — the RQUANT method of the ``qmap`` package.

    Same node table as :class:`QuantQM`, but each node's value is smoothed
    against its neighbours by local linear regression, so a node no longer
    rests only on the handful of days nearest its own probability.

    Because the nodes sit at fixed probability positions, the smoother is a
    fixed ``(n_nodes, n_nodes)`` hat matrix and the whole domain is one matrix
    product.  Smoothing can break monotonicity, so the smoothed nodes are
    re-sorted afterwards.

    The smoother is applied to the **residual from the linear fit**, and the
    linear part is added back.  Smoothing ``y`` directly against the node index
    would bias the result wherever the q–q curve is bent: quantiles of a normal
    sample trace an S-shape against probability, and a local linear smoother
    cuts the corners off it.  Measured on a perfect ``y = x`` relation, direct
    smoothing left a 0.12 °C error where it should have left none.  Taking out
    the linear trend first makes the method exact for any linear relation — the
    common case for temperature — while still smoothing the departures from it.

    Note this method is **not** in Gudmundsson et al. (2012) — it is a third
    non-parametric option in the same authors' ``qmap`` R package, and should be
    cited to the package rather than the paper.
    """

    name = "rquant"
    is_batched = True
    hp_grid = {"wsize": [3, 5, 9, 15, 25, 41, 65]}
    hp_default = {"wsize": 9}

    def _fit(self, nodes: QQNodes) -> None:
        x = nodes.x.astype(np.float64)
        y = nodes.y.astype(np.float64)
        self.xn_ = x
        a, b = _ols_columns(x, y)
        trend = a[None, :] + b[None, :] * x
        S = self._hat_matrix(nodes.n_nodes, int(self.hp["wsize"]))
        self.yn_ = np.maximum.accumulate(trend + S @ (y - trend), axis=0)
        # Anchored for the same reason as QuantQM: an unanchored table clamps
        # flat below its first node and above its last.
        self.xn_, self.yn_ = _anchor_table(
            self.xn_, self.yn_,
            nodes.x_lo.astype(np.float64), nodes.x_hi.astype(np.float64),
            nodes.y_lo.astype(np.float64), nodes.y_hi.astype(np.float64),
        )

    @staticmethod
    def _hat_matrix(n: int, wsize: int) -> np.ndarray:
        """Local linear smoother on the node index, with a tricube kernel."""
        half = max(int(wsize) // 2, 1)
        t = np.arange(n, dtype=float)
        S = np.zeros((n, n))
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            u = (t[lo:hi] - t[i]) / (half + 1.0)
            w = (1.0 - np.abs(u) ** 3) ** 3
            w = np.maximum(w, 1e-12)
            # Weighted linear fit evaluated at t[i]; u is already centred there.
            sw = w.sum()
            su = (w * u).sum()
            suu = (w * u * u).sum()
            denom = sw * suu - su * su
            if abs(denom) < 1e-12:
                S[i, lo:hi] = w / sw
            else:
                # Intercept of the weighted fit in the centred coordinate.
                S[i, lo:hi] = (w * (suu - su * u)) / denom
        return S

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        return _interp_columns(x, self.xn_, self.yn_)

    def _param_diagnostics(self) -> dict[str, np.ndarray]:
        return {"p_wsize": np.full(self.n_pix_, int(self.hp["wsize"]), dtype=np.int16)}


class SSplinQM(QQTransform):
    """Cubic smoothing spline through the q–q relation — the SSPLIN method.

    Minimises ``sum (y_i - f(x_i))^2 + lam * integral f''^2``, so ``lam`` sets
    the trade-off between following the q–q cloud and ignoring its sampling
    noise.

    ``lam`` is **fixed per season** and chosen on a log-spaced grid by the same
    inner cross-validation that picks every other hyper-parameter here.

    Two alternatives were rejected.  Letting ``make_smoothing_spline`` choose it
    by generalised cross-validation on each fit (``lam=None``) costs 27x more —
    measured at 30 ms against 1.1 ms per fit, which is 2.5 hours against 5.6
    minutes over the full domain — and gives every pixel its own amount of
    smoothing, which cannot be described in a methods section.  It is also
    unusable in practice: scipy returns a plain ``BSpline`` that does not carry
    the chosen ``lam``, so the value cannot be recovered and reported.

    Choosing ``lam`` on held-out years instead of by GCV is the better criterion
    on this data regardless of cost.  GCV assumes independent residuals, while
    daily temperature is autocorrelated over roughly a week — the effective
    sample size behind ~820 training days is nearer 150-250 — so GCV would
    systematically under-smooth.  Held-out prediction error carries no such
    assumption.

    The grid spans 1e-2 to 1e2.  Measured at the extremes, the mean absolute
    residual at the nodes runs from 0.027 °C at ``lam=0.01`` to 0.100 °C at
    ``lam=100``, so the range covers genuinely different amounts of smoothing.
    """

    name = "ssplin"
    is_batched = False
    hp_grid = {"lam": [0.01, 0.1, 1.0, 10.0, 100.0]}
    hp_default = {"lam": 1.0}

    def _fit(self, nodes: QQNodes) -> None:
        lam = self.hp["lam"]
        x = nodes.x.astype(np.float64)
        y = nodes.y.astype(np.float64)
        self.splines_: list[Any] = [None] * nodes.n_pix
        self.lin_a_, self.lin_b_ = _ols_columns(x, y)

        for p in range(nodes.n_pix):
            if not nodes.valid[p]:
                self.converged_[p] = False
                continue
            try:
                self.splines_[p] = make_smoothing_spline(
                    x[:, p], y[:, p], lam=lam
                )
            except (ValueError, np.linalg.LinAlgError):
                self.converged_[p] = False

        n_fail = int((~self.converged_ & nodes.valid).sum())
        if n_fail:
            log.info(
                "SSplinQM season %s fold %s: %d spline fit(s) failed; using linear.",
                nodes.season, nodes.hold_out, n_fail,
            )

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float64)
        for p in range(x.shape[1]):
            spl = self.splines_[p]
            if spl is None:
                out[:, p] = self.lin_a_[p] + self.lin_b_[p] * x[:, p]
            else:
                out[:, p] = spl(x[:, p])
        return out

    def _param_diagnostics(self) -> dict[str, np.ndarray]:
        return {"p_lam": np.full(self.n_pix_, float(self.hp["lam"]), dtype=np.float32)}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Name -> class, in the order results should be reported.
TRANSFORMS: dict[str, type[QQTransform]] = {
    "raw": IdentityQM,
    "normal": NormalQM,
    "linear": LinearQM,
    "poly": PolyQM,
    "quant": QuantQM,
    "rquant": RQuantQM,
    "ssplin": SSplinQM,
}

#: The fitted methods, excluding the uncorrected baseline.
METHOD_NAMES: list[str] = [k for k in TRANSFORMS if k != RAW_METHOD]

#: Hyper-parameter search space per method; empty means nothing to choose.
HYPER_GRIDS: dict[str, dict[str, list]] = {
    k: dict(v.hp_grid) for k, v in TRANSFORMS.items()
}
