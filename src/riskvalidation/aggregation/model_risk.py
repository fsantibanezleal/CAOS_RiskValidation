"""The model risk measures of dossier 06 B.12 over the set of aggregation models of the band: sensitivity information,
never a
regulatory capital figure (dossier 06 l.699-704).

- Barrieu and Scandolo (2015), "Assessing financial model risk", European Journal of Operational Research
242(2):546-556:
  the add-on multiplier AM = max rho / rho(X0) - 1 ("generalizes the Basel multiplier") and the relative measure RM =
  (max rho - rho(X0)) / (max rho - min rho) (dossier 06 l.692-698).
- Glasserman and Xu (2014), "Robust risk measurement and model risk", Quantitative Finance 14(1):29-58: the worst case
of
  E[m V] over the likelihood ratios m with relative entropy E[m log m] <= eta is attained at m_theta = exp(theta V) /
  E[exp(theta V)], with theta > 0 set (by bisection) so that the entropy constraint binds (dossier 06 l.684-691); the
  worst
  case is computed by reweighting the baseline sample.

Contract: `engine-aggregation-api` 1.0.0, module aggregation.model_risk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from riskvalidation.portfolio.portfolio import float_vector

MAX_DOUBLINGS = 200
MAX_BISECTIONS = 300


@dataclass(frozen=True)
class ModelRisk:
    """`rho_x0` the risk measure under the reference model, `rho_max` and `rho_min` over the set (money), `am` =
    rho_max /
    rho_x0 - 1 and `rm` = (rho_max - rho_x0) / (rho_max - rho_min) in [0, 1] (decimals)."""

    rho_x0: float
    rho_max: float
    rho_min: float
    am: float
    rm: float


def barrieu_scandolo(rhos: Sequence[float], reference_index: int) -> ModelRisk:
    """AM and RM of the risk measures `rhos` (money, one per model of the set, the band rows) with the reference model
    at
    `reference_index`. RM is 0 when rho_max = rho_min (this contract's convention for the undefined 0 / 0; UNVERIFIED in
    the paper). Golden: (100, 130, 90) with reference 0 gives AM 0.3 and RM 0.75."""
    x = float_vector(rhos, "rhos")
    if isinstance(reference_index, bool) or not isinstance(reference_index, (int, np.integer)):
        raise ValueError(f"reference_index must be an integer, got {reference_index!r}")
    if not 0 <= reference_index < x.size:
        raise ValueError(f"reference_index must lie in [0, {x.size - 1}], got {reference_index}")
    rho_x0 = float(x[reference_index])
    if not rho_x0 > 0.0:
        raise ValueError(f"rho_x0 (the risk measure under the reference model) must be positive, got {rho_x0}")
    rho_max, rho_min = float(x.max()), float(x.min())
    span = rho_max - rho_min
    rm = 0.0 if span == 0.0 else (rho_max - rho_x0) / span
    return ModelRisk(rho_x0=rho_x0, rho_max=rho_max, rho_min=rho_min, am=rho_max / rho_x0 - 1.0, rm=rm)


@dataclass(frozen=True)
class GlassermanXu:
    """`theta` the tilt (in the inverse units of V), `relative_entropy` E[m log m] at theta (equal to the budget to
    `tol`),
    `baseline` = E[V] and `worst_case` = E[m V] in the units of V."""

    theta: float
    relative_entropy: float
    baseline: float
    worst_case: float


def _entropy(x: np.ndarray, theta: float) -> tuple[float, np.ndarray]:
    """The relative entropy E[m log m] of m = exp(theta x) / E exp(theta x) for x <= 0 (stable), and m."""
    log_w = theta * x
    w = np.exp(log_w)
    mean_w = float(w.mean())
    m = w / mean_w
    return float(np.mean(m * (log_w - math.log(mean_w)))), m


def glasserman_xu_worst_case(v: np.ndarray, budget: float, tol: float = 1e-10) -> GlassermanXu:
    """sup over m with E[m log m] <= budget of E[m V] on the baseline sample `v` (S,), by the exponential tilt of the
    module
    docstring. `budget` > 0 is the relative entropy budget eta. Refused: a budget that is not attainable on the sample
    (the entropy of the tilt rises to log(S / k) with k the number of sample points at the maximum, at most log S), a
    constant V and a nonpositive budget. Golden: V = (0, 1) equally likely, theta = 1: m = (0.5379, 1.4621), entropy
    0.1109,
    worst case 0.7311 against the baseline 0.5; the budget 0.1109 recovers theta = 1."""
    x = float_vector(v, "v")
    if x.size < 2:
        raise ValueError("v needs at least 2 values")
    eta, eps = float(budget), float(tol)
    if not (math.isfinite(eta) and eta > 0.0):
        raise ValueError(f"budget must be positive, got {budget}")
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"tol must be positive, got {tol}")
    spread = float(np.ptp(x))
    if spread == 0.0:
        raise ValueError("v is constant: no tilt changes its mean")
    top = float(x.max())
    attainable = math.log(x.size / int(np.sum(x == top)))
    if eta >= attainable:
        raise ValueError(f"budget {eta} must be below the largest entropy attainable on the sample, log(S / k) = "
                         f"{attainable:.6g} (at most log S = {math.log(x.size):.6g})")
    scaled = (x - top) / spread  # in [-1, 0]
    hi = 1.0
    for _ in range(MAX_DOUBLINGS):
        if _entropy(scaled, hi)[0] >= eta:
            break
        hi *= 2.0
    else:
        raise ValueError(f"budget {eta} is not reached by any tilt of the sample")
    lo = 0.0
    theta_s, h, m = hi, 0.0, None
    for _ in range(MAX_BISECTIONS):
        theta_s = 0.5 * (lo + hi)
        h, m = _entropy(scaled, theta_s)
        if abs(h - eta) <= eps:
            break
        if h < eta:
            lo = theta_s
        else:
            hi = theta_s
    else:
        raise ValueError(f"the entropy constraint did not bind to tol {eps} (reached {h}, budget {eta})")
    assert m is not None
    return GlassermanXu(theta=theta_s / spread, relative_entropy=h, baseline=float(x.mean()),
                        worst_case=float(np.mean(m * x)))

__all__ = [
    "GlassermanXu",
    "ModelRisk",
    "barrieu_scandolo",
    "glasserman_xu_worst_case",
]
