"""Rungs 3 and 4: delta-normal, Cornish-Fisher and EWMA (dossier 05 A.2 rows 4 to 6)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from riskvalidation.market.conventions import check_alpha, vec


def delta_normal_var_es(mu: float, sigma: float, alpha: float) -> tuple[float, float]:
    """`var = mu + Phi^-1(alpha) sigma`, `es = mu - sigma phi(Phi^-1(alpha)) / alpha` (A.2 row 4, return convention)."""
    alpha = check_alpha(alpha)
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be positive, got {sigma}")
    z = float(stats.norm.ppf(alpha))
    return float(mu) + z * float(sigma), float(mu) - float(sigma) * float(stats.norm.pdf(z)) / alpha


@dataclass(frozen=True)
class CornishFisher:
    var: float      # return convention
    z_cf: float     # the adjusted quantile
    valid: bool     # the validity domain holds


def cornish_fisher_var(mu: float, sigma: float, skew: float, excess_kurtosis: float, alpha: float) -> CornishFisher:
    """Cornish-Fisher VaR (A.2 row 5): `z_cf = z + (z^2-1) S/6 + (z^3-3z) Ke/24 - (2 z^3 - 5 z) S^2/36`, `var = mu +
    z_cf sigma`.

    Decision D-6: `valid` is true when the derivative of the map `z -> z_cf(z)`,
    `1 + 2 z S/6 + (3 z^2 - 3) Ke/24 - (6 z^2 - 5) S^2/36`, is positive at every point of `[Phi^-1(alpha), 0]` (a grid
    of 201 points): the expansion is then monotone over the tail it is used on.
    """
    alpha = check_alpha(alpha)
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be positive, got {sigma}")
    s, k = float(skew), float(excess_kurtosis)
    z = float(stats.norm.ppf(alpha))
    z_cf = z + (z**2 - 1.0) * s / 6.0 + (z**3 - 3.0 * z) * k / 24.0 - (2.0 * z**3 - 5.0 * z) * s**2 / 36.0
    grid = np.linspace(z, 0.0, 201)
    slope = 1.0 + 2.0 * grid * s / 6.0 + (3.0 * grid**2 - 3.0) * k / 24.0 - (6.0 * grid**2 - 5.0) * s**2 / 36.0
    return CornishFisher(var=float(mu) + z_cf * float(sigma), z_cf=float(z_cf), valid=bool(np.all(slope > 0.0)))


def ewma_variance(returns: np.ndarray, lam: float, initial_variance: float) -> np.ndarray:
    """One-step variances `(T+1,)`: element 0 the initial variance, element `t+1` the forecast after observing `r_t`.

    `sigma2_{t+1|t} = lam sigma2_{t|t-1} + (1 - lam) r_t^2` (A.2 row 6; the zero-mean assumption is part of the method).
    """
    r = vec(returns, "returns", allow_empty=True)
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must lie in (0, 1), got {lam}")
    if not (np.isfinite(initial_variance) and initial_variance > 0.0):
        raise ValueError(f"initial_variance must be positive, got {initial_variance}")
    out = np.empty(r.size + 1)
    out[0] = initial_variance
    s2 = float(initial_variance)
    for t in range(r.size):
        s2 = lam * s2 + (1.0 - lam) * float(r[t]) ** 2
        out[t + 1] = s2
    return out


def ewma_effective_days(lam: float, tolerance: float) -> float:
    """`K = ln(tolerance) / ln(lam)` (A.2 row 6): 74.4 for (0.94, 0.01) and 151.2 for (0.97, 0.01)."""
    if not (0.0 < lam < 1.0) or not (0.0 < tolerance < 1.0):
        raise ValueError("lam and tolerance must lie in (0, 1)")
    return math.log(tolerance) / math.log(lam)


def ewma_var_es(returns: np.ndarray, lam: float, alpha: float, initial_variance: float) -> tuple[float, float]:
    """Rung 4: the normal quantile with the last EWMA variance and a zero mean."""
    variance = float(ewma_variance(returns, lam, initial_variance)[-1])
    return delta_normal_var_es(0.0, math.sqrt(variance), alpha)
