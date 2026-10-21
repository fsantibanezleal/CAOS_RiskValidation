"""The granularity adjustment of Gordy and Lutkebohmert (2013), dossier 04 H.5 (ladder rung C5).

Gordy, M.B. and Lutkebohmert, E. (2013), Granularity adjustment for regulatory capital assessment, International
Journal of Central Banking 9(3):33-48,
https://www.ijcb.org/sites/default/files/journal/v9n3/ijcb-v9n3-granularity-adjustment-regulatory-capital-assessment.pdf

The systematic factor of the single-factor CreditRisk+ model is X ~ Gamma with mean 1 and variance 1/xi (shape xi,
rate xi; paper p. 43), which reproduces the paper's test vector delta = 4.83 at xi = 0.25 and q = 0.999. The V[LGD] of
eq. 16 is gamma E[LGD] (1 - E[LGD]).

The adjustment (eq. 9, p. 41) is GA = alpha_q(L_n) - mu(alpha_q(X)), the gap of the portfolio loss rate L_n = sum_i
s_i U_i (a loss per unit of EAD) above the infinitely granular portfolio, so it is already a share of EAD and is NOT
scaled by K*. The simplified form (eq. 17, p. 46) is

    GA = (1 / (2 K*)) sum_i s_i^2 C_i (delta (K_i + R_i) - K_i),   C_i = (V[LGD_i] + E[LGD_i]^2) / E[LGD_i]  (eq. 11),
    R_i = E[LGD_i] PD_i,   K* = sum_i s_i K_i,   delta = (alpha_q - 1)(xi + (1 - xi) / alpha_q)  (eq. 15).

K_i is an input, as in the paper's regulatory algorithm (p. 47, step iv: the UL capital requirement from the IRB
formula); eq. 13 (p. 45), K_i = E[LGD_i] PD_i omega_i (alpha_q - 1), is `k_creditrisk_plus` for a caller that has
omega_i and no IRB capital. The exact adjustment (eq. 14) with second-order terms is not implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from riskvalidation.portfolio.portfolio import float_vector


def _check_xi_q(xi: float, q: float) -> tuple[float, float]:
    x, p = float(xi), float(q)
    if not (math.isfinite(x) and x > 0.0):
        raise ValueError(f"xi must be positive, got {xi}")
    if not (math.isfinite(p) and 0.0 < p < 1.0):
        raise ValueError(f"q must lie in (0, 1), got {q}")
    return x, p


def gamma_quantile(xi: float, q: float) -> float:
    """alpha_q(X): the q-quantile of X ~ Gamma(shape xi, rate xi) (mean 1, variance 1/xi)."""
    x, p = _check_xi_q(xi, q)
    return float(stats.gamma.ppf(p, a=x, scale=1.0 / x))


def delta(xi: float, q: float) -> float:
    """delta = (alpha_q - 1)(xi + (1 - xi) / alpha_q) with alpha_q = gamma_quantile(xi, q) (eq. 15)."""
    x, p = _check_xi_q(xi, q)
    a = gamma_quantile(x, p)
    return (a - 1.0) * (x + (1.0 - x) / a)


def lgd_variance(mean_lgd: np.ndarray, gamma: float) -> np.ndarray:
    """V[LGD] = gamma E[LGD] (1 - E[LGD]) (eq. 16); zero at E[LGD] equal to 0 or 1."""
    g = float(gamma)
    if not (math.isfinite(g) and g >= 0.0):
        raise ValueError("gamma must be a non-negative number")
    m = np.array(mean_lgd, dtype=float)
    if not np.all(np.isfinite(m)) or np.any(m < 0.0) or np.any(m > 1.0):
        raise ValueError("E[LGD] must lie in [0, 1]")
    return g * m * (1.0 - m)


def k_creditrisk_plus(pd: np.ndarray, lgd_mean: np.ndarray, omega: np.ndarray, alpha_q: float) -> np.ndarray:
    """K_i = E[LGD_i] PD_i omega_i (alpha_q - 1) (eq. 13): the UL capital per unit of EAD of the single-factor
    CreditRisk+ model.

    Equal to E[LGD_i] (pi_i(alpha_q) - PD_i) with the conditional PD pi_i(x) = PD_i (1 + omega_i (x - 1)); omega_i in
    [0, 1] is the weight of name i on the systematic factor.
    """
    p = float_vector(pd, "pd")
    lg = float_vector(lgd_mean, "lgd_mean")
    om = float_vector(omega, "omega")
    a = float(alpha_q)
    if not (p.size == lg.size == om.size):
        raise ValueError("pd, lgd_mean and omega must have the same length")
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("pd must lie strictly between 0 and 1")
    if np.any(lg < 0.0) or np.any(lg > 1.0):
        raise ValueError("lgd_mean must lie in [0, 1]")
    if np.any(om < 0.0) or np.any(om > 1.0):
        raise ValueError("omega must lie in [0, 1]")
    if not (math.isfinite(a) and a > 0.0):
        raise ValueError("alpha_q must be positive")
    return lg * p * om * (a - 1.0)


@dataclass(frozen=True)
class GranularityResult:
    """ga: the adjustment of eq. 17 as a share of EAD (eq. 9: not scaled by K*); k_star = sum_i s_i K_i, a share of EAD;
    the delta and alpha_q used; c the C_i of eq. 11 per name."""

    ga: float
    k_star: float
    delta: float
    alpha_q: float
    c: np.ndarray


def granularity_adjustment(share: np.ndarray, k: np.ndarray, pd: np.ndarray, lgd_mean: np.ndarray, lgd_gamma: float,
                           xi: float, q: float = 0.999) -> GranularityResult:
    """The simplified granularity adjustment of eq. 17 for exposure shares `share` (summing to 1) and the UL capital `k`
    per unit of EAD of each name (the IRB capital in the regulatory algorithm, or `k_creditrisk_plus`).

    For n identical names ga * n is constant (ga is proportional to 1 / n) and ga tends to 0 as n grows. Raises
    ValueError for shares that do not sum to 1 within 1e-9, a negative k, an expected LGD of 0 (C_i is undefined) and
    K* = 0.
    """
    s = float_vector(share, "share")
    kk = float_vector(k, "k")
    p = float_vector(pd, "pd")
    lg = float_vector(lgd_mean, "lgd_mean")
    if not (s.size == kk.size == p.size == lg.size):
        raise ValueError("share, k, pd and lgd_mean must have the same length")
    if np.any(s < 0.0) or abs(float(s.sum()) - 1.0) > 1e-9:
        raise ValueError("share must be nonnegative and sum to 1 within 1e-9")
    if np.any(kk < 0.0):
        raise ValueError("k must be nonnegative")
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("pd must lie strictly between 0 and 1")
    if np.any(lg <= 0.0) or np.any(lg > 1.0):
        raise ValueError("lgd_mean must lie in (0, 1]: C_i divides by E[LGD_i]")
    x, level = _check_xi_q(xi, q)
    k_star = float(s @ kk)
    if not k_star > 0.0:
        raise ValueError("K* = sum s_i K_i is zero: the adjustment is undefined")
    alpha = gamma_quantile(x, level)
    d = delta(x, level)
    c = (lgd_variance(lg, lgd_gamma) + lg**2) / lg
    r = lg * p
    ga = float(np.sum(s * s * c * (d * (kk + r) - kk))) / (2.0 * k_star)
    return GranularityResult(ga=ga, k_star=k_star, delta=d, alpha_q=alpha, c=c)

__all__ = [
    "GranularityResult",
    "delta",
    "gamma_quantile",
    "granularity_adjustment",
    "k_creditrisk_plus",
    "lgd_variance",
]
