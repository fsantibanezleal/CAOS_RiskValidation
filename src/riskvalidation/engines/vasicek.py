"""The one-factor (Vasicek) model and its large homogeneous portfolio limit.

Obligor i defaults when its asset return A_i = sqrt(rho) Z + sqrt(1 - rho) e_i falls below G(PD), with Z and e_i
independent standard normals (Vasicek 2002; Gordy 2003, doi:10.1016/S1042-9573(03)00040-8). Given the systematic
factor Z = z the default probability is

    p(z) = N((G(PD) - sqrt(rho) z) / sqrt(1 - rho)),

and in the infinitely granular portfolio the default fraction converges to p(Z), so

    P(L <= x) = N((sqrt(1 - rho) G(x) - G(PD)) / sqrt(rho)),   q-quantile N((G(PD) + sqrt(rho) G(q)) / sqrt(1 - rho)).

The IRB formula's conditional PD is this quantile at q = 0.999 (BCBS explanatory note, 2005). An observed default rate
DR implies the factor z = (G(PD) - sqrt(1 - rho) G(DR)) / sqrt(rho), the inverse of p(z): the credit-cycle index of a
through-the-cycle PD (dossier 04 C.1, the Z-factor of Belkin, Suchower and Forest 1998).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr, ndtri

VASICEK_2002 = "Vasicek (2002), Loan portfolio value, Risk, December 2002, 160-162"
GORDY_2003 = (
    "Gordy (2003), A risk-factor model foundation for ratings-based bank capital rules, "
    "doi:10.1016/S1042-9573(03)00040-8"
)


def _check(pd: float, rho: float) -> None:
    if not 0.0 < pd < 1.0:
        raise ValueError("pd must lie in (0, 1)")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie in (0, 1)")


def conditional_pd(pd: float, rho: float, z: float | np.ndarray) -> float | np.ndarray:
    """The PD given the systematic factor z (a low z is a bad state of the economy)."""
    _check(pd, rho)
    return ndtr((ndtri(pd) - math.sqrt(rho) * np.asarray(z, dtype=float)) / math.sqrt(1.0 - rho))


def lhp_cdf(x: float | np.ndarray, pd: float, rho: float) -> float | np.ndarray:
    """P(L <= x) for the default fraction L of the large homogeneous portfolio."""
    _check(pd, rho)
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return ndtr((math.sqrt(1.0 - rho) * ndtri(x) - ndtri(pd)) / math.sqrt(rho))


def lhp_quantile(q: float | np.ndarray, pd: float, rho: float) -> float | np.ndarray:
    """The q-quantile of the default fraction of the large homogeneous portfolio."""
    _check(pd, rho)
    return ndtr((ndtri(pd) + math.sqrt(rho) * ndtri(np.asarray(q, dtype=float))) / math.sqrt(1.0 - rho))


def lhp_pdf(x: float | np.ndarray, pd: float, rho: float) -> float | np.ndarray:
    """The density of the default fraction on (0, 1)."""
    _check(pd, rho)
    g = ndtri(np.asarray(x, dtype=float))
    return math.sqrt((1.0 - rho) / rho) * np.exp(0.5 * g**2 - (math.sqrt(1.0 - rho) * g - ndtri(pd)) ** 2 / (2.0 * rho))


def implied_factor(default_rate: float | np.ndarray, pd: float, rho: float) -> float | np.ndarray:
    """The systematic factor z at which the conditional PD equals the observed default rate."""
    _check(pd, rho)
    dr = np.asarray(default_rate, dtype=float)
    if np.any((dr <= 0.0) | (dr >= 1.0)):
        raise ValueError("an implied factor needs a default rate in (0, 1)")
    return (ndtri(pd) - math.sqrt(1.0 - rho) * ndtri(dr)) / math.sqrt(rho)

__all__ = [
    "conditional_pd",
    "implied_factor",
    "lhp_cdf",
    "lhp_pdf",
    "lhp_quantile",
]
