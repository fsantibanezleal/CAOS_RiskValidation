"""Most prudent estimation of PDs for low-default portfolios (Pluto and Tasche).

Grades are ordered best first. The PD of grade i is estimated as an upper confidence bound at level gamma under the
most prudent assumption that every grade from i downward has the same PD, so the obligors and defaults of grades i to
k are pooled (Pluto and Tasche 2005, arXiv cond-mat/0411699, sections 2 and 3). With N pooled obligors and D pooled
defaults, independent defaults, the bound p solves

    P(Binomial(N, p) <= D) = 1 - gamma,            (zero defaults: p = 1 - (1 - gamma)^(1/N)),

the Clopper-Pearson upper bound. With asset correlation rho in the one-factor model (section 4, eq. 4.1) the binomial
probability is averaged over the systematic factor, computed here by Gauss-Hermite quadrature:

    1 - gamma = integral of phi(y) P(Binomial(N, p(y)) <= D) dy,   p(y) = N((G(p) - sqrt(rho) y) / sqrt(1 - rho)).

The bounds are monotone across grades by construction. The paper notes that the choice of gamma is arbitrary and
that levels below 95% "seem more appropriate"; the level is the caller's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import optimize, stats
from scipy.special import ndtr, ndtri

PLUTO_TASCHE = (
    "Pluto and Tasche (2005), Estimating probabilities of default for low default portfolios, arXiv cond-mat/0411699"
)

_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(256)
_WEIGHTS = _WEIGHTS / math.sqrt(2.0 * math.pi)


def _pooled(obligors: Sequence[int], defaults: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(obligors, dtype=int)
    d = np.asarray(defaults, dtype=int)
    if n.shape != d.shape or n.ndim != 1 or len(n) == 0:
        raise ValueError("obligors and defaults are one number per grade")
    if np.any(n <= 0) or np.any(d < 0) or np.any(d > n):
        raise ValueError("each grade needs obligors > 0 and 0 <= defaults <= obligors")
    # grade i pools grades i..k (best first, so the pool grows towards the best grade)
    return np.cumsum(n[::-1])[::-1], np.cumsum(d[::-1])[::-1]


def _independent_bound(n: int, d: int, gamma: float) -> float:
    if d >= n:
        return 1.0
    return float(stats.beta.ppf(gamma, d + 1, n - d))


def _correlated_bound(n: int, d: int, gamma: float, rho: float) -> float:
    target = 1.0 - gamma

    def tail(p: float) -> float:
        cond = ndtr((ndtri(p) - math.sqrt(rho) * _NODES) / math.sqrt(1.0 - rho))
        return float(np.sum(_WEIGHTS * stats.binom.cdf(d, n, cond))) - target

    lo, hi = 1e-12, 1.0 - 1e-12
    return float(optimize.brentq(tail, lo, hi, xtol=1e-15, rtol=1e-12, maxiter=500))


def most_prudent_pd(obligors: Sequence[int], defaults: Sequence[int], gamma: float, rho: float = 0.0) -> dict:
    """Most prudent PD bounds per grade (best first) at confidence ``gamma``, independent or with asset correlation
    ``rho``. Returns the bounds and the pooled counts each one rests on."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    big_n, big_d = _pooled(obligors, defaults)
    if rho == 0.0:
        bounds = [_independent_bound(int(n), int(d), gamma) for n, d in zip(big_n, big_d, strict=True)]
    else:
        bounds = [_correlated_bound(int(n), int(d), gamma, rho) for n, d in zip(big_n, big_d, strict=True)]
    return {
        "pd": bounds,
        "pooled_obligors": [int(v) for v in big_n],
        "pooled_defaults": [int(v) for v in big_d],
        "gamma": gamma,
        "rho": rho,
        "reference": PLUTO_TASCHE,
    }
