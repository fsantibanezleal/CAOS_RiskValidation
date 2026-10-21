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


def scale_bounds(bounds: Sequence[float], obligors: Sequence[int], target: float) -> dict:
    """Section 5, equations 5.1 and 5.2: one factor K for every grade, such that the obligor-weighted average of the
    scaled bounds equals ``target``. Scaling keeps the bounds' ordering and their ratios (the "relative" PDs the most
    prudent principle derives) and gives up their interpretation as confidence bounds."""
    b = np.asarray(bounds, dtype=float)
    n = np.asarray(obligors, dtype=float)
    if b.shape != n.shape or b.ndim != 1 or np.any(n <= 0) or np.any(b <= 0):
        raise ValueError("bounds and obligors are one positive value per grade")
    if not 0.0 < target < 1.0:
        raise ValueError("the scaling target must lie in (0, 1)")
    k = target * float(n.sum()) / float(np.sum(b * n))
    return {"pd": [float(v) for v in k * b], "k": k, "target": float(target),
            "reference": f"{PLUTO_TASCHE}, section 5, equations 5.1 and 5.2"}


def most_prudent_pd_scaled(obligors: Sequence[int], defaults: Sequence[int], gamma: float, rho: float = 0.0, *,
                           target: str | float = "upper_bound") -> dict:
    """The most prudent bounds scaled by section 5: to the portfolio's observed default rate (``"central_tendency"``,
    Tables 9 and 10; needs at least one default), to the most prudent bound of the whole portfolio at the same level,
    which is the best grade's bound (``"upper_bound"``, Tables 11 and 12, the paper's proposal), or to a given PD."""
    raw = most_prudent_pd(obligors, defaults, gamma, rho)
    n, d = np.asarray(obligors, dtype=float), np.asarray(defaults, dtype=float)
    if target == "central_tendency":
        if d.sum() <= 0:
            raise ValueError("scaling to the central tendency needs at least one default (section 5)")
        level = float(d.sum() / n.sum())
    elif target == "upper_bound":
        level = raw["pd"][0]
    elif isinstance(target, (int, float)) and not isinstance(target, bool):
        level = float(target)
    else:
        raise ValueError('target is "central_tendency", "upper_bound" or a PD in (0, 1)')
    scaled = scale_bounds(raw["pd"], obligors, level)
    return {**raw, "unscaled": raw["pd"], "pd": scaled["pd"], "k": scaled["k"], "target": level,
            "target_kind": target if isinstance(target, str) else "given",
            "reference": f"{PLUTO_TASCHE}, sections 3 to 5"}


def _multiperiod_tail(p: float, n: int, k: int, rho: float, factors: np.ndarray) -> float:
    """(6.8b): P[at most k of n obligors default within the period], averaged over the simulated factor paths."""
    cond = ndtr((ndtri(p) - math.sqrt(rho) * factors) / math.sqrt(1.0 - rho))
    pi = 1.0 - np.prod(1.0 - cond, axis=1)  # (6.8a): an obligor's probability to default within the period
    return float(np.mean(stats.binom.cdf(k, n, pi)))


def most_prudent_pd_multiperiod(obligors: Sequence[int], defaults: Sequence[int], gamma: float, *, rho: float,
                                theta: float, years: int, n_sim: int = 100_000, seed: int = 0) -> dict:
    """Section 6: most prudent bounds from a cohort observed for ``years`` years, with ``defaults`` the defaults of
    each grade over the whole period. The systematic factors of the years are standard normal with correlation
    theta^|s - t| (6.5b); obligor i defaults in year t when sqrt(rho) S_t + sqrt(1 - rho) xi_it first falls below
    Phi^{-1}(p) (6.1 to 6.4), so p is a long-run one-year PD. Given the factors an obligor defaults within the period
    with probability 1 - prod_t (1 - p(S_t)) (6.7, 6.8a), and the bound is the largest p with
    P[at most k defaults] >= 1 - gamma (6.6, 6.8b). The expectation over the factors is a Monte Carlo average over
    ``n_sim`` paths drawn with ``seed``, the same paths for every p (so the bound is a smooth function of p)."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie in (0, 1)")
    if not 0.0 <= theta < 1.0:
        raise ValueError("theta must lie in [0, 1)")
    if years < 1 or n_sim < 1000:
        raise ValueError("years >= 1 and n_sim >= 1000")
    big_n, big_d = _pooled(obligors, defaults)
    t = np.arange(years)
    corr = theta ** np.abs(np.subtract.outer(t, t)).astype(float)
    factors = np.random.default_rng(seed).standard_normal((n_sim, years)) @ np.linalg.cholesky(corr).T
    bounds = []
    for n, d in zip(big_n, big_d, strict=True):
        if d >= n:
            bounds.append(1.0)
            continue
        bounds.append(float(optimize.brentq(lambda p, n=int(n), d=int(d): _multiperiod_tail(p, n, d, rho, factors)
                                            - (1.0 - gamma), 1e-12, 1.0 - 1e-12, xtol=1e-15, rtol=1e-12)))
    return {
        "pd": bounds,
        "pooled_obligors": [int(v) for v in big_n],
        "pooled_defaults": [int(v) for v in big_d],
        "gamma": gamma,
        "rho": rho,
        "theta": theta,
        "years": int(years),
        "n_sim": int(n_sim),
        "seed": int(seed),
        "reference": f"{PLUTO_TASCHE}, section 6, equations 6.1 to 6.8",
    }

__all__ = [
    "most_prudent_pd",
    "most_prudent_pd_multiperiod",
    "most_prudent_pd_scaled",
    "scale_bounds",
]
