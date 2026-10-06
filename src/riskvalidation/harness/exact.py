r"""Exact rejection probabilities of the count tests: no simulation needed.

`pd.binomial`, `pd.binomial_vasicek` and `pd.jeffreys` see the data only through one default count $D$ of $n$
obligors, and their p-value decreases in $D$; so each rejects at level $\alpha$ exactly when $D \ge k^*$, the smallest
count whose p-value is below $\alpha$ under the PD applied. Under a true PD $\pi$ and asset correlation $\rho$:

$$P(\text{reject}) = P_{\pi,\rho}(D \ge k^*) = \int \bar F_{\mathrm{Bin}(n, p(x))}(k^* - 1)\,\varphi(x)\,dx,\qquad
p(x) = \Phi\big((\Phi^{-1}(\pi) - \sqrt\rho\,x)/\sqrt{1-\rho}\big),$$

the binomial tail when $\rho = 0$. $k^*$ is found by evaluating the registered test itself, so the exact value checks
the generator, the harness and the test together.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import special, stats

from riskvalidation.core.registry import spec as get_spec
from riskvalidation.core.result import TestResult
from riskvalidation.generators._base import GH_NODES, GH_WEIGHTS
from riskvalidation.validation.calibration import (
    WP14_TRAFFIC_PROBS,
    default_count_sf,
    traffic_light_colours,
    traffic_profile_cdf,
)

COUNT_TESTS = ("pd.binomial", "pd.binomial_vasicek", "pd.jeffreys")


def _func(test: str | Callable[..., TestResult]) -> Callable[..., TestResult]:
    return get_spec(test).func if isinstance(test, str) else test


def count_pmf(n: int, pd: float, rho: float = 0.0) -> np.ndarray:
    """P(D = k) for k = 0..n: binomial when ``rho`` is 0, the one-factor mixture (256-node Gauss-Hermite) otherwise."""
    k = np.arange(n + 1)
    if rho == 0.0:
        return stats.binom.pmf(k, n, pd)
    if not 0.0 < rho < 1.0:
        raise ValueError("asset correlation must lie in [0, 1)")
    cond = special.ndtr((special.ndtri(pd) - np.sqrt(rho) * GH_NODES) / np.sqrt(1.0 - rho))
    return stats.binom.pmf(k[:, None], n, cond[None, :]) @ GH_WEIGHTS


def critical_count(test: str | Callable[..., TestResult], n: int, pd_applied: float, alpha: float,
                   **test_kwargs: Any) -> int | None:
    """Smallest count k in 0..n whose p-value is below ``alpha``; ``None`` when no count rejects.

    The p-value must not increase in the count (true of the three count tests); the bisection checks the bracket."""
    func = _func(test)

    def p(k: int) -> float:
        r = func(n, k, pd_applied, **test_kwargs)
        return float(r.p_value)

    if p(n) >= alpha:
        return None
    if p(0) < alpha:
        return 0
    lo, hi = 0, n  # p(lo) >= alpha > p(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p(mid) < alpha:
            hi = mid
        else:
            lo = mid
    return hi


def rejection_probability(test: str | Callable[..., TestResult], n: int, pd_applied: float, alpha: float, *,
                          pd_true: float | None = None, rho_true: float = 0.0, **test_kwargs: Any) -> dict[str, Any]:
    """Exact probability that a count test rejects at ``alpha`` when the true PD is ``pd_true`` (the PD applied when
    omitted: the size) and the true asset correlation ``rho_true``. ``test_kwargs`` go to the test (for example the
    assumed ``rho`` of `pd.binomial_vasicek`)."""
    pt = pd_applied if pd_true is None else pd_true
    k = critical_count(test, n, pd_applied, alpha, **test_kwargs)
    prob = 0.0 if k is None else default_count_sf(k, n, pt, rho_true)
    return {"probability": float(prob), "critical_count": k, "n": n, "pd_applied": pd_applied, "pd_true": pt,
            "rho_true": rho_true, "alpha": alpha}


def wp14_true_confidence(pd: float, n: int, rho: float, q_uncorrelated: float) -> float:
    """BCBS WP14's large-portfolio approximation of the true confidence of an independence-based binomial test of
    confidence ``q_uncorrelated`` when defaults follow the one-factor model with asset correlation ``rho`` (page 59):
    Phi((sqrt(1 - rho) Phi^-1(PD + Phi^-1(q) sqrt(PD (1 - PD) / n)) - Phi^-1(PD)) / sqrt(rho))."""
    c = pd + special.ndtri(q_uncorrelated) * np.sqrt(pd * (1.0 - pd) / n)
    if not 0.0 < c < 1.0:
        raise ValueError("the critical default rate leaves (0, 1)")
    return float(special.ndtr((np.sqrt(1.0 - rho) * special.ndtri(c) - special.ndtri(pd)) / np.sqrt(rho)))




def traffic_lights_rejection_probability(n: Any, pd_applied: Any, alpha: float, *, pd_true: Any = None,
                                         probs: Any = WP14_TRAFFIC_PROBS, ties: str = "lower") -> dict[str, Any]:
    """Exact probability that the traffic-lights test rejects at ``alpha`` when the years are independent (no asset
    correlation, no factor correlation): each year's colour probabilities from its binomial count under the true PD,
    coloured by the test's own rule, the profile distribution by convolution over the years, and the rejection region
    from the test's own cumulative profile table."""
    n_a = np.asarray(n, dtype=np.int64).reshape(-1)
    pf = np.broadcast_to(np.asarray(pd_applied, dtype=float).reshape(-1), n_a.shape)
    pt = pf if pd_true is None else np.broadcast_to(np.asarray(pd_true, dtype=float).reshape(-1), n_a.shape)
    years = n_a.size
    colour_p = np.zeros((years, 4))
    for t in range(years):
        d = np.arange(int(n_a[t]) + 1)
        col = traffic_light_colours(np.full(d.size, n_a[t]), d, np.full(d.size, pf[t]), probs=probs, ties=ties)
        colour_p[t] = np.bincount(col, weights=stats.binom.pmf(d, int(n_a[t]), pt[t]), minlength=4)
    dist: dict[tuple[int, int, int, int], float] = {(0, 0, 0, 0): 1.0}
    for t in range(years):
        nxt: dict[tuple[int, int, int, int], float] = {}
        for prof, pr in dist.items():
            for c in range(4):
                if colour_p[t, c] > 0.0:
                    q = list(prof)
                    q[c] += 1
                    key = (q[0], q[1], q[2], q[3])
                    nxt[key] = nxt.get(key, 0.0) + pr * colour_p[t, c]
        dist = nxt
    cdf = traffic_profile_cdf(years, probs)
    prob = sum(pr for prof, pr in dist.items() if cdf[prof] < alpha)
    return {"probability": float(prob), "colour_probabilities": colour_p.tolist(), "ties": ties, "alpha": alpha}


__all__ = [
    "COUNT_TESTS",
    "count_pmf",
    "critical_count",
    "rejection_probability",
    "traffic_lights_rejection_probability",
    "wp14_true_confidence",
]
