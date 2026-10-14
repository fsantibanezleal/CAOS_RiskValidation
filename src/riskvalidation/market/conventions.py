"""The sign convention of dossier 05 A.0 and VaR and ES at a tail level from a sample or a distribution.

Return convention in the engine: a return `y` is a decimal of the portfolio value; at a tail level `alpha` the engine
stores `VaR = v = F^-1(alpha)` and `ES = e = (1/alpha) int_0^alpha F^-1(u) du`, both negative in practice, with `e <=
v`.
Positive loss numbers (`loss = -y`) exist only at the reporting layer; `to_loss_numbers` is the one place the sign is
flipped.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np


def vec(x: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    """A one-dimensional float64 copy; a NaN or an infinity raises (a function never receives a missing return)."""
    a = np.array(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (the caller removes a missing day, never fills it)")
    return a


def check_alpha(alpha: float, *, upper_inclusive: bool = False) -> float:
    a = float(alpha)
    if not (0.0 < a < 0.5 or (upper_inclusive and a == 0.5)):
        raise ValueError(f"alpha must lie in (0, 0.5{']' if upper_inclusive else ')'}, got {alpha}")
    return a


def to_loss_numbers(values: np.ndarray) -> np.ndarray:
    """Returns, VaR or ES in the return convention to positive loss numbers: `-values`; a NaN stays NaN."""
    return -np.asarray(values, dtype=float)


def from_loss_numbers(values: np.ndarray) -> np.ndarray:
    """Positive loss numbers to the return convention: the exact inverse of `to_loss_numbers`."""
    return -np.asarray(values, dtype=float)


def var_es_from_sample(returns: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR and ES in the return convention (decision D-3 and review F4).

    With `k = ceil(alpha N)` and the sorted returns `r(1) <= ... <= r(N)`: `var = r(k)` (the left-continuous inverse
    `inf{y : F(y) >= alpha}`) and `es = (sum_{i<k} r(i) + (alpha N - (k - 1)) r(k)) / (alpha N)`, the tail integral of
    the empirical quantile function: the k-th return carries the fractional weight `alpha N - (k - 1)`.
    """
    # the golden of volatility_updated_hs_var_es uses alpha = 0.5, so the sample rule accepts 0.5 (contract
    # inconsistency)
    alpha = check_alpha(alpha, upper_inclusive=True)
    r = np.sort(vec(returns, "returns"))
    n = r.size
    an = alpha * n
    k = max(1, math.ceil(an - 1e-9))   # a product a hair above an integer must not move k
    es = (float(np.sum(r[: k - 1])) + (an - (k - 1)) * float(r[k - 1])) / an
    return float(r[k - 1]), float(es)


def var_es_from_distribution(ppf: Callable[[np.ndarray], np.ndarray], alpha: float,
                             n_nodes: int = 200) -> tuple[float, float]:
    """`var = ppf(alpha)` and `es = (1/alpha) int_0^alpha ppf(u) du` by Gauss-Legendre quadrature with `n_nodes` nodes.

    The integrand is unbounded at 0 for an unbounded left tail; the substitution `u = alpha s^3` (`du = 3 alpha s^2 ds`)
    makes it smooth enough for the quadrature, and is still a Gauss-Legendre rule on (0, 1).
    """
    alpha = check_alpha(alpha)
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    s = 0.5 * (x + 1.0)
    ws = 0.5 * w
    u = alpha * s**3
    q = np.asarray(ppf(u), dtype=float)
    es = float(np.sum(ws * q * 3.0 * s**2))
    var = float(np.asarray(ppf(np.array([alpha])), dtype=float)[0])
    return var, es
