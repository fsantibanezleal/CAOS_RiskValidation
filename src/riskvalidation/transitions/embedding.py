"""The embedding problem: is a one-year transition matrix the exponential of a generator, and which generator to use
when it is not.

Israel, Rosenthal and Wei (2001), Finding generators for Markov chains via empirical transition matrices, with
applications to credit ratings, Mathematical Finance 11(2), 245-265, doi:10.1111/1467-9965.00114 (read in the authors'
version, https://www.probability.ca/jeff/ftpdir/wei.pdf). A generator Q has non-negative off-diagonal entries and rows
summing to 0; the problem is to find one with exp(Q) = P.

- Theorem 1: with S = max{(a - 1)^2 + b^2 : a + bi an eigenvalue of P}, if S < 1 the series
  Q~ = (P - I) - (P - I)^2/2 + (P - I)^3/3 - ... converges geometrically and exp(Q~) = P exactly, rows summing to 0.
  Theorem 2: every p_ii > 1/2 implies S < 1, and then P has at most one generator (Cuthbert 1972, 1973).
- Q~ may have negative off-diagonal entries. Equation (2), the diagonal adjustment: q_ij = max(q~_ij, 0) for j != i,
  q_ii = q~_ii + sum_{j != i} min(q~_ij, 0). Equation (2'), the weighted adjustment: G_i = |q~_ii| +
  sum_{j != i} max(q~_ij, 0), B_i = sum_{j != i} max(-q~_ij, 0); q_ij = 0 where i != j and q~_ij < 0,
  q_ij = q~_ij - B_i |q~_ij| / G_i otherwise if G_i > 0, q_ij = q~_ij if G_i = 0. Neither keeps exp(Q) = P.
- Equation (3), Jarrow, Lando and Turnbull's approximation (at most one transition a year): q_ii = log(p_ii),
  q_ij = p_ij log(p_ii) / (p_ii - 1).
- Theorem 3: no exact generator exists if (a) det(P) <= 0, (b) det(P) > prod_i p_ii, or (c) some j is accessible
  from i while p_ij = 0.
- Lemma 1 (Jarrow, Lando and Turnbull 1997, p. 495): a chain is stochastically monotonic when sum_{j >= k} p_ij is
  nondecreasing in i for every k.

The distance they report is the L1 norm, the sum of the absolute entries of P - exp(Q).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.linalg import expm, logm

ISRAEL_ROSENTHAL_WEI_2001 = (
    "Israel, Rosenthal and Wei (2001), Finding generators for Markov chains via empirical transition matrices, "
    "with applications to credit ratings, doi:10.1111/1467-9965.00114"
)


def _stochastic(p: Sequence[Sequence[float]], *, tol: float = 1e-3) -> np.ndarray:
    m = np.asarray(p, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"a transition matrix must be square, got shape {m.shape}")
    if np.any(m < 0) or not np.all(np.isfinite(m)):
        raise ValueError("a transition matrix must be finite and non-negative")
    if np.max(np.abs(m.sum(axis=1) - 1.0)) > tol:
        raise ValueError("every row of a transition matrix must sum to 1")
    return m


def eigen_condition(p: Sequence[Sequence[float]]) -> float:
    """S = max |lambda - 1|^2 over the eigenvalues of P (Theorem 1: the log series converges when S < 1)."""
    lam = np.linalg.eigvals(_stochastic(p))
    return float(np.max(np.abs(lam - 1.0) ** 2))


def log_series(p: Sequence[Sequence[float]], *, tol: float = 1e-14, max_terms: int = 2000) -> dict[str, Any]:
    """Theorem 1's series for log P, summed until a term's largest entry falls below ``tol``."""
    m = _stochastic(p)
    n = m.shape[0]
    a = m - np.eye(n)
    q = np.zeros_like(a)
    power = np.eye(n)
    converged = False
    terms = 0
    for k in range(1, max_terms + 1):
        power = power @ a
        term = power / k * (1.0 if k % 2 == 1 else -1.0)
        q += term
        terms = k
        if np.max(np.abs(term)) < tol:
            converged = True
            break
    return {
        "generator": q,
        "terms": terms,
        "converged": converged,
        "S": eigen_condition(m),
        "distance_to_logm": float(np.max(np.abs(q - np.real(logm(m))))) if converged else None,
        "reference": f"{ISRAEL_ROSENTHAL_WEI_2001}, Theorem 1",
    }


def _reachable(p: np.ndarray) -> np.ndarray:
    """reach[i, j]: j can be reached from i in one or more steps of positive probability."""
    step = p > 0
    reach = step.copy()
    for _ in range(p.shape[0]):
        nxt = reach | ((reach.astype(int) @ step.astype(int)) > 0)
        if np.array_equal(nxt, reach):
            break
        reach = nxt
    return reach


def monotonicity_violations(p: Sequence[Sequence[float]], *, tol: float = 0.0) -> list[dict[str, Any]]:
    """Lemma 1 (a): every (i, k) where sum_{j >= k} p_ij falls from row i to row i + 1 by more than ``tol``."""
    m = _stochastic(p)
    tail = np.cumsum(m[:, ::-1], axis=1)[:, ::-1]  # tail[i, k] = sum_{j >= k} p_ij
    out = []
    for i in range(m.shape[0] - 1):
        for k in range(1, m.shape[0]):
            if tail[i + 1, k] < tail[i, k] - tol:
                out.append({"row": i, "next_row": i + 1, "column": k, "tail": float(tail[i, k]),
                            "next_tail": float(tail[i + 1, k])})
    return out


def embedding_diagnostics(p: Sequence[Sequence[float]], *, tol: float = 1e-12) -> dict[str, Any]:
    """What can be said of P's embeddability: Theorem 1's S, the three no-generator conditions of Theorem 3, the
    negative off-diagonal entries of the log series, and Lemma 1's monotonicity. An entry counts as negative below
    ``-tol`` (the series leaves round-off of order 1e-17 where the true entry is 0)."""
    m = _stochastic(p)
    n = m.shape[0]
    det = float(np.linalg.det(m))
    prod_diag = float(np.prod(np.diag(m)))
    reach = _reachable(m)
    zero_but_reachable = [(int(i), int(j)) for i in range(n) for j in range(n) if reach[i, j] and m[i, j] == 0]
    series = log_series(m)
    q = series["generator"]
    off = ~np.eye(n, dtype=bool)
    negative = [(int(i), int(j), float(q[i, j]))
                for i, j in zip(*np.nonzero(off & (q < -tol)), strict=True)]
    return {
        "S": series["S"],
        "series_converges": series["converged"],
        "all_diagonal_above_half": bool(np.all(np.diag(m) > 0.5)),
        "det": det,
        "prod_diagonal": prod_diag,
        "theorem3_a": det <= 0,
        "theorem3_b": det > prod_diag,
        "theorem3_c": zero_but_reachable,
        "exact_generator_excluded": bool(det <= 0 or det > prod_diag or zero_but_reachable),
        "series_negative_off_diagonal": negative,
        "stochastically_monotone": not monotonicity_violations(m),
        "monotonicity_violations": monotonicity_violations(m),
        "reference": f"{ISRAEL_ROSENTHAL_WEI_2001}, Theorems 1 to 3 and Lemma 1",
    }


def diagonal_adjustment(q_tilde: np.ndarray) -> np.ndarray:
    """Equation (2): negative off-diagonal entries to 0, their sum added to the diagonal."""
    q = np.array(q_tilde, dtype=float)
    n = q.shape[0]
    off = ~np.eye(n, dtype=bool)
    neg = np.where(off & (q < 0), q, 0.0)
    q[off & (q < 0)] = 0.0
    q[np.diag_indices(n)] += neg.sum(axis=1)
    return q


def weighted_adjustment(q_tilde: np.ndarray) -> np.ndarray:
    """Equation (2'): negative off-diagonal entries to 0, their total taken from the row's correct-sign entries in
    proportion to their absolute values."""
    qt = np.asarray(q_tilde, dtype=float)
    n = qt.shape[0]
    q = qt.copy()
    for i in range(n):
        off = np.arange(n) != i
        g = abs(qt[i, i]) + np.sum(np.maximum(qt[i, off], 0.0))
        b = np.sum(np.maximum(-qt[i, off], 0.0))
        for j in range(n):
            if j != i and qt[i, j] < 0:
                q[i, j] = 0.0
            elif g > 0:
                q[i, j] = qt[i, j] - b * abs(qt[i, j]) / g
    return q


def jlt_generator(p: Sequence[Sequence[float]]) -> np.ndarray:
    """Equation (3): q_ii = log p_ii, q_ij = p_ij log(p_ii) / (p_ii - 1); an absorbing row (p_ii = 1) stays 0."""
    m = _stochastic(p)
    n = m.shape[0]
    q = np.zeros((n, n))
    for i in range(n):
        pii = m[i, i]
        if pii >= 1.0:
            continue
        if pii <= 0.0:
            raise ValueError(f"row {i} has p_ii = 0: the JLT approximation needs log(p_ii)")
        q[i, i] = np.log(pii)
        for j in range(n):
            if j != i:
                q[i, j] = m[i, j] * np.log(pii) / (pii - 1.0)
    return q


def l1_distance(p: Sequence[Sequence[float]], q: np.ndarray, *, t: float = 1.0) -> float:
    """The sum of the absolute entries of P - exp(tQ), the distance Israel et al. report."""
    return float(np.sum(np.abs(np.asarray(p, dtype=float) - expm(t * np.asarray(q, dtype=float)))))


def is_generator(q: np.ndarray, *, tol: float = 1e-12) -> bool:
    """Non-negative off-diagonal entries and rows summing to 0."""
    q = np.asarray(q, dtype=float)
    off = ~np.eye(q.shape[0], dtype=bool)
    return bool(np.all(q[off] >= -tol) and np.all(np.abs(q.sum(axis=1)) <= 1e-9))


def generator(p: Sequence[Sequence[float]], *, method: str = "weighted") -> dict[str, Any]:
    """A generator for P by ``method``: ``"series"`` (Theorem 1, possibly invalid), ``"diagonal"`` (2),
    ``"weighted"`` (2') or ``"jlt"`` (3), with its validity and its L1 distance to P."""
    m = _stochastic(p)
    if method == "jlt":
        q = jlt_generator(m)
    else:
        series = log_series(m)
        if not series["converged"]:
            raise ValueError("the log series does not converge for this matrix (S >= 1)")
        qt = series["generator"]
        q = {"series": lambda x: x, "diagonal": diagonal_adjustment, "weighted": weighted_adjustment}.get(method)
        if q is None:
            raise ValueError("method must be 'series', 'diagonal', 'weighted' or 'jlt'")
        q = q(qt)
    return {
        "generator": q,
        "method": method,
        "valid": is_generator(q),
        "l1_distance": l1_distance(m, q),
        "reference": ISRAEL_ROSENTHAL_WEI_2001,
    }
