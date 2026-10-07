r"""The through-the-cycle portfolio a migration matrix implies, and the projection that drifts towards it.

Engelmann (2024), Spurious default probability projections in credit risk stress testing models, arXiv 2401.08892v1
(read in full): a portfolio $W$ over $n$ states (default last) is propagated one period by a transition matrix $T$,
its defaulted part written off and re-originated by the vector $O$ ($\sum_i o_i = 1$), keeping the balance constant,

$$W_t' = W_{t-1}' T(z_t) I_w + (W_{t-1}' T(z_t) V_w)\, O' \qquad (9)$$

with $I_w$ the identity whose last diagonal entry is 0 and $V_w = (0, \dots, 0, 1)'$. Theorem 1: if the performing
block $T_p = (p_{ij})_{i,j<n}$ is primitive (some power strictly positive) and $o_n = 0$, there is a unique portfolio
with

$$W_{ttc}' = W_{ttc}' T I_w + (W_{ttc}' T V_w)\, O' \qquad (10)$$

computed by iterating (9) with the unstressed $T$ "for an arbitrary initial portfolio" (the proof applies
Perron-Frobenius to $M = T I_w + T V_w O'$). "Independent of the stress imposed to the model, the current portfolio
will have a tendency to propagate towards the through-the-cycle portfolio", and the convergence "is not monotonic":
a projection under the unstressed matrix can show a spurious recession or boom. The portfolio PD is
$\sum_i w_i\, T_{in}$. Section 4's example (the Trueck and Rachev 2009 matrix (11)) is reproduced in
``tests/test_reproductions.py``.

A printed matrix has rows summing to 1 only up to its rounding (the matrix (11): 1 +- 0.0001). It is used as printed,
and the balance is kept constant, as the paper's step 5 states ("to keep total loan balance constant over time"), by
rescaling the portfolio to unit balance after each period. That reproduces section 4 to the printed digits;
renormalising the rows instead moves two entries of the printed TTC portfolio by 0.0002.

Primitivity is checked by Wielandt's bound: a non-negative $k \times k$ matrix is primitive if and only if its
$((k - 1)^2 + 1)$-th power is strictly positive.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ENGELMANN_2024 = (
    "Engelmann (2024), Spurious default probability projections in credit risk stress testing models, arXiv "
    "2401.08892v1, equations (9) and (10), Theorem 1"
)


def _transition(t: Any, atol: float) -> tuple[np.ndarray, float]:
    """A transition matrix, default last and absorbing, its rows summing to 1 within ``atol`` (a printed matrix's
    rounding), kept as given. Returns the matrix and the largest row-sum deviation."""
    a = np.asarray(t, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 3:
        raise ValueError(f"the transition matrix must be square with at least two grades and default, got {a.shape}")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("the transition matrix must be finite and non-negative")
    dev = float(np.max(np.abs(a.sum(axis=1) - 1.0)))
    if dev > atol:
        raise ValueError(f"a row sums to 1 +- {dev:.2g}, beyond atol {atol}")
    n = a.shape[0]
    if not np.allclose(a[-1], np.eye(n)[-1], atol=atol):
        raise ValueError("default (the last state) must be absorbing")
    return a, dev


def _origination(o: Any, n: int, atol: float) -> np.ndarray:
    v = np.asarray(o, dtype=float)
    if v.shape == (n - 1,):
        v = np.append(v, 0.0)
    if v.shape != (n,) or not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError(f"the origination vector must give {n - 1} or {n} non-negative shares")
    if v[-1] > 0:
        raise ValueError("origination into default breaks Theorem 1's second condition (o_n = 0)")
    if abs(v.sum() - 1.0) > atol:
        raise ValueError(f"the origination shares sum to {v.sum():.6g}, not 1")
    return v / v.sum()


def is_primitive(a: Any) -> bool:
    """Wielandt: a non-negative k x k matrix is primitive iff its ((k - 1)^2 + 1)-th power is strictly positive."""
    b = (np.asarray(a, dtype=float) > 0).astype(np.int64)
    k = b.shape[0]
    power = (k - 1) ** 2 + 1
    result = np.eye(k, dtype=np.int64)
    base = b
    while power:
        if power & 1:
            result = np.minimum(result @ base, 1)
        base = np.minimum(base @ base, 1)
        power >>= 1
    return bool(np.all(result > 0))


def propagation_matrix(transition: Any, origination: Any, *, atol: float = 1e-3) -> np.ndarray:
    """M = T I_w + (T V_w) O', so that (9) with a constant T is W_t' = W_{t-1}' M."""
    t, _ = _transition(transition, atol)
    o = _origination(origination, t.shape[0], atol)
    m = t.copy()
    m[:, -1] = 0.0
    return m + np.outer(t[:, -1], o)


def ttc_portfolio(transition: Any, origination: Any, *, atol: float = 1e-3, tol: float = 1e-14,
                  max_iter: int = 1_000_000) -> dict[str, Any]:
    """Engelmann's W_ttc: iterate (9) from the uniform performing portfolio, at unit balance, until the L1 change is
    below ``tol``; refuses a matrix whose performing block is not primitive or an origination into default (Theorem
    1). ``eigenvector_distance`` checks the result against the Perron vector of M."""
    t, dev = _transition(transition, atol)
    n = t.shape[0]
    if not is_primitive(t[:-1, :-1]):
        raise ValueError("the performing block is not primitive (Theorem 1's first condition): no unique TTC "
                         "portfolio")
    m = propagation_matrix(t, origination, atol=atol)
    w = np.append(np.full(n - 1, 1.0 / (n - 1)), 0.0)
    converged = False
    iterations = 0
    while not converged and iterations < max_iter:
        iterations += 1
        nxt = w @ m
        nxt = nxt / nxt.sum()
        converged = float(np.abs(nxt - w).sum()) < tol
        w = nxt
    vals, vecs = np.linalg.eig(m.T)
    v = np.real(vecs[:, np.argmax(np.real(vals))])
    v = v / v.sum()
    return {
        "portfolio": w,
        "default_rate": float(w @ t[:, -1]),
        "iterations": iterations,
        "converged": converged,
        "eigenvector_distance": float(np.abs(v - w).sum()),
        "row_sum_deviation": dev,
        "reference": ENGELMANN_2024,
    }


def project(transition: Any, portfolio: Any, origination: Any, years: int | None = None, *,
            atol: float = 1e-3) -> dict[str, Any]:
    """Propagate ``portfolio`` by (9) at unit balance: ``transition`` one matrix (the unstressed projection, over
    ``years``) or one matrix per year (a stressed path T(z_t)). Returns the portfolio at each date (years + 1 rows),
    the portfolio PD at each date before the last (``default_rate[t]`` = W_t' T_{t+1} V_w, the default rate of year
    t + 1), and, for one matrix, the TTC portfolio and each date's L1 distance to it."""
    arr = np.asarray(transition, dtype=float)
    if arr.ndim == 2:
        if years is None or years < 1:
            raise ValueError("one matrix needs years >= 1")
        mats = [_transition(arr, atol)[0]] * int(years)
    elif arr.ndim == 3:
        if years is not None and years != arr.shape[0]:
            raise ValueError("years must equal the number of matrices")
        mats = [_transition(a, atol)[0] for a in arr]
    else:
        raise ValueError("transition is one matrix or one matrix per year")
    n = mats[0].shape[0]
    w0 = np.asarray(portfolio, dtype=float)
    if w0.shape == (n - 1,):
        w0 = np.append(w0, 0.0)
    if w0.shape != (n,) or np.any(w0 < 0) or abs(w0.sum() - 1.0) > atol:
        raise ValueError(f"the portfolio must give {n - 1} or {n} non-negative shares summing to 1")
    o = _origination(origination, n, atol)
    path = [w0 / w0.sum()]
    rates = []
    for t in mats:
        w = path[-1]
        rates.append(float(w @ t[:, -1]))
        moved = w @ t
        nxt = np.append(moved[:-1], 0.0) + moved[-1] * o
        path.append(nxt / nxt.sum())
    out: dict[str, Any] = {"portfolio": np.array(path), "default_rate": np.array(rates), "reference": ENGELMANN_2024}
    if arr.ndim == 2:
        ttc = ttc_portfolio(arr, o, atol=atol)
        out["ttc"] = ttc
        out["distance_to_ttc"] = np.abs(out["portfolio"] - ttc["portfolio"]).sum(axis=1)
    return out


__all__ = ["is_primitive", "project", "propagation_matrix", "ttc_portfolio"]
