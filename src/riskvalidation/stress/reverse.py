"""Reverse stress: the most plausible scenario that reaches a pre-defined failure point (dossier 05 H.5, 01 J.4).

EBA/GL/2018/04 defines a reverse stress test as one that starts from the identification of a pre-defined outcome and
then explores the scenarios that might cause it (01 l.884-889). The formal statement is Contraste's, not regulatory text
(01 l.891-894; 05 H.5): for a loss model `L(x)` over factor deviations `x` from the baseline with covariance `Sigma`,
the most plausible scenario reaching the loss threshold `l` maximises the factor density, which for Gaussian (and
elliptical) factors is the minimum Mahalanobis distance scenario

    x* = argmin x' Sigma^-1 x   subject to   L(x) >= l,

in closed form for a linear loss and by `scipy.optimize` (SLSQP) otherwise, with the plausibility of the solution. The
attribution of the elliptical result to a specific paper is UNVERIFIED in the dossier (05 l.412). Empirical likelihood
(Glasserman, Kang and Kang) is out: only its abstract is in the dossier.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class ReverseSolution:
    """A reverse stress solution; maps onto the document block `solution`."""

    method: str  # "closed-form-linear" or "scipy-optimize"
    x_star: np.ndarray  # (K,) deviation of the factors from the baseline, natural units
    z_star: np.ndarray  # (K,) x_star over the factor sd
    distance: float  # sqrt(x' Sigma^-1 x)
    loss_at_x: float
    threshold: float
    converged: bool


def _cov(cov: np.ndarray) -> np.ndarray:
    c = np.array(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.size == 0:
        raise ValueError(f"cov must be a square matrix, got shape {c.shape}")
    if not np.all(np.isfinite(c)):
        raise ValueError("cov contains non-finite values")
    if not np.allclose(c, c.T, rtol=1e-9, atol=1e-12):
        raise ValueError("cov is not symmetric")
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov is not positive definite") from exc
    return c


def _threshold(threshold: float, *, positive: bool) -> float:
    v = float(threshold)
    if not np.isfinite(v):
        raise ValueError("threshold is non-finite")
    if v < 0.0 or (positive and v == 0.0):
        raise ValueError(f"threshold must be {'positive' if positive else 'not negative'}, got {v}")
    return v


def reverse_stress_linear(gradient: np.ndarray, cov: np.ndarray, threshold: float) -> ReverseSolution:
    """The closed form for the linear loss `L(x) = g'x`: `x* = Sigma g l / (g' Sigma g)` and `distance = l / sqrt(g'
    Sigma g)`.
    `gradient` (K,) is the loss per unit of each factor (money per unit), `cov` (K, K) the factor covariance,
    `threshold` money."""
    c = _cov(cov)
    g = np.array(gradient, dtype=float)
    if g.shape != (c.shape[0],) or not np.all(np.isfinite(g)):
        raise ValueError(f"gradient must be a finite vector of length {c.shape[0]}, got shape {g.shape}")
    thr = _threshold(threshold, positive=False)
    q = float(g @ c @ g)
    if q <= 0.0:
        raise ValueError("the gradient gives g' Sigma g = 0: the loss does not move with any factor")
    x = c @ g * (thr / q)
    sd = np.sqrt(np.diag(c))
    return ReverseSolution(
        method="closed-form-linear",
        x_star=x,
        z_star=x / sd,
        distance=thr / math.sqrt(q),
        loss_at_x=float(g @ x),
        threshold=thr,
        converged=True,
    )


def reverse_stress(
    loss_fn: Callable[[np.ndarray], float],
    cov: np.ndarray,
    threshold: float,
    starts: Sequence[np.ndarray],
    tolerance: float = 1e-6,
) -> tuple[ReverseSolution, tuple[ReverseSolution, ...]]:
    """Minimise `x' Sigma^-1 x` subject to `L(x) >= threshold` with `scipy.optimize` (SLSQP) from each start. Returns
    the best converged solution (the smallest distance) and the per-start results. A start converges when SLSQP succeeds
    and `|L(x) - threshold| / threshold <= tolerance`. At least three starts are required (the search is local: one
    start proves nothing); `ValueError` when no start converges. `loss_fn` maps the factor deviation to money; the
    threshold is
    positive."""
    c = _cov(cov)
    thr = _threshold(threshold, positive=True)
    tol = float(tolerance)
    if not (np.isfinite(tol) and tol > 0.0):
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")
    if len(starts) < 3:
        raise ValueError(f"at least three starts are required, got {len(starts)}")
    k = c.shape[0]
    x0s = []
    for i, s in enumerate(starts):
        a = np.array(s, dtype=float)
        if a.shape != (k,) or not np.all(np.isfinite(a)):
            raise ValueError(f"start {i} must be a finite vector of length {k}, got shape {a.shape}")
        x0s.append(a)
    prec = np.linalg.inv(c)
    sd = np.sqrt(np.diag(c))
    results = []
    for x0 in x0s:
        res = optimize.minimize(
            lambda x: float(x @ prec @ x),
            x0,
            jac=lambda x: 2.0 * prec @ x,
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": lambda x: float(loss_fn(np.array(x))) / thr - 1.0}],
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        x = np.array(res.x, dtype=float)
        loss = float(loss_fn(x.copy()))
        ok = bool(res.success) and np.all(np.isfinite(x)) and np.isfinite(loss) and abs(loss - thr) / thr <= tol
        results.append(
            ReverseSolution(
                method="scipy-optimize",
                x_star=x,
                z_star=x / sd,
                distance=math.sqrt(float(x @ prec @ x)),
                loss_at_x=loss,
                threshold=thr,
                converged=bool(ok),
            )
        )
    converged = [s for s in results if s.converged]
    if not converged:
        raise ValueError(
            "no start converged: SLSQP did not reach the threshold within the tolerance from any of the "
            f"{len(results)} starts"
        )
    return min(converged, key=lambda s: s.distance), tuple(results)


def plausibility(distance: float, df: int) -> float:
    """`P(chi2_df >= distance^2)`: the tail probability of a scenario at least as far (Mahalanobis distance `distance`)
    from the
    baseline, `df` the number of factors. 1.0 at distance 0, decreasing in the distance."""
    d = float(distance)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError(f"distance must be a finite non-negative number, got {distance!r}")
    if int(df) != df or df < 1:
        raise ValueError(f"df must be an integer of at least 1, got {df!r}")
    return float(stats.chi2.sf(d * d, int(df)))
