"""CVA, DVA and wrong-way risk (dossier 05 C.2; Hull and White 2012, "CVA and wrong way risk", Financial Analysts
Journal 68(5):58-69, addendum R1): the credit triangle, the discretised CVA and DVA, and the hazard rate that depends on
the portfolio value, ``h(t) = exp[a(t) + b w(t)]``, with the sequential calibration of ``a(t)``
of the paper's Appendix.

CVA is a positive cost and DVA a positive benefit. The simple bilateral form ``BCVA = CVA - DVA`` is
the only one; the first-to-default form is not implemented. ORE's wrong-way example is a different model (a correlated
credit LGM), so no ORE golden applies to ``wwr_cva``. Contract: ``engine-ccr-api`` 1.0.0, module cva.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from riskvalidation.lgd_ead._checks import mat, same_length, scalar, vec

LOG_A_BRACKET = (-60.0, 5.0)
"""The starting bracket of the search for ``a(t_k)``; the upper end is extended until the average survival is below the
target, the lower end is the hazard exp(-60), essentially zero."""


def _lgd(x: float, name: str) -> float:
    v = scalar(x, name)
    if not 0.0 < v <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1], got {v}")
    return v


def survival_from_spread(spread: np.ndarray, t: np.ndarray, lgd: float) -> np.ndarray:
    """The credit triangle ``lambda = s / LGD`` and
    ``S(t) = exp(-s t / LGD)`` (dossier 05 C.2; Hull and White equation 6 and page 13).
    ``S(0) = 1``, non-increasing in ``t``, 1 for a zero spread."""
    s = np.array(spread, dtype=float)
    tt = np.array(t, dtype=float)
    if not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("spread must be finite and nonnegative")
    if not np.all(np.isfinite(tt)) or np.any(tt < 0.0):
        raise ValueError("t must be finite and nonnegative")
    return np.exp(-s * tt / _lgd(lgd, "lgd"))


def default_probabilities(survival: np.ndarray) -> np.ndarray:
    """(K,) ``PD(t_{i-1}, t_i) = S(t_{i-1}) - S(t_i)`` from the (K+1,) survival curve at
    ``t_0 = 0, ..., t_K``; non-negative and summing to ``S(0) - S(t_K)``. A survival curve
    that rises or leaves (0, 1] is refused."""
    s = vec(survival, "survival")
    if s.size < 2:
        raise ValueError("survival needs at least 2 points")
    if np.any(s <= 0.0) or np.any(s > 1.0):
        raise ValueError("survival must lie in (0, 1]")
    pd = s[:-1] - s[1:]
    if np.any(pd < 0.0):
        raise ValueError("survival must be non-increasing")
    return pd


def _loss(exposure: np.ndarray, pd: np.ndarray, lgd: float, exposure_name: str, pd_name: str,
          lgd_name: str) -> tuple[float, np.ndarray]:
    e, p = vec(exposure, exposure_name), vec(pd, pd_name)
    same_length(**{exposure_name: e, pd_name: p})
    if np.any(e < 0.0):
        raise ValueError(f"{exposure_name} must be nonnegative")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"{pd_name} must lie in [0, 1]")
    terms = p * _lgd(lgd, lgd_name) * e
    return float(np.sum(terms)), terms


def cva(epe_discounted: np.ndarray, pd: np.ndarray, lgd: float) -> tuple[float, np.ndarray]:
    """``(cva, terms)``: ``CVA = sum PD(t_{i-1}, t_i) LGD EPE(t_i)`` (dossier 05 C.2,
    the advanced discretisation), a positive cost; ``terms`` (K,) are the bucket terms in money. A zero spread
    (``pd = 0``) gives exactly 0; linear in ``lgd`` and in the EE."""
    return _loss(epe_discounted, pd, lgd, "epe_discounted", "pd", "lgd")


def dva(ene_discounted: np.ndarray, pd_bank: np.ndarray, lgd_bank: float) -> tuple[float, np.ndarray]:
    """``(dva, terms)``: ``DVA = sum PD_Bank LGD_Bank ENE(t_i)`` with the bank's own
    default probabilities and the discounted ENE (dossier 05 C.2), a positive benefit; zero with a zero bank spread."""
    return _loss(ene_discounted, pd_bank, lgd_bank, "ene_discounted", "pd_bank", "lgd_bank")


def b_from_two_points(w1: float, hazard1: float, w2: float, hazard2: float) -> float:
    """``b = (ln h2 - ln h1) / (w2 - w1)`` for a flat hazard term
    structure (Hull and White 2012 page 16), per unit of ``w`` (USD million in the paper); ``h1 = h2`` gives
    0."""
    a, b = scalar(w1, "w1"), scalar(w2, "w2")
    h1, h2 = scalar(hazard1, "hazard1"), scalar(hazard2, "hazard2")
    if h1 <= 0.0:
        raise ValueError("hazard1 must be positive")
    if h2 <= 0.0:
        raise ValueError("hazard2 must be positive")
    if a == b:
        raise ValueError("w2 must differ from w1")
    return (math.log(h2) - math.log(h1)) / (b - a)


def wwr_calibrate_a(w: np.ndarray, survival_target: np.ndarray, dt: np.ndarray, b: float, sigma_noise: float = 0.0,
                    eps: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(a, h)`` with ``a`` (K,) and the hazards ``h`` (P, K) per year:
    ``h_ij = exp(a_i + b w_ij + sigma eps_ij)``. Sequential calibration (Hull
    and White Appendix, page 22): for each ``k`` the average over paths of
    ``exp(-sum_{i <= k} h_ij dt_i)`` equals ``survival_target[k]``; ``k = 1`` by a root
    search from ``w_1j``, then ``k = 2`` from ``w_2j`` and the ``h_1j``, and so on. ``w`` is the simulated
    portfolio value in the unit ``b`` is stated in and ``survival_target`` the spread-implied survival
    ``exp(-s_k t_k / (1 - R))``. With ``b = 0`` every path has the same hazard. A
    target that nonnegative hazards cannot reach (the survival would have to rise) is refused."""
    w_ = mat(w, "w")
    p, k = w_.shape
    tgt, d = vec(survival_target, "survival_target"), vec(dt, "dt")
    if tgt.size != k:
        raise ValueError(f"survival_target must have {k} entries (one per column of w), got {tgt.size}")
    if d.size != k:
        raise ValueError(f"dt must have {k} entries (one per column of w), got {d.size}")
    if np.any(d <= 0.0):
        raise ValueError("dt must be positive")
    if np.any(tgt <= 0.0) or np.any(tgt > 1.0):
        raise ValueError("survival_target must lie in (0, 1]")
    b_, noise = scalar(b, "b"), scalar(sigma_noise, "sigma_noise")
    if noise < 0.0:
        raise ValueError("sigma_noise must be nonnegative")
    shift = b_ * w_
    if noise > 0.0:
        if eps is None:
            raise ValueError("eps is required when sigma_noise is positive")
        e = mat(eps, "eps")
        if e.shape != w_.shape:
            raise ValueError(f"eps must have the shape of w {w_.shape}, got {e.shape}")
        shift = shift + noise * e
    a = np.empty(k)
    h = np.empty((p, k))
    cum = np.zeros(p)
    for i in range(k):
        base = float(np.mean(np.exp(-cum)))
        if not tgt[i] < base:
            raise ValueError(f"survival_target[{i}] = {tgt[i]} cannot be reached with nonnegative hazards: the average "
                             f"survival before the step is already {base}")

        def gap(x: float, i: int = i, cum: np.ndarray = cum) -> float:
            with np.errstate(over="ignore"):
                return float(np.mean(np.exp(-cum - d[i] * np.exp(x + shift[:, i])))) - float(tgt[i])

        lo, hi = LOG_A_BRACKET
        grow = 0
        while gap(hi) > 0.0 and grow < 40:
            hi += 5.0
            grow += 1
        if gap(lo) <= 0.0 or gap(hi) > 0.0:
            raise ValueError(f"survival_target[{i}] = {tgt[i]} is not reachable within the search bracket")
        a[i] = brentq(gap, lo, hi, xtol=1e-15, maxiter=500)
        with np.errstate(over="ignore"):
            h[:, i] = np.exp(a[i] + shift[:, i])
        cum = cum + h[:, i] * d[i]
    return a, h


def wwr_cva(w: np.ndarray, exposure_discounted: np.ndarray, survival_target: np.ndarray, dt: np.ndarray, lgd: float,
            b: float) -> tuple[float, float]:
    """``(cva, mc_se)`` of the Hull and White model:
    ``CVA = LGD sum_i mean_j [PD_ij E_ij]`` with ``PD_ij`` the path-wise default
    probability of bucket ``i``, ``S_ij(t_{i-1}) - S_ij(t_i)``, from the hazards of ``wwr_calibrate_a``;
    ``E_ij`` is the discounted positive exposure (P, K) and ``mc_se`` the standard error of the path mean. With
    ``b = 0`` the hazard does not depend on the path and the value is that of the independent ``cva`` on the
    same paths, computed by it (bit for bit). Wrong-way risk (``b > 0`` with a positively correlated exposure)
    raises CVA."""
    w_ = mat(w, "w")
    e = mat(exposure_discounted, "exposure_discounted")
    if e.shape != w_.shape:
        raise ValueError(f"exposure_discounted must have the shape of w {w_.shape}, got {e.shape}")
    if np.any(e < 0.0):
        raise ValueError("exposure_discounted must be nonnegative")
    lg = _lgd(lgd, "lgd")
    tgt = vec(survival_target, "survival_target")
    p = w_.shape[0]
    if scalar(b, "b") == 0.0:
        if tgt.size != w_.shape[1]:
            raise ValueError(f"survival_target must have {w_.shape[1]} entries, got {tgt.size}")
        pd = default_probabilities(np.concatenate([[1.0], tgt]))
        value, _ = cva(e.mean(axis=0), pd, lg)
        per_path = lg * (e @ pd)
    else:
        _, h = wwr_calibrate_a(w_, tgt, dt, b)
        surv = np.exp(-np.cumsum(h * vec(dt, "dt")[None, :], axis=1))
        full = np.hstack([np.ones((p, 1)), surv])
        per_path = lg * np.sum((full[:, :-1] - full[:, 1:]) * e, axis=1)
        value = float(per_path.mean())
    return value, float(per_path.std(ddof=1) / math.sqrt(p))
