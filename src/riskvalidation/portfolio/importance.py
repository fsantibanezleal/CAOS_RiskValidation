"""Importance sampling for the Gaussian copula (dossier 04 H.7; Glasserman and Li 2005, doi:10.1287/mnsc.1050.0415;
rung C6).

Given the systematic factor z, each conditional default probability p_i(z) is twisted to

    p_{i,theta}(z) = p_i(z) e^{theta c_i} / (1 + p_i(z) (e^{theta c_i} - 1)),   c_i = EAD_i LGD_i,

with theta = theta(z) the root of sum_i c_i p_{i,theta}(z) = x (the twisted conditional mean loss equals the threshold
x; theta = 0 where the untwisted mean is already at or above x). The factor is drawn with mean mu instead of 0. The
likelihood ratio dP/dQ of one scenario is

    exp(-theta L + sum_i ln(1 + p_i (e^{theta c_i} - 1))) exp(-mu' z + mu' mu / 2),

so that E_Q[w] = 1 and E_Q[w L] = EL for ANY theta(z) >= 0 and ANY mu: unbiasedness does not depend on how the twist
or the shift are chosen, only the variance does.

UNVERIFIED (dossier 04 H.7): the transcription of the twist and of the factor mean shift (the mode of the
zero-variance density) was not checked against the paper, which is paywalled. This module therefore accepts nothing on
trust: its acceptance is the unbiasedness against plain Monte Carlo (the weights average 1, the weighted loss averages
the EL, the weighted tail probability, VaR and ES agree with `simulate_gaussian`), tested for several thresholds and
shifts. The default shift maximises -theta(z) x + ln-mgf(theta(z), z) - z'z / 2 over z by Nelder-Mead from z = 0
(deterministic).

Only the Gaussian copula with uncorrelated factors is supported; the multi-factor twist is a failure mode of the ladder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, ndtr, ndtri

from riskvalidation.portfolio.portfolio import (
    QUANTILE_GUARD,
    Portfolio,
    check_level,
    float_vector,
    idiosyncratic_scale,
    positive_int,
)

THETA_TOL = 1e-12
BATCHES = 20
MIN_PER_BATCH = 10


def _logit(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(p) - np.log1p(-p)


def _twist(logit_p: np.ndarray, c: np.ndarray, theta: np.ndarray | float) -> np.ndarray:
    return expit(logit_p + np.asarray(theta) * c)


def twisted_pd(p: np.ndarray, c: np.ndarray, theta: float) -> np.ndarray:
    """p e^{theta c} / (1 + p (e^{theta c} - 1)); equals p at theta 0, stays in [0, 1]; p of 0 or 1 stays there."""
    pv, cv = float_vector(p, "p"), float_vector(c, "c")
    if pv.shape != cv.shape:
        raise ValueError("p and c must have the same length")
    if np.any(pv < 0.0) or np.any(pv > 1.0):
        raise ValueError("p must lie in [0, 1]")
    if np.any(cv < 0.0):
        raise ValueError("c must be nonnegative")
    th = float(theta)
    if not (math.isfinite(th) and th >= 0.0):
        raise ValueError("theta must be a nonnegative number")
    if th == 0.0:
        return pv.copy()
    return _twist(_logit(pv), cv, th)


def solve_theta(p: np.ndarray, c: np.ndarray, threshold: float) -> float:
    """The root theta >= 0 of sum_i c_i p_{theta,i} = threshold; 0.0 when sum c p >= threshold.

    Raises ValueError when the threshold is unreachable (it is at or above sum c, or above the loss of the names whose
    conditional PD is positive). Bracketed root finding with a fixed tolerance of 1e-12: deterministic.
    """
    pv, cv = float_vector(p, "p"), float_vector(c, "c")
    if pv.shape != cv.shape:
        raise ValueError("p and c must have the same length")
    x = float(threshold)
    if not math.isfinite(x):
        raise ValueError("threshold must be finite")
    if np.any(pv < 0.0) or np.any(pv > 1.0) or np.any(cv < 0.0):
        raise ValueError("p must lie in [0, 1] and c must be nonnegative")
    if x >= float(cv.sum()):
        raise ValueError(f"the threshold {x} is unreachable: it is at or above the total loss {float(cv.sum())}")
    if float(cv @ pv) >= x:
        return 0.0
    lp = _logit(pv)

    def gap(theta: float) -> float:
        return float(cv @ _twist(lp, cv, theta)) - x

    hi = 1.0 / max(float(cv.max()), 1e-300)
    for _ in range(80):
        if gap(hi) > 0.0:
            return float(brentq(gap, 0.0, hi, xtol=THETA_TOL))
        hi *= 2.0
    raise ValueError("the threshold is unreachable: no finite theta twists the mean loss up to it")


@dataclass(frozen=True)
class ImportanceResult:
    """Losses (money) drawn under the twisted measure, their likelihood ratios dP/dQ and how they were drawn."""

    losses: np.ndarray
    weights: np.ndarray
    factor_shift: np.ndarray
    n_scenarios: int
    seed: int
    chunk_scenarios: int


def _theta_rows(logit_p: np.ndarray, c: np.ndarray, x: float, mean0: np.ndarray) -> np.ndarray:
    """theta(z) per row: safeguarded Newton on sum c p_theta = x for the rows below the threshold, 0 elsewhere."""
    r = logit_p.shape[0]
    theta = np.zeros(r)
    act = np.flatnonzero(mean0 < x)
    if act.size == 0:
        return theta
    lp = logit_p[act]
    c2 = c * c
    hi = np.full(act.size, 1.0 / max(float(c.max()), 1e-300))
    for _ in range(80):  # bracket: double theta until the twisted mean passes the threshold (or give up at the cap)
        below = _twist(lp, c, hi[:, None]) @ c <= x
        if not below.any():
            break
        hi[below] *= 2.0
    lo = np.zeros(act.size)
    th = 0.5 * hi
    for _ in range(100):
        tw = _twist(lp, c, th[:, None])
        f = tw @ c - x
        done = np.abs(f) <= 1e-10 * x
        if done.all():
            break
        lo = np.where(f < 0.0, th, lo)
        hi = np.where(f > 0.0, th, hi)
        slope = (tw * (1.0 - tw)) @ c2
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = th - f / slope
        ok = np.isfinite(newton) & (newton > lo) & (newton < hi)
        nxt = np.where(ok, newton, 0.5 * (lo + hi))
        th = np.where(done, th, nxt)
    theta[act] = th
    return theta


def _log_mgf(p: np.ndarray, c: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """sum_i ln(1 + p_i (e^{theta c_i} - 1)) per row, stable."""
    with np.errstate(divide="ignore"):
        return np.logaddexp(np.log1p(-p), np.log(p) + theta[:, None] * c).sum(axis=1)


def _cond_p(pt: Portfolio, z: np.ndarray, idio: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return ndtr((thr - z @ pt.loadings.T) / idio)


def _default_shift(pt: Portfolio, c: np.ndarray, x: float) -> np.ndarray:
    idio = idiosyncratic_scale(pt)
    thr = ndtri(pt.pd)
    f = pt.loadings.shape[1]

    def neg_obj(z: np.ndarray) -> float:
        zz = z[None, :]
        p = _cond_p(pt, zz, idio, thr)
        theta = _theta_rows(_logit(p), c, x, p @ c)
        return float(theta[0] * x - _log_mgf(p, c, theta)[0] + 0.5 * float(z @ z))

    res = minimize(neg_obj, np.zeros(f), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 400 * f})
    return np.asarray(res.x, dtype=float)


def importance_sample_gaussian(p: Portfolio, threshold: float, n_scenarios: int, seed: int,
                               factor_shift: np.ndarray | None = None,
                               chunk_scenarios: int = 100_000) -> ImportanceResult:
    """Importance sampling of the Gaussian copula loss targeting the loss level `threshold` (money).

    `factor_shift` (F,) is the mean of the systematic factor under the sampling measure; None solves it
    deterministically (see the module docstring). Raises ValueError for correlated factors (no multi-factor twist),
    for a threshold outside (0, sum of EAD LGD), for a non-positive count and for a shift of the wrong length.
    """
    s = positive_int(n_scenarios, "n_scenarios")
    chunk = positive_int(chunk_scenarios, "chunk_scenarios")
    n, f = p.loadings.shape
    if not np.allclose(p.factor_corr, np.eye(f), atol=1e-12, rtol=0):
        raise ValueError("importance sampling needs uncorrelated factors: the multi-factor twist for correlated "
                         "factors is not supported")
    c = p.ead * p.lgd
    x = float(threshold)
    if not (math.isfinite(x) and 0.0 < x < float(c.sum())):
        raise ValueError(f"threshold must lie in (0, {float(c.sum())}), got {threshold}")
    if factor_shift is None:
        mu = _default_shift(p, c, x)
    else:
        mu = float_vector(factor_shift, "factor_shift")
        if mu.shape != (f,):
            raise ValueError(f"factor_shift must have shape ({f},), got {mu.shape}")
    seed = int(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    idio = idiosyncratic_scale(p)
    thr = ndtri(p.pd)
    losses = np.empty(s)
    logw = np.empty(s)
    block = max(1, chunk // 6)
    for start in range(0, s, block):
        r = min(block, s - start)
        z = mu[None, :] + rng.standard_normal((r, f))
        pc = _cond_p(p, z, idio, thr)
        lp = _logit(pc)
        theta = _theta_rows(lp, c, x, pc @ c)
        tw = _twist(lp, c, theta[:, None])
        defaulted = rng.random((r, n)) < tw
        loss = defaulted @ c
        losses[start:start + r] = loss
        logw[start:start + r] = -theta * loss + _log_mgf(pc, c, theta) - z @ mu + 0.5 * float(mu @ mu)
    return ImportanceResult(losses=losses, weights=np.exp(logw), factor_shift=np.asarray(mu, dtype=float).copy(),
                            n_scenarios=s, seed=seed, chunk_scenarios=chunk)


def _weighted_var_es(losses: np.ndarray, weights: np.ndarray, q: float) -> tuple[float, float]:
    """The tail estimators built on the upper tail probability P(L > x) = sum_{L_i > x} w_i / S.

    The tail is normalised by the scenario count S, as in the contract's sum(w 1{L > x}) / S, never by the realised
    sum of the weights: the weights average 1 only in expectation, the bulk of an importance sampling sample is
    sparse and its weights are noisy, and at a threshold far in the tail the realised sum is far below S (mean
    weight 0.034 in the review). The quantile is the smallest loss whose upper tail probability is at most 1 - q,
    and the boundary scenario carries the fractional tail mass, so that the tail has exactly mass 1 - q.
    """
    s = losses.size
    order = np.argsort(losses, kind="stable")
    x, w = losses[order], weights[order]
    above = (w.sum() - np.cumsum(w)) / s  # upper tail probability P(L > x_k) for each sorted scenario
    tail_mass = 1.0 - q
    if float(w.sum()) / s < tail_mass:
        raise ValueError(f"the weights account for a total mass {float(w.sum()) / s:.6f} of the scenario count, below "
                         f"the tail mass {tail_mass:.6f}: use more scenarios or a lower threshold")
    k = int(np.argmax(above <= tail_mass + QUANTILE_GUARD / s))
    tail = float((w[k + 1:] * x[k + 1:]).sum()) / s
    boundary = max(0.0, tail_mass - float(above[k])) * float(x[k])
    return float(x[k]), (tail + boundary) / tail_mass


def weighted_var_es(res: ImportanceResult, q: float) -> tuple[float, float, float, float]:
    """(var, es, se_var, se_es): the smallest loss whose weighted upper tail probability sum(w 1{L > x}) / S is at
    most 1 - q, and the weighted Acerbi-Tasche ES, both normalised by the scenario count S (not by the realised sum of
    the weights); the standard errors by the batch-means estimator on 20 consecutive batches, NaN when the sample
    is too short for 20 batches of 10. With unit weights they equal the plain estimators. Raises ValueError when
    the weights cannot account for the tail mass 1 - q."""
    q = check_level(q)
    x = float_vector(res.losses, "losses")
    w = float_vector(res.weights, "weights")
    if x.shape != w.shape or np.any(w <= 0.0):
        raise ValueError("losses and weights must have the same length and the weights must be positive")
    var, es = _weighted_var_es(x, w, q)
    per = x.size // BATCHES
    if per < MIN_PER_BATCH:
        return var, es, float("nan"), float("nan")
    est = np.array([_weighted_var_es(x[i * per:(i + 1) * per], w[i * per:(i + 1) * per], q) for i in range(BATCHES)])
    root = math.sqrt(BATCHES)
    return var, es, float(np.std(est[:, 0], ddof=1) / root), float(np.std(est[:, 1], ddof=1) / root)


def variance_ratio(plain_se: float, is_se: float) -> float:
    """(plain_se / is_se)^2: the variance reduction at equal sample size; raises for a non-positive is_se."""
    a, b = float(plain_se), float(is_se)
    if not (math.isfinite(b) and b > 0.0):
        raise ValueError("is_se must be positive")
    if not (math.isfinite(a) and a >= 0.0):
        raise ValueError("plain_se must be nonnegative")
    return (a / b) ** 2
