"""Rung 9: the dynamic semiparametric VaR and ES models of Patton, Ziegel and Chen (2019), estimated by minimising the
average FZ0 loss (dossier 05 A.1 and A.2 row 12).

Recursions transcribed from the preprint (arXiv 1707.05108, J. Econometrics 211(2):388-413), with the FZ0 score of its
eqs 10 to 12 and 18, in the return convention (`v`, `e` negative, `e <= v`):

- two-factor GAS (eqs 9 to 16): `(v, e)_{t+1} = w + B (v, e)_t + A (lam_v, lam_e)_t` with `lam_v = -v_t (1{y_t <= v_t}
- alpha)`
  and `lam_e = (1/alpha) 1{y_t <= v_t} y_t - e_t`; `B` diagonal (Table 7 reports one persistence per equation) and `A` a
  full 2 x 2 matrix; `params = [w_v, w_e, B_vv, B_ee, A_vv, A_ve, A_ev, A_ee]`; the start is the sample `(v, e)`.
- one-factor GAS (eqs 17 to 20): `v_t = a exp(kappa_t)`, `e_t = b exp(kappa_t)`, `b < a < 0`,
  `kappa_{t+1} = beta kappa_t + gamma (-1/e_t)((1/alpha) 1{y_t <= v_t} y_t - e_t)`; the intercept is not identified
  and is fixed at 0 (the paper does the same); `kappa_1 = 0`; `params = [beta, gamma, a, b]`.
- hybrid GAS/GARCH (eq 27, the "hybrid GARCH-FZ" of the contract): the one-factor recursion plus `delta log|y_t|`,
  `kappa_{t+1} = beta kappa_t + gamma (-1/e_t)(...) + delta log|y_t|`, intercept 0, `kappa_1 = delta m / (1 - beta)`
  with `m` the sample mean of `log|y|`; `params = [beta, gamma, delta, a, b]`.

The objective is the average FZ0 loss over the estimation sample; a path that leaves `e <= v < 0` costs a large
constant, so the optimiser stays in the domain. Each fit minimises from `n_starts` starts (the first is the constant
empirical VaR and ES, which makes the benchmark nested) by Nelder-Mead, deterministic for the same `seed`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from riskvalidation.market.conventions import check_alpha, var_es_from_sample, vec

SPECS = ("one-factor", "two-factor")
MIN_OBS = 500
_BIG = 1e9
_EPS_LOG = 1e-8


@dataclass(frozen=True)
class GasFzFit:
    spec: str                    # "one-factor", "two-factor" or "hybrid"
    alpha: float
    params: np.ndarray
    mean_fz0: float              # the minimised average FZ0 loss on the estimation sample
    n_starts: int
    converged: bool


def _fz0(y: float, v: float, e: float, alpha: float, hit: bool) -> float:
    return (-(v - y) / (alpha * e) if hit else 0.0) + v / e + math.log(-e) - 1.0


def _run_one_factor(params: np.ndarray, y: np.ndarray, alpha: float, hybrid: bool,
                    kappa0: float) -> tuple[float, float, float]:
    """The one-factor or hybrid recursion: (average FZ0 over y, v_{N+1}, e_{N+1}); `inf` off the domain."""
    if hybrid:
        beta, gamma, delta, a, b = (float(p) for p in params)
    else:
        beta, gamma, a, b = (float(p) for p in params)
        delta = 0.0
    if not (abs(beta) < 1.0 and b <= a < 0.0):
        return math.inf, math.nan, math.nan
    kappa, total = kappa0, 0.0
    try:
        for yt in y.tolist():
            ek = math.exp(kappa)
            v, e = a * ek, b * ek
            if not (e <= v < 0.0):
                return math.inf, math.nan, math.nan
            hit = yt <= v
            total += _fz0(yt, v, e, alpha, hit)
            s = (-1.0 / e) * ((yt / alpha if hit else 0.0) - e)
            kappa = beta * kappa + gamma * s + (delta * math.log(max(abs(yt), _EPS_LOG)) if hybrid else 0.0)
        ek = math.exp(kappa)
    except (OverflowError, ValueError, ZeroDivisionError):
        return math.inf, math.nan, math.nan
    v, e = a * ek, b * ek
    if not (e <= v < 0.0):
        return math.inf, math.nan, math.nan
    return total / y.size, v, e


def _run_two_factor(params: np.ndarray, y: np.ndarray, alpha: float, v0: float,
                    e0: float) -> tuple[float, float, float]:
    w1, w2, b1, b2, a11, a12, a21, a22 = (float(p) for p in params)
    v, e, total = v0, e0, 0.0
    try:
        for yt in y.tolist():
            if not (e <= v < 0.0):
                return math.inf, math.nan, math.nan
            hit = yt <= v
            total += _fz0(yt, v, e, alpha, hit)
            lam_v = -v * ((1.0 if hit else 0.0) - alpha)
            lam_e = (yt / alpha if hit else 0.0) - e
            v, e = w1 + b1 * v + a11 * lam_v + a12 * lam_e, w2 + b2 * e + a21 * lam_v + a22 * lam_e
    except (OverflowError, ValueError, ZeroDivisionError):
        return math.inf, math.nan, math.nan
    if not (e <= v < 0.0):
        return math.inf, math.nan, math.nan
    return total / y.size, v, e


def _one_factor_params(theta: np.ndarray, hybrid: bool) -> np.ndarray:
    """From the unconstrained vector to `[beta, gamma, (delta,) a, b]` with `b <= a < 0`."""
    a = -math.exp(min(theta[-2], 50.0))
    b = a - math.exp(min(theta[-1], 50.0))
    return np.concatenate([theta[:-2], [a, b]])


def _kappa0(params: np.ndarray, y: np.ndarray, hybrid: bool) -> float:
    if not hybrid:
        return 0.0
    beta, delta = float(params[0]), float(params[2])
    m = float(np.mean(np.log(np.maximum(np.abs(y), _EPS_LOG))))
    return delta * m / (1.0 - beta) if abs(beta) < 1.0 else 0.0


def _fit_one_factor(y: np.ndarray, alpha: float, hybrid: bool, n_starts: int,
                    seed: int) -> tuple[np.ndarray, float, bool]:
    v_bar, e_bar = var_es_from_sample(y, alpha)
    pa, pb = math.log(-v_bar), math.log(max(v_bar - e_bar, 1e-12))
    nlead = 3 if hybrid else 2
    rng = np.random.default_rng(seed)
    starts = [np.concatenate([np.zeros(nlead), [pa, pb]])]
    for _ in range(n_starts - 1):
        lead = [rng.uniform(0.5, 0.98), rng.uniform(-0.05, 0.01)] + ([rng.uniform(0.0, 0.05)] if hybrid else [])
        starts.append(np.concatenate([lead, [pa + rng.normal(0, 0.15), pb + rng.normal(0, 0.15)]]))

    def objective(theta: np.ndarray) -> float:
        par = _one_factor_params(theta, hybrid)
        val = _run_one_factor(par, y, alpha, hybrid, _kappa0(par, y, hybrid))[0]
        return val if math.isfinite(val) else _BIG

    steps = np.concatenate([[0.05, 0.02], [0.02] if hybrid else [], [0.1, 0.1]])
    best_t, best_v, best_ok = starts[0], objective(starts[0]), False
    for x0 in starts:
        simplex = np.vstack([x0, x0 + np.diag(steps)])
        res = optimize.minimize(objective, x0, method="Nelder-Mead",
                                options={"initial_simplex": simplex, "maxiter": 400 * len(x0), "maxfev": 500 * len(x0),
                                         "xatol": 1e-7, "fatol": 1e-10})
        if res.fun < best_v:
            best_t, best_v, best_ok = res.x, float(res.fun), bool(res.success)
    return _one_factor_params(best_t, hybrid), best_v, bool(best_ok and best_v < _BIG)


def _fit_two_factor(y: np.ndarray, alpha: float, n_starts: int, seed: int) -> tuple[np.ndarray, float, bool]:
    v_bar, e_bar = var_es_from_sample(y, alpha)
    rng = np.random.default_rng(seed)
    starts = [np.array([v_bar, e_bar, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    for _ in range(n_starts - 1):
        b1, b2 = rng.uniform(0.7, 0.98, size=2)
        a = rng.normal(0.0, 0.02, size=4)
        starts.append(np.array([(1 - b1) * v_bar, (1 - b2) * e_bar, b1, b2, *a]))

    def objective(p: np.ndarray) -> float:
        val = _run_two_factor(p, y, alpha, v_bar, e_bar)[0]
        return val if math.isfinite(val) else _BIG

    steps = np.array([abs(v_bar) * 0.1, abs(e_bar) * 0.1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02])
    best_p, best_v, best_ok = starts[0], objective(starts[0]), False
    for x0 in starts:
        simplex = np.vstack([x0, x0 + np.diag(steps)])
        res = optimize.minimize(objective, x0, method="Nelder-Mead",
                                options={"initial_simplex": simplex, "maxiter": 300 * len(x0), "maxfev": 400 * len(x0),
                                         "xatol": 1e-8, "fatol": 1e-10})
        if res.fun < best_v:
            best_p, best_v, best_ok = res.x, float(res.fun), bool(res.success)
    return best_p, best_v, bool(best_ok and best_v < _BIG)


def _checks(y: np.ndarray, alpha: float, n_starts: int) -> tuple[np.ndarray, float]:
    alpha = check_alpha(alpha)
    y = vec(y, "y")
    if y.size < MIN_OBS:
        raise ValueError(f"the GAS-FZ fits need at least {MIN_OBS} returns, got {y.size}")
    if n_starts < 2:
        raise ValueError("n_starts must be at least 2")
    return y, alpha


def fit_gas_fz(y: np.ndarray, alpha: float, spec: str = "one-factor", n_starts: int = 20, seed: int = 0) -> GasFzFit:
    """The best of `n_starts` starts of the average-FZ0 minimisation; `mean_fz0` does not exceed that of the constant
    empirical VaR and ES (the first start, a nested benchmark). `converged = False` is a result, not an error (the
    estimation is unstable for small alpha). Same inputs and seed, same fit. `N >= 500`."""
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS} (use fit_hybrid_garch_fz for the hybrid), got {spec!r}")
    y, alpha = _checks(y, alpha, n_starts)
    if spec == "one-factor":
        params, val, ok = _fit_one_factor(y, alpha, False, n_starts, seed)
    else:
        params, val, ok = _fit_two_factor(y, alpha, n_starts, seed)
    return GasFzFit(spec=spec, alpha=alpha, params=np.asarray(params, dtype=float), mean_fz0=float(val),
                    n_starts=int(n_starts), converged=ok)


def fit_hybrid_garch_fz(y: np.ndarray, alpha: float, n_starts: int = 20, seed: int = 0) -> GasFzFit:
    """The hybrid GAS/GARCH model of Patton, Ziegel and Chen (2019, eq 27) by average-FZ0 minimisation."""
    y, alpha = _checks(y, alpha, n_starts)
    params, val, ok = _fit_one_factor(y, alpha, True, n_starts, seed)
    return GasFzFit(spec="hybrid", alpha=alpha, params=np.asarray(params, dtype=float), mean_fz0=float(val),
                    n_starts=int(n_starts), converged=ok)


def gas_fz_forecast(fit: GasFzFit, y: np.ndarray) -> tuple[float, float]:
    """The `(var, es)` of the next day (return convention): the recursion of the fit run through `y`; nothing beyond
    `y[-1]` is read. ValueError if the path leaves `e <= v < 0` (a fit that did not converge can)."""
    alpha = check_alpha(fit.alpha)
    y = vec(y, "y")
    if fit.spec == "two-factor":
        v0, e0 = var_es_from_sample(y, alpha)
        val, v, e = _run_two_factor(fit.params, y, alpha, v0, e0)
    elif fit.spec in ("one-factor", "hybrid"):
        hybrid = fit.spec == "hybrid"
        val, v, e = _run_one_factor(fit.params, y, alpha, hybrid, _kappa0(fit.params, y, hybrid))
    else:
        raise ValueError(f"unknown spec {fit.spec!r}")
    if not math.isfinite(val):
        raise ValueError("the fitted recursion leaves the domain e <= v < 0 on this series")
    return float(v), float(e)
