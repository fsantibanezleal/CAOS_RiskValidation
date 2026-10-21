"""Rung 8: CAViaR (Engle and Manganelli 2004), the ES extension of Taylor (2019) and the dynamic quantile statistic
(dossier 05 A.2 row 11).

Specifications, with `y` the return and `f_t` the VaR of day `t` in the return convention:

- `sav`: `f_t = b1 + b2 f_{t-1} + b3 |y_{t-1}|`
- `as`:  `f_t = b1 + b2 f_{t-1} + b3 (y_{t-1})^+ + b4 (y_{t-1})^-` with `(y)^+ = max(y, 0)` and `(y)^- = -min(y, 0)`.

The recursion starts at the empirical `alpha`-quantile of the estimation sample (`f_1`, the order statistic of
`conventions.var_es_from_sample`). Both recursions are linear filters, run with `scipy.signal.lfilter`. The objective
is the tick loss `sum_t (alpha - 1{y_t < f_t})(y_t - f_t)` (non-convex and non-smooth: several starts, Nelder-Mead
from each).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal, stats

from riskvalidation.market.conventions import check_alpha, var_es_from_sample, vec

SPECS = ("sav", "as")
MIN_OBS = 500
_BIG = 1e12


@dataclass(frozen=True)
class CaviarFit:
    spec: str                    # "sav" or "as"
    alpha: float
    params: np.ndarray           # (3,) for "sav", (4,) for "as"
    tick_loss: float             # the minimised sum of tick losses
    n_starts: int
    converged: bool


def _inputs(y: np.ndarray, spec: str) -> np.ndarray:
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, got {spec!r}")
    return vec(y, "y")


def _regressors(y: np.ndarray, spec: str) -> list[np.ndarray]:
    """The columns multiplying b3 (and b4) at days 2..N, built from the previous day's return."""
    prev = y[:-1]
    if spec == "sav":
        return [np.abs(prev)]
    return [np.maximum(prev, 0.0), -np.minimum(prev, 0.0)]


def _path(params: np.ndarray, y: np.ndarray, spec: str, f1: float) -> np.ndarray:
    """`f_1 .. f_N` and `f_{N+1}`: an array of N+1 values (the last is the forecast for the day after `y`)."""
    cols = _regressors(np.append(y, 0.0), spec)         # day 2 .. N+1 from the returns 1 .. N
    u = params[0] + sum(c * p for c, p in zip(cols, params[2:], strict=True))
    out = np.empty(y.size + 1)
    out[0] = f1
    with np.errstate(over="ignore", invalid="ignore"):
        out[1:], _ = signal.lfilter([1.0], [1.0, -params[1]], u, zi=[params[1] * f1])
    return out


def _tick_sum(y: np.ndarray, f: np.ndarray, alpha: float) -> float:
    return float(np.sum((alpha - (y < f)) * (y - f)))


def fit_caviar(y: np.ndarray, alpha: float, spec: str = "sav", n_starts: int = 20, seed: int = 0) -> CaviarFit:
    """Minimise the tick loss over `n_starts` starts (the first is the constant empirical quantile, the others random
    with the stationary level at the empirical quantile), each polished by two Nelder-Mead runs; deterministic for the
    same `seed`. `converged` is the termination flag of the best start. `N >= 500` (ValueError otherwise)."""
    alpha = check_alpha(alpha)
    y = _inputs(y, spec)
    if y.size < MIN_OBS:
        raise ValueError(f"fit_caviar needs at least {MIN_OBS} returns, got {y.size}")
    if n_starts < 2:
        raise ValueError("n_starts must be at least 2 (the objective is non-convex)")
    q = var_es_from_sample(y, alpha)[0]
    npar = 3 if spec == "sav" else 4
    n_slope = npar - 2
    mean_cols = [float(c.mean()) for c in _regressors(y, spec)]
    rng = np.random.default_rng(seed)

    def objective(p: np.ndarray) -> float:
        f = _path(p, y, spec, q)[:-1]
        if not np.all(np.isfinite(f)):
            return _BIG
        return _tick_sum(y, f, alpha)

    starts = [np.concatenate([[q, 0.0], np.zeros(n_slope)])]
    for _ in range(n_starts - 1):
        b2 = rng.uniform(0.5, 0.99)
        slopes = -rng.uniform(0.05, 0.6, size=n_slope)
        b1 = q * (1.0 - b2) - float(np.dot(slopes, mean_cols))
        starts.append(np.concatenate([[b1, b2], slopes]))
    steps = np.concatenate([[max(abs(q) * 0.1, 1e-6), 0.1], np.full(n_slope, 0.1)])
    best_p, best_v, best_ok = starts[0], objective(starts[0]), False
    for x0 in starts:
        simplex = np.vstack([x0, x0 + np.diag(steps)])
        p, ok = x0, False
        for _ in range(2):
            res = optimize.minimize(objective, p, method="Nelder-Mead",
                                    options={"initial_simplex": simplex, "maxiter": 600 * npar, "maxfev": 900 * npar,
                                             "xatol": 1e-9, "fatol": 1e-12})
            p, ok = res.x, bool(res.success)
            simplex = np.vstack([p, p + np.diag(steps * 0.3)])
        v = objective(p)
        if v < best_v:
            best_p, best_v, best_ok = p, v, ok
    return CaviarFit(spec=spec, alpha=alpha, params=np.asarray(best_p, dtype=float), tick_loss=float(best_v),
                     n_starts=int(n_starts), converged=bool(best_ok and np.isfinite(best_v)))


def caviar_var_forecast(fit: CaviarFit, y: np.ndarray) -> float:
    """The next-day VaR `f_{T+1}` (return convention): the recursion is run through `y` from the empirical quantile of
    `y` to the last day; nothing beyond `y[-1]` is read."""
    y = _inputs(y, fit.spec)
    f1 = var_es_from_sample(y, fit.alpha)[0]
    return float(_path(fit.params, y, fit.spec, f1)[-1])


@dataclass(frozen=True)
class CaviarEsFit:
    """The result of the ES-CAViaR estimation, with the estimator that produced it (decision D-10)."""

    var: float                 # next-day VaR of the tick-loss fit (return convention)
    es: float                  # next-day ES (return convention), at most `var`
    gamma: np.ndarray          # (g0, g1, g2) of the x recursion
    log_likelihood: float      # the asymmetric Laplace log-likelihood at the estimate
    estimator: str             # "two-step": see `caviar_es_fit`


def caviar_es_forecast(fit: CaviarFit, y: np.ndarray) -> float:
    """The next-day ES (return convention) of `caviar_es_fit`, by the ES-CAViaR of Taylor (2019) in a **two-step**
    estimation (decision D-10, review M3): the VaR path is that of the tick-loss fit and only the ES parameters are
    estimated, where Taylor (2019) estimates VaR and ES jointly by the asymmetric Laplace pseudo-likelihood. See
    `caviar_es_fit`."""
    return caviar_es_fit(fit, y).es


def caviar_es_fit(fit: CaviarFit, y: np.ndarray) -> CaviarEsFit:
    """The ES-CAViaR of Taylor (2019, JBES 37(1):121-133), autoregressive form, estimated in **two steps** (decision
    D-10).

        ES_t = VaR_t - x_t,   x_t = g0 + g1 (VaR_{t-1} - r_{t-1}) + g2 x_{t-1}   if r_{t-1} <= VaR_{t-1}, else x_{t-1},

    Step one is `fit_caviar` (the VaR path by tick-loss minimisation); step two, here, fixes that path and estimates
    `g0, g1 >= 0` and `0 <= g2 < 1` (so `x_t >= 0` and `es <= var`) by maximising the asymmetric Laplace likelihood
    `f(r) = ((alpha - 1) / ES_t) exp((r - VaR_t)(alpha - 1{r <= VaR_t}) / (alpha ES_t))`. Taylor (2019) maximises that
    likelihood jointly over the VaR and the ES parameters (arXiv 2603.02357 section 3.2); the two-step estimator is the
    declared deviation, kept so that the ES forecast is consistent with `caviar_var_forecast` (`es <= var`) and the
    result says so in `estimator`. `x_1` is the sample gap `VaR - ES` of `y`. The equations are those restated in
    Section 3.2 of arXiv 2603.02357 (the primary text was not retrieved, see docs/market/04_caviar-and-gas-fz.md).
    Deterministic.
    """
    y = _inputs(y, fit.spec)
    alpha = fit.alpha
    f1, e1 = var_es_from_sample(y, alpha)
    f = _path(fit.params, y, fit.spec, f1)               # f_1 .. f_{T+1}
    hit = y <= f[:-1]
    x0 = max(f1 - e1, 1e-9)
    hit_days = np.flatnonzero(hit)
    gap = (f[:-1] - y)[hit_days]                         # VaR_t - r_t on the hit days

    def x_path(g: np.ndarray) -> np.ndarray:
        """x_1 .. x_{T+1}: x changes only the day after a hit."""
        x = np.empty(f.size)
        cur, start = x0, 0
        for d, gp in zip(hit_days, gap, strict=True):
            x[start:d + 1] = cur
            cur = g[0] + g[1] * gp + g[2] * cur
            start = d + 1
        x[start:] = cur
        return x

    def negll(theta: np.ndarray) -> float:
        g = _gamma(theta)
        es = f - x_path(g)
        if not np.all(es[:-1] < 0.0):
            return _BIG
        e, v = es[:-1], f[:-1]
        ll = np.log((alpha - 1.0) / e) + (y - v) * (alpha - hit) / (alpha * e)
        return float(-np.sum(ll))

    scale = max(x0, 1e-6)
    starts = [np.array([np.log(0.5 * scale), np.log(0.1), 0.0]), np.array([np.log(0.1 * scale), np.log(0.5), 1.0]),
              np.array([np.log(scale), np.log(0.05), -1.0])]
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0s in starts:
            res = optimize.minimize(negll, x0s, method="Nelder-Mead",
                                    options={"maxiter": 1200, "xatol": 1e-9, "fatol": 1e-12})
            if best is None or res.fun < best.fun:
                best = res
    g = _gamma(best.x)
    x = x_path(g)
    return CaviarEsFit(var=float(f[-1]), es=float(f[-1] - x[-1]), gamma=g, log_likelihood=float(-best.fun),
                       estimator="two-step")


def _gamma(theta: np.ndarray) -> np.ndarray:
    return np.array([np.exp(theta[0]), np.exp(theta[1]), 1.0 / (1.0 + np.exp(-theta[2]))])


def dynamic_quantile_statistic(hits: np.ndarray, var: np.ndarray, alpha: float, lags: int = 4) -> tuple[float, float]:
    """The dynamic quantile (DQ) statistic of Engle and Manganelli (2004): the Wald-type quadratic form of the
    regression of `hits - alpha` on a constant, `lags` lagged hits and the contemporaneous VaR, `Hit'X (X'X)^+ X'Hit /
    (alpha (1 - alpha))`, asymptotically chi-square with the rank of `X` degrees of freedom (`lags + 2` in general).
    The first `lags` days only supply the lags. `p_value` in [0, 1]. Invariant to an affine change of `var`, so the
    sign convention of the VaR does not
    matter."""
    alpha = check_alpha(alpha)
    h = vec(hits, "hits")
    v = vec(var, "var")
    if h.shape != v.shape:
        raise ValueError(f"hits and var must have the same length, got {h.size} and {v.size}")
    if not np.all((h == 0.0) | (h == 1.0)):
        raise ValueError("hits must contain only 0 and 1")
    if lags < 0 or h.size - lags < 2:
        raise ValueError(f"need more than lags + 1 = {lags + 1} observations, got {h.size}")
    stat, rank = dq_statistic_unchecked(h, v, alpha, int(lags))
    return stat, float(stats.chi2.sf(stat, max(rank, 1)))


def dq_statistic_unchecked(h: np.ndarray, v: np.ndarray, alpha: float, lags: int) -> tuple[float, int]:
    """The DQ quadratic form and the rank of `X` from validated float arrays (the Monte Carlo p-value's inner loop).

    `Hit'X (X'X)^+ X'Hit` through the eigendecomposition of the small matrix `X'X` (a direction is dropped when its
    eigenvalue is below 1e-12 of the largest, which removes the exact collinearities: a constant VaR, no hit)."""
    n = h.size
    cols = [np.ones(n - lags)] + [h[lags - j:n - j] for j in range(1, lags + 1)] + [v[lags:]]
    x = np.column_stack(cols)
    y = h[lags:] - alpha
    w, vecs = np.linalg.eigh(x.T @ x)
    keep = w > w.max() * 1e-12
    proj = vecs[:, keep].T @ (x.T @ y)
    return float(np.sum(proj**2 / w[keep])) / (alpha * (1.0 - alpha)), int(keep.sum())

__all__ = [
    "CaviarEsFit",
    "CaviarFit",
    "caviar_es_fit",
    "caviar_es_forecast",
    "caviar_var_forecast",
    "dq_statistic_unchecked",
    "dynamic_quantile_statistic",
    "fit_caviar",
]
