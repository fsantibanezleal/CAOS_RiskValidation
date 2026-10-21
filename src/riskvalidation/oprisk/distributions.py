"""Frequency and severity laws of dossier 05 F.2, their truncation above the collection threshold and their fits.

Truncation: internal data are collected above a threshold ``H``, so the severity is fitted as ``f(x) / (1 - F(H))`` and
the frequency is rescaled, ``lam_full = lam_obs / (1 - F(H))``; ignoring this biases capital (Mignola and Ugoccioni
2007; Cope et al. 2009, dossier 05 F.2). Severity laws are named with a parameter tuple in a fixed order:
``"lognormal"`` ``(mu, sigma)``; ``"weibull"`` ``(shape, scale)``; ``"gpd"`` ``(xi, beta, location)`` with ``Y = X -
location`` generalised Pareto, ``G(y) = 1 - (1 + xi y / beta)^(-1/xi)`` and ``xi > 0``; ``"lognormal-gpd-splice"``
``(mu, sigma, u, xi, beta, w)``, the lognormal body below the splice threshold ``u``, the GPD tail above it and
``w = P(X <= u)``. Frequency laws: ``"poisson"`` ``(lam,)``, ``"negative-binomial"`` ``(r, q)`` with
``p_n = Gamma(n + r) / (n! Gamma(r)) q^r (1 - q)^n`` and ``"binomial"`` ``(m, q)``.

A tail index of 1 or more has no mean: ``severity_moment`` returns ``math.inf`` for a moment that does not exist.
Money is in the caller's currency; the engine never converts. Every function is pure and deterministic.
Contract: ``engine-oprisk-api`` 1.0.0, module distributions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.special import comb, gammaln, log_ndtr, ndtr, ndtri

from riskvalidation.oprisk._checks import params_of, positive, scalar, spread_is_zero, vec

FREQUENCY_LAWS = ("poisson", "negative-binomial", "binomial")
SEVERITY_LAWS = ("lognormal", "weibull", "gpd", "lognormal-gpd-splice")
FITTED_SEVERITY_LAWS = ("lognormal", "weibull", "gpd")
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_INF = math.inf
_NAN = math.nan


# ----------------------------------------------------------------------------------------------------------------
# parameter validation


def _freq_params(law: str, params: Sequence[float]) -> tuple[float, ...]:
    if law == "poisson":
        (lam,) = params_of(params, 1, law)
        positive(lam, "lam")
        return (lam,)
    if law == "negative-binomial":
        r, q = params_of(params, 2, law)
        positive(r, "r")
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must lie in (0, 1), got {q}")
        return (r, q)
    if law == "binomial":
        m, q = params_of(params, 2, law)
        if m < 1 or m != math.floor(m):
            raise ValueError(f"m must be a positive integer, got {m}")
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must lie in (0, 1), got {q}")
        return (m, q)
    raise ValueError(f"unknown frequency law {law!r}; expected one of {FREQUENCY_LAWS}")


def _sev_params(law: str, params: Sequence[float]) -> tuple[float, ...]:
    if law == "lognormal":
        mu, sigma = params_of(params, 2, law)
        positive(sigma, "sigma")
        return (mu, sigma)
    if law == "weibull":
        shape, scale = params_of(params, 2, law)
        positive(shape, "shape")
        positive(scale, "scale")
        return (shape, scale)
    if law == "gpd":
        xi, beta, loc = params_of(params, 3, law)
        positive(xi, "xi")
        positive(beta, "beta")
        return (xi, beta, loc)
    if law == "lognormal-gpd-splice":
        mu, sigma, u, xi, beta, w = params_of(params, 6, law)
        positive(sigma, "sigma")
        positive(u, "u")
        positive(xi, "xi")
        positive(beta, "beta")
        if not 0.0 < w < 1.0:
            raise ValueError(f"the body weight w must lie in (0, 1), got {w}")
        return (mu, sigma, u, xi, beta, w)
    raise ValueError(f"unknown severity law {law!r}; expected one of {SEVERITY_LAWS}")


# ----------------------------------------------------------------------------------------------------------------
# severity primitives on validated parameters: cdf, survival function, log density, quantile, inverse survival


def _ln_cdf(p: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    pos = x > 0.0
    out[pos] = ndtr((np.log(x[pos]) - p[0]) / p[1])
    return out


def _ln_sf(p: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    pos = x > 0.0
    out[pos] = ndtr(-(np.log(x[pos]) - p[0]) / p[1])
    return out


def _wb_cdf(p: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    pos = x > 0.0
    with np.errstate(over="ignore"):
        out[pos] = -np.expm1(-((x[pos] / p[1]) ** p[0]))
    return out


def _wb_sf(p: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    pos = x > 0.0
    with np.errstate(over="ignore"):
        out[pos] = np.exp(-((x[pos] / p[1]) ** p[0]))
    return out


def _gpd_y_sf(xi: float, beta: float, y: np.ndarray) -> np.ndarray:
    out = np.ones_like(y)
    pos = y > 0.0
    with np.errstate(over="ignore"):
        out[pos] = np.exp(-np.log1p(xi * y[pos] / beta) / xi)
    return out


def _gpd_y_cdf(xi: float, beta: float, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    pos = y > 0.0
    with np.errstate(over="ignore"):
        out[pos] = -np.expm1(-np.log1p(xi * y[pos] / beta) / xi)
    return out


def _cdf(law: str, p: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    if law == "lognormal":
        return _ln_cdf(p, x)
    if law == "weibull":
        return _wb_cdf(p, x)
    if law == "gpd":
        return _gpd_y_cdf(p[0], p[1], x - p[2])
    mu, sigma, u, xi, beta, w = p
    body = (mu, sigma)
    out = np.zeros_like(x)
    below = x <= u
    fu = float(ndtr((math.log(u) - mu) / sigma))
    out[below] = w * _ln_cdf(body, x[below]) / fu
    out[~below] = w + (1.0 - w) * _gpd_y_cdf(xi, beta, x[~below] - u)
    return out


def _sf(law: str, p: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    if law == "lognormal":
        return _ln_sf(p, x)
    if law == "weibull":
        return _wb_sf(p, x)
    if law == "gpd":
        return _gpd_y_sf(p[0], p[1], x - p[2])
    mu, sigma, u, xi, beta, w = p
    out = np.ones_like(x)
    below = x <= u
    fu = float(ndtr((math.log(u) - mu) / sigma))
    out[below] = 1.0 - w * _ln_cdf((mu, sigma), x[below]) / fu
    out[~below] = (1.0 - w) * _gpd_y_sf(xi, beta, x[~below] - u)
    return out


def _gpd_ppf_y(xi: float, beta: float, q: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return beta / xi * np.expm1(-xi * np.log1p(-q))


def _ppf(law: str, p: tuple[float, ...], q: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if law == "lognormal":
            return np.exp(p[0] + p[1] * ndtri(q))
        if law == "weibull":
            return p[1] * (-np.log1p(-q)) ** (1.0 / p[0])
        if law == "gpd":
            return p[2] + _gpd_ppf_y(p[0], p[1], q)
        mu, sigma, u, xi, beta, w = p
        fu = float(ndtr((math.log(u) - mu) / sigma))
        body = q <= w
        out = np.empty_like(q)
        out[body] = np.exp(mu + sigma * ndtri(q[body] / w * fu))
        out[~body] = u + _gpd_ppf_y(xi, beta, (q[~body] - w) / (1.0 - w))
        return out


def severity_isf(law: str, params: Sequence[float], tail: np.ndarray) -> np.ndarray:
    """The inverse survival function: the loss ``x`` with ``P(X > x) = tail``, accurate for a small ``tail``.

    ``severity_ppf(q)`` is ``severity_isf(1 - q)`` but loses digits when ``1 - q`` is small; the single loss
    approximation asks for ``tail = (1 - level) / E[N]`` and uses this.
    """
    p = _sev_params(law, params)
    t = np.array(tail, dtype=float, ndmin=1)
    if np.any(~(t >= 0.0)) or np.any(t > 1.0):
        raise ValueError("tail probabilities must lie in [0, 1]")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if law == "lognormal":
            return np.exp(p[0] - p[1] * ndtri(t))
        if law == "weibull":
            return p[1] * (-np.log(t)) ** (1.0 / p[0])
        if law == "gpd":
            return p[2] + p[0] ** -1 * p[1] * np.expm1(-p[0] * np.log(t))
        mu, sigma, u, xi, beta, w = p
        fu = float(ndtr((math.log(u) - mu) / sigma))
        tail_part = t <= 1.0 - w
        out = np.empty_like(t)
        out[tail_part] = u + beta / xi * np.expm1(-xi * np.log(t[tail_part] / (1.0 - w)))
        out[~tail_part] = np.exp(mu + sigma * ndtri((1.0 - t[~tail_part]) / w * fu))
        return out


def _logpdf(law: str, p: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    """Log density of the three fitted laws at ``x`` inside the support."""
    if law == "lognormal":
        z = (np.log(x) - p[0]) / p[1]
        return -np.log(x) - math.log(p[1]) - _LOG_SQRT_2PI - 0.5 * z * z
    if law == "weibull":
        k, lam = p
        return math.log(k) - math.log(lam) + (k - 1.0) * (np.log(x) - math.log(lam)) - (x / lam) ** k
    xi, beta, loc = p
    return -math.log(beta) - (1.0 + 1.0 / xi) * np.log1p(xi * (x - loc) / beta)


def _log_sf_scalar(law: str, p: tuple[float, ...], h: float) -> float:
    """``ln(1 - F(h))`` for a fitted law, stable in the upper tail."""
    if law == "lognormal":
        return float(log_ndtr(-(math.log(h) - p[0]) / p[1])) if h > 0.0 else 0.0
    if law == "weibull":
        return -((h / p[1]) ** p[0]) if h > 0.0 else 0.0
    return 0.0  # the GPD of a fit has its location at the threshold


# ----------------------------------------------------------------------------------------------------------------
# frequency


def frequency_pmf(law: str, params: Sequence[float], n_max: int) -> np.ndarray:
    """``p_0 .. p_n_max`` of a frequency law (counts per year); the tail beyond ``n_max`` is not added."""
    p = _freq_params(law, params)
    if int(n_max) != n_max or n_max < 0:
        raise ValueError(f"n_max must be a non-negative integer, got {n_max}")
    n = np.arange(int(n_max) + 1, dtype=float)
    if law == "poisson":
        return np.exp(n * math.log(p[0]) - p[0] - gammaln(n + 1.0))
    if law == "negative-binomial":
        r, q = p
        return np.exp(gammaln(n + r) - gammaln(n + 1.0) - gammaln(r) + r * math.log(q) + n * math.log1p(-q))
    m, q = p
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = gammaln(m + 1.0) - gammaln(n + 1.0) - gammaln(m - n + 1.0) + n * math.log(q) + (m - n) * math.log1p(-q)
    return np.where(n <= m, np.exp(logp), 0.0)


# ----------------------------------------------------------------------------------------------------------------
# severity


def _no_nan(x: np.ndarray) -> np.ndarray:
    a = np.array(x, dtype=float, ndmin=1)
    if np.any(np.isnan(a)):
        raise ValueError("x contains NaN")
    return a


def severity_cdf(law: str, params: Sequence[float], x: np.ndarray) -> np.ndarray:
    """``F(x)``: zero at and below the origin of the law (0, or the location of the GPD)."""
    p = _sev_params(law, params)
    return _cdf(law, p, _no_nan(x))


def severity_ppf(law: str, params: Sequence[float], q: np.ndarray) -> np.ndarray:
    """The inverse of ``severity_cdf``: ``q = 0`` gives the origin and ``q = 1`` gives ``math.inf``."""
    p = _sev_params(law, params)
    qa = np.array(q, dtype=float, ndmin=1)
    if np.any(~(qa >= 0.0)) or np.any(qa > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    return _ppf(law, p, qa)


def _gpd_y_moment(xi: float, beta: float, j: int) -> float:
    """``E[Y^j]`` of the zero-location GPD, finite exactly when ``j xi < 1``."""
    if j * xi >= 1.0:
        return _INF
    value = beta ** j * math.factorial(j)
    for i in range(1, j + 1):
        value /= 1.0 - i * xi
    return value


def _gpd_shifted_moment(xi: float, beta: float, loc: float, k: int) -> float:
    if k * xi >= 1.0:
        return _INF
    return float(sum(comb(k, j) * loc ** (k - j) * _gpd_y_moment(xi, beta, j) for j in range(k + 1)))


def severity_moment(law: str, params: Sequence[float], k: int) -> float:
    """The raw moment ``E[X^k]``, ``k >= 1``; ``math.inf`` when it does not exist (``k xi >= 1`` for the GPD)."""
    p = _sev_params(law, params)
    if int(k) != k or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    k = int(k)
    if law == "lognormal":
        return math.exp(k * p[0] + 0.5 * k * k * p[1] ** 2)
    if law == "weibull":
        return p[1] ** k * math.gamma(1.0 + k / p[0])
    if law == "gpd":
        return _gpd_shifted_moment(p[0], p[1], p[2], k)
    mu, sigma, u, xi, beta, w = p
    tail = _gpd_shifted_moment(xi, beta, u, k)
    if math.isinf(tail):
        return _INF
    fu = float(ndtr((math.log(u) - mu) / sigma))
    cut = float(ndtr((math.log(u) - mu - k * sigma ** 2) / sigma))
    body = w / fu * math.exp(k * mu + 0.5 * k * k * sigma ** 2) * cut
    return body + (1.0 - w) * tail


def _sf_at_threshold(law: str, p: tuple[float, ...], threshold: float) -> float:
    sf = float(_sf(law, p, np.array([threshold]))[0])
    if not sf > 0.0:
        raise ValueError(f"F(H) = 1 at the threshold {threshold}: no probability mass above it")
    return sf


def truncated_severity_cdf(law: str, params: Sequence[float], x: np.ndarray, threshold: float) -> np.ndarray:
    """``(F(x) - F(H)) / (1 - F(H))`` for ``x >= H`` and 0 below; ``ValueError`` when ``F(H) = 1``."""
    p = _sev_params(law, params)
    h = scalar(threshold, "threshold")
    xa = _no_nan(x)
    sf_h = _sf_at_threshold(law, p, h)
    above = xa >= h
    out = np.zeros_like(xa)
    out[above] = np.clip(1.0 - _sf(law, p, xa[above]) / sf_h, 0.0, 1.0)
    return out


def rescale_frequency(rate_observed: float, law: str, params: Sequence[float], threshold: float) -> float:
    """``lam_full = lam_obs / (1 - F(H))``: losses per year of the full severity (dossier 05 F.2)."""
    rate = scalar(rate_observed, "rate_observed")
    if rate < 0.0:
        raise ValueError(f"rate_observed must not be negative, got {rate}")
    p = _sev_params(law, params)
    return rate / _sf_at_threshold(law, p, scalar(threshold, "threshold"))


# ----------------------------------------------------------------------------------------------------------------
# numerical helpers for the fits


def _hessian(f: Callable[[np.ndarray], float], x: np.ndarray, scale: np.ndarray, rel: float = 2e-4) -> np.ndarray:
    """Central-difference Hessian of ``f`` at ``x`` with steps ``rel * scale``."""
    k = x.size
    h = rel * scale
    hess = np.empty((k, k))
    f0 = f(x)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        hess[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / (h[i] ** 2)
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            cross = f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            hess[i, j] = hess[j, i] = cross / (4.0 * h[i] * h[j])
    return hess


def _standard_errors(nll: Callable[[np.ndarray], float], x: np.ndarray,
                     scale: np.ndarray) -> tuple[tuple[float, ...], bool]:
    """Standard errors from the inverse observed information; ``nan`` and ``False`` when it is not positive definite."""
    try:
        hess = _hessian(nll, x, scale)
        if not np.all(np.isfinite(hess)):
            raise np.linalg.LinAlgError
        eig = np.linalg.eigvalsh(0.5 * (hess + hess.T))
        if eig.min() <= 1e-12 * max(1.0, abs(eig.max())):
            raise np.linalg.LinAlgError
        cov = np.linalg.inv(hess)
        diag = np.diag(cov)
        if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
            raise np.linalg.LinAlgError
        return tuple(float(v) for v in np.sqrt(diag)), True
    except np.linalg.LinAlgError:
        return tuple(_NAN for _ in range(x.size)), False


def _nelder_mead(fun: Callable[[np.ndarray], float], x0: np.ndarray) -> optimize.OptimizeResult:
    """Nelder-Mead restarted once from its own optimum: deterministic, derivative free, tolerances at 100 machine
    epsilons
    of the objective (a tolerance below the roundoff of the objective can never be met)."""
    f0 = abs(float(fun(x0)))
    opts = {"xatol": 1e-10, "fatol": 1e-14 * max(1.0, f0), "maxiter": 20000, "maxfev": 20000}
    res = optimize.minimize(fun, x0, method="Nelder-Mead", options=opts)
    res2 = optimize.minimize(fun, res.x, method="Nelder-Mead", options=opts)
    return res2 if res2.fun <= res.fun else res


# ----------------------------------------------------------------------------------------------------------------
# GPD maximum likelihood by profiling the scale ratio (Grimshaw's reduction)

_GRID_T = np.logspace(-6.0, 6.0, 241)


def _gpd_profile(theta: float, y: np.ndarray) -> tuple[float, float, float]:
    """(profile log-likelihood, xi, beta) at ``theta = xi / beta`` for the excesses ``y``."""
    xi = float(np.mean(np.log1p(theta * y)))
    if not xi > 0.0:
        return -_INF, 0.0, _NAN
    n = y.size
    return -n * math.log(xi / theta) - n * (xi + 1.0), xi, xi / theta


def _fit_gpd_excess(y: np.ndarray) -> tuple[float, float, bool]:
    """Maximum likelihood ``(xi, beta, interior)`` of the GPD (``xi > 0``) for the excesses ``y >= 0``.

    ``interior`` is False when the optimum sits on the boundary of the search range (the exponential limit
    ``xi -> 0`` or an unbounded tail index) or the sample has no spread: the pair is then the boundary value.
    """
    ybar = float(np.mean(y))
    if y.size < 2 or not ybar > 0.0:
        return _NAN, _NAN, False
    thetas = _GRID_T / ybar
    values = np.array([_gpd_profile(t, y)[0] for t in thetas])
    best = int(np.argmax(values))
    if best == 0 or best == len(thetas) - 1 or not np.isfinite(values[best]):
        ll, xi, beta = _gpd_profile(float(thetas[best]), y)
        return xi, beta, False
    lo, hi = math.log(thetas[best - 1]), math.log(thetas[best + 1])
    res = optimize.minimize_scalar(lambda s: -_gpd_profile(math.exp(s), y)[0], bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-13})
    _, xi, beta = _gpd_profile(math.exp(res.x), y)
    return xi, beta, True


def _gpd_nll(y: np.ndarray) -> Callable[[np.ndarray], float]:
    n = y.size

    def nll(p: np.ndarray) -> float:
        xi, beta = float(p[0]), float(p[1])
        if xi <= 0.0 or beta <= 0.0:
            return _INF
        return n * math.log(beta) + (1.0 + 1.0 / xi) * float(np.sum(np.log1p(xi * y / beta)))

    return nll


def _fit_gpd(y: np.ndarray) -> tuple[float, float, float, tuple[float, float], bool]:
    """``(xi, beta, loglik, (se_xi, se_beta), converged)`` for the excesses ``y``."""
    xi, beta, interior = _fit_gpd_excess(y)
    if not interior:
        return xi, beta, _NAN, (_NAN, _NAN), False
    nll = _gpd_nll(y)
    se, ok = _standard_errors(nll, np.array([xi, beta]), np.array([xi, beta]))
    return xi, beta, -nll(np.array([xi, beta])), (se[0], se[1]), ok


# ----------------------------------------------------------------------------------------------------------------
# frequency fits


@dataclass(frozen=True)
class FrequencyFit:
    """A fitted frequency law: ``params`` as in the module docstring (``lam`` in losses per year), ``std_errors`` from
    the observed information (same length, nonnegative), ``loglik`` the natural-log likelihood."""

    law: str
    params: tuple[float, ...]
    std_errors: tuple[float, ...]
    n_years: int
    loglik: float


def _counts(counts: np.ndarray) -> np.ndarray:
    c = vec(counts, "counts")
    if c.size < 2:
        raise ValueError("fit_frequency needs at least 2 years of counts")
    if np.any(c < 0.0) or np.any(c != np.floor(c)):
        raise ValueError("counts must be non-negative integers")
    return c


def fit_frequency(counts: np.ndarray, law: str = "poisson") -> FrequencyFit:
    """Maximum likelihood fit of the recorded annual counts (to be rescaled by ``rescale_frequency``).

    Poisson: ``lam = mean(counts)`` and ``se = sqrt(lam / Y)``. Negative binomial: the sample variance must exceed the
    mean, otherwise ``ValueError("underdispersed")`` and the caller uses Poisson. The binomial is not fitted.
    """
    c = _counts(counts)
    y = int(c.size)
    if law == "poisson":
        lam = float(c.mean())
        if not lam > 0.0:
            raise ValueError("all counts are zero: there is no frequency to fit")
        loglik = float(np.sum(c * math.log(lam) - lam - gammaln(c + 1.0)))
        return FrequencyFit("poisson", (lam,), (math.sqrt(lam / y),), y, loglik)
    if law != "negative-binomial":
        raise ValueError(f"fit_frequency supports 'poisson' and 'negative-binomial', got {law!r}")
    mean, var = float(c.mean()), float(c.var(ddof=1))
    if not var > mean:
        raise ValueError("underdispersed: the sample variance does not exceed the mean; use the Poisson law")

    def nll_unconstrained(z: np.ndarray) -> float:
        r, q = math.exp(z[0]), 1.0 / (1.0 + math.exp(-z[1]))
        return -float(np.sum(gammaln(c + r) - gammaln(c + 1.0) - gammaln(r) + r * math.log(q) + c * math.log1p(-q)))

    q0 = mean / var
    r0 = mean * q0 / (1.0 - q0)
    res = _nelder_mead(nll_unconstrained, np.array([math.log(r0), math.log(q0 / (1.0 - q0))]))
    r, q = math.exp(res.x[0]), 1.0 / (1.0 + math.exp(-res.x[1]))

    def nll(p: np.ndarray) -> float:
        rr, qq = float(p[0]), float(p[1])
        if rr <= 0.0 or not 0.0 < qq < 1.0:
            return _INF
        return -float(np.sum(gammaln(c + rr) - gammaln(c + 1.0) - gammaln(rr) + rr * math.log(qq)
                             + c * math.log1p(-qq)))

    se, _ = _standard_errors(nll, np.array([r, q]), np.array([r, q]))
    return FrequencyFit("negative-binomial", (r, q), se, y, -nll(np.array([r, q])))


# ----------------------------------------------------------------------------------------------------------------
# severity fits


@dataclass(frozen=True)
class SeverityFit:
    """A fitted severity law: ``params`` as in the module docstring (EUR for scales and locations), ``threshold`` the
    collection threshold, ``truncated`` whether the likelihood was left-truncated at it, ``std_errors`` from the inverse
    observed information (``nan`` with ``converged = False`` for a parameter on a boundary or a singular information
    matrix; the fixed location of a GPD fit has standard error 0)."""

    law: str
    params: tuple[float, ...]
    std_errors: tuple[float, ...]
    threshold: float
    truncated: bool
    n: int
    loglik: float
    converged: bool


def _fit_lognormal(x: np.ndarray, h: float, truncated: bool) -> SeverityFit:
    lx = np.log(x)
    n = x.size
    mu0 = float(lx.mean())
    sigma0 = float(np.sqrt(np.mean((lx - mu0) ** 2)))
    if spread_is_zero(lx):
        raise ValueError("the losses have no spread: a lognormal cannot be fitted")
    if not truncated or h <= 0.0:
        ll = float(np.sum(_logpdf("lognormal", (mu0, sigma0), x)))
        se = (sigma0 / math.sqrt(n), sigma0 / math.sqrt(2.0 * n))
        return SeverityFit("lognormal", (mu0, sigma0), se, h, truncated, n, ll, True)

    def nll(p: np.ndarray) -> float:
        mu, sigma = float(p[0]), float(p[1])
        if sigma <= 0.0:
            return _INF
        return -(float(np.sum(_logpdf("lognormal", (mu, sigma), x))) - n * _log_sf_scalar("lognormal", (mu, sigma), h))

    res = _nelder_mead(lambda z: nll(np.array([z[0], math.exp(z[1])])), np.array([mu0, math.log(sigma0)]))
    mu, sigma = float(res.x[0]), math.exp(float(res.x[1]))
    p = np.array([mu, sigma])
    se, ok = _standard_errors(nll, p, np.array([max(abs(mu), 1.0), sigma]))
    return SeverityFit("lognormal", (mu, sigma), se, h, True, n, -nll(p), bool(ok and res.success))


def _fit_weibull(x: np.ndarray, h: float, truncated: bool) -> SeverityFit:
    lx = np.log(x)
    n = x.size
    if spread_is_zero(lx):
        raise ValueError("the losses have no spread: a Weibull cannot be fitted")
    k0 = math.pi / (math.sqrt(6.0) * float(lx.std()))
    lam0 = math.exp(float(lx.mean()) + 0.5772156649 / k0)
    cut = h if truncated else 0.0

    def nll(p: np.ndarray) -> float:
        k, lam = float(p[0]), float(p[1])
        if k <= 0.0 or lam <= 0.0:
            return _INF
        return -(float(np.sum(_logpdf("weibull", (k, lam), x))) - n * _log_sf_scalar("weibull", (k, lam), cut))

    res = _nelder_mead(lambda z: nll(np.exp(z)), np.log(np.array([k0, lam0])))
    p = np.exp(res.x)
    se, ok = _standard_errors(nll, p, p.copy())
    return SeverityFit("weibull", (float(p[0]), float(p[1])), se, h, truncated, n, -nll(p), bool(ok and res.success))


def _fit_gpd_severity(x: np.ndarray, h: float, truncated: bool) -> SeverityFit:
    y = x - h
    xi, beta, ll, se, ok = _fit_gpd(y)
    return SeverityFit("gpd", (xi, beta, h), (se[0], se[1], 0.0), h, truncated, int(x.size), ll, ok)


def fit_severity(law: str, losses: np.ndarray, collection_threshold: float = 0.0,
                 truncated: bool = True) -> SeverityFit:
    """Maximum likelihood fit of a severity law to losses recorded at or above ``collection_threshold``.

    With ``truncated = True`` the log-likelihood is ``sum ln f(x_i) - N ln(1 - F(H))`` (dossier 05 F.2); with
    ``truncated = False`` it is ``sum ln f(x_i)`` and the threshold is ignored (rung 1, the biased baseline). The
    untruncated lognormal is closed form (``mu = mean(ln x)``, ``sigma`` with divisor ``N``). For ``"gpd"`` the
    location is the threshold, so the fit is of the excesses over it (the truncation is then a no-op). A loss below the
    threshold raises ``ValueError``.
    """
    if law not in FITTED_SEVERITY_LAWS:
        raise ValueError(f"fit_severity supports {FITTED_SEVERITY_LAWS}, got {law!r}")
    x = vec(losses, "losses")
    h = scalar(collection_threshold, "collection_threshold")
    if h < 0.0:
        raise ValueError(f"collection_threshold must not be negative, got {h}")
    if x.size < 2:
        raise ValueError("fit_severity needs at least 2 losses")
    if np.any(x < h):
        raise ValueError("a loss lies below the collection threshold: below-threshold rows are rejected")
    if law != "gpd" and np.any(x <= 0.0):
        raise ValueError(f"the {law} law needs positive losses")
    if law == "lognormal":
        return _fit_lognormal(x, h, truncated)
    if law == "weibull":
        return _fit_weibull(x, h, truncated)
    return _fit_gpd_severity(x, h, truncated)


# ----------------------------------------------------------------------------------------------------------------
# splice


@dataclass(frozen=True)
class SpliceFit:
    """A fitted lognormal body and GPD tail: ``params = (mu, sigma, u, xi, beta, w)``, ``weight = w``; the standard
    errors of ``u`` (fixed by the caller) and ``w`` (derived from the observed tail share) are ``nan``."""

    params: tuple[float, ...]
    std_errors: tuple[float, ...]
    collection_threshold: float
    splice_threshold: float
    weight: float
    n_body: int
    n_tail: int
    converged: bool


def _log_mass(zl: float, zu: float) -> float:
    """``ln(Phi(zu) - Phi(zl))`` for ``zl < zu``, taken on the side that avoids cancellation."""
    if zl >= 0.0:
        a, b = float(log_ndtr(-zl)), float(log_ndtr(-zu))
    else:
        a, b = float(log_ndtr(zu)), float(log_ndtr(zl))
    if b >= a:
        return -_INF
    return a + math.log1p(-math.exp(b - a))


def fit_splice(losses: np.ndarray, collection_threshold: float, splice_threshold: float,
               min_tail: int = 10) -> SpliceFit:
    """Fit the lognormal body (truncated to ``[H, u]``, by maximum likelihood on the losses at or below ``u``) and the
    GPD tail (the excesses over ``u``), with the body weight ``w = (1 - t) / (1 - t c)`` where ``t = n_tail / N`` and
    ``c = F_b(H) / F_b(u)``, so that ``(1 - w) / (1 - w c) = t`` (``P(X > u) = 1 - w``, ``P(X >= H) = 1 - w c``).
    """
    x = vec(losses, "losses")
    h = scalar(collection_threshold, "collection_threshold")
    u = scalar(splice_threshold, "splice_threshold")
    if h < 0.0:
        raise ValueError("collection_threshold must not be negative")
    if not u > h:
        raise ValueError("splice_threshold must exceed the collection threshold")
    if np.any(x < h):
        raise ValueError("a loss lies below the collection threshold")
    if np.any(x <= 0.0):
        raise ValueError("the splice needs positive losses")
    body = x[x <= u]
    tail_excess = x[x > u] - u
    if body.size == 0:
        raise ValueError("no loss at or below the splice threshold")
    if tail_excess.size < min_tail:
        raise ValueError(f"only {tail_excess.size} losses above the splice threshold, fewer than min_tail = {min_tail}")
    if body.size < 2:
        raise ValueError("the body needs at least 2 losses")
    lb = np.log(body)
    if spread_is_zero(lb):
        raise ValueError("the body losses have no spread")
    lu = math.log(u)
    lh = math.log(h) if h > 0.0 else -_INF

    def body_nll(p: np.ndarray) -> float:
        mu, sigma = float(p[0]), float(p[1])
        if sigma <= 0.0:
            return _INF
        zu = (lu - mu) / sigma
        zl = (lh - mu) / sigma if h > 0.0 else -_INF
        log_mass = _log_mass(zl, zu) if h > 0.0 else float(log_ndtr(zu))
        return -(float(np.sum(_logpdf("lognormal", (mu, sigma), body))) - body.size * log_mass)

    mu0 = float(lb.mean())
    sigma0 = float(lb.std())
    res = _nelder_mead(lambda z: body_nll(np.array([z[0], math.exp(z[1])])), np.array([mu0, math.log(sigma0)]))
    mu, sigma = float(res.x[0]), math.exp(float(res.x[1]))
    se_body, ok_body = _standard_errors(body_nll, np.array([mu, sigma]), np.array([max(abs(mu), 1.0), sigma]))
    xi, beta, _, se_tail, ok_tail = _fit_gpd(tail_excess)
    t = tail_excess.size / x.size
    zu = (lu - mu) / sigma
    c = math.exp(float(log_ndtr((lh - mu) / sigma)) - float(log_ndtr(zu))) if h > 0.0 else 0.0
    w = (1.0 - t) / (1.0 - t * c)
    params = (mu, sigma, u, xi, beta, w)
    return SpliceFit(params, (se_body[0], se_body[1], _NAN, se_tail[0], se_tail[1], _NAN), h, u, w, int(body.size),
                     int(tail_excess.size), bool(ok_body and ok_tail and res.success))


# ----------------------------------------------------------------------------------------------------------------
# tail index profile


@dataclass(frozen=True)
class TailProfile:
    """The fitted GPD tail index of the excesses over each threshold, arrays of shape ``(T,)``: a threshold with fewer
    exceedances than the minimum is listed with ``xi``, ``xi_se`` and ``beta`` equal to ``nan`` and ``converged``
    False."""

    thresholds: np.ndarray
    n_exceedances: np.ndarray
    xi: np.ndarray
    xi_se: np.ndarray
    beta: np.ndarray
    converged: np.ndarray


def tail_index_profile(losses: np.ndarray, thresholds: np.ndarray, min_exceedances: int = 10) -> TailProfile:
    """The threshold-stability profile of dossier 05 F.3: a GPD fitted by maximum likelihood to the excesses ``x - u``
    of the losses above each strictly increasing threshold ``u``. For exact GPD data ``beta_hat(u) - xi_hat(u) u`` is
    constant in expectation."""
    x = vec(losses, "losses")
    th = vec(thresholds, "thresholds")
    if th.size > 1 and not np.all(np.diff(th) > 0.0):
        raise ValueError("thresholds must be strictly increasing")
    if min_exceedances < 2:
        raise ValueError("min_exceedances must be at least 2")
    k = th.size
    n_exc = np.zeros(k, dtype=np.int64)
    xi = np.full(k, _NAN)
    se = np.full(k, _NAN)
    beta = np.full(k, _NAN)
    conv = np.zeros(k, dtype=bool)
    for i, u in enumerate(th):
        y = x[x > u] - u
        n_exc[i] = y.size
        if y.size < min_exceedances:
            continue
        f_xi, f_beta, _, (f_se, _), ok = _fit_gpd(y)
        xi[i], beta[i], se[i], conv[i] = f_xi, f_beta, f_se, ok
    return TailProfile(th.copy(), n_exc, xi, se, beta, conv)

__all__ = [
    "FrequencyFit",
    "SeverityFit",
    "SpliceFit",
    "TailProfile",
    "fit_frequency",
    "fit_severity",
    "fit_splice",
    "frequency_pmf",
    "rescale_frequency",
    "severity_cdf",
    "severity_isf",
    "severity_moment",
    "severity_ppf",
    "tail_index_profile",
    "truncated_severity_cdf",
]
