"""The fit of the Vasicek distribution to a series of yearly loss or default rates (dossier 04 C.4; contract
`engine-portfolio-api`).

In the large homogeneous portfolio the yearly loss rate is p(Z) with Z ~ N(0, 1) and p the conditional PD of
`engines.vasicek`, with the density `engines.vasicek.lhp_pdf` and the quantile function `lhp_quantile`. Two
estimators:

- maximum likelihood on the density: (pd, rho) maximising sum_t ln lhp_pdf(rate_t; pd, rho), by Nelder-Mead on the
  logit scale from
  a fixed start with a fixed tolerance (deterministic);
- the method of moments: pd = mean(rates) and rho the root of E[L^2] = P(two obligors default) = Phi_2(G(pd),
  G(pd); rho) =
  mean(rates^2) (the second moment of the large-portfolio default fraction is the joint default probability of two
  names).

The uncertainty of the correlation: the profile log-likelihood and the likelihood-ratio interval (the parameter values
whose profile log-likelihood is at least max - chi2_1(level) / 2), and a seeded parametric bootstrap. The implied
factor series is the Z-factor of dossier 04 C.4, z_t = (G(pd) - sqrt(1 - rho) G(rate_t)) / sqrt(rho) (a derivation,
not a quoted formula).

A rate that is NaN is a year the table lacks: it is excluded, counted in `n_missing` and never interpolated. A PD, a
correlation and a loss rate are open at every estimate site: a value outside (0, 1) raises ValueError.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import integrate, stats
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import expit, logit

from riskvalidation.engines.vasicek import conditional_pd, implied_factor, lhp_pdf
from riskvalidation.portfolio.portfolio import check_level, float_vector, positive_int

BOUND_LOW = 1e-4            # an interval endpoint that reaches the parameter space is returned as these bounds
BOUND_HIGH = 0.9999
MIN_YEARS = 3
PD_SEARCH = (1e-6, 0.99)    # the search range of the PD at a fixed correlation
RHO_SEARCH = (1e-4, 0.9999)
GRID_POINTS = 41
MIN_BOOT = 100
PENALTY = 1e300


@dataclass(frozen=True)
class AsrfFit:
    """A fit: 0 < pd < 1, 0 < rho < 1, the log-likelihood in nats, how the estimate was reached and the years used."""

    pd: float
    rho: float
    log_likelihood: float
    converged: bool
    method: str
    n_years: int
    n_missing: int


class BootstrapIntervals(dict):
    """The percentile intervals {"pd": (low, high), "rho": (low, high)}, with how many refits were dropped."""

    n_boot: int
    n_dropped: int


def _clean(rates: Any) -> tuple[np.ndarray, int]:
    """The usable rates (NaN removed) and the number of NaN; a value outside (0, 1) or infinite raises ValueError."""
    a = np.array(rates, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"rates must be one-dimensional, got shape {a.shape}")
    miss = np.isnan(a)
    good = a[~miss]
    if np.any(np.isinf(good)) or np.any(good <= 0.0) or np.any(good >= 1.0):
        raise ValueError("every rate must lie strictly between 0 and 1 (NaN marks a missing year)")
    return good, int(miss.sum())


def log_likelihood(rates: np.ndarray, pd: float, rho: float) -> float:
    """sum_t ln lhp_pdf(rate_t; pd, rho); permutation-invariant. NaN rates are the callers' to exclude and count."""
    a = float_vector(rates, "rates")
    if np.any(a <= 0.0) or np.any(a >= 1.0):
        raise ValueError("every rate must lie strictly between 0 and 1")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return float(np.sum(np.log(lhp_pdf(a, pd, rho))))


def _usable(rates: Any) -> tuple[np.ndarray, int]:
    r, miss = _clean(rates)
    if r.size < MIN_YEARS:
        raise ValueError(f"at least {MIN_YEARS} usable years are needed, got {r.size}")
    return r, miss


def _no_spread(r: np.ndarray) -> bool:
    return float(np.ptp(r)) <= 1e-12 * max(1.0, float(np.max(np.abs(r))))


def fit_ml(rates: np.ndarray) -> AsrfFit:
    """The maximum-likelihood (pd, rho). Raises ValueError for fewer than 3 usable years or a series with zero spread.

    `converged` is False when the optimiser stops at its iteration cap or at the bound of the parameter space (rho or
    pd outside [1e-4, 0.9999]).
    """
    r, miss = _usable(rates)
    if _no_spread(r):
        raise ValueError("the rates have zero spread: the correlation is not identified")

    def nll(t: np.ndarray) -> float:
        v = log_likelihood(r, float(expit(t[0])), float(expit(t[1])))
        return -v if math.isfinite(v) else PENALTY

    x0 = np.array([logit(min(max(float(r.mean()), BOUND_LOW), BOUND_HIGH)), logit(0.1)])
    opts = {"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000, "maxfev": 8000}
    res = minimize(nll, x0, method="Nelder-Mead", options=opts)
    res = minimize(nll, res.x, method="Nelder-Mead", options=opts)  # one restart from the optimum: simplex stalls
    pd, rho = float(expit(res.x[0])), float(expit(res.x[1]))
    inside = BOUND_LOW < rho < BOUND_HIGH and BOUND_LOW < pd < BOUND_HIGH
    return AsrfFit(pd=pd, rho=rho, log_likelihood=-float(res.fun), converged=bool(res.success and inside),
                   method="maximum-likelihood", n_years=int(r.size), n_missing=miss)


def _joint_default(pd: float, rho: float) -> float:
    """E[p(Z)^2] = P(two obligors default) by quadrature of the engine's conditional PD."""
    def integrand(z: float) -> float:
        return float(conditional_pd(pd, rho, z)) ** 2 * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    val, _ = integrate.quad(integrand, -12.0, 12.0, epsabs=1e-16, epsrel=1e-12, limit=200)
    return float(val)


def fit_moments(rates: np.ndarray) -> AsrfFit:
    """pd = mean(rates) and rho solving E[p(Z)^2] = mean(rates^2).

    When mean(rates^2) <= pd^2 there is no excess dispersion, the correlation is not identified and ValueError is
    raised (never 0). `log_likelihood` is evaluated at the moment estimates for comparison.
    """
    r, miss = _usable(rates)
    pd = float(r.mean())
    m2 = float(np.mean(r * r))
    if m2 <= pd * pd * (1.0 + 1e-12):
        raise ValueError("no excess dispersion: mean(rates^2) <= pd^2, the correlation is not identified")
    lo, hi = 1e-9, 1.0 - 1e-9

    def gap(rho: float) -> float:
        return _joint_default(pd, rho) - m2

    if gap(hi) < 0.0:
        raise ValueError("the second moment is above the joint default probability at rho = 1 - 1e-9: "
                         "no correlation fits")
    rho = float(brentq(gap, lo, hi, xtol=1e-14, rtol=1e-13))
    return AsrfFit(pd=pd, rho=rho, log_likelihood=log_likelihood(r, pd, rho), converged=True, method="moments",
                   n_years=int(r.size), n_missing=miss)


def _maximize_logit(fun, lo: float, hi: float) -> tuple[float, float]:
    """The maximiser x in (lo, hi) of fun by a coarse grid on the logit scale and a bounded refinement; (x, fun(x))."""
    grid = np.linspace(logit(lo), logit(hi), GRID_POINTS)
    vals = np.array([fun(float(expit(t))) for t in grid])
    vals = np.where(np.isfinite(vals), vals, -PENALTY)
    i = int(np.argmax(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, GRID_POINTS - 1)]
    res = minimize_scalar(lambda t: -fun(float(expit(t))), bounds=(a, b), method="bounded", options={"xatol": 1e-10})
    best_x, best_v = float(expit(res.x)), -float(res.fun)
    if vals[i] > best_v:
        best_x, best_v = float(expit(grid[i])), float(vals[i])
    return best_x, best_v


def _profile_at_rho(r: np.ndarray, rho: float) -> tuple[float, float]:
    pd, v = _maximize_logit(lambda p: log_likelihood(r, p, rho), *PD_SEARCH)
    return v, pd


def _profile_at_pd(r: np.ndarray, pd: float) -> tuple[float, float]:
    rho, v = _maximize_logit(lambda p: log_likelihood(r, pd, p), *RHO_SEARCH)
    return v, rho


def profile_likelihood(rates: np.ndarray, rho_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(log_likelihood, pd_at_rho): at each correlation of a strictly increasing grid in (0, 1), the PD that
    maximises the likelihood (a one-dimensional bounded search) and the maximised log-likelihood. NaN rates are
    excluded."""
    r, _ = _usable(rates)
    g = float_vector(rho_grid, "rho_grid")
    if np.any(g <= 0.0) or np.any(g >= 1.0) or np.any(np.diff(g) <= 0.0):
        raise ValueError("rho_grid must be strictly increasing inside (0, 1)")
    out = [_profile_at_rho(r, float(rho)) for rho in g]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def _endpoint(profile, start: float, bound: float, cutoff: float) -> float:
    """Where the profile (a function of one parameter) falls to the cutoff going from `start` to `bound`."""
    pts = expit(np.linspace(logit(start), logit(bound), GRID_POINTS))
    prev = float(pts[0])
    for x in pts[1:]:
        if profile(float(x)) < cutoff:
            return float(brentq(lambda t: profile(t) - cutoff, prev, float(x), xtol=1e-9, rtol=1e-10))
        prev = float(x)
    return float(bound)


def profile_interval(rates: np.ndarray, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """The likelihood-ratio interval {"pd": (low, high), "rho": (low, high)}: the values whose profile log-likelihood is
    at least max - chi2_1(level) / 2 (for 0.95 the cutoff is max - 1.920729). An endpoint that reaches the
    parameter space is returned as the bound (1e-4 or 0.9999); no error is raised. Deterministic."""
    level = check_level(level, "level")
    r, _ = _usable(rates)
    est = fit_ml(r)
    cut_drop = float(stats.chi2.ppf(level, 1)) / 2.0
    top = max(est.log_likelihood, _profile_at_rho(r, est.rho)[0], _profile_at_pd(r, est.pd)[0])
    cutoff = top - cut_drop

    def p_rho(x: float) -> float:
        return _profile_at_rho(r, x)[0]

    def p_pd(x: float) -> float:
        return _profile_at_pd(r, x)[0]

    rho_hat = min(max(est.rho, BOUND_LOW), BOUND_HIGH)
    pd_hat = min(max(est.pd, BOUND_LOW), BOUND_HIGH)
    return {
        "pd": (_endpoint(p_pd, pd_hat, BOUND_LOW, cutoff), _endpoint(p_pd, pd_hat, BOUND_HIGH, cutoff)),
        "rho": (_endpoint(p_rho, rho_hat, BOUND_LOW, cutoff), _endpoint(p_rho, rho_hat, BOUND_HIGH, cutoff)),
    }


def bootstrap_interval(rates: np.ndarray, estimator: str, n_boot: int, seed: int,
                       level: float = 0.95) -> dict[str, tuple[float, float]]:
    """A parametric bootstrap: fit the estimator, draw `n_boot` series of the same length with z_t ~ N(0, 1) and
    rate_t = conditional_pd(pd, rho, z_t), refit each, and return the (1 - level)/2 and (1 + level)/2 percentiles of
    the refits.

    A refit that fails (a ValueError: moments with no excess dispersion, a series with no spread) is dropped and
    counted in the `n_dropped` attribute of the result; a maximum-likelihood refit that stops at a bound is kept.
    Reproducible for a seed. The interval is not required to contain the estimate (a skewed refit distribution).
    """
    fitters = {"maximum-likelihood": fit_ml, "moments": fit_moments}
    if estimator not in fitters:
        raise ValueError(f"estimator must be one of {sorted(fitters)}, got {estimator!r}")
    nb = positive_int(n_boot, "n_boot")
    if nb < MIN_BOOT:
        raise ValueError(f"n_boot must be at least {MIN_BOOT}")
    level = check_level(level, "level")
    fit = fitters[estimator](rates)
    t = fit.n_years
    z = np.random.Generator(np.random.PCG64(int(seed))).standard_normal((nb, t))
    pds, rhos = [], []
    for row in z:
        sim = np.asarray(conditional_pd(fit.pd, fit.rho, row))
        try:
            refit = fitters[estimator](sim)
        except ValueError:
            continue
        pds.append(refit.pd)
        rhos.append(refit.rho)
    if len(pds) < 2:
        raise ValueError("fewer than 2 bootstrap refits succeeded: no interval")
    lo, hi = 100.0 * (1.0 - level) / 2.0, 100.0 * (1.0 + level) / 2.0
    out = BootstrapIntervals(
        pd=(float(np.percentile(pds, lo)), float(np.percentile(pds, hi))),
        rho=(float(np.percentile(rhos, lo)), float(np.percentile(rhos, hi))),
    )
    out.n_boot = nb
    out.n_dropped = nb - len(pds)
    return out


def implied_factors(rates: np.ndarray, pd: float, rho: float) -> np.ndarray:
    """z_t = implied_factor(rate_t; pd, rho) elementwise (decreasing in the rate), NaN where the rate is NaN."""
    a = np.array(rates, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"rates must be one-dimensional, got shape {a.shape}")
    implied_factor(0.5, pd, rho)  # validates pd and rho
    miss = np.isnan(a)
    out = np.full(a.shape, np.nan)
    if np.any((a[~miss] <= 0.0) | (a[~miss] >= 1.0)):
        raise ValueError("every rate must lie strictly between 0 and 1")
    if (~miss).any():
        out[~miss] = implied_factor(a[~miss], pd, rho)
    return out

__all__ = [
    "AsrfFit",
    "BootstrapIntervals",
    "bootstrap_interval",
    "fit_ml",
    "fit_moments",
    "implied_factors",
    "log_likelihood",
    "profile_interval",
    "profile_likelihood",
]
