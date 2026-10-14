"""Zero curves and curve fits: interpolation, discount factors, Nelson-Siegel, Svensson and dynamic Nelson-Siegel
(dossier 05 D.5; dossier 08 C.2).

Rates are decimals and discounting is continuously compounded, ``DF(t) = exp(-R t)`` (SRP31.129). The functions are
pure NumPy and SciPy: QuantLib is at most an optional test extra that cross-checks the discount factors. Contract:
``engine-irrbb-api`` 1.0.0, module curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from riskvalidation.lgd_ead._checks import mat, same_length, scalar, vec

DIEBOLD_LI_LAMBDA = 0.0609
"""The Diebold and Li (2006) decay per month. The curvature loading peaks at x* / lam = 29.446 months, not at the
"exactly 30" the paper rounds to (dossier 05 D.5; recomputed by Contraste)."""

DEFAULT_TAU_GRID = np.arange(1, 41) / 4.0
"""The default tau grid of ``fit_nelson_siegel``: 0.25 to 10.0 years in steps of 0.25 (it contains 2.0 exactly)."""

_SVENSSON_GRID = np.arange(2, 81) / 4.0  # 0.5 to 20.0 years in steps of 0.25 (contains 2.0 and 6.0)


def _increasing(x: np.ndarray, name: str) -> None:
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def bootstrap_zero_curve(tenors: np.ndarray, par_yields: np.ndarray, frequency: int = 2) -> np.ndarray:
    """Continuously compounded zero rates at the tenors from bond-equivalent par yields with ``frequency`` coupons a
    year.

    A tenor at or below ``1 / frequency`` is read as a simple-interest money-market rate, ``R = ln(1 + y t) / t``. The
    par yield is interpolated linearly between pillars (flat outside) on the coupon dates ``k / frequency`` and the
    zero curve on those dates is bootstrapped from the par condition, ``DF_k = (1 - c_k sum_{j<k} DF_j) / (1 + c_k)``
    with ``c_k = y(t_k) / frequency``; a pillar above ``1 / frequency`` that is not a coupon date is read from the
    bootstrapped dates by linear interpolation of the zero rate (its own par bond then reprices at par only on the
    coupon dates).

    A flat par curve gives a flat zero curve at every tenor that is a multiple of ``1 / frequency``; below ``1 /
    frequency`` the money-market rule gives a slightly different rate, by design. A missing tenor is absent from
    ``tenors``, never filled."""
    t, y = vec(tenors, "tenors"), vec(par_yields, "par_yields")
    same_length(tenors=t, par_yields=y)
    _increasing(t, "tenors")
    if t[0] <= 0.0:
        raise ValueError("tenors must be positive")
    if int(frequency) != frequency or frequency < 1:
        raise ValueError("frequency must be a positive integer")
    f = int(frequency)
    if np.any(1.0 + y * t <= 0.0):
        raise ValueError("a par yield makes 1 + y t nonpositive")
    n = int(np.ceil(t[-1] * f - 1e-9))
    grid = np.arange(1, max(n, 1) + 1) / f
    coupon = np.interp(grid, t, y) / f
    df = np.empty(grid.size)
    running = 0.0
    for k in range(grid.size):
        df[k] = (1.0 - coupon[k] * running) / (1.0 + coupon[k])
        if df[k] <= 0.0:
            raise ValueError("the par curve implies a nonpositive discount factor")
        running += df[k]
    grid_zero = -np.log(df) / grid
    out = np.interp(t, grid, grid_zero)
    money = t <= 1.0 / f + 1e-12
    out[money] = np.log1p(y[money] * t[money]) / t[money]
    on_grid = np.isclose(t * f, np.round(t * f), rtol=0.0, atol=1e-9) & ~money
    out[on_grid] = grid_zero[np.round(t[on_grid] * f).astype(int) - 1]
    return out


def zero_rates_at(tenors: np.ndarray, zero_rates: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Linear interpolation in tenor with flat extrapolation outside the grid; exact at the pillars."""
    t, r = vec(tenors, "tenors"), vec(zero_rates, "zero_rates")
    same_length(tenors=t, zero_rates=r)
    _increasing(t, "tenors")
    a = vec(at, "at", allow_empty=True)
    return np.interp(a, t, r)


def discount_factors(rates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """``DF = exp(-R t)`` (SRP31.129); times are nonnegative years."""
    r, t = vec(rates, "rates", allow_empty=True), vec(times, "times", allow_empty=True)
    same_length(rates=r, times=t)
    if np.any(t < 0.0):
        raise ValueError("times must be nonnegative")
    return np.exp(-r * t)


def _humps(m: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """The slope loading ``(1 - e^-x) / x`` and the curvature loading ``(1 - e^-x) / x - e^-x`` at ``x = m / tau``,
    each at its limit (1 and 0) for ``m = 0`` instead of a division by zero."""
    x = m / tau
    safe = np.where(x > 0.0, x, 1.0)
    slope = np.where(x > 0.0, -np.expm1(-safe) / safe, 1.0)
    return slope, slope - np.exp(-x)


def nelson_siegel_yield(m: np.ndarray, beta0: float, beta1: float, beta2: float, tau: float) -> np.ndarray:
    """``y(m) = b0 + b1 (1 - e^-x) / x + b2 ((1 - e^-x) / x - e^-x)``, ``x = m / tau`` (Nelson and Siegel 1987)."""
    mm = vec(m, "m")
    if np.any(mm < 0.0):
        raise ValueError("m must be nonnegative")
    if not scalar(tau, "tau") > 0.0:
        raise ValueError("tau must be positive")
    slope, curv = _humps(mm, float(tau))
    return scalar(beta0, "beta0") + scalar(beta1, "beta1") * slope + scalar(beta2, "beta2") * curv


def svensson_yield(m: np.ndarray, beta0: float, beta1: float, beta2: float, beta3: float, tau1: float,
                   tau2: float) -> np.ndarray:
    """Nelson-Siegel plus the second hump ``b3 ((1 - e^-x2) / x2 - e^-x2)``, ``x2 = m / tau2`` (Svensson 1994)."""
    mm = vec(m, "m")
    if not scalar(tau2, "tau2") > 0.0:
        raise ValueError("tau2 must be positive")
    _, curv2 = _humps(mm, float(tau2))
    return nelson_siegel_yield(mm, beta0, beta1, beta2, tau1) + scalar(beta3, "beta3") * curv2


@dataclass(frozen=True)
class NelsonSiegelFit:
    beta0: float
    beta1: float
    beta2: float
    tau: float
    rmse: float          # decimals


def fit_nelson_siegel(maturities: np.ndarray, yields: np.ndarray,
                      tau_grid: np.ndarray | None = None) -> NelsonSiegelFit:
    """The betas by least squares for each tau of the grid, the best tau kept (default grid ``DEFAULT_TAU_GRID``)."""
    m, y = vec(maturities, "maturities"), vec(yields, "yields")
    same_length(maturities=m, yields=y)
    if m.size < 4:
        raise ValueError("fit_nelson_siegel needs at least 4 maturities")
    if np.any(m <= 0.0):
        raise ValueError("maturities must be positive")
    grid = DEFAULT_TAU_GRID if tau_grid is None else vec(tau_grid, "tau_grid")
    if np.any(grid <= 0.0):
        raise ValueError("tau_grid must be positive")
    best: tuple[float, float, np.ndarray] | None = None
    for tau in grid:
        slope, curv = _humps(m, float(tau))
        design = np.column_stack([np.ones_like(m), slope, curv])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        sse = float(np.sum((y - design @ beta) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(tau), beta)
    assert best is not None
    sse, tau, beta = best
    return NelsonSiegelFit(float(beta[0]), float(beta[1]), float(beta[2]), tau, float(np.sqrt(sse / m.size)))


@dataclass(frozen=True)
class SvenssonFit:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float
    rmse: float


def _svensson_design(m: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    s1, c1 = _humps(m, tau1)
    _, c2 = _humps(m, tau2)
    return np.column_stack([np.ones_like(m), s1, c1, c2])


def _svensson_sse(m: np.ndarray, y: np.ndarray, tau1: float, tau2: float) -> tuple[float, np.ndarray]:
    design = _svensson_design(m, tau1, tau2)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.sum((y - design @ beta) ** 2)), beta


def fit_svensson(maturities: np.ndarray, yields: np.ndarray) -> SvenssonFit:
    """Variable projection: the four betas by least squares for each ``(tau1, tau2)``, a fixed grid search followed by
    a bounded polish of the two decays. Deterministic (fixed grid, start and tolerance); the family nests
    Nelson-Siegel (the Nelson-Siegel tau is on the grid and ``beta3 = 0`` is feasible), so the RMSE is never above the
    Nelson-Siegel RMSE of the same data. A polish that does not converge raises ``ValueError``."""
    m, y = vec(maturities, "maturities"), vec(yields, "yields")
    same_length(maturities=m, yields=y)
    if m.size < 6:
        raise ValueError("fit_svensson needs at least 6 maturities")
    if np.any(m <= 0.0):
        raise ValueError("maturities must be positive")
    best = (np.inf, 0.0, 0.0)
    for i, t1 in enumerate(_SVENSSON_GRID):
        for t2 in _SVENSSON_GRID[i + 1:]:
            sse, _ = _svensson_sse(m, y, float(t1), float(t2))
            if sse < best[0]:
                best = (sse, float(t1), float(t2))
    sse0, tau1, tau2 = best

    def resid(p: np.ndarray) -> np.ndarray:
        design = _svensson_design(m, p[0], p[1])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        return y - design @ beta

    sol = least_squares(resid, [tau1, tau2], bounds=([0.05, 0.05], [60.0, 60.0]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    if not sol.success:
        raise ValueError(f"fit_svensson did not converge: {sol.message}")
    sse, beta = _svensson_sse(m, y, float(sol.x[0]), float(sol.x[1]))
    if sse > sse0:  # the polish never ends above the grid optimum
        sse, beta = _svensson_sse(m, y, tau1, tau2)
        sol.x = np.array([tau1, tau2])
    if not np.isfinite(sse):
        raise ValueError("fit_svensson did not converge: non-finite residual")
    return SvenssonFit(float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3]), float(sol.x[0]), float(sol.x[1]),
                       float(np.sqrt(sse / m.size)))


def dns_loadings(maturities_months: np.ndarray, lam: float = DIEBOLD_LI_LAMBDA) -> np.ndarray:
    """(N, 3): the level loading 1, the slope loading ``(1 - e^-lam m) / (lam m)`` and the curvature loading
    ``(1 - e^-lam m) / (lam m) - e^-lam m`` (Diebold and Li 2006); ``m`` in months, strictly positive."""
    m = vec(maturities_months, "maturities_months")
    if np.any(m <= 0.0):
        raise ValueError("maturities_months must be strictly positive")
    if not scalar(lam, "lam") > 0.0:
        raise ValueError("lam must be positive")
    x = lam * m
    slope = -np.expm1(-x) / x
    return np.column_stack([np.ones_like(m), slope, slope - np.exp(-x)])


def dns_yield(maturities_months: np.ndarray, level: float, slope: float, curvature: float,
              lam: float = DIEBOLD_LI_LAMBDA) -> np.ndarray:
    """``y = level + slope L_slope(m) + curvature L_curv(m)``."""
    return dns_loadings(maturities_months, lam) @ np.array(
        [scalar(level, "level"), scalar(slope, "slope"), scalar(curvature, "curvature")])


def fit_dns_factors(maturities_months: np.ndarray, yields: np.ndarray, lam: float = DIEBOLD_LI_LAMBDA) -> np.ndarray:
    """(T, 3) level, slope and curvature by least squares at each date; ``yields`` is (T, N) with ``N >= 3``."""
    m = vec(maturities_months, "maturities_months")
    y = mat(yields, "yields")
    if m.size < 3:
        raise ValueError("fit_dns_factors needs at least 3 maturities")
    if y.shape[1] != m.size:
        raise ValueError(f"yields has {y.shape[1]} columns for {m.size} maturities")
    return np.linalg.lstsq(dns_loadings(m, lam), y.T, rcond=None)[0].T


def forecast_dns_factors(factors: np.ndarray, horizon: int, n_paths: int, seed: int) -> np.ndarray:
    """(P, horizon, 3) simulated factor paths from an AR(1) fitted by least squares to each factor,
    ``f_t = c + phi f_{t-1} + e_t`` with independent Gaussian innovations of the fitted residual variance
    (``SSE / (T - 3)``; Diebold and Li 2006). A path starts from the last row of the history and uses no later data. A
    history with no spread is forecast as constant."""
    f = mat(factors, "factors")
    if f.shape[1] != 3:
        raise ValueError("factors must have 3 columns (level, slope, curvature)")
    if f.shape[0] < 4:
        raise ValueError("forecast_dns_factors needs at least 4 dates")
    if int(horizon) < 1 or int(n_paths) < 1:
        raise ValueError("horizon and n_paths must be at least 1")
    h, p = int(horizon), int(n_paths)
    const = np.empty(3)
    phi = np.empty(3)
    sd = np.empty(3)
    for j in range(3):
        x, y = f[:-1, j], f[1:, j]
        if np.ptp(x) <= 1e-15 * max(1.0, float(np.max(np.abs(x)))):
            const[j], phi[j], sd[j] = float(y.mean()), 0.0, 0.0
            continue
        design = np.column_stack([np.ones_like(x), x])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        res = y - design @ coef
        const[j], phi[j] = float(coef[0]), float(coef[1])
        sd[j] = float(np.sqrt(res @ res / (x.size - 2)))
    z = np.random.default_rng(seed).standard_normal((p, h, 3))
    out = np.empty((p, h, 3))
    prev = np.tile(f[-1], (p, 1))
    for step in range(h):
        prev = const + phi * prev + sd * z[:, step, :]
        out[:, step, :] = prev
    return out
