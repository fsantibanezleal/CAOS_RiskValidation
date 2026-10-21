"""The Hull-White one-factor model on any time grid: ``r_t = phi(t) + x_t`` with ``x_t`` an
Ornstein-Uhlenbeck process ``dx = -a x dt + sigma dW`` (addendum R4, QuantLib 1.44
``hullwhite.hpp``), its affine bond prices, the zero-coupon bond option and the Jamshidian swaption, and the historical
AR(1) estimate of ``a`` and ``sigma`` (addendum R6, Brigo et al. arXiv 0812.4210).

The curve and forward functions of ``irrbb.curves`` and ``irrbb.simulation`` are reused by import, never copied, and
``irrbb.simulation.hw1f_scenarios`` is unchanged (contract ``engine-irrbb-api``). Contract: ``engine-ccr-api`` 1.0.0,
module hw1f. A curve is passed as two callables on arrays of years: ``p0(t)`` today's discount factor ``P(0, t)``
and ``forward(t)`` the instantaneous forward ``f(0, t)``; ``curve_functions`` builds them from a zero curve. Every
function with randomness takes a seed and uses ``numpy.random.default_rng(seed)``.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from riskvalidation.irrbb.curves import zero_rates_at
from riskvalidation.irrbb.simulation import _forward_curve
from riskvalidation.lgd_ead._checks import mat, same_length, scalar, vec

SQRT_EPS = math.sqrt(float(np.finfo(float).eps))
"""Below this mean reversion the closed forms use their limit (QuantLib ``hullwhite.cpp``: ``a < sqrt(eps)``)."""

JAMSHIDIAN_BRACKET = (-10.0, 10.0)
"""The bracket of the Brent search for r* (QuantLib ``jamshidianswaptionengine.cpp``, addendum R4)."""

JAMSHIDIAN_XTOL = 1e-12
"""The tolerance of the Brent search. QuantLib uses 1e-8; the engine solves tighter so that the parity identity of the
swaption holds to 1e-9 for any bracket."""

DAYS_PER_YEAR = 365.25
"""Calendar days to years, used only for the default step and the history span of ``calibrate_ar1``."""

Curve = Callable[[np.ndarray], np.ndarray]


def _eval(fn: Callable, x: np.ndarray | float) -> np.ndarray:
    """A curve callable on an array (or a scalar) of times, returned as a float array of the same shape."""
    arg = np.asarray(x, dtype=float)
    return np.asarray(fn(arg), dtype=float).reshape(arg.shape)


def _mean_reversion(a: float) -> float:
    v = scalar(a, "a")
    if v < 0.0:
        raise ValueError("a (mean reversion) must be nonnegative")
    return v


def _volatility(sigma: float) -> float:
    v = scalar(sigma, "sigma")
    if v < 0.0:
        raise ValueError("sigma must be nonnegative")
    return v


def _times(x: np.ndarray | float, name: str) -> np.ndarray:
    a = np.array(x, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(a < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return a


def curve_functions(zero_rates: np.ndarray, tenors: np.ndarray) -> tuple[Curve, Curve]:
    """``(p0, forward)`` of a continuously compounded zero curve, linear in tenor with flat extrapolation:
    ``P(0, t) = exp(-R(t) t)`` through ``irrbb.curves.zero_rates_at`` and
    ``f(0, t) = R(t) + t R'(t)`` through ``irrbb.simulation._forward_curve``. Both accept
    an array or a scalar of years."""
    z, ten = vec(zero_rates, "zero_rates"), vec(tenors, "tenors")
    same_length(zero_rates=z, tenors=ten)
    if np.any(np.diff(ten) <= 0.0) or ten[0] <= 0.0:
        raise ValueError("tenors must be strictly increasing and positive")

    def p0(t: np.ndarray) -> np.ndarray:
        flat = np.asarray(t, dtype=float).reshape(-1)
        return np.exp(-zero_rates_at(ten, z, flat) * flat).reshape(np.shape(t))

    def forward(t: np.ndarray) -> np.ndarray:
        flat = np.asarray(t, dtype=float).reshape(-1)
        return _forward_curve(z, ten, flat).reshape(np.shape(t))

    return p0, forward


def phi(t: np.ndarray, a: float, sigma: float, forward: Curve) -> np.ndarray:
    """The fitting parameter
    ``phi(t) = f(0, t) + (sigma (1 - exp(-a t)) / a)^2 / 2``
    (QuantLib ``FittingParameter``); for ``a`` below ``sqrt(eps)`` the bracket is ``sigma t``.
    ``phi(0) = f(0, 0)`` and ``sigma = 0`` gives ``f(0, t)``."""
    tt = _times(t, "t")
    a_, s_ = _mean_reversion(a), _volatility(sigma)
    bracket = s_ * bond_b(np.zeros_like(tt), tt, a_)
    return _eval(forward, tt) + 0.5 * bracket**2


def bond_b(t: np.ndarray, big_t: np.ndarray, a: float) -> np.ndarray:
    """``B(t, T) = (1 - exp(-a (T - t))) / a``, ``T - t`` for ``a`` below ``sqrt(eps)``; zero at ``T = t``."""
    t_, big = _times(t, "t"), _times(big_t, "big_t")
    a_ = _mean_reversion(a)
    tau = big - t_
    if np.any(tau < -1e-12):
        raise ValueError("big_t must not be before t")
    tau = np.maximum(tau, 0.0)
    if a_ < SQRT_EPS:
        return tau
    return -np.expm1(-a_ * tau) / a_


def bond_a(t: np.ndarray, big_t: np.ndarray, a: float, sigma: float, p0: Curve, forward: Curve) -> np.ndarray:
    """``A(t, T) = P(0, T) / P(0, t) exp(B f(0, t) - sigma^2 / (4 a) (1 - exp(-2 a t)) B^2)``
    (``HullWhite::A``, written as ``0.25 (sigma B)^2 B(0, 2 t)`` so that the limit ``a`` to 0
    needs no division). ``A(t, t) = 1``; at ``t = 0`` it is
    ``P(0, T) exp(B(0, T) f(0, 0))``, not ``P(0, T)``."""
    t_, big = _times(t, "t"), _times(big_t, "big_t")
    a_, s_ = _mean_reversion(a), _volatility(sigma)
    b = bond_b(t_, big, a_)
    b2t = bond_b(np.zeros_like(t_), 2.0 * t_, a_)
    ratio = _eval(p0, big) / _eval(p0, t_)
    return ratio * np.exp(b * _eval(forward, t_) - 0.25 * (s_ * b) ** 2 * b2t)


def bond_price(r: np.ndarray, t: float, big_t: float, a: float, sigma: float, p0: Curve, forward: Curve) -> np.ndarray:
    """``P(t, T) = A(t, T) exp(-B(t, T) r)`` for the short rates ``r`` at ``t``. At
    ``t = 0`` with ``r = phi(0)`` it reprices ``P(0, T)`` to 1e-12; decreasing in ``r``."""
    rr = np.array(r, dtype=float)
    if not np.all(np.isfinite(rr)):
        raise ValueError("r contains non-finite values")
    tt, big = scalar(t, "t"), scalar(big_t, "big_t")
    if tt < 0.0 or big < tt - 1e-12:
        raise ValueError(f"t must satisfy 0 <= t <= big_t, got t={tt} and big_t={big}")
    big = max(big, tt)
    ta, ba = np.array(tt), np.array(big)
    return bond_a(ta, ba, a, sigma, p0, forward) * np.exp(-bond_b(ta, ba, a) * rr)


def simulate_state(grid: np.ndarray, a: float, sigma: float, n_paths: int, seed: int,
                   antithetic: bool = True) -> np.ndarray:
    """(P, K) paths of the Ornstein-Uhlenbeck state with ``x(0) = 0``, simulated exactly:
    ``x(t_i) = x(t_{i-1}) e^{-a dt} + sigma sqrt((1 - e^{-2 a dt}) / (2 a)) z``
    (addendum R6, equations 49 to 52), so the law does not depend on the step. Antithetic: the second half mirrors the
    first (path ``i + P / 2`` is minus path ``i``), so the sample mean is zero and ``n_paths`` must
    be even."""
    g = vec(grid, "grid")
    if g[0] != 0.0 or np.any(np.diff(g) <= 0.0):
        raise ValueError("grid must start at 0 and be strictly increasing")
    a_, s_ = _mean_reversion(a), _volatility(sigma)
    n = int(n_paths)
    if n != n_paths or n < 1:
        raise ValueError("n_paths must be a positive integer")
    if antithetic and n % 2:
        raise ValueError(f"n_paths must be even with antithetic paths, got {n}")
    rng = np.random.default_rng(seed)
    half = n // 2 if antithetic else n
    z = rng.standard_normal((half, g.size - 1))
    if antithetic:
        z = np.vstack([z, -z])
    dt = np.diff(g)
    decay = np.exp(-a_ * dt)
    if a_ < SQRT_EPS:
        vol = s_ * np.sqrt(dt)
    else:
        vol = s_ * np.sqrt(-np.expm1(-2.0 * a_ * dt) / (2.0 * a_))
    x = np.zeros((n, g.size))
    for k in range(g.size - 1):
        x[:, k + 1] = x[:, k] * decay[k] + vol[k] * z[:, k]
    return x


def simulate_short_rate(grid: np.ndarray, a: float, sigma: float, forward: Curve, n_paths: int,
                        seed: int) -> np.ndarray:
    """(P, K) short rates ``r = phi(t) + x`` on the grid, antithetic (``n_paths`` even). The mean of
    ``r(t)`` is ``phi(t)`` and the discounted bond prices are martingales (``bank_account`` is the numeraire)."""
    x = simulate_state(grid, a, sigma, n_paths, seed)
    return x + phi(np.asarray(grid, dtype=float), a, sigma, forward)[None, :]


PHI_SERIES_THRESHOLD = 0.5
"""Below ``a t`` of this size the integral of ``phi`` uses its power series in ``a t`` (the limit ``t^3 / 6`` and its
corrections, truncated at about 1e-12 relative); above it the closed form, whose cancellation is about
``eps / (a t)^2`` relative to the term (4e-16)."""

_PHI_SERIES = tuple((-1.0) ** k * (2.0**k - 2.0) / math.factorial(k + 1) / 2.0 for k in range(2, 19))
"""Coefficients ``c_k`` of ``(1 / (2 a^2)) [t - 2 (1 - e^{-a t}) / a + (1 - e^{-2 a t}) / (2 a)]``, which is
``t^3 sum_k c_k (a t)^{k-2}``,
from the expansions of the two exponentials; ``c_2 = 1/6`` is the limit."""


def _phi_integral_term(t: np.ndarray, a: float) -> np.ndarray:
    """``(1 / (2 a^2)) [t - 2 (1 - e^{-a t}) / a + (1 - e^{-2 a t}) / (2 a)]``, the part of the integral of ``phi``
    from 0 to ``t`` that is multiplied by ``sigma^2``; ``t^3 / 6`` in the limit ``a`` to 0. Where ``a t`` is below
    ``PHI_SERIES_THRESHOLD`` the series is used, so that the closed form never loses its digits to cancellation."""
    tt = np.asarray(t, dtype=float)
    u = a * tt
    series = np.zeros_like(tt)
    for c in reversed(_PHI_SERIES):
        series = series * u + c
    series = tt**3 * series
    if a == 0.0:
        return series
    closed = (tt + (2.0 * np.expm1(-u) - 0.5 * np.expm1(-2.0 * u)) / a) / (2.0 * a**2)
    return np.where(u < PHI_SERIES_THRESHOLD, series, closed)


def bank_account(grid: np.ndarray, short_rate: np.ndarray, a: float, sigma: float, p0: Curve,
                 forward: Curve) -> np.ndarray:
    """(P, K) the numeraire ``N(t) = exp(integral of r)`` of the paths, ``N(0) = 1``, the numeraire of
    ``exposure.expected_exposure`` in the discounted measure.

    With ``r = phi + x`` the deterministic part is integrated exactly,
    ``integral_0^t phi = -ln P(0, t) + sigma^2 (1 / (2 a^2)) [t - 2 (1 - e^{-a t}) / a + (1 - e^{-2 a t}) / (2 a)]``
    (``sigma^2 t^3 / 6`` as ``a`` tends to 0), and only the state ``x = r - phi`` is integrated by the trapezoid rule
    on the grid. The trapezoid on ``r`` itself misses the integral of ``phi`` by an amount that depends on the grid
    (the forward of a zero curve jumps at its pillars), a bias in every discounted exposure. At ``sigma = 0`` the
    discount factor ``1 / N(t)`` is ``P(0, t)`` on any grid. ``a``, ``sigma``, ``p0`` and ``forward`` are those of
    ``simulate_short_rate``."""
    g = vec(grid, "grid")
    if g[0] != 0.0 or np.any(np.diff(g) <= 0.0):
        raise ValueError("grid must start at 0 and be strictly increasing")
    r = mat(short_rate, "short_rate")
    if r.shape[1] != g.size:
        raise ValueError(f"short_rate has {r.shape[1]} columns for a grid of {g.size} points")
    a_, s_ = _mean_reversion(a), _volatility(sigma)
    x = r - phi(g, a_, s_, forward)[None, :]
    area = np.cumsum(0.5 * (x[:, 1:] + x[:, :-1]) * np.diff(g)[None, :], axis=1)
    det = -np.log(_eval(p0, g)) + s_**2 * _phi_integral_term(g, a_)
    return np.exp(np.hstack([np.zeros((r.shape[0], 1)), area]) + det[None, :])


def bond_option_vol(expiry: float, bond_maturity: float, a: float, sigma: float) -> float:
    """``sigma_p = sigma B(T, S) sqrt((1 - e^{-2 a T}) / (2 a))``
    of ``HullWhite::discountBondOption`` (``sigma B sqrt(T)`` for ``a`` below ``sqrt(eps)``)."""
    big_t, s = scalar(expiry, "expiry"), scalar(bond_maturity, "bond_maturity")
    if big_t < 0.0:
        raise ValueError("expiry must be nonnegative")
    if s < big_t:
        raise ValueError("bond_maturity must not be before expiry")
    a_, s_ = _mean_reversion(a), _volatility(sigma)
    b = float(bond_b(np.array(big_t), np.array(s), a_))
    variance = float(bond_b(np.array(0.0), np.array(2.0 * big_t), a_)) / 2.0
    return s_ * b * math.sqrt(variance)


def _zbo(is_call: bool, strike: np.ndarray, p0t: float, p0s: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """Black on the forward bond: call
    ``P(0, S) N(h) - K P(0, T) N(h - s)``, put by the mirror; intrinsic for
    ``s = 0``."""
    k = strike * p0t
    out = np.empty_like(p0s)
    flat = sp <= 0.0
    sign = 1.0 if is_call else -1.0
    out[flat] = np.maximum(sign * (p0s[flat] - k[flat]), 0.0)
    live = ~flat
    h = np.log(p0s[live] / k[live]) / sp[live] + 0.5 * sp[live]
    if is_call:
        out[live] = p0s[live] * ndtr(h) - k[live] * ndtr(h - sp[live])
    else:
        out[live] = k[live] * ndtr(sp[live] - h) - p0s[live] * ndtr(-h)
    return np.maximum(out, 0.0)


def bond_option(is_call: bool, strike: float, expiry: float, bond_maturity: float, a: float, sigma: float,
                p0: Curve) -> float:
    """The price per unit nominal of a European option, expiring at ``T``, on a zero-coupon bond maturing at ``S`` with
    strike ``K`` (a bond price):
    ``ZBC = P(0, S) N(h) - K P(0, T) N(h - sigma_p)`` with
    ``h = ln(P(0, S) / (K P(0, T))) / sigma_p + sigma_p / 2``;
    put-call parity ``ZBC - ZBP = P(0, S) - K P(0, T)``. With
    ``sigma_p = 0`` the forward intrinsic value."""
    k = scalar(strike, "strike")
    if k <= 0.0:
        raise ValueError("strike must be positive")
    sp = bond_option_vol(expiry, bond_maturity, a, sigma)
    p0t = float(_eval(p0, np.array(float(expiry))))
    p0s = _eval(p0, np.array([float(bond_maturity)]))
    return float(_zbo(bool(is_call), np.array([k]), p0t, p0s, np.array([sp]))[0])


def _infer_nominal(amounts: np.ndarray) -> float:
    """The nominal added to the last amount of a constant-coupon fixed leg: ``a_n - a_1``."""
    if amounts.size < 2 or not np.allclose(amounts[:-1], amounts[0], rtol=1e-12, atol=0.0):
        raise ValueError("nominal cannot be inferred from fixed_amounts (it needs at least two payments with equal "
                         "coupons); pass nominal")
    n = float(amounts[-1] - amounts[0])
    if n <= 0.0:
        raise ValueError("nominal inferred from fixed_amounts is not positive; pass nominal")
    return n


def swaption_jamshidian(is_payer: bool, expiry: float, fixed_times: np.ndarray, fixed_amounts: np.ndarray, a: float,
                        sigma: float, p0: Curve, forward: Curve, nominal: float | None = None) -> float:
    """The European swaption on the remaining swap starting at the expiry, by the Jamshidian decomposition (addendum R4,
    QuantLib ``jamshidianswaptionengine.cpp``): ``r*`` solves
    ``N - sum c_i P(T_e, T_i; r*) = 0`` by Brent, the leg strikes are
    ``K_i = P(T_e, T_i; r*)`` and the value is ``sum c_i ZBO(K_i)``, a put for a payer
    swaption and a call for a receiver. ``fixed_times`` are the payment times after the expiry, ``fixed_amounts`` the
    coupon amounts with the nominal added to the last; the result is in the money of the nominal, at time 0.

    ``nominal`` is the notional ``N`` of the equation. The contract's signature has no such argument; it is optional and
    inferred as the last amount minus the first when the coupons are constant, and otherwise required (a one-payment leg
    cannot be split into coupon and nominal). Payer minus receiver equals the forward swap value."""
    te = scalar(expiry, "expiry")
    if te < 0.0:
        raise ValueError("expiry must be nonnegative")
    times, amounts = vec(fixed_times, "fixed_times"), vec(fixed_amounts, "fixed_amounts")
    same_length(fixed_times=times, fixed_amounts=amounts)
    if np.any(np.diff(times) <= 0.0) or times[0] <= te:
        raise ValueError("fixed_times must be strictly increasing and after the expiry")
    if np.any(amounts < 0.0) or amounts[-1] <= 0.0:
        raise ValueError("fixed_amounts must be nonnegative with a positive last amount")
    n = _infer_nominal(amounts) if nominal is None else scalar(nominal, "nominal")
    if n <= 0.0:
        raise ValueError("nominal must be positive")
    a_, s_ = _mean_reversion(a), _volatility(sigma)
    ta, tb = np.full(times.size, te), times
    big_a = bond_a(ta, tb, a_, s_, p0, forward)
    big_b = bond_b(ta, tb, a_)

    def gap(r: float) -> float:
        return n - float(np.sum(amounts * big_a * np.exp(-big_b * r)))

    lo, hi = JAMSHIDIAN_BRACKET
    if gap(lo) * gap(hi) > 0.0:
        raise ValueError("the coupon bond equation has no root in the bracket [-10, 10]")
    r_star = brentq(gap, lo, hi, xtol=JAMSHIDIAN_XTOL)
    strikes = big_a * np.exp(-big_b * r_star)
    p0t = float(_eval(p0, np.array(te)))
    p0s = _eval(p0, times)
    sp = np.array([bond_option_vol(te, float(s), a_, s_) for s in times])
    legs = _zbo(not bool(is_payer), strikes, p0t, p0s, sp)
    return float(np.sum(amounts * legs))


@dataclass(frozen=True)
class Ar1Fit:
    a: float
    sigma: float
    theta: float
    a_se: float
    sigma_se: float
    n_obs: int
    sample_start: str
    sample_end: str


def ar1_to_hw1f(b: float, delta: float, dt: float) -> tuple[float, float]:
    """The map of addendum R6, equations 55 to 57: ``a = -ln b / dt`` and
    ``sigma = delta / sqrt((b^2 - 1) dt / (2 ln b))``; ``b`` must
    lie in (0, 1)."""
    b_, d_, dt_ = scalar(b, "b"), scalar(delta, "delta"), scalar(dt, "dt")
    if not 0.0 < b_ < 1.0:
        raise ValueError(f"the series is not mean-reverting: the AR(1) coefficient b = {b_} must lie in (0, 1)")
    if d_ < 0.0 or dt_ <= 0.0:
        raise ValueError("delta must be nonnegative and dt positive")
    a_ = -math.log(b_) / dt_
    sigma = d_ / math.sqrt((b_**2 - 1.0) * dt_ / (2.0 * math.log(b_)))
    return a_, sigma


def calibrate_ar1(dates: list[str], short_rate: np.ndarray, before: str, min_years: float = 3.0,
                  dt: float | None = None) -> Ar1Fit:
    """OLS of ``x(t_i)`` on ``x(t_{i-1})`` for the observations strictly before ``before`` (no look-ahead), mapped to
    ``a`` and ``sigma`` by ``ar1_to_hw1f``, ``theta = c / (1 - b)`` (addendum R6, Brigo et
    al. arXiv 0812.4210). The residual spread is ``delta = std(resid, ddof = 1)``, as the listing of addendum R6 has it
    (the OLS variance of ``b`` keeps its divisor ``m - 2``). The
    standard errors are the OLS standard error of ``b`` propagated to ``a`` by the delta method, and for ``sigma`` the
    delta method on ``(b, delta)`` with the OLS standard error of ``b`` and the normal-theory
    ``delta / sqrt(2 (m - 2))`` of the residual spread, the two taken as independent.

    ``dt`` defaults to the median spacing of the dates in years of ``DAYS_PER_YEAR`` days. Fewer than ``min_years`` of
    history raises (CRE53.51); so does ``b`` outside (0, 1). ``n_obs`` is the number of observations used."""
    try:
        parsed = [_dt.date.fromisoformat(str(d)) for d in dates]
        cut = _dt.date.fromisoformat(str(before))
    except ValueError as exc:
        raise ValueError(f"dates and before must be ISO dates: {exc}") from exc
    x_all = vec(short_rate, "short_rate")
    if len(parsed) != x_all.size:
        raise ValueError(f"dates and short_rate must have the same length, got {len(parsed)} and {x_all.size}")
    days = np.array([d.toordinal() for d in parsed], dtype=float)
    if np.any(np.diff(days) <= 0.0):
        raise ValueError("dates must be strictly increasing")
    keep = np.array([d < cut for d in parsed], dtype=bool)
    x, d_days = x_all[keep], days[keep]
    n_obs = int(x.size)
    if n_obs < 4:
        raise ValueError(f"short_rate needs at least 4 observations before {before}, got {n_obs}")
    span = (d_days[-1] - d_days[0]) / DAYS_PER_YEAR
    min_y = scalar(min_years, "min_years")
    if span < min_y:
        raise ValueError(f"min_years: {span:.3f} years of history before {before} is below the {min_y} required "
                         "(CRE53.51)")
    step = float(np.median(np.diff(d_days)) / DAYS_PER_YEAR) if dt is None else scalar(dt, "dt")
    if step <= 0.0:
        raise ValueError("dt must be positive")
    y, xl = x[1:], x[:-1]
    m = y.size
    design = np.column_stack([np.ones(m), xl])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ coef
    df = m - 2
    s2 = float(resid @ resid) / df
    c, b = float(coef[0]), float(coef[1])
    cov = s2 * np.linalg.inv(design.T @ design)
    se_b = math.sqrt(max(float(cov[1, 1]), 0.0))
    if not 0.0 < b < 1.0:
        raise ValueError(f"the series is not mean-reverting: the AR(1) coefficient b = {b} must lie in (0, 1)")
    delta = float(np.std(resid, ddof=1))
    a_hat, sigma_hat = ar1_to_hw1f(b, delta, step)
    a_se = se_b / (b * step)
    h = 2.0 * math.log(b) / ((b**2 - 1.0) * step)                 # sigma = delta sqrt(h)
    dh = ((2.0 / b) * (b**2 - 1.0) - 4.0 * b * math.log(b)) / ((b**2 - 1.0) ** 2 * step)
    dsigma_db = delta * dh / (2.0 * math.sqrt(h))
    dsigma_ddelta = math.sqrt(h)
    se_delta = delta / math.sqrt(2.0 * df)
    sigma_se = math.sqrt((dsigma_db * se_b) ** 2 + (dsigma_ddelta * se_delta) ** 2)
    kept = [d for d, k in zip(parsed, keep, strict=True) if k]
    return Ar1Fit(a_hat, sigma_hat, c / (1.0 - b), a_se, sigma_se, n_obs, kept[0].isoformat(), kept[-1].isoformat())

__all__ = [
    "Ar1Fit",
    "ar1_to_hw1f",
    "bank_account",
    "bond_a",
    "bond_b",
    "bond_option",
    "bond_option_vol",
    "bond_price",
    "calibrate_ar1",
    "curve_functions",
    "phi",
    "simulate_short_rate",
    "simulate_state",
    "swaption_jamshidian",
]
