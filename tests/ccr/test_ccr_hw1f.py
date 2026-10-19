"""RV-950 to RV-956 and RV-984: the Hull-White one-factor module of `riskvalidation.ccr` (contract `engine-ccr-api`,
module hw1f). Every golden value is recomputed here through a second route (the printed formula typed out with `math`),
never typed as the answer alone; the Monte Carlo checks are seeded."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from riskvalidation.ccr import hw1f

TENORS = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
ZEROS = np.array([0.030, 0.032, 0.035, 0.040, 0.042, 0.043])
SQRT_EPS = math.sqrt(np.finfo(float).eps)


def curve():
    return hw1f.curve_functions(ZEROS, TENORS)


def flat(rate: float = 0.03):
    def p0(t):
        return np.exp(-rate * np.asarray(t, dtype=float))

    def fwd(t):
        return np.full_like(np.asarray(t, dtype=float), rate)

    return p0, fwd


def arr(*x):
    return np.array(x, dtype=float)


# ---------------------------------------------------------------------------------------------------- RV-950


def test_phi_golden_and_invariants():
    p0, fwd = flat(0.03)
    got = hw1f.phi(arr(2.0), 0.1, 0.01, fwd)
    by_hand = 0.03 + 0.5 * (0.01 * (1.0 - math.exp(-0.1 * 2.0)) / 0.1) ** 2
    assert got[0] == pytest.approx(by_hand, abs=1e-15)
    assert got[0] == pytest.approx(0.030164, abs=5e-7)
    # phi(0) = f(0, 0); sigma = 0 gives the forward curve
    _, fc = curve()
    assert hw1f.phi(arr(0.0), 0.1, 0.01, fc)[0] == pytest.approx(float(fc(arr(0.0))[0]), abs=1e-15)
    t = arr(0.25, 1.0, 3.0, 8.0)
    assert np.allclose(hw1f.phi(t, 0.2, 0.0, fc), fc(t), atol=1e-15, rtol=0)
    # below sqrt(eps) the bracket is sigma t (and the limit is continuous)
    tiny = hw1f.phi(arr(2.0), 1e-12, 0.01, fwd)[0]
    assert tiny == pytest.approx(0.03 + 0.5 * (0.01 * 2.0) ** 2, abs=1e-15)
    assert hw1f.phi(arr(2.0), 1e-6, 0.01, fwd)[0] == pytest.approx(tiny, abs=1e-9)


def test_bond_b_golden_and_invariants():
    got = hw1f.bond_b(arr(0.0), arr(5.0), 0.1)
    assert got[0] == pytest.approx((1.0 - math.exp(-0.5)) / 0.1, abs=1e-15)
    assert got[0] == pytest.approx(3.934693, abs=5e-7)
    assert hw1f.bond_b(arr(1.5), arr(1.5), 0.3)[0] == 0.0
    assert hw1f.bond_b(arr(1.0), arr(4.0), 1e-12)[0] == pytest.approx(3.0, abs=1e-12)
    # broadcasts over an array of maturities
    out = hw1f.bond_b(arr(0.0), arr(1.0, 2.0, 5.0), 0.1)
    assert out.shape == (3,) and np.all(np.diff(out) > 0.0)


# ---------------------------------------------------------------------------------------------------- RV-951


def test_bond_a_golden_and_t0_identity():
    p0, fwd = flat(0.03)
    got = hw1f.bond_a(arr(0.0), arr(5.0), 0.1, 0.01, p0, fwd)[0]
    by_hand = math.exp(-0.15) * math.exp((1.0 - math.exp(-0.5)) / 0.1 * 0.03)
    assert got == pytest.approx(by_hand, abs=1e-14)
    assert got == pytest.approx(0.968546, abs=5e-7)
    assert float(p0(5.0)) == pytest.approx(0.860708, abs=5e-7)
    # at t = 0 and any a, sigma: A(0, T) = P(0, T) exp(B(0, T) f(0, 0)), not P(0, T)
    p1, f1 = curve()
    big_t = arr(1.0, 3.0, 7.0)
    for a, sigma in [(0.05, 0.0), (0.1, 0.01), (0.8, 0.03)]:
        a0 = hw1f.bond_a(np.zeros(3), big_t, a, sigma, p1, f1)
        want = p1(big_t) * np.exp(hw1f.bond_b(np.zeros(3), big_t, a) * float(f1(arr(0.0))[0]))
        assert np.allclose(a0, want, atol=1e-13, rtol=0)
    # A(t, t) = 1
    t = arr(0.5, 2.0, 6.0)
    assert np.allclose(hw1f.bond_a(t, t, 0.1, 0.01, p1, f1), 1.0, atol=1e-14, rtol=0)
    # the formula at t > 0, typed out with the QuantLib form 0.25 (sigma B)^2 B(0, 2t)
    a, sigma, tt, big = 0.15, 0.012, 2.0, 6.0
    b = (1.0 - math.exp(-a * (big - tt))) / a
    b2t = (1.0 - math.exp(-a * 2.0 * tt)) / a
    want = float(p1(big) / p1(tt)) * math.exp(b * float(f1(arr(tt))[0]) - 0.25 * (sigma * b) ** 2 * b2t)
    assert hw1f.bond_a(arr(tt), arr(big), a, sigma, p1, f1)[0] == pytest.approx(want, abs=1e-13)


def test_bond_price_reprices_the_curve_to_1e12():
    grid_t = np.concatenate([TENORS, arr(0.25, 0.75, 1.5, 3.3, 7.7, 12.0)])
    for p0, fwd in (curve(), flat(0.03)):
        for a, sigma in [(0.02, 0.0), (0.1, 0.01), (0.5, 0.02), (1.5, 0.05)]:
            r0 = hw1f.phi(arr(0.0), a, sigma, fwd)
            for big_t in grid_t:
                price = hw1f.bond_price(np.repeat(r0, 3), 0.0, float(big_t), a, sigma, p0, fwd)
                assert np.max(np.abs(price - float(p0(float(big_t))))) < 1e-12
    # decreasing in r, P(t, t) = 1
    p0, fwd = curve()
    r = np.linspace(-0.02, 0.1, 25)
    price = hw1f.bond_price(r, 1.0, 4.0, 0.1, 0.01, p0, fwd)
    assert np.all(np.diff(price) < 0.0)
    assert np.allclose(hw1f.bond_price(r, 2.0, 2.0, 0.1, 0.01, p0, fwd), 1.0, atol=1e-14, rtol=0)


# ---------------------------------------------------------------------------------------------------- RV-952


def _se_var(var: float, n: int) -> float:
    """Standard error of a sample variance of n iid normals (antithetic pairs halve the effective count)."""
    return var * math.sqrt(2.0 / (n / 2.0))


def test_simulate_state_golden_and_invariants():
    a, sigma = 0.1, 0.01
    theory = sigma**2 * (1.0 - math.exp(-2.0 * a * 5.0)) / (2.0 * a)
    assert theory == pytest.approx(0.00031606, abs=5e-9)
    n = 20_000
    for grid in (arr(0.0, 5.0), np.linspace(0.0, 5.0, 61)):          # the law does not depend on the step
        x = hw1f.simulate_state(grid, a, sigma, n, 7)
        assert x.shape == (n, grid.size) and np.all(x[:, 0] == 0.0)
        assert abs(float(np.var(x[:, -1])) - theory) <= 3.0 * _se_var(theory, n)
        assert np.max(np.abs(x.mean(axis=0))) < 1e-12                # antithetic
    # variance along the path: Var(x(t)) = sigma^2 (1 - e^(-2 a t)) / (2 a)
    grid = np.linspace(0.0, 5.0, 11)
    x = hw1f.simulate_state(grid, a, sigma, n, 11)
    for k in (2, 5, 10):
        v = sigma**2 * (1.0 - math.exp(-2.0 * a * grid[k])) / (2.0 * a)
        assert abs(float(np.var(x[:, k])) - v) <= 3.0 * _se_var(v, n)
    # reproducible for a seed, different for another
    assert np.array_equal(hw1f.simulate_state(grid, a, sigma, 100, 3), hw1f.simulate_state(grid, a, sigma, 100, 3))
    assert not np.array_equal(hw1f.simulate_state(grid, a, sigma, 100, 3), hw1f.simulate_state(grid, a, sigma, 100, 4))
    # without antithetic an odd number of paths is allowed and the mean is not forced to zero
    odd = hw1f.simulate_state(grid, a, sigma, 101, 3, antithetic=False)
    assert odd.shape == (101, grid.size)
    # the grid is refused unless it starts at 0 and strictly increases
    for bad in (arr(0.1, 1.0), arr(0.0, 1.0, 1.0), arr(0.0, 2.0, 1.0)):
        with pytest.raises(ValueError, match="grid"):
            hw1f.simulate_state(bad, a, sigma, 10, 1)
    with pytest.raises(ValueError, match="n_paths"):
        hw1f.simulate_state(grid, a, sigma, 101, 1)                  # antithetic needs an even count


# ---------------------------------------------------------------------------------------------------- RV-953


def test_bank_account_is_the_trapezoid_of_the_short_rate():
    grid = arr(0.0, 0.5, 1.0, 2.0)
    r = np.array([[0.02, 0.02, 0.02, 0.02], [0.0, 0.01, 0.02, 0.03]])
    n = hw1f.bank_account(grid, r)
    assert np.allclose(n[0], np.exp(0.02 * grid), atol=1e-15, rtol=0)
    want = np.exp(np.array([0.0, 0.5 * (0.0 + 0.01) / 2, 0.5 * 0.005 + 0.5 * (0.01 + 0.02) / 2,
                            0.5 * 0.005 + 0.5 * 0.015 + 1.0 * (0.02 + 0.03) / 2]))
    assert np.allclose(n[1], want, atol=1e-15, rtol=0)
    assert np.all(n[:, 0] == 1.0)


def test_simulate_short_rate_martingale_and_mean():
    p0, fwd = curve()
    a, sigma, n = 0.1, 0.01, 20_000
    grid = np.arange(0, 61) / 12.0                                   # monthly to 5 years
    r = hw1f.simulate_short_rate(grid, a, sigma, fwd, n, 5)
    assert r.shape == (n, grid.size)
    phi = hw1f.phi(grid, a, sigma, fwd)
    se = r.std(axis=0, ddof=1) / math.sqrt(n)
    assert np.all(np.abs(r.mean(axis=0) - phi) <= 4.0 * np.maximum(se, 1e-12))
    assert np.all(r[:, 0] == phi[0])                                 # x(0) = 0
    # martingale: E[exp(-integral r) P(t, T; r_t)] = P(0, T) at every grid point
    numeraire = hw1f.bank_account(grid, r)
    for big_t in (5.0, 7.0, 10.0):
        for k in range(1, grid.size, 6):
            price = hw1f.bond_price(r[:, k], float(grid[k]), big_t, a, sigma, p0, fwd) / numeraire[:, k]
            err = abs(float(price.mean()) - float(p0(big_t)))
            assert err <= 4.0 * float(price.std(ddof=1)) / math.sqrt(n), (big_t, k, err)
    assert np.array_equal(r, hw1f.simulate_short_rate(grid, a, sigma, fwd, n, 5))


# ---------------------------------------------------------------------------------------------------- RV-954


def _zbc(p0s, p0t, strike, sp):
    from scipy.stats import norm
    h = math.log(p0s / (strike * p0t)) / sp + sp / 2.0
    return p0s * norm.cdf(h) - strike * p0t * norm.cdf(h - sp)


def test_bond_option_golden_parity_and_monte_carlo():
    a, sigma, t, s = 0.1, 0.01, 2.0, 5.0
    b = (1.0 - math.exp(-a * (s - t))) / a
    sp = sigma * b * math.sqrt((1.0 - math.exp(-2.0 * a * t)) / (2.0 * a))
    assert hw1f.bond_option_vol(t, s, a, sigma) == pytest.approx(sp, abs=1e-15)
    assert hw1f.bond_option_vol(t, s, a, sigma) == pytest.approx(0.033276, abs=5e-7)
    assert hw1f.bond_option_vol(t, s, 1e-12, sigma) == pytest.approx(sigma * (s - t) * math.sqrt(t), abs=1e-12)
    p0, _ = curve()
    p0s, p0t = float(p0(s)), float(p0(t))
    for strike in (0.8, 0.9, p0s / p0t, 0.99, 1.1):
        call = hw1f.bond_option(True, strike, t, s, a, sigma, p0)
        put = hw1f.bond_option(False, strike, t, s, a, sigma, p0)
        assert call >= 0.0 and put >= 0.0
        assert call - put == pytest.approx(p0s - strike * p0t, abs=1e-12)
        assert call == pytest.approx(_zbc(p0s, p0t, strike, sp), abs=1e-14)
    # sigma = 0: the forward intrinsic value
    assert hw1f.bond_option(True, 0.5, t, s, a, 0.0, p0) == pytest.approx(max(p0s - 0.5 * p0t, 0.0), abs=1e-15)
    assert hw1f.bond_option(False, 0.5, t, s, a, 0.0, p0) == 0.0
    # Monte Carlo of bond_price under the bank account numeraire
    n = 100_000
    p1, fwd = curve()
    grid = np.linspace(0.0, t, 25)
    r = hw1f.simulate_short_rate(grid, a, sigma, fwd, n, 21)
    price = hw1f.bond_price(r[:, -1], t, s, a, sigma, p1, fwd)
    numeraire = hw1f.bank_account(grid, r)[:, -1]
    strike = p0s / p0t
    pay = np.maximum(price - strike, 0.0) / numeraire
    mc, se = float(pay.mean()), float(pay.std(ddof=1)) / math.sqrt(n)
    assert abs(mc - hw1f.bond_option(True, strike, t, s, a, sigma, p1)) <= 4.0 * se
    pay_put = np.maximum(strike - price, 0.0) / numeraire
    se_put = float(pay_put.std(ddof=1)) / math.sqrt(n)
    assert abs(float(pay_put.mean()) - hw1f.bond_option(False, strike, t, s, a, sigma, p1)) <= 4.0 * se_put


# ---------------------------------------------------------------------------------------------------- RV-955


def _leg(expiry, n_pay, rate, nominal=1.0):
    times = expiry + np.arange(1, n_pay + 1) * 0.5
    amounts = np.full(n_pay, nominal * rate * 0.5)
    amounts[-1] += nominal
    return times, amounts


def test_swaption_jamshidian_parity_and_monte_carlo():
    p0, fwd = curve()
    a, sigma = 0.1, 0.01
    expiry = 1.0
    for rate in (0.01, 0.035, 0.0381, 0.06, 0.20):
        times, amounts = _leg(expiry, 8, rate)
        payer = hw1f.swaption_jamshidian(True, expiry, times, amounts, a, sigma, p0, fwd)
        recv = hw1f.swaption_jamshidian(False, expiry, times, amounts, a, sigma, p0, fwd)
        assert payer >= 0.0 and recv >= 0.0
        forward_swap = float(p0(expiry)) - float(np.sum(amounts * p0(times)))      # payer pays fixed, receives float
        assert payer - recv == pytest.approx(forward_swap, abs=1e-9)
    # the nominal is inferred for a constant coupon and refused when it cannot be
    times, amounts = _leg(expiry, 4, 0.04)
    assert hw1f.swaption_jamshidian(True, expiry, times, amounts, a, sigma, p0, fwd) == pytest.approx(
        hw1f.swaption_jamshidian(True, expiry, times, amounts, a, sigma, p0, fwd, nominal=1.0), abs=1e-15)
    with pytest.raises(ValueError, match="nominal"):
        hw1f.swaption_jamshidian(True, expiry, arr(2.0), arr(1.02), a, sigma, p0, fwd)
    # sigma tending to zero: intrinsic when in the money, zero when out of the money
    times, amounts = _leg(expiry, 8, 0.01)
    itm = hw1f.swaption_jamshidian(True, expiry, times, amounts, a, 1e-7, p0, fwd)
    assert itm == pytest.approx(float(p0(expiry)) - float(np.sum(amounts * p0(times))), abs=1e-6)
    times, amounts = _leg(expiry, 8, 0.20)
    assert hw1f.swaption_jamshidian(True, expiry, times, amounts, a, 1e-9, p0, fwd) < 1e-12
    assert hw1f.swaption_jamshidian(True, expiry, times, amounts, a, 0.0, p0, fwd) == 0.0
    # Monte Carlo: the discounted payoff of the payer swaption at the expiry
    n = 200_000
    times, amounts = _leg(expiry, 8, 0.038)
    grid = np.linspace(0.0, expiry, 25)
    r = hw1f.simulate_short_rate(grid, a, sigma, fwd, n, 33)
    bonds = np.stack([hw1f.bond_price(r[:, -1], expiry, float(tt), a, sigma, p0, fwd) for tt in times], axis=1)
    numeraire = hw1f.bank_account(grid, r)[:, -1]
    for is_payer in (True, False):
        value = 1.0 - bonds @ amounts if is_payer else bonds @ amounts - 1.0
        pay = np.maximum(value, 0.0) / numeraire
        mc, se = float(pay.mean()), float(pay.std(ddof=1)) / math.sqrt(n)
        assert abs(mc - hw1f.swaption_jamshidian(is_payer, expiry, times, amounts, a, sigma, p0, fwd)) <= 4.0 * se


# ---------------------------------------------------------------------------------------------------- RV-956


def _monthly_dates(n: int, start=dt.date(1990, 1, 31)) -> list[str]:
    out, year, month = [], start.year, start.month
    for _ in range(n):
        last = (dt.date(year + month // 12, month % 12 + 1, 1) - dt.timedelta(days=1)).day
        out.append(dt.date(year, month, min(start.day, last)).isoformat())
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return out


def test_calibrate_ar1_golden_and_recovery():
    # the mapping of R6 equations 55 to 57, typed out
    a, sigma = hw1f.ar1_to_hw1f(0.99, 0.002, 1.0 / 12.0)
    assert a == pytest.approx(-math.log(0.99) * 12.0, abs=1e-15)
    assert a == pytest.approx(0.120604, abs=5e-7)
    assert sigma == pytest.approx(0.002 / math.sqrt((0.99**2 - 1.0) * (1.0 / 12.0) / (2.0 * math.log(0.99))), abs=1e-15)
    assert sigma == pytest.approx(0.006963, abs=5e-7)
    # a series simulated from simulate_state with known a, sigma is recovered within three standard errors
    true_a, true_sigma, n = 0.4, 0.01, 1200
    dates = _monthly_dates(n)
    hits = 0
    for seed in range(1, 9):
        x = hw1f.simulate_state(np.arange(n) / 12.0, true_a, true_sigma, 2, seed)[0]
        fit = hw1f.calibrate_ar1(dates, 0.03 + x, "2100-01-01", dt=1.0 / 12.0)
        assert fit.n_obs == n and fit.sample_start == dates[0] and fit.sample_end == dates[-1]
        assert fit.a_se > 0.0 and fit.sigma_se > 0.0
        assert fit.theta == pytest.approx(0.03, abs=0.01)
        hits += abs(fit.a - true_a) <= 3.0 * fit.a_se and abs(fit.sigma - true_sigma) <= 3.0 * fit.sigma_se
    assert hits >= 7
    # no look-ahead: only observations strictly before `before`, and later data cannot change the fit
    x = hw1f.simulate_state(np.arange(n) / 12.0, true_a, true_sigma, 2, 3)[0]
    series = 0.03 + x
    before = dates[600]
    fit = hw1f.calibrate_ar1(dates, series, before, dt=1.0 / 12.0)
    assert fit.sample_end < before and fit.n_obs == 600 and fit.sample_end == dates[599]
    changed = series.copy()
    changed[600:] += 5.0
    assert hw1f.calibrate_ar1(dates, changed, before, dt=1.0 / 12.0) == fit
    # the default step is the median spacing in years
    weekly = [(dt.date(2000, 1, 3) + dt.timedelta(days=7 * i)).isoformat() for i in range(800)]
    xw = hw1f.simulate_state(np.arange(800) * 7.0 / 365.25, 0.5, 0.01, 2, 5)[0]
    f_default = hw1f.calibrate_ar1(weekly, xw, "2100-01-01")
    f_explicit = hw1f.calibrate_ar1(weekly, xw, "2100-01-01", dt=7.0 / 365.25)
    assert f_default.a == pytest.approx(f_explicit.a, abs=1e-12)
    # fewer than min_years raises (CRE53.51), as does a series that is not mean-reverting
    with pytest.raises(ValueError, match="min_years"):
        hw1f.calibrate_ar1(dates[:30], series[:30], "2100-01-01", dt=1.0 / 12.0)
    assert hw1f.calibrate_ar1(dates[:30], series[:30], "2100-01-01", min_years=2.0, dt=1.0 / 12.0).n_obs == 30
    with pytest.raises(ValueError, match="mean-revert"):
        hw1f.calibrate_ar1(dates[:60], 0.01 * 1.01 ** np.arange(60), "2100-01-01", dt=1.0 / 12.0)
    with pytest.raises(ValueError, match="mean-revert"):                 # b = -1: not in (0, 1)
        hw1f.calibrate_ar1(dates[:60], 0.03 + 0.01 * (-1.0) ** np.arange(60), "2100-01-01", dt=1.0 / 12.0)
    with pytest.raises(ValueError, match="mean-revert"):
        hw1f.ar1_to_hw1f(1.0, 0.002, 1.0 / 12.0)
    with pytest.raises(ValueError, match="dates"):
        hw1f.calibrate_ar1(dates[:60][::-1], series[:60], "2100-01-01", dt=1.0 / 12.0)
    with pytest.raises(ValueError, match="short_rate"):
        hw1f.calibrate_ar1(dates[:60], series[:59], "2100-01-01", dt=1.0 / 12.0)


# ---------------------------------------------------------------------------------------------------- RV-984


def test_hw1f_refuses_invalid_inputs():
    p0, fwd = curve()
    with pytest.raises(ValueError, match="a"):
        hw1f.bond_b(arr(0.0), arr(1.0), -0.1)
    with pytest.raises(ValueError, match="sigma"):
        hw1f.phi(arr(1.0), 0.1, -0.01, fwd)
    with pytest.raises(ValueError, match="sigma"):
        hw1f.simulate_state(arr(0.0, 1.0), 0.1, -0.01, 10, 1)
    with pytest.raises(ValueError, match="n_paths"):
        hw1f.simulate_state(arr(0.0, 1.0), 0.1, 0.01, 0, 1)
    with pytest.raises(ValueError, match="strike"):
        hw1f.bond_option(True, -1.0, 1.0, 2.0, 0.1, 0.01, p0)
    with pytest.raises(ValueError, match="bond_maturity"):
        hw1f.bond_option(True, 0.9, 2.0, 1.0, 0.1, 0.01, p0)
    with pytest.raises(ValueError, match="fixed_times"):
        hw1f.swaption_jamshidian(True, 1.0, arr(2.0, 1.5), arr(0.02, 1.02), 0.1, 0.01, p0, fwd)
    with pytest.raises(ValueError, match="fixed_amounts"):
        hw1f.swaption_jamshidian(True, 1.0, arr(1.5, 2.0), arr(0.02, 1.02, 1.0), 0.1, 0.01, p0, fwd)
    with pytest.raises(ValueError, match="t"):
        hw1f.bond_price(arr(0.01), 2.0, 1.0, 0.1, 0.01, p0, fwd)
    with pytest.raises(ValueError, match="zero_rates"):
        hw1f.curve_functions(ZEROS[:3], TENORS)
