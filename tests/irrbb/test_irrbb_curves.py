"""RV-1001 to RV-1006 and RV-1031: zero curves, the Nelson-Siegel, Svensson and dynamic Nelson-Siegel curves (contract
`engine-irrbb-api`, module curves). Each golden value is reproduced by an independent recomputation written in the
test with
`math`, never by a function under test."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from riskvalidation.irrbb import curves as c


def arr(*x):
    return np.array(x, dtype=float)


def ns_scalar(m, b0, b1, b2, tau):
    x = m / tau
    slope = (1 - math.exp(-x)) / x
    return b0 + b1 * slope + b2 * (slope - math.exp(-x))


# ---------------------------------------------------------------------------------------------------- RV-1001


def test_zero_rates_at_golden_and_invariants():
    tenors, rates = arr(1, 2), arr(0.01, 0.03)
    out = c.zero_rates_at(tenors, rates, arr(1.5, 0.5, 3))
    assert np.allclose(out, [0.02, 0.01, 0.03], atol=1e-15)
    # exact at the pillars, flat outside, linear inside (independent: the line through the two pillars)
    assert np.allclose(c.zero_rates_at(tenors, rates, tenors), rates, atol=1e-15)
    at = np.linspace(1, 2, 11)
    assert np.allclose(c.zero_rates_at(tenors, rates, at), 0.01 + 0.02 * (at - 1), atol=1e-15)
    assert c.zero_rates_at(tenors, rates, arr(0.0, 50.0)).tolist() == [0.01, 0.03]
    mids = c.zero_rates_at(arr(0.25, 1, 5, 10, 30), arr(0.02, 0.025, 0.03, 0.035, 0.04),
                           np.array([0.0028, 0.0417, 0.1667, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5,
                                     7.5, 8.5, 9.5, 12.5, 17.5, 25.0]))
    assert mids.shape == (19,) and np.all(np.diff(mids) >= 0)
    with pytest.raises(ValueError):
        c.zero_rates_at(arr(2, 1), arr(0.01, 0.02), arr(1.5))
    with pytest.raises(ValueError):
        c.zero_rates_at(arr(1, 2), arr(0.01), arr(1.5))
    with pytest.raises(ValueError):
        c.zero_rates_at(arr(1, 2), arr(0.01, float("nan")), arr(1.5))


def test_discount_factors_golden_and_invariants():
    assert c.discount_factors(arr(0.02), arr(2.5))[0] == pytest.approx(math.exp(-0.05), abs=1e-15)
    assert c.discount_factors(arr(0.02), arr(2.5))[0] == pytest.approx(0.951229, abs=5e-7)
    assert c.discount_factors(arr(0.07), arr(0.0))[0] == 1.0
    t = np.linspace(0, 30, 31)
    df_t = c.discount_factors(np.full(31, 0.03), t)
    assert np.all(np.diff(df_t) < 0) and np.all((df_t > 0) & (df_t <= 1))
    r = np.linspace(0.0, 0.1, 21)
    assert np.all(np.diff(c.discount_factors(r, np.full(21, 5.0))) < 0)
    with pytest.raises(ValueError):
        c.discount_factors(arr(0.02, 0.03), arr(1.0))
    with pytest.raises(ValueError):
        c.discount_factors(arr(0.02), arr(-1.0))


# ---------------------------------------------------------------------------------------------------- RV-1002


def par_price(zero_at, y, big_t, f=2):
    """Price of a par bond of maturity ``big_t`` (a multiple of 1 / f) from the zero rates (continuous compounding)."""
    n = int(round(big_t * f))
    return sum(y / f * math.exp(-zero_at(k / f) * k / f) for k in range(1, n + 1)) + math.exp(-zero_at(big_t) * big_t)


def test_bootstrap_zero_curve_golden_and_par_repricing():
    # golden: a flat 4 percent par curve with f = 2 gives 2 ln(1.02) at the tenors that are multiples of 1 / f
    t = arr(0.5, 1, 2, 5, 10)
    flat = c.bootstrap_zero_curve(t, np.full(5, 0.04))
    assert flat.tolist() == pytest.approx([2 * math.log(1.02)] * 5, abs=1e-14)
    assert flat[0] == pytest.approx(0.039605, abs=5e-7)
    # below 1 / f the money-market rule gives a slightly different rate, by design (the flatness holds on the multiples)
    short = c.bootstrap_zero_curve(arr(1 / 12, 0.25, 0.5, 1.0), np.full(4, 0.04))
    assert short[0] == pytest.approx(math.log(1 + 0.04 / 12) * 12, abs=1e-14)
    assert short[0] == pytest.approx(0.039933, abs=5e-7) and short[1] == pytest.approx(0.039801, abs=5e-7)
    assert short[1] == pytest.approx(math.log(1 + 0.04 * 0.25) / 0.25, abs=1e-14)
    assert short[2] == pytest.approx(2 * math.log(1.02), abs=1e-14) and short[3] == pytest.approx(short[2], abs=1e-14)
    # a sloped par curve on every coupon date: each par bond reprices at 1 with the bootstrapped zero rates
    tt = np.arange(1, 21) / 2
    y = 0.03 + 0.001 * np.sqrt(tt) + 0.0002 * tt
    z = c.bootstrap_zero_curve(tt, y)

    def zero_at(x):
        return float(np.interp(x, tt, z))

    for big_t, yy in zip(tt, y, strict=True):
        assert par_price(zero_at, yy, big_t) == pytest.approx(1.0, abs=1e-13)
    # a different frequency: quarterly coupons
    zq = c.bootstrap_zero_curve(np.arange(1, 9) / 4, np.full(8, 0.04), frequency=4)
    assert zq.tolist() == pytest.approx([4 * math.log(1.01)] * 8, abs=1e-14)
    # a pillar that is not a coupon date is read from the bootstrapped dates (linear in the zero rate)
    off = c.bootstrap_zero_curve(arr(0.5, 1.0, 1.2, 2.0), arr(0.03, 0.032, 0.033, 0.035))
    assert 1.2 > 1.0 and min(off[1], off[3]) <= off[2] <= max(off[1], off[3])
    # inputs are not mutated and the result is deterministic
    t0, y0 = tt.copy(), y.copy()
    assert np.array_equal(c.bootstrap_zero_curve(tt, y), z) and np.array_equal(tt, t0) and np.array_equal(y, y0)
    for bad in (dict(tenors=arr(1, 1, 2), par_yields=arr(0.03, 0.03, 0.03)),
                dict(tenors=arr(2, 1), par_yields=arr(0.03, 0.03)),
                dict(tenors=arr(0, 1), par_yields=arr(0.03, 0.03)),
                dict(tenors=arr(-1, 1), par_yields=arr(0.03, 0.03)),
                dict(tenors=arr(1, 2), par_yields=arr(0.03))):
        with pytest.raises(ValueError):
            c.bootstrap_zero_curve(**bad)
    with pytest.raises(ValueError):
        c.bootstrap_zero_curve(arr(1, 2), arr(0.03, 0.03), frequency=0)


# ---------------------------------------------------------------------------------------------------- RV-1003


def test_nelson_siegel_yield_golden_and_limits():
    y = c.nelson_siegel_yield(arr(1.0), 0.05, -0.02, 0.01, 2.0)
    assert y[0] == pytest.approx(ns_scalar(1.0, 0.05, -0.02, 0.01, 2.0), abs=1e-15)
    assert y[0] == pytest.approx(0.036065, abs=5e-7)
    short = c.nelson_siegel_yield(arr(1e-9, 0.0), 0.05, -0.02, 0.01, 2.0)
    assert short[0] == pytest.approx(0.03, abs=1e-9) and short[1] == pytest.approx(0.03, abs=1e-15)
    assert c.nelson_siegel_yield(arr(1e9), 0.05, -0.02, 0.01, 2.0)[0] == pytest.approx(0.05, abs=1e-6)
    grid = np.linspace(0.01, 40, 400)
    expect = np.array([ns_scalar(m, 0.05, -0.02, 0.01, 2.0) for m in grid])
    assert np.allclose(c.nelson_siegel_yield(grid, 0.05, -0.02, 0.01, 2.0), expect, atol=1e-14)
    with pytest.raises(ValueError):
        c.nelson_siegel_yield(arr(1.0), 0.05, -0.02, 0.01, 0.0)
    with pytest.raises(ValueError):
        c.nelson_siegel_yield(arr(-1.0), 0.05, -0.02, 0.01, 2.0)


def test_svensson_reduces_to_nelson_siegel():
    m = arr(0.5, 1, 2, 5, 10, 30)
    ns = c.nelson_siegel_yield(m, 0.05, -0.02, 0.01, 2.0)
    for tau2 in (0.3, 5.0, 17.0):
        assert np.allclose(c.svensson_yield(m, 0.05, -0.02, 0.01, 0.0, 2.0, tau2), ns, atol=1e-15)
    assert c.svensson_yield(arr(1.0), 0.05, -0.02, 0.01, 0.0, 2.0, 5.0)[0] == pytest.approx(0.036065, abs=5e-7)
    with_hump = c.svensson_yield(arr(1.0), 0.05, -0.02, 0.01, 0.005, 2.0, 6.0)[0]
    x = 1.0 / 6.0
    hump = 0.005 * ((1 - math.exp(-x)) / x - math.exp(-x))
    assert with_hump == pytest.approx(ns_scalar(1.0, 0.05, -0.02, 0.01, 2.0) + hump, abs=1e-15)
    with pytest.raises(ValueError):
        c.svensson_yield(arr(1.0), 0.05, -0.02, 0.01, 0.005, 2.0, -1.0)


# ---------------------------------------------------------------------------------------------------- RV-1004


def test_fit_nelson_siegel_recovers_golden():
    m = arr(0.5, 1, 2, 3, 5, 7, 10)
    y = c.nelson_siegel_yield(m, 0.05, -0.02, 0.01, 2.0)
    m0, y0 = m.copy(), y.copy()
    fit = c.fit_nelson_siegel(m, y)
    assert (fit.beta0, fit.beta1, fit.beta2, fit.tau) == pytest.approx((0.05, -0.02, 0.01, 2.0), abs=1e-8)
    assert fit.rmse < 1e-9
    assert np.array_equal(m, m0) and np.array_equal(y, y0)
    assert c.fit_nelson_siegel(m, y) == fit
    # the betas are the least-squares betas of the chosen tau: a noisy curve is not fitted exactly, and its residuals
    # are orthogonal to the three loadings
    rng = np.random.default_rng(3)
    yn = y + rng.normal(0, 2e-4, y.size)
    nf = c.fit_nelson_siegel(m, yn)
    resid = yn - c.nelson_siegel_yield(m, nf.beta0, nf.beta1, nf.beta2, nf.tau)
    x = m / nf.tau
    slope = (1 - np.exp(-x)) / x
    design = np.column_stack([np.ones_like(m), slope, slope - np.exp(-x)])
    assert np.allclose(design.T @ resid, 0.0, atol=1e-12)
    assert nf.rmse == pytest.approx(float(np.sqrt(np.mean(resid**2))), abs=1e-15)
    # a custom grid is used
    only = c.fit_nelson_siegel(m, y, tau_grid=arr(1.0, 3.0))
    assert only.tau in (1.0, 3.0) and only.rmse > 1e-6
    with pytest.raises(ValueError):
        c.fit_nelson_siegel(m[:3], y[:3])
    with pytest.raises(ValueError):
        c.fit_nelson_siegel(m, y[:-1])


def test_fit_svensson_recovers_golden_and_nests_nelson_siegel():
    m = arr(0.5, 1, 2, 3, 5, 7, 10, 20)
    y = c.svensson_yield(m, 0.05, -0.02, 0.01, 0.005, 2.0, 6.0)
    m0, y0 = m.copy(), y.copy()
    fit = c.fit_svensson(m, y)
    assert fit.rmse < 1e-6
    assert np.array_equal(m, m0) and np.array_equal(y, y0)
    refit = c.svensson_yield(m, fit.beta0, fit.beta1, fit.beta2, fit.beta3, fit.tau1, fit.tau2)
    assert float(np.sqrt(np.mean((refit - y) ** 2))) == pytest.approx(fit.rmse, abs=1e-12)
    assert c.fit_svensson(m, y) == fit
    # the family nests Nelson-Siegel: its RMSE is never above the Nelson-Siegel RMSE of the same data
    rng = np.random.default_rng(11)
    yn = y + rng.normal(0, 3e-4, y.size)
    assert c.fit_svensson(m, yn).rmse <= c.fit_nelson_siegel(m, yn).rmse + 1e-12
    with pytest.raises(ValueError):
        c.fit_svensson(m[:5], y[:5])


# ---------------------------------------------------------------------------------------------------- RV-1005


def test_dns_loadings_goldens_and_maximiser():
    load = c.dns_loadings(arr(12, 30, 60))
    assert load.shape == (3, 3) and np.all(load[:, 0] == 1.0)
    assert load[:, 1] == pytest.approx([0.709464, 0.459280, 0.266588], abs=5e-7)
    assert load[:, 2] == pytest.approx([0.227941, 0.298384, 0.240701], abs=5e-7)
    lam = 0.0609
    for m, (s, k) in zip((12, 30, 60), load[:, 1:], strict=True):
        x = lam * m
        assert s == pytest.approx((1 - math.exp(-x)) / x, abs=1e-15)
        assert k == pytest.approx((1 - math.exp(-x)) / x - math.exp(-x), abs=1e-15)
    # the curvature loading peaks at x* / lam = 29.446 months (not exactly 30); a grid search finds the same month
    months = np.linspace(20, 40, 200001)
    peak = months[int(np.argmax(c.dns_loadings(months)[:, 2]))]
    assert peak == pytest.approx(29.4463, abs=2e-4)
    assert abs(peak - 30.0) > 0.5
    assert c.dns_loadings(arr(29.446))[0, 2] > c.dns_loadings(arr(30.0))[0, 2]
    with pytest.raises(ValueError):
        c.dns_loadings(arr(0.0, 12.0))
    with pytest.raises(ValueError):
        c.dns_loadings(arr(12.0), lam=0.0)


def test_dns_yield_and_fit_roundtrip():
    m = arr(3, 12, 24, 60, 120)
    assert c.dns_yield(arr(30.0), 0.05, -0.02, 0.01)[0] == pytest.approx(0.043798, abs=5e-7)
    y = c.dns_yield(m, 0.05, -0.02, 0.01)
    assert np.allclose(y, c.dns_loadings(m) @ arr(0.05, -0.02, 0.01), atol=1e-15)
    assert c.dns_yield(arr(1e7), 0.05, -0.02, 0.01)[0] == pytest.approx(0.05, abs=1e-6)
    f = c.fit_dns_factors(m, y[None, :])
    assert f.shape == (1, 3) and np.allclose(f[0], [0.05, -0.02, 0.01], atol=1e-10)
    hist = np.vstack([c.dns_yield(m, 0.05 + 0.001 * k, -0.02 + 0.002 * k, 0.01 - 0.001 * k) for k in range(5)])
    fall = c.fit_dns_factors(m, hist)
    assert fall.shape == (5, 3)
    # each date is fitted on its own: refitting a single row gives the same factors
    assert np.allclose(c.fit_dns_factors(m, hist[2:3])[0], fall[2], atol=1e-12)
    assert np.allclose(fall[:, 0], 0.05 + 0.001 * np.arange(5), atol=1e-10)
    with pytest.raises(ValueError):
        c.fit_dns_factors(m[:2], y[None, :2])
    with pytest.raises(ValueError):
        c.fit_dns_factors(m, y[None, :4])


# ---------------------------------------------------------------------------------------------------- RV-1006


def test_forecast_dns_factors_invariants():
    const = np.tile(arr(0.05, -0.02, 0.01), (12, 1))
    paths = c.forecast_dns_factors(const, horizon=6, n_paths=4, seed=1)
    assert paths.shape == (4, 6, 3)
    assert np.allclose(paths, arr(0.05, -0.02, 0.01), atol=1e-12)
    rng = np.random.default_rng(5)
    hist = np.zeros((120, 3))
    phi = np.array([0.9, 0.8, 0.5])
    for t in range(1, 120):
        hist[t] = 0.01 + phi * hist[t - 1] + rng.normal(0, 0.01, 3)
    h0 = hist.copy()
    p1 = c.forecast_dns_factors(hist, 12, 3000, seed=7)
    assert np.array_equal(hist, h0)
    assert np.array_equal(p1, c.forecast_dns_factors(hist, 12, 3000, seed=7))
    assert not np.array_equal(p1, c.forecast_dns_factors(hist, 12, 3000, seed=8))
    # the first step depends on the last row only: two histories sharing the last row and the AR fit give one
    # conditional mean; and the cross-path mean of the first step is the AR(1) conditional mean of the last row
    x, yv = hist[:-1], hist[1:]
    for j in range(3):
        design = np.column_stack([np.ones(119), x[:, j]])
        (c0, ph), *_ = np.linalg.lstsq(design, yv[:, j], rcond=None)
        resid = yv[:, j] - design @ np.array([c0, ph])
        sd = math.sqrt(resid @ resid / (119 - 2))
        first = p1[:, 0, j]
        assert first.mean() == pytest.approx(c0 + ph * hist[-1, j], abs=4 * sd / math.sqrt(3000))
        assert first.std(ddof=1) == pytest.approx(sd, rel=0.06)
    # no look-ahead: changing nothing but a value inside the history changes the fit, not the dependence on rows after
    # it
    with pytest.raises(ValueError):
        c.forecast_dns_factors(hist[:3], 12, 10, 0)
    with pytest.raises(ValueError):
        c.forecast_dns_factors(hist, 0, 10, 0)
    with pytest.raises(ValueError):
        c.forecast_dns_factors(hist, 12, 0, 0)
    with pytest.raises(ValueError):
        c.forecast_dns_factors(hist[:, :2], 12, 10, 0)


# ---------------------------------------------------------------------------------------------------- RV-1031


def test_quantlib_cross_check_of_discount_factors():
    ql = pytest.importorskip("QuantLib", reason="QuantLib wheel not installed: the optional cross-check is skipped")
    today = ql.Date(2, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    cal, dc = ql.NullCalendar(), ql.Thirty360(ql.Thirty360.BondBasis)
    tenors = np.arange(1, 21) / 2
    par = 0.03 + 0.001 * np.sqrt(tenors) + 0.0002 * tenors
    helpers = []
    for t, y in zip(tenors, par, strict=True):
        maturity = today + ql.Period(int(round(t * 12)), ql.Months)
        schedule = ql.Schedule(today, maturity, ql.Period(6, ql.Months), cal, ql.Unadjusted, ql.Unadjusted,
                               ql.DateGeneration.Forward, False)
        helpers.append(ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(100.0)), 0, 100.0, schedule, [float(y)], dc,
                                              ql.Unadjusted, 100.0))
    curve = ql.PiecewiseLogLinearDiscount(today, helpers, dc)
    zero = c.bootstrap_zero_curve(tenors, par)
    for t, r in zip(tenors, zero, strict=True):
        date = today + ql.Period(int(round(t * 12)), ql.Months)
        assert c.discount_factors(arr(r), arr(t))[0] == pytest.approx(curve.discount(date), rel=1e-12)


def test_modules_do_not_import_quantlib():
    import riskvalidation.irrbb.curves as mod
    assert "QuantLib" not in copy.copy(vars(mod))
