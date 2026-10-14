"""RV-802 to RV-804: the historical-simulation family (contract `engine-market-api`, module market.historical)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.market import conventions as c
from riskvalidation.market import historical as h


def a(*x):
    return np.array(x, dtype=float)


def test_age_weights():
    w = h.age_weights(0.98, 100)
    assert w.shape == (100,)
    assert w[0] == pytest.approx(0.023058, abs=5e-7)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(w) < 0)
    assert w[5] == pytest.approx((1 - 0.98) / (1 - 0.98**100) * 0.98**5, abs=1e-15)
    # the worked example of Boudoukh, Richardson and Whitelaw (1998): the return 3 days ago weighs 0.0221
    assert w[2] == pytest.approx(0.0221, abs=5e-5)
    assert w[65 - 1] == pytest.approx(0.0063, abs=5e-5)
    for bad in ((1.0, 10), (0.0, 10), (0.9, 0)):
        with pytest.raises(ValueError):
            h.age_weights(*bad)


def test_historical_var_es():
    r = -0.001 * np.arange(1, 101)
    v, e = h.historical_var_es(r, 0.05, 100)
    assert v == pytest.approx(-0.096, abs=1e-12) and e == pytest.approx(-0.098, abs=1e-12)
    assert h.historical_var_es(r, 0.05, 40) == c.var_es_from_sample(r[-40:], 0.05)
    with pytest.raises(ValueError):
        h.historical_var_es(r, 0.05, 101)
    # the ghost effect: a crisis return that leaves the window moves the result
    x = np.full(300, 0.001)
    x[10] = -0.09
    inside = h.historical_var_es(x[:259], 0.01, 250)
    outside = h.historical_var_es(x[:261], 0.01, 250)
    assert inside[1] < outside[1]


def test_age_weighted_var_es_midpoint_rule():
    # three returns, window 3, lam 0.5: the weights of 1, 2 and 3 days ago are 4/7, 2/7 and 1/7
    r = a(-0.03, -0.02, -0.01)  # oldest first; observed 3, 2 and 1 days ago
    lam = 0.5
    w = h.age_weights(lam, 3)
    assert w == pytest.approx([4 / 7, 2 / 7, 1 / 7])
    # ascending: -0.03 (1/7), -0.02 (2/7), -0.01 (4/7). BRW 1998 pages 6 and 7: half of a return's weight on each side, up
    # to the point halfway to the neighbouring return, so the quantile function has the knots (C_j - w_j/2, r_j) and
    # (C_j, (r_j + r_{j+1})/2): (1/14, -0.03), (1/7, -0.025), (2/7, -0.02), (3/7, -0.015), (5/7, -0.01)
    alpha = 0.2
    expected_var = -0.025 + (alpha - 1 / 7) / (2 / 7 - 1 / 7) * 0.005
    v, e = h.age_weighted_var_es(r, alpha, lam, 3)
    assert v == pytest.approx(expected_var, abs=1e-12)
    # D-9: the ES is the tail integral of that piecewise linear function, constant -0.03 on (0, 1/14)
    area = (-0.03 * (1 / 14) + 0.5 * (-0.03 + -0.025) * (1 / 7 - 1 / 14)
            + 0.5 * (-0.025 + expected_var) * (alpha - 1 / 7))
    assert e == pytest.approx(area / alpha, abs=1e-12)
    assert e <= v
    # below the first knot the VaR is the smallest return and the ES equals it
    v0, e0 = h.age_weighted_var_es(r, 0.05, lam, 3)
    assert v0 == -0.03 and e0 == -0.03
    # the window is the last `window` returns
    long = np.concatenate([a(0.5, 0.5), r])
    assert h.age_weighted_var_es(long, alpha, lam, 3) == (v, e)


def brw_window(ages_and_returns):
    """A 100-day window (oldest first) with the given returns at the given ages (1 = observed one day ago) and benign
    positive returns elsewhere."""
    x = np.linspace(0.0, 0.01, 100)
    for age, ret in ages_and_returns.items():
        x[100 - age] = ret
    return x


def test_age_weighted_var_es_reproduces_the_brw_worked_example():
    """Boudoukh, Richardson and Whitelaw (1998) pages 6 and 7: lam 0.98, K 100; the 5% VaR is 2.63% at the initial date and
    2.34% twenty-five days later (the six lowest returns -3.3, -2.9, -2.7, -2.5, -2.4 and -2.3% observed 3, 2, 65, 45, 5 and
    30 days ago, then 28, 27, 90, 70, 30 and 55 days ago). The knots at the cumulated weights, valued halfway between
    adjacent returns, are what the paper's own arithmetic uses (2.70 - 0.10 (0.05 - 0.0478) / (0.0511 - 0.0478))."""
    rets = (-0.033, -0.029, -0.027, -0.025, -0.024, -0.023)
    first = brw_window(dict(zip((3, 2, 65, 45, 5, 30), rets, strict=True)))
    later = brw_window(dict(zip((28, 27, 90, 70, 30, 55), rets, strict=True)))
    v1, e1 = h.age_weighted_var_es(first, 0.05, 0.98, 100)
    v2, e2 = h.age_weighted_var_es(later, 0.05, 0.98, 100)
    assert round(-100 * v1, 2) == 2.63 and round(-100 * v2, 2) == 2.34
    assert v1 == pytest.approx(-(0.027 - 0.001 * (0.05 - 0.0478) / (0.0511 - 0.0478)), abs=2e-5)
    assert e1 <= v1 and e2 <= v2 and e1 >= -0.033
    # the paper's table: the cumulated weights of the first block
    w = h.age_weights(0.98, 100)
    assert np.cumsum(w[[2, 1, 64]]) == pytest.approx([0.0221, 0.0447, 0.0511], abs=6e-5)


def test_age_weighted_var_es_limits_and_invariants():
    r = -0.001 * np.arange(1, 101)
    hs = h.historical_var_es(r, 0.05, 100)
    aw = h.age_weighted_var_es(r, 0.05, 0.9999999, 100)
    assert abs(aw[0] - hs[0]) <= 0.0011 and abs(aw[1] - hs[1]) <= 0.0011
    rng = np.random.default_rng(3)
    x = rng.standard_t(4, size=600) * 0.01
    for lam in (0.98, 0.99):
        v, e = h.age_weighted_var_es(x, 0.01, lam, 250)
        assert e <= v
    # a recent loss weighs more than an old one
    old = np.zeros(250)
    new = np.zeros(250)
    old[0] = -0.1
    new[-1] = -0.1
    assert h.age_weighted_var_es(new, 0.02, 0.98, 250)[1] < h.age_weighted_var_es(old, 0.02, 0.98, 250)[1]
    with pytest.raises(ValueError):
        h.age_weighted_var_es(x[:10], 0.01, 0.98, 250)
    with pytest.raises(ValueError):
        h.age_weighted_var_es(x, 0.01, 1.0, 250)


def test_filtered_hs_var_es():
    z = a(-2, -1, 0, 1, 2)
    v, e = h.filtered_hs_var_es(z, 0.0, 0.01, 0.2)
    assert v == pytest.approx(-0.02, abs=1e-15) and e == pytest.approx(-0.02, abs=1e-15)
    rng = np.random.default_rng(4)
    zz = rng.standard_t(6, size=500)
    base = h.filtered_hs_var_es(zz, 0.0, 0.01, 0.025)
    dbl = h.filtered_hs_var_es(zz, 0.0, 0.02, 0.025)
    assert dbl[0] == pytest.approx(2 * base[0]) and dbl[1] == pytest.approx(2 * base[1])
    shifted = h.filtered_hs_var_es(zz, 0.001, 0.01, 0.025)
    assert shifted[0] == pytest.approx(base[0] + 0.001)
    assert c.var_es_from_sample(0.001 + 0.01 * zz, 0.025) == pytest.approx(shifted)
    for bad in (0.0, -0.01):
        with pytest.raises(ValueError):
            h.filtered_hs_var_es(zz, 0.0, bad, 0.025)


def test_volatility_updated_hs_var_es():
    v, e = h.volatility_updated_hs_var_es(a(-0.02, 0.01), a(0.01, 0.02), 0.015, 0.5)
    assert v == pytest.approx(-0.03, abs=1e-15) and e == pytest.approx(-0.03, abs=1e-15)
    rng = np.random.default_rng(5)
    r = rng.standard_t(5, size=400) * 0.01
    const = h.volatility_updated_hs_var_es(r, np.full(400, 0.012), 0.012, 0.025)
    assert const == pytest.approx(c.var_es_from_sample(r, 0.025), abs=1e-15)
    assert const == pytest.approx(h.historical_var_es(r, 0.025, 400), abs=1e-15)
    with pytest.raises(ValueError):
        h.volatility_updated_hs_var_es(r, np.full(399, 0.01), 0.01, 0.025)
    with pytest.raises(ValueError):
        h.volatility_updated_hs_var_es(r, np.full(400, 0.0), 0.01, 0.025)
