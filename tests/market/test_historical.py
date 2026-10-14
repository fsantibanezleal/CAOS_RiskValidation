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
    # three returns, window 3, lam such that the weights are known: the most recent is the largest
    r = a(-0.03, -0.02, -0.01)  # oldest first; observed 3, 2 and 1 days ago
    lam = 0.5
    w = h.age_weights(lam, 3)  # for 1, 2 and 3 days ago: 4/7, 2/7, 1/7
    assert w == pytest.approx([4 / 7, 2 / 7, 1 / 7])
    # ascending: -0.03 (weight w[2] = 1/7), -0.02 (2/7), -0.01 (4/7); midpoints 1/14, 1/7 + 1/7 = 2/7, 3/7 + 2/7 = 5/7
    p = np.array([1 / 14, 3 / 7 - 1 / 7, 3 / 7 + 2 / 7])
    assert p == pytest.approx([1 / 14, 2 / 7, 5 / 7])
    alpha = 0.2
    expected_var = -0.03 + (alpha - 1 / 14) / (2 / 7 - 1 / 14) * 0.01
    v, e = h.age_weighted_var_es(r, alpha, lam, 3)
    assert v == pytest.approx(expected_var, abs=1e-12)
    # the ES is the tail integral of the interpolated quantile function: constant -0.03 on (0, 1/14), then linear
    q_alpha = expected_var
    area = -0.03 * (1 / 14) + 0.5 * (-0.03 + q_alpha) * (alpha - 1 / 14)
    assert e == pytest.approx(area / alpha, abs=1e-12)
    assert e <= v
    # below the first midpoint the VaR is the smallest return and the ES equals it
    v0, e0 = h.age_weighted_var_es(r, 0.05, lam, 3)
    assert v0 == -0.03 and e0 == -0.03
    # the window is the last `window` returns
    long = np.concatenate([a(0.5, 0.5), r])
    assert h.age_weighted_var_es(long, alpha, lam, 3) == (v, e)


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
