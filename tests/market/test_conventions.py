"""RV-801: the sign convention and VaR and ES from a sample or a distribution (contract `engine-market-api`, module
market.conventions)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from riskvalidation.market import conventions as c


def a(*x):
    return np.array(x, dtype=float)


def test_loss_number_conversion():
    out = c.to_loss_numbers(a(-0.023263, -0.026652))
    assert np.array_equal(out, a(0.023263, 0.026652))
    assert np.array_equal(c.from_loss_numbers(a(0.02, 0.03)), a(-0.02, -0.03))
    x = np.random.default_rng(1).normal(size=(4, 3))
    assert np.array_equal(c.from_loss_numbers(c.to_loss_numbers(x)), x)
    assert np.array_equal(c.to_loss_numbers(c.from_loss_numbers(x)), x)
    n = c.to_loss_numbers(a(np.nan, -0.01))
    assert np.isnan(n[0]) and n[1] == 0.01
    assert c.to_loss_numbers(x).shape == x.shape


def test_var_es_from_sample_goldens():
    r = -0.001 * np.arange(1, 101)
    v, e = c.var_es_from_sample(r, 0.05)
    assert v == pytest.approx(-0.096, abs=1e-12) and e == pytest.approx(-0.098, abs=1e-12)
    v, e = c.var_es_from_sample(r, 0.025)  # k = 3, alpha N = 2.5: the third return carries the weight 0.5
    assert v == pytest.approx(-0.098, abs=1e-12)
    assert e == pytest.approx((-0.100 - 0.099 + 0.5 * -0.098) / 2.5, abs=1e-12)
    assert e == pytest.approx(-0.0992, abs=1e-12)
    # the 250-day rung: alpha N is 2.5 at 0.01 (k = 3) and 6.25 at 0.025 (k = 7)
    y = np.arange(1, 251, dtype=float)
    v, e = c.var_es_from_sample(y, 0.01)
    assert v == 3.0 and e == pytest.approx((1 + 2 + 0.5 * 3) / 2.5)
    v, e = c.var_es_from_sample(y, 0.025)
    assert v == 7.0 and e == pytest.approx((sum(range(1, 7)) + 0.25 * 7) / 6.25)


def test_var_es_from_sample_invariants_and_refusals():
    rng = np.random.default_rng(2)
    r = rng.standard_t(5, size=777) * 0.01
    v, e = c.var_es_from_sample(r, 0.01)
    assert e <= v
    v2, e2 = c.var_es_from_sample(rng.permutation(r), 0.01)
    assert (v, e) == (v2, e2)
    before = r.copy()
    c.var_es_from_sample(r, 0.025)
    assert np.array_equal(r, before)
    for bad_alpha in (0.0, 0.51, -0.1, 0.7):
        with pytest.raises(ValueError):
            c.var_es_from_sample(r, bad_alpha)
    # the golden of volatility_updated_hs_var_es uses alpha = 0.5, so the closed upper bound is accepted
    assert c.var_es_from_sample(a(1, 2, 3, 4), 0.5) == (2.0, 1.5)
    with pytest.raises(ValueError):
        c.var_es_from_sample(a(), 0.05)
    with pytest.raises(ValueError):
        c.var_es_from_sample(a(0.1, np.nan, -0.2), 0.05)
    # a floating product just above an integer must not move k
    k5 = c.var_es_from_sample(-np.arange(1, 101, dtype=float), 0.05)
    assert k5[0] == -96.0


def test_var_es_from_distribution():
    v, e = c.var_es_from_distribution(stats.norm.ppf, 0.01)
    assert v == pytest.approx(-2.326348, abs=1e-5) and e == pytest.approx(-2.665214, abs=1e-5)
    v, e = c.var_es_from_distribution(stats.norm.ppf, 0.025)
    assert v == pytest.approx(-1.959964, abs=1e-5) and e == pytest.approx(-2.337803, abs=1e-5)
    mu, sg = 0.002, 0.013
    v2, e2 = c.var_es_from_distribution(lambda u: mu + sg * stats.norm.ppf(u), 0.01)
    assert v2 == pytest.approx(mu + sg * -2.326348, abs=1e-6) and e2 == pytest.approx(mu + sg * -2.665214, abs=1e-6)
    assert e2 <= v2
    t = stats.t(5)
    vt, et = c.var_es_from_distribution(t.ppf, 0.025)
    closed = -(t.pdf(t.ppf(0.025)) / 0.025) * (5 + t.ppf(0.025) ** 2) / 4
    assert et == pytest.approx(closed, abs=1e-4) and et <= vt
    with pytest.raises(ValueError):
        c.var_es_from_distribution(stats.norm.ppf, 0.6)
