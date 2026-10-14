"""RV-805 and RV-806: delta-normal, Cornish-Fisher and EWMA (contract `engine-market-api`, module market.parametric)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.market import parametric as p


def a(*x):
    return np.array(x, dtype=float)


def test_delta_normal_var_es():
    v, e = p.delta_normal_var_es(0.0, 1.0, 0.01)
    assert v == pytest.approx(-2.326348, abs=5e-7) and e == pytest.approx(-2.665214, abs=5e-7)
    v, e = p.delta_normal_var_es(0.0, 1.0, 0.025)
    assert v == pytest.approx(-1.959964, abs=5e-7) and e == pytest.approx(-2.337803, abs=5e-7)
    v1, e1 = p.delta_normal_var_es(0.001, 0.02, 0.01)
    v0, e0 = p.delta_normal_var_es(0.0, 1.0, 0.01)
    assert v1 == pytest.approx(0.001 + 0.02 * v0) and e1 == pytest.approx(0.001 + 0.02 * e0)
    assert e1 <= v1
    z = stats.norm.ppf(0.01)
    assert e0 == pytest.approx(-stats.norm.pdf(z) / 0.01, abs=1e-12)
    with pytest.raises(ValueError):
        p.delta_normal_var_es(0.0, 0.0, 0.01)
    with pytest.raises(ValueError):
        p.delta_normal_var_es(0.0, 1.0, 0.6)


def test_cornish_fisher_var():
    r = p.cornish_fisher_var(0.0, 1.0, -0.5, 1.0, 0.01)
    assert r.z_cf == pytest.approx(-2.833709, abs=5e-7) and r.var == pytest.approx(-2.833709, abs=5e-7)
    assert r.valid is True
    r = p.cornish_fisher_var(0.0, 1.0, -3.0, 20.0, 0.01)
    assert r.z_cf == pytest.approx(-5.821010, abs=5e-7) and r.valid is False
    n = p.cornish_fisher_var(0.002, 0.01, 0.0, 0.0, 0.025)
    assert n.var == pytest.approx(p.delta_normal_var_es(0.002, 0.01, 0.025)[0], abs=1e-15) and n.valid
    z = stats.norm.ppf(0.025)
    s, k = 0.3, 2.0
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    c = p.cornish_fisher_var(0.0, 2.0, s, k, 0.025)
    assert c.z_cf == pytest.approx(zcf, abs=1e-12) and c.var == pytest.approx(2.0 * zcf, abs=1e-12)
    assert isinstance(c, p.CornishFisher)
    with pytest.raises(ValueError):
        p.cornish_fisher_var(0.0, -1.0, 0.0, 0.0, 0.01)


def test_ewma_variance_and_effective_days():
    out = p.ewma_variance(a(0.02), 0.94, 0.0001)
    assert out == pytest.approx([0.0001, 0.000118], abs=1e-15)
    r = np.random.default_rng(6).normal(0, 0.01, size=50)
    s2 = p.ewma_variance(r, 0.94, 1e-4)
    assert s2.shape == (51,) and np.all(s2 > 0)
    assert s2[10] == pytest.approx(0.94 * s2[9] + 0.06 * r[9] ** 2, abs=1e-18)
    with pytest.raises(ValueError):
        p.ewma_variance(r, 0.94, 0.0)
    with pytest.raises(ValueError):
        p.ewma_variance(r, 1.0, 1e-4)
    assert p.ewma_effective_days(0.94, 0.01) == pytest.approx(74.427, abs=5e-4)
    assert p.ewma_effective_days(0.97, 0.01) == pytest.approx(151.191, abs=5e-4)
    assert math.floor(p.ewma_effective_days(0.94, 0.01)) == 74 and math.floor(p.ewma_effective_days(0.97, 0.01)) == 151
    assert p.ewma_effective_days(0.99, 0.01) > p.ewma_effective_days(0.94, 0.01)


def test_ewma_var_es():
    v, e = p.ewma_var_es(a(0.02), 0.94, 0.01, 0.0001)
    assert v == pytest.approx(-0.025271, abs=5e-7) and e == pytest.approx(-0.028952, abs=5e-7)
    r = np.random.default_rng(7).normal(0, 0.01, size=300)
    s2 = p.ewma_variance(r, 0.94, 1e-4)[-1]
    assert p.ewma_var_es(r, 0.94, 0.025, 1e-4) == p.delta_normal_var_es(0.0, math.sqrt(s2), 0.025)
