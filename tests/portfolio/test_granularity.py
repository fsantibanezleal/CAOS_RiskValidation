"""RV-713 and RV-714: the granularity adjustment of Gordy and Lutkebohmert (2013), dossier 04 H.5, ladder rung C5."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from riskvalidation.portfolio.granularity import (
    GranularityResult,
    delta,
    gamma_quantile,
    granularity_adjustment,
    k_creditrisk_plus,
    lgd_variance,
)


def test_gamma_quantile_delta_and_lgd_variance_goldens():
    a = gamma_quantile(0.25, 0.999)
    assert a == pytest.approx(17.505777, abs=5e-7)
    assert a == pytest.approx(float(stats.gamma.ppf(0.999, a=0.25, scale=4.0)), rel=1e-12)  # shape xi, rate xi
    d = delta(0.25, 0.999)
    assert d == pytest.approx(4.8336, abs=5e-5) and abs(d - 4.83) < 0.005
    assert d == pytest.approx((a - 1.0) * (0.25 + 0.75 / a), rel=1e-14)
    assert abs(d - 1.0867) > 3.0  # the other reading of xi (variance xi) is not the paper's
    # mean 1 and variance 1/xi
    assert float(stats.gamma.mean(a=0.25, scale=4.0)) == 1.0
    assert float(stats.gamma.var(a=0.25, scale=4.0)) == pytest.approx(4.0)
    qs = [gamma_quantile(0.25, q) for q in (0.5, 0.9, 0.99, 0.999)]
    assert qs == sorted(qs) and gamma_quantile(0.25, 0.9) > 1.0
    assert delta(0.25, 0.999) > delta(0.25, 0.99) > 0
    assert lgd_variance(0.45, 0.25) == pytest.approx(0.061875, abs=5e-7)
    v = lgd_variance(np.array([0.0, 0.45, 1.0]), 0.25)
    assert v == pytest.approx([0.0, 0.061875, 0.0], abs=5e-7)
    for bad_xi, bad_q in ((0.0, 0.999), (-1.0, 0.999), (0.25, 0.0), (0.25, 1.0)):
        with pytest.raises(ValueError):
            gamma_quantile(bad_xi, bad_q)
        with pytest.raises(ValueError):
            delta(bad_xi, bad_q)
    with pytest.raises(ValueError):
        lgd_variance(np.array([0.5]), -0.1)
    with pytest.raises(ValueError):
        lgd_variance(np.array([1.2]), 0.25)
    with pytest.raises(ValueError):
        lgd_variance(np.array([-0.1]), 0.25)


def _identical(n: int, k: float):
    s = np.full(n, 1.0 / n)
    return dict(share=s, k=np.full(n, k), pd=np.full(n, 0.01), lgd_mean=np.full(n, 0.45), lgd_gamma=0.25, xi=0.25,
                q=0.999)


def test_k_creditrisk_plus_golden():
    a = gamma_quantile(0.25, 0.999)
    k = k_creditrisk_plus(np.array([0.01]), np.array([0.45]), np.array([0.5]), a)
    assert k[0] == pytest.approx(0.037138, abs=5e-7) and k[0] == pytest.approx(0.45 * 0.01 * 0.5 * 16.505777, abs=5e-7)
    # the same as E[LGD] (pi(alpha_q) - PD) with the conditional PD pi(x) = PD (1 + omega (x - 1))
    pi = 0.01 * (1.0 + 0.5 * (a - 1.0))
    assert k[0] == pytest.approx(0.45 * (pi - 0.01), rel=1e-12)
    assert k_creditrisk_plus(np.array([0.01]), np.array([0.45]), np.array([0.0]), a)[0] == 0.0
    with pytest.raises(ValueError):
        k_creditrisk_plus(np.array([0.01]), np.array([0.45]), np.array([1.5]), a)
    with pytest.raises(ValueError):
        k_creditrisk_plus(np.array([0.01, 0.02]), np.array([0.45]), np.array([0.5]), a)


def test_granularity_adjustment_goldens_and_scaling():
    k = float(k_creditrisk_plus(np.array([0.01]), np.array([0.45]), np.array([0.5]), gamma_quantile(0.25, 0.999))[0])
    r10 = granularity_adjustment(**_identical(10, k))
    assert isinstance(r10, GranularityResult)
    assert r10.c == pytest.approx(np.full(10, 0.5875), abs=5e-7) and 0.45 * 0.01 == pytest.approx(0.0045)
    assert r10.k_star == pytest.approx(0.037138, abs=5e-7) and r10.delta == pytest.approx(4.8336, abs=5e-5)
    assert r10.alpha_q == pytest.approx(17.505777, abs=5e-7)
    assert r10.ga == pytest.approx(0.129817, abs=5e-7)  # a share of EAD (eq. 9), not scaled by K*
    r100 = granularity_adjustment(**_identical(100, k))
    assert r100.ga == pytest.approx(0.012982, abs=5e-7)
    # proportional to 1 / n for identical names, nonnegative, tending to zero
    sizes = (10, 50, 100, 1000, 10_000)
    vals = [granularity_adjustment(**_identical(n, k)).ga for n in sizes]
    for n, v in zip(sizes, vals, strict=True):
        assert v * n == pytest.approx(10 * vals[0], rel=1e-12) and v >= 0.0
    assert vals[-1] < 1e-3 * vals[0] * 1.0001
    # the formula by hand for two unequal names (eq. 17)
    s = np.array([0.7, 0.3])
    kk = np.array([0.04, 0.02])
    pdv = np.array([0.01, 0.03])
    lg = np.array([0.45, 0.3])
    res = granularity_adjustment(s, kk, pdv, lg, 0.25, 0.25, 0.999)
    cc = (0.25 * lg * (1 - lg) + lg**2) / lg
    d = (17.505777 - 1.0) * (0.25 + 0.75 / 17.505777)
    by_hand = float(np.sum(s**2 * cc * (d * (kk + lg * pdv) - kk))) / (2.0 * float(s @ kk))
    assert res.ga == pytest.approx(by_hand, rel=1e-6) and res.k_star == pytest.approx(0.7 * 0.04 + 0.3 * 0.02)
    # nonnegative at q >= 0.99, and a larger LGD variance gives a larger adjustment
    for q in (0.99, 0.999):
        assert granularity_adjustment(**{**_identical(20, 0.03), "q": q}).ga >= 0.0
    hi = granularity_adjustment(**{**_identical(20, 0.03), "lgd_gamma": 0.5}).ga
    assert hi > granularity_adjustment(**{**_identical(20, 0.03), "lgd_gamma": 0.0}).ga
    base = _identical(4, 0.03)
    bad_cases = (dict(share=np.array([0.5, 0.2, 0.2, 0.2])), dict(k=np.array([-0.01, 0.03, 0.03, 0.03])),
                 dict(k=np.zeros(4)), dict(lgd_mean=np.zeros(4)), dict(pd=np.full(4, 1.0)), dict(xi=0.0), dict(q=1.0),
                 dict(k=np.full(3, 0.03)))
    for bad in bad_cases:
        with pytest.raises(ValueError):
            granularity_adjustment(**{**base, **bad})
