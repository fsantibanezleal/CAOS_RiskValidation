"""RV-101, RV-102, RV-107, RV-108, RV-109: PD calibration against published values and independent computations."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import special, stats

from riskvalidation import Light
from riskvalidation.validation import (
    critical_default_count,
    pd_binomial,
    pd_binomial_vasicek,
    pd_brier,
    pd_chi2_grades,
    pd_ece,
    pd_hosmer_lemeshow,
    pd_jeffreys,
    pd_jeffreys_grades,
    pd_normal_multiperiod,
    pd_spiegelhalter,
    pd_traffic_lights,
)


def test_jeffreys_matches_beta_cdf():
    for n, d, pd in [(1000, 12, 0.01), (50, 0, 0.02), (250, 9, 0.015), (10, 10, 0.5)]:
        r = pd_jeffreys(n, d, pd)
        # independent reference: the regularised incomplete beta function
        assert r.p_value == pytest.approx(special.betainc(d + 0.5, n - d + 0.5, pd), abs=1e-12)
        assert r.n == n and r.n_events == d
    # more defaults at the same PD give a smaller p (stronger evidence of underestimation)
    ps = [pd_jeffreys(1000, d, 0.01).p_value for d in range(5, 25)]
    assert all(a > b for a, b in itertools.pairwise(ps))
    # K + 1 results; the portfolio PD is the obligor-weighted average PD
    res = pd_jeffreys_grades([100, 300, 600], [1, 5, 20], [0.005, 0.02, 0.03], grades=["A", "B", "C"])
    assert [r.segment for r in res] == ["A", "B", "C", "portfolio"]
    port_pd = (100 * 0.005 + 300 * 0.02 + 600 * 0.03) / 1000
    assert res[-1].extras["pd_applied"] == pytest.approx(port_pd)
    assert res[-1].p_value == pytest.approx(special.betainc(26.5, 974.5, port_pd), abs=1e-12)
    with pytest.raises(ValueError):
        pd_jeffreys(10, 11, 0.1)


def test_wp14_critical_values():
    # BCBS WP14 (2005) appendix: PD 1%, N 1,000, confidence 99%
    expected = {0.0: 19, 0.05: 35, 0.10: 49, 0.15: 63, 0.20: 77}
    for rho, k in expected.items():
        assert critical_default_count(1000, 0.01, 0.99, rho) == k
    assert pd_binomial(1000, 10, 0.01).extras["critical_count"] == 19
    # the test rejects at D >= k*: p <= 1 - q at k*, p > 1 - q just below it
    at = pd_binomial_vasicek(1000, 35, 0.01, 0.05)
    below = pd_binomial_vasicek(1000, 34, 0.01, 0.05)
    assert at.p_value <= 0.01 < below.p_value
    assert at.light is Light.RED  # default policy: red below 1%
    assert pd_binomial(1000, 19, 0.01).p_value <= 0.01 < pd_binomial(1000, 18, 0.01).p_value


def test_chi2_spiegelhalter_hl_normal_formulas():
    # chi-square over grades, hand computation
    r = pd_chi2_grades([100, 200], [2, 10], [0.01, 0.04])
    t = (1 - 2) ** 2 / (100 * 0.01 * 0.99) + (8 - 10) ** 2 / (200 * 0.04 * 0.96)
    assert r.statistic == pytest.approx(t)
    assert r.extras["dof"] == 2
    assert r.p_value == pytest.approx(stats.chi2.sf(t, 2))

    rng = np.random.default_rng(7)
    f = rng.uniform(0.005, 0.2, 400)
    y = (rng.uniform(size=400) < f).astype(int)
    # Spiegelhalter, written out independently
    o = np.mean((f - y) ** 2)
    e = np.mean(f * (1 - f))
    v = np.sum(f * (1 - f) * (1 - 2 * f) ** 2) / f.size**2
    sp = pd_spiegelhalter(y, f)
    assert sp.statistic == pytest.approx((o - e) / math.sqrt(v))
    assert sp.p_value == pytest.approx(stats.norm.sf((o - e) / math.sqrt(v)))
    # Hosmer-Lemeshow with explicit decile groups
    order = np.argsort(f, kind="mergesort")
    hl = 0.0
    for g in np.array_split(order, 10):
        og, eg = y[g].sum(), f[g].sum()
        hl += (og - eg) ** 2 / (eg * (1 - eg / g.size))
    res = pd_hosmer_lemeshow(y, f)  # PDs given before the outcomes: as many degrees of freedom as groups
    assert res.statistic == pytest.approx(hl)
    assert res.p_value == pytest.approx(stats.chi2.sf(hl, 10))
    assert res.extras["dof"] == 10 and res.extras["fitted"] is False
    fitted = pd_hosmer_lemeshow(y, f, fitted=True)  # PDs estimated on these observations: G - 2
    assert fitted.p_value == pytest.approx(stats.chi2.sf(hl, 8)) and fitted.extras["dof"] == 8
    assert pd_hosmer_lemeshow(y, f, dof=7).extras["dof"] == 7
    # normal multi-period test
    dr = np.array([0.012, 0.018, 0.009, 0.025, 0.015])
    pdt = np.full(5, 0.012)
    d = dr - pdt
    s2 = (np.sum(d**2) - np.sum(d) ** 2 / 5) / 4
    stat = np.sum(d) / (math.sqrt(5) * math.sqrt(s2))
    nm = pd_normal_multiperiod(dr, pdt)
    assert nm.statistic == pytest.approx(stat)
    assert nm.p_value == pytest.approx(stats.norm.sf(stat))
    flat = pd_normal_multiperiod([0.02, 0.02], [0.01, 0.01])
    assert flat.light is Light.NOT_EVALUATED


def test_traffic_lights_multinomial():
    q = (0.5, 0.3, 0.15, 0.05)
    n = [1000, 1000, 1000, 1000]
    pd = [0.01] * 4
    d = [8, 14, 16, 22]
    res = pd_traffic_lights(n, d, pd, probs=q)
    # colours from the standardised rates and the WP14 cut-offs
    rstd = (np.array(d) - 10) / math.sqrt(1000 * 0.01 * 0.99)
    cuts = stats.norm.ppf(np.cumsum(q)[:3])
    colours = [int(np.searchsorted(cuts, x, side="left")) for x in rstd]
    counts = tuple(colours.count(c) for c in range(4))
    assert (res.extras["green"], res.extras["yellow"], res.extras["orange"], res.extras["red"]) == counts
    # independent brute force: every colour sequence over 4 years
    v_obs = 1000 * counts[0] + 100 * counts[1] + 10 * counts[2] + counts[3]
    p = 0.0
    for seq in itertools.product(range(4), repeat=4):
        c = tuple(seq.count(k) for k in range(4))
        v = 1000 * c[0] + 100 * c[1] + 10 * c[2] + c[3]
        if v <= v_obs:
            p += math.prod(q[k] for k in seq)
    assert res.statistic == v_obs
    assert res.p_value == pytest.approx(p, abs=1e-12)
    # all years green is the best profile: P(V <= v) = 1
    assert pd_traffic_lights(n, [5, 5, 5, 5], pd).p_value == pytest.approx(1.0)


def test_brier_murphy_identity():
    rng = np.random.default_rng(11)
    grades = rng.integers(0, 6, 2000)
    grade_pd = np.array([0.002, 0.006, 0.015, 0.04, 0.09, 0.2])
    f = grade_pd[grades]
    y = (rng.uniform(size=f.size) < f * 1.3).astype(int)
    r = pd_brier(y, f, groups=grades)
    x = r.extras
    assert x["decomposition_exact"] is True
    assert r.metric == pytest.approx(x["reliability"] - x["resolution"] + x["uncertainty"], abs=1e-12)
    assert r.metric == pytest.approx(np.mean((f - y) ** 2))
    assert x["brier_skill"] == pytest.approx(1 - r.metric / x["uncertainty"])
    # ECE on two hand-made bins
    y2 = np.array([0, 0, 1, 0, 1, 1])
    f2 = np.array([0.1, 0.1, 0.1, 0.6, 0.6, 0.6])
    ece = pd_ece(y2, f2, bins=2)
    assert ece.metric == pytest.approx(0.5 * abs(1 / 3 - 0.1) + 0.5 * abs(2 / 3 - 0.6))
    assert len(ece.extras["bins"]) == 2


def test_traffic_lights_table_equals_enumeration():
    """RV-313: the cached profile table equals a fresh multinomial enumeration for every profile of up to nine years;
    a year on the green boundary is green under the printed rule and yellow under ``ties="upper"``."""
    from riskvalidation.validation.calibration import traffic_light_colours, traffic_profile_cdf

    for probs in ((0.5, 0.3, 0.15, 0.05), (0.25, 0.25, 0.25, 0.25), (0.6, 0.2, 0.15, 0.05)):
        for t in range(1, 10):
            table = traffic_profile_cdf(t, probs)
            profiles = sorted(table)
            assert len(profiles) == (t + 1) * (t + 2) * (t + 3) // 6
            for prof in profiles:
                fresh = sum(stats.multinomial.pmf(q, t, probs) for q in profiles if q <= prof)
                assert table[prof] == pytest.approx(min(1.0, fresh), abs=1e-12)
    n, pd = np.full(5, 1000), np.full(5, 0.03)
    d = np.array([30, 30, 30, 30, 30])  # exactly N p defaults every year: R = 0, on the green boundary
    assert list(traffic_light_colours(n, d, pd)) == [0] * 5
    assert list(traffic_light_colours(n, d, pd, ties="upper")) == [1] * 5
    lower = pd_traffic_lights(n, d, pd)
    upper = pd_traffic_lights(n, d, pd, ties="upper")
    assert lower.extras["green"] == 5 and upper.extras["yellow"] == 5 and upper.p_value < lower.p_value
    assert lower.p_value == pytest.approx(1.0) and upper.p_value == pytest.approx(0.5**5)  # no green year
    with pytest.raises(ValueError):
        pd_traffic_lights(n, d, pd, ties="middle")
