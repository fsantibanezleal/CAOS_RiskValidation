"""RV-816 and RV-817: the FRTB statistics of dossier 01 D.4 (contract `engine-market-api`, module market.frtb)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from riskvalidation.market import frtb as f

TABLE_1 = [8.11, 28.58, 54.32, 75.81, 89.22, 95.88, 98.63, 99.60, 99.89, 99.97, 99.99]


def a(*x):
    return np.array(x, dtype=float)


def test_binomial_cumulative_probability_table():
    for k, pct in enumerate(TABLE_1):
        assert round(100 * f.binomial_cumulative_probability(k, 250, 0.01), 2) == pct
    cum = [f.binomial_cumulative_probability(k, 250, 0.01) for k in range(0, 251)]
    assert all(x <= y for x, y in zip(cum, cum[1:], strict=False)) and cum[-1] == pytest.approx(1.0, abs=1e-15)
    assert f.binomial_cumulative_probability(7, 40, 0.3) == pytest.approx(stats.binom.cdf(7, 40, 0.3), abs=1e-14)
    assert f.binomial_cumulative_probability(4, 250, 0.01) == pytest.approx(0.892185, abs=5e-6)
    with pytest.raises(ValueError):
        f.binomial_cumulative_probability(1, 0, 0.01)
    with pytest.raises(ValueError):
        f.binomial_cumulative_probability(1, 10, 1.2)


def test_traffic_light_zone_and_multiplier():
    assert [f.traffic_light_zone(k) for k in (0, 4)] == ["green", "green"]
    assert [f.traffic_light_zone(k) for k in (5, 9)] == ["amber", "amber"]
    assert [f.traffic_light_zone(k) for k in (10, 11, 40)] == ["red", "red", "red"]
    zones = [f.traffic_light_zone(k) for k in range(0, 30)]
    order = {"green": 0, "amber": 1, "red": 2}
    assert [order[z] for z in zones] == sorted(order[z] for z in zones)
    # the zones are the 95 percent and 99.99 percent cumulative levels, also for another window
    n, cov = 500, 0.99
    amber = next(k for k in range(n + 1) if f.binomial_cumulative_probability(k, n, 1 - cov) >= 0.95)
    red = next(k for k in range(n + 1) if f.binomial_cumulative_probability(k, n, 1 - cov) >= 0.9999)
    assert f.traffic_light_zone(amber - 1, n, cov) == "green" and f.traffic_light_zone(amber, n, cov) == "amber"
    assert f.traffic_light_zone(red - 1, n, cov) == "amber" and f.traffic_light_zone(red, n, cov) == "red"
    assert [f.mar32_multiplier(k) for k in (0, 4)] == [1.50, 1.50]
    assert [f.mar32_multiplier(k) for k in range(5, 10)] == [1.70, 1.76, 1.83, 1.88, 1.92]
    assert [f.mar32_multiplier(k) for k in (10, 25)] == [2.00, 2.00]
    m = [f.mar32_multiplier(k) for k in range(0, 15)]
    assert m == sorted(m)
    with pytest.raises(ValueError):
        f.traffic_light_zone(-1)
    with pytest.raises(ValueError):
        f.mar32_multiplier(-1)


def test_desk_limit_and_exceptions():
    assert f.desk_exception_limit(0.01) == 12 and f.desk_exception_limit(0.025) == 30
    for bad in (0.05, 0.02, 0.0):
        with pytest.raises(ValueError):
            f.desk_exception_limit(bad)
    loss = a(0.01, 0.03, np.nan, 0.02)
    var = a(0.02, 0.02, 0.02, 0.02)
    assert f.desk_exceptions(loss, var, missing_as_exception=False) == 1
    assert f.desk_exceptions(loss, var, missing_as_exception=True) == 2
    assert f.desk_exceptions(loss, var) == 2
    assert f.desk_exceptions(a(0.02), a(0.02)) == 0  # equality is no exception
    assert f.desk_exceptions(a(0.01, 0.01), a(np.nan, 0.02), missing_as_exception=True) == 1
    assert f.desk_exceptions(loss, var) <= loss.size
    with pytest.raises(ValueError):
        f.desk_exceptions(loss, var[:3])


def test_pla_spearman():
    h = np.arange(250, dtype=float)
    assert f.pla_spearman(h, h) == pytest.approx(1.0, abs=1e-15)
    assert f.pla_spearman(h, h[::-1].copy()) == pytest.approx(-1.0, abs=1e-15)
    sw = h.copy()
    sw[0], sw[1] = sw[1], sw[0]
    assert f.pla_spearman(h, sw) == pytest.approx(1 - 6 * 2 / (250 * (250**2 - 1)), abs=1e-12)
    assert round(f.pla_spearman(h, sw), 6) == 0.999999
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=250), rng.normal(size=250)
    base = f.pla_spearman(x, y)
    assert f.pla_spearman(np.exp(3 * x), y**3) == pytest.approx(base, abs=1e-12)
    assert base == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)
    tied = np.round(x, 1)  # ties take the average rank
    assert f.pla_spearman(tied, y) == pytest.approx(stats.spearmanr(tied, y).statistic, abs=1e-12)
    with pytest.raises(ValueError):
        f.pla_spearman(h[:249], h[:249])
    with pytest.raises(ValueError):
        f.pla_spearman(np.where(h == 3, np.nan, h), h)
    with pytest.raises(ValueError):
        f.pla_spearman(np.ones(250), h)


def test_pla_ks_is_an_exact_step_count():
    h = np.arange(250, dtype=float)
    assert f.pla_ks(h, h + 0.5) == 0.004
    assert f.pla_ks(h, h) == 0.0
    ks30 = f.pla_ks(h, h + 29.5)
    assert ks30 == 0.12  # exactly the float 0.12, not 0.12000000000000002
    assert f.pla_ks(h, h + 30.5) == pytest.approx(0.124, abs=1e-15)
    assert f.pla_ks(h + 30.5, h) == f.pla_ks(h, h + 30.5)  # symmetric
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=250), rng.normal(size=250)
    ref = stats.ks_2samp(x, y).statistic
    assert f.pla_ks(x, y) == pytest.approx(ref, abs=1e-12) and 0.0 <= f.pla_ks(x, y) <= 1.0
    with pytest.raises(ValueError):
        f.pla_ks(h[:100], h[:100])
    with pytest.raises(ValueError):
        f.pla_ks(np.where(h == 3, np.nan, h), h)


def test_pla_spearman_comparison_is_exact_arithmetic():
    """Review m4: a zone is decided on integer ranks, never on a float quotient compared with 0.80 or 0.70."""
    x = np.arange(1.0, 6.0)
    y = np.array([2.0, 1.0, 4.0, 3.0, 5.0])         # sum d^2 = 4: rho = 1 - 6 * 4 / (5 * 24) = 0.8 exactly
    assert f.compare_spearman(x, y, 0.80) == 0
    assert f.compare_spearman(x, y, 0.79) == 1 and f.compare_spearman(x, y, 0.81) == -1
    assert f.compare_spearman(x, y[::-1].copy(), -0.8) in (-1, 0, 1)
    tied_a = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
    tied_b = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0])
    rho = stats.spearmanr(tied_a, tied_b).statistic
    assert f.compare_spearman(tied_a, tied_b, rho - 1e-9) == 1 and f.compare_spearman(tied_a, tied_b, rho + 1e-9) == -1
    assert f.compare_spearman(x, x, 1.0) == 0 and f.compare_spearman(x, x[::-1].copy(), -1.0) == 0
    with pytest.raises(ValueError):
        f.compare_spearman(np.ones(5), x, 0.8)
    # the zone of two 250-day series agrees with pla_zone away from the boundaries, and decides the KS on the integer gap
    h = np.arange(250, dtype=float)
    for shift, zone in ((0.5, "green"), (22.5, "green"), (23.5, "amber"), (29.5, "amber"), (30.5, "red")):
        assert f.pla_zone_exact(h, h + shift) == zone, shift
    rng = np.random.default_rng(9)
    for _ in range(20):
        x1, y1 = rng.normal(size=250), rng.normal(size=250) * 0.7 + 0.9 * rng.normal(size=250)
        assert f.pla_zone_exact(x1, y1) == f.pla_zone(f.pla_spearman(x1, y1), f.pla_ks(x1, y1))


def test_pla_zone():
    cases = [((0.81, 0.08), "green"), ((0.80, 0.05), "amber"), ((0.85, 0.09), "amber"), ((0.69, 0.05), "red"),
             ((0.70, 0.05), "amber"), ((0.95, 0.121), "red"), ((0.95, 0.12), "amber"), ((0.95, 0.0), "green"),
             ((0.60, 0.20), "red")]
    for (rho, ks), zone in cases:
        assert f.pla_zone(rho, ks) == zone, (rho, ks)
    h = np.arange(250, dtype=float)
    assert f.pla_zone(f.pla_spearman(h, h + 29.5), f.pla_ks(h, h + 29.5)) == "amber"
    assert f.pla_zone(f.pla_spearman(h, h + 30.5), f.pla_ks(h, h + 30.5)) == "red"
