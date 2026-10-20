"""RV-1316: the dependence band (contract `engine-aggregation-api`, module aggregation.band)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.aggregation import band as bd
from riskvalidation.aggregation import bounds as b
from riskvalidation.aggregation import copulas as c
from riskvalidation.aggregation import simple as s
from riskvalidation.portfolio import euler
from riskvalidation.portfolio.portfolio import es_from_sample
from riskvalidation.validation.portfolio import euler_additivity

R5 = np.array([[1.0, 0.5], [0.5, 1.0]])
EC = np.array([3.0, 4.0])
EL0 = np.zeros(2)
SCALE_999 = 0.001 ** -0.5 - 1.0  # F^-1(0.999) of the unit Pareto(2) of the tests


def _pareto(scale):
    def ppf(u):
        with np.errstate(divide="ignore"):  # F^-1(1) is +infinity, the last cell of the upper matrix
            return scale * ((1.0 - np.asarray(u, dtype=float)) ** -0.5 - 1.0)

    return ppf


PPFS = [_pareto(3.0 / SCALE_999), _pareto(4.0 / SCALE_999)]  # the quantile at 0.999 is the EC of each marginal
KW = dict(level=0.999, n_draws=200_000, seed=5, chunk_size=50_000, bounds_n=2000, bounds_epsilon=1e-3)


@pytest.fixture(scope="module")
def band():
    return bd.dependence_band(PPFS, EC, EL0, R5, [4.0, 8.0], **KW)


def test_dependence_band_golden_and_invariants(band):
    methods = [r.method for r in band.rows]
    assert methods == ["lower-bound", "simple-sum", "variance-covariance", "gaussian-copula", "t-copula", "t-copula",
                       "worst-case"]
    by = {r.method: r for r in band.rows if r.method != "t-copula"}
    assert by["simple-sum"].capital == 7.0
    assert by["variance-covariance"].capital == pytest.approx(6.0828, abs=5e-5)
    assert by["variance-covariance"].capital == pytest.approx(math.sqrt(37.0), abs=1e-12)
    assert band.headline == by["simple-sum"].capital
    assert (band.horizon_years, band.confidence_level, band.capital_basis) == (1, 0.999, "economic-capital-var-less-el")
    assert [r.nu for r in band.rows] == [None, None, None, None, 4.0, 8.0, None]
    simple = by["simple-sum"].capital
    for r in band.rows:
        assert r.status == ("headline-capital" if r.method == "simple-sum" else "information-not-capital")
        assert r.diversification_factor == pytest.approx(r.capital / simple, rel=1e-12)
        if r.method in ("gaussian-copula", "t-copula"):
            assert r.mc_se is not None and r.mc_se > 0.0
        else:
            assert r.mc_se is None
    lower, worst = by["lower-bound"].capital, by["worst-case"].capital
    for r in band.rows:
        assert lower - 1e-9 <= r.capital <= worst + 1e-9
    assert lower <= 6.0828 <= worst  # the contract's golden
    # every simulated row is the reused copula aggregate on the same marginals and seed, less the same expected loss
    g = c.copula_aggregate(PPFS, R5, "gaussian", None, 0.0, 0.999, 200_000, 5, 50_000)
    assert by["gaussian-copula"].capital == pytest.approx(g.ec, rel=1e-12)
    assert by["gaussian-copula"].mc_se == pytest.approx(g.var_se, rel=1e-12)
    t4 = c.copula_aggregate(PPFS, R5, "t", 4.0, 0.0, 0.999, 200_000, 5, 50_000)
    assert [r for r in band.rows if r.nu == 4.0][0].capital == pytest.approx(t4.ec, rel=1e-12)
    wb = b.worst_var_bounds(PPFS, 0.999, n=2000, epsilon=1e-3, seed=5)
    assert by["lower-bound"].capital == pytest.approx(wb.lower_range[0], rel=1e-12)
    assert by["worst-case"].capital == pytest.approx(wb.upper_range[1], rel=1e-12)


def test_band_expected_loss_and_signed_lower_bound():
    el = np.array([0.5, 1.0])
    tiny = dict(KW, n_draws=20_000, bounds_n=500)
    out = bd.dependence_band(PPFS, EC - el, el, R5, [5.0], **tiny)
    wb = b.worst_var_bounds(PPFS, 0.999, n=500, epsilon=1e-3, seed=5)
    by = {r.method: r for r in out.rows}
    assert by["lower-bound"].capital == pytest.approx(wb.lower_range[0] - 1.5, rel=1e-12)
    assert by["worst-case"].capital == pytest.approx(wb.upper_range[1] - 1.5, rel=1e-12)
    # the simulated rows are EC = VaR - EL, never the VaR (review F1)
    g = c.copula_aggregate(PPFS, R5, "gaussian", None, 1.5, 0.999, 20_000, 5, 50_000)
    t5 = c.copula_aggregate(PPFS, R5, "t", 5.0, 1.5, 0.999, 20_000, 5, 50_000)
    assert by["gaussian-copula"].capital == pytest.approx(g.ec, rel=1e-12)
    assert by["gaussian-copula"].capital == pytest.approx(g.var - 1.5, rel=1e-12)
    assert by["t-copula"].capital == pytest.approx(t5.ec, rel=1e-12)
    assert by["t-copula"].capital == pytest.approx(t5.var - 1.5, rel=1e-12)
    # a lower bound below the expected loss is reported as computed, signed (Table 4, d = 56: 45.82 against an EL of 56)
    big_el = np.array([1.7, 1.7])  # the best VaR is 3.2: below the expected loss of 3.4
    neg = bd.dependence_band(PPFS, EC - big_el, big_el, R5, [5.0], **tiny)
    assert [r for r in neg.rows if r.method == "lower-bound"][0].capital < 0.0
    assert all(r.capital > 0.0 for r in neg.rows if r.method != "lower-bound")  # every other row is not below zero


def test_band_euler_additivity_of_the_aggregate():
    losses = c.simulate_component_losses(PPFS, R5, "gaussian", None, 50_000, seed=2)
    total = losses.sum(axis=1)
    rc = euler.es_contributions(losses, total, 0.999)
    assert rc.sum() == pytest.approx(es_from_sample(total, 0.999), rel=1e-12)
    res = euler_additivity(rc, es_from_sample(total, 0.999), tolerance=1e-9)
    assert res.test_id == "portfolio.euler_additivity" and res.light.value == "green"
    # the closed-form allocation of the variance-covariance row adds up to the row as well
    cov = np.outer(EC, EC) * R5
    assert s.std_contributions(cov).sum() == pytest.approx(
        [r for r in bd.dependence_band(PPFS, EC, EL0, R5, [5.0], **dict(KW, n_draws=2000, bounds_n=200)).rows
         if r.method == "variance-covariance"][0].capital, rel=1e-12)


def test_dependence_band_refusals_and_seed(band):
    again = bd.dependence_band(PPFS, EC, EL0, R5, [4.0, 8.0], **KW)
    assert again == band
    small = dict(KW, n_draws=2000, bounds_n=100)
    for bad in ([], [8.0, 4.0], [4.0, 4.0], [2.0, 4.0], [1.0]):
        with pytest.raises(ValueError, match="nu_set"):
            bd.dependence_band(PPFS, EC, EL0, R5, bad, **small)
    for lv in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="level"):
            bd.dependence_band(PPFS, EC, EL0, R5, [4.0], **dict(small, level=lv))
    with pytest.raises(ValueError, match="horizon_years"):
        bd.dependence_band(PPFS, EC, EL0, R5, [4.0], horizon_years=0, **small)
    with pytest.raises(ValueError, match="same length|length"):
        bd.dependence_band(PPFS, EC, np.zeros(3), R5, [4.0], **small)
    with pytest.raises(ValueError, match="same length|length"):
        bd.dependence_band(PPFS[:1], EC, EL0, R5, [4.0], **small)
    with pytest.raises(ValueError, match="positive semi-definite"):
        bd.dependence_band(PPFS, EC, EL0, np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]]), [4.0], **small)
    with pytest.raises(ValueError, match="ec"):
        bd.dependence_band(PPFS, np.zeros(2), EL0, R5, [4.0], **small)
    with pytest.raises(ValueError, match="ec"):
        bd.dependence_band(PPFS, np.array([3.0, -4.0]), EL0, R5, [4.0], **small)
    with pytest.raises(ValueError, match="el"):
        bd.dependence_band(PPFS, EC, np.array([0.0, float("nan")]), R5, [4.0], **small)


def test_band_row_invariants():
    ok = bd.BandRow("t-copula", 4.0, 5.0, 0.7, "information-not-capital", 0.1)
    assert ok.nu == 4.0
    with pytest.raises(ValueError, match="status"):
        bd.BandRow("simple-sum", None, 7.0, 1.0, "information-not-capital", None)
    with pytest.raises(ValueError, match="status"):
        bd.BandRow("gaussian-copula", None, 5.0, 0.7, "headline-capital", None)
    with pytest.raises(ValueError, match="nu"):
        bd.BandRow("gaussian-copula", 4.0, 5.0, 0.7, "information-not-capital", None)
    with pytest.raises(ValueError, match="nu"):
        bd.BandRow("t-copula", None, 5.0, 0.7, "information-not-capital", None)
    with pytest.raises(ValueError, match="method"):
        bd.BandRow("median", None, 5.0, 0.7, "information-not-capital", None)


def test_band_refuses_ec_that_contradicts_the_marginals():
    # review F6: ec = (100, 100) on unit Pareto(2) marginals whose 0.999 quantile is about 30.6
    with pytest.raises(ValueError, match="ec_consistency_tolerance|marginal 0"):
        bd.dependence_band([_pareto(1.0), _pareto(1.0)], np.array([100.0, 100.0]), EL0, R5, [4.0], **KW)
    with pytest.raises(ValueError, match="marginal 1"):
        bd.dependence_band(PPFS[:1] + [_pareto(1.0)], EC, EL0, R5, [4.0], **KW)
    # consistent marginals pass at the default tolerance, and a loose tolerance admits the contradiction
    bd.dependence_band(PPFS, EC, EL0, R5, [4.0], **dict(KW, n_draws=2000, bounds_n=200))
    bd.dependence_band([_pareto(1.0), _pareto(1.0)], np.array([100.0, 100.0]), EL0, R5, [4.0],
                       **dict(KW, n_draws=2000, bounds_n=200), ec_consistency_tolerance=10.0)
