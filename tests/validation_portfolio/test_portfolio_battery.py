"""RV-724 to RV-727: the economic capital validation tests (dossier 05 G.5); each returns one `TestResult`."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from riskvalidation.core.result import Light, TestResult
from riskvalidation.engines.vasicek import lhp_quantile
from riskvalidation.portfolio.creditrisk_plus import CreditRiskPlusInput, pmf_stable
from riskvalidation.validation import portfolio as val
from riskvalidation.validation.portfolio import (
    correlation_size_power,
    crp_recursion_vs_mc,
    euler_additivity,
    mc_convergence,
    tail_backtest,
)

ROW_FIELDS = {
    "test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
    "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras",
}
ONE_YEAR = np.array([0.0045, 0.0085, 0.0160, 0.0120, 0.0080, 0.0050])
TWO_YEARS = np.array([0.0045, 0.0085, 0.0160, 0.0230, 0.0080, 0.0050])


def test_tail_backtest_golden():
    q = [float(lhp_quantile(x, 0.003, 0.06)) for x in (0.95, 0.99, 0.999)]
    assert q == pytest.approx([0.007791, 0.012340, 0.020018], abs=5e-7)
    r = tail_backtest(ONE_YEAR, 0.003, 0.06, model_id="m", segment="cards")
    assert isinstance(r, TestResult)
    assert r.test_id == "portfolio.tail_backtest"
    assert r.model_id == "m"
    assert r.segment == "cards"
    assert r.statistic == 1.0 and r.metric == pytest.approx(0.06, abs=1e-12)
    assert r.p_value == pytest.approx(0.058520, abs=5e-7) and r.light is Light.GREEN
    assert (r.n, r.n_events) == (6, 1) and (r.alpha_amber, r.alpha_red) == (0.05, 0.01)
    levels = {row["level"]: row for row in r.extras["levels"]}
    assert set(levels) == {0.95, 0.99, 0.999}
    assert levels[0.99]["fitted_quantile"] == pytest.approx(0.012340, abs=5e-7) and levels[0.99]["observed"] == 1
    assert levels[0.99]["years_exceeding"] == [2] and levels[0.99]["expected"] == pytest.approx(0.06)
    assert levels[0.95]["observed"] == 4 and levels[0.95]["years_exceeding"] == [1, 2, 3, 4]
    assert levels[0.95]["p_value"] == pytest.approx(stats.binom.sf(3, 6, 0.05), abs=1e-12)
    assert levels[0.999]["observed"] == 0 and levels[0.999]["p_value"] == pytest.approx(1.0)
    assert r.extras["n_missing"] == 0
    two = tail_backtest(TWO_YEARS, 0.003, 0.06)
    assert two.statistic == 2.0 and two.p_value == pytest.approx(0.001460, abs=5e-7) and two.light is Light.RED
    # another primary level, and a year the table lacks (excluded, counted, never interpolated)
    p95 = tail_backtest(np.append(ONE_YEAR, np.nan), 0.003, 0.06, primary_level=0.95)
    assert p95.statistic == 4.0
    assert p95.metric == pytest.approx(6 * 0.05)
    assert p95.n == 6
    assert p95.extras["n_missing"] == 1
    assert p95.p_value == pytest.approx(stats.binom.sf(3, 6, 0.05), abs=1e-12)
    # the exceedance is strict
    edge = tail_backtest(np.array([float(lhp_quantile(0.99, 0.003, 0.06))] * 2), 0.003, 0.06)
    assert edge.statistic == 0.0 and edge.p_value == pytest.approx(1.0)
    assert r.to_dict()["inputs_hash"] and tail_backtest(ONE_YEAR, 0.003, 0.06).inputs_hash == r.inputs_hash
    # the light follows the p-value and the alphas
    assert tail_backtest(TWO_YEARS, 0.003, 0.06, alpha_red=0.0001, alpha_amber=0.05).light is Light.AMBER


def test_mc_convergence_golden():
    red = mc_convergence(0.1030, 0.0010, 0.1000, quantity="var", model_id="m")
    assert red.test_id == "portfolio.mc_convergence"
    assert red.statistic == pytest.approx(3.0, abs=1e-9) and red.p_value == pytest.approx(0.002700, abs=5e-7)
    assert red.metric == pytest.approx(0.003, abs=1e-12) and red.light is Light.RED
    assert red.extras["estimate"] == 0.1030 and red.extras["anchor"] == 0.1 and red.extras["se"] == 0.001
    assert red.extras["quantity"] == "var"
    green = mc_convergence(0.1015, 0.0010, 0.1000)
    assert green.statistic == pytest.approx(1.5, abs=1e-9) and green.p_value == pytest.approx(0.133614, abs=5e-7)
    assert green.light is Light.GREEN
    amber = mc_convergence(0.1022, 0.0010, 0.1000)  # z = 2.2, p = 0.0278
    assert amber.light is Light.AMBER
    below = mc_convergence(0.0970, 0.0010, 0.1000)
    assert below.statistic == pytest.approx(-3.0, abs=1e-9) and below.p_value == pytest.approx(0.0027, abs=5e-7)
    assert mc_convergence(0.1, 0.001, 0.1, quantity="es").extras["quantity"] == "es"


def test_euler_additivity_golden():
    ok = euler_additivity(np.array([0.5, 0.3, 0.2]), 1.0, model_id="m")
    assert ok.test_id == "portfolio.euler_additivity"
    assert ok.statistic == pytest.approx(1.0) and ok.metric == pytest.approx(0.0, abs=1e-15) and ok.p_value is None
    assert ok.light is Light.GREEN and ok.extras["thresholds"] == {"tolerance": 1e-6}
    assert ok.extras["sum"] == pytest.approx(1.0) and ok.extras["total"] == 1.0
    bad = euler_additivity(np.array([0.5, 0.3, 0.1]), 1.0)
    assert bad.statistic == pytest.approx(0.9) and bad.metric == pytest.approx(0.1) and bad.light is Light.RED
    loose = euler_additivity(np.array([0.5, 0.3, 0.19]), 1.0, tolerance=0.02)
    assert loose.light is Light.GREEN and loose.extras["thresholds"] == {"tolerance": 0.02}
    exact = euler_additivity(np.array([0.995, 0.0]), 1.0, tolerance=0.005)
    assert exact.metric == pytest.approx(0.005) and exact.light is Light.GREEN  # on the tolerance is within it
    neg = euler_additivity(np.array([3.0, 4.0]), -7.0)
    assert neg.metric == pytest.approx(2.0)  # relative to the absolute total


def test_euler_additivity_writes_its_basis():
    """D16-17: the basis is the field the packaged policy keys its tolerance by (1e-06 and 0.01)."""
    es = euler_additivity(np.array([0.5, 0.3, 0.2]), 1.0, basis="exact-allocation")
    kernel = euler_additivity(np.array([0.5, 0.3, 0.195]), 1.0, tolerance=0.01, basis="kernel-var")
    assert es.extras["basis"] == "exact-allocation" and kernel.extras["basis"] == "kernel-var"
    assert euler_additivity(np.array([0.5, 0.3, 0.2]), 1.0).extras["basis"] is None
    assert es.policy_version == kernel.policy_version == "contraste-portfolio-battery 1.0.0"


def test_crp_recursion_vs_mc_golden():
    pmf = np.array([0.5, 0.3, 0.2])
    same = crp_recursion_vs_mc(pmf, np.repeat([0, 1, 2], [5, 3, 2]))
    assert same.test_id == "portfolio.crp_recursion_vs_mc"
    assert same.statistic == pytest.approx(0.0, abs=1e-15)
    assert same.p_value == pytest.approx(1.0)
    assert same.light is Light.GREEN
    assert same.metric == pytest.approx(0.0, abs=1e-15) and same.extras["negative_entries"] == 0
    off = crp_recursion_vs_mc(pmf, np.repeat([0, 1, 2], [3, 3, 4]))
    assert off.statistic == pytest.approx(0.2, abs=1e-12) and off.p_value == pytest.approx(0.818621, abs=5e-7)
    assert off.light is Light.GREEN and off.n == 10
    far = crp_recursion_vs_mc(pmf, np.repeat([0, 1, 2], [200, 300, 500]))
    assert far.light is Light.RED and far.p_value < 0.01
    # any negative entry of the recursion is red whatever the p-value
    neg = crp_recursion_vs_mc(np.array([0.5, 0.31, 0.2, -0.01]), np.repeat([0, 1, 2], [5, 3, 2]))
    assert neg.extras["negative_entries"] == 1 and neg.light is Light.RED


def test_crp_recursion_agrees_with_monte_carlo_of_the_model():
    rng = np.random.default_rng(2024)
    n = 30
    nu = rng.integers(1, 5, size=n)
    pd = rng.uniform(0.01, 0.06, size=n)
    w = rng.dirichlet([2.0, 1.0, 1.0], size=n)
    s2 = np.array([0.5, 1.0])
    inp = CreditRiskPlusInput(nu=nu, pd=pd, w0=w[:, 0], weights=w[:, 1:], sigma2=s2, loss_unit=1.0, currency="USD")
    pmf = pmf_stable(inp, 400)
    s = 200_000
    sectors = np.column_stack([rng.gamma(1.0 / v, v, size=s) for v in s2])
    counts = rng.poisson(pd[None, :] * (w[None, :, 0] + sectors @ w[:, 1:].T))
    r = crp_recursion_vs_mc(pmf, counts @ nu)
    assert r.light is Light.GREEN and r.extras["negative_entries"] == 0 and r.n == s
    # a wrong recursion (the Poisson one, without the sectors) is told apart from the Monte Carlo of the sectors model
    poisson = CreditRiskPlusInput(nu=nu, pd=pd, w0=np.ones(n), weights=np.zeros((n, 0)), sigma2=np.zeros(0),
                                  loss_unit=1.0, currency="USD")
    wrong = crp_recursion_vs_mc(pmf_stable(poisson, 400), counts @ nu)
    assert wrong.light is Light.RED


def test_correlation_size_power_golden():
    size = correlation_size_power(np.arange(1000) < 52, 0.10, 0.10, alpha=0.05, estimator="ml", model_id="sim")
    assert size.test_id == "portfolio.correlation_size_power" and size.extras["role"] == "size"
    assert size.statistic == pytest.approx(0.290191, abs=5e-7) and size.p_value == pytest.approx(0.771671, abs=5e-7)
    assert size.metric == pytest.approx(0.052) and size.light is Light.GREEN and size.n == 1000 and size.n_events == 52
    e = size.extras
    assert e["known_rho"] == 0.10
    assert e["null_rho"] == 0.10
    assert e["estimator"] == "ml"
    assert e["rate"] == pytest.approx(0.052)
    se = (0.052 * 0.948 / 1000) ** 0.5
    assert e["se"] == pytest.approx(se)
    z = 1.959963984540054
    centre = (0.052 + z * z / 2000) / (1 + z * z / 1000)
    half = z * ((0.052 * 0.948 / 1000 + z * z / 4e6) ** 0.5) / (1 + z * z / 1000)
    assert e["ci_low"] == pytest.approx(centre - half, abs=1e-9)
    assert e["ci_high"] == pytest.approx(centre + half, abs=1e-9)
    assert e["thresholds"] == {"power_green": 0.8, "power_amber": 0.5}
    over = correlation_size_power(np.arange(1000) < 120, 0.10, 0.10)
    assert over.light is Light.RED and over.statistic > 0
    power = correlation_size_power(np.arange(1000) < 830, 0.18, 0.10)
    assert power.extras["role"] == "power" and power.statistic is None and power.p_value is None
    assert power.metric == pytest.approx(0.83) and power.light is Light.GREEN
    mid = correlation_size_power(np.arange(1000) < 600, 0.18, 0.10)
    assert mid.metric == pytest.approx(0.6) and mid.light is Light.AMBER
    low = correlation_size_power(np.arange(1000) < 400, 0.18, 0.10)
    assert low.metric == pytest.approx(0.4) and low.light is Light.RED
    edge = correlation_size_power(np.arange(1000) < 500, 0.18, 0.10, power_amber=0.5)
    assert edge.light is Light.AMBER  # on the gate is within it


def test_degenerate_inputs_are_not_evaluated():
    results = [
        tail_backtest(np.array([np.nan, np.nan]), 0.003, 0.06),
        tail_backtest(np.array([]), 0.003, 0.06),
        mc_convergence(0.1, 0.0, 0.1),
        mc_convergence(0.1, -0.001, 0.1),
        mc_convergence(float("nan"), 0.001, 0.1),
        mc_convergence(0.1, 0.001, float("inf")),
        euler_additivity(np.array([0.0, 0.0]), 0.0),
        crp_recursion_vs_mc(np.array([0.5, 0.3, 0.2]), np.array([0, 1, 2, 1, 0])),   # fewer than 10
        crp_recursion_vs_mc(np.array([0.5, 0.3, 0.2]), np.array([0, 1, 2, 3] * 5)),  # a sample above the grid
        correlation_size_power(np.zeros(0, dtype=bool), 0.1, 0.1),
    ]
    for r in results:
        assert r.light is Light.NOT_EVALUATED, r.test_id
        d = r.to_dict()
        assert d["statistic"] is None and d["metric"] is None and d["p_value"] is None
        assert d["notes"].startswith("not evaluated") and len(d["notes"]) > 20
    assert "fewer than 10" in results[7].notes and "above" in results[8].notes


def test_every_result_serialises_with_its_reference():
    rng = np.random.default_rng(0)
    results = [
        tail_backtest(ONE_YEAR, 0.003, 0.06),
        mc_convergence(0.1015, 0.001, 0.1),
        euler_additivity(np.array([0.5, 0.5]), 1.0),
        crp_recursion_vs_mc(np.array([0.5, 0.3, 0.2]), np.repeat([0, 1, 2], [5, 3, 2])),
        correlation_size_power(rng.random(500) < 0.05, 0.1, 0.1),
        correlation_size_power(rng.random(500) < 0.9, 0.2, 0.1),
        mc_convergence(0.1, 0.0, 0.1),
    ]
    for r in results:
        d = r.to_dict()
        assert set(d) == ROW_FIELDS
        assert d["reference"].strip() and d["h0"].strip() and d["policy_version"].strip() and d["alternative"].strip()
        assert d["test_id"].startswith("portfolio.") and d["light"] in {"green", "amber", "red", "not_evaluated"}
        json.dumps(d, allow_nan=False)
    refs = " ".join(r.reference for r in results)
    assert "dossier 05 G.5" in refs
    # the module states every test it owns
    assert {n for n in dir(val) if not n.startswith("_") and callable(getattr(val, n))} >= {
        "tail_backtest", "mc_convergence", "euler_additivity", "crp_recursion_vs_mc", "correlation_size_power",
    }


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        tail_backtest(ONE_YEAR, 0.0, 0.06)
    with pytest.raises(ValueError):
        tail_backtest(ONE_YEAR, 0.003, 1.0)
    with pytest.raises(ValueError):
        tail_backtest(ONE_YEAR, 0.003, 0.06, levels=(0.95, 0.99), primary_level=0.999)
    with pytest.raises(ValueError):
        tail_backtest(ONE_YEAR, 0.003, 0.06, levels=(0.95, 1.0))
    with pytest.raises(ValueError):
        tail_backtest(ONE_YEAR, 0.003, 0.06, alpha_amber=0.01, alpha_red=0.05)
    with pytest.raises(ValueError):
        tail_backtest(np.array([0.01, np.inf]), 0.003, 0.06)
    with pytest.raises(ValueError):
        tail_backtest(np.array([0.01, 1.5]), 0.003, 0.06)
    with pytest.raises(ValueError):
        mc_convergence(0.1, 0.001, 0.1, quantity="mean")
    with pytest.raises(ValueError):
        mc_convergence(0.1, 0.001, 0.1, alpha_amber=0.001, alpha_red=0.01)
    with pytest.raises(ValueError):
        euler_additivity(np.array([0.5, 0.5]), 1.0, tolerance=0.0)
    with pytest.raises(ValueError):
        euler_additivity(np.array([0.5, np.nan]), 1.0)
    with pytest.raises(ValueError):
        euler_additivity(np.array([0.5, 0.5]), float("nan"))
    with pytest.raises(ValueError):
        euler_additivity(np.array([]), 1.0)
    with pytest.raises(ValueError):
        crp_recursion_vs_mc(np.array([0.5, np.nan, 0.2]), np.zeros(20, dtype=int))
    with pytest.raises(ValueError):
        crp_recursion_vs_mc(np.array([0.5, 0.3, 0.2]), np.full(20, -1))
    with pytest.raises(ValueError):
        crp_recursion_vs_mc(np.array([0.5, 0.3, 0.2]), np.full(20, 0.5))
    with pytest.raises(ValueError):
        correlation_size_power(np.array([1.0, 0.5, 0.0]), 0.1, 0.1)
    with pytest.raises(ValueError):
        correlation_size_power(np.zeros(10, dtype=bool), 0.0, 0.1)
    with pytest.raises(ValueError):
        correlation_size_power(np.zeros(10, dtype=bool), 0.1, 1.0)
    with pytest.raises(ValueError):
        correlation_size_power(np.zeros(10, dtype=bool), 0.1, 0.1, alpha=0.0)
    with pytest.raises(ValueError):
        correlation_size_power(np.zeros(10, dtype=bool), 0.1, 0.1, power_green=0.4, power_amber=0.5)
