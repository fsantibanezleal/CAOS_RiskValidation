"""RV-961 to RV-967: the exposure measures of `riskvalidation.ccr` (contract `engine-ccr-api`, module exposure): EE and
ENE, the effective measures, PFE, netting, effective maturity, the analytic EE of a single swap (ladder rung 2) and rung
1."""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.stats import norm

from riskvalidation.ccr import exposure as ex
from riskvalidation.ccr import hw1f, swaps
from riskvalidation.ccr.swaps import Swap


def arr(*x):
    return np.array(x, dtype=float)


# ---------------------------------------------------------------------------------------------------- RV-961


def test_expected_exposure():
    values = np.array([[-1.0, 2.0], [3.0, -4.0]])
    ee, se = ex.expected_exposure(values)
    assert np.array_equal(ee, arr(1.5, 1.0))
    ene, _ = ex.expected_exposure(-values)
    assert np.array_equal(ene, arr(0.5, 2.0))
    # the standard error is std / sqrt(P) of the positive parts
    assert np.allclose(se, np.std(np.maximum(values, 0.0), axis=0, ddof=1) / math.sqrt(2), atol=1e-15, rtol=0)
    rng = np.random.default_rng(1)
    v = rng.normal(0.3, 2.0, (5000, 7))
    ee, se = ex.expected_exposure(v)
    ene, _ = ex.expected_exposure(-v)
    assert np.all(ee >= 0.0) and np.all(se > 0.0)
    assert np.allclose(ee - ene, v.mean(axis=0), atol=1e-9, rtol=0)
    # the discounted EE is at most the undiscounted one for positive rates (numeraire >= 1)
    grid = np.linspace(0.1, 5.0, 7)
    numeraire = np.exp(0.03 * grid)[None, :] * np.ones((5000, 1))
    ee_d, se_d = ex.expected_exposure(v, numeraire)
    assert np.all(ee_d <= ee) and np.allclose(ee_d, np.mean(np.maximum(v, 0.0) / numeraire, axis=0), atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="numeraire"):
        ex.expected_exposure(v, numeraire[:, :3])
    with pytest.raises(ValueError, match="numeraire"):
        ex.expected_exposure(v, np.zeros_like(numeraire))
    with pytest.raises(ValueError, match="values"):
        ex.expected_exposure(arr(1.0, 2.0))
    with pytest.raises(ValueError, match="values"):
        ex.expected_exposure(np.array([[1.0, 2.0]]))                   # one path has no standard error


# ---------------------------------------------------------------------------------------------------- RV-962


def test_effective_ee_epe():
    assert np.array_equal(ex.effective_ee(arr(1.0, 3.0, 2.0)), arr(1.0, 3.0, 3.0))
    assert np.array_equal(ex.effective_ee(arr(1.0, 3.0, 2.0), current_exposure=2.0), arr(2.0, 3.0, 3.0))
    rng = np.random.default_rng(2)
    e = rng.uniform(0.0, 5.0, 40)
    eff = ex.effective_ee(e, 0.5)
    assert np.all(np.diff(eff) >= 0.0) and np.all(eff >= e) and eff[0] >= 0.5
    # effective EPE: golden 2.5, constant profile returns its value, the covered time normalises
    grid = arr(0.25, 0.5, 1.0)
    assert ex.effective_epe(grid, arr(1.0, 3.0, 3.0)) == pytest.approx((1 * 0.25 + 3 * 0.25 + 3 * 0.5) / 1.0, abs=1e-15)
    assert ex.effective_epe(grid, arr(1.0, 3.0, 3.0)) == pytest.approx(2.5, abs=1e-15)
    assert ex.effective_epe(grid, arr(4.0, 4.0, 4.0)) == pytest.approx(4.0, abs=1e-15)
    assert ex.effective_epe(arr(0.25, 0.5), arr(1.0, 3.0)) == pytest.approx((0.25 + 0.75) / 0.5, abs=1e-15)
    # only the first year counts
    assert ex.effective_epe(arr(0.25, 0.5, 1.0, 2.0), arr(1.0, 3.0, 3.0, 9.0)) == pytest.approx(2.5, abs=1e-15)
    assert ex.effective_epe(grid, arr(1.0, 3.0, 3.0), horizon=0.5) == pytest.approx(2.0, abs=1e-15)
    # at most the maximum of the profile it came from
    ee = rng.uniform(0.0, 5.0, 12)
    g = np.linspace(1.0 / 12.0, 1.0, 12)
    assert ex.effective_epe(g, ex.effective_ee(ee)) <= ee.max() + 1e-15
    # plain EPE: golden 2.0, at most the effective one
    assert ex.epe(grid, arr(1.0, 3.0, 2.0)) == pytest.approx(0.25 + 0.75 + 1.0, abs=1e-15)
    assert ex.epe(grid, arr(1.0, 3.0, 2.0)) == pytest.approx(2.0, abs=1e-15)
    for seed in range(5):
        ee = np.random.default_rng(seed).uniform(0.0, 5.0, 12)
        assert ex.epe(g, ee) <= ex.effective_epe(g, ex.effective_ee(ee)) + 1e-15
    for bad in (arr(0.25, 0.25, 1.0), arr(0.5, 0.25, 1.0)):
        with pytest.raises(ValueError, match="grid"):
            ex.epe(bad, arr(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="horizon"):
        ex.epe(arr(1.5, 2.0), arr(1.0, 2.0))
    with pytest.raises(ValueError, match="eff_ee"):
        ex.effective_epe(grid, arr(1.0, 2.0))


# ---------------------------------------------------------------------------------------------------- RV-963


def test_pfe():
    values = np.arange(1.0, 101.0).reshape(100, 1)
    p, se = ex.pfe(values, 0.95)
    assert p[0] == 95.0 and se[0] > 0.0
    assert ex.pfe(values, 0.5)[0][0] == 50.0
    assert np.array_equal(ex.pfe(-values, 0.95)[0], arr(0.0))          # the positive part
    rng = np.random.default_rng(4)
    v = rng.normal(0.0, 1.0, (20_000, 3))
    qs = (0.5, 0.8, 0.9, 0.95, 0.99)
    levels = np.array([ex.pfe(v, q, n_bootstrap=20)[0] for q in qs])
    assert np.all(np.diff(levels, axis=0) >= 0.0) and np.all(levels >= 0.0)
    ee, _ = ex.expected_exposure(v)
    for q in (0.9, 0.95, 0.99):                                       # a symmetric value: PFE above EE
        assert np.all(ex.pfe(v, q, n_bootstrap=20)[0] >= ee)
    # the bootstrap standard error is the order-statistic one: sqrt(q (1 - q) / P) / density
    p95, se95 = ex.pfe(v, 0.95, seed=3)
    analytic = math.sqrt(0.95 * 0.05 / 20_000) / float(norm.pdf(norm.ppf(0.95)))
    assert np.all(np.abs(se95 / analytic - 1.0) < 0.3)
    # reproducible for the seed
    again = ex.pfe(v, 0.95, seed=3)
    assert np.array_equal(again[0], p95) and np.array_equal(again[1], se95)
    assert not np.array_equal(ex.pfe(v, 0.95, seed=4)[1], se95)
    for q in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="quantile"):
            ex.pfe(v, q)
    with pytest.raises(ValueError, match="n_bootstrap"):
        ex.pfe(v, 0.9, n_bootstrap=1)


# ---------------------------------------------------------------------------------------------------- RV-964


def test_netting():
    one_path = np.array([[[1.0]], [[-1.0]]])                          # (S, P, K) = (2, 1, 1)
    net = ex.net_values(one_path)
    assert net.shape == (1, 1) and net[0, 0] == 0.0
    assert ex.expected_exposure(np.array([[0.0], [0.0]]))[0][0] == 0.0
    single = sum(ex.expected_exposure(np.array([[v]] * 2))[0] for v in (1.0, -1.0))
    assert single[0] == 1.0                                           # the single-trade EE sum to 1, netted 0
    rng = np.random.default_rng(5)
    trades = rng.normal(0.0, 1.0, (4, 3000, 6))
    netted_ee, _ = ex.expected_exposure(ex.net_values(trades))
    each = np.stack([ex.expected_exposure(trades[s])[0] for s in range(4)])
    assert np.all(netted_ee <= each.sum(axis=0) + 1e-12)
    benefit = ex.netting_benefit(each, netted_ee)
    assert np.all(benefit >= -1e-12) and np.allclose(benefit, each.sum(axis=0) - netted_ee, atol=1e-15, rtol=0)
    assert np.array_equal(ex.netting_benefit(arr(1.0, 2.0) * np.ones((2, 2)), arr(1.5, 3.0)), arr(0.5, 1.0))
    # netting is on the sum, not on the sum of positive parts
    assert np.allclose(ex.net_values(trades), trades.sum(axis=0), atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="values_by_trade"):
        ex.net_values(np.ones((3, 4)))
    with pytest.raises(ValueError, match="ee_netted"):
        ex.netting_benefit(each, netted_ee[:3])


# ---------------------------------------------------------------------------------------------------- RV-965


def test_effective_maturity():
    grid = arr(0.25, 0.5, 1.0)
    ee = arr(1.0, 3.0, 2.0)
    df = np.exp(-0.03 * grid)
    assert ex.effective_maturity(grid, ee, ex.effective_ee(ee), df) == 1.0        # exactly 1
    g = arr(0.5, 1.0, 2.0, 3.0)
    ee = arr(1.0, 2.0, 3.0, 2.0)
    eff = ex.effective_ee(ee)
    d = np.exp(-0.03 * g)
    dt = np.diff(np.concatenate([[0.0], g]))
    s1 = float(np.sum((eff * dt * d)[g <= 1.0]))
    s2 = float(np.sum((ee * dt * d)[g > 1.0]))
    assert ex.effective_maturity(g, ee, eff, d) == pytest.approx((s1 + s2) / s1, abs=1e-14)
    assert ex.effective_maturity(g, ee, eff, d) >= 1.0
    # capped at 5 years, and the cap is a parameter
    big = arr(1.0, 2.0, 100.0, 100.0)
    assert ex.effective_maturity(g, big, ex.effective_ee(arr(1.0, 2.0, 3.0, 2.0)), d) == 5.0
    assert ex.effective_maturity(g, big, ex.effective_ee(arr(1.0, 2.0, 3.0, 2.0)), d, cap=2.0) == 2.0
    with pytest.raises(ValueError, match="eff_ee"):
        ex.effective_maturity(grid, np.zeros(3), np.zeros(3), df)
    with pytest.raises(ValueError, match="df"):
        ex.effective_maturity(grid, ee[:3], eff[:3], df[:2])
    with pytest.raises(ValueError, match="cap"):
        ex.effective_maturity(g, ee, eff, d, cap=0.5)


# ---------------------------------------------------------------------------------------------------- RV-966


def _swap(payer, rate, maturity=3.0, notional=1.0, start=0.0, freq=2):
    return Swap("x", notional, rate, payer, start, maturity, freq)


def test_analytic_ee_single_swap():
    p0, fwd = hw1f.curve_functions(np.array([0.030, 0.032, 0.035, 0.040, 0.042, 0.043]),
                                   np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0]))
    a, sigma = 0.1, 0.01
    par = swaps.par_fixed_rate(3.0, 2, p0)
    for payer in (True, False):
        s = _swap(payer, par + 0.004, notional=2.5)
        grid = swaps.reset_times(s)
        ee = ex.analytic_ee_single_swap(s, np.append(grid, 3.0), a, sigma, p0, fwd)
        assert ee.shape == (grid.size + 1,) and np.all(ee >= 0.0) and ee[-1] == 0.0
        # at t = 1: Swaption(0; 1) / P(0, 1) on the remaining swap (flows after 1), typed out by hand
        times = np.array([1.5, 2.0, 2.5, 3.0])
        amounts = 2.5 * (par + 0.004) * 0.5 * np.ones(4)
        amounts[-1] += 2.5
        want = hw1f.swaption_jamshidian(payer, 1.0, times, amounts, a, sigma, p0, fwd, nominal=2.5) / float(p0(1.0))
        assert ee[2] == pytest.approx(want, abs=1e-14)
        # t = 0: the intrinsic value of the swap
        v0 = 2.5 * (1.0 - float(p0(3.0))) - 2.5 * (par + 0.004) * 0.5 * float(sum(p0(np.arange(1, 7) * 0.5)))
        assert ee[0] == pytest.approx(max(v0 if payer else -v0, 0.0), abs=1e-10)
    with pytest.raises(ValueError, match="grid"):
        ex.analytic_ee_single_swap(_swap(True, par), arr(0.0, 0.25), a, sigma, p0, fwd)      # not a reset date
    with pytest.raises(ValueError, match="grid"):
        ex.analytic_ee_single_swap(_swap(True, par, start=1.0), arr(0.5), a, sigma, p0, fwd)  # before the start
    assert np.array_equal(ex.analytic_ee_single_swap(_swap(True, par), arr(3.0, 4.0), a, sigma, p0, fwd), arr(0.0, 0.0))
    # against the Monte Carlo of swap_values_on_grid at the reset dates, within the error (risk-neutral measure: the
    # discounted simulated EE equals P(0, t) times the analytic EE)
    n = 40_000
    s = _swap(True, par + 0.004, notional=1.0)
    grid = swaps.reset_times(s)
    sim = swaps.simulation_grid(grid, [s])
    r = hw1f.simulate_short_rate(sim, a, sigma, fwd, n, 99)
    v = swaps.swap_values_on_grid([s], sim, r, a, sigma, p0, fwd)
    ee_mc, se_mc = ex.expected_exposure(ex.net_values(v), hw1f.bank_account(sim, r))
    analytic = ex.analytic_ee_single_swap(s, sim, a, sigma, p0, fwd) * p0(sim)
    assert np.all(np.abs(ee_mc - analytic)[1:] <= 4.0 * se_mc[1:])
    assert analytic[1:-1].min() > 0.0


# ---------------------------------------------------------------------------------------------------- RV-967


def test_rung1_ead_delegates_to_sa_ccr(monkeypatch):
    calls = []

    @dataclass(frozen=True)
    class AddOnByClass:
        interest_rate: float = 0.0
        fx: float = 0.0
        credit: float = 0.0
        equity: float = 0.0
        commodity: float = 0.0

    @dataclass(frozen=True)
    class Result:
        ead: float

    def netting_set_ead(v, c, addon_by_class, margined=False, th=0.0, mta=0.0, nica=0.0, alpha=1.4):
        calls.append((v, c, addon_by_class, margined, th, mta, nica, alpha))
        return Result(123.456)

    stub = types.ModuleType("riskvalidation.regulatory.sa_ccr")
    stub.AddOnByClass, stub.netting_set_ead = AddOnByClass, netting_set_ead
    monkeypatch.setitem(sys.modules, "riskvalidation.regulatory.sa_ccr", stub)
    out = ex.rung1_ead(-120.0, 0.0, 1400.96, margined=True, th=10.0, mta=2.0, nica=3.0)
    assert out == 123.456
    v, c, addon, margined, th, mta, nica, alpha = calls[0]
    assert (v, c, margined, th, mta, nica, alpha) == (-120.0, 0.0, True, 10.0, 2.0, 3.0, 1.4)
    assert addon == AddOnByClass(interest_rate=1400.96)               # the aggregate carried whole, the rest zero
    with pytest.raises(ValueError, match="addon_aggregate"):
        ex.rung1_ead(0.0, 0.0, -1.0)


def test_rung1_ead_cre99_golden():
    """Netting set 5 of CRE99 (`engine-regulatory-api`): V - C = -120, aggregate add-on 1,400.96238, EAD 1,879. Runs
    once `riskvalidation.regulatory.sa_ccr` (unit U9-E1) is in the tree."""
    sa = pytest.importorskip("riskvalidation.regulatory.sa_ccr")
    got = ex.rung1_ead(-120.0, 0.0, 1400.962380)
    assert got == sa.netting_set_ead(-120.0, 0.0, sa.AddOnByClass(interest_rate=1400.962380)).ead
    assert abs(got - 1878.9) < 1.0
    assert got == pytest.approx(1879.212632, abs=1e-5)
