"""RV-976 to RV-982, RV-984 and RV-986: the CCR validation tests of `riskvalidation.validation.ccr` (contract
`engine-ccr-api`, module validation.ccr). Each returns one `TestResult`; the rows serialise into the fields of
`common/test-result`. The size and power of the PIT backtest are measured under the model itself."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from scipy.stats import chi2, norm

from riskvalidation.core.result import Light
from riskvalidation.validation import ccr as v

ROW_FIELDS = {"test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
              "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras"}


def arr(*x):
    return np.array(x, dtype=float)


# ---------------------------------------------------------------------------------------------------- RV-976


def test_ee_analytic_vs_simulated():
    sim = arr(1.0, 2.0, 3.0, 2.0)
    se = arr(0.1, 0.1, 0.2, 0.1)
    r = v.ee_analytic_vs_simulated(sim.copy(), sim, se)
    assert r.test_id == "ccr.ee_analytic_vs_simulated" and r.light is Light.GREEN
    assert r.statistic == 0.0 and r.metric == 0.0
    assert r.extras["thresholds"] == {"k_se": 3.0}
    off = sim.copy()
    off[2] += 4.0 * se[2]
    r = v.ee_analytic_vs_simulated(off, sim, se)
    assert r.light is Light.RED and r.statistic == pytest.approx(4.0, abs=1e-12)
    assert r.metric == pytest.approx(0.8, abs=1e-12)
    inside = sim.copy()
    inside[1] += 2.99 * se[1]
    assert v.ee_analytic_vs_simulated(inside, sim, se).light is Light.GREEN
    assert v.ee_analytic_vs_simulated(off, sim, se, k_se=5.0).light is Light.GREEN
    assert v.ee_analytic_vs_simulated(off, sim, se, k_se=5.0).extras["thresholds"] == {"k_se": 5.0}
    # a point with no Monte Carlo error (EE zero after the maturity): equal values pass, a difference does not
    z = arr(1.0, 0.0)
    assert v.ee_analytic_vs_simulated(z, z, arr(0.1, 0.0)).light is Light.GREEN
    bad = v.ee_analytic_vs_simulated(arr(1.0, 0.5), z, arr(0.1, 0.0))
    assert bad.light is Light.RED and bad.notes
    assert v.ee_analytic_vs_simulated(sim, sim, se).n == 4


def test_martingale_discounted_bonds():
    p0 = arr(0.99, 0.97, 0.94)
    se = arr(1e-4, 2e-4, 3e-4)
    r = v.martingale_discounted_bonds(p0.copy(), p0, se)
    assert r.test_id == "ccr.martingale_discounted_bonds" and r.light is Light.GREEN and r.statistic == 0.0
    off = p0.copy()
    off[1] -= 4.0 * se[1]
    r = v.martingale_discounted_bonds(off, p0, se)
    assert r.light is Light.RED and r.statistic == pytest.approx(4.0, abs=1e-9)
    assert r.extras["thresholds"] == {"k_se": 3.0}
    assert v.martingale_discounted_bonds(p0 + 2.0 * se, p0, se).light is Light.GREEN


# ---------------------------------------------------------------------------------------------------- RV-977


def test_eepe_and_cva_recomputed():
    grid, ee = arr(0.25, 0.5, 1.0), arr(1.0, 3.0, 2.0)
    r = v.eepe_recomputed(grid, ee, 2.5)
    assert r.test_id == "ccr.eepe_recomputed" and r.light is Light.GREEN
    assert r.statistic == pytest.approx(0.0, abs=1e-12) and r.metric == pytest.approx(0.0, abs=1e-12)
    r = v.eepe_recomputed(grid, ee, 2.0)                               # the plain EPE, not the effective one
    assert r.light is Light.RED and r.statistic == pytest.approx(0.5, abs=1e-12)
    assert r.metric == pytest.approx(0.2, abs=1e-12)
    assert v.eepe_recomputed(grid, ee, 2.0, rel_tol=0.25).light is Light.GREEN
    assert v.eepe_recomputed(grid, ee, 2.5 + 1e-12).light is Light.GREEN
    r = v.cva_recomputed(arr(10.0, 20.0), arr(0.1, 0.1), 0.6, 1.8)
    assert r.test_id == "ccr.cva_recomputed" and r.light is Light.GREEN and r.statistic == pytest.approx(0.0, abs=1e-12)
    r = v.cva_recomputed(arr(10.0, 20.0), arr(0.1, 0.1), 0.6, 1.9)
    assert r.light is Light.RED and r.statistic == pytest.approx(0.1, abs=1e-12)
    assert r.metric == pytest.approx(0.1 / 1.8, abs=1e-12)
    # a zero value reported as zero is exact
    assert v.cva_recomputed(arr(10.0, 20.0), np.zeros(2), 0.6, 0.0).light is Light.GREEN


# ---------------------------------------------------------------------------------------------------- RV-978


def test_path_convergence():
    ee = arr(0.0, 1.0, 2.0, 1.5, 0.5)
    se1, se4 = arr(0.0, 0.02, 0.04, 0.03, 0.01), arr(0.0, 0.01, 0.02, 0.015, 0.005)
    ses = {"ee_se_by_path_count": {1000: se1, 4000: se4}, "cva_se_by_path_count": {1000: 0.05, 4000: 0.025}}
    r = v.path_convergence({1000: ee.copy(), 4000: ee.copy()}, {1000: 1.8, 4000: 1.8}, **ses)
    assert r.test_id == "ccr.path_convergence" and r.light is Light.GREEN and r.statistic == 0.0 and r.metric == 0.0
    assert r.extras["thresholds"]["k_se"] == 3.0
    # the band is k_se times the standard error of the smaller count: statistic = the largest |change| / se (EE),
    # metric = |CVA change| / se (CVA); green within k_se, amber within 2 k_se, red beyond
    for z, light in ((2.9, Light.GREEN), (4.0, Light.AMBER), (6.5, Light.RED)):
        big = ee.copy()
        small = ee.copy()
        small[2] = ee[2] - z * se1[2]
        r = v.path_convergence({1000: small, 4000: big}, {1000: 1.8, 4000: 1.8}, **ses)
        assert r.light is light, (z, r.statistic)
        assert r.statistic == pytest.approx(z, abs=1e-9) and r.metric == 0.0
        r = v.path_convergence({1000: ee, 4000: ee}, {1000: 1.8 - z * 0.05, 4000: 1.8}, **ses)
        assert r.light is light and r.metric == pytest.approx(z, abs=1e-9) and r.statistic == 0.0
    # the two largest counts are compared, and a point with no standard error needs a zero change
    r = v.path_convergence({500: ee * 0.1, 1000: ee * 0.99, 4000: ee}, {500: 0.1, 1000: 1.8, 4000: 1.8},
                           ee_se_by_path_count={500: se1, 1000: se1, 4000: se4},
                           cva_se_by_path_count={500: 0.1, 1000: 0.05, 4000: 0.025})
    assert r.statistic == pytest.approx(0.01 * 2.0 / 0.04, abs=1e-12) and r.light is Light.GREEN
    r = v.path_convergence({1000: ee + arr(0.5, 0, 0, 0, 0), 4000: ee}, {1000: 1.8, 4000: 1.8}, **ses)
    assert r.light is Light.RED and math.isinf(r.statistic)
    # fewer than two path counts: not evaluated, with its reason and no number
    one = v.path_convergence({1000: ee}, {1000: 1.0}, ee_se_by_path_count={1000: se1}, cva_se_by_path_count={1000: 0.1})
    assert one.light is Light.NOT_EVALUATED and "path count" in one.notes
    row = one.to_dict()
    assert row["statistic"] is None and row["p_value"] is None and row["metric"] is None
    with pytest.raises(ValueError, match="cva_by_path_count"):
        v.path_convergence({1000: ee, 4000: ee}, {1000: 1.0, 2000: 1.0}, **ses)
    with pytest.raises(ValueError, match="ee_by_path_count"):
        v.path_convergence({1000: ee, 4000: ee[:3]}, {1000: 1.0, 4000: 1.0}, **ses)
    with pytest.raises(ValueError, match="ee_se_by_path_count"):
        v.path_convergence({1000: ee, 4000: ee}, {1000: 1.8, 4000: 1.8}, ee_se_by_path_count={1000: se1},
                           cva_se_by_path_count=ses["cva_se_by_path_count"])
    with pytest.raises(ValueError, match="cva_se_by_path_count"):
        v.path_convergence({1000: ee, 4000: ee}, {1000: 1.8, 4000: 1.8}, ee_se_by_path_count=ses["ee_se_by_path_count"],
                           cva_se_by_path_count={1000: 0.05})
    with pytest.raises(ValueError, match="ee_se_by_path_count"):
        v.path_convergence({1000: ee, 4000: ee}, {1000: 1.8, 4000: 1.8},
                           ee_se_by_path_count={1000: se1[:3], 4000: se4},
                           cva_se_by_path_count=ses["cva_se_by_path_count"])
    with pytest.raises(TypeError):                                      # the standard errors are required, by keyword
        v.path_convergence({1000: ee, 4000: ee}, {1000: 1.8, 4000: 1.8})


def test_path_convergence_same_model_green_and_planted_shift_red():
    """H10: two path counts drawn from the same model are green with the Monte Carlo standard errors (an at-the-money
    swap has a coefficient of variation above 1, which the removed unit-CV band could not tolerate); a planted shift
    beyond the band is red."""
    rng = np.random.default_rng(11)
    grid_n = 6

    def draw(n):
        w = rng.normal(0.0, 1.0, (n, grid_n)).cumsum(axis=1)             # at-the-money, mean zero
        pay = np.maximum(w, 0.0)
        disc = pay * 0.8
        return (pay.mean(axis=0), pay.std(axis=0, ddof=1) / math.sqrt(n), float(disc.mean()),
                float(disc.mean(axis=1).std(ddof=1)) / math.sqrt(n))

    lights = []
    for _ in range(20):
        (e1, s1, c1, cs1), (e2, s2, c2, cs2) = draw(2000), draw(8000)
        lights.append(v.path_convergence({2000: e1, 8000: e2}, {2000: c1, 8000: c2},
                                         ee_se_by_path_count={2000: s1, 8000: s2},
                                         cva_se_by_path_count={2000: cs1, 8000: cs2}).light)
    assert sum(light is Light.GREEN for light in lights) >= 17, lights     # a 3-sigma band over 7 contrasts
    (e1, s1, c1, cs1), (e2, s2, c2, cs2) = draw(2000), draw(8000)
    r = v.path_convergence({2000: e1 + 8.0 * s1, 8000: e2}, {2000: c1, 8000: c2},
                           ee_se_by_path_count={2000: s1, 8000: s2}, cva_se_by_path_count={2000: cs1, 8000: cs2})
    assert r.light is Light.RED


# ---------------------------------------------------------------------------------------------------- RV-979


def test_reconcile_ee_profile():
    ref = arr(1.0, 2.0, 3.0)
    se = arr(0.1, 0.1, 0.2)
    rse = arr(0.05, 0.1, 0.15)
    r = v.reconcile_ee_profile(ref.copy(), se, ref, rse)
    assert r.test_id == "ccr.reconcile_ee_profile" and r.light is Light.GREEN and r.statistic == 0.0
    comb = np.sqrt(se**2 + rse**2)
    ee = ref.copy()
    ee[1] += 3.0 * comb[1]                                             # 3 combined standard errors
    r = v.reconcile_ee_profile(ee, se, ref, rse)
    assert r.light is Light.RED and r.statistic == pytest.approx(3.0, abs=1e-12)
    assert v.reconcile_ee_profile(ee, se, ref, rse, k_se=4.0).light is Light.GREEN
    assert r.extras["thresholds"] == {"k_se": 2.0}
    agree = r.extras["agreement"]
    assert len(agree) == 3
    assert [round(a["difference"], 12) for a in agree] == [0.0, round(3.0 * comb[1], 12), 0.0]
    assert [round(a["combined_se"], 12) for a in agree] == [round(float(c), 12) for c in comb]
    within = ref.copy()
    within[0] += 1.9 * comb[0]
    assert v.reconcile_ee_profile(within, se, ref, rse).light is Light.GREEN
    # no error on either side at a point: equal passes, a difference does not
    z = arr(0.0, 1.0)
    assert v.reconcile_ee_profile(z, arr(0.0, 0.1), z, arr(0.0, 0.1)).light is Light.GREEN
    assert v.reconcile_ee_profile(arr(0.5, 1.0), arr(0.0, 0.1), z, arr(0.0, 0.1)).light is Light.RED


# ---------------------------------------------------------------------------------------------------- RV-980


def _kupiec(n, x, q):
    p = 1.0 - q
    ll0 = (n - x) * math.log(1.0 - p) + x * math.log(p)
    ll1 = (n - x) * math.log(1.0 - x / n) if x < n else 0.0
    ll1 += x * math.log(x / n) if x > 0 else 0.0
    return max(-2.0 * (ll0 - ll1), 0.0)


def test_pfe_exceedance_backtest():
    n, q = 100, 0.95
    forecast = np.full(n, 10.0)
    realised = np.full(n, 5.0)
    realised[:5] = 11.0                                                # 5 exceedances = the expected number
    r = v.pfe_exceedance_backtest(realised, forecast, q)
    assert r.test_id == "ccr.pfe_exceedance_backtest" and r.light is Light.GREEN
    assert r.n_events == 5 and r.n == 100 and r.statistic == pytest.approx(0.0, abs=1e-9)
    assert r.p_value == pytest.approx(1.0, abs=1e-9) and r.metric == pytest.approx(0.05, abs=1e-15)
    none = v.pfe_exceedance_backtest(np.full(n, 5.0), forecast, q)
    assert none.n_events == 0 and none.light is Light.RED
    assert none.statistic == pytest.approx(-2.0 * n * math.log(q), abs=1e-9)
    assert none.statistic == pytest.approx(10.2587, abs=5e-5)
    assert none.p_value == pytest.approx(float(chi2.sf(none.statistic, 1)), abs=1e-15)
    assert none.alpha_amber == 0.05 and none.alpha_red == 0.01
    assert "BCBS 185" in none.extras["caveat"] and "paragraph 27" in none.extras["caveat"]
    # an exceedance is strictly above the forecast
    assert v.pfe_exceedance_backtest(np.full(n, 10.0), forecast, q).n_events == 0
    # every light, against a likelihood ratio computed here
    n2, q2 = 250, 0.99
    seen = set()
    for x in range(0, 40):
        rl = np.zeros(n2)
        rl[:x] = 1.0
        res = v.pfe_exceedance_backtest(rl, np.full(n2, 0.5), q2)
        lr = _kupiec(n2, x, q2)
        assert res.statistic == pytest.approx(lr, abs=1e-9) and res.n_events == x
        p = float(chi2.sf(lr, 1))
        want = Light.GREEN if p >= 0.05 else Light.AMBER if p >= 0.01 else Light.RED
        assert res.light is want, (x, p)
        seen.add(want)
    assert seen == {Light.GREEN, Light.AMBER, Light.RED}
    # all dates exceeded
    allx = v.pfe_exceedance_backtest(np.ones(20), np.zeros(20), 0.95)
    assert allx.n_events == 20 and allx.light is Light.RED
    assert allx.statistic == pytest.approx(_kupiec(20, 20, 0.95), abs=1e-9)
    with pytest.raises(ValueError, match="pfe_forecast"):
        v.pfe_exceedance_backtest(realised, forecast[:5], q)
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="quantile"):
            v.pfe_exceedance_backtest(realised, forecast, bad)


# ---------------------------------------------------------------------------------------------------- RV-981


def _w2(pits):
    x = sorted(pits)
    n = len(x)
    s = sum((2 * k - 1) / n * (math.log(x[k - 1]) + math.log(1.0 - x[n - k])) for k in range(1, n + 1))
    return -n - s


def _backtest_from_pits(pit_indices, p_sim):
    """A realised vector and a simulated sample (rows 0..p_sim-1) whose PITs are (i + 1) / (p_sim + 2)."""
    simulated = np.tile(np.arange(float(p_sim)), (len(pit_indices), 1))
    realised = np.array(pit_indices, dtype=float) - 0.5
    return realised, simulated


def test_pit_backtest_golden_and_light():
    # the mapping (i + 1) / (N + 2) with the two tails; ties count only the values strictly below
    sim = np.array([[4.0, 1.0, 3.0, 2.0]] * 3)
    pits = v.pit_from_simulated(arr(0.5, 2.5, 5.0), sim)
    assert np.allclose(pits, arr(1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0), atol=1e-15, rtol=0)
    assert v.pit_from_simulated(arr(2.0), sim[:1])[0] == pytest.approx(2.0 / 6.0, abs=1e-15)
    # the Anderson-Darling statistic: PITs 0.25, 0.5, 0.75 give 0.269431
    assert v.anderson_darling_w2(arr(0.25, 0.5, 0.75)) == pytest.approx(_w2([0.25, 0.5, 0.75]), abs=1e-14)
    assert v.anderson_darling_w2(arr(0.25, 0.5, 0.75)) == pytest.approx(0.269431, abs=5e-7)
    assert v.anderson_darling_w2(arr(0.75, 0.25, 0.5)) == v.anderson_darling_w2(arr(0.25, 0.5, 0.75))     # order-free
    uniform = (np.arange(50) + 1.0) / 52.0                              # uniform PITs: a small statistic
    assert v.anderson_darling_w2(uniform) < 0.5
    assert v.anderson_darling_w2(arr(0.01, 0.02, 0.03, 0.04)) > 6.0     # all in the lower tail: far above 0.1 percent
    for bad in (arr(0.0, 0.5), arr(0.5, 1.0), arr(-0.1, 0.5), arr(0.5, 1.5)):
        with pytest.raises(ValueError, match="pits"):
            v.anderson_darling_w2(bad)
    assert v.IID_CRITICAL_VALUES == {0.25: 1.248, 0.15: 1.610, 0.10: 1.933, 0.05: 2.492, 0.01: 3.880, 0.005: 4.500,
                                     0.001: 6.000}
    # not evaluated without simulated critical values: no number anywhere
    realised, simulated = _backtest_from_pits([10, 20, 30, 40, 50, 60, 70, 80], 99)
    ne = v.pit_exposure_backtest(realised, simulated, None)
    assert ne.test_id == "ccr.pit_exposure_backtest" and ne.light is Light.NOT_EVALUATED
    assert "no simulated critical values" in ne.notes
    row = ne.to_dict()
    assert row["statistic"] is None and row["p_value"] is None and row["metric"] is None
    # the light against the simulated critical values
    idx = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    realised, simulated = _backtest_from_pits(idx, 99)
    pit_values = (np.array(idx) + 1.0) / 101.0
    w2 = _w2(list(pit_values))
    assert v.pit_from_simulated(realised, simulated) == pytest.approx(pit_values, abs=1e-15)
    cases = (({"p05": 1.5 * w2, "p01": 2.0 * w2}, Light.GREEN), ({"p05": 0.8 * w2, "p01": 1.2 * w2}, Light.AMBER),
             ({"p05": 0.5 * w2, "p01": 0.8 * w2}, Light.RED))
    for crit, light in cases:
        r = v.pit_exposure_backtest(realised, simulated, crit, effective_sample_size=7.5)
        assert r.light is light
        assert r.statistic == pytest.approx(w2, abs=1e-12) and r.p_value is None and r.n == 10
        assert np.allclose(r.extras["pits"], pit_values, atol=1e-15, rtol=0)
        assert r.extras["simulated_critical_values"] == crit
        assert r.extras["iid_critical_values"] == {str(k): x for k, x in v.IID_CRITICAL_VALUES.items()}
        assert r.extras["effective_sample_size"] == 7.5
        assert r.extras["thresholds"]["simulated_p05"] == crit["p05"]
    # the effective sample size is computed from the PITs when not given
    r = v.pit_exposure_backtest(realised, simulated, {"p05": 3.0, "p01": 4.0})
    assert r.extras["effective_sample_size"] == pytest.approx(v.effective_sample_size(pit_values), abs=1e-12)
    for crit in ({"p05": 3.0}, {"p01": 4.0}, {"p05": 4.0, "p01": 3.0}, {"p05": -1.0, "p01": 3.0}):
        with pytest.raises(ValueError, match="critical_simulated"):
            v.pit_exposure_backtest(realised, simulated, crit)
    with pytest.raises(ValueError, match="simulated"):
        v.pit_exposure_backtest(realised, simulated[:5], {"p05": 3.0, "p01": 4.0})


def _model_backtest(rng, n_dates=60, p_sim=199, scale=1.0):
    """PITs of one backtest whose realised values come from N(0, scale^2) while the model forecasts N(0, 1)."""
    sim = rng.standard_normal((n_dates, p_sim))
    realised = scale * rng.standard_normal(n_dates)
    return v.pit_from_simulated(realised, sim)


def test_pit_size_and_power():
    crit = v.simulate_pit_critical_values(lambda rng: _model_backtest(rng), 2000, seed=1)
    rng = np.random.default_rng(2)
    n_rep = 600
    w2 = np.array([v.anderson_darling_w2(_model_backtest(rng)) for _ in range(n_rep)])
    rate05 = float(np.mean(w2 > crit.p05))
    rate01 = float(np.mean(w2 > crit.p01))
    assert abs(rate05 - 0.05) <= 3.0 * math.sqrt(0.05 * 0.95 / n_rep)           # the size is the nominal level
    assert rate01 <= 0.01 + 3.0 * math.sqrt(0.01 * 0.99 / n_rep)
    # power: the realised path has half the forecast sigma
    rng = np.random.default_rng(3)
    narrow = np.array([v.anderson_darling_w2(_model_backtest(rng, scale=0.5)) for _ in range(200)])
    assert float(np.mean(narrow > crit.p01)) >= 0.9
    # and through the test: green under the model, red for the halved sigma
    sim = np.random.default_rng(4).standard_normal((60, 199))
    for scale, light in ((1.0, Light.GREEN), (0.5, Light.RED)):
        realised = scale * np.random.default_rng(5).standard_normal(60)
        r = v.pit_exposure_backtest(realised, sim, {"p05": crit.p05, "p01": crit.p01})
        assert r.light is light, (scale, r.statistic, crit.p05, crit.p01)


# ---------------------------------------------------------------------------------------------------- RV-982


def test_effective_sample_size_and_critical_values():
    z = arr(1.0, 0.0, -1.0, 0.0)
    assert v.effective_sample_size(norm.cdf(z), max_lag=1) == pytest.approx(4.0, abs=1e-9)
    rng = np.random.default_rng(1)
    n = 1000
    iid = norm.cdf(rng.standard_normal(n))
    assert 0.7 * n <= v.effective_sample_size(iid) <= n
    # fully overlapping horizons give fewer; recomputed through a plain-Python route
    zz = np.zeros(n)
    eps = rng.standard_normal(n)
    for i in range(1, n):
        zz[i] = 0.9 * zz[i - 1] + math.sqrt(1.0 - 0.81) * eps[i]
    pits = norm.cdf(zz)
    ess = v.effective_sample_size(pits)
    assert 1.0 <= ess < 0.3 * n
    lag = int(n ** (1.0 / 3.0))
    zbar = float(np.mean(zz))
    denom = sum((x - zbar) ** 2 for x in zz)
    rho = [sum((zz[t] - zbar) * (zz[t + k] - zbar) for t in range(n - k)) / denom for k in range(1, lag + 1)]
    want = n / (1.0 + 2.0 * sum((1.0 - k / (lag + 1.0)) * rho[k - 1] for k in range(1, lag + 1)))
    assert ess == pytest.approx(min(max(want, 1.0), float(n)), abs=1e-8)
    assert v.effective_sample_size(pits, max_lag=5) != ess
    ramp = norm.cdf(np.linspace(-2.0, 2.0, 40))                        # near-perfect dependence: clipped into [1, n]
    assert 1.0 <= v.effective_sample_size(ramp) <= 40.0
    for bad, field in ((np.full(10, 0.5), "pits"), (arr(0.5), "pits"), (arr(0.0, 0.5, 0.7), "pits")):
        with pytest.raises(ValueError, match=field):
            v.effective_sample_size(bad)
    for lag_bad in (0, 10):
        with pytest.raises(ValueError, match="max_lag"):
            v.effective_sample_size(iid[:10], max_lag=lag_bad)
    # the critical values of independent uniforms are the iid values of Formenti et al. Table 1
    crit = v.simulate_pit_critical_values(lambda g: g.uniform(0.0, 1.0, 100), 4000, seed=11)
    assert abs(crit.p05 - 2.492) <= 0.15 and abs(crit.p01 - 3.880) <= 0.35
    assert crit.n_backtests == 4000 and crit.w2_samples.shape == (4000,)
    assert crit.p05 == float(np.quantile(crit.w2_samples, 0.95, method="inverted_cdf"))
    assert crit.p01 == float(np.quantile(crit.w2_samples, 0.99, method="inverted_cdf"))
    assert crit.p01 >= crit.p05 > 0.0 and 1.0 <= crit.effective_sample_size <= 100.0
    again = v.simulate_pit_critical_values(lambda g: g.uniform(0.0, 1.0, 100), 4000, seed=11)
    assert np.array_equal(again.w2_samples, crit.w2_samples) and again.p05 == crit.p05
    other = v.simulate_pit_critical_values(lambda g: g.uniform(0.0, 1.0, 100), 4000, seed=12)
    assert not np.array_equal(other.w2_samples, crit.w2_samples)
    # the effective sample size is the one of the first simulated series
    first = np.random.default_rng(11).uniform(0.0, 1.0, 100)
    assert crit.effective_sample_size == pytest.approx(v.effective_sample_size(first), abs=1e-12)
    # the contract refuses fewer than 100 backtests and PITs outside (0, 1)
    with pytest.raises(ValueError, match="n_backtests"):
        v.simulate_pit_critical_values(lambda g: g.uniform(0.0, 1.0, 100), 99, seed=1)
    for bad_value in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="pit"):
            v.simulate_pit_critical_values(lambda g, b=bad_value: np.append(g.uniform(0.1, 0.9, 9), b), 100, seed=1)
    with pytest.raises(ValueError, match="pit"):
        v.simulate_pit_critical_values(lambda g: g.uniform(0.1, 0.9, (3, 3)), 100, seed=1)
    # frozen, and its invariants are enforced
    with pytest.raises(dataclasses.FrozenInstanceError):
        crit.p05 = 1.0                                                  # type: ignore[misc]
    with pytest.raises(ValueError, match="p01"):
        v.CriticalValues(p05=3.0, p01=2.0, effective_sample_size=5.0, n_backtests=100, w2_samples=np.ones(100))
    with pytest.raises(ValueError, match="p05"):
        v.CriticalValues(p05=0.0, p01=2.0, effective_sample_size=5.0, n_backtests=100, w2_samples=np.ones(100))
    with pytest.raises(ValueError, match="effective_sample_size"):
        v.CriticalValues(p05=2.0, p01=3.0, effective_sample_size=0.5, n_backtests=100, w2_samples=np.ones(100))


# ---------------------------------------------------------------------------------------------------- RV-984


def test_refusals_name_the_field():
    sim = arr(1.0, 2.0)
    se = arr(0.1, 0.1)
    with pytest.raises(ValueError, match="k_se"):
        v.ee_analytic_vs_simulated(sim, sim, se, k_se=0.0)
    with pytest.raises(ValueError, match="mc_se"):
        v.ee_analytic_vs_simulated(sim, sim, arr(0.1, -0.1))
    with pytest.raises(ValueError, match="ee_simulated"):
        v.ee_analytic_vs_simulated(sim, arr(1.0, 2.0, 3.0), se)
    with pytest.raises(ValueError, match="mc_se"):
        v.martingale_discounted_bonds(sim, sim, arr(0.1, np.nan))
    with pytest.raises(ValueError, match="rel_tol"):
        v.eepe_recomputed(arr(0.5, 1.0), sim, 1.0, rel_tol=-1.0)
    with pytest.raises(ValueError, match="grid"):
        v.eepe_recomputed(arr(1.0, 0.5), sim, 1.0)
    with pytest.raises(ValueError, match="pd"):
        v.cva_recomputed(sim, arr(0.1), 0.6, 1.0)
    with pytest.raises(ValueError, match="reference_se"):
        v.reconcile_ee_profile(sim, se, sim, arr(0.1))
    with pytest.raises(ValueError, match="k_se"):
        v.reconcile_ee_profile(sim, se, sim, se, k_se=-1.0)
    with pytest.raises(ValueError, match="realised"):
        v.pit_exposure_backtest(arr(1.0), np.ones((1, 5)), {"p05": 2.0, "p01": 3.0})
    with pytest.raises(ValueError, match="effective_sample_size"):
        v.pit_exposure_backtest(arr(1.0, 2.0), np.ones((2, 5)), {"p05": 2.0, "p01": 3.0}, effective_sample_size=0.0)


# ---------------------------------------------------------------------------------------------------- RV-986


def _numbers(x):
    if isinstance(x, bool):
        return []
    if isinstance(x, (int, float)):
        return [x]
    if isinstance(x, dict):
        return [n for y in x.values() for n in _numbers(y)]
    if isinstance(x, (list, tuple)):
        return [n for y in x for n in _numbers(y)]
    return []


def test_rows_serialise_and_not_evaluated_carry_no_number():
    sim = arr(1.0, 2.0, 3.0)
    se = arr(0.1, 0.1, 0.1)
    realised, simulated = _backtest_from_pits([5, 15, 25, 35, 45, 55], 99)
    rows = [
        v.ee_analytic_vs_simulated(sim, sim, se),
        v.martingale_discounted_bonds(sim, sim, se),
        v.eepe_recomputed(arr(0.25, 0.5, 1.0), arr(1.0, 3.0, 2.0), 2.5),
        v.cva_recomputed(arr(10.0, 20.0), arr(0.1, 0.1), 0.6, 1.8),
        v.path_convergence({1000: sim, 4000: sim}, {1000: 1.0, 4000: 1.0}, ee_se_by_path_count={1000: se, 4000: se},
                           cva_se_by_path_count={1000: 0.1, 4000: 0.05}),
        v.path_convergence({1000: sim}, {1000: 1.0}, ee_se_by_path_count={1000: se}, cva_se_by_path_count={1000: 0.1}),
        v.reconcile_ee_profile(sim, se, sim, se),
        v.pfe_exceedance_backtest(np.full(100, 5.0), np.full(100, 10.0), 0.95),
        v.pit_exposure_backtest(realised, simulated, {"p05": 2.492, "p01": 3.880}),
        v.pit_exposure_backtest(realised, simulated, None),
    ]
    ids = set()
    for r in rows:
        d = r.to_dict()
        assert set(d) == ROW_FIELDS
        assert d["test_id"].startswith("ccr.") and d["policy_version"] and d["h0"] and d["reference"]
        json.dumps(d, allow_nan=False)                                  # serialises with no NaN or infinity
        ids.add(d["test_id"])
        if r.light is Light.NOT_EVALUATED:
            assert d["statistic"] is None and d["p_value"] is None and d["metric"] is None
            assert d["notes"] and _numbers(d["extras"]) == []           # no number stands in
    assert ids == {"ccr.ee_analytic_vs_simulated", "ccr.martingale_discounted_bonds", "ccr.eepe_recomputed",
                   "ccr.cva_recomputed", "ccr.path_convergence", "ccr.reconcile_ee_profile",
                   "ccr.pfe_exceedance_backtest", "ccr.pit_exposure_backtest"}
    assert sum(r.light is Light.NOT_EVALUATED for r in rows) == 2
