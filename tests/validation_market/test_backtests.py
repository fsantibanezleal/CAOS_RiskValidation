"""RV-818 to RV-821 and RV-825: the VaR back-tests as `TestResult`s at their golden values and light rules (contract
`engine-market-api`, module validation.market)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, stats

from riskvalidation.core.result import Light, TestResult
from riskvalidation.validation import market as m


def ex(t, positions):
    out = np.zeros(t, dtype=bool)
    out[list(positions)] = True
    return out


def test_kupiec_pof():
    r = m.kupiec_pof(ex(250, [10, 60, 110, 160, 210]), 0.01, model_id="m1")
    assert isinstance(r, TestResult) and r.test_id == "market.kupiec_pof"
    assert r.statistic == pytest.approx(1.956810, abs=5e-7)
    assert r.p_value == pytest.approx(0.161855, abs=5e-7)
    assert r.metric == pytest.approx(0.107812, abs=5e-7)
    assert r.light is Light.GREEN and r.segment == "alpha=0.01" and r.model_id == "m1"
    assert r.n == 250 and r.n_events == 5 and (r.alpha_amber, r.alpha_red) == (0.10, 0.05)
    assert r.extras["expected_events"] == pytest.approx(2.5)
    assert r.extras["p_value_exact_basis"] == "binomial-minlike"
    assert "power" in r.notes.lower()
    z = m.kupiec_pof(np.zeros(250, dtype=bool), 0.01)
    assert z.statistic == pytest.approx(5.025168, abs=5e-7) and z.p_value == pytest.approx(0.024982, abs=5e-7)
    assert z.light is Light.RED
    assert stats.chi2.ppf(0.95, 1) == pytest.approx(3.841, abs=5e-4)
    # the exact p-value is the total probability of the outcomes at most as likely as x
    pmf = stats.binom.pmf(np.arange(251), 250, 0.01)
    assert r.metric == pytest.approx(pmf[pmf <= pmf[5] * (1 + 1e-7)].sum(), abs=1e-12)
    # the full-count case x = T uses 0 ln 0 = 0 and stays finite
    allx = m.kupiec_pof(np.ones(50, dtype=bool), 0.05)
    assert np.isfinite(allx.statistic) and allx.light is Light.RED
    # lights: amber between the two alphas, by the asymptotic p-value
    amber = m.kupiec_pof(ex(250, [10, 60, 110, 160, 210]), 0.01, alpha_amber=0.2, alpha_red=0.05)
    assert amber.light is Light.AMBER and 0.05 <= amber.p_value < 0.2
    empty = m.kupiec_pof(np.zeros(0, dtype=bool), 0.01)
    assert empty.light is Light.NOT_EVALUATED and empty.p_value is None and empty.metric is None
    assert empty.to_dict()["statistic"] is None
    with pytest.raises(ValueError):
        m.kupiec_pof(ex(250, [1]), 0.0)
    with pytest.raises(ValueError):
        m.kupiec_pof(ex(250, [1]), 0.01, alpha_amber=0.01, alpha_red=0.05)


GOLDEN_SEQ = np.array([0] * 10 + [1, 1] + [0] * 8, dtype=bool)


def test_christoffersen_independence():
    r = m.christoffersen_independence(GOLDEN_SEQ, 0.01)
    assert r.test_id == "market.christoffersen_independence" and r.segment == "alpha=0.01"
    assert r.extras["transitions"] == [16, 1, 1, 1]
    assert r.statistic == pytest.approx(2.407835, abs=5e-7) and r.p_value == pytest.approx(0.120729, abs=5e-7)
    assert r.n == 20 and r.n_events == 2 and r.light is Light.GREEN
    for seq in (np.zeros(50, dtype=bool), np.ones(50, dtype=bool)):
        n = m.christoffersen_independence(seq, 0.05)
        assert n.light is Light.NOT_EVALUATED
        assert n.to_dict()["statistic"] is None and n.to_dict()["p_value"] is None
        assert "no exceedance" in n.notes or "all" in n.notes
    # clustering raises the statistic
    spread = ex(200, range(0, 200, 20))
    clustered = ex(200, range(0, 10))
    assert (m.christoffersen_independence(clustered, 0.05).statistic
            > m.christoffersen_independence(spread, 0.05).statistic)
    one = m.christoffersen_independence(ex(1, []), 0.05)
    assert one.light is Light.NOT_EVALUATED


def test_christoffersen_conditional_coverage():
    cc = m.christoffersen_conditional_coverage(GOLDEN_SEQ, 0.01)
    pof = m.kupiec_pof(GOLDEN_SEQ, 0.01)
    ind = m.christoffersen_independence(GOLDEN_SEQ, 0.01)
    assert pof.statistic == pytest.approx(5.779174, abs=5e-7)
    assert cc.statistic == pytest.approx(8.187009, abs=5e-7) and cc.p_value == pytest.approx(0.016681, abs=5e-7)
    assert cc.statistic == pytest.approx(pof.statistic + ind.statistic, abs=1e-12)
    assert cc.statistic >= pof.statistic
    assert stats.chi2.ppf(0.95, 2) == pytest.approx(5.991, abs=5e-4)
    assert cc.p_value == pytest.approx(stats.chi2.sf(cc.statistic, 2), abs=1e-12)
    assert cc.test_id == "market.christoffersen_conditional_coverage" and cc.light is Light.RED
    for seq in (np.zeros(50, dtype=bool), np.ones(50, dtype=bool)):
        n = m.christoffersen_conditional_coverage(seq, 0.05)
        assert n.light is Light.NOT_EVALUATED and n.to_dict()["statistic"] is None


def weibull_lr(exceed):
    """An independent likelihood ratio of b = 1 against the Weibull hazard, by direct optimisation in (a, b)."""
    idx = np.flatnonzero(exceed)
    t = exceed.size
    d_unc = np.diff(idx).astype(float)
    # the first spell is censored only when the sample does not start with a hit (review m2)
    cens = ([float(idx[0] + 1)] if idx[0] > 0 else []) + ([float(t - 1 - idx[-1])] if idx[-1] < t - 1 else [])
    cens = [c for c in cens if c > 0]

    def nll(par, force_b1=False):
        a, b = np.exp(par[0]), (1.0 if force_b1 else np.exp(par[1]))
        ll = np.sum(np.log(a**b * b * d_unc ** (b - 1)) - (a * d_unc) ** b) - sum((a * c) ** b for c in cens)
        return -ll

    w = optimize.minimize(nll, [np.log(1 / d_unc.mean()), 0.0], method="Nelder-Mead",
                          options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
    e = optimize.minimize(lambda p: nll([p[0], 0.0], True), [np.log(1 / d_unc.mean())], method="Nelder-Mead",
                          options={"xatol": 1e-10, "fatol": 1e-13})
    return 2.0 * (e.fun - w.fun)


def test_duration_test():
    rng = np.random.default_rng(5)
    iid = rng.random(600) < 0.05
    r1 = m.duration_test(iid, 0.05, n_mc=499, seed=3)
    r2 = m.duration_test(iid, 0.05, n_mc=499, seed=3)
    assert r1.test_id == "market.duration_test" and r1.segment == "alpha=0.05"
    assert r1.p_value == r2.p_value and r1.statistic == r2.statistic  # reproducible for the same seed
    assert r1.n == 600 and r1.n_events == int(iid.sum())
    assert r1.statistic == pytest.approx(weibull_lr(iid), abs=2e-3)
    assert 1 / 500 <= r1.p_value <= 1.0 and r1.statistic >= 0.0
    # clustered exceedances (a long no-hit spell, then a burst) raise the statistic and the rejection
    pos = list(range(100, 112)) + list(range(400, 410)) + [20, 250, 560]
    clus = ex(600, pos)
    rc = m.duration_test(clus, 0.05, n_mc=499, seed=3)
    assert rc.statistic == pytest.approx(weibull_lr(clus), abs=2e-3)
    assert rc.statistic > r1.statistic and rc.p_value < 0.05 and rc.light is Light.RED
    for pos in ([], [7]):
        n = m.duration_test(ex(100, pos), 0.05, n_mc=99)
        assert n.light is Light.NOT_EVALUATED and n.to_dict()["statistic"] is None and n.to_dict()["p_value"] is None
        assert "completed" in n.notes or "fewer than 2" in n.notes
    with pytest.raises(ValueError):
        m.duration_test(iid, 0.05, n_mc=0)


def test_duration_first_day_hit_and_the_bound_on_b():
    # review m2: a hit on the first day leaves no first spell to censor
    comp, cens = m._spells(ex(60, [0, 10, 25]))
    assert comp.tolist() == [10.0, 15.0] and cens.tolist() == [34.0]
    comp, cens = m._spells(ex(60, [3, 10, 59]))
    assert comp.tolist() == [7.0, 49.0] and cens.tolist() == [4.0]
    # review m3: perfectly regular spells have an unbounded MLE of b; the cap is reported, not hidden
    regular = ex(200, range(9, 200, 10))
    r = m.duration_test(regular, 0.1, n_mc=99, seed=1)
    assert r.extras["weibull_b_at_bound"] is True and "bound" in r.notes
    irregular = np.random.default_rng(4).random(300) < 0.05
    assert m.duration_test(irregular, 0.05, n_mc=99, seed=1).extras["weibull_b_at_bound"] is False


def test_basel_traffic_light():
    table = {4: (89.22, "green", 1.50), 5: (95.88, "amber", 1.70), 9: (99.97, "amber", 1.92),
             10: (99.99, "red", 2.00)}
    for k, (pct, zone, mult) in table.items():
        hist = np.zeros(300, dtype=bool)
        hist[:40] = True  # older exceptions outside the last 250 days do not count
        hist[300 - 250 + np.arange(k) * 20] = True
        r = m.basel_traffic_light(hist, model_id="m1")
        assert r.test_id == "market.basel_traffic_light" and r.segment == "alpha=0.01"
        assert r.statistic == k and r.n == 250 and r.n_events == k
        assert round(r.metric, 2) == pct and r.light is Light(zone) and r.p_value is None
        assert r.extras["mar32_multiplier"] == mult and r.extras["zone"] == zone
        assert r.alpha_amber is None and r.alpha_red is None
    zones = [m.basel_traffic_light(np.r_[np.zeros(250 - k, dtype=bool), np.ones(k, dtype=bool)]).light
             for k in range(0, 14)]
    order = {Light.GREEN: 0, Light.AMBER: 1, Light.RED: 2}
    assert [order[z] for z in zones] == sorted(order[z] for z in zones)
    short = m.basel_traffic_light(np.zeros(249, dtype=bool))
    assert short.light is Light.NOT_EVALUATED and short.to_dict()["statistic"] is None and short.metric is None
    assert short.p_value is None


def test_dynamic_quantile_test():
    from riskvalidation.market.caviar import dynamic_quantile_statistic
    rng = np.random.default_rng(8)
    t = 500
    var = -0.02 + 0.004 * rng.standard_normal(t)
    hits = rng.random(t) < 0.05
    r = m.dynamic_quantile_test(hits, var, 0.05, lags=4, model_id="m1", n_mc=199, seed=3)
    stat, pv = dynamic_quantile_statistic(hits.astype(float), var, 0.05, lags=4)
    assert r.test_id == "market.dynamic_quantile_test" and r.segment == "alpha=0.05"
    # review M1: the light follows a Monte Carlo p-value (Dufour form, Bernoulli(alpha) hits with the same VaR column);
    # the asymptotic chi-square p-value is kept in extras
    assert r.statistic == stat and r.n == t and r.n_events == int(hits.sum())
    assert r.extras["p_value_asymptotic"] == pv and r.extras["p_value_basis"] == "monte-carlo"
    assert r.extras["n_mc"] == 199 and 1 / 200 <= r.p_value <= 1.0 and abs(r.p_value * 200 - round(r.p_value * 200)) < 1e-9
    again = m.dynamic_quantile_test(hits, var, 0.05, lags=4, n_mc=199, seed=3)
    assert again.p_value == r.p_value
    with pytest.raises(ValueError):
        m.dynamic_quantile_test(hits, var, 0.05, n_mc=0)
    assert r.light in (Light.GREEN, Light.AMBER, Light.RED)
    small = m.dynamic_quantile_test(hits[:6], var[:6], 0.05, lags=4)
    assert small.light is Light.NOT_EVALUATED and small.to_dict()["statistic"] is None
    assert m.dynamic_quantile_test(hits[:7], var[:7], 0.05, lags=4).light is not Light.NOT_EVALUATED
    with pytest.raises(ValueError):
        m.dynamic_quantile_test(hits, var[:-1], 0.05)


def test_pla_test():
    h = np.arange(250, dtype=float)
    r = m.pla_test(h, h.copy(), model_id="d1")
    assert r.test_id == "market.pla_test" and r.segment is None and r.n == 250
    assert r.statistic == pytest.approx(1.0) and r.metric == 0.0 and r.light is Light.GREEN
    assert r.p_value is None and r.alpha_amber is None and r.alpha_red is None
    assert r.extras["zone"] == "green"
    assert r.extras["thresholds"] == {"spearman_green": 0.80, "spearman_red": 0.70, "ks_green": 0.09, "ks_red": 0.12}
    amber = m.pla_test(h, h + 29.5)
    assert amber.metric == 0.12 and amber.light is Light.AMBER
    assert m.pla_test(h, h + 30.5).light is Light.RED
    rev = m.pla_test(h, h[::-1].copy())
    assert rev.light is Light.RED and rev.statistic == pytest.approx(-1.0)
    for bad in (np.arange(249, dtype=float), np.where(h == 3, np.nan, h), np.ones(250)):
        n = m.pla_test(h, bad)
        assert n.light is Light.NOT_EVALUATED
        d = n.to_dict()
        assert d["statistic"] is None and d["metric"] is None and d["p_value"] is None


def test_desk_backtest():
    for alpha, limit in ((0.01, 12), (0.025, 30)):
        for count, light in ((limit, Light.GREEN), (limit + 1, Light.RED), (0, Light.GREEN)):
            loss = np.full(250, 0.01)
            var = np.full(250, 0.02)
            loss[:count] = 0.03
            r = m.desk_backtest(loss, var, alpha, model_id="d1", basis="HPL")
            assert r.test_id == "market.desk_backtest" and r.segment == f"alpha={alpha}"
            assert r.statistic == count and r.metric == limit and r.n == 250 and r.n_events == count
            assert r.light is light and r.p_value is None and r.extras == {"basis": "HPL"}
    assert m.desk_backtest(np.full(250, 0.01), np.full(250, 0.02), 0.01, basis="APL").extras == {"basis": "APL"}
    short = m.desk_backtest(np.full(249, 0.01), np.full(249, 0.02), 0.01)
    assert short.light is Light.NOT_EVALUATED and short.to_dict()["metric"] is None
    with pytest.raises(ValueError):
        m.desk_backtest(np.full(250, 0.01), np.full(250, 0.02), 0.05)
    with pytest.raises(ValueError):
        m.desk_backtest(np.full(250, 0.01), np.full(250, 0.02), 0.01, basis="XPL")
    # a missing P&L or VaR counts as an exception (the flag of desk_exceptions defaults to True)
    loss = np.full(250, 0.01)
    loss[:13] = np.nan
    assert m.desk_backtest(loss, np.full(250, 0.02), 0.01).light is Light.RED
