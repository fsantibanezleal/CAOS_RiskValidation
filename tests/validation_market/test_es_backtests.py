"""RV-822 to RV-824: the ES back-tests of Acerbi and Szekely, Du and Escanciano, and McNeil and Frey (contract
`engine-market-api`, module validation.market)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from riskvalidation.core.result import Light
from riskvalidation.validation import market as m


def zeros_sim(rng):
    return np.zeros(5)


def test_acerbi_szekely():
    loss = np.array([0, 0, 0, 0, 0.05])
    var, es = np.full(5, 0.02), np.full(5, 0.03)
    r = m.acerbi_szekely(loss, var, es, 0.025, zeros_sim, model_id="m1", n_sim=10000, seed=1)
    assert r.test_id == "market.acerbi_szekely" and r.segment == "alpha=0.025" and r.model_id == "m1"
    assert r.statistic == pytest.approx(-12.333333, abs=5e-7)
    assert r.p_value == pytest.approx(1 / 10001, abs=1e-15)  # every simulated Z2 is 1, none at or below -12.33
    assert r.light is Light.RED and (r.alpha_amber, r.alpha_red) == (0.05, 0.0001)
    assert r.extras["statistic_name"] == "Z2" and r.extras["n_draws"] == 10000
    assert r.extras["reference"] == {"yellow": -0.70, "red": -1.8}
    assert "critical_5pct" in r.extras and "critical_0_01pct" in r.extras
    assert r.n == 5 and r.n_events == 1
    z1 = m.acerbi_szekely(np.array([0.05, 0, 0, 0, 0.04]), var, es, 0.025, lambda g: np.array([0.05, 0, 0, 0, 0.04]),
                          statistic="Z1", n_sim=200, seed=1)
    assert z1.statistic == pytest.approx(-0.5, abs=1e-12) and z1.extras["statistic_name"] == "Z1"
    assert z1.p_value == pytest.approx(1.0, abs=1e-12)  # the simulated path equals the observed one: every draw ties
    # a correctly specified model: the statistic is centred and the simulated critical values are ordered
    alpha, t = 0.025, 250
    z = stats.norm.ppf(1 - alpha)
    es_true = stats.norm.pdf(z) / alpha
    sim = lambda g: g.standard_normal(t)  # noqa: E731
    obs = np.random.default_rng(3).standard_normal(t)
    ok = m.acerbi_szekely(obs, np.full(t, z), np.full(t, es_true), alpha, sim, n_sim=2000, seed=5)
    again = m.acerbi_szekely(obs, np.full(t, z), np.full(t, es_true), alpha, sim, n_sim=2000, seed=5)
    assert ok.p_value == again.p_value and ok.statistic == again.statistic
    assert ok.extras["critical_0_01pct"] <= ok.extras["critical_5pct"] < 0.5
    assert abs(np.mean([m._z_statistic(sim(np.random.default_rng(i)), np.full(t, z), np.full(t, es_true), alpha, "Z2")
                        for i in range(300)])) < 0.15
    # underestimated risk: losses 1.5 times heavier than the forecast are rejected
    heavy = m.acerbi_szekely(1.5 * obs, np.full(t, z), np.full(t, es_true), alpha, sim, n_sim=2000, seed=5)
    assert heavy.statistic < ok.statistic and heavy.p_value < 0.01 and heavy.light in (Light.AMBER, Light.RED)
    # an undefined ES means the test cannot run
    es_nan = es.copy()
    es_nan[2] = np.nan
    n = m.acerbi_szekely(loss, var, es_nan, 0.025, zeros_sim, n_sim=100)
    assert n.light is Light.NOT_EVALUATED and n.to_dict()["statistic"] is None and n.to_dict()["p_value"] is None
    with pytest.raises(ValueError):
        m.acerbi_szekely(loss, var, es, 0.025, zeros_sim, statistic="Z3")
    with pytest.raises(ValueError):
        m.acerbi_szekely(loss, var, es, 0.025, lambda g: np.zeros(4), n_sim=10)  # a path of the wrong length


def test_du_escanciano_unconditional():
    alpha = 0.025
    h_mean, h_var = alpha / 2, alpha * (1 / 3 - alpha / 4)
    assert h_mean == pytest.approx(0.0125, abs=1e-15) and h_var == pytest.approx(0.0081770833, abs=1e-10)
    r = m.du_escanciano_unconditional(np.full(100, 0.5), alpha, model_id="m1")
    assert r.test_id == "market.du_escanciano_unconditional" and r.segment == "alpha=0.025"
    assert r.statistic == pytest.approx(-1.382327, abs=5e-7)
    assert r.p_value == pytest.approx(stats.norm.sf(-1.382327), abs=1e-6) and r.light is Light.GREEN
    assert r.n == 100
    # PITs piled at zero (risk understated) give a large positive statistic and a small upper-tail p-value
    low = m.du_escanciano_unconditional(np.full(100, 0.001), alpha)
    assert low.statistic > 5.0 and low.p_value < 1e-6 and low.light is Light.RED
    u = np.random.default_rng(1).random(400)
    h = np.where(u <= 0.1, (0.1 - u) / 0.1, 0.0)
    r = m.du_escanciano_unconditional(u, 0.1)
    expected = np.sqrt(400) * (h.mean() - 0.05) / np.sqrt(0.1 * (1 / 3 - 0.1 / 4))
    assert r.statistic == pytest.approx(expected, rel=1e-12)
    n = m.du_escanciano_unconditional(np.full(10, np.nan), alpha)
    assert n.light is Light.NOT_EVALUATED and n.to_dict()["statistic"] is None
    assert m.du_escanciano_unconditional(np.array([0.3]), alpha).light is Light.NOT_EVALUATED
    with pytest.raises(ValueError):
        m.du_escanciano_unconditional(np.array([0.3, 1.4]), alpha)
    with pytest.raises(ValueError):
        m.du_escanciano_unconditional(np.array([0.3, np.nan]), alpha)


def test_du_escanciano_conditional():
    alpha, lags = 0.1, 3
    u = np.random.default_rng(2).random(500)
    r = m.du_escanciano_conditional(u, alpha, lags=lags)
    h = np.where(u <= alpha, (alpha - u) / alpha, 0.0)
    c = h - h.mean()
    rho = np.array([np.sum(c[j:] * c[:-j]) / np.sum(c * c) for j in range(1, lags + 1)])
    expected = 500 * np.sum(rho**2)
    assert r.test_id == "market.du_escanciano_conditional" and r.extras == {"lags": lags}
    assert r.statistic == pytest.approx(expected, rel=1e-10) and r.statistic >= 0.0
    assert r.p_value == pytest.approx(stats.chi2.sf(expected, lags), abs=1e-12) and r.n == 500
    assert stats.chi2.ppf(0.95, 1) == pytest.approx(3.841, abs=5e-4) and stats.chi2.ppf(0.95, 2) == pytest.approx(5.991, abs=5e-4)
    # clustered tail PITs (a burst of low values) are rejected
    v = np.random.default_rng(3).random(500)
    v[100:130] = 0.01
    v[300:330] = 0.02
    burst = m.du_escanciano_conditional(v, alpha, lags=lags)
    assert burst.p_value < 0.001 and burst.light is Light.RED
    const = m.du_escanciano_conditional(np.full(100, 0.5), alpha)
    assert const.light is Light.NOT_EVALUATED and const.to_dict()["statistic"] is None
    assert m.du_escanciano_conditional(np.full(5, np.nan), alpha).light is Light.NOT_EVALUATED
    assert m.du_escanciano_conditional(u[:5], alpha, lags=5).light is Light.NOT_EVALUATED  # T <= lags
    with pytest.raises(ValueError):
        m.du_escanciano_conditional(u, alpha, lags=0)


def test_mcneil_frey_residual_test():
    loss = np.array([0, 6.0, 0, 7.0, 0, 8.0, 0])
    var, es, sigma = np.full(7, 1.0), np.full(7, 5.0), np.full(7, 1.0)
    r = m.mcneil_frey_residual_test(loss, var, es, sigma, 0.025, model_id="m1", n_boot=10000, seed=4)
    assert r.test_id == "market.mcneil_frey_residual_test" and r.segment == "alpha=0.025"
    assert r.statistic == pytest.approx(2.0, abs=1e-12) and r.n_events == 3 and r.n == 7
    assert r.p_value == pytest.approx(1 / 10001, abs=1e-15) and r.light is Light.RED
    same = m.mcneil_frey_residual_test(loss, var, es, sigma, 0.025, n_boot=500, seed=4)
    assert same.p_value == m.mcneil_frey_residual_test(loss, var, es, sigma, 0.025, n_boot=500, seed=4).p_value
    # residuals centred on zero are not rejected; sigma scales the residuals
    rng = np.random.default_rng(6)
    resid = rng.exponential(1.0, 60) - 1.0
    loss = np.concatenate([[0.0], 3.0 + 2.0 * resid])
    ok = m.mcneil_frey_residual_test(loss, np.full(61, 1.0), np.full(61, 3.0), np.full(61, 2.0), 0.05, n_boot=2000,
                                     seed=1)
    assert ok.statistic == pytest.approx(resid.mean(), abs=1e-12) and ok.p_value > 0.05
    assert ok.n_events == 60
    for loss_bad in (np.array([0.0, 6.0, 0.0]), np.array([0.0, 0.0, 0.0])):  # one exceedance, then none
        n = m.mcneil_frey_residual_test(loss_bad, np.full(3, 1.0), np.full(3, 5.0), np.full(3, 1.0), 0.025)
        assert n.light is Light.NOT_EVALUATED and n.to_dict()["statistic"] is None
    es_nan = np.array([5.0, np.nan, 5.0, 5.0, 5.0, 5.0, 5.0])
    n = m.mcneil_frey_residual_test(np.array([0, 6.0, 0, 7.0, 0, 8.0, 0]), var, es_nan, sigma, 0.025)
    assert n.light is Light.NOT_EVALUATED and n.to_dict()["p_value"] is None
    with pytest.raises(ValueError):
        m.mcneil_frey_residual_test(loss, np.full(61, 1.0), np.full(61, 3.0), np.full(61, 0.0), 0.05)
