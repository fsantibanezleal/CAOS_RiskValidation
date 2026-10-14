"""RV-808 and RV-809: conditional EVT of McNeil and Frey (2000) (contract `engine-market-api`, module market.evt)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from riskvalidation.market import evt


def test_evt_var_es_golden_and_limits():
    fit = evt.GpdFit(threshold=1.0, xi=0.5, beta=1.0, n=1000, n_exceed=100, converged=True)
    z, es = evt.evt_var_es(fit, 0.99)
    assert z == pytest.approx(5.324555, abs=5e-7) and es == pytest.approx(11.649111, abs=5e-7)
    assert es >= z
    # xi = 0 is the limit (exponential excesses): z_q = u - beta ln(n (1 - q) / N_u), es = z_q + beta
    f0 = evt.GpdFit(threshold=1.0, xi=0.0, beta=1.0, n=1000, n_exceed=100, converged=True)
    z0, e0 = evt.evt_var_es(f0, 0.99)
    assert z0 == pytest.approx(1.0 - np.log(0.1), abs=1e-12) and e0 == pytest.approx(z0 + 1.0, abs=1e-12)
    near = evt.evt_var_es(evt.GpdFit(1.0, 1e-9, 1.0, 1000, 100, True), 0.99)
    assert near[0] == pytest.approx(z0, abs=1e-6)
    for xi in (0.1, 0.5, 0.9):
        zz, ee = evt.evt_var_es(evt.GpdFit(1.0, xi, 1.0, 1000, 100, True), 0.995)
        assert ee >= zz
    # xi at or above 1: the ES is undefined, never a number
    for xi in (1.0, 1.4):
        zz, ee = evt.evt_var_es(evt.GpdFit(1.0, xi, 1.0, 1000, 100, True), 0.99)
        assert ee is None and np.isfinite(zz)
    # the quantile is increasing in q
    assert evt.evt_var_es(fit, 0.995)[0] > evt.evt_var_es(fit, 0.99)[0]
    with pytest.raises(ValueError):
        evt.evt_var_es(fit, 1.0)


def test_fit_gpd_recovers_simulated_parameters():
    rng = np.random.default_rng(41)
    n = 20000
    z = stats.t.rvs(5, size=n, random_state=rng)  # a heavy tail: xi = 1/5
    f = evt.fit_gpd(z, 0.05)
    assert isinstance(f, evt.GpdFit) and f.converged and f.n == n
    assert f.n_exceed == int(np.sum(z > np.quantile(z, 0.95)))
    assert f.threshold == pytest.approx(np.quantile(z, 0.95))
    assert f.beta > 0 and f.xi == pytest.approx(0.2, abs=0.12)
    # an exact GPD sample: the threshold excesses of a GPD are a GPD with the same shape
    y = stats.genpareto.rvs(0.3, scale=2.0, size=6000, random_state=rng)
    g = evt.fit_gpd(y, 0.5 - 1e-9)
    assert g.xi == pytest.approx(0.3, abs=0.1)
    # the loss convention: the right tail of z_loss, so the sign matters
    left = evt.fit_gpd(-z, 0.05)
    assert left.threshold == pytest.approx(np.quantile(-z, 0.95))


def test_fit_gpd_refuses_few_excesses():
    z = np.random.default_rng(42).normal(size=100)
    with pytest.raises(ValueError):
        evt.fit_gpd(z, 0.05)  # 5 excesses
    with pytest.raises(ValueError):
        evt.fit_gpd(z, 0.0)
    with pytest.raises(ValueError):
        evt.fit_gpd(z, 0.5)
    with pytest.raises(ValueError):
        evt.fit_gpd(np.append(z, np.nan), 0.2)


def test_threshold_stability():
    z = stats.t.rvs(5, size=5000, random_state=np.random.default_rng(43))
    fr = np.array([0.15, 0.05, 0.10, 0.02])
    out = evt.threshold_stability(z, fr)
    assert [row[0] for row in out] == [0.02, 0.05, 0.10, 0.15]
    for frac, xi, ne in out:
        assert xi == pytest.approx(evt.fit_gpd(z, frac).xi) and ne == evt.fit_gpd(z, frac).n_exceed
    counts = [row[2] for row in out]
    assert counts == sorted(counts)


def test_conditional_evt_forecast():
    pytest.importorskip("arch")
    from market_sim import simulate_garch
    y = simulate_garch(1500, nu=5, seed=51)
    f1 = evt.conditional_evt_forecast(y, 0.01, 0.1)
    f2 = evt.conditional_evt_forecast(y, 0.01, 0.1)
    assert f1 == f2 and isinstance(f1, evt.EvtForecast)
    f5 = evt.conditional_evt_forecast(y, 0.05, 0.1)
    assert f1.var < f5.var < 0.02
    assert f1.n_exceed >= 10
    if f1.xi < 1:
        assert f1.es is not None and f1.es <= f1.var
    else:
        assert f1.es is None
    # a different window end moves the forecast: the last return is read
    g = evt.conditional_evt_forecast(np.append(y[:-1], -0.15), 0.01, 0.1)
    assert g.var != f1.var
    gjr = evt.conditional_evt_forecast(y, 0.01, 0.1, kind="gjr")
    assert gjr.var < 0
    with pytest.raises(ValueError):
        evt.conditional_evt_forecast(y[:100], 0.01, 0.1)
