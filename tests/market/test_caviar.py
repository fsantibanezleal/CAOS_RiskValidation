"""RV-810 and RV-811: CAViaR, its ES extension and the dynamic quantile statistic (contract `engine-market-api`, module
market.caviar)."""

from __future__ import annotations

import numpy as np
import pytest
from market_sim import simulate_garch
from scipy import stats

from riskvalidation.market import caviar as cv
from riskvalidation.market.conventions import var_es_from_sample


def order_stat(y, alpha):
    """The empirical quantile the recursion starts from: the k-th smallest return, k = ceil(alpha N)."""
    return float(np.sort(y)[int(np.ceil(alpha * y.size - 1e-9)) - 1])


def tick(y, f, alpha):
    return float(np.sum((alpha - (y < f)) * (y - f)))


def sav_path(params, y, f1):
    f = np.empty(y.size)
    f[0] = f1
    for t in range(1, y.size):
        f[t] = params[0] + params[1] * f[t - 1] + params[2] * abs(y[t - 1])
    return f


def as_path(params, y, f1):
    f = np.empty(y.size)
    f[0] = f1
    for t in range(1, y.size):
        yp, ym = max(y[t - 1], 0.0), -min(y[t - 1], 0.0)
        f[t] = params[0] + params[1] * f[t - 1] + params[2] * yp + params[3] * ym
    return f


@pytest.fixture(scope="module")
def series():
    return simulate_garch(900, nu=6, seed=61)


def test_fit_caviar_beats_the_constant_benchmark(series):
    alpha = 0.05
    for spec, npar in (("sav", 3), ("as", 4)):
        f = cv.fit_caviar(series, alpha, spec=spec, n_starts=4, seed=1)
        assert isinstance(f, cv.CaviarFit) and f.spec == spec and f.alpha == alpha
        assert f.params.shape == (npar,) and f.n_starts == 4 and f.n_starts >= 2
        bench = tick(series, np.full(series.size, np.quantile(series, alpha)), alpha)
        assert f.tick_loss <= bench + 1e-12
        assert f.tick_loss < bench  # a GARCH series has quantile dynamics the recursion captures
        # the reported tick loss is the loss of the reported parameters on the stated path
        path = (sav_path if spec == "sav" else as_path)(f.params, series, order_stat(series, alpha))
        assert tick(series, path, alpha) == pytest.approx(f.tick_loss, rel=1e-9)
        assert f.converged


def test_caviar_recursions_and_forecast(series):
    alpha = 0.05
    f1 = order_stat(series, alpha)
    fit = cv.CaviarFit(spec="sav", alpha=alpha, params=np.array([-0.002, 0.8, -0.3]), tick_loss=0.0, n_starts=2,
                       converged=True)
    path = sav_path(fit.params, series, f1)
    expected = fit.params[0] + fit.params[1] * path[-1] + fit.params[2] * abs(series[-1])
    assert cv.caviar_var_forecast(fit, series) == pytest.approx(expected, abs=1e-15)
    fit_as = cv.CaviarFit(spec="as", alpha=alpha, params=np.array([-0.002, 0.8, -0.1, -0.5]), tick_loss=0.0,
                          n_starts=2, converged=True)
    path = as_path(fit_as.params, series, f1)
    y = series[-1]
    expected = (fit_as.params[0] + fit_as.params[1] * path[-1] + fit_as.params[2] * max(y, 0.0)
                + fit_as.params[3] * -min(y, 0.0))
    assert cv.caviar_var_forecast(fit_as, series) == pytest.approx(expected, abs=1e-15)
    # the last return is read and nothing after it
    changed = series.copy()
    changed[-1] = -0.2
    assert cv.caviar_var_forecast(fit, changed) != cv.caviar_var_forecast(fit, series)


def test_caviar_es_forecast(series):
    fit = cv.fit_caviar(series, 0.05, spec="sav", n_starts=3, seed=2)
    var = cv.caviar_var_forecast(fit, series)
    es = cv.caviar_es_forecast(fit, series)
    assert es <= var < 0.0
    assert cv.caviar_es_forecast(fit, series) == es  # deterministic
    # the fitted gap between ES and VaR is of the order of the empirical one in the sample
    v, e = var_es_from_sample(series, 0.05)
    assert 0.3 * (v - e) < var - es < 4.0 * (v - e)


def test_caviar_es_fit_declares_the_two_step_estimator(series):
    """Decision D-10 (review M3): the ES parameters are estimated in a second step on the VaR path of the tick-loss
    fit, not jointly as in Taylor (2019); the result, the docstrings and the page say so."""
    fit = cv.fit_caviar(series, 0.05, spec="sav", n_starts=3, seed=2)
    ef = cv.caviar_es_fit(fit, series)
    assert isinstance(ef, cv.CaviarEsFit) and ef.estimator == "two-step"
    assert ef.var == cv.caviar_var_forecast(fit, series) and ef.es == cv.caviar_es_forecast(fit, series)
    assert ef.es <= ef.var < 0.0 and ef.gamma.shape == (3,) and np.all(ef.gamma >= 0.0) and ef.gamma[2] < 1.0
    assert np.isfinite(ef.log_likelihood)
    assert "two-step" in (cv.caviar_es_forecast.__doc__ or "") and "D-10" in (cv.caviar_es_forecast.__doc__ or "")
    assert "two-step" in (cv.caviar_es_fit.__doc__ or "")


def test_fit_caviar_refusals_and_determinism(series):
    a = cv.fit_caviar(series, 0.05, n_starts=3, seed=7)
    b = cv.fit_caviar(series, 0.05, n_starts=3, seed=7)
    assert np.array_equal(a.params, b.params) and a.tick_loss == b.tick_loss
    with pytest.raises(ValueError):
        cv.fit_caviar(series[:499], 0.05)
    with pytest.raises(ValueError):
        cv.fit_caviar(series, 0.05, spec="igarch")
    with pytest.raises(ValueError):
        cv.fit_caviar(series, 0.05, n_starts=1)
    with pytest.raises(ValueError):
        cv.fit_caviar(series, 0.7)
    with pytest.raises(ValueError):
        cv.fit_caviar(np.append(series, np.nan), 0.05)
    before = series.copy()
    cv.fit_caviar(series, 0.05, n_starts=2, seed=1)
    assert np.array_equal(series, before)


def test_dynamic_quantile_statistic():
    rng = np.random.default_rng(71)
    t, alpha, lags = 800, 0.05, 4
    var = -0.02 + 0.005 * rng.standard_normal(t)
    hits = (rng.random(t) < alpha).astype(float)
    stat, pv = cv.dynamic_quantile_statistic(hits, var, alpha, lags=lags)
    # direct computation: rows lags..T-1, regressors: constant, lagged hits, the VaR of the day
    rows = range(lags, t)
    X = np.array([[1.0, *[hits[i - j] for j in range(1, lags + 1)], var[i]] for i in rows])
    y = np.array([hits[i] - alpha for i in rows])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    direct = float(beta @ (X.T @ X) @ beta) / (alpha * (1 - alpha))
    assert stat == pytest.approx(direct, rel=1e-9)
    assert pv == pytest.approx(stats.chi2.sf(direct, lags + 2), abs=1e-12) and 0.0 <= pv <= 1.0
    # invariant to an affine change of the VaR (the sign convention does not matter)
    s2, p2 = cv.dynamic_quantile_statistic(hits, -3.0 * var + 0.1, alpha, lags=lags)
    assert s2 == pytest.approx(stat, rel=1e-8) and p2 == pytest.approx(pv, abs=1e-8)
    # clustered hits are rejected
    clustered = np.zeros(t)
    clustered[100:140:2] = 1
    clustered[300:340:2] = 1
    sc, pc = cv.dynamic_quantile_statistic(clustered, var, alpha, lags=lags)
    assert pc < 0.001 and sc > stat
    with pytest.raises(ValueError):
        cv.dynamic_quantile_statistic(hits[:-1], var, alpha)
    with pytest.raises(ValueError):
        cv.dynamic_quantile_statistic(hits[:5], var[:5], alpha, lags=4)
