"""RV-812: the dynamic semiparametric VaR and ES models of Patton, Ziegel and Chen (2019), estimated by minimising the
average FZ0 loss (contract `engine-market-api`, module market.gas_fz; recursions: arXiv 1707.05108 eqs 9 to 20 and 27)."""

from __future__ import annotations

import numpy as np
import pytest
from market_sim import simulate_garch

from riskvalidation.market import gas_fz as gz
from riskvalidation.market.conventions import var_es_from_sample
from riskvalidation.market.scoring import fz0_loss


@pytest.fixture(scope="module")
def series():
    return simulate_garch(700, nu=6, seed=81)


def one_factor_path(params, y, v0, e0, alpha):
    beta, gamma, a, b = params
    kappa, v, e = 0.0, [], []
    for t in range(y.size):
        v.append(a * np.exp(kappa))
        e.append(b * np.exp(kappa))
        hit = 1.0 if y[t] <= v[-1] else 0.0
        kappa = beta * kappa + gamma * (-1.0 / e[-1]) * (hit * y[t] / alpha - e[-1])
    return np.array(v), np.array(e), a * np.exp(kappa), b * np.exp(kappa)


def test_fit_gas_fz_beats_the_constant_benchmark(series):
    alpha = 0.05
    v0, e0 = var_es_from_sample(series, alpha)
    bench = float(np.mean(fz0_loss(series, np.full(series.size, v0), np.full(series.size, e0), alpha)))
    for spec in ("one-factor", "two-factor"):
        f = gz.fit_gas_fz(series, alpha, spec=spec, n_starts=3, seed=1)
        assert isinstance(f, gz.GasFzFit) and f.spec == spec and f.alpha == alpha and f.n_starts == 3
        assert f.mean_fz0 <= bench + 1e-9
        assert f.params.shape == ((4,) if spec == "one-factor" else (8,))
    f1 = gz.fit_gas_fz(series, alpha, spec="one-factor", n_starts=3, seed=1)
    assert f1.mean_fz0 < bench  # the volatility clustering of the series is captured


def test_gas_fz_recursion_matches_a_direct_computation(series):
    alpha = 0.05
    f = gz.GasFzFit(spec="one-factor", alpha=alpha, params=np.array([0.95, -0.02, -1.6, -2.1]), mean_fz0=0.0,
                    n_starts=2, converged=True)
    # the scale (a, b) is in units of the recursion: the path is a e^kappa with kappa from eq 20
    v, e, v_next, e_next = one_factor_path(f.params, series, None, None, alpha)
    var, es = gz.gas_fz_forecast(f, series)
    assert var == pytest.approx(v_next, rel=1e-12) and es == pytest.approx(e_next, rel=1e-12)
    assert es <= var < 0
    # the last return is read and nothing after it
    changed = series.copy()
    changed[-1] = -0.2
    assert gz.gas_fz_forecast(f, changed) != (var, es)
    two = gz.fit_gas_fz(series, alpha, spec="two-factor", n_starts=2, seed=3)
    v2, e2 = gz.gas_fz_forecast(two, series)
    assert e2 <= v2 < 0
    # the two-factor recursion on its own parameters: w + B (v, e) + A lambda (eqs 9 to 12, diagonal B)
    w1, w2, b1, b2, a11, a12, a21, a22 = two.params
    v0, e0 = var_es_from_sample(series, alpha)
    vt, et = v0, e0
    for y in series:
        hit = 1.0 if y <= vt else 0.0
        lam_v = -vt * (hit - alpha)
        lam_e = hit * y / alpha - et
        vt, et = w1 + b1 * vt + a11 * lam_v + a12 * lam_e, w2 + b2 * et + a21 * lam_v + a22 * lam_e
    assert (v2, e2) == pytest.approx((vt, et), rel=1e-12)


def test_hybrid_fit_and_forecast(series):
    alpha = 0.05
    f = gz.fit_hybrid_garch_fz(series, alpha, n_starts=3, seed=5)
    assert f.spec == "hybrid" and f.params.shape == (5,) and f.n_starts == 3
    v0, e0 = var_es_from_sample(series, alpha)
    bench = float(np.mean(fz0_loss(series, np.full(series.size, v0), np.full(series.size, e0), alpha)))
    assert f.mean_fz0 <= bench + 1e-9
    var, es = gz.gas_fz_forecast(f, series)
    assert es <= var < 0
    again = gz.fit_hybrid_garch_fz(series, alpha, n_starts=3, seed=5)
    assert np.array_equal(f.params, again.params) and f.mean_fz0 == again.mean_fz0


def test_gas_fz_refusals_and_determinism(series):
    a = gz.fit_gas_fz(series, 0.05, n_starts=2, seed=9)
    b = gz.fit_gas_fz(series, 0.05, n_starts=2, seed=9)
    assert np.array_equal(a.params, b.params) and a.mean_fz0 == b.mean_fz0 and a.converged == b.converged
    with pytest.raises(ValueError):
        gz.fit_gas_fz(series[:499], 0.05)
    with pytest.raises(ValueError):
        gz.fit_gas_fz(series, 0.05, spec="three-factor")
    with pytest.raises(ValueError):
        gz.fit_gas_fz(series, 0.05, spec="hybrid")
    with pytest.raises(ValueError):
        gz.fit_gas_fz(series, 0.05, n_starts=1)
    with pytest.raises(ValueError):
        gz.fit_gas_fz(series, 0.6)
    with pytest.raises(ValueError):
        gz.fit_gas_fz(np.append(series, np.nan), 0.05)
    before = series.copy()
    gz.fit_gas_fz(series, 0.05, n_starts=2, seed=1)
    assert np.array_equal(series, before)
