"""BS-16: adaptive conformal recalibration of VaR and ES forecasts (module market.conformal)."""

from __future__ import annotations

import numpy as np
import pytest
from market_sim import simulate_garch
from scipy import stats

from riskvalidation.market import conformal as cf
from riskvalidation.market.historical import historical_var_es
from riskvalidation.market.scoring import fz0_loss
from riskvalidation.validation import market as vm

ALPHA = 0.05


def normal_forecaster(sigma_of_t, scale=1.0):
    """A model with the right shape and a scale error: VaR and ES of N(0, (scale sigma_t)^2) at the asked level."""
    def f(t, level):
        s = scale * sigma_of_t[t]
        v = float(stats.norm.ppf(level))
        return s * v, -s * float(stats.norm.pdf(v)) / level
    return f


def hs(r, window=250):
    return lambda t, level: historical_var_es(r[t - window:t], level, window)


# ----------------------------------------------------------------------------------------------- the level update
def test_update_closed_form():
    # no hit: the level rises by gamma alpha; hit: it falls by gamma (1 - alpha)
    assert cf.aci_update(0.05, False, 0.05, 0.01) == pytest.approx(0.05 + 0.01 * 0.05)
    assert cf.aci_update(0.05, True, 0.05, 0.01) == pytest.approx(0.05 - 0.01 * 0.95)


def test_update_is_clipped_to_the_band():
    assert cf.aci_update(0.0011, True, 0.05, 0.01, level_min=0.001, level_max=0.25) == 0.001
    assert cf.aci_update(0.249, False, 0.05, 0.5, level_min=0.001, level_max=0.25) == 0.25


@pytest.mark.parametrize("kw", [dict(alpha=0.0), dict(alpha=0.5), dict(alpha=0.05, gamma=-0.1),
                                dict(alpha=0.05, level_min=0.06), dict(alpha=0.05, level_max=0.04),
                                dict(alpha=0.05, level_max=0.6)])
def test_update_rejects_bad_parameters(kw):
    args = dict(alpha=0.05, gamma=0.01, level_min=0.001, level_max=0.25)
    args.update(kw)
    with pytest.raises(ValueError):
        cf.aci_update(0.05, False, **args)


# ----------------------------------------------------------------------------------------------- the wrapper
def make_data(n=1500, seed=3):
    return simulate_garch(n, nu=5, seed=seed)


def test_gamma_zero_is_the_unwrapped_forecast():
    r = make_data(700)
    f = hs(r)
    res = cf.aci_var_es(r, f, ALPHA, gamma=0.0, start=250)
    base = np.array([f(t, ALPHA) for t in range(250, r.size)])
    assert np.allclose(res.levels, ALPHA)
    assert np.allclose(res.var, base[:, 0]) and np.allclose(res.es, base[:, 1])
    assert np.array_equal(res.hits, r[250:] < base[:, 0])


def test_recursion_matches_the_update_step_by_step():
    r = make_data(400)
    f = hs(r)
    res = cf.aci_var_es(r, f, ALPHA, gamma=0.02, start=250)
    lvl = ALPHA
    for k, t in enumerate(range(250, r.size)):
        assert res.levels[k] == pytest.approx(lvl)
        v, e = f(t, lvl)
        assert res.var[k] == v and res.es[k] == e
        lvl = cf.aci_update(lvl, r[t] < v, ALPHA, 0.02, res.level_min, res.level_max)


def test_scale_error_is_repaired_and_unwrapped_is_not():
    # the model volatility is half the truth: the unwrapped hit rate is far above alpha, ACI brings it back
    rng = np.random.default_rng(5)
    n = 6000
    sig = np.full(n, 0.01)
    r = rng.standard_normal(n) * sig
    f = normal_forecaster(sig, scale=0.7)
    wrapped = cf.aci_var_es(r, f, ALPHA, gamma=0.002, start=1, level_min=1e-3, level_max=0.25)
    plain = cf.aci_var_es(r, f, ALPHA, gamma=0.0, start=1)
    assert plain.hits.mean() > 0.10
    assert abs(wrapped.hits.mean() - ALPHA) < 0.012
    # Gibbs and Candes (2021) proposition 3.1 bound of the unclipped recursion, as a scale for the gap
    assert abs(wrapped.hits.mean() - ALPHA) <= (max(ALPHA, 1 - ALPHA) + 0.002) / (0.002 * wrapped.hits.size)


def test_no_look_ahead_in_the_levels():
    r = make_data(600)
    base = cf.aci_var_es(r, hs(r), ALPHA, gamma=0.01, start=250)
    r2 = r.copy()
    r2[500:] *= 3.0
    pert = cf.aci_var_es(r2, hs(r2), ALPHA, gamma=0.01, start=250)
    k = 500 - 250
    assert np.array_equal(base.levels[: k + 1], pert.levels[: k + 1])
    assert not np.array_equal(base.levels, pert.levels)


def test_deterministic_and_inputs_not_mutated():
    r = make_data(500)
    keep = r.copy()
    a = cf.aci_var_es(r, hs(r), ALPHA, gamma=0.01, start=250)
    b = cf.aci_var_es(r, hs(r), ALPHA, gamma=0.01, start=250)
    assert np.array_equal(r, keep)
    assert np.array_equal(a.var, b.var) and np.array_equal(a.levels, b.levels)


def test_clipped_days_are_counted():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(600) * 0.01
    f = normal_forecaster(np.full(600, 0.01), scale=0.2)
    res = cf.aci_var_es(r, f, ALPHA, gamma=0.05, start=1, level_min=0.01, level_max=0.25)
    assert res.n_clipped == int(res.clipped.sum()) > 0
    assert res.levels.min() >= 0.01 - 1e-15


def test_wrapper_validation():
    r = make_data(300)
    ok = hs(r)
    with pytest.raises(ValueError, match="start"):
        cf.aci_var_es(r, ok, ALPHA, start=0)
    with pytest.raises(ValueError, match="no forecast day"):
        cf.aci_var_es(r, ok, ALPHA, start=300)
    with pytest.raises(ValueError, match="finite"):
        cf.aci_var_es(r, lambda t, level: (float("nan"), -0.1), ALPHA, start=250)
    with pytest.raises(ValueError, match="es <= var"):
        cf.aci_var_es(r, lambda t, level: (-0.02, -0.01), ALPHA, start=250)
    bad = r.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        cf.aci_var_es(bad, ok, ALPHA, start=250)


def test_historical_forecaster_matches_the_engine():
    r = make_data(400)
    f = cf.historical_forecaster(r, 250)
    assert f(300, 0.03) == historical_var_es(r[50:300], 0.03, 250)
    with pytest.raises(ValueError):
        f(100, 0.03)


# ----------------------------------------------------------------------------------------------- the scores
def test_scores_match_the_back_tests():
    rng = np.random.default_rng(9)
    y = rng.standard_t(5, 800) * 0.01
    v = np.full(800, -0.015)
    e = np.full(800, -0.021)
    s = cf.forecast_scores(y, v, e, ALPHA)
    ex = y < v
    assert s.n == 800 and s.n_hits == int(ex.sum())
    assert s.kupiec_p == pytest.approx(vm.kupiec_pof(ex, ALPHA).p_value)
    assert s.independence_p == pytest.approx(vm.christoffersen_independence(ex, ALPHA).p_value)
    assert s.cc_p == pytest.approx(vm.christoffersen_conditional_coverage(ex, ALPHA).p_value)
    assert s.fz0_mean == pytest.approx(float(np.mean(fz0_loss(y, v, e, ALPHA))))
    assert s.z2 == pytest.approx(vm._z_statistic(-y, -v, -e, ALPHA, "Z2"))


def test_scores_not_evaluated_form_has_reasons():
    y = np.full(50, 0.001)
    v = np.full(50, -0.02)
    e = np.full(50, -0.03)
    s = cf.forecast_scores(y, v, e, ALPHA)
    # no hit: independence is undefined, reported as None with a reason, never a number
    assert s.n_hits == 0 and s.independence_p is None and s.cc_p is None
    assert "independence" in s.reasons[0]
    assert s.kupiec_p is not None


def test_compare_reports_both_sides_and_the_fz0_change():
    rng = np.random.default_rng(5)
    n = 2500
    sig = np.full(n, 0.01)
    r = rng.standard_normal(n) * sig
    cmp_ = cf.aci_compare(r, normal_forecaster(sig, scale=0.5), ALPHA, gamma=0.005, start=1, level_min=1e-4)
    assert cmp_.unwrapped.kupiec_p < 0.01
    assert cmp_.wrapped.n == cmp_.unwrapped.n == n - 1
    assert cmp_.fz0_change == pytest.approx(
        (cmp_.wrapped.fz0_mean - cmp_.unwrapped.fz0_mean) / abs(cmp_.unwrapped.fz0_mean))
    assert cmp_.wrapped.fz0_mean < cmp_.unwrapped.fz0_mean


def test_volatility_break_hs250_hit_rate_moves_toward_alpha():
    """The plan hypothesis in test time, one path: after a volatility break HS-250 under-covers and the wrapped
    version has the hit rate closer to alpha (the study over many paths is in the docs page)."""
    rng = np.random.default_rng(21)
    n = 1250
    sig = np.where(np.arange(n) < 1000, 0.007, 0.02)
    r = rng.standard_t(6, n) * sig * np.sqrt(4 / 6)
    cmp_ = cf.aci_compare(r, cf.historical_forecaster(r, 250), 0.01, gamma=0.01, start=250)
    assert abs(cmp_.wrapped.hit_rate - 0.01) < abs(cmp_.unwrapped.hit_rate - 0.01)
