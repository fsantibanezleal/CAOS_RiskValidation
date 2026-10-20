"""RV-1459 to RV-1461: the decision model, the protected effect and the rate-spread model against statsmodels."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.fairness.conditional import (
    DecisionModel,
    Effect,
    fit_decision_model,
    fit_rate_spread_model,
    protected_effect,
)


def _data(seed=5, n=1500, q=2, c=3, intercept=0.3):
    rng = np.random.default_rng(seed)
    protected = (rng.random((n, q)) < 0.3).astype(float)
    controls = rng.normal(size=(n, c))
    eta = intercept - 0.6 * protected[:, 0] + 0.2 * protected[:, 1] + controls @ np.array([0.8, -0.4, 0.2])
    decision = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return decision, protected, controls, eta


def test_fit_decision_model_matches_statsmodels_logit():
    sm = pytest.importorskip("statsmodels.api", reason="statsmodels is the optional reference")
    decision, protected, controls, _ = _data()
    model = fit_decision_model(decision, protected, controls)
    assert isinstance(model, DecisionModel)
    assert model.converged and model.n == 1500
    assert len(model.coefficients) == len(model.standard_errors) == 1 + 2 + 3
    x = np.column_stack([np.ones(len(decision)), protected, controls])
    ref = sm.Logit(decision, x).fit(disp=0, tol=1e-12, maxiter=200)
    assert np.allclose(model.coefficients, ref.params, atol=1e-6, rtol=0)
    assert np.allclose(model.standard_errors, ref.bse, atol=1e-6, rtol=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.n = 3  # type: ignore[misc]
    # the true log-odds of the planted protected effect lies within 4 standard errors (not a golden: a sanity check)
    assert abs(model.coefficients[1] - (-0.6)) < 4 * model.standard_errors[1]
    # a model without controls: one column, still valid
    m0 = fit_decision_model(decision, protected, np.empty((len(decision), 0)))
    assert len(m0.coefficients) == 3 and m0.converged
    # a single protected column given as a 1-D array
    m1 = fit_decision_model(decision, protected[:, 0], controls)
    assert len(m1.coefficients) == 1 + 1 + 3


def test_fit_decision_model_separation_is_not_converged():
    rng = np.random.default_rng(1)
    n = 200
    controls = rng.normal(size=(n, 1))
    protected = (rng.random((n, 1)) < 0.4).astype(float)
    decision = (controls[:, 0] > 0).astype(int)  # the control separates the classes completely
    model = fit_decision_model(decision, protected, controls)
    assert model.converged is False
    assert model.n == n


def test_fit_decision_model_refuses_invalid_inputs():
    decision, protected, controls, _ = _data(n=100)
    with pytest.raises(ValueError, match="decision"):
        fit_decision_model(np.where(decision == 1, 2, 0), protected, controls)
    with pytest.raises(ValueError, match="decision"):
        fit_decision_model(np.zeros(100, dtype=int), protected, controls)  # one class only
    bad = controls.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="controls"):
        fit_decision_model(decision, protected, bad)
    badp = protected.copy()
    badp[0, 0] = np.inf
    with pytest.raises(ValueError, match="protected"):
        fit_decision_model(decision, badp, controls)
    with pytest.raises(ValueError, match="protected"):
        fit_decision_model(decision, protected[:50], controls)
    with pytest.raises(ValueError, match="rank"):
        fit_decision_model(decision, protected, np.column_stack([controls, controls[:, 0]]))  # collinear
    with pytest.raises(ValueError, match="protected"):
        fit_decision_model(decision, np.empty((100, 0)), controls)


def test_protected_effect_identities():
    decision, protected, controls, _ = _data()
    model = fit_decision_model(decision, protected, controls)
    for index in (1, 2):
        e = protected_effect(model, index)
        assert isinstance(e, Effect)
        z = stats.norm.ppf(0.975)
        assert e.coefficient == model.coefficients[index]
        assert e.odds_ratio == pytest.approx(math.exp(e.coefficient), abs=1e-15)
        assert e.lo == pytest.approx(e.coefficient - z * model.standard_errors[index], abs=1e-12)
        assert e.hi == pytest.approx(e.coefficient + z * model.standard_errors[index], abs=1e-12)
        assert e.lo <= e.coefficient <= e.hi
        assert e.level == 0.95
    wide = protected_effect(model, 1, level=0.99)
    assert wide.hi - wide.lo > protected_effect(model, 1).hi - protected_effect(model, 1).lo
    with pytest.raises(dataclasses.FrozenInstanceError):
        wide.lo = 0.0  # type: ignore[misc]


def test_protected_effect_refusals():
    decision, protected, controls, _ = _data(n=300)
    model = fit_decision_model(decision, protected, controls)
    for index in (-1, len(model.coefficients), 99):
        with pytest.raises(ValueError, match="index"):
            protected_effect(model, index)
    for level in (0.0, 1.0, -0.2, 1.5, float("nan")):
        with pytest.raises(ValueError, match="level"):
            protected_effect(model, 1, level=level)
    unconverged = DecisionModel((0.0, 1.0), (math.nan, math.nan), 10, False)
    with pytest.raises(ValueError, match="standard error"):
        protected_effect(unconverged, 1)


def _spread(seed=9, n=900):
    rng = np.random.default_rng(seed)
    protected = (rng.random((n, 1)) < 0.35).astype(float)
    controls = rng.normal(size=(n, 2))
    spread = 0.4 + 0.25 * protected[:, 0] + controls @ np.array([0.3, -0.2]) + rng.normal(scale=0.5, size=n)
    return spread, protected, controls


def test_fit_rate_spread_model_matches_statsmodels_ols():
    sm = pytest.importorskip("statsmodels.api", reason="statsmodels is the optional reference")
    spread, protected, controls = _spread()
    model = fit_rate_spread_model(spread, protected, controls)
    assert model.converged and model.n == len(spread)
    x = np.column_stack([np.ones(len(spread)), protected, controls])
    ref = sm.OLS(spread, x).fit()
    assert np.allclose(model.coefficients, ref.params, atol=1e-8, rtol=0)
    assert np.allclose(model.standard_errors, ref.bse, atol=1e-8, rtol=0)
    e = protected_effect(model, 1)
    assert e.lo < 0.25 < e.hi or abs(e.coefficient - 0.25) < 4 * model.standard_errors[1]


def test_fit_rate_spread_model_drops_missing_rows_and_refuses_invalid():
    spread, protected, controls = _spread(n=300)
    holed = spread.copy()
    holed[[3, 10, 11]] = np.nan
    model = fit_rate_spread_model(holed, protected, controls)
    assert model.n == 297  # three rows dropped, the count is len(input) - n; nothing imputed
    keep = ~np.isnan(holed)
    again = fit_rate_spread_model(spread[keep], protected[keep], controls[keep])
    assert np.allclose(model.coefficients, again.coefficients, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="rank"):
        fit_rate_spread_model(spread, protected, np.column_stack([controls, controls[:, 0] * 2.0]))
    bad = controls.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="controls"):
        fit_rate_spread_model(spread, protected, bad)
    with pytest.raises(ValueError, match="rate_spread"):
        fit_rate_spread_model(np.full(300, np.nan), protected, controls)
    with pytest.raises(ValueError, match="rate_spread"):
        fit_rate_spread_model(spread[:100], protected, controls)


def test_protected_effect_refuses_the_intercept_and_reads_indicator_j_at_index_1_plus_j():
    """F2: index is the position in `coefficients`; index 0 is the intercept, protected indicator j is index 1 + j."""
    decision, protected, controls, _ = _data()
    assert protected.shape[1] == 2  # two protected indicators
    model = fit_decision_model(decision, protected, controls)
    with pytest.raises(ValueError, match="index 0 is the intercept, not a protected indicator; indicator j is index 1"):
        protected_effect(model, 0)
    for j in (0, 1):
        assert protected_effect(model, 1 + j).coefficient == model.coefficients[1 + j]
    assert "1 + j" in (protected_effect.__doc__ or "")
