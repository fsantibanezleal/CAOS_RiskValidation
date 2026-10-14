"""RV-1217: scenario analysis and its combination with loss data (dossier 05 F.5): pseudo-observations, the tail
fitted to scenario-implied quantiles and the scenario influence function."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.oprisk import scenarios as sc
from riskvalidation.oprisk.scenarios import ScenarioPoint as P


def test_scenario_point_validation_and_frozen():
    p = P("s1", "percentile", 100.0, 0.1)
    assert (p.scenario_id, p.kind, p.loss, p.exceedance_frequency) == ("s1", "percentile", 100.0, 0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.loss = 1.0  # type: ignore[misc]
    assert P("s2", "individual", 5.0, 1.0).kind == "individual"
    for bad in (("s", "interval", 1.0, 0.1), ("s", "percentile", 0.0, 0.1), ("s", "percentile", -5.0, 0.1),
                ("s", "percentile", 1.0, 0.0), ("s", "percentile", 1.0, -1.0), ("s", "percentile", math.nan, 0.1),
                ("s", "percentile", 1.0, math.inf)):
        with pytest.raises(ValueError):
            P(*bad)


def test_pseudo_observations_golden():
    pts = [P("a", "percentile", 100.0, 0.1), P("b", "percentile", 1000.0, 0.01)]
    losses, weights = sc.scenario_pseudo_observations(pts, 10.0)
    assert losses.tolist() == [100.0, 1000.0]
    assert weights == pytest.approx([1.0, 0.1], abs=1e-15)
    with pytest.raises(ValueError):
        sc.scenario_pseudo_observations([P("a", "percentile", 100.0, 0.1)], 0.0)
    with pytest.raises(ValueError):
        sc.scenario_pseudo_observations([], 10.0)


def gpd_points(xi, beta, location, xs, frequency):
    tail = stats.genpareto.sf(np.asarray(xs) - location, xi, scale=beta)
    pairs = enumerate(zip(xs, tail, strict=True))
    return [P(f"p{i}", "percentile", float(x), float(frequency * s)) for i, (x, s) in pairs]


def test_fit_tail_to_scenarios_golden_two_points_on_a_gpd():
    pts = [P("a", "percentile", 200.0, 2.5), P("b", "percentile", 1800.0, 0.1)]
    fit = sc.fit_tail_to_scenarios(pts, 0.0, 10.0)
    assert fit.law == "gpd" and fit.params[2] == 0.0 and fit.threshold == 0.0
    assert fit.params[0] == pytest.approx(0.5, abs=1e-6) and fit.params[1] == pytest.approx(100.0, rel=1e-6)
    assert fit.n == 2


def test_fit_tail_to_scenarios_exact_on_many_points_and_with_a_threshold():
    xs = [30_000.0, 60_000.0, 150_000.0, 500_000.0, 2e6]
    pts = gpd_points(0.8, 20_000.0, 10_000.0, xs, 12.0)
    fit = sc.fit_tail_to_scenarios(pts, 10_000.0, 12.0)
    assert fit.params[0] == pytest.approx(0.8, abs=1e-6) and fit.params[1] == pytest.approx(20_000.0, rel=1e-6)
    assert fit.params[2] == 10_000.0
    assert fit.std_errors[0] == pytest.approx(0.0, abs=1e-6) or math.isnan(fit.std_errors[0])
    # off the GPD the fit is the least squares of the log conditional exceedance (a non-zero residual)
    factors = (1.1, 0.9, 1.05, 0.95, 1.0)
    noisy = [P(p.scenario_id, p.kind, p.loss, p.exceedance_frequency * f) for p, f in zip(pts, factors, strict=True)]
    nf = sc.fit_tail_to_scenarios(noisy, 10_000.0, 12.0)
    assert abs(nf.params[0] - 0.8) < 0.2 and nf.params[0] != fit.params[0]
    again = sc.fit_tail_to_scenarios(noisy, 10_000.0, 12.0)  # deterministic (the fit has no likelihood: loglik is nan)
    assert again.params == nf.params and again.std_errors == nf.std_errors and math.isnan(nf.loglik)


def test_fit_tail_to_scenarios_refusals():
    one = [P("a", "percentile", 200.0, 2.5)]
    with pytest.raises(ValueError):
        sc.fit_tail_to_scenarios(one, 0.0, 10.0)
    same = [P("a", "percentile", 200.0, 2.5), P("b", "percentile", 200.0, 1.0)]
    with pytest.raises(ValueError):
        sc.fit_tail_to_scenarios(same, 0.0, 10.0)
    two = [P("a", "percentile", 200.0, 2.5), P("b", "percentile", 1800.0, 0.1)]
    with pytest.raises(ValueError):                           # a conditional probability of 1 or more
        sc.fit_tail_to_scenarios(two, 0.0, 2.5)
    with pytest.raises(ValueError):
        sc.fit_tail_to_scenarios(two, 0.0, 0.0)
    with pytest.raises(ValueError):                           # a loss at or below the threshold
        sc.fit_tail_to_scenarios(two, 300.0, 10.0)


def test_scenario_influence_golden_and_call_count():
    pts = [P("a", "percentile", 100.0, 0.1), P("b", "percentile", 1000.0, 0.01), P("c", "individual", 50.0, 1.0)]
    calls = []

    def capital(points):
        calls.append(len(points))
        return sum(p.loss * p.exceedance_frequency for p in points)

    infl = sc.scenario_influence(pts, capital)
    assert infl == pytest.approx([0.142857, 0.142857, 0.714286], abs=5e-7)
    assert len(calls) == len(pts) + 1 and calls[0] == 3 and set(calls[1:]) == {2}
    assert infl.shape == (3,)
    # a point whose removal leaves the capital unchanged has influence 0
    flat = sc.scenario_influence(pts, lambda ps: 10.0)
    assert flat.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        sc.scenario_influence(pts, lambda ps: 0.0)
    with pytest.raises(ValueError):
        sc.scenario_influence([], lambda ps: 1.0)
