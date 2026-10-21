"""RV-1319 to RV-1324: the aggregation validation tests (contract `engine-aggregation-api`, module
validation.aggregation). Each returns one `TestResult`; the rows serialise into the fields of `common/test-result`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.aggregation import simple as s
from riskvalidation.core.result import Light
from riskvalidation.validation import aggregation as v

ROW_FIELDS = {"test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
              "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras"}
EC = np.array([3.0, 4.0])


def test_simple_sum_and_variance_covariance_golden():
    r = v.simple_sum_and_variance_covariance(EC, np.eye(2), 5.0)
    assert r.test_id == "aggregation.simple_sum_and_variance_covariance"
    assert r.statistic == pytest.approx(1.0, abs=1e-12)
    assert r.metric == pytest.approx(0.7143, abs=5e-5)
    assert r.metric == pytest.approx(5.0 / 7.0, rel=1e-12)
    assert r.light is Light.GREEN and r.p_value is None
    assert r.extras["simple_sum"] == 7.0
    assert r.extras["variance_covariance"] == pytest.approx(5.0, abs=1e-12)
    assert r.extras["thresholds"] == {"amber": 0.10, "tolerance": 1e-9}
    assert r.alpha_amber is None and r.alpha_red is None
    assert v.simple_sum_and_variance_covariance(EC, np.eye(2), 8.0).light is Light.RED
    amber = v.simple_sum_and_variance_covariance(EC, np.eye(2), 5.6)
    assert amber.statistic == pytest.approx(1.12, abs=1e-12) and amber.light is Light.AMBER
    # an aggregate exactly on the simple sum is not above it; a custom gate moves the amber light
    assert v.simple_sum_and_variance_covariance(EC, np.eye(2), 7.0).light is Light.AMBER
    assert v.simple_sum_and_variance_covariance(EC, np.eye(2), 5.6, amber=0.2).light is Light.GREEN
    assert v.simple_sum_and_variance_covariance(EC, np.eye(2), 5.0, model_id="m", segment="g").model_id == "m"
    assert s.variance_covariance(EC, np.eye(2)) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="positive semi-definite"):
        v.simple_sum_and_variance_covariance(np.ones(3), np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]]), 1.0)
    with pytest.raises(ValueError, match="amber"):
        v.simple_sum_and_variance_covariance(EC, np.eye(2), 5.0, amber=-0.1)


def test_copula_choice_sensitivity_golden():
    r = v.copula_choice_sensitivity(100.0, ((4, 107.0), (8, 103.5), (30, 101.0)))
    assert r.test_id == "aggregation.copula_choice_sensitivity"
    assert r.statistic == pytest.approx(0.07, abs=1e-12)
    assert r.metric == pytest.approx(0.07, abs=1e-12)  # the signed value at the smallest nu
    assert r.light is Light.AMBER
    assert r.extras["thresholds"] == {"amber": 0.05, "red": 0.15}
    assert r.extras["differences"] == pytest.approx({"4": 0.07, "8": 0.035, "30": 0.01}, abs=1e-12)
    assert v.copula_choice_sensitivity(100.0, ((4, 103.0), (30, 101.0))).light is Light.GREEN
    assert v.copula_choice_sensitivity(100.0, ((4, 120.0),)).light is Light.RED
    neg = v.copula_choice_sensitivity(100.0, ((30, 101.0), (4, 96.0)))  # the smallest nu is the second pair
    assert neg.metric == pytest.approx(-0.04, abs=1e-12) and neg.statistic == pytest.approx(0.04, abs=1e-12)
    with pytest.raises(ValueError, match="amber"):
        v.copula_choice_sensitivity(100.0, ((4, 107.0),), amber=0.2, red=0.1)


def test_worst_case_bracket_golden():
    r = v.worst_case_bracket(4.0, (5.0, 6.0828, 7.0), 8.0)
    assert r.test_id == "aggregation.worst_case_bracket"
    assert r.statistic == pytest.approx(4.0, abs=1e-12)
    assert r.metric == pytest.approx(4.0 / 7.0, abs=1e-12)
    assert r.light is Light.GREEN and r.extras["thresholds"] == {"tolerance": 1e-9}
    assert v.worst_case_bracket(4.0, (5.0, 6.0828, 7.0), 6.5).light is Light.RED  # the simple sum is above the worst
    assert v.worst_case_bracket(5.5, (5.0, 6.0, 7.0), 8.0).light is Light.RED  # a row below the lower bound
    assert v.worst_case_bracket(9.0, (5.0,), 8.0).light is Light.RED  # lower above worst
    assert v.worst_case_bracket(4.0, (4.0, 8.0), 8.0).light is Light.GREEN  # on the edge is inside
    neg = v.worst_case_bracket(-2.0, (-1.0, 0.0), 1.0)  # a signed lower bound: max(rows) is zero, no ratio
    assert neg.light is Light.GREEN and neg.statistic == pytest.approx(3.0) and neg.metric is None


def test_correlation_stress_golden():
    r = v.correlation_stress(100.0, 105.0)
    assert r.test_id == "aggregation.correlation_stress"
    assert r.statistic == pytest.approx(0.05, abs=1e-12) and r.metric == r.statistic
    assert r.light is Light.GREEN and r.extras["thresholds"] == {"amber": 0.10, "tolerance": 1e-9}
    assert v.correlation_stress(100.0, 112.0).light is Light.AMBER
    assert v.correlation_stress(100.0, 95.0).light is Light.RED
    assert v.correlation_stress(100.0, 100.0).light is Light.GREEN
    assert v.correlation_stress(100.0, 110.0).light is Light.GREEN  # exactly the gate
    with pytest.raises(ValueError, match="amber"):
        v.correlation_stress(100.0, 105.0, amber=-1.0)


def test_diversification_sign_golden():
    r = v.diversification_sign((0.87, 0.9, 0.93))
    assert r.test_id == "aggregation.diversification_sign"
    assert r.light is Light.GREEN
    assert r.statistic == pytest.approx(0.93) and r.metric == pytest.approx(1.0 - 0.87)
    assert r.extras["count_above_one"] == 0 and r.extras["thresholds"] == {"tolerance": 1e-9}
    red = v.diversification_sign((0.87, 1.02))
    assert red.light is Light.RED and red.extras["count_above_one"] == 1
    assert v.diversification_sign((0.87, 1.0 + 1e-12)).light is Light.GREEN
    assert v.diversification_sign((1.0,)).light is Light.GREEN


def test_horizon_confidence_consistency_golden():
    ok = v.horizon_confidence_consistency((1, 1, 1, 1), (0.999,) * 4)
    assert ok.test_id == "aggregation.horizon_confidence_consistency"
    assert ok.statistic == 0.0 and ok.light is Light.GREEN and ok.metric is None
    assert ok.extras["thresholds"] == {"expected_horizon": 1.0, "expected_level": 0.999}
    bad = v.horizon_confidence_consistency((1, 1, 1, 1), (0.999, 0.999, 0.99, 0.999))
    assert bad.statistic == 1.0 and bad.light is Light.RED and bad.extras["offending_indices"] == [2]
    both = v.horizon_confidence_consistency((1, 10 / 250, 1), (0.999, 0.99, 0.999))
    assert both.statistic == 1.0 and both.extras["offending_indices"] == [1]  # one marginal counted once
    two = v.horizon_confidence_consistency((1, 0.5, 1), (0.99, 0.999, 0.999))
    assert two.statistic == 2.0 and two.extras["offending_indices"] == [0, 1]
    with pytest.raises(ValueError, match="same length|length"):
        v.horizon_confidence_consistency((1, 1), (0.999,))


def test_elliptical_mc_agreement_golden():
    r = v.elliptical_mc_agreement(100.4, 0.2, 100.0)
    assert r.test_id == "aggregation.elliptical_mc_agreement"
    assert r.statistic == pytest.approx(2.0, abs=1e-9) and r.light is Light.GREEN
    assert r.metric == pytest.approx(0.004, abs=1e-12)
    assert r.extras["thresholds"] == {"z": 3.0}
    red = v.elliptical_mc_agreement(101.0, 0.2, 100.0)
    assert red.statistic == pytest.approx(5.0, abs=1e-9) and red.light is Light.RED
    assert v.elliptical_mc_agreement(99.4, 0.2, 100.0).light is Light.GREEN  # symmetric
    assert v.elliptical_mc_agreement(99.0, 0.2, 100.0).light is Light.RED
    assert v.elliptical_mc_agreement(100.6, 0.2, 100.0).light is Light.GREEN  # exactly z standard errors


def test_rosenberg_schuermann_reference_golden():
    w = np.array([0.041, 0.264, 0.695])  # the benchmark risk weights (market, credit, operational)
    corr = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.2], [0.2, 0.2, 1.0]])
    gaussian = s.variance_covariance(w, corr)
    assert gaussian == pytest.approx(0.80624, abs=5e-6)
    assert math.sqrt(w @ corr @ w) == pytest.approx(0.80624, abs=5e-6)
    r = v.rosenberg_schuermann_reference(float(w.sum()), gaussian)
    assert r.test_id == "aggregation.rosenberg_schuermann_reference"
    assert r.statistic == pytest.approx(0.2403, abs=5e-5) and r.metric == r.statistic
    assert r.light is Light.GREEN
    assert r.extras["role"] == "reference-not-a-gate"
    assert r.extras["thresholds"] == {"low": 0.30, "high": 0.40}
    assert "outside" in r.notes
    inside = v.rosenberg_schuermann_reference(1.35, 1.0)
    assert inside.light is Light.GREEN and "inside" in inside.notes and inside.statistic == pytest.approx(0.35)
    far = v.rosenberg_schuermann_reference(3.0, 1.0)  # never a gate: green even far outside
    assert far.light is Light.GREEN and "outside" in far.notes
    with pytest.raises(ValueError, match="low"):
        v.rosenberg_schuermann_reference(1.0, 1.0, low=0.5, high=0.4)


def _not_evaluated_cases():
    return [
        v.simple_sum_and_variance_covariance(np.zeros(2), np.eye(2), 0.0),
        v.copula_choice_sensitivity(100.0, ()),
        v.copula_choice_sensitivity(0.0, ((4, 1.0),)),
        v.worst_case_bracket(1.0, (), 2.0),
        v.correlation_stress(0.0, 1.0),
        v.diversification_sign(()),
        v.horizon_confidence_consistency((), ()),
        v.elliptical_mc_agreement(1.0, 0.0, 1.0),
        v.elliptical_mc_agreement(1.0, 0.1, 0.0),
        v.rosenberg_schuermann_reference(1.0, 0.0),
    ]


def test_not_evaluated_carries_no_number():
    for r in _not_evaluated_cases():
        assert r.light is Light.NOT_EVALUATED, r.test_id
        row = r.to_dict()
        assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None, r.test_id
        assert r.notes.startswith("not evaluated:") and len(r.notes) > len("not evaluated: "), r.test_id


def _every_result():
    w = np.array([0.041, 0.264, 0.695])
    return [
        v.simple_sum_and_variance_covariance(EC, np.eye(2), 5.0),
        v.copula_choice_sensitivity(100.0, ((4, 107.0), (8, 103.5))),
        v.worst_case_bracket(4.0, (5.0, 7.0), 8.0),
        v.correlation_stress(100.0, 105.0),
        v.diversification_sign((0.87, 0.9)),
        v.horizon_confidence_consistency((1, 1), (0.999, 0.999)),
        v.elliptical_mc_agreement(100.4, 0.2, 100.0),
        v.rosenberg_schuermann_reference(float(w.sum()), 0.80624),
    ] + _not_evaluated_cases()


def test_every_result_serialises_with_its_reference():
    for r in _every_result():
        row = r.to_dict()
        assert set(row) == ROW_FIELDS
        assert row["test_id"].startswith("aggregation.")
        assert row["reference"].strip() and row["h0"].strip()
        assert row["policy_version"] == "contraste-aggregation-battery 1.0.0"
        assert isinstance(row["extras"].get("thresholds"), dict), r.test_id
        assert row["alpha_amber"] is None and row["alpha_red"] is None
    assert v.correlation_stress(100.0, 105.0).inputs_hash == v.correlation_stress(100.0, 105.0).inputs_hash
    assert v.correlation_stress(100.0, 105.0).inputs_hash != v.correlation_stress(100.0, 106.0).inputs_hash


def test_non_finite_inputs_raise():
    nan, inf = float("nan"), float("inf")
    with pytest.raises(ValueError, match="aggregate"):
        v.simple_sum_and_variance_covariance(EC, np.eye(2), nan)
    with pytest.raises(ValueError, match="ec"):
        v.simple_sum_and_variance_covariance(np.array([3.0, nan]), np.eye(2), 5.0)
    with pytest.raises(ValueError, match="gaussian"):
        v.copula_choice_sensitivity(inf, ((4, 1.0),))
    with pytest.raises(ValueError, match="t_capitals"):
        v.copula_choice_sensitivity(100.0, ((4, nan),))
    with pytest.raises(ValueError, match="lower"):
        v.worst_case_bracket(nan, (1.0,), 2.0)
    with pytest.raises(ValueError, match="rows"):
        v.worst_case_bracket(1.0, (nan,), 2.0)
    with pytest.raises(ValueError, match="stressed"):
        v.correlation_stress(100.0, inf)
    with pytest.raises(ValueError, match="factors"):
        v.diversification_sign((0.9, nan))
    with pytest.raises(ValueError, match="levels"):
        v.horizon_confidence_consistency((1.0,), (nan,))
    with pytest.raises(ValueError, match="mc_capital"):
        v.elliptical_mc_agreement(nan, 0.1, 1.0)
    with pytest.raises(ValueError, match="simple_sum"):
        v.rosenberg_schuermann_reference(inf, 1.0)
