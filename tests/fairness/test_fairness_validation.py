"""RV-1473 to RV-1477: the `fair.` validation tests (contract `engine-fairness-api`, module validation.fairness). Each
returns one `TestResult`; the rows serialise into the fields of `common/test-result`."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.core.result import Light, TestResult
from riskvalidation.fairness.conditional import Effect
from riskvalidation.fairness.metrics import (
    auc_by_group as auc_by_group_metric,
)
from riskvalidation.fairness.metrics import (
    calibration_within_groups as calibration_metric,
)
from riskvalidation.fairness.metrics import (
    equalized_odds_difference,
    group_selection_rates,
)
from riskvalidation.fairness.replication import Replication
from riskvalidation.fairness.situation import SituationResult
from riskvalidation.validation import fairness as fv

ROW_FIELDS = {"test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
              "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras"}


def rates_for(**counts: tuple[int, int]):
    decision, group = [], []
    for g, (x, n) in counts.items():
        decision += [1] * x + [0] * (n - x)
        group += [g] * n
    return group_selection_rates(np.array(decision), np.array(group))


def effect(coef: float, se: float = 0.1, level: float = 0.95) -> Effect:
    z = float(stats.norm.ppf(0.5 + level / 2))
    return Effect(coefficient=coef, odds_ratio=math.exp(coef), lo=coef - z * se, hi=coef + z * se, level=level)


def outcome_table(tpr, fpr, n_pos=200, n_neg=200):
    """y, decision, group with the true and false positive rate of each group exact (two groups, a and b)."""
    ys, ds, gs = [], [], []
    for g, t, f in zip(("a", "b"), tpr, fpr, strict=True):
        tp, fp = round(t * n_pos), round(f * n_neg)
        ys += [1] * n_pos + [0] * n_neg
        ds += [1] * tp + [0] * (n_pos - tp) + [1] * fp + [0] * (n_neg - fp)
        gs += [g] * (n_pos + n_neg)
    return np.array(ys), np.array(ds), np.array(gs)


# ---------------------------------------------------------------------------------------------------- RV-1473


def test_demographic_parity_result():
    r = fv.demographic_parity("m1", rates_for(a=(70, 100), b=(60, 100)), "a")
    assert r.test_id == "fair.demographic_parity" and r.model_id == "m1"
    assert r.metric == pytest.approx(0.10, abs=1e-15) and r.light is Light.AMBER
    assert r.p_value is None
    assert r.extras["differences"] == {"a": 0.0, "b": pytest.approx(-0.10, abs=1e-15)}
    assert r.extras["thresholds"]["amber"] == 0.05 and r.extras["thresholds"]["strict"] is True
    assert r.reference and r.h0 and r.alternative
    green = fv.demographic_parity("m1", rates_for(a=(55, 100), b=(50, 100)), "b")
    assert green.light is Light.GREEN and green.metric == 0.05
    never_red = fv.demographic_parity("m1", rates_for(a=(100, 100), b=(0, 100)), "a")
    assert never_red.metric == 1.0 and never_red.light is Light.AMBER  # an indicator: amber above the tolerance
    three = fv.demographic_parity("m1", rates_for(a=(50, 100), b=(52, 100), c=(40, 100)), "a")
    assert three.metric == pytest.approx(0.10, abs=1e-15)  # the largest absolute difference, across the groups
    assert three.n == 300 and three.n_events == 142
    only_one = fv.demographic_parity("m1", rates_for(a=(5, 10)), "a")
    assert only_one.light is Light.NOT_EVALUATED


def test_adverse_impact_result_flags_on_exact_counts():
    exact = fv.adverse_impact("m", rates_for(h=(75, 100), g=(60, 100)), "h")
    assert exact.test_id == "fair.adverse_impact" and exact.light is Light.GREEN  # exactly 4/5 does not flag
    flagged = fv.adverse_impact("m", rates_for(h=(75, 100), g=(5999, 10000)), "h")
    assert flagged.light is Light.AMBER
    assert flagged.extras["flagged_groups"] == ["g"]
    assert fv.adverse_impact("m", rates_for(h=(75, 100), g=(6001, 10000)), "h").light is Light.GREEN
    scenario = fv.adverse_impact("m", rates_for(white=(750, 1000), asian=(900, 1000), black=(650, 1000)), "white")
    assert scenario.light is Light.AMBER and scenario.extras["flagged_groups"] == ["black"]
    assert scenario.metric == pytest.approx(650 / 900, abs=1e-12)  # the lowest ratio to the highest-rate group
    assert scenario.extras["parity_ratio_to_reference"]["black"] == pytest.approx(650 / 750, abs=1e-12)
    assert scenario.extras["highest_rate_group"] == "asian"
    assert scenario.extras["is_credit_law_threshold"] is False
    for text in (scenario.reference, scenario.notes):
        assert "employment benchmark" in text and "29 CFR 1607.4(D)" in text
    assert "credit" in scenario.notes
    assert scenario.extras["thresholds"]["four_fifths"] == 0.8
    none_favourable = fv.adverse_impact("m", rates_for(a=(0, 10), b=(0, 10)), "a")
    assert none_favourable.light is Light.NOT_EVALUATED


# ---------------------------------------------------------------------------------------------------- RV-1474


def test_two_proportion_z_result_lights():
    for counts, light in (((20, 100, 40, 100), Light.RED), ((30, 100, 45, 100), Light.AMBER),
                          ((30, 100, 32, 100), Light.GREEN)):
        r = fv.two_proportion_z("m", *counts)
        x1, n1, x2, n2 = counts
        pooled = (x1 + x2) / (n1 + n2)
        z = (x1 / n1 - x2 / n2) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        assert r.test_id == "fair.two_proportion_z" and r.light is light, counts
        assert r.statistic == pytest.approx(z, abs=1e-12)
        assert r.p_value == pytest.approx(2 * stats.norm.sf(abs(z)), abs=1e-12)
        assert r.metric == pytest.approx(x1 / n1 - x2 / n2, abs=1e-15)
        assert (r.alpha_amber, r.alpha_red) == (0.05, 0.01)
        assert (r.n, r.n_events) == (n1 + n2, x1 + x2)
    degenerate = fv.two_proportion_z("m", 0, 10, 0, 20)
    assert degenerate.p_value == 1.0 and degenerate.light is Light.GREEN and "pooled" in degenerate.notes
    assert fv.two_proportion_z("m", 0, 0, 3, 10).light is Light.NOT_EVALUATED


def test_fisher_exact_result_lights():
    for counts, light in (((2, 40, 15, 40), Light.RED), ((6, 40, 15, 40), Light.AMBER),
                          ((12, 40, 14, 40), Light.GREEN)):
        r = fv.fisher_exact("m", *counts)
        x1, n1, x2, n2 = counts
        _, p = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])
        assert r.test_id == "fair.fisher_exact" and r.light is light, (counts, r.p_value)
        assert r.p_value == pytest.approx(p, abs=1e-12)
        assert r.statistic == pytest.approx((x1 * (n2 - x2)) / ((n1 - x1) * x2), abs=1e-12)
        assert r.metric == pytest.approx(x1 / n1 - x2 / n2, abs=1e-15)
    undefined_or = fv.fisher_exact("m", 5, 10, 0, 10)
    assert undefined_or.statistic is None and undefined_or.p_value is not None and undefined_or.metric == 0.5
    assert fv.fisher_exact("m", 3, 10, 0, 0).light is Light.NOT_EVALUATED


# ---------------------------------------------------------------------------------------------------- RV-1475


def _calibration_data(extra: float, n=8000, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.8, n)
    g = np.where(np.arange(n) % 2 == 0, "a", "b")
    y = (rng.random(n) < np.where(g == "b", p + extra, p)).astype(int)
    return p, y, g


def _auc_data(signal_b: float, n=8000, seed=4):
    rng = np.random.default_rng(seed)
    g = np.where(np.arange(n) % 2 == 0, "a", "b")
    y = (rng.random(n) < 0.3).astype(int)
    score = rng.normal(size=n) + np.where(g == "a", 1.0, signal_b) * y
    return score, y, g


def test_outcome_conditioned_results_need_an_outcome():
    d = np.array([1, 0, 1, 0, 1, 1])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    p = np.array([0.2, 0.4, 0.6, 0.3, 0.5, 0.7])
    results = [
        fv.equalized_odds("m", None, d, g),
        fv.equal_opportunity("m", None, d, g),
        fv.calibration_within_groups("m", p, None, g),
        fv.auc_by_group("m", p, None, g),
    ]
    assert [r.test_id for r in results] == ["fair.equalized_odds", "fair.equal_opportunity",
                                            "fair.calibration_within_groups", "fair.auc_by_group"]
    for r in results:
        assert r.light is Light.NOT_EVALUATED
        assert "outcome" in r.notes
        row = r.to_dict()
        assert row["statistic"] is None and row["p_value"] is None and row["metric"] is None


def test_outcome_conditioned_results_lights_and_omitted_groups():
    for tpr, fpr, light in (((0.80, 0.78), (0.20, 0.19), Light.GREEN), ((0.80, 0.73), (0.20, 0.20), Light.AMBER),
                            ((0.80, 0.80), (0.20, 0.35), Light.RED), ((0.80, 0.70), (0.20, 0.20), Light.RED)):
        y, d, g = outcome_table(tpr, fpr)
        eo = fv.equalized_odds("m", y, d, g)
        assert eo.test_id == "fair.equalized_odds" and eo.light is light, (tpr, fpr, eo.metric)
        assert eo.metric == equalized_odds_difference(y, d, g)
        assert (eo.alpha_amber, eo.alpha_red) == (0.05, 0.10)
        assert eo.extras["rates_by_group"]["a"]["tpr"] == tpr[0]
        tpr_gap = abs(tpr[0] - tpr[1])
        opp = fv.equal_opportunity("m", y, d, g)
        assert opp.test_id == "fair.equal_opportunity" and opp.metric == pytest.approx(tpr_gap, abs=1e-12)
        expect = Light.GREEN if tpr_gap < 0.05 else Light.AMBER if tpr_gap < 0.10 else Light.RED
        assert opp.light is expect, (tpr, opp.metric)
    # the boundary is inclusive: a gap of exactly 0.10 is red, exactly 0.05 amber (exact arithmetic on the counts)
    y, d, g = outcome_table((0.90, 0.80), (0.2, 0.2))
    assert fv.equal_opportunity("m", y, d, g).light is Light.RED
    y, d, g = outcome_table((0.90, 0.85), (0.2, 0.2))
    assert fv.equal_opportunity("m", y, d, g).light is Light.AMBER

    for extra, light in ((0.0, Light.GREEN), (0.07, Light.AMBER), (0.20, Light.RED)):
        p, y, g = _calibration_data(extra)
        cal = fv.calibration_within_groups("m", p, y, g)
        assert cal.test_id == "fair.calibration_within_groups" and cal.light is light, (extra, cal.metric)
        assert cal.metric == max(calibration_metric(p, y, g).values())
    p, y, g = _calibration_data(0.0)
    g2 = np.concatenate([g, np.array(["tiny"] * 4)])
    cal = fv.calibration_within_groups("m", np.concatenate([p, [0.5] * 4]), np.concatenate([y, [0, 1, 0, 1]]), g2)
    assert cal.extras["omitted_groups"] == ["tiny"] and cal.extras["n_omitted_groups"] == 1

    levels = {}
    for signal_b in (1.0, 0.7, 0.2):
        score, y, g = _auc_data(signal_b)
        r = fv.auc_by_group("m", score, y, g)
        aucs = auc_by_group_metric(score, y, g)
        assert r.test_id == "fair.auc_by_group"
        assert r.metric == pytest.approx(max(aucs.values()) - min(aucs.values()), abs=1e-15)
        levels[signal_b] = r.light
    assert levels == {1.0: Light.GREEN, 0.7: Light.AMBER, 0.2: Light.RED}
    score, y, g = _auc_data(1.0)
    one = np.where(g == "b", 0, y)  # group b has a single class: omitted, so one group is left
    assert fv.auc_by_group("m", score, one, g).light is Light.NOT_EVALUATED
    assert fv.auc_by_group("m", score, np.where(g == "b", 0, y), g).extras["omitted_groups"] == ["b"]


# ---------------------------------------------------------------------------------------------------- RV-1476


def test_conditional_and_rate_spread_results():
    controls, missing = ["income", "loan_amount", "dti_bin"], ["credit_score"]
    for fn, metric_of in ((fv.conditional_disparity, lambda e: e.odds_ratio),
                          (fv.rate_spread_disparity, lambda e: e.coefficient)):
        excludes = fn("m", effect(0.4), controls, missing)
        assert excludes.light is Light.AMBER and excludes.metric == metric_of(effect(0.4))
        assert fn("m", effect(-0.4), controls, missing).light is Light.AMBER  # below 0 is also an exclusion
        covers = fn("m", effect(0.05), controls, missing)
        assert covers.light is Light.GREEN
        huge = fn("m", effect(5.0, se=0.01), controls, missing)
        assert huge.light is Light.AMBER  # never red: no finding of a violation
        z = 0.4 / 0.1
        assert excludes.statistic == pytest.approx(z, abs=1e-9)
        assert excludes.p_value == pytest.approx(2 * stats.norm.sf(z), abs=1e-9)
        assert excludes.alpha_amber == pytest.approx(0.05, abs=1e-12)
        assert excludes.extras["controls"] == controls and excludes.extras["missing_controls"] == missing
        assert "credit_score" in excludes.notes and "no legal claim" in excludes.notes
        assert "approximation" in excludes.notes
        assert excludes.extras["thresholds"]["interval_level"] == 0.95
        assert excludes.extras["effect"]["lo"] == effect(0.4).lo
    assert fv.conditional_disparity("m", effect(0.4), controls, []).extras["missing_controls"] == []
    assert fv.conditional_disparity("m", effect(0.4), controls, []).light is Light.AMBER
    assert fv.conditional_disparity("m", effect(0.4), controls, ["x"]).test_id == "fair.conditional_disparity"
    assert fv.rate_spread_disparity("m", effect(0.4), controls, ["x"]).test_id == "fair.rate_spread_disparity"
    bad = Effect(float("nan"), float("nan"), float("nan"), float("nan"), 0.95)
    assert fv.conditional_disparity("m", bad, controls, missing).light is Light.NOT_EVALUATED
    assert fv.conditional_disparity("m", None, controls, missing).light is Light.NOT_EVALUATED  # type: ignore[arg-type]
    flat = Effect(0.2, math.exp(0.2), 0.2, 0.2, 0.95)  # a zero-width interval: the statistic is undefined
    assert fv.conditional_disparity("m", flat, controls, missing).light is Light.NOT_EVALUATED


def test_replication_result():
    same = fv.replication("m", Replication(True, 0.8, 0.5, 0.4))
    assert same.test_id == "fair.replication" and same.light is Light.GREEN and same.metric == 0.8
    flipped = fv.replication("m", Replication(False, -0.5, 0.4, -0.2))
    assert flipped.light is Light.AMBER and flipped.metric == -0.5
    assert flipped.extras["fit_coefficient"] == 0.4 and flipped.extras["confirm_coefficient"] == -0.2
    undefined = fv.replication("m", Replication(False, None, 0.0, 0.2))
    assert undefined.light is Light.NOT_EVALUATED and "fit coefficient" in undefined.notes
    assert same.extras["thresholds"]["rule"]
    assert fv.replication("m", Replication(True, 1.0, 0.3, 0.3)).metric == 1.0


def test_situation_testing_result():
    res = lambda sig, n=100: SituationResult(n, n, sig, 30, 0.0, 0.05)  # noqa: E731
    low = fv.situation_testing("m", res(9))
    assert low.test_id == "fair.situation_testing" and low.light is Light.GREEN and low.metric == 0.09
    assert fv.situation_testing("m", res(19)).light is Light.GREEN
    assert fv.situation_testing("m", res(20)).light is Light.AMBER  # inclusive at the cut point 0.20 (F4)
    allsig = fv.situation_testing("m", res(100))
    assert allsig.light is Light.AMBER and allsig.metric == 1.0  # never red
    assert (allsig.n, allsig.n_events) == (100, 100)
    assert allsig.extras["k"] == 30 and allsig.extras["tau"] == 0.0 and allsig.extras["alpha"] == 0.05
    assert "no legal claim" in allsig.notes
    assert fv.situation_testing("m", SituationResult(0, 0, 0, 30, 0.0, 0.05)).light is Light.NOT_EVALUATED


# ---------------------------------------------------------------------------------------------------- RV-1477


def _all_results() -> list[TestResult]:
    y, d, g = outcome_table((0.8, 0.7), (0.2, 0.3))
    p, yc, gc = _calibration_data(0.07)
    score, ya, ga = _auc_data(0.8)
    return [
        fv.demographic_parity("m", rates_for(a=(70, 100), b=(60, 100)), "a"),
        fv.adverse_impact("m", rates_for(h=(75, 100), g=(59, 100)), "h"),
        fv.two_proportion_z("m", 20, 100, 40, 100),
        fv.fisher_exact("m", 2, 40, 15, 40),
        fv.equalized_odds("m", y, d, g),
        fv.equal_opportunity("m", y, d, g),
        fv.calibration_within_groups("m", p, yc, gc),
        fv.auc_by_group("m", score, ya, ga),
        fv.conditional_disparity("m", effect(0.4), ["income"], ["credit_score"]),
        fv.rate_spread_disparity("m", effect(0.4), ["income"], ["credit_score"]),
        fv.replication("m", Replication(True, 0.8, 0.5, 0.4)),
        fv.situation_testing("m", SituationResult(100, 100, 12, 30, 0.0, 0.05)),
    ]


def _not_evaluated_results() -> list[TestResult]:
    d = np.array([1, 0, 1, 0])
    g = np.array(["a", "a", "b", "b"])
    return [
        fv.demographic_parity("m", rates_for(a=(5, 10)), "a"),
        fv.adverse_impact("m", rates_for(a=(0, 10), b=(0, 10)), "a"),
        fv.two_proportion_z("m", 0, 0, 3, 10),
        fv.fisher_exact("m", 3, 10, 0, 0),
        fv.equalized_odds("m", None, d, g),
        fv.equal_opportunity("m", None, d, g),
        fv.calibration_within_groups("m", np.array([0.2, 0.4, 0.6, 0.8]), None, g),
        fv.auc_by_group("m", np.array([0.2, 0.4, 0.6, 0.8]), None, g),
        fv.conditional_disparity("m", None, ["income"], ["credit_score"]),  # type: ignore[arg-type]
        fv.rate_spread_disparity("m", None, ["income"], ["credit_score"]),  # type: ignore[arg-type]
        fv.replication("m", Replication(False, None, 0.0, 0.1)),
        fv.situation_testing("m", SituationResult(0, 0, 0, 30, 0.0, 0.05)),
    ]


def test_not_evaluated_carries_no_number():
    results = _not_evaluated_results()
    assert len(results) == 12
    for r in results:
        assert r.light is Light.NOT_EVALUATED, r.test_id
        assert r.notes.startswith("not evaluated"), (r.test_id, r.notes)
        assert r.p_value is None and r.metric is None
        row = r.to_dict()
        assert row["statistic"] is None and row["p_value"] is None and row["metric"] is None, r.test_id
        assert r.test_id.startswith("fair.") and r.reference and r.h0 and r.alternative
    # even when the computation succeeded but the policy holds no row, no number travels with a missing colour
    assert {r.test_id for r in results} == {r.test_id for r in _all_results()}


def test_every_result_serialises_with_its_reference():
    results = _all_results()
    assert len({r.test_id for r in results}) == 12
    for r in results:
        row = r.to_dict()
        assert set(row) == ROW_FIELDS
        assert row["test_id"].startswith("fair.")
        for field in ("reference", "h0", "alternative", "policy_version"):
            assert row[field], (r.test_id, field)
        assert "thresholds" in row["extras"], r.test_id
        assert row["light"] in ("green", "amber", "red"), (r.test_id, row["light"])
        json.dumps(row, allow_nan=False)
        assert row["model_id"] == "m"
        assert r.policy_version.startswith("riskvalidation-default")
        assert isinstance(row["inputs_hash"], str)
    # no result states a legal finding: the conditional and the situation results say so
    by_id = {r.test_id: r for r in results}
    for key in ("fair.conditional_disparity", "fair.rate_spread_disparity", "fair.situation_testing"):
        assert "no legal claim" in by_id[key].notes


def test_invalid_inputs_raise_with_the_field_named():
    rates = rates_for(a=(5, 10), b=(2, 10))
    y, d, g = outcome_table((0.8, 0.7), (0.2, 0.3))
    p, yc, gc = _calibration_data(0.0, n=400)
    cases = [
        (lambda: fv.demographic_parity("m", rates, "zzz"), "reference"),
        (lambda: fv.adverse_impact("m", rates, "zzz"), "reference"),
        (lambda: fv.adverse_impact("m", {}, "a"), "rates"),
        (lambda: fv.demographic_parity("m", "not rates", "a"), "rates"),  # type: ignore[arg-type]
        (lambda: fv.two_proportion_z("m", 11, 10, 3, 10), "x1"),
        (lambda: fv.two_proportion_z("m", 3, 10, -1, 10), "x2"),
        (lambda: fv.fisher_exact("m", 3, 10.5, 1, 10), "n1"),
        (lambda: fv.equalized_odds("m", np.where(y == 1, 2, 0), d, g), r"\by\b"),
        (lambda: fv.equalized_odds("m", y, np.where(d == 1, 3, 0), g), "decision"),
        (lambda: fv.equal_opportunity("m", y[:-1], d, g), r"\by\b"),
        (lambda: fv.equal_opportunity("m", y, d, g[:-1]), "group"),
        (lambda: fv.calibration_within_groups("m", np.where(p > 0.5, 1.5, p), yc, gc), r"\bp\b"),
        (lambda: fv.calibration_within_groups("m", p, yc[:-2], gc), r"\by\b"),
        (lambda: fv.auc_by_group("m", np.where(p > 0.5, np.nan, p), yc, gc), "score"),
        (lambda: fv.conditional_disparity("m", "effect", [], []), "effect"),  # type: ignore[arg-type]
        (lambda: fv.conditional_disparity("m", effect(0.1), "income", []), "controls"),  # type: ignore[arg-type]
        (lambda: fv.rate_spread_disparity("m", effect(0.1), [], "x"), "missing_controls"),  # type: ignore[arg-type]
        (lambda: fv.replication("m", "r"), "replication"),  # type: ignore[arg-type]
        (lambda: fv.situation_testing("m", "r"), "result"),  # type: ignore[arg-type]
    ]
    for call, field in cases:
        with pytest.raises(ValueError, match=field):
            call()


def test_demographic_parity_is_never_red_under_a_caller_policy():
    """F1: a caller policy whose red cut is 0.10 never makes the parity indicator red: red is capped at amber."""
    from riskvalidation.core.policy import Policy, Threshold

    caller = Policy("caller", "1", "red cut 0.10", {"fair.demographic_parity": Threshold("upper_bad", 0.05, 0.10)})
    r = fv.demographic_parity("m", rates_for(a=(100, 100), b=(0, 100)), "a", policy=caller)
    assert r.metric == 1.0 and r.light is Light.AMBER
    assert "never red by contract; the policy's red cut is not applied" in r.notes
    assert fv.demographic_parity("m", rates_for(a=(70, 100), b=(60, 100)), "a", policy=caller).light is Light.AMBER


def test_situation_testing_is_never_red_under_a_caller_policy():
    """F1: a caller policy whose red cut is 0.10 never makes situation testing red: red is capped at amber."""
    from riskvalidation.core.policy import Policy, Threshold

    caller = Policy("caller", "1", "red cut 0.10", {"fair.situation_testing": Threshold("upper_bad", 0.05, 0.10)})
    r = fv.situation_testing("m", SituationResult(100, 100, 100, 30, 0.0, 0.05), policy=caller)
    assert r.metric == 1.0 and r.light is Light.AMBER
    assert "never red by contract; the policy's red cut is not applied" in r.notes
