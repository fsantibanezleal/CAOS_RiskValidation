"""The fix round of the Opus review of U14-E1 (engine-ml-api 1.0.0 to 1.1.0): one test per decision and finding.

Q1 the Beta-Binomial null of the conformal coverage test (M-2), Q2 the CBPE row is not_evaluated (M-4), Q3 the reason
codes by signed attribution, Q4 the exact permutation reference of the shift resilience test (M-3), m3 the 12-decimal
rounding of the strict boundaries, and the `threshold_rows()` mapping the integrator writes into `default.json`."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from ml_testkit import assert_not_evaluated, assert_row, sigmoid_predict
from scipy import stats

from riskvalidation.core.policy import Policy, Threshold
from riskvalidation.core.result import Light
from riskvalidation.core.suite import SuiteRun
from riskvalidation.ml.cbpe import cbpe_auc
from riskvalidation.ml.conformal import conformal_coverage
from riskvalidation.ml.explanations import reason_code_overlap
from riskvalidation.ml.generalisation import overfit_gap
from riskvalidation.ml.policy import ML_KEYS, ml_policy, ml_thresholds, threshold_rows
from riskvalidation.ml.robustness import (
    RobustnessCurve,
    ShiftReference,
    shift_development_reference,
    shift_score_rank,
    worst_subsample_performance,
)
from riskvalidation.ml.seeds import SeedDispersion
from riskvalidation.validation import ml as v

# ------------------------------------------------------------------------------------------------ Q1 (M-2)


def _sets_covering(m: int, covered: int) -> tuple[np.ndarray, np.ndarray]:
    sets = np.array([[True, False]] * covered + [[False, True]] * (m - covered))
    return sets, np.zeros(m, dtype=int)


def _betabinom_two_sided(c: int, m: int, k: int, n_cal: int) -> float:
    pmf = stats.betabinom.pmf(np.arange(m + 1), m, k, n_cal + 1 - k)
    return float(pmf[pmf <= pmf[c] * (1.0 + 1e-7)].sum())


def test_conformal_coverage_p_value_is_the_beta_binomial_one_and_the_binomial_stays_beside_it():
    m, n_cal, alpha, covered = 2000, 2000, 0.1, 1830
    sets, y = _sets_covering(m, covered)
    res = conformal_coverage(sets, y, alpha, n_cal=n_cal)
    k = math.ceil((n_cal + 1) * (1 - alpha))
    assert res.n_cal == n_cal and res.k == k == 1801 and res.n == m
    # the scenario: the binomial test alarms, the Beta-Binomial one does not (the quantile is itself random)
    assert res.p_value_binomial == stats.binomtest(covered, m, 0.9, alternative="two-sided").pvalue
    assert res.p_value_binomial < 0.05 < res.p_value
    assert res.p_value == pytest.approx(_betabinom_two_sided(covered, m, k, n_cal), abs=1e-12)
    light = v.conformal_coverage("m", res)
    assert light.light is Light.GREEN and light.p_value == res.p_value  # the light never reads the binomial value
    assert light.extras["p_value_binomial"] == res.p_value_binomial and light.extras["n_cal"] == n_cal


def test_conformal_coverage_requires_n_cal_as_a_keyword():
    sets, y = _sets_covering(10, 9)
    with pytest.raises(TypeError):
        conformal_coverage(sets, y, 0.1)
    with pytest.raises(TypeError):
        conformal_coverage(sets, y, 0.1, 100)  # keyword-only
    with pytest.raises(ValueError, match="n_cal"):
        conformal_coverage(sets, y, 0.1, n_cal=0)


def test_conformal_coverage_is_not_evaluated_when_k_exceeds_n_cal():
    sets = np.ones((20, 2), dtype=bool)  # qhat = +inf: every label, coverage 1 by construction
    res = conformal_coverage(sets, np.zeros(20, dtype=int), 0.1, n_cal=5)  # k = ceil(6 * 0.9) = 6 > 5
    assert res.k == 6 and res.n_cal == 5 and res.coverage == 1.0 and math.isnan(res.p_value)
    r = v.conformal_coverage("m", res)
    assert_not_evaluated(r, "ml.conformal_coverage")
    assert "every label" in r.notes and "coverage is 1" in r.notes


# ------------------------------------------------------------------------------------------------ Q2 (M-4)


def _seeds(sd: float) -> SeedDispersion:
    return SeedDispersion(n_seeds=5, mean=0.7, sd=sd, minimum=0.6, maximum=0.8)


def test_cbpe_estimate_is_always_not_evaluated_and_never_moves_the_worst_light():
    est = cbpe_auc(np.random.default_rng(1).beta(0.8, 2.5, 500))
    r = v.cbpe_estimate("m", est)
    row = assert_row(r, "ml.cbpe_estimate")
    assert r.light is Light.NOT_EVALUATED
    assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None
    assert row["notes"].startswith("not evaluated: an estimate, not a test: CBPE estimates performance without labels")
    assert "06 C.7 l.1052-1058" in row["notes"]
    for a in est.assumptions:
        assert a in row["notes"]  # the four assumptions are in the notes
    assert row["extras"]["estimate"]["estimated_auc"] == est.estimated_auc
    assert row["extras"]["estimate"]["label"] == "estimate"
    labelled = v.cbpe_estimate("m", est, labels_supplied=True)
    assert_not_evaluated(labelled, "ml.cbpe_estimate")
    assert "label is supplied" in labelled.notes
    # SuiteRun.worst(): a green battery plus the CBPE row is green
    pol = ml_policy()
    run = SuiteRun(policy_version=pol.label, results=(v.seed_stability("m", _seeds(0.001), pol), r))
    assert run.worst() is Light.GREEN and run.counts()["green"] == 1 and run.counts()["not_evaluated"] == 1
    assert run.counts()["amber"] == 0


# ------------------------------------------------------------------------------------------------ Q3


def test_reason_codes_are_the_largest_signed_contributions_toward_the_event():
    a1 = np.array([[-9.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    a2 = np.array([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    # signed: refit 1 reasons are columns {1, 2, 3, 4}, refit 2 are {0, 1, 2, 3}: three of four kept. By absolute value
    # both would be {0, 1, 2, 3} and the share would be 1.
    res = reason_code_overlap(np.stack([a1, a2]))
    assert res.mean == pytest.approx(0.75, abs=1e-15)
    # a large negative attribution (a push away from the event) is never a reason
    only_negative = np.array([[-9.0, 0.5, 0.4, 0.3, 0.2, 0.1]])
    assert reason_code_overlap(np.stack([only_negative, only_negative.copy()]), top_k=1).mean == 1.0
    a3 = np.array([[-9.0, 0.5, 0.9, 0.3, 0.2, 0.1]])
    a4 = np.array([[9.0, 0.5, 0.9, 0.3, 0.2, 0.1]])  # the top reason is column 2 against column 0
    assert reason_code_overlap(np.stack([a3, a4]), top_k=1).mean == 0.0
    assert "positive attribution pushes toward" in (reason_code_overlap.__doc__ or "")
    assert "over obligors" in (reason_code_overlap.__doc__ or "")  # m7: the 5th percentile is over obligors


def test_reason_code_ties_are_broken_by_column_order():
    a1 = np.array([[1.0, 1.0, 1.0, 0.0]])
    a2 = np.array([[1.0, 1.0, 0.0, 1.0]])
    # ties by column order: the top two of both refits are columns {0, 1}
    assert reason_code_overlap(np.stack([a1, a2]), top_k=2).mean == 1.0
    a3 = np.array([[0.0, 1.0, 1.0, 1.0]])
    assert reason_code_overlap(np.stack([a1, a3]), top_k=1).mean == 0.0  # {0} against {1}


# ------------------------------------------------------------------------------------------------ Q4 (M-3)


def _shift_data(n: int = 120, n_dev: int = 360, seed: int = 3, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    w = np.array([1.0, 0.5, 0.0])
    x_dev, x_eval = rng.standard_normal((n_dev, 3)), rng.standard_normal((n, 3))
    x_eval[: n // 4, 0] += shift

    def p(x):
        return 1.0 / (1.0 + np.exp(-(-0.3 + x @ w)))

    y_dev = (rng.random(n_dev) < p(x_dev)).astype(int)
    y_eval = (rng.random(n) < p(x_eval)).astype(int)
    return x_dev, y_dev, x_eval, y_eval, sigmoid_predict(w, -0.3)


def test_shift_development_reference_formula_determinism_and_purity():
    x_dev, y_dev, x_eval, y_eval, predict = _shift_data()
    score = shift_score_rank(x_dev, x_eval, 5)
    copy = (x_dev.copy(), y_dev.copy(), x_eval.copy(), y_eval.copy())
    args = (predict, x_dev, y_dev, x_eval, y_eval, score, 0.25)
    ref = shift_development_reference(*args, shift_permutations=19, seed=7)
    again = shift_development_reference(*args, shift_permutations=19, seed=7)
    other = shift_development_reference(*args, shift_permutations=19, seed=8)
    assert isinstance(ref, ShiftReference) and ref == again and ref.null_drops != other.null_drops
    for a, b in zip((x_dev, y_dev, x_eval, y_eval), copy, strict=True):
        np.testing.assert_array_equal(a, b)
    assert ref.n_permutations == 19 and len(ref.null_drops) == 19 and ref.share == 0.25 and ref.labelled
    obs = worst_subsample_performance(predict, x_eval, y_eval, score, 0.25)
    assert ref.observed_drop == obs.auc_full - obs.auc
    ge = sum(d >= ref.observed_drop for d in ref.null_drops)
    assert ref.p_value == (1 + ge) / 20  # (1 + #{null >= observed}) / (R + 1)
    assert ref.enough_development and ref.n_undefined == 0
    assert 1 / 20 <= ref.p_value <= 1.0


def test_shift_development_reference_default_is_199_and_without_development_labels_is_not_evaluated():
    assert inspect.signature(shift_development_reference).parameters["shift_permutations"].default == 199
    x_dev, _, x_eval, y_eval, predict = _shift_data(n=120, n_dev=240)
    score = shift_score_rank(x_dev, x_eval, 5)
    ref = shift_development_reference(predict, x_dev, None, x_eval, y_eval, score, 0.25, shift_permutations=9, seed=1)
    assert not ref.labelled and ref.null_drops == () and math.isnan(ref.p_value)
    obs = worst_subsample_performance(predict, x_eval, y_eval, score, 0.25)
    r = v.shift_resilience("m", obs, ref)
    assert_not_evaluated(r, "ml.shift_resilience")
    assert "development labels" in r.notes
    with pytest.raises(ValueError, match="shift_permutations"):
        shift_development_reference(predict, x_dev, None, x_eval, y_eval, score, 0.25, shift_permutations=0)


def test_shift_development_reference_needs_twice_the_evaluation_rows_in_development():
    x_dev, y_dev, x_eval, y_eval, predict = _shift_data(n=120, n_dev=200)  # 200 < 2 * 120
    score = shift_score_rank(x_dev, x_eval, 5)
    ref = shift_development_reference(predict, x_dev, y_dev, x_eval, y_eval, score, 0.25, shift_permutations=9)
    assert not ref.enough_development and ref.null_drops == () and math.isnan(ref.p_value)
    obs = worst_subsample_performance(predict, x_eval, y_eval, score, 0.25)
    r = v.shift_resilience("m", obs, ref)
    assert_not_evaluated(r, "ml.shift_resilience")
    assert "twice" in r.notes


def test_shift_development_null_holds_no_evaluation_row():
    """The null drops come from development rows only: a development set with a huge outlier row that the evaluation
    set lacks must still give identical results when the evaluation set changes (only its size enters)."""
    x_dev, y_dev, x_eval, y_eval, predict = _shift_data()
    score = shift_score_rank(x_dev, x_eval, 5)
    a = shift_development_reference(predict, x_dev, y_dev, x_eval, y_eval, score, 0.25, shift_permutations=9, seed=2)
    b = shift_development_reference(
        predict, x_dev, y_dev, x_eval + 5.0, y_eval, score, 0.25, shift_permutations=9, seed=2
    )
    assert a.null_drops == b.null_drops


def test_shift_resilience_light_reads_the_permutation_p_value():
    x_dev, _, x_eval, y_eval, predict = _shift_data(n=120)
    score = shift_score_rank(x_dev, x_eval, 5)
    obs = worst_subsample_performance(predict, x_eval, y_eval, score, 0.25)
    drop = obs.auc_full - obs.auc
    nulls = (0.01, 0.02, 0.03)

    def ref(p):
        return ShiftReference(
            observed_drop=drop, null_drops=nulls, p_value=p, n_permutations=len(nulls), share=0.25, labelled=True
        )

    for p, light in (
        (0.5, Light.GREEN),
        (0.05, Light.GREEN),
        (0.0499, Light.AMBER),
        (0.01, Light.AMBER),
        (0.0099, Light.RED),
        (0.005, Light.RED),
    ):
        r = v.shift_resilience("m", obs, ref(p))
        assert r.light is light, p
        row = assert_row(r, "ml.shift_resilience")
        assert row["p_value"] == p and row["statistic"] == drop and row["metric"] == drop
        assert row["alpha_amber"] == 0.05 and row["alpha_red"] == 0.01
        assert row["extras"]["null_mean"] == pytest.approx(0.02)
        assert row["extras"]["null_p95"] == pytest.approx(np.percentile(nulls, 95))
        assert row["extras"]["shift_permutations"] == 3
    with pytest.raises(ValueError, match="observed"):
        v.shift_resilience("m", obs, ShiftReference(drop + 0.1, (0.0,), 0.5, 1, 0.25, True))
    empty = Policy("p", "1", "no ml keys", {})
    assert_not_evaluated(v.shift_resilience("m", obs, ref(0.5), empty), "ml.shift_resilience")


def test_the_shift_resilience_row_is_a_p_value_row():
    t = ml_thresholds()["ml.shift_resilience"]
    assert (t.kind, t.amber, t.red) == ("p_value", 0.05, 0.01) and t.source.strip()


# ------------------------------------------------------------------------------------------------ m3


def test_strict_boundaries_hold_on_decimal_equal_gaps():
    assert overfit_gap(0.80, 0.75, 0.05).flag is False  # 0.80 - 0.75 = 0.050000000000000044 in floats
    assert overfit_gap(0.85, 0.80, 0.05).flag is False  # 0.04999999999999993
    assert overfit_gap(0.80, 0.75, 0.05).gap == 0.05 and overfit_gap(0.85, 0.80, 0.05).gap == 0.05
    assert overfit_gap(0.8501, 0.80, 0.05).flag is True
    for train, test in ((0.80, 0.75), (0.85, 0.80)):
        r = v.overfit("m", overfit_gap(train, test, 0.05))
        assert r.light is Light.GREEN and r.metric == 0.05
    for a, b in ((0.80, 0.78), (0.85, 0.83), (0.90, 0.88)):  # a loss of exactly 0.02
        c = RobustnessCurve(grid=(0.0, 0.5), auc=(a, b), brier=(0.1, 0.1), auc_delta=(0.0, b - a))
        assert v.robustness("m", c, 0.02, 0.5).light is Light.GREEN, (a, b)
    c = RobustnessCurve(grid=(0.0, 0.5), auc=(0.8, 0.7799), brier=(0.1, 0.1), auc_delta=(0.0, -0.0201))
    assert v.robustness("m", c, 0.02, 0.5).light is Light.AMBER


# ------------------------------------------------------------------------------------------------ the mapping


def test_threshold_rows_is_the_mapping_for_default_json():
    rows = threshold_rows()
    assert set(rows) == set(ML_KEYS)
    for key, row in rows.items():
        t = ml_thresholds()[key]
        assert row == {"kind": t.kind, "amber": t.amber, "red": t.red, "source": t.source}
        assert isinstance(row["amber"], float) and isinstance(row["red"], float) and row["source"].strip()
    assert rows["ml.robustness"]["amber"] == math.nextafter(0.02, math.inf)
    assert rows["ml.shift_resilience"]["kind"] == "p_value"
    assert threshold_rows(perturbation_tolerance=0.01)["ml.robustness"]["amber"] == math.nextafter(0.01, math.inf)
    assert Threshold(**rows["ml.overfit"]).kind == "upper_bad"


def test_draws_with_a_single_class_worst_subsample_are_left_out_and_counted(monkeypatch):
    from riskvalidation.ml import robustness as rb

    x_dev, y_dev, x_eval, y_eval, predict = _shift_data(n=120, n_dev=240)
    score = shift_score_rank(x_dev, x_eval, 5)
    real, calls = rb.worst_subsample_performance, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1  # call 1 is the observed subsample; then every second permutation has no AUC
        if calls["n"] > 1 and calls["n"] % 2 == 1:
            raise rb.SingleClassSubsampleError("single class")
        return real(*args, **kwargs)

    monkeypatch.setattr(rb, "worst_subsample_performance", flaky)
    ref = rb.shift_development_reference(predict, x_dev, y_dev, x_eval, y_eval, score, 0.25, shift_permutations=10)
    assert ref.n_undefined == 5 and len(ref.null_drops) == 5 and ref.n_permutations == 10
    ge = sum(d >= ref.observed_drop for d in ref.null_drops)
    assert ref.p_value == (1 + ge) / 6
