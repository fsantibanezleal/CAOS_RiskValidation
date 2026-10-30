"""E15 (BS-15): e-process sequential monitors for VaR hits, defaults against predicted PD by grade and a battery.

The defining properties are tested directly: the exact null expectation of every e-process by full enumeration (so the
Ville bound is a theorem of the code, not a hope), the closed form of a one-point alternative, the anytime-valid size
under optional stopping against the repeated Kupiec rule (which exceeds it), and the planted-defect detection."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import special, stats

from riskvalidation.core.result import Light
from riskvalidation.generators import VarHits
from riskvalidation.validation.sequential import (
    EProcessResult,
    GradeEProcessResult,
    anytime_p_value,
    detection_time,
    eprocess_battery,
    pd_grade_eprocess,
    var_hit_clustering_eprocess,
    var_hit_eprocess,
)


def _enumerate_hits(t: int):
    for bits in itertools.product([0, 1], repeat=t):
        yield np.array(bits, dtype=bool)


def _null_prob(bits: np.ndarray, alpha: float) -> float:
    k = int(bits.sum())
    return alpha**k * (1.0 - alpha) ** (bits.size - k)


# ---------------------------------------------------------------------------------------------------- the math


def test_one_point_alternative_is_the_likelihood_ratio_closed_form():
    alpha, p = 0.05, 0.15
    hits = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    res = var_hit_eprocess(hits, alpha, multipliers=[p / alpha])
    k = np.cumsum(hits)
    t = np.arange(1, hits.size + 1)
    expected = k * math.log(p / alpha) + (t - k) * math.log((1 - p) / (1 - alpha))
    np.testing.assert_allclose(res.log_e, expected, rtol=0, atol=1e-12)


def test_var_hit_eprocess_has_mean_one_under_the_null_at_every_time():
    alpha, t = 0.2, 10
    seqs = list(_enumerate_hits(t))
    probs = np.array([_null_prob(s, alpha) for s in seqs])
    assert probs.sum() == pytest.approx(1.0)
    for direction in ("two-sided", "upper", "lower"):
        paths = np.array([np.exp(var_hit_eprocess(s, alpha, direction=direction).log_e) for s in seqs])
        np.testing.assert_allclose(probs @ paths, np.ones(t), rtol=0, atol=1e-10)


def test_clustering_eprocess_has_mean_one_under_the_null_at_every_time():
    alpha, t = 0.2, 10
    seqs = list(_enumerate_hits(t))
    probs = np.array([_null_prob(s, alpha) for s in seqs])
    paths = np.array([np.exp(var_hit_clustering_eprocess(s, alpha).log_e) for s in seqs])
    np.testing.assert_allclose(probs @ paths, np.ones(t), rtol=0, atol=1e-10)


def test_clustering_eprocess_ignores_a_hit_rate_that_is_right_but_reacts_to_hit_after_hit():
    alpha = 0.1
    rng = np.random.default_rng(3)
    iid = rng.random(400) < alpha
    clustered = np.zeros(400, dtype=bool)
    clustered[::20] = True
    clustered[1::20] = True  # a hit is followed by a hit: pairs, the rate stays 0.1
    assert clustered.mean() == pytest.approx(alpha)
    assert var_hit_clustering_eprocess(iid, alpha).max_log_e < math.log(20)
    assert var_hit_clustering_eprocess(clustered, alpha).rejected
    assert not var_hit_eprocess(clustered, alpha).rejected  # the rate monitor sees nothing: it is the right rate


def test_pd_one_point_alternative_is_the_binomial_likelihood_ratio_closed_form():
    pd = np.array([0.02, 0.10])
    n = np.array([[100, 50], [120, 60], [90, 70]])
    d = np.array([[3, 9], [2, 11], [4, 8]])
    res = pd_grade_eprocess(d, n, pd, multipliers=[2.0], combine="product")
    m = 2.0
    per = np.cumsum(d * math.log(m) + (n - d) * np.log((1 - m * pd) / (1 - pd)), axis=0)
    np.testing.assert_allclose(res.grade_log_e, per, atol=1e-12)
    np.testing.assert_allclose(res.log_e, per.sum(axis=1), atol=1e-12)


def test_pd_eprocess_has_mean_one_under_the_null_for_both_combinations():
    pd = np.array([0.1, 0.3])
    n = np.array([[3, 2], [2, 4]])
    outcomes = []
    for d in itertools.product(*[range(int(v) + 1) for v in n.ravel()]):
        d = np.array(d, dtype=float).reshape(n.shape)
        prob = float(np.prod(stats.binom.pmf(d, n, pd)))
        outcomes.append((d, prob))
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)
    for combine in ("product", "average"):
        mean = np.zeros(2)
        for d, prob in outcomes:
            mean += prob * np.exp(pd_grade_eprocess(d, n, pd, combine=combine).log_e)
        np.testing.assert_allclose(mean, np.ones(2), atol=1e-10)


def test_pd_eprocess_with_a_pd_that_changes_by_step_has_mean_one_under_the_null():
    pd = np.array([[0.1, 0.3], [0.2, 0.05]])
    n = np.array([[3, 2], [2, 4]])
    mean = np.zeros(2)
    for d in itertools.product(*[range(int(v) + 1) for v in n.ravel()]):
        d = np.array(d, dtype=float).reshape(n.shape)
        mean += float(np.prod(stats.binom.pmf(d, n, pd))) * np.exp(pd_grade_eprocess(d, n, pd).log_e)
    np.testing.assert_allclose(mean, np.ones(2), atol=1e-10)
    one_point = pd_grade_eprocess(np.array([[1.0, 0.0], [0.0, 1.0]]), n, pd, multipliers=[2.0])
    m = 2.0
    d = np.array([[1.0, 0.0], [0.0, 1.0]])
    step = d * math.log(m) + (n - d) * np.log((1 - m * pd) / (1 - pd))
    np.testing.assert_allclose(one_point.grade_log_e, np.cumsum(step, axis=0), atol=1e-12)


def test_pd_default_weights_are_jeffreys_type_and_normalised():
    res = pd_grade_eprocess(np.zeros((1, 1)), np.ones((1, 1)) * 10, np.array([0.05]))
    w = np.asarray(res.extras["weights"][0])
    m = np.asarray(res.extras["multipliers"][0])
    assert w.sum() == pytest.approx(1.0)
    jeffreys = np.sqrt(m * 0.05 / (1.0 - m * 0.05))
    np.testing.assert_allclose(w, jeffreys / jeffreys.sum(), rtol=1e-12)
    assert 1.0 not in m


def test_battery_average_and_product_combine_logs():
    a = var_hit_eprocess(np.array([0, 1, 0, 1, 1, 0], dtype=bool), 0.3)
    b = var_hit_clustering_eprocess(np.array([0, 1, 1, 0, 1, 0], dtype=bool), 0.3)
    prod = eprocess_battery({"a": a, "b": b}, combine="product")
    avg = eprocess_battery({"a": a, "b": b}, combine="average")
    np.testing.assert_allclose(prod.log_e, a.log_e + b.log_e, atol=1e-12)
    np.testing.assert_allclose(avg.log_e, np.logaddexp(a.log_e, b.log_e) - math.log(2.0), atol=1e-12)
    assert set(prod.extras["components"]) == {"a", "b"}


def test_battery_average_of_the_same_series_monitors_has_mean_one_under_the_null():
    alpha, t = 0.2, 8
    seqs = list(_enumerate_hits(t))
    probs = np.array([_null_prob(s, alpha) for s in seqs])
    paths = []
    for s in seqs:
        comps = {"rate": var_hit_eprocess(s, alpha), "cluster": var_hit_clustering_eprocess(s, alpha)}
        paths.append(np.exp(eprocess_battery(comps, combine="average").log_e))
    np.testing.assert_allclose(probs @ np.array(paths), np.ones(t), atol=1e-10)


def test_detection_time_and_anytime_p_value():
    path = np.log([1.0, 3.0, 25.0, 4.0, 400.0])
    assert detection_time(path, 0.05) == 3
    assert detection_time(path, 0.001) is None
    assert detection_time(path, 0.0025) == 5
    assert anytime_p_value(path) == pytest.approx(1.0 / 400.0)
    assert anytime_p_value(np.log([0.5, 0.2])) == 1.0
    assert anytime_p_value(np.array([])) is None


def test_the_record_maps_to_a_test_result():
    rej = var_hit_eprocess(np.ones(30, dtype=bool), 0.01)
    ok = var_hit_eprocess(np.zeros(30, dtype=bool), 0.01, direction="upper")
    r1, r2 = rej.to_test_result(model_id="m"), ok.to_test_result()
    assert r1.light is Light.RED and r2.light is Light.GREEN
    assert r1.p_value == pytest.approx(rej.anytime_p_value) and r1.model_id == "m" and r1.n == 30
    assert r1.to_dict()["extras"]["rejection_time"] == rej.rejection_time
    assert isinstance(rej, EProcessResult) and rej.rejected and not ok.rejected


def test_the_path_is_read_only_and_deterministic():
    hits = np.random.default_rng(1).random(100) < 0.02
    a, b = var_hit_eprocess(hits, 0.01), var_hit_eprocess(hits, 0.01)
    np.testing.assert_array_equal(a.log_e, b.log_e)
    assert a.inputs_hash == b.inputs_hash and a.inputs_hash
    with pytest.raises(ValueError):
        a.log_e[0] = 1.0


def test_an_upper_monitor_does_not_reward_a_conservative_model_and_a_two_sided_one_does():
    hits = np.zeros(900, dtype=bool)  # alpha = 0.05 but never a hit: the VaR is too high
    assert not var_hit_eprocess(hits, 0.05, direction="upper").rejected
    assert var_hit_eprocess(hits, 0.05, direction="lower").rejected
    assert var_hit_eprocess(hits, 0.05, direction="two-sided").rejected


def test_the_grade_result_reads_each_grade():
    pd = np.array([0.02, 0.05, 0.10])
    n = np.full((6, 3), 400)
    d = np.zeros((6, 3))
    d[:, 1] = 20  # grade 2 defaults at 5 % against a predicted 5 %... too high for 5 %? 20 of 400 is exactly 5 %
    d[:, 2] = 80  # grade 3: 20 % against 10 %
    res = pd_grade_eprocess(d, n, pd, grades=["A", "B", "C"], direction="upper")
    assert isinstance(res, GradeEProcessResult) and res.grades == ("A", "B", "C")
    times = res.grade_rejection_times()
    assert times["C"] == 1 and times["A"] is None and times["B"] is None
    assert res.rejected and res.grade_log_e.shape == (6, 3)


# --------------------------------------------------------------------------------------- the edge cases


@pytest.mark.parametrize("bad", [[0, 1, 2], [0.0, float("nan"), 1.0], [[0, 1], [1, 0]]])
def test_hits_must_be_flags_without_gaps(bad):
    with pytest.raises(ValueError):
        var_hit_eprocess(bad, 0.01)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan")])
def test_alpha_must_be_open_unit(alpha):
    with pytest.raises(ValueError):
        var_hit_eprocess(np.zeros(5, dtype=bool), alpha)


def test_bad_direction_levels_and_multipliers_raise():
    h = np.zeros(5, dtype=bool)
    with pytest.raises(ValueError):
        var_hit_eprocess(h, 0.01, direction="sideways")
    with pytest.raises(ValueError):
        var_hit_eprocess(h, 0.01, alpha_level=1.5)
    with pytest.raises(ValueError):
        var_hit_eprocess(h, 0.01, multipliers=[1.0])
    with pytest.raises(ValueError):
        var_hit_eprocess(h, 0.5, multipliers=[3.0])  # every alternative hit rate would be 1.5: no alternative left


def test_an_empty_series_is_the_not_evaluated_form():
    res = var_hit_eprocess(np.array([], dtype=bool), 0.01)
    assert res.n == 0 and not res.rejected and res.rejection_time is None and res.anytime_p_value is None
    assert res.to_test_result().light is Light.NOT_EVALUATED


def test_pd_inputs_are_checked():
    n = np.full((3, 2), 10)
    d = np.ones((3, 2))
    pd = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        pd_grade_eprocess(d, n, np.array([0.0, 0.2]))  # a zero predicted PD makes any default infinitely surprising
    with pytest.raises(ValueError):
        pd_grade_eprocess(d, n, np.array([0.1, 1.0]))
    with pytest.raises(ValueError):
        pd_grade_eprocess(d, n, np.array([0.1]))  # one PD for two grades
    with pytest.raises(ValueError):
        pd_grade_eprocess(d + 20, n, pd)  # more defaults than obligors
    with pytest.raises(ValueError):
        pd_grade_eprocess(d * float("nan"), n, pd)
    with pytest.raises(ValueError):
        pd_grade_eprocess(d * 0.5, n, pd)  # a fraction of a default
    with pytest.raises(ValueError):
        pd_grade_eprocess(d, n, pd, combine="sum")
    with pytest.raises(ValueError):
        pd_grade_eprocess(d, n[:2], pd)
    ok = pd_grade_eprocess(np.zeros((2, 2)), np.zeros((2, 2)), pd)  # no obligors at risk: nothing learned
    np.testing.assert_allclose(ok.log_e, 0.0, atol=1e-12)
    assert pd_grade_eprocess(np.zeros((0, 2)), np.zeros((0, 2)), pd).to_test_result().light is Light.NOT_EVALUATED


def test_battery_refuses_misaligned_or_empty_components():
    a = var_hit_eprocess(np.zeros(5, dtype=bool), 0.1)
    b = var_hit_eprocess(np.zeros(6, dtype=bool), 0.1)
    with pytest.raises(ValueError):
        eprocess_battery({"a": a, "b": b})
    with pytest.raises(ValueError):
        eprocess_battery({})
    with pytest.raises(ValueError):
        eprocess_battery({"a": a}, combine="max")
    with pytest.raises(TypeError):
        eprocess_battery({"a": np.zeros(5)})  # type: ignore[dict-item]


# ----------------------------------------------------------------- the hypothesis of the plan, at test size


def _kupiec_lr(k: np.ndarray, t: np.ndarray, p: float) -> np.ndarray:
    pi = k / t
    return -2.0 * (special.xlogy(t - k, 1 - p) + special.xlogy(k, p) - special.xlogy(t - k, 1 - pi)
                   - special.xlogy(k, pi))


def _hits(rate_factor: float, t: int, reps: int, seed: int, alpha: float = 0.01, cluster: float | None = None):
    gen = VarHits(t=t, alpha=alpha, rate_factor=rate_factor, cluster=cluster, with_var=False)
    rng = np.random.default_rng(seed)
    return np.array([gen.draw(rng)["hits"] for _ in range(reps)])


def test_size_under_optional_stopping_is_held_where_the_repeated_kupiec_rule_exceeds_it():
    reps, t, alpha = 1500, 500, 0.01
    hits = _hits(1.0, t, reps, seed=20261028)
    eproc = np.mean([var_hit_eprocess(h, alpha).rejected for h in hits])
    days = np.arange(1, t + 1)
    k = np.cumsum(hits, axis=1)
    lr = _kupiec_lr(k.astype(float), days.astype(float), alpha)
    kupiec = np.mean((stats.chi2.sf(lr, 1) < 0.05)[:, 49:].any(axis=1))  # looked at every day from day 50
    mc_error = 3.0 * math.sqrt(0.05 * 0.95 / reps)
    assert eproc <= 0.05 + mc_error, eproc
    assert kupiec > 0.05 + mc_error, kupiec


def test_a_planted_four_fold_hit_rate_is_detected_inside_the_250_day_window():
    reps = 300
    hits = _hits(4.0, 250, reps, seed=7)
    times = [detection_time(var_hit_eprocess(h, 0.01).log_e) for h in hits]
    detected = [x for x in times if x is not None]
    assert len(detected) / reps > 0.6
    assert np.median(detected) < 250


def test_planted_clustering_is_detected_by_the_clustering_monitor_at_the_right_rate():
    reps = 200
    hits = _hits(1.0, 750, reps, seed=11, alpha=0.05, cluster=0.5)
    cl = np.mean([var_hit_clustering_eprocess(h, 0.05).rejected for h in hits])
    assert cl > 0.6
