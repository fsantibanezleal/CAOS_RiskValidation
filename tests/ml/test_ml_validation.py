"""RV-1430 to RV-1436: the `ml.` validation tests of `riskvalidation.validation.ml` and the cut points they read
(`riskvalidation.ml.policy`), contract engine-ml-api 1.0.0, section Policy thresholds."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ml_testkit import assert_not_evaluated, assert_row

from riskvalidation.core.policy import Policy, Threshold, default_policy
from riskvalidation.core.result import Light
from riskvalidation.ml.cbpe import CbpeAuc
from riskvalidation.ml.challenger import PairedBrier
from riskvalidation.ml.conformal import CoverageResult
from riskvalidation.ml.dataquality import DataQualityCount
from riskvalidation.ml.explanations import OverlapResult
from riskvalidation.ml.generalisation import OverfitResult, WeakSpot
from riskvalidation.ml.leakage import AucResult, SingleFeatureAuc
from riskvalidation.ml.policy import ML_KEYS, ml_policy, ml_thresholds
from riskvalidation.ml.robustness import RobustnessCurve, ShiftReference, SubsampleResult
from riskvalidation.ml.seeds import SeedDispersion
from riskvalidation.validation import ml as v

NA = math.nextafter


def _curve(delta: float) -> RobustnessCurve:
    return RobustnessCurve(grid=(0.0, 0.5), auc=(0.8, 0.8 + delta), brier=(0.1, 0.1), auc_delta=(0.0, delta))


# ---------------------------------------------------------------------------------------------- the cut points


def test_policy_rows_are_the_contract_table():
    t = ml_thresholds()
    expect = {  # key: (kind, amber, red) with the strict rows one float above their cut
        "ml.robustness": ("upper_bad", NA(0.02, math.inf), NA(0.04, math.inf)),
        "ml.shift_resilience": ("p_value", 0.05, 0.01),
        "ml.seed_stability": ("upper_bad", 0.01, 0.02),
        "ml.reason_code_stability": ("lower_bad", 0.80, 0.60),
        "ml.effect_agreement": ("lower_bad", 0.70, 0.40),
        "ml.additivity": ("upper_bad", NA(1e-9, math.inf), NA(1e-9, math.inf)),
        "ml.overfit": ("upper_bad", NA(0.05, math.inf), NA(0.05, math.inf)),
        "ml.weak_spot": ("upper_bad", NA(2.0, math.inf), NA(1e12, math.inf)),
        "ml.conformal_coverage": ("p_value", 0.05, 0.01),
        "ml.leakage_screen.single_feature_auc": ("upper_bad", NA(0.95, math.inf), NA(0.95, math.inf)),
        "ml.leakage_screen.resemblance_auc": ("upper_bad", NA(0.70, math.inf), NA(0.70, math.inf)),
        "ml.leakage_screen.temporal_order": ("upper_bad", 1.0, 1.0),
        "ml.leakage_screen.entity_overlap": ("upper_bad", 1.0, 1.0),
        "ml.data_quality": ("upper_bad", 0.01, 0.05),
    }
    assert set(t) == set(expect) == set(ML_KEYS)
    for key, (kind, amber, red) in expect.items():
        assert (t[key].kind, t[key].amber, t[key].red) == (kind, amber, red), key
        assert t[key].source.strip(), key  # every threshold states where it comes from
    assert "ml.cbpe_estimate" not in t and "ml.challenger_brier" not in t  # no policy threshold, stated in the contract


def test_ml_policy_fills_only_the_missing_keys_of_the_base_policy():
    base = default_policy()
    pol = ml_policy()
    assert isinstance(pol, Policy) and pol.policy_id == base.policy_id and pol.version == base.version
    for key in base.thresholds:
        assert pol.thresholds[key] == base.thresholds[key]  # the base thresholds are untouched
    for key in ML_KEYS:
        assert key in pol.thresholds
    own = Policy("custom", "9", "x", {"ml.shift_resilience": Threshold("upper_bad", 0.2, 0.4, "mine")})
    merged = ml_policy(own)
    assert merged.thresholds["ml.shift_resilience"].amber == 0.2  # a key the base holds is kept
    assert merged.thresholds["ml.seed_stability"].amber == 0.01
    custom = ml_policy(
        perturbation_tolerance=0.01, overfit_tolerance=0.1, weak_spot_multiple=3.0, additivity_tolerance=1e-6
    )
    assert custom.thresholds["ml.robustness"].amber == NA(0.01, math.inf)
    assert custom.thresholds["ml.robustness"].red == NA(0.02, math.inf)
    assert custom.thresholds["ml.overfit"].amber == NA(0.1, math.inf)
    assert custom.thresholds["ml.weak_spot"].amber == NA(3.0, math.inf)
    assert custom.thresholds["ml.additivity"].amber == NA(1e-6, math.inf)


# ---------------------------------------------------------------------------------------------- robustness


def test_robustness_boundaries_agree_with_the_strict_finding():
    from riskvalidation.ml.robustness import robustness_finding

    cases = [
        (-0.02, Light.GREEN),
        (-0.0200001, Light.AMBER),
        (-0.04, Light.AMBER),
        (-0.0400001, Light.RED),
        (0.0, Light.GREEN),
        (0.03, Light.GREEN),
        (-0.1, Light.RED),
    ]
    for delta, light in cases:
        curve = _curve(delta)
        res = v.robustness("m1", curve, 0.02, 0.5)
        row = assert_row(res, "ml.robustness")
        assert res.light is light, delta
        assert res.statistic == res.metric == -delta
        assert (res.light is not Light.GREEN) == robustness_finding(curve, 0.02, 0.5)  # light and boolean agree
        assert row["model_id"] == "m1" and set(row["extras"]["thresholds"]) >= {"perturbation_tolerance"}
        assert row["extras"]["thresholds"]["perturbation_tolerance"] == 0.02
        assert row["alpha_amber"] == NA(0.02, math.inf) and row["alpha_red"] == NA(0.04, math.inf)
        assert "perturbation_tolerance=0.02" in row["policy_version"]


def test_robustness_policy_argument_tolerance_and_off_grid():
    pol = ml_policy(perturbation_tolerance=0.01)
    assert v.robustness("m", _curve(-0.015), 0.01, 0.5, pol).light is Light.AMBER
    assert v.robustness("m", _curve(-0.015), 0.02, 0.5).light is Light.GREEN
    with pytest.raises(ValueError, match="tolerance"):
        v.robustness("m", _curve(-0.015), 0.02, 0.5, pol)  # a tolerance the policy's cut points do not state
    with pytest.raises(ValueError, match="preset_eps"):
        v.robustness("m", _curve(-0.015), 0.02, 0.75)
    nan_curve = RobustnessCurve(
        grid=(0.0, 0.5), auc=(0.8, float("nan")), brier=(0.1, 0.1), auc_delta=(0.0, float("nan"))
    )
    assert_not_evaluated(v.robustness("m", nan_curve, 0.02, 0.5), "ml.robustness")
    empty = Policy("p", "1", "no ml keys", {})
    assert_not_evaluated(v.robustness("m", _curve(-0.1), 0.02, 0.5, empty), "ml.robustness")


# ---------------------------------------------------------------------------------------------- shift, seeds, codes


def test_shift_resilience_lights():
    """The light reads the permutation p-value (docs/ml/01); the observed drop is the statistic. The full lights and the
    reference are tested in test_ml_review_fixes.py."""
    obs = SubsampleResult(n=100, auc=0.7, brier=0.2, auc_full=0.8, brier_full=0.1)

    def ref(p):
        return ShiftReference(
            observed_drop=0.8 - 0.7, null_drops=(0.01, 0.02), p_value=p, n_permutations=2, share=0.25, labelled=True
        )

    for p, light in ((0.5, Light.GREEN), (0.05, Light.GREEN), (0.049, Light.AMBER), (0.0099, Light.RED)):
        r = v.shift_resilience("m", obs, ref(p))
        assert r.light is light, p
        assert r.statistic == pytest.approx(0.1, abs=1e-12) and r.n == 100 and r.p_value == p
        assert_row(r, "ml.shift_resilience")
    custom = Policy("c", "1", "x", {"ml.shift_resilience": Threshold("p_value", 0.2, 0.1)})
    assert v.shift_resilience("m", obs, ref(0.15), custom).light is Light.AMBER
    assert v.shift_resilience("m", obs, ref(0.05), custom).light is Light.RED


def test_seed_stability_lights():
    def res(sd):
        return v.seed_stability("m", SeedDispersion(n_seeds=5, mean=0.7, sd=sd, minimum=0.6, maximum=0.8))

    for sd, light in (
        (0.0, Light.GREEN),
        (0.0099, Light.GREEN),
        (0.01, Light.AMBER),
        (0.019, Light.AMBER),
        (0.02, Light.RED),
        (0.05, Light.RED),
    ):
        r = res(sd)
        assert r.light is light, sd
        assert r.metric == sd and r.n == 5
        assert_row(r, "ml.seed_stability")


def test_reason_code_stability_and_effect_agreement_lights():
    def rc(mean):
        return v.reason_code_stability("m", OverlapResult(top_k=4, n_refits=3, mean=mean, p05=mean))

    for mean, light in (
        (1.0, Light.GREEN),
        (0.81, Light.GREEN),
        (0.80, Light.AMBER),
        (0.7, Light.AMBER),
        (0.60, Light.RED),
        (0.2, Light.RED),
    ):
        r = rc(mean)
        assert r.light is light, mean
        assert r.metric == mean and r.n == 3
        assert_row(r, "ml.reason_code_stability")
    for rho, light in (
        (0.95, Light.GREEN),
        (0.71, Light.GREEN),
        (0.70, Light.AMBER),
        (0.5, Light.AMBER),
        (0.40, Light.RED),
        (-0.3, Light.RED),
    ):
        r = v.effect_agreement("m", rho, 12)
        assert r.light is light, rho
        assert r.metric == rho and r.n == 12
        assert_row(r, "ml.effect_agreement")
    assert_not_evaluated(v.effect_agreement("m", float("nan"), 12), "ml.effect_agreement")
    assert_not_evaluated(v.effect_agreement("m", 0.9, 1), "ml.effect_agreement")


# ---------------------------------------------------------------------------------------------- additivity, overfit,
# spots


def test_additivity_overfit_and_weak_spot_lights():
    for gap, light in ((0.0, Light.GREEN), (1e-9, Light.GREEN), (1.1e-9, Light.RED), (1e-3, Light.RED)):
        r = v.additivity("m", gap)
        assert r.light is light, gap
        assert r.metric == gap
        assert_row(r, "ml.additivity")
    assert v.additivity("m", 1e-7, 1e-6, ml_policy(additivity_tolerance=1e-6)).light is Light.GREEN
    with pytest.raises(ValueError, match="tolerance"):
        v.additivity("m", 1e-7, 1e-6)
    for gap, flag, light in ((0.05, False, Light.GREEN), (0.0501, True, Light.RED), (-0.1, False, Light.GREEN)):
        r = v.overfit("m", OverfitResult(gap=gap, tolerance=0.05, flag=flag))
        assert r.light is light, gap
        assert r.metric == gap
        assert (r.light is Light.RED) == flag  # the strict boolean and the light agree at the boundary
        assert_row(r, "ml.overfit")
    with pytest.raises(ValueError, match="tolerance"):
        v.overfit("m", OverfitResult(gap=0.1, tolerance=0.07, flag=True))
    none = v.weak_spot("m", [], 2.0, 8)
    assert none.light is Light.GREEN and none.metric == 0.0 and none.n == 8
    assert_row(none, "ml.weak_spot")
    amber = v.weak_spot("m", [WeakSpot(label=3, n=100, mean_error=3.0, ratio=2.14), WeakSpot(1, 50, 2.5, 2.05)], 2.0, 5)
    assert amber.light is Light.AMBER and amber.metric == 2.14  # the worst slice
    assert amber.extras["worst_label"] == 3
    boundary = v.weak_spot("m", [WeakSpot(label=1, n=50, mean_error=2.0, ratio=2.0)], 2.0, 5)
    assert boundary.light is Light.GREEN  # a ratio equal to k is not a weak spot
    assert v.weak_spot("m", [WeakSpot(0, 40, 9.0, 1e6)], 2.0, 5).light is Light.AMBER  # never red
    with pytest.raises(ValueError, match="k"):
        v.weak_spot("m", [], 3.0, 8)  # k the policy does not state


# ---------------------------------------------------------------------------------------------- conformal, leakage, dq


def test_conformal_coverage_lights():
    def res(p, cov=0.85):
        return v.conformal_coverage(
            "m", CoverageResult(n=500, coverage=cov, target=0.9, p_value=p, n_cal=5000, k=4501, p_value_binomial=0.5)
        )

    for p, light in (
        (0.5, Light.GREEN),
        (0.05, Light.GREEN),
        (0.0499, Light.AMBER),
        (0.01, Light.AMBER),
        (0.0099, Light.RED),
        (0.0, Light.RED),
    ):
        r = res(p)
        assert r.light is light, p
        assert r.p_value == p and r.metric == 0.85 and r.n == 500
        assert r.statistic == pytest.approx(0.85 - 0.9, abs=1e-15)  # review minor m1 (mutant M26): coverage - target
        row = assert_row(r, "ml.conformal_coverage")
        assert row["alpha_amber"] == 0.05 and row["alpha_red"] == 0.01


def test_leakage_screen_is_the_worst_of_the_four_screens():
    clean = dict(
        temporal=0,
        single=SingleFeatureAuc(0.55, 1, False),
        overlap=0.0,
        resemblance=AucResult(auc=0.52, variance=1e-4, n=2000),
    )
    r = v.leakage_screen("m", **clean)
    assert r.light is Light.GREEN and r.statistic == 0.0 and r.n == 2000
    assert_row(r, "ml.leakage_screen")
    trips = {
        "temporal": dict(clean, temporal=1),
        "single": dict(clean, single=SingleFeatureAuc(0.951, 2, True)),
        "overlap": dict(clean, overlap=0.01),
        "resemblance": dict(clean, resemblance=AucResult(auc=0.71, variance=1e-4, n=2000)),
    }
    for name, kw in trips.items():
        r = v.leakage_screen("m", **kw)
        assert r.light is Light.RED and r.statistic == 1.0, name
        assert r.extras["screens"][name]["light"] == "red"
        assert_row(r, "ml.leakage_screen")
    edge = v.leakage_screen(
        "m",
        **dict(clean, single=SingleFeatureAuc(0.95, 0, False), resemblance=AucResult(auc=0.70, variance=1e-4, n=10)),
    )
    assert edge.light is Light.GREEN  # strict rows: a value equal to the cut is not a trip
    both = v.leakage_screen("m", **dict(clean, temporal=3, overlap=1.0))
    assert both.light is Light.RED and both.statistic == 2.0
    assert both.extras["screens"]["temporal"]["value"] == 3


def test_data_quality_lights_and_counts():
    def dq(*failed):
        counts = [
            DataQualityCount(check=f"c{i}", n_checked=1000, n_failed=f, share_ok=1.0 - f / 1000.0)
            for i, f in enumerate(failed)
        ]
        return v.data_quality("m", counts)

    for failed, light in (
        ((0, 0), Light.GREEN),
        ((9, 0), Light.GREEN),
        ((10, 0), Light.AMBER),
        ((10, 49), Light.AMBER),
        ((0, 50), Light.RED),
    ):
        r = dq(*failed)
        assert r.light is light, failed
        assert r.metric == pytest.approx(max(failed) / 1000.0, abs=1e-15)
        assert r.n == 2000 and r.n_events == sum(failed)
        assert_row(r, "ml.data_quality")
    assert_not_evaluated(v.data_quality("m", []), "ml.data_quality")


def test_challenger_brier_light_follows_the_paired_interval():
    def ch(diff, lo, hi):
        return v.challenger_brier("m", PairedBrier(diff=diff, lo=lo, hi=hi, n_boot=1000))

    assert ch(0.01, 0.002, 0.02).light is Light.AMBER  # the interval excludes zero with the challenger worse
    assert ch(0.01, -0.002, 0.02).light is Light.GREEN  # it does not exclude zero
    assert ch(0.0, 0.0, 0.0).light is Light.GREEN  # identical forecasts
    assert ch(-0.01, -0.02, -0.002).light is Light.GREEN  # a better challenger is not a finding against the champion
    assert ch(0.01, 0.0, 0.02).light is Light.GREEN  # an interval touching zero does not exclude it
    r = ch(0.01, 0.002, 0.02)
    row = assert_row(r, "ml.challenger_brier")
    assert r.statistic == r.metric == 0.01 and row["extras"]["lo"] == 0.002 and row["extras"]["hi"] == 0.02
    assert row["extras"]["level"] == 0.95 and r.n is None


# ---------------------------------------------------------------------------------------------- invariants


def _ref(drop: float, p: float) -> ShiftReference:
    return ShiftReference(
        observed_drop=drop, null_drops=(0.01, 0.02), p_value=p, n_permutations=2, share=0.25, labelled=True
    )


def test_every_function_is_pure_deterministic_and_serialises():
    calls = [
        lambda: v.robustness("m", _curve(-0.03), 0.02, 0.5),
        lambda: v.shift_resilience("m", SubsampleResult(100, 0.7, 0.2, 0.8, 0.1), _ref(0.1, 0.03)),
        lambda: v.seed_stability("m", SeedDispersion(5, 0.7, 0.015, 0.6, 0.8)),
        lambda: v.reason_code_stability("m", OverlapResult(4, 3, 0.7, 0.5)),
        lambda: v.effect_agreement("m", 0.6, 9),
        lambda: v.additivity("m", 1e-12),
        lambda: v.overfit("m", OverfitResult(0.1, 0.05, True)),
        lambda: v.weak_spot("m", [WeakSpot(1, 40, 3.0, 2.5)], 2.0, 4),
        lambda: v.conformal_coverage("m", CoverageResult(100, 0.85, 0.9, 0.03, 999, 900, 0.02)),
        lambda: v.leakage_screen("m", 0, SingleFeatureAuc(0.6, 0, False), 0.0, AucResult(0.5, 1e-4, 100)),
        lambda: v.data_quality("m", [DataQualityCount("completeness", 10, 1, 0.9)]),
        lambda: v.cbpe_estimate("m", CbpeAuc(0.7, "estimate", ("a", "b", "c", "d"))),
        lambda: v.challenger_brier("m", PairedBrier(0.01, -0.01, 0.02, 100)),
    ]
    for call in calls:
        a, b = call(), call()
        assert a.to_dict() == b.to_dict() and a.light is b.light  # (a == b is False for a NaN statistic)
        assert a.test_id.startswith("ml.") and a.h0 and a.alternative and a.reference
        assert a.inputs_hash and len(a.inputs_hash) == 64
        assert_row(a, a.test_id)
        if a.light is not Light.NOT_EVALUATED:
            assert "thresholds" in a.extras or a.test_id in ("ml.cbpe_estimate", "ml.challenger_brier")


def test_not_evaluated_rows_carry_no_number_for_every_function_with_missing_data():
    nan = float("nan")
    rows = [
        (v.seed_stability("m", SeedDispersion(2, 0.7, nan, 0.6, 0.8)), "ml.seed_stability"),
        (v.shift_resilience("m", SubsampleResult(10, 0.0, 0.2, 0.8, 0.1), _ref(0.8, nan)), "ml.shift_resilience"),
        (v.reason_code_stability("m", OverlapResult(4, 3, nan, nan)), "ml.reason_code_stability"),
        (v.additivity("m", nan), "ml.additivity"),
        (v.overfit("m", OverfitResult(nan, 0.05, False)), "ml.overfit"),
        (v.conformal_coverage("m", CoverageResult(10, 0.9, 0.9, nan, 99, 90, nan)), "ml.conformal_coverage"),
        (v.leakage_screen("m", 0, SingleFeatureAuc(0.6, 0, False), 0.0, AucResult(nan, 0.0, 10)), "ml.leakage_screen"),
        (v.challenger_brier("m", PairedBrier(nan, nan, nan, 10)), "ml.challenger_brier"),
    ]
    for res, tid in rows:
        assert_not_evaluated(res, tid)
        assert res.extras is not None
    # a policy that lacks a key: never a colour
    bare = Policy("bare", "1", "none", {})
    for res, tid in [
        (v.seed_stability("m", SeedDispersion(5, 0.7, 0.05, 0.6, 0.8), bare), "ml.seed_stability"),
        (v.conformal_coverage("m", CoverageResult(10, 0.9, 0.9, 0.5, 99, 90, 0.5), bare), "ml.conformal_coverage"),
        (v.data_quality("m", [DataQualityCount("c", 10, 1, 0.9)], bare), "ml.data_quality"),
    ]:
        assert_not_evaluated(res, tid)
    # a leakage screen with one missing key is not evaluated as a whole
    partial = Policy(
        "p", "1", "x", {k: t for k, t in ml_thresholds().items() if k != "ml.leakage_screen.entity_overlap"}
    )
    assert_not_evaluated(
        v.leakage_screen("m", 0, SingleFeatureAuc(0.6, 0, False), 0.0, AucResult(0.5, 1e-4, 10), partial),
        "ml.leakage_screen",
    )


def test_inputs_are_not_mutated_by_the_validation_functions():
    curve = _curve(-0.03)
    snapshot = (curve.grid, curve.auc, curve.brier, curve.auc_delta)
    v.robustness("m", curve, 0.02, 0.5)
    assert (curve.grid, curve.auc, curve.brier, curve.auc_delta) == snapshot
    counts = [DataQualityCount("c", 10, 1, 0.9)]
    v.data_quality("m", counts)
    assert counts == [DataQualityCount("c", 10, 1, 0.9)]
    spots = np.array([1.0])  # nothing to mutate in a tuple-based API: the arrays of the engine stay untouched
    v.additivity("m", float(spots[0]))
    assert spots[0] == 1.0
