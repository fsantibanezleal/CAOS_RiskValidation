"""U16 second fix round (review N1, N2, N5): a `Threshold` hashes, pickles and deep-copies as in 0.9.0; two that
give different lights are never equal; `Policy.classify` never returns a wrong light for a rule or keyed entry."""

from __future__ import annotations

import copy
import pickle

import pytest

from riskvalidation.core.policy import (
    Policy,
    Threshold,
    default_policy,
    packaged_labels,
    packaged_policy,
)
from riskvalidation.core.result import Light


def _all_thresholds():
    for label in packaged_labels():
        for test_id, th in packaged_policy(label).thresholds.items():
            yield label, test_id, th


def test_threshold_of_0_9_0_is_hashable_picklable_and_deepcopyable():
    th = Threshold("upper_bad", 0.1, 0.2, "source")
    assert hash(th) == hash(Threshold("upper_bad", 0.1, 0.2, "source"))
    assert pickle.loads(pickle.dumps(th)) == th
    assert copy.deepcopy(th) == th
    assert len({th, Threshold("upper_bad", 0.1, 0.2, "source")}) == 1


@pytest.mark.parametrize("label", packaged_labels())
def test_every_packaged_threshold_and_policy_hashes_pickles_and_deep_copies(label):
    policy = packaged_policy(label)
    for test_id, th in policy.thresholds.items():
        assert isinstance(hash(th), int), test_id
        assert pickle.loads(pickle.dumps(th)) == th, test_id
        assert copy.deepcopy(th) == th, test_id
    again = pickle.loads(pickle.dumps(policy))
    assert isinstance(again, Policy) and again.label == policy.label
    assert dict(again.thresholds) == dict(policy.thresholds)
    assert copy.deepcopy(policy).thresholds == policy.thresholds
    assert len(set(policy.thresholds.values())) >= 1


def test_the_parameters_and_cases_of_a_threshold_are_immutable():
    th = packaged_policy("contraste-portfolio-battery 1.0.0").threshold("portfolio.euler_additivity")
    with pytest.raises(TypeError):
        th.cases["new"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        th.cases["kernel-var"]["tolerance"] = 1.0  # type: ignore[index]
    p = default_policy().threshold("fair.demographic_parity")
    assert p is not None
    with pytest.raises(TypeError):
        p.parameters["alpha_amber"] = 0.5  # type: ignore[index]


def test_two_thresholds_that_give_different_lights_are_never_equal():
    base = packaged_policy("contraste-aggregation-battery 1.0.0").threshold("aggregation.correlation_stress")
    other = Threshold(
        base.kind, base.amber, base.red, base.source, parameters={"amber": 0.1, "tolerance": 0.5}, rule=base.rule
    )
    assert base.parameters != other.parameters
    row = {"statistic": -0.2}
    assert base.classify_row(row) is not other.classify_row(row)
    assert base != other and hash(base) != hash(other)
    # same pair, same rule, same parameters, same cases: equal whatever the source says about them
    same = Threshold(base.kind, base.amber, base.red, base.source, parameters=dict(base.parameters), rule=base.rule)
    assert base == same
    keyed = packaged_policy("contraste-portfolio-battery 1.0.0").threshold("portfolio.euler_additivity")
    assert keyed != Threshold(keyed.kind, keyed.amber, keyed.red, keyed.source, rule=keyed.rule)  # no cases


def test_ml_policy_with_the_default_arguments_replaces_nothing_and_a_changed_one_is_labelled():
    from riskvalidation.ml.policy import ml_policy

    base = default_policy()
    same = ml_policy()
    assert {k: (t.kind, t.amber, t.red) for k, t in same.thresholds.items()} == {
        k: (t.kind, t.amber, t.red) for k, t in base.thresholds.items()
    }
    assert same.label == base.label
    custom = ml_policy(overfit_tolerance=0.10)
    assert custom.label == base.label + "+custom"


# ---- N2: classify never returns a wrong light ------------------------------------------------------------------------

KEYED_OR_RULE = [
    (label, test_id)
    for label, test_id, th in _all_thresholds()
    if th.rule or th.by
]


def test_the_cases_of_n2_from_the_review_give_the_row_light():
    market = packaged_policy("contraste-market-battery 1.0.0")
    row = {"statistic": 20, "segment": "alpha=0.025"}
    assert market.classify_row("market.desk_backtest", row)[0] is Light.GREEN  # limit 30 at 97.5 percent
    # the old two-cut-point reading said red; classify with the row says green, without the row it refuses
    assert market.classify("market.desk_backtest", metric=20, row=row)[0] is Light.GREEN
    with pytest.raises(ValueError, match="classify_row"):
        market.classify("market.desk_backtest", metric=20)
    ifrs9 = packaged_policy("contraste-ifrs9-battery 1.0.0")
    with pytest.raises(ValueError, match="classify_row"):
        ifrs9.classify("ifrs9.ecl_backtest", metric=0.6)
    assert ifrs9.classify("ifrs9.ecl_backtest", metric=0.6, row={"metric": 0.6})[0] is Light.AMBER
    stress = packaged_policy("contraste-stress-battery 1.0.0")
    assert stress.classify("stress.scenario_severity", metric=0.5, row={"statistic": 0.5})[0] is Light.AMBER  # no red


@pytest.mark.parametrize("label,test_id", KEYED_OR_RULE)
def test_classify_never_answers_a_rule_or_keyed_entry_without_its_row(label, test_id):
    policy = packaged_policy(label)
    with pytest.raises(ValueError, match="classify_row"):
        policy.classify(test_id, p_value=0.5, metric=0.5)
    with pytest.raises(ValueError, match="classify_row"):
        policy.threshold(test_id).classify(0.5)


def test_plain_entries_classify_as_before_and_the_boundary_is_stated():
    plain = next(t for t in default_policy().thresholds.values() if t.kind == "upper_bad" and not t.rule)
    assert plain.classify(plain.amber) in (Light.AMBER, Light.RED)  # no rule, upper_bad: amber starts AT amber
    # under the rule upper_le a value AT amber is green: the two meanings are separate and documented in docs/policies.md
    upper = packaged_policy("contraste-liquidity-battery 1.0.0").threshold("liq.nlp_identity")
    assert upper.classify_row({"statistic": upper.amber}) is Light.GREEN
