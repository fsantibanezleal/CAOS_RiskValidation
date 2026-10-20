"""RV-1472: the lights of the `fair.` tests come from `core.policy.Policy` rows with the contract's cut points."""

from __future__ import annotations

import math

import numpy as np

from riskvalidation.core.policy import Policy, Threshold, default_policy
from riskvalidation.core.result import Light
from riskvalidation.fairness.metrics import group_selection_rates
from riskvalidation.fairness.policy import FAIR_KEYS, fairness_default_policy
from riskvalidation.validation import fairness as fv

# key -> (kind, amber, red, strict): the table "Policy thresholds" of the contract engine-fairness-api 1.0.0
CONTRACT = {
    "fair.demographic_parity": ("upper_bad", 0.05, 2.0, True),
    "fair.two_proportion_z": ("p_value", 0.05, 0.01, False),
    "fair.fisher_exact": ("p_value", 0.05, 0.01, False),
    "fair.equalized_odds": ("upper_bad", 0.05, 0.10, False),
    "fair.equal_opportunity": ("upper_bad", 0.05, 0.10, False),
    "fair.calibration_within_groups": ("upper_bad", 0.05, 0.10, False),
    "fair.auc_by_group": ("upper_bad", 0.05, 0.10, False),
    "fair.situation_testing": ("upper_bad", 0.20, 2.0, False),
}
NO_ROW = ("fair.adverse_impact", "fair.conditional_disparity", "fair.rate_spread_disparity", "fair.replication")


def test_fair_policy_rows_match_the_contract():
    pol = fairness_default_policy()
    assert set(FAIR_KEYS) == set(CONTRACT)
    for key, (kind, amber, red, strict) in CONTRACT.items():
        th = pol.threshold(key)
        assert th is not None and th.kind == kind, key
        assert th.red == red
        assert th.amber == (math.nextafter(amber, math.inf) if strict else amber), key
        assert th.source, f"{key} has no source text"
    for key in NO_ROW:
        assert pol.threshold(key) is None, f"{key} states no policy threshold"
    # the packaged default policy is kept whole
    base = default_policy()
    for key, th in base.thresholds.items():
        assert pol.threshold(key) == th
    assert pol.policy_id.startswith(base.policy_id)
    assert "fair" in pol.version


def test_strict_row_boundary_is_green():
    th = fairness_default_policy().threshold("fair.demographic_parity")
    assert th.classify(0.05) is Light.GREEN  # a delta equal to the tolerance is not a finding
    assert th.classify(math.nextafter(0.05, math.inf)) is Light.AMBER
    assert th.classify(0.0500001) is Light.AMBER
    # through the result: 55 of 100 against 50 of 100 differs by exactly 0.05 and is green
    d = np.array([1] * 55 + [0] * 45 + [1] * 50 + [0] * 50)
    g = np.array(["a"] * 100 + ["b"] * 100)
    assert fv.demographic_parity("m", group_selection_rates(d, g), "b").light is Light.GREEN
    # the inclusive rows: a metric equal to the amber cut is amber, equal to the red cut is red
    eo = fairness_default_policy().threshold("fair.equalized_odds")
    assert eo.classify(0.05) is Light.AMBER and eo.classify(0.10) is Light.RED and eo.classify(0.0499999) is Light.GREEN


def test_never_red_rows_use_a_sentinel():
    pol = fairness_default_policy()
    for key in ("fair.demographic_parity", "fair.situation_testing"):
        th = pol.threshold(key)
        assert th.red == 2.0
        assert th.classify(1.0) is Light.AMBER  # the largest value the metric can take is never red
    assert Threshold("upper_bad", 0.05, 2.0).classify(1.0) is Light.AMBER


def test_explicit_policy_without_the_key_is_not_evaluated():
    empty = Policy("empty", "1", "no fair keys", {})
    d = np.array([1, 0, 1, 1, 0, 0])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    rates = group_selection_rates(d, g)
    r = fv.demographic_parity("m", rates, "a", policy=empty)
    assert r.light is Light.NOT_EVALUATED and "fair.demographic_parity" in r.notes
    row = r.to_dict()
    assert row["statistic"] is None and row["p_value"] is None and row["metric"] is None
    z = fv.two_proportion_z("m", 5, 10, 8, 10, policy=empty)
    assert z.light is Light.NOT_EVALUATED and z.to_dict()["p_value"] is None
    # a supplied policy that carries the key is used as given
    custom = Policy("c", "9", "strict", {"fair.two_proportion_z": Threshold("p_value", 0.5, 0.1)})
    assert fv.two_proportion_z("m", 5, 10, 8, 10, policy=custom).light is Light.AMBER


def test_fairness_default_policy_returns_the_default_policy_when_it_holds_every_fair_key():
    """F5: once the integrator copies the rows into default.json, every `fair.` row carries the default policy's own
    label (no `+fair-1.0.0` suffix); a policy that lacks any key still gets the rows and the suffix."""
    import riskvalidation.fairness.policy as fp

    base = default_policy()
    full = Policy(base.policy_id, base.version, base.description,
                  {**base.thresholds, **{key: fp.fair_threshold(key) for key in FAIR_KEYS}})
    fp.fairness_default_policy.cache_clear()
    original = fp.default_policy
    fp.default_policy = lambda: full
    try:
        got = fp.fairness_default_policy()
    finally:
        fp.default_policy = original
        fp.fairness_default_policy.cache_clear()
    assert got is full and got.label == full.label and "fair-1.0.0" not in got.label
    assert "fair-1.0.0" in fairness_default_policy().label  # the packaged policy has no `fair.` rows yet


def test_situation_testing_amber_cut_is_0_20():
    """F4: the amber cut of `fair.situation_testing` is 0.20 (measured size 0.039 at n 400, k 30; docs/fairness/06)."""
    th = fairness_default_policy().threshold("fair.situation_testing")
    assert th.amber == 0.20 and th.classify(0.19) is Light.GREEN and th.classify(0.20) is Light.AMBER
