"""RV-002, RV-004, RV-007: policy-driven lights, never hard-coded thresholds."""

from __future__ import annotations

import json

import pytest

from riskvalidation import Light, Policy, Threshold, default_policy, load_policy
from riskvalidation.core.build import make_result


def test_missing_policy_no_light():
    empty = Policy(policy_id="empty", version="0", description="", thresholds={})
    r = make_result("selftest.p", p_value=0.0001, policy=empty)
    assert r.light is Light.NOT_EVALUATED
    assert r.alpha_amber is None and r.alpha_red is None
    assert "no thresholds" in r.notes


def test_policy_version_carried(policy, tmp_path):
    r = make_result("selftest.p", p_value=0.2, policy=policy)
    assert r.policy_version == "test-policy 1.2.3"
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            {
                "policy_id": "file-policy",
                "version": "2026.10.1",
                "thresholds": {"selftest.p": {"kind": "p_value", "amber": 0.10, "red": 0.02}},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_policy(path)
    r2 = make_result("selftest.p", p_value=0.05, policy=loaded)
    assert r2.policy_version == "file-policy 2026.10.1"
    assert r2.light is Light.AMBER  # same p-value, different policy, different light
    assert default_policy().version  # the packaged policy loads


def test_threshold_boundaries():
    p = Threshold(kind="p_value", amber=0.05, red=0.01)
    assert p.classify(0.01) is Light.AMBER  # red only strictly below 0.01
    assert p.classify(0.0099999) is Light.RED
    assert p.classify(0.05) is Light.GREEN  # amber only strictly below 0.05
    assert p.classify(0.0499999) is Light.AMBER
    up = Threshold(kind="upper_bad", amber=0.10, red=0.25)
    assert up.classify(0.0999) is Light.GREEN
    assert up.classify(0.10) is Light.AMBER
    assert up.classify(0.25) is Light.RED
    lo = Threshold(kind="lower_bad", amber=0.70, red=0.60)
    assert lo.classify(0.7001) is Light.GREEN
    assert lo.classify(0.70) is Light.AMBER
    assert lo.classify(0.60) is Light.RED
    assert p.classify(float("nan")) is Light.NOT_EVALUATED
    assert p.classify(None) is Light.NOT_EVALUATED


def test_threshold_rejects_inverted_bands():
    with pytest.raises(ValueError):
        Threshold(kind="p_value", amber=0.01, red=0.05)
    with pytest.raises(ValueError):
        Threshold(kind="upper_bad", amber=0.25, red=0.10)
    with pytest.raises(ValueError):
        Threshold(kind="sideways", amber=0.1, red=0.2)
