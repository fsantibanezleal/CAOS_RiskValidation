"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the policy identity of a
test row (contract `engine-governance-api` 1.0.0, module policy; decisions W15-6 and W15-12).
"""

from __future__ import annotations

import pytest

from riskvalidation.core.policy import default_policy
from riskvalidation.governance import policy as gpolicy
from riskvalidation.governance.policy import resolve_policy

PACKAGED = default_policy().label
INLINE = "contraste-ifrs9-battery-1.0.0; nl_amber=1.0; nl_red=3.0"
IDENTITY_KEYS = {
    "label",
    "kind",
    "kind_reason",
    "direction",
    "direction_reason",
    "alpha_amber",
    "alpha_red",
    "parameters",
    "version",
    "source",
}


@pytest.fixture
def ifrs9_unregistered(monkeypatch):
    """At the batch 3 head the ifrs9 tests are not registered: `spec()` raises KeyError for them, whatever the
    registry holds later.
    """
    real = gpolicy.spec

    def spec(test_id):
        if test_id.startswith("ifrs9."):
            raise KeyError(f"no registered test {test_id!r}")
        return real(test_id)

    monkeypatch.setattr(gpolicy, "spec", spec)


def test_packaged_label_resolves_through_the_policy_file():
    assert PACKAGED == "riskvalidation-default 2026.10.1"
    identity, reason = resolve_policy(PACKAGED, "pd.jeffreys", 0.05, 0.01)
    assert reason is None and set(identity) >= IDENTITY_KEYS
    assert (
        identity["source"] == "packaged-policy" and identity["version"] == "2026.10.1" and identity["label"] == PACKAGED
    )
    assert identity["kind"] == "p_value" and identity["kind_reason"] is None
    assert identity["alpha_amber"] == 0.05 and identity["alpha_red"] == 0.01 and identity["parameters"] == {}


def test_packaged_direction_is_read_from_the_policy_file():
    thresholds = default_policy().thresholds
    upper = next(k for k, t in thresholds.items() if t.kind == "upper_bad")
    identity, _ = resolve_policy(PACKAGED, upper)
    assert identity["direction"] == "upper_bad" and identity["direction_reason"] is None
    identity, _ = resolve_policy(PACKAGED, "pd.jeffreys")  # a p-value threshold has no direction
    assert identity["direction"] is None and identity["direction_reason"]
    identity, _ = resolve_policy(PACKAGED, "disc.auc")  # a descriptive test holds no threshold
    assert identity["direction"] is None and identity["direction_reason"] and identity["kind"] == "metric"


def test_inline_battery_string_is_parsed(ifrs9_unregistered):
    identity, reason = resolve_policy(INLINE, "ifrs9.scenario_sensitivity")
    assert reason is None
    assert identity["source"] == "inline-battery" and identity["version"] == "1.0.0" and identity["label"] == INLINE
    assert identity["parameters"] == {"nl_amber": 1.0, "nl_red": 3.0}
    assert identity["kind"] is None and "registry" in identity["kind_reason"]
    assert identity["direction"] is None and identity["direction_reason"] == "no policy file (BL-085)"


def test_the_same_inline_string_for_a_registered_test_has_a_kind():
    identity, reason = resolve_policy(INLINE, "pd.jeffreys")
    assert reason is None and identity["kind"] == "p_value" and identity["kind_reason"] is None
    assert identity["direction"] is None and identity["direction_reason"] == "no policy file (BL-085)"
    identity, _ = resolve_policy(INLINE, "disc.auc")
    assert identity["kind"] == "metric"


def test_an_unresolvable_identity_returns_none_with_its_reason():
    for text in ("x", "", "contraste-battery-1.0.0", "contraste-ifrs9-battery-1.0; a=1", "other-policy 2026.10.1"):
        identity, reason = resolve_policy(text, "pd.jeffreys")
        assert identity is None and reason, text
    identity, reason = resolve_policy("riskvalidation-default 1999.1.1", "pd.jeffreys")  # not the packaged version
    assert identity is None and "2026.10.1" in reason


def test_exactly_one_of_identity_and_reason_is_none():
    for text in (PACKAGED, INLINE, "x"):
        identity, reason = resolve_policy(text, "pd.jeffreys")
        assert (identity is None) != (reason is None)


def test_non_numeric_parameters_stay_strings_and_numbers_are_numbers():
    identity, _ = resolve_policy("contraste-stress-battery-1.0.0; rule=red-on-any", "stress.reverse")
    assert identity["parameters"] == {"rule": "red-on-any"}
    identity, _ = resolve_policy(
        "contraste-ifrs9-battery-2.1.3; tolerance=1e-09; r_green=0.8-1.0; basis=tolerance * max(1, G); n=4", "x.y"
    )
    assert identity["parameters"] == {
        "tolerance": 1e-09,
        "r_green": "0.8-1.0",
        "basis": "tolerance * max(1, G)",
        "n": 4.0,
    }
    assert identity["version"] == "2.1.3"
    identity, _ = resolve_policy("contraste-stress-battery-1.0.0", "stress.reverse")
    assert identity["parameters"] == {}


def test_the_alphas_pass_through_and_the_function_is_pure():
    a = resolve_policy(INLINE, "pd.jeffreys", 0.1, 0.02)
    b = resolve_policy(INLINE, "pd.jeffreys", 0.1, 0.02)
    assert a == b and a[0]["alpha_amber"] == 0.1 and a[0]["alpha_red"] == 0.02
    assert resolve_policy(INLINE, "pd.jeffreys")[0]["alpha_amber"] is None
