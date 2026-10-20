"""The packaged default policy holds the rows of the batch 3 test families, equal to the rows their modules define
(U14 decision G5: `run_suite` passes `default_policy()` explicitly, so a row held only in a module gives no colour)."""
from __future__ import annotations

from riskvalidation.core.policy import default_policy
from riskvalidation.fairness.policy import FAIR_KEYS, fair_threshold


def test_default_policy_holds_every_fair_row_as_the_module_defines_it():
    thresholds = default_policy().thresholds
    for key in FAIR_KEYS:
        assert key in thresholds, f"policies/default.json has no row for {key}"
        held, defined = thresholds[key], fair_threshold(key)
        assert (held.kind, held.amber, held.red) == (defined.kind, defined.amber, defined.red), key


def test_fairness_default_policy_is_the_default_policy_once_the_rows_are_in_it():
    from riskvalidation.fairness.policy import fairness_default_policy

    assert fairness_default_policy().label == default_policy().label
