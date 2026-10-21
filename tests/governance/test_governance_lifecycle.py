"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the B.4 lifecycle (contract
`engine-governance-api` 1.0.0, module lifecycle; dossier 06 B.4 l.576-583).
"""

from __future__ import annotations

from datetime import date

import pytest

from riskvalidation.governance.lifecycle import allowed_next, transition

STATES = (
    "proposed",
    "in_development",
    "submitted_for_validation",
    "in_validation",
    "approved",
    "approved_with_conditions",
    "in_use",
    "under_monitoring_alert",
    "material_change_pending",
    "restricted",
    "in_remediation",
    "decommissioned",
)
ALLOWED = {
    ("proposed", "in_development"),
    ("in_development", "submitted_for_validation"),
    ("submitted_for_validation", "in_validation"),
    ("in_validation", "approved"),
    ("in_validation", "approved_with_conditions"),
    ("in_validation", "in_development"),
    ("approved", "in_use"),
    ("approved_with_conditions", "in_use"),
    ("in_use", "under_monitoring_alert"),
    ("in_use", "material_change_pending"),
    ("in_use", "in_validation"),
    ("in_use", "restricted"),
    ("in_use", "decommissioned"),
    ("under_monitoring_alert", "in_use"),
    ("under_monitoring_alert", "restricted"),
    ("under_monitoring_alert", "in_remediation"),
    ("restricted", "in_remediation"),
    ("restricted", "in_use"),
    ("restricted", "decommissioned"),
    ("in_remediation", "in_validation"),
    ("in_remediation", "restricted"),
    ("in_remediation", "decommissioned"),
    ("material_change_pending", "submitted_for_validation"),
    ("material_change_pending", "in_use"),
}
EXCEPTION = ("use before validation", "Chief Risk Officer", date(2027, 1, 31))


def test_the_contract_lists_twenty_four_allowed_transitions_of_twelve_states():
    assert len(ALLOWED) == 24 and len(STATES) == 12


def test_allowed_next_is_exactly_the_allowed_pairs():
    for state in STATES:
        assert set(allowed_next(state)) == {to for frm, to in ALLOWED if frm == state}, state


def test_every_allowed_pair_moves_and_every_other_pair_is_refused():
    for frm in STATES:
        for to in STATES:
            if (frm, to) in ALLOWED:
                assert transition(frm, to) == to
            else:
                with pytest.raises(ValueError, match=f"cannot move from '{frm}' to '{to}'"):
                    transition(frm, to)


def test_decommissioned_is_terminal_and_unknown_states_raise():
    assert allowed_next("decommissioned") == ()
    for to in STATES:
        with pytest.raises(ValueError):
            transition("decommissioned", to)
        with pytest.raises(ValueError):
            transition("decommissioned", to, EXCEPTION)
    with pytest.raises(ValueError):
        allowed_next("retired")
    with pytest.raises(ValueError):
        transition("retired", "in_use")
    with pytest.raises(ValueError):
        transition("proposed", "retired")


def test_transition_goldens():
    assert transition("proposed", "in_development") == "in_development"
    with pytest.raises(ValueError) as exc:
        transition("proposed", "in_use")
    assert str(exc.value) == "cannot move from 'proposed' to 'in_use'"
    with pytest.raises(ValueError) as exc:
        transition("in_validation", "in_use")
    assert str(exc.value) == "cannot move from 'in_validation' to 'in_use'"
    assert transition("in_validation", "in_use", EXCEPTION) == "in_use"


def test_use_before_validation_needs_a_complete_exception_record():
    for frm in ("in_development", "submitted_for_validation", "in_validation"):
        assert transition(frm, "in_use", EXCEPTION) == "in_use"
        with pytest.raises(ValueError, match=f"'{frm}' to 'in_use'"):
            transition(frm, "in_use")
        for bad in (
            ("", "Chief Risk Officer", date(2027, 1, 31)),
            ("use before validation", "", date(2027, 1, 31)),
            ("use before validation", "Chief Risk Officer", None),
        ):
            with pytest.raises(ValueError):
                transition(frm, "in_use", bad)  # type: ignore[arg-type]


def test_an_exception_record_opens_no_other_transition():
    for frm in ("proposed", "approved", "restricted", "in_remediation"):
        if (frm, "in_use") not in ALLOWED:
            with pytest.raises(ValueError, match="cannot move"):
                transition(frm, "in_use", EXCEPTION)
    with pytest.raises(ValueError):
        transition("proposed", "approved", EXCEPTION)
    assert transition("approved", "in_use", EXCEPTION) == "in_use"  # an allowed pair is unchanged by a record
