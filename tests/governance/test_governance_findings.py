"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the findings workflow
(contract `engine-governance-api` 1.0.0, module findings; dossier 06 B.7 l.614-628).
"""

from __future__ import annotations

import dataclasses
import itertools
from datetime import date

import pytest

from riskvalidation.governance.findings import (
    DEADLINE_DAYS,
    Finding,
    ageing_days,
    anchor_from_test_row,
    deadline_days,
    due_date,
    is_overdue,
    map_product_status,
    register_counts,
    rwea_share_of_anchor,
    severity_to_ecb,
    transition_finding,
)

STATES = (
    "open",
    "action_plan_agreed",
    "in_progress",
    "submitted_for_closure",
    "validated_closed",
    "overdue",
    "risk_accepted",
)
ALLOWED = {
    ("open", "action_plan_agreed"),
    ("open", "risk_accepted"),
    ("open", "overdue"),
    ("action_plan_agreed", "in_progress"),
    ("action_plan_agreed", "risk_accepted"),
    ("action_plan_agreed", "overdue"),
    ("in_progress", "submitted_for_closure"),
    ("in_progress", "risk_accepted"),
    ("in_progress", "overdue"),
    ("submitted_for_closure", "validated_closed"),
    ("submitted_for_closure", "in_progress"),
    ("submitted_for_closure", "overdue"),
    ("overdue", "in_progress"),
    ("overdue", "submitted_for_closure"),
    ("overdue", "risk_accepted"),
    ("risk_accepted", "open"),
}
RISK = {"approver": "not-applicable", "expiry": None}
DEADLINES = {
    "S1": (30, 45, 60, 90),
    "S2": (60, 90, 120, 180),
    "S3": (90, 120, 180, 270),
    "S4": (180, 270, 365, 365),
}


def _finding(**kw) -> Finding:
    base = dict(
        case_id="C01",
        variant_id="v",
        id="f1",
        severity="S2",
        product_status="open",
        evidence=("pd.jeffreys@P0",),
        raised=date(2026, 10, 12),
        tier=1,
        status="open",
        rwea_share=None,
    )
    base.update(kw)
    return Finding(**base)


# ---- Finding


def test_finding_key_is_the_triple_and_it_is_frozen():
    f = _finding()
    assert f.key == "C01/v/f1"
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.status = "closed"  # type: ignore[misc]


def test_finding_invariants():
    with pytest.raises(ValueError):
        _finding(evidence=())
    with pytest.raises(ValueError):
        _finding(severity="S5")
    with pytest.raises(ValueError):
        _finding(status="approved")
    with pytest.raises(ValueError):
        _finding(tier=5)
    with pytest.raises(ValueError):
        _finding(product_status="done")
    with pytest.raises(ValueError):
        _finding(rwea_share=-0.1)


# ---- severity, status


def test_severity_to_ecb_goldens():
    assert [severity_to_ecb(s) for s in ("S1", "S2", "S3", "S4")] == ["F4", "F3", "F2", "F1"]
    for bad in ("S5", "F1", "", "s1"):
        with pytest.raises(ValueError):
            severity_to_ecb(bad)


def test_map_product_status_goldens():
    assert map_product_status("open") == "open"
    assert map_product_status("accepted") == "risk_accepted"
    assert map_product_status("closed") == "validated_closed"
    for bad in ("overdue", "risk_accepted", "validated_closed", "", "Open"):
        with pytest.raises(ValueError):
            map_product_status(bad)


# ---- deadline, due, ageing, overdue


def test_deadline_days_whole_default_table():
    assert DEADLINE_DAYS == DEADLINES
    for sev, row in DEADLINES.items():
        for tier, days in zip(range(1, 5), row, strict=True):
            assert deadline_days(sev, tier) == days


def test_deadline_days_is_non_decreasing_in_tier_and_in_severity_number():
    for sev in DEADLINES:
        assert [deadline_days(sev, t) for t in range(1, 5)] == sorted(deadline_days(sev, t) for t in range(1, 5))
    for tier in range(1, 5):
        col = [deadline_days(s, tier) for s in ("S1", "S2", "S3", "S4")]
        assert col == sorted(col)


def test_deadline_days_refusals_and_other_tables():
    with pytest.raises(ValueError):
        deadline_days("S9", 1)
    for tier in (0, 5):
        with pytest.raises(ValueError):
            deadline_days("S1", tier)
    with pytest.raises(ValueError):
        deadline_days("S1", 1, {"S1": (1, 2, 3)})
    assert deadline_days("S1", 3, {"S1": (1, 2, 3, 4)}) == 3


def test_due_date_goldens_and_identity():
    assert due_date(date(2026, 10, 12), 30) == date(2026, 11, 11)
    assert due_date(date(2026, 10, 16), 60) == date(2026, 12, 15)
    for d in (0, 1, 30, 365):
        assert (due_date(date(2026, 10, 12), d) - date(2026, 10, 12)).days == d
    with pytest.raises(ValueError):
        due_date(date(2026, 10, 12), -1)


def test_ageing_days_goldens_and_refusal():
    assert ageing_days(date(2026, 10, 12), date(2026, 12, 31)) == 80
    assert ageing_days(date(2026, 10, 16), date(2026, 10, 16)) == 0
    with pytest.raises(ValueError):
        ageing_days(date(2026, 10, 16), date(2026, 10, 15))


def test_is_overdue_goldens_at_the_due_date():
    due = date(2026, 11, 11)
    assert is_overdue("open", due, date(2026, 11, 11)) is False  # on the due date
    assert is_overdue("open", due, date(2026, 11, 12)) is True
    assert is_overdue("in_progress", due, date(2026, 12, 31)) is True
    assert is_overdue("risk_accepted", due, date(2027, 6, 1)) is False
    assert is_overdue("validated_closed", due, date(2027, 6, 1)) is False


def test_is_overdue_only_for_the_five_open_states_after_the_due_date():
    due = date(2026, 11, 11)
    for state in STATES:
        want = state in ("open", "action_plan_agreed", "in_progress", "submitted_for_closure", "overdue")
        assert is_overdue(state, due, date(2026, 11, 12)) is want
        assert is_overdue(state, due, date(2026, 11, 10)) is False
    with pytest.raises(ValueError):
        is_overdue("approved", due, date(2026, 11, 12))


# ---- the B.7 transitions


def test_transition_finding_sixteen_allowed_and_thirty_three_refused_pairs():
    assert len(ALLOWED) == 16
    refused = 0
    for frm, to in itertools.product(STATES, STATES):
        if (frm, to) in ALLOWED:
            assert transition_finding(frm, to, RISK) == to
        else:
            refused += 1
            with pytest.raises(ValueError, match=f"cannot move from '{frm}' to '{to}'"):
                transition_finding(frm, to, RISK)
    assert refused == 33 and sum(1 for s in STATES if (s, s) not in ALLOWED) == 7


def test_transition_finding_refused_examples():
    with pytest.raises(ValueError) as exc:
        transition_finding("validated_closed", "open")
    assert str(exc.value) == "cannot move from 'validated_closed' to 'open'"
    for frm, to in (
        ("open", "open"),
        ("open", "in_progress"),
        ("open", "validated_closed"),
        ("action_plan_agreed", "validated_closed"),
        ("risk_accepted", "validated_closed"),
        ("validated_closed", "overdue"),
    ):
        with pytest.raises(ValueError):
            transition_finding(frm, to, RISK)


def test_validated_closed_is_terminal_and_risk_accepted_leaves_only_to_open():
    assert [to for to in STATES if (("validated_closed", to) in ALLOWED)] == []
    assert [to for to in STATES if ("risk_accepted", to) in ALLOWED] == ["open"]
    for to in STATES:
        if to != "open":
            with pytest.raises(ValueError):
                transition_finding("risk_accepted", to, RISK)
    assert transition_finding("risk_accepted", "open") == "open"  # no record is needed to leave


def test_entering_risk_accepted_needs_an_acceptance_record_with_approver_and_expiry_fields():
    for frm in ("open", "action_plan_agreed", "in_progress", "overdue"):
        for bad in (None, {}, {"approver": "x"}, {"expiry": None}):
            with pytest.raises(ValueError):
                transition_finding(frm, "risk_accepted", bad)
        assert transition_finding(frm, "risk_accepted", RISK) == "risk_accepted"
        assert (
            transition_finding(frm, "risk_accepted", {"approver": "CRO", "expiry": date(2027, 1, 1)}) == "risk_accepted"
        )
    with pytest.raises(ValueError):
        transition_finding("open", "risk_accepted")


def test_overdue_is_reached_from_any_open_state_and_leaves_to_three_states():
    for frm in ("open", "action_plan_agreed", "in_progress", "submitted_for_closure"):
        assert transition_finding(frm, "overdue") == "overdue"
    assert {to for to in STATES if ("overdue", to) in ALLOWED} == {
        "in_progress",
        "submitted_for_closure",
        "risk_accepted",
    }
    with pytest.raises(ValueError):
        transition_finding("proposed", "open")
    with pytest.raises(ValueError):
        transition_finding("open", "closed")


# ---- anchor_from_test_row, rwea_share_of_anchor


def test_anchor_golden_statistic():
    rows = [
        {"test_id": "pd.hosmer_lemeshow", "model_id": "P1-scorecard", "segment": None, "statistic": 12.5, "metric": 0.3}
    ]
    anchor, reason = anchor_from_test_row(["pd.hosmer_lemeshow@P1-scorecard"], rows)
    assert reason is None
    assert (
        anchor["metric"] == "statistic"
        and anchor["value"] == 12.5
        and anchor["unit"] == "test statistic (dimensionless)"
    )
    assert (
        anchor["test_id"] == "pd.hosmer_lemeshow" and anchor["model_id"] == "P1-scorecard" and anchor["segment"] is None
    )
    assert set(anchor) == {"metric", "test_id", "model_id", "segment", "value", "unit", "basis"}
    assert anchor["basis"].endswith("#/tests/0/statistic")


def test_anchor_reads_metric_when_the_statistic_is_null_and_names_the_document():
    rows = [
        {"test_id": "t.a", "model_id": "m", "segment": "s1", "statistic": None, "metric": 0.5},
        {"test_id": "t.b", "model_id": "m", "segment": None, "statistic": 1.0, "metric": 2.0},
    ]
    anchor, reason = anchor_from_test_row(["t.a@m@s1"], rows, document="data/derived/C01/base.json")
    assert reason is None and anchor["metric"] == "metric" and anchor["value"] == 0.5 and anchor["segment"] == "s1"
    assert anchor["unit"] == "metric of the test row (the unit its test defines)"
    assert anchor["basis"] == "data/derived/C01/base.json#/tests/0/metric"


def test_anchor_is_none_with_a_reason_when_nothing_is_cited_or_read():
    rows = [{"test_id": "t.a", "model_id": "m", "segment": None, "statistic": None, "metric": None}]
    for ev in (["design:b1-minimum-normal-individual"], ["rate:rating.migration_ztests@r2-n500"], ["pd.jeffreys"], []):
        anchor, reason = anchor_from_test_row(ev, rows)
        assert anchor is None and reason
    anchor, reason = anchor_from_test_row(["t.a@m"], rows)  # both null
    assert anchor is None and reason
    anchor, reason = anchor_from_test_row(["t.zz@m"], rows)  # no row matches
    assert anchor is None and reason


def test_anchor_cites_the_first_cited_evidence_only():
    rows = [
        {"test_id": "t.a", "model_id": "m", "segment": None, "statistic": 1.0, "metric": None},
        {"test_id": "t.b", "model_id": "m", "segment": None, "statistic": 2.0, "metric": None},
    ]
    anchor, _ = anchor_from_test_row(["design:x", "t.b@m", "t.a@m"], rows)
    assert anchor["test_id"] == "t.b" and anchor["value"] == 2.0
    anchor, reason = anchor_from_test_row(
        ["t.zz@m", "t.a@m"], rows
    )  # the first cited one finds no row: a reason, no second try
    assert anchor is None and reason


def test_anchor_takes_a_unit_the_row_states_and_never_guesses_one():
    rows = [
        {"test_id": "t.a", "model_id": "m", "segment": None, "statistic": None, "metric": 0.08, "unit": "share-of-rwea"}
    ]
    anchor, _ = anchor_from_test_row(["t.a@m"], rows)
    assert anchor["unit"] == "share-of-rwea"


def test_rwea_share_goldens():
    assert rwea_share_of_anchor({"unit": "share-of-rwea", "value": 0.08}) == (0.08, None)
    share, reason = rwea_share_of_anchor({"unit": "rwea-amount", "value": 750000.0}, 7492853.0)
    assert reason is None and round(share, 4) == 0.1001
    for anchor in ({"unit": "test statistic (dimensionless)", "value": 12.5}, None):
        share, reason = rwea_share_of_anchor(anchor)
        assert share is None and reason


def test_rwea_share_invariants():
    with pytest.raises(ValueError):
        rwea_share_of_anchor({"unit": "rwea-amount", "value": 1.0})
    with pytest.raises(ValueError):
        rwea_share_of_anchor({"unit": "rwea-amount", "value": 1.0}, 0.0)
    with pytest.raises(ValueError):
        rwea_share_of_anchor({"unit": "rwea-amount", "value": 1.0}, -5.0)
    with pytest.raises(ValueError):
        rwea_share_of_anchor({"unit": "share-of-rwea", "value": -0.1})
    share, reason = rwea_share_of_anchor(None, 100.0)
    assert share is None and reason  # never a share without an anchor


# ---- register_counts


def test_register_counts_golden_on_the_fixture_register(findings_doc):
    counts = register_counts(findings_doc["rows"])
    assert counts["total"] == 81
    assert counts["by_severity"] == {"S1": 2, "S2": 14, "S3": 41, "S4": 24}
    assert counts["by_status"] == {
        "open": 49,
        "action_plan_agreed": 0,
        "in_progress": 0,
        "submitted_for_closure": 0,
        "validated_closed": 4,
        "overdue": 0,
        "risk_accepted": 28,
    }
    assert counts["overdue"] == 7
    assert counts == findings_doc["counts"]


def test_register_counts_invariants(findings_doc):
    counts = register_counts(findings_doc["rows"])
    assert sum(counts["by_severity"].values()) == counts["total"] == sum(counts["by_status"].values())
    assert sum(c["total"] for c in counts["by_case"]) == counts["total"]
    assert [c["case_id"] for c in counts["by_case"]] == sorted({r["case_id"] for r in findings_doc["rows"]})
    assert set(counts["by_case"][0]) == {"case_id", "total", "by_severity", "by_status", "overdue"}
    empty = register_counts([])
    assert empty["total"] == 0 and empty["by_case"] == [] and empty["overdue"] == 0
