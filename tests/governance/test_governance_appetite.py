"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the model risk appetite
metrics (contract `engine-governance-api` 1.0.0, module appetite; dossier 06 B.6 l.606-612). Every threshold is a
convention, never a regulatory value."""

from __future__ import annotations

import copy

import pytest

from riskvalidation.governance.appetite import (
    APPETITE_THRESHOLDS,
    appetite_metrics,
    appetite_population,
    entries_in_breach,
    entries_under_exception,
    light,
    open_high_severity,
    overdue_remediation,
    owner_and_complete_record,
    pma_magnitude,
    tier1_validated_on_time,
)

from .governance_support import AS_OF

IDS = [
    "tier1-validated-on-time",
    "open-high-severity",
    "overdue-remediation",
    "entries-under-exception",
    "entries-in-breach",
    "pma-magnitude",
    "owner-and-complete-record",
]


def test_light_goldens():
    assert light(0, "lower-is-better", 0, 3) == "green"
    assert light(3, "lower-is-better", 0, 3) == "amber"
    assert light(4, "lower-is-better", 0, 3) == "red"
    assert light(0.9, "higher-is-better", 0.9, 0.75) == "green"
    assert light(0.8999, "higher-is-better", 0.9, 0.75) == "amber"
    assert light(0.75, "higher-is-better", 0.9, 0.75) == "amber"
    assert light(0.7499, "higher-is-better", 0.9, 0.75) == "red"
    assert light(None, "lower-is-better", 0, 3) == "not_evaluated"
    with pytest.raises(ValueError):
        light(1, "sideways", 0, 3)


def test_default_thresholds_are_the_stated_conventions():
    assert APPETITE_THRESHOLDS == {
        "tier1-validated-on-time": ("higher-is-better", 0.9, 0.75),
        "open-high-severity": ("lower-is-better", 0, 3),
        "overdue-remediation": ("lower-is-better", 0, 3),
        "entries-under-exception": ("lower-is-better", 0, 1),
        "entries-in-breach": ("lower-is-better", 0, 1),
        "pma-magnitude": ("lower-is-better", 0.01, 0.03),
        "owner-and-complete-record": ("higher-is-better", 0.9, 0.5),
    }


def test_tier1_validated_on_time_golden(inventory_doc):
    assert tier1_validated_on_time(inventory_doc["entries"], AS_OF) == (2, 2, 1.0)


def test_tier1_validated_on_time_counts_the_review_date_on_or_after_the_as_of_date(inventory_doc):
    entries = copy.deepcopy(inventory_doc["entries"])
    tier1 = [e for e in entries if e["tier_block"]["tier"] == 1]
    tier1[0]["schedule"]["next_review"] = "2026-12-30"
    assert tier1_validated_on_time(entries, AS_OF)[:2] == (len(tier1) - 1, len(tier1))
    tier1[0]["schedule"]["next_review"] = "2026-12-31"  # on the as-of date: on time
    assert tier1_validated_on_time(entries, AS_OF)[0] == len(tier1)


def test_tier1_validated_on_time_is_none_without_a_tier1_entry(inventory_doc):
    entries = copy.deepcopy(inventory_doc["entries"])
    for e in entries:
        e["tier_block"]["tier"] = 4
    num, den, value = tier1_validated_on_time(entries, AS_OF)
    assert (num, den, value) == (0, 0, None)


def test_open_high_severity_golden(findings_doc):
    assert open_high_severity(findings_doc["rows"]) == (4, 39, 4, 8)


def test_overdue_remediation_golden(findings_doc):
    assert overdue_remediation(findings_doc["rows"]) == (3, 39, 3, 7)


def test_distinct_finding_counts_count_a_finding_repeated_in_variants_once():
    rows = [
        {"entry_key": "E/f", "id": "x", "variant_id": v, "status": "open", "severity": "S1", "overdue": True}
        for v in ("a", "b", "c")
    ]
    rows += [
        {
            "entry_key": "E/f",
            "id": "y",
            "variant_id": "a",
            "status": "validated_closed",
            "severity": "S2",
            "overdue": False,
        },
        {"entry_key": "F/f", "id": "x", "variant_id": "a", "status": "overdue", "severity": "S2", "overdue": True},
    ]
    assert open_high_severity(rows) == (2, 3, 2, 4)  # (E/f, x) and (F/f, x); rows: three of E/f x and one of F/f x
    assert overdue_remediation(rows) == (2, 3, 2, 4)
    rows.append({"entry_key": "G/f", "id": "w", "variant_id": "a", "status": "open", "severity": "S4", "overdue": True})
    assert overdue_remediation(rows) == (
        3,
        4,
        3,
        5,
    )  # a minor overdue finding counts, a high-severity filter does not apply
    assert open_high_severity(rows) == (2, 4, 2, 4)
    low = [{"entry_key": "E/f", "id": "z", "variant_id": "a", "status": "open", "severity": "S3", "overdue": False}]
    assert open_high_severity(low) == (0, 1, 0, 0)


def test_entries_under_exception_golden_and_count(inventory_doc):
    assert entries_under_exception(inventory_doc["entries"]) == (0, 4, 0)
    entries = copy.deepcopy(inventory_doc["entries"])
    entries[0]["lifecycle"]["exception"] = {"type": "use before validation", "approver": "CRO", "expiry": "2027-01-31"}
    assert entries_under_exception(entries) == (1, 4, 1)


def test_entries_in_breach_golden_and_a_red_on_any_variant_counts_the_entry_once(inventory_doc):
    assert entries_in_breach(inventory_doc["entries"]) == (3, 4, 3)
    reds = {e["key"]: e["attributes"]["A36"]["value"]["red"] for e in inventory_doc["entries"]}
    assert reds == {"C01/german": 13, "C07/banco-andino": 0, "C18/banco-andino": 24, "C22/generators": 8}
    entries = copy.deepcopy(inventory_doc["entries"])
    entries[0]["attributes"]["A36"]["value"]["red"] = 400
    assert entries_in_breach(entries) == (3, 4, 3)
    entries[1]["attributes"]["A36"]["value"]["red"] = 1
    assert entries_in_breach(entries) == (4, 4, 4)


def test_entries_in_breach_refuses_an_entry_with_no_light_counts(inventory_doc):
    entries = copy.deepcopy(inventory_doc["entries"])
    entries[0]["attributes"]["A36"] = {"value": None, "status": "missing", "basis": "no field"}
    with pytest.raises(ValueError, match="C01/german"):
        entries_in_breach(entries)


def test_pma_magnitude_golden(inventory_doc):
    num, den, value = pma_magnitude(inventory_doc["entries"])
    assert num == 324129997.0 and den == 97592002519.53
    assert value == pytest.approx(0.0033213, abs=5e-8)
    assert value == pytest.approx(num / den, rel=1e-12)


def test_pma_magnitude_none_without_a_pma_and_refuses_mixed_currencies(inventory_doc):
    entries = copy.deepcopy(inventory_doc["entries"])
    pma_entry = next(e for e in entries if e["attributes"]["A26"]["status"] == "present")
    assert pma_magnitude([e for e in entries if e is not pma_entry]) == (None, None, None)
    other = copy.deepcopy(pma_entry)
    other["attributes"]["A26"]["value"]["currency"] = "USD"
    with pytest.raises(ValueError, match="currenc"):
        pma_magnitude([pma_entry, other])
    twin = copy.deepcopy(pma_entry)
    num, den, value = pma_magnitude([pma_entry, twin])
    assert num == 2 * 324129997.0 and den == 2 * 97592002519.53


def test_owner_and_complete_record_golden_and_rule(inventory_doc):
    assert owner_and_complete_record(inventory_doc["entries"]) == (0, 4, 0.0)
    entries = copy.deepcopy(inventory_doc["entries"])
    e = entries[0]
    for a in e["attributes"].values():
        if a["status"] == "missing":
            a.update(
                value=None, status="not-applicable", basis="a stated reason of enough length"
            )  # not-applicable is complete
    assert owner_and_complete_record(entries) == (1, 4, 0.25)
    e["attributes"]["A28"] = {"value": None, "status": "missing", "basis": "no field"}  # no owner
    assert owner_and_complete_record(entries) == (0, 4, 0.0)


def test_appetite_metrics_golden_on_the_fixture(inventory_doc, findings_doc):
    metrics = appetite_metrics(inventory_doc["entries"], findings_doc["rows"], AS_OF)
    assert [m["id"] for m in metrics] == IDS
    assert [m["light"] for m in metrics] == ["green", "red", "amber", "green", "red", "green", "red"]


def test_appetite_metrics_invariants(inventory_doc, findings_doc):
    metrics = appetite_metrics(inventory_doc["entries"], findings_doc["rows"], AS_OF)
    for m in metrics:
        assert m["threshold"]["label"] == "convention" and m["threshold"]["basis"].strip()
        d, g, a = APPETITE_THRESHOLDS[m["id"]]
        assert (m["threshold"]["direction"], m["threshold"]["green_limit"], m["threshold"]["amber_limit"]) == (d, g, a)
        if m["unit"] == "count":
            assert m["value"] == m["numerator"]
        else:
            assert m["value"] == pytest.approx(m["numerator"] / m["denominator"])
        assert m["light"] == light(m["value"], d, g, a)
        assert (m["register_rows"] is not None) == (m["id"] in ("open-high-severity", "overdue-remediation"))
    assert [m["register_rows"] for m in metrics] == [None, 8, 7, None, None, None, None]


def test_a_metric_with_no_value_is_not_evaluated_with_a_reason(inventory_doc, findings_doc):
    entries = [copy.deepcopy(e) for e in inventory_doc["entries"] if e["attributes"]["A26"]["status"] != "present"]
    for e in entries:
        e["tier_block"]["tier"] = 3
    metrics = appetite_metrics(entries, findings_doc["rows"], AS_OF)
    by = {m["id"]: m for m in metrics}
    for mid in ("tier1-validated-on-time", "pma-magnitude"):
        assert by[mid]["value"] is None and by[mid]["light"] == "not_evaluated" and by[mid]["reason"]


def test_appetite_metrics_take_other_thresholds(inventory_doc, findings_doc):
    th = dict(APPETITE_THRESHOLDS)
    th["open-high-severity"] = ("lower-is-better", 10, 20)
    metrics = appetite_metrics(inventory_doc["entries"], findings_doc["rows"], AS_OF, th)
    assert metrics[1]["light"] == "green" and metrics[1]["threshold"]["green_limit"] == 10


def test_appetite_population(inventory_doc, findings_doc):
    assert appetite_population(inventory_doc["entries"], findings_doc["rows"]) == {
        "entries": 4,
        "findings": 81,
        "distinct_findings": 39,
        "tier1_entries": 2,
    }
