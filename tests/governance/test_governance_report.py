"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the I.3 validation report
(contract `engine-governance-api` 1.0.0, module report; dossier 06 I.3 l.1219-1237).
"""

from __future__ import annotations

import copy
import json
from datetime import date

import pytest

from riskvalidation.core.result import Light, TestResult
from riskvalidation.core.suite import SuiteRun
from riskvalidation.governance.report import build_report, executive_counts

from .governance_support import AS_OF, report_inputs

SECTION_KEYS = [
    "executive_summary",
    "model_description",
    "conceptual_soundness",
    "data",
    "implementation",
    "outcomes_analysis",
    "sensitivity_robustness_stress",
    "ongoing_monitoring",
    "limitations_and_controls",
    "findings_register",
    "opinion_and_recommendation",
    "appendices",
]
KEYS = ("C07/banco-andino", "C18/banco-andino")
LABEL_EN = "the rule's output on the evidence; Contraste approves no model"


def _build(extract, inventory_doc, findings_doc, key: str) -> dict:
    return build_report(*report_inputs(extract, inventory_doc, findings_doc, key), AS_OF)


STALE_KIND = "the test id is not in the engine registry (spec() raises KeyError), so it has no decision"


def _stale_ids(fixture: dict) -> set[str]:
    """The test ids whose policy identity the fixture wrote before the 0.08.000 catalogue registered them."""
    return {
        r["test_id"]
        for r in fixture["sections"][5]["blocks"][0]["rows"]
        if r["policy"].get("kind_reason") == STALE_KIND
    }


def _normalise(doc: dict, stale: frozenset[str] = frozenset()) -> dict:
    """Drop what the engine does not state the way the fixture builder wrote it: the free-text lines that name the
    batch 3 head or the IFRS 9 provision, and the `kind` and `kind_reason` of the ids that the fixture, built at the
    batch 3 head, wrote as unregistered (the 0.08.000 catalogue registers them; see
    `test_stale_kind_ids_are_registered_and_resolve_their_decision`)."""
    out = copy.deepcopy(doc)
    sec = out["sections"]
    sec[5]["missing"] = "text"
    if sec[7]["status"] == "filled":
        sec[7]["blocks"] = "text"
    for r in sec[5]["blocks"][0]["rows"]:
        if r["test_id"] in stale:
            r["policy"].pop("kind", None)
            r["policy"].pop("kind_reason", None)
    return out


# ---- executive_counts


def test_executive_counts_golden_on_the_default_variant_of_c18(extract):
    rows = extract["cases"]["C18"]["variants"][0]["tests"]
    assert len(rows) == 17
    assert executive_counts(rows) == {"green": 6, "amber": 2, "red": 6, "not_evaluated": 3, "error": 0}


def test_executive_counts_is_the_suite_counting():
    results = tuple(
        TestResult(test_id="t.a", h0="h", reference="r", policy_version="p", light=light, metric=1.0)
        for light in (Light.GREEN, Light.RED, Light.RED, Light.AMBER, Light.NOT_EVALUATED, Light.ERROR, Light.ERROR)
    )
    rows = [r.to_dict() for r in results]
    assert executive_counts(rows) == SuiteRun("p", results).counts()
    assert sum(executive_counts(rows).values()) == len(rows) and executive_counts(rows)["error"] == 2
    assert executive_counts([]) == {"green": 0, "amber": 0, "red": 0, "not_evaluated": 0, "error": 0}
    with pytest.raises(ValueError):
        executive_counts([{"light": "purple"}])


# ---- build_report goldens


def test_report_c18_golden(extract, inventory_doc, findings_doc):
    rep = _build(extract, inventory_doc, findings_doc, "C18/banco-andino")
    secs = rep["sections"]
    assert [s["key"] for s in secs] == SECTION_KEYS and [s["number"] for s in secs] == list(range(1, 13))
    assert secs[7]["status"] == "absent" and secs[7]["absent_reason"]["en"].strip() and secs[7]["blocks"] == []
    op = secs[10]["opinion"]
    assert (
        op["level"] == 4
        and op["recommendation"]["outcome"] == "remediate before use"
        and len(op["open_findings"]) == 29
    )
    assert op["label"]["en"] == LABEL_EN
    assert secs[0]["opinion_level"] == 4
    assert rep["tier"] == 1 and rep["entry_key"] == "C18/banco-andino"


def test_report_c07_golden(extract, inventory_doc, findings_doc):
    rep = _build(extract, inventory_doc, findings_doc, "C07/banco-andino")
    secs = rep["sections"]
    assert secs[10]["opinion"]["level"] == 3 and secs[0]["opinion_level"] == 3
    assert secs[7]["status"] == "filled" and secs[7]["inputs"] and secs[7]["blocks"]
    assert "324129997" in secs[7]["blocks"][0]["text"]["en"]


@pytest.mark.parametrize("key", KEYS)
def test_report_reproduces_the_fixture(key, extract, inventory_doc, findings_doc, report_docs):
    rep = _build(extract, inventory_doc, findings_doc, key)
    stale = frozenset(_stale_ids(report_docs[key]))
    assert _normalise(rep, stale) == _normalise(report_docs[key], stale)


@pytest.mark.parametrize("key", KEYS)
def test_stale_kind_ids_are_registered_and_resolve_their_decision(
    key, extract, inventory_doc, findings_doc, report_docs
):
    """The only difference from the Contraste fixture copy is the `kind` of ids that the fixture wrote as unregistered
    at the batch 3 head: the engine registers them now and states `spec(test_id).decision` with no reason."""
    from riskvalidation.core.registry import spec

    stale = _stale_ids(report_docs[key])
    assert stale
    rows = _build(extract, inventory_doc, findings_doc, key)["sections"][5]["blocks"][0]["rows"]
    seen = set()
    for r in rows:
        if r["test_id"] in stale:
            seen.add(r["test_id"])
            assert r["policy"]["kind"] == spec(r["test_id"]).decision and r["policy"]["kind_reason"] is None
    assert seen == stale


@pytest.mark.parametrize("key", KEYS)
def test_report_computed_fields_equal_the_fixture(key, extract, inventory_doc, findings_doc, report_docs):
    rep, fix = _build(extract, inventory_doc, findings_doc, key), report_docs[key]
    assert rep["test_counts"] == fix["test_counts"]
    assert rep["sections"][10]["opinion"] == fix["sections"][10]["opinion"]
    assert rep["sections"][9]["blocks"] == fix["sections"][9]["blocks"]
    assert rep["sections"][11]["blocks"] == fix["sections"][11]["blocks"]
    assert [
        r["policy"]["parameters"]
        for r in rep["sections"][5]["blocks"][0]["rows"]
        if r["policy"]["source"] == "inline-battery"
    ] == [
        r["policy"]["parameters"]
        for r in fix["sections"][5]["blocks"][0]["rows"]
        if r["policy"]["source"] == "inline-battery"
    ]


# ---- invariants


@pytest.mark.parametrize("key", KEYS)
def test_report_is_deterministic_and_does_not_mutate_its_inputs(key, extract, inventory_doc, findings_doc):
    inputs = report_inputs(extract, inventory_doc, findings_doc, key)
    before = copy.deepcopy(inputs)
    one = build_report(*inputs, AS_OF)
    two = build_report(*inputs, AS_OF)
    assert json.dumps(one, sort_keys=False) == json.dumps(two, sort_keys=False)
    assert inputs == before


@pytest.mark.parametrize("key", KEYS)
def test_report_validates_against_the_schema_and_the_contract_checker(
    key, extract, inventory_doc, findings_doc, validate_schema, checker
):
    rep = _build(extract, inventory_doc, findings_doc, key)
    validate_schema(rep, "governance-report")
    assert checker.violations(rep) == []


@pytest.mark.parametrize("key", KEYS)
def test_report_states_no_approval_and_no_regulatory_convention(key, extract, inventory_doc, findings_doc, checker):
    rep = _build(extract, inventory_doc, findings_doc, key)
    for text in checker.strings(rep):
        assert checker.approval_claims(text) == [], text
        assert checker.regulatory_claims(text) == [], text


@pytest.mark.parametrize("key", KEYS)
def test_a_section_is_filled_exactly_when_it_names_an_input(key, extract, inventory_doc, findings_doc):
    for s in _build(extract, inventory_doc, findings_doc, key)["sections"]:
        if s["status"] == "filled":
            assert s["inputs"] and s["blocks"] and s["absent_reason"] is None
        else:
            assert s["status"] == "absent" and not s["inputs"] and not s["blocks"] and s["absent_reason"]["en"].strip()


def test_section_6_lists_every_test_row_of_the_default_variant_with_its_policy_identity(
    extract, inventory_doc, findings_doc
):
    rep = _build(extract, inventory_doc, findings_doc, "C18/banco-andino")
    rows = rep["sections"][5]["blocks"][0]["rows"]
    default_rows = extract["cases"]["C18"]["variants"][0]["tests"]
    assert [(r["test_id"], r["model_id"], r["segment"]) for r in rows] == [
        (t["test_id"], t["model_id"], t["segment"]) for t in default_rows
    ]
    assert rep["sections"][5]["blocks"][0]["variant_id"] == "baseline"
    assert all(r["policy"]["source"] in ("packaged-policy", "inline-battery", "unresolved") for r in rows)
    assert all(r["family"] == r["test_id"].split(".")[0] for r in rows)


def test_section_10_lists_the_entrys_register_rows_and_section_11_the_open_ones(extract, inventory_doc, findings_doc):
    rep = _build(extract, inventory_doc, findings_doc, "C18/banco-andino")
    rows = [r for r in findings_doc["rows"] if r["entry_key"] == "C18/banco-andino"]
    table = rep["sections"][9]["blocks"][0]["rows"]
    assert [row[0] for row in table] == [r["key"] for r in rows]
    open_keys = [
        r["key"]
        for r in rows
        if r["status"] in ("open", "action_plan_agreed", "in_progress", "submitted_for_closure", "overdue")
    ]
    assert [f["key"] for f in rep["sections"][10]["opinion"]["open_findings"]] == open_keys


def test_section_12_lists_the_sha256_of_the_documents_read(extract, inventory_doc, findings_doc):
    rep = _build(extract, inventory_doc, findings_doc, "C18/banco-andino")
    table = rep["sections"][11]["blocks"][0]["rows"]
    read = {d["path"]: d["sha256"] for d in inventory_doc["provenance"]["documents_read"]}
    assert len(table) == 8 and all(read[path] == sha for path, sha in table)


def test_the_opinion_follows_the_register_a_closed_or_accepted_finding_is_not_open(
    extract, inventory_doc, findings_doc
):
    entry, inv, register, variants = report_inputs(extract, inventory_doc, findings_doc, "C18/banco-andino")
    register = [dict(r, status="validated_closed", overdue=False) for r in register]
    rep = build_report(entry, inv, register, variants, AS_OF)
    op = rep["sections"][10]["opinion"]
    assert op["level"] == 1 and op["open_findings"] == [] and op["recommendation"]["outcome"] == "approve"
    assert rep["sections"][0]["opinion_level"] == 1
    assert (
        "approve" in op["recommendation"]["text"]["en"]
        and "Contraste approves no model" in op["recommendation"]["text"]["en"]
    )


def test_an_rwea_share_raises_the_level_and_sets_the_cut_point_flag(extract, inventory_doc, findings_doc):
    entry, inv, register, variants = report_inputs(extract, inventory_doc, findings_doc, "C07/banco-andino")
    open_rows = [r for r in register if r["status"] == "open" and r["severity"] in ("S3", "S4")]
    register = [
        dict(r, status="validated_closed", overdue=False) if r["severity"] in ("S1", "S2") else dict(r)
        for r in register
    ]
    assert build_report(entry, inv, register, variants, AS_OF)["sections"][10]["opinion"]["level"] == 2
    target = open_rows[0]["key"]
    register = [dict(r, rwea_share=0.12) if r["key"] == target else r for r in register]
    op = build_report(entry, inv, register, variants, AS_OF)["sections"][10]["opinion"]
    assert op["level"] == 4 and op["rwea_cut_points_applied"] is True


def test_a_report_without_a_pma_has_section_8_absent_and_one_with_it_filled(extract, inventory_doc, findings_doc):
    entry, inv, register, variants = report_inputs(extract, inventory_doc, findings_doc, "C07/banco-andino")
    entry = copy.deepcopy(entry)
    entry["attributes"]["A26"] = {
        "value": None,
        "status": "not-applicable",
        "basis": "no PMA or overlay exists in this case",
    }
    assert build_report(entry, inv, register, variants, AS_OF)["sections"][7]["status"] == "absent"


def test_the_as_of_date_is_carried_and_a_date_earlier_than_a_raised_date_is_refused(
    extract, inventory_doc, findings_doc
):
    inputs = report_inputs(extract, inventory_doc, findings_doc, "C18/banco-andino")
    assert build_report(*inputs, AS_OF)["as_of"] == "2026-12-31"
    with pytest.raises(ValueError, match="as_of"):
        build_report(*inputs, date(2026, 10, 1))


def test_a_variant_row_with_an_unreadable_policy_is_an_unresolved_row_with_its_reason(
    extract, inventory_doc, findings_doc
):
    entry, inv, register, variants = report_inputs(extract, inventory_doc, findings_doc, "C18/banco-andino")
    variants = copy.deepcopy(variants)
    variants[0]["tests"][0]["policy_version"] = "x"
    rows = build_report(entry, inv, register, variants, AS_OF)["sections"][5]["blocks"][0]["rows"]
    assert rows[0]["policy"]["source"] == "unresolved" and rows[0]["policy"]["reason"]
    assert rows[1]["policy"]["reason"] is None


def test_an_entry_with_no_variant_is_refused(extract, inventory_doc, findings_doc):
    entry, inv, register, _ = report_inputs(extract, inventory_doc, findings_doc, "C18/banco-andino")
    with pytest.raises(ValueError, match="variant"):
        build_report(entry, inv, register, [], AS_OF)
