"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the `gov.` validation tests
(contract `engine-governance-api` 1.0.0, module validation.governance; dossier 06 B.2 l.539-551, B.3 l.574-575,
B.7 l.614-628, B.8 l.630-640, I.2). Every cut point is a convention, never a regulatory threshold."""

from __future__ import annotations

import copy
import json
import math

import pytest

from riskvalidation.core.policy import Policy, default_policy
from riskvalidation.core.registry import catalogue, spec
from riskvalidation.core.result import Light, TestResult
from riskvalidation.core.suite import run_suite
from riskvalidation.validation.governance import (
    GOV_KEYS,
    GOV_ROWS,
    edges_acyclic,
    findings_overdue,
    gov_default_policy,
    gov_threshold,
    inventory_completeness,
    tier_override_documented,
    validation_due,
)

from .governance_support import AS_OF

IDS = [
    "gov.inventory_completeness",
    "gov.findings_overdue",
    "gov.validation_due",
    "gov.tier_override_documented",
    "gov.edges_acyclic",
]


def test_the_five_tests_are_registered_in_the_family_governance():
    specs = {s.test_id: s for s in catalogue("governance")}
    assert sorted(specs) == sorted(IDS)
    for s in specs.values():
        assert s.family == "governance" and s.decision == "metric" and s.h0.strip() and s.reference.strip()
        assert spec(s.test_id) is s
    assert tuple(GOV_KEYS) == tuple(IDS) and set(GOV_ROWS) == set(IDS)


# ---- goldens


def test_inventory_completeness_golden(inventory_doc):
    r = inventory_completeness(inventory_doc)
    assert (r.n, r.n_events) == (160, 46) and r.metric == pytest.approx(0.7125) and r.light is Light.AMBER
    assert r.statistic is None and r.p_value is None and r.model_id is None and r.segment is None
    assert r.test_id == "gov.inventory_completeness"


def test_findings_overdue_golden(findings_doc):
    r = findings_overdue(findings_doc["rows"], AS_OF)
    assert (
        (r.n, r.n_events) == (49, 7)
        and r.metric == pytest.approx(7 / 49)
        and round(r.metric, 4) == 0.1429
        and r.light is Light.AMBER
    )
    assert r.statistic is None and r.p_value is None


def test_validation_due_golden(inventory_doc):
    r = validation_due(inventory_doc, AS_OF)
    assert (r.n, r.metric, r.n_events) == (4, 0, 0) and r.light is Light.GREEN


def test_tier_override_documented_golden(inventory_doc):
    r = tier_override_documented(inventory_doc)
    assert (r.n, r.metric) == (0, 0) and r.light is Light.GREEN


def test_edges_acyclic_golden(inventory_doc):
    r = edges_acyclic(inventory_doc)
    assert (r.n, r.metric) == (2, 0) and r.light is Light.GREEN
    cyc = copy.deepcopy(inventory_doc)
    cyc["edges"] = [
        {"upstream": a, "downstream": b, "kind": "document", "stated_source": "s"}
        for a, b in (
            ("C01/german", "C07/banco-andino"),
            ("C07/banco-andino", "C18/banco-andino"),
            ("C18/banco-andino", "C01/german"),
        )
    ]
    r = edges_acyclic(cyc)
    assert r.metric == 3 and r.light is Light.RED and r.n == 3
    assert "dependency cycle: C01/german, C07/banco-andino, C18/banco-andino, C01/german" in r.notes


# ---- the boundaries of every light


def _inv_with_missing(missing: int, total: int = 100) -> dict:
    """An inventory of `total` attribute pairs of which `missing` are missing (a quarter-entry per 40: 5 entries of
    40 give 200 pairs).
    """
    n_entries = math.ceil(total / 40)
    entries = []
    left = missing
    for i in range(n_entries):
        attrs = {}
        for j in range(1, 41):
            miss = left > 0
            left -= 1 if miss else 0
            attrs[f"A{j:02d}"] = (
                {"value": None, "status": "missing", "basis": "no field"}
                if miss
                else {"value": 1, "status": "convention", "basis": "a basis"}
            )
        entries.append(
            {
                "key": f"E{i}/f",
                "attributes": attrs,
                "schedule": {"next_review": "2030-01-01"},
                "tier_block": {"override": None},
            }
        )
    return {"entries": entries, "edges": []}


def test_inventory_completeness_boundaries_are_strict():
    # 40 pairs: share 0.9 is green (amber is "below 0.9"), 0.875 amber, 0.7 amber (red is "below 0.7"), 0.675 red
    assert inventory_completeness(_inv_with_missing(4, 40)).light is Light.GREEN
    assert inventory_completeness(_inv_with_missing(5, 40)).light is Light.AMBER
    assert inventory_completeness(_inv_with_missing(12, 40)).light is Light.AMBER
    assert inventory_completeness(_inv_with_missing(13, 40)).light is Light.RED
    r = inventory_completeness(_inv_with_missing(0, 40))
    assert r.light is Light.GREEN and r.metric == 1.0 and r.n_events == 0


def test_findings_overdue_boundaries_are_strict():
    def rows(n_open: int, n_over: int):
        return [{"status": "open", "due": "2026-12-30" if i < n_over else "2027-01-30"} for i in range(n_open)]

    assert findings_overdue(rows(10, 0), AS_OF).light is Light.GREEN
    assert findings_overdue(rows(10, 1), AS_OF).light is Light.AMBER
    assert findings_overdue(rows(4, 1), AS_OF).light is Light.AMBER  # 0.25 is "above 0.25" false: amber
    assert findings_overdue(rows(10, 3), AS_OF).light is Light.RED  # 0.3
    assert findings_overdue(rows(3, 1), AS_OF).light is Light.RED  # 0.333


def test_findings_overdue_counts_the_overdue_rows_as_is_overdue_does():
    rows = [
        {"status": "open", "due": "2026-12-31"},
        {"status": "in_progress", "due": "2026-12-30"},
        {"status": "overdue", "due": "2026-12-30"},
        {"status": "risk_accepted", "due": "2020-01-01"},
        {"status": "validated_closed", "due": "2020-01-01"},
        {"status": "submitted_for_closure", "due": "2027-01-01"},
    ]
    r = findings_overdue(rows, AS_OF)
    assert (r.n, r.n_events) == (
        4,
        2,
    )  # risk_accepted and validated_closed are not open-state rows; a row on its due date is not overdue


def test_findings_overdue_with_no_open_state_row_is_not_evaluated():
    r = findings_overdue([{"status": "validated_closed", "due": "2020-01-01"}], AS_OF)
    assert r.light is Light.NOT_EVALUATED and r.n == 0 and r.to_dict()["metric"] is None and r.notes
    assert findings_overdue([], AS_OF).light is Light.NOT_EVALUATED


def test_validation_due_boundaries(inventory_doc):
    def inv(dates):
        return {
            "entries": [{"key": f"E{i}/f", "schedule": {"next_review": d}} for i, d in enumerate(dates)],
            "edges": [],
        }

    assert (
        validation_due(inv(["2026-12-31", "2027-01-01"]), AS_OF).light is Light.GREEN
    )  # on the as-of date is not past
    r = validation_due(inv(["2026-12-30", "2027-01-01"]), AS_OF)
    assert r.light is Light.AMBER and r.metric == 1 and r.n_events == 1 and r.n == 2
    r = validation_due(inv(["2026-12-30", "2026-01-01", "2027-01-01"]), AS_OF)
    assert r.light is Light.RED and r.metric == 2
    assert validation_due(inv(["2026-12-30"] * 5), AS_OF).light is Light.RED


def test_tier_override_documented_counts_the_overrides_without_a_rationale_or_an_approver(inventory_doc):
    inv = copy.deepcopy(inventory_doc)
    inv["entries"][0]["tier_block"]["override"] = {
        "to_tier": 1,
        "rationale": "the model feeds the stress test",
        "approver": "CRO",
    }
    r = tier_override_documented(inv)
    assert (r.n, r.metric, r.light) == (1, 0, Light.GREEN)
    inv["entries"][1]["tier_block"]["override"] = {"to_tier": 2, "rationale": "", "approver": "CRO"}
    r = tier_override_documented(inv)
    assert (r.n, r.metric, r.light) == (2, 1, Light.RED)
    inv["entries"][2]["tier_block"]["override"] = {"to_tier": 2, "rationale": "a reason", "approver": " "}
    assert tier_override_documented(inv).metric == 2


def test_edges_acyclic_counts_unknown_endpoints_and_names_the_cycle(inventory_doc):
    inv = copy.deepcopy(inventory_doc)
    inv["edges"].append({"upstream": "C01/german", "downstream": "ZZ/none", "kind": "code", "stated_source": "s"})
    r = edges_acyclic(inv)
    assert (r.n, r.metric, r.light) == (3, 1, Light.RED)
    cyc = copy.deepcopy(inventory_doc)
    cyc["edges"] = [
        {"upstream": "C01/german", "downstream": "C22/generators", "kind": "code", "stated_source": "s"},
        {"upstream": "C22/generators", "downstream": "C01/german", "kind": "code", "stated_source": "s"},
        {"upstream": "C22/generators", "downstream": "C18/banco-andino", "kind": "code", "stated_source": "s"},
    ]
    r = edges_acyclic(cyc)
    assert (
        r.metric == 2 and "dependency cycle: C01/german, C22/generators, C01/german" in r.notes
    )  # C18 is downstream, not on the cycle


# ---- rows, policy, serialisation


def test_every_result_serialises_with_the_test_result_fields(inventory_doc, findings_doc):
    results = [
        inventory_completeness(inventory_doc),
        findings_overdue(findings_doc["rows"], AS_OF),
        validation_due(inventory_doc, AS_OF),
        tier_override_documented(inventory_doc),
        edges_acyclic(inventory_doc),
    ]
    assert [r.test_id for r in results] == IDS
    for r in results:
        assert isinstance(r, TestResult)
        d = r.to_dict()
        assert d["h0"].strip() and d["reference"].strip() and d["policy_version"].strip() and d["inputs_hash"]
        assert (
            d["light"] in ("green", "amber", "red", "not_evaluated", "error")
            and d["model_id"] is None
            and d["segment"] is None
        )
        assert d["extras"]["thresholds"]["label"] == "convention"
        json.dumps(d)
    assert "dossier 06" in results[0].reference


def test_results_are_deterministic_and_do_not_mutate_the_inputs(inventory_doc, findings_doc):
    inv, rows = copy.deepcopy(inventory_doc), copy.deepcopy(findings_doc["rows"])
    a = inventory_completeness(inv).to_dict(), findings_overdue(rows, AS_OF).to_dict()
    b = inventory_completeness(inv).to_dict(), findings_overdue(rows, AS_OF).to_dict()
    assert a == b and inv == inventory_doc and rows == findings_doc["rows"]


def test_an_empty_inventory_is_not_evaluated_with_its_reason():
    r = inventory_completeness({"entries": [], "edges": []})
    assert (
        r.light is Light.NOT_EVALUATED
        and "no entry" in r.notes
        and r.to_dict()["metric"] is None
        and r.to_dict()["statistic"] is None
    )
    for fn in (tier_override_documented, edges_acyclic):
        assert fn({"entries": [], "edges": []}).n in (0, None)
    with pytest.raises(ValueError, match="entries"):
        inventory_completeness({})


def test_the_policy_rows_are_conventions_with_strict_cuts():
    assert GOV_ROWS["gov.inventory_completeness"].kind == "lower_bad" and GOV_ROWS["gov.inventory_completeness"].strict
    assert (GOV_ROWS["gov.inventory_completeness"].amber, GOV_ROWS["gov.inventory_completeness"].red) == (0.9, 0.7)
    assert (
        GOV_ROWS["gov.findings_overdue"].kind,
        GOV_ROWS["gov.findings_overdue"].amber,
        GOV_ROWS["gov.findings_overdue"].red,
    ) == ("upper_bad", 0.0, 0.25)
    assert (GOV_ROWS["gov.validation_due"].amber, GOV_ROWS["gov.validation_due"].red) == (1, 2)
    assert GOV_ROWS["gov.tier_override_documented"].red == 1 and GOV_ROWS["gov.edges_acyclic"].red == 1
    for key in IDS:
        assert GOV_ROWS[key].source.startswith("convention")
        th = gov_threshold(key)
        assert th.kind == GOV_ROWS[key].kind
    th = gov_threshold("gov.inventory_completeness")
    assert th.classify(0.9).value == "green" and th.classify(math.nextafter(0.9, 0)).value == "amber"
    assert th.classify(0.7).value == "amber" and th.classify(math.nextafter(0.7, 0)).value == "red"
    th = gov_threshold("gov.findings_overdue")
    assert (
        th.classify(0.0).value == "green" and th.classify(1e-9).value == "amber" and th.classify(0.25).value == "amber"
    )
    assert th.classify(0.2501).value == "red"


def test_gov_default_policy_is_the_default_policy_once_it_holds_every_row():
    pol = gov_default_policy()
    for key in IDS:
        assert key in pol.thresholds
    base = default_policy()
    if all(k in base.thresholds for k in IDS):
        assert pol is base
    else:
        assert pol.label.startswith(base.label) and "gov-1.0.0" in pol.label


def test_an_explicit_policy_without_the_key_gives_not_evaluated_never_a_colour(inventory_doc):
    pol = Policy("p", "1", "empty", {})
    r = inventory_completeness(inventory_doc, policy=pol)
    assert r.light is Light.NOT_EVALUATED and r.metric == pytest.approx(0.7125)


def test_the_tests_run_in_a_suite(inventory_doc, findings_doc):
    run = run_suite(
        [
            (inventory_completeness, {"inventory": inventory_doc}),
            (findings_overdue, {"register": findings_doc["rows"], "as_of": AS_OF}),
            (validation_due, {"inventory": inventory_doc, "as_of": AS_OF}),
            (tier_override_documented, {"inventory": inventory_doc}),
            (edges_acyclic, {"inventory": inventory_doc}),
        ],
        policy=gov_default_policy(),
    )
    assert [r.light for r in run.results] == [Light.AMBER, Light.AMBER, Light.GREEN, Light.GREEN, Light.GREEN]
