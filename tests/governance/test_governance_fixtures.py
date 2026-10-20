"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the functions of
`riskvalidation.governance`, called on the inputs of the four contract fixtures (CAOS_Contraste `task/u15-s` at
5dce395: C01/german, C07/banco-andino, C18/banco-andino, C22/generators), reproduce every computed field of the
fixtures."""

from __future__ import annotations

import copy
from datetime import date

import pytest

from riskvalidation.governance.appetite import appetite_metrics, appetite_population
from riskvalidation.governance.findings import (
    ageing_days,
    anchor_from_test_row,
    deadline_days,
    due_date,
    is_overdue,
    map_product_status,
    register_counts,
    rwea_share_of_anchor,
    severity_to_ecb,
)
from riskvalidation.governance.inventory import (
    attribute_completeness,
    build_inventory,
    graph_metrics,
    topological_order,
)
from riskvalidation.governance.schedule import next_dates
from riskvalidation.governance.scope import ScopeAnswers, scope_record
from riskvalidation.governance.tiering import (
    apply_override,
    complexity,
    exposure_band,
    materiality,
    matrix_tier,
    tier_counts,
    tier_sensitivity,
)

from .governance_support import AS_OF, edges_of, entries_of

RELEASE = {"0.07.000": date(2026, 10, 12), "0.12.000": date(2026, 10, 16)}  # CHANGELOG.md release dates (A D1 A34)


def _table(conv: dict) -> dict:
    return {
        t["tier"]: (t["review_months"], t["revalidation_months"], t["monitoring_months"])
        for t in conv["frequency"]["tiers"]
    }


# ---- the inventory


def test_the_fixture_entries_and_edges_build_an_inventory(inventory_doc):
    inv = build_inventory(
        entries_of(inventory_doc), edges_of(inventory_doc), related=[(r["a"], r["b"]) for r in inventory_doc["related"]]
    )
    assert [e.key for e in inv.entries] == [e["key"] for e in inventory_doc["entries"]]
    assert len(inv.edges) == 2


def test_tier_blocks_are_reproduced(inventory_doc):
    cuts = tuple(inventory_doc["conventions"]["exposure_band_cuts"]["cuts"])
    for e in inventory_doc["entries"]:
        tb = e["tier_block"]
        mat, cx = tb["materiality"], tb["complexity"]
        band = exposure_band(mat["exposure"]["share"], cuts)
        assert mat["exposure"]["band"] == band, e["key"]
        reg, dec = mat["regulatory_use"], mat["decision_criticality"]["score"]
        assert mat["m"] == {
            "literal": materiality(band, reg["literal"], dec),
            "graded": materiality(band, reg["graded"], dec),
        }, e["key"]
        scores = [f["score"] for f in cx["factors"] if f["score"] is not None]
        assert cx["scored"] == len(scores) and cx["sum"] == sum(scores)
        assert cx["mean"] == pytest.approx(sum(scores) / len(scores), abs=0.006)
        assert cx["c"] == {"mean": complexity(scores), "maximum": complexity(scores, aggregation="maximum")}, e["key"]
        rows = tier_sensitivity(mat["exposure"]["share"], reg["literal"], reg["graded"], dec, scores, cuts)
        assert [
            (r["reading"], r["regulatory_use"], r["complexity"], r["m"], r["c"], r["tier"])
            for r in tb["tier_sensitivity"]
        ] == list(rows)
        computed = matrix_tier(mat["m"]["graded"], cx["c"]["mean"])
        assert tb["computed_tier"] == computed and tb["adopted_reading"] == "B"
        override = (
            None
            if tb["override"] is None
            else (tb["override"]["to_tier"], tb["override"]["rationale"], tb["override"]["approver"])
        )
        assert tb["tier"] == apply_override(computed, override)
        a = e["attributes"]
        assert (a["A22"]["value"], a["A23"]["value"], a["A24"]["value"]) == (
            mat["m"]["graded"],
            cx["c"]["mean"],
            tb["tier"],
        )


def test_tier_counts_per_reading_are_reproduced(inventory_doc):
    for rec in inventory_doc["tier_counts"]:
        idx = "ABCD".index(rec["reading"])
        tiers = [e["tier_block"]["tier_sensitivity"][idx]["tier"] for e in inventory_doc["entries"]]
        assert list(tier_counts(tiers)) == rec["counts"]


def test_schedules_are_reproduced(inventory_doc):
    table = _table(inventory_doc["conventions"])
    for e in inventory_doc["entries"]:
        s = e["schedule"]
        review, full, monitoring = next_dates(
            e["tier_block"]["tier"], date.fromisoformat(s["last_validation"]["date"]), table, s["profile"]
        )
        assert s["next_review"] == review.isoformat()
        assert s["next_full_revalidation"] == (full.isoformat() if full else None)
        assert s["next_monitoring"] == (monitoring.isoformat() if monitoring else None)
        assert (s["null_reason"] is None) == (full is not None and monitoring is not None)
        assert e["attributes"]["A35"]["value"] == s["next_review"]


def test_graph_fields_and_completeness_are_reproduced(inventory_doc):
    inv = build_inventory(entries_of(inventory_doc), edges_of(inventory_doc))
    metrics = graph_metrics(inv)
    for e in inventory_doc["entries"]:
        assert metrics[e["key"]] == (e["graph"]["depth"], e["graph"]["fan_in"], e["graph"]["fan_out"])
        assert e["map"]["generation"] == e["graph"]["depth"]
    gens = topological_order(
        [e["key"] for e in inventory_doc["entries"]], [(x["upstream"], x["downstream"]) for x in inventory_doc["edges"]]
    )
    assert {k: i for i, layer in enumerate(gens) for k in layer} == {k: v[0] for k, v in metrics.items()}
    assert attribute_completeness(inv.entries) == inventory_doc["completeness"]["attributes"]


def test_scope_profiles_are_reproduced(inventory_doc):
    for e in inventory_doc["entries"]:
        rec = scope_record(ScopeAnswers(**e["scope"]["answers"]))
        assert [(d.profile, d.in_scope) for d in rec] == [(p["profile"], p["in_scope"]) for p in e["scope"]["profiles"]]


def test_the_inventory_document_with_its_recomputed_fields_satisfies_the_contract_checker(inventory_doc, checker):
    assert checker.violations(inventory_doc) == []


# ---- the findings register


def _register_rows(extract: dict, inventory_doc: dict) -> list[dict]:
    """The rows the governance bake builds, field by field, through the engine functions."""
    tiers = {e["key"]: e["tier_block"]["tier"] for e in inventory_doc["entries"]}
    rows = []
    for key, tier in tiers.items():
        case, fit = key.split("/")
        for v in extract["cases"][case]["variants"]:
            raised = RELEASE[v["code_version"]]
            for f in v["findings"]:
                anchor, anchor_reason = anchor_from_test_row(f["evidence"], v["tests"], document=v["path"])
                share, share_reason = rwea_share_of_anchor(anchor)
                days = deadline_days(f["severity"], tier)
                due = due_date(raised, days)
                status = map_product_status(f["status"])
                rows.append(
                    {
                        "key": f"{case}/{v['variant_id']}/{f['id']}",
                        "ecb_f": severity_to_ecb(f["severity"]),
                        "status": status,
                        "deadline_days": days,
                        "due": due.isoformat(),
                        "ageing_days": ageing_days(raised, AS_OF),
                        "overdue": is_overdue(status, due, AS_OF),
                        "anchor": anchor,
                        "anchor_reason": anchor_reason,
                        "rwea_share": share,
                        "rwea_share_reason": share_reason,
                        "raised": raised.isoformat(),
                        "tier": tier,
                    }
                )
    return rows


def test_every_register_row_is_reproduced(extract, inventory_doc, findings_doc):
    mine = {r["key"]: r for r in _register_rows(extract, inventory_doc)}
    fixture = {r["key"]: r for r in findings_doc["rows"]}
    assert mine.keys() == fixture.keys() and len(fixture) == 81
    for key, row in mine.items():
        fx = fixture[key]
        for field in (
            "ecb_f",
            "status",
            "deadline_days",
            "due",
            "ageing_days",
            "overdue",
            "anchor",
            "rwea_share",
            "raised",
            "tier",
        ):
            assert row[field] == fx[field], (key, field)
        assert row["anchor_reason"] == (fx["anchor_reason"]["en"] if fx["anchor_reason"] else None), key
        assert row["rwea_share_reason"] == (fx["rwea_share_reason"]["en"] if fx["rwea_share_reason"] else None), key


def test_the_register_counts_are_reproduced(findings_doc):
    assert register_counts(findings_doc["rows"]) == findings_doc["counts"]


def test_the_findings_document_satisfies_the_contract_checker(findings_doc, checker):
    assert checker.violations(findings_doc) == []


# ---- the appetite


def _drop_basis(metric: dict) -> dict:
    out = copy.deepcopy(metric)
    out["threshold"].pop("basis")
    return out


def test_the_appetite_metrics_are_reproduced(inventory_doc, findings_doc, appetite_doc):
    mine = appetite_metrics(inventory_doc["entries"], findings_doc["rows"], AS_OF)
    assert [_drop_basis(m) for m in mine] == [_drop_basis(m) for m in appetite_doc["metrics"]]
    assert appetite_population(inventory_doc["entries"], findings_doc["rows"]) == appetite_doc["population"]


def test_the_documents_agree_across_the_bake(inventory_doc, findings_doc, appetite_doc, report_docs, checker):
    assert checker.violations(appetite_doc) == []
    assert (
        checker.cross_violations(
            {"inventory": inventory_doc, "findings": findings_doc, "appetite": appetite_doc, "reports": report_docs}
        )
        == []
    )
