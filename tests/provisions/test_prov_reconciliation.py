"""RV-513: the IFRS 7 35H and 35I roll-forwards and the identity check, on the contract's C06 reporting-date fixture
(copied from the Contraste fixture `variant-reporting-date-fixture.json`: outputs.waterfall, overlays,
reconciliation)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from riskvalidation.provisions import reconciliation as rc
from riskvalidation.provisions.overlays import Overlay

FIXTURE = Path(__file__).parent / "fixtures" / "c06_reporting_date_reconciliation.json"


def load():
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return d


def rows(d, key):
    make = rc.allowance_rollforward if key == "allowance" else rc.gca_rollforward
    return [make(r["stage"], r["opening"], {m["line_id"]: m["amount"] for m in r["movements"]}, r["closing"])
            for r in d["reconciliation"][key]]


def overlays_of(d):
    return [Overlay(id=o["id"], rationale_en=o["rationale"]["en"], rationale_es=o["rationale"]["es"],
                    method=o["method"], target=o["target"], segment_id=o["segment_id"], stage=o["stage"],
                    amount=o["amount"],
                    staging_effect=o["staging_effect"], review_date=o["review_date"], currency=o["currency"])
            for o in d["overlays"]]


def stage_ecl(d):
    return {r["stage"]: r["ecl"] for r in d["stage_summary"]["rows"]}


def test_rollforward_rows_goldens_and_refusals():
    a = rc.allowance_rollforward("stage1", 1000.0, {"originations": 200.0, "write-off": -50.0}, 1150.0)
    assert a.residual == 0.0 and a.opening == 1000.0 and a.closing == 1150.0
    assert a.movements == (("originations", 200.0), ("write-off", -50.0))  # order preserved
    g = rc.gca_rollforward("stage2", 100000.0, {"repayments": -5000.0, "originations": 20000.0}, 115000.0)
    assert g.residual == 0.0
    assert [m[0] for m in g.movements] == ["repayments", "originations"]
    # the residual is the difference itself, not clipped or tolerance-adjusted
    off = rc.allowance_rollforward("stage2", 100.0, {"originations": 10.0}, 110.5)
    assert off.residual == pytest.approx(0.5, abs=1e-12)
    under = rc.gca_rollforward("stage1", 100.0, {"repayments": -10.0}, 80.0)
    assert under.residual == pytest.approx(-10.0, abs=1e-12)
    # the allowance of stages 1 to 3 is not negative; POCI may be (5.5.14)
    with pytest.raises(ValueError, match="allowance"):
        rc.allowance_rollforward("stage3", 10.0, {"write-off": -20.0}, -10.0)
    assert rc.allowance_rollforward("poci", 10.0, {"remeasurement": -20.0}, -10.0).residual == 0.0
    with pytest.raises(ValueError, match="stage"):
        rc.allowance_rollforward("stage9", 0.0, {}, 0.0)
    with pytest.raises(ValueError, match="line_id"):
        rc.gca_rollforward("stage1", 0.0, {"mystery": 1.0}, 1.0)
    with pytest.raises(ValueError, match="finite"):
        rc.allowance_rollforward("stage1", float("nan"), {}, 0.0)
    with pytest.raises(ValueError, match="gross carrying amount"):
        rc.gca_rollforward("stage1", -1.0, {}, -1.0)
    assert rc.allowance_rollforward("stage1", 0.0, {}, 0.0).movements == ()


def test_check_identities_on_the_c06_fixture():
    d = load()
    out = rc.check_identities(rows(d, "allowance"), rows(d, "gross_carrying_amount"), d["waterfall"]["lines"],
                              stage_ecl(d), overlays_of(d))
    assert out == []
    # the stage 2 closing 1758.76 differs from its ECL 1758.7575 by 0.0025, inside 1e-6 * 230000 = 0.23
    assert abs(1758.76 - stage_ecl(d)["stage2"]) == pytest.approx(0.0025, abs=1e-9)
    gca_total = sum(abs(r.closing) for r in rows(d, "gross_carrying_amount"))
    assert gca_total == 230000.0
    # moving one movement by 1.0 fails an identity
    moved = copy.deepcopy(d)
    moved["reconciliation"]["allowance"][1]["movements"][0]["amount"] += 1.0
    bad = rc.check_identities(rows(moved, "allowance"), rows(d, "gross_carrying_amount"), d["waterfall"]["lines"],
                              stage_ecl(d), overlays_of(d))
    assert bad
    # a movement moved by 0.1 stays inside the tolerance 0.23
    small = copy.deepcopy(d)
    small["reconciliation"]["allowance"][1]["movements"][0]["amount"] += 0.1
    assert rc.check_identities(rows(small, "allowance"), rows(d, "gross_carrying_amount"), d["waterfall"]["lines"],
                               stage_ecl(d), overlays_of(d)) == []


def test_check_identities_names_each_failure():
    d = load()
    a, g, w = rows(d, "allowance"), rows(d, "gross_carrying_amount"), d["waterfall"]["lines"]
    e, o = stage_ecl(d), overlays_of(d)
    # allowance roll-forward residual
    broken = copy.deepcopy(d)
    broken["reconciliation"]["allowance"][0]["closing"] += 5.0
    got = rc.check_identities(rows(broken, "allowance"), g, w, e, o)
    assert "allowance-rollforward:stage1" in got
    assert "waterfall-equals-stage-closings" in got and "stage-closing-equals-ecl:stage1" in got
    # gca roll-forward residual
    gb = copy.deepcopy(d)
    gb["reconciliation"]["gross_carrying_amount"][2]["movements"][1]["amount"] += 3.0
    assert rc.check_identities(a, rows(gb, "gross_carrying_amount"), w, e, o) == ["gca-rollforward:stage3"]
    # waterfall that does not close
    wb = copy.deepcopy(w)
    wb[-1]["amount"] += 10.0
    got = rc.check_identities(a, g, wb, e, o)
    assert "waterfall-closes" in got and "waterfall-equals-stage-closings" in got
    # a line whose amount is not the sum of the stage movements; an unknown line id
    lb = copy.deepcopy(w)
    next(x for x in lb if x["id"] == "transfer-1-2")["amount"] += 7.0
    got = rc.check_identities(a, g, lb, e, o)
    assert "waterfall-lines-equal-movements" in got and "waterfall-closes" in got
    unknown = [x for x in copy.deepcopy(w) if x["id"] != "unwinding"]
    assert "waterfall-lines-equal-movements" in rc.check_identities(a, g, unknown, e, o)
    # a stage ECL that does not match the closing; the overlay carried in stage1 matters
    eb = dict(e)
    eb["stage3"] += 50.0
    assert rc.check_identities(a, g, w, eb, o) == ["stage-closing-equals-ecl:stage3"]
    assert "stage-closing-equals-ecl:stage1" in rc.check_identities(a, g, w, e, [])
    # an overlay with no stage is carried in stage1 (the convention of the contract)
    portfolio = [Overlay(**{**o[0].__dict__, "stage": None})]
    assert rc.check_identities(a, g, w, e, portfolio) == []
    # the function never raises on a mismatch, and the tolerance scales with the total gross carrying amount
    assert rc.check_identities(a, g, w, e, o, tolerance=1e-12) == ["stage-closing-equals-ecl:stage2"]


def test_identity_residuals_are_the_largest_absolute_differences():
    d = load()
    res = rc.identity_residuals(rows(d, "allowance"), rows(d, "gross_carrying_amount"), d["waterfall"]["lines"],
                                stage_ecl(d), overlays_of(d))
    assert max(v for k, v in res.items() if not k.startswith("stage-closing-equals-ecl")) <= 1e-9
    assert res["stage-closing-equals-ecl:stage2"] == pytest.approx(0.0025, abs=1e-9)


def synthetic_document(rng):
    """A random consistent document with a POCI row and a portfolio-level overlay, unlike the C06 fixture:
    every identity closes by construction."""
    line_ids = ["originations", "transfer-1-2", "transfer-2-3", "remeasurement", "write-off", "unwinding"]
    stages = ["stage1", "stage2", "stage3", "poci"]
    overlay = Overlay(id="PORT", rationale_en="r", rationale_es="r", method="scenario", target="staging-share",
                      segment_id=None, stage=None, amount=float(rng.uniform(5, 50)), staging_effect="none",
                      review_date="2024-06-30", currency="CLP")
    rows, ecl_by_stage = [], {}
    for stage in stages:
        opening = float(rng.uniform(100, 5000))
        moves = {lid: float(rng.uniform(-50, 200)) for lid in line_ids}
        if stage == "stage1":
            moves["overlay-PORT"] = overlay.amount  # a portfolio-level overlay is carried in the stage1 closing
        closing = opening + sum(moves.values())
        ecl_by_stage[stage] = closing - (overlay.amount if stage == "stage1" else 0.0)
        rows.append(rc.allowance_rollforward(stage, opening, moves, closing))
    waterfall = [{"id": "opening", "kind": "opening", "amount": sum(r.opening for r in rows), "stage": None}]
    for lid in line_ids:
        waterfall.append({"id": lid, "kind": lid, "amount": sum(dict(r.movements)[lid] for r in rows), "stage": None})
    waterfall.append({"id": "overlay-PORT", "kind": "overlay", "amount": overlay.amount, "stage": None})
    waterfall.append({"id": "closing", "kind": "closing", "amount": sum(r.closing for r in rows), "stage": None})
    gca = [rc.gca_rollforward(s, 1e6, {"repayments": -1e5, "originations": 3e5}, 1.2e6) for s in stages]
    return rows, gca, waterfall, ecl_by_stage, overlay


def test_identities_close_on_random_documents_and_open_on_any_perturbation():
    rng = np.random.default_rng(11)
    for _ in range(20):
        rows, gca, waterfall, e, overlay = synthetic_document(rng)
        assert rc.check_identities(rows, gca, waterfall, e, [overlay]) == []
        # without the overlay the stage1 closing no longer equals its ECL
        assert "stage-closing-equals-ecl:stage1" in rc.check_identities(rows, gca, waterfall, e, [])
        tol = 1e-6 * max(1.0, rc.gross_total(gca))
        for k, r in enumerate(rows):
            bad = list(rows)
            bad[k] = rc.RollForwardRow(r.stage, r.opening, r.movements, r.closing + 10 * tol, r.residual)
            assert f"allowance-rollforward:{r.stage}" in rc.check_identities(bad, gca, waterfall, e, [overlay])
        # a gap inside the tolerance does not fail an identity
        r = rows[0]
        near = [rc.RollForwardRow(r.stage, r.opening, r.movements, r.closing + 0.1 * tol, r.residual), *rows[1:]]
        assert "allowance-rollforward:stage1" not in rc.check_identities(near, gca, waterfall, e, [overlay])
