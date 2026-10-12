"""RV-512: overlays as explicit line items, never merged into model ECL (dossier 02 I.1 item 9; ECB July 2024;
d350 paras 50 and 51)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from riskvalidation.provisions import ecl, overlays


def ovl(oid="OVL-RATES", amount=1500.0, stage="stage1", **kw) -> overlays.Overlay:
    base = {
        "id": oid, "rationale_en": "Interest-rate shock", "rationale_es": "Shock de tasas", "method": "expert",
        "target": "pd", "segment_id": "s1", "stage": stage, "amount": amount, "staging_effect": "none",
        "review_date": "2020-12-31", "currency": "USD",
    }
    base.update(kw)
    return overlays.Overlay(**base)


def test_overlay_record_validation_and_lines():
    o = ovl()
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.amount = 1.0  # type: ignore[misc]
    for kw in (
        {"method": "guess"}, {"target": "ecl"}, {"staging_effect": "transfers-to-stage3"}, {"stage": "stage4"},
        {"amount": float("nan")}, {"amount": float("inf")}, {"id": ""}, {"review_date": "31/12/2020"},
        {"currency": "usd"},
    ):
        with pytest.raises(ValueError):
            ovl(**kw)
    assert ovl(stage=None).stage is None  # portfolio level
    assert ovl(segment_id=None).segment_id is None
    lines = overlays.overlay_lines([o])
    assert lines == [{"id": "overlay-OVL-RATES", "kind": "overlay", "overlay_id": "OVL-RATES", "amount": 1500.0,
                      "stage": "stage1"}]
    # one line per overlay, in id order, never merged; null stage stays null
    many = [ovl("B", 2.5, stage=None), ovl("A", -1.0, stage="stage2"), ovl("C", 3.0)]
    ls = overlays.overlay_lines(many)
    assert [x["overlay_id"] for x in ls] == ["A", "B", "C"]
    assert [x["id"] for x in ls] == ["overlay-A", "overlay-B", "overlay-C"]
    assert ls[1]["stage"] is None
    assert sum(x["amount"] for x in ls) == sum(x.amount for x in many)
    assert overlays.overlay_lines([]) == []
    with pytest.raises(ValueError, match="unique"):
        overlays.overlay_lines([ovl("A"), ovl("A", 2.0)])


def test_overlay_share_total_allowance_and_separation():
    assert overlays.overlay_share(5.0, 80.0) == 0.0625
    assert overlays.overlay_share(0.0, 80.0) == 0.0
    assert 0.0 <= overlays.overlay_share(80.0, 80.0) <= 1.0
    for total in (0.0, -1.0):
        with pytest.raises(ValueError, match="total_allowance"):
            overlays.overlay_share(5.0, total)
    assert overlays.total_allowance(75.0, [ovl("O", 5.0)]) == 80.0
    assert overlays.total_allowance(75.0, []) == 75.0
    with pytest.raises(ValueError, match="unique"):
        overlays.total_allowance(75.0, [ovl("A"), ovl("A")])
    # equals the portfolio allowance for the same inputs, and overlays stay out of the model ECL
    stage = np.array([1, 2, 3])
    pa = ecl.portfolio_allowance(stage, np.array([10.0, 20.0, 30.0]), np.array([15.0, 40.0, 45.0]),
                                 np.array([0.0, 0.0, 25.0]), overlays=[ovl("O", 5.0)])
    assert pa.total == overlays.total_allowance(pa.model_total, [ovl("O", 5.0)])
    no_overlay = ecl.portfolio_allowance(stage, np.array([10.0, 20.0, 30.0]), np.array([15.0, 40.0, 45.0]),
                                         np.array([0.0, 0.0, 25.0]))
    assert no_overlay.by_stage == pa.by_stage and no_overlay.model_total == pa.model_total
