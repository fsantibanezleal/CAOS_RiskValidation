"""Shared helpers of the U9-E1 regulatory tests: the regulatory-calculators fixture, the SA-CCR blocks of the C12
documents
and the trade-level path of the five CRE99 netting sets (transcribed trade inputs, intermediates carried unrounded)."""

from __future__ import annotations

import json
import math
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SQRT2 = math.sqrt(2.0)


def load_calculators() -> dict:
    return json.loads((FIXTURES / "calculators-fixture.json").read_text(encoding="utf-8"))


def load_c12_blocks() -> dict:
    return json.loads((FIXTURES / "c12-blocks.json").read_text(encoding="utf-8"))


def example(doc: dict, example_id: str) -> dict:
    return next(e for e in doc["official_examples"] + doc["invariant_cases"] if e["id"] == example_id)


def named(entries: list[dict]) -> dict[str, object]:
    return {e["name"]: e["value"] for e in entries}


def terms(trade: dict) -> dict[str, object]:
    return {t["name"]: t["value"] for t in trade["terms"]}


def cre99_netting_set(ex: dict) -> dict:
    """Run one CRE99 example from its trades through the engine; returns every intermediate by the fixture's name."""
    from riskvalidation.regulatory import sa_ccr as s

    inputs = named(ex["inputs"])
    mpor = inputs.get("mpor_business_days")
    margined = mpor is not None
    ir, credit, commodity = [], [], {}
    for trade in ex["trades"]:
        t = terms(trade)
        asset = t["asset_class"]
        if asset == "interest-rate":
            if t["option_type"] == "none":
                delta = 1.0 if t["position"] == "pay-fixed-receive-floating" else -1.0
            else:
                assert t["option_type"] == "bought-put"
                delta = s.supervisory_delta_option(
                    t["option_price"],
                    t["option_strike"],
                    t["option_volatility"],
                    t["option_expiry_years"],
                    is_call=False,
                    is_long=True,
                )
            ir.append(
                s.IrTrade(t["currency"], float(t["start_years"]), float(t["end_years"]), float(t["notional"]), delta)
            )
        elif asset == "credit":
            delta = 1.0 if t["position"] == "protection-buyer" else -1.0
            en = s.credit_trade_effective_notional(
                delta, t["notional"], float(t["start_years"]), float(t["end_years"]), mpor
            )
            credit.append(s.CreditEntity(t["rating"], t["reference_kind"] == "index", en))
        else:
            assert asset == "commodity"
            delta = 1.0 if t["position"] == "long" else -1.0
            en = s.commodity_trade_effective_notional(delta, t["adjusted_notional"], float(t["maturity_years"]), mpor)
            key = (t["hedging_set"], t["commodity_type"])
            commodity[key] = commodity.get(key, 0.0) + en
    out: dict[str, object] = {"margined": margined, "mpor": mpor}
    ir_result = s.ir_addon(ir, mpor_business_days=mpor) if ir else None
    out["ir"] = ir_result
    addon_ir = ir_result.addon if ir_result else 0.0
    addon_credit = s.credit_addon(credit) if credit else 0.0
    sf = s.supervisory_parameter("commodity-oil-gas-metals-agricultural-other", "supervisory_factor").value
    rho = s.supervisory_parameter("commodity-oil-gas-metals-agricultural-other", "correlation").value
    sets: dict[str, list] = {}
    for (hedging_set, _kind), en in commodity.items():
        sets.setdefault(hedging_set, []).append(s.CommodityEntity(sf, rho, en))
    addon_commodity = s.commodity_addon(sets) if sets else 0.0
    out.update(addon_ir=addon_ir, addon_credit=addon_credit, addon_commodity=addon_commodity)
    out["commodity_sets"] = {k: s.commodity_addon({k: v}) for k, v in sets.items()}
    res = s.netting_set_ead(
        float(inputs["v"]),
        float(inputs["c"]),
        s.AddOnByClass(interest_rate=addon_ir, credit=addon_credit, commodity=addon_commodity),
        margined=margined,
        th=float(inputs.get("th", 0.0)),
        mta=float(inputs.get("mta", 0.0)),
        nica=float(inputs.get("nica", 0.0)),
    )
    out["result"] = res
    out["credit_entities"] = credit
    return out
