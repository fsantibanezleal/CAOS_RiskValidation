"""RV-915: the five CRE99 netting sets of SA-CCR (CRE99.20 to CRE99.97, USD thousands) from their transcribed trade
inputs. The trade-level path carries every intermediate unrounded (CRE99.20): `round(ead)` equals 569, 381, 5,406, 936
and 1,879 exactly (dossier 01 M.3 item 2) and every intermediate of dossier 01 E.2 held by the fixture is reproduced
step by step. The
path from the printed (rounded) intermediates holds to one unit."""

from __future__ import annotations

import math

import pytest
from reg_helpers import cre99_netting_set, example, load_calculators, named

from riskvalidation.regulatory import sa_ccr as s

DOC = load_calculators()
IDS = (
    "cre99-netting-set-1",
    "cre99-netting-set-2",
    "cre99-netting-set-3",
    "cre99-netting-set-4",
    "cre99-netting-set-5",
)
PUBLISHED_EAD = (569, 381, 5406, 936, 1879)


def test_cre99_five_netting_sets_round_exactly():
    got = []
    for ex_id in IDS:
        ex = example(DOC, ex_id)
        run = cre99_netting_set(ex)
        res = run["result"]
        published = named(ex["published"])["ead"]
        assert round(res.ead) == published, (ex_id, res.ead)
        got.append(round(res.ead))
        # alpha (RC + PFE) with PFE = multiplier x AddOn
        assert res.ead == pytest.approx(1.4 * (res.rc + res.multiplier * res.addon_aggregate), rel=1e-14)
    assert got == list(PUBLISHED_EAD)


def test_cre99_intermediates_step_by_step():
    checked = 0
    for ex_id in IDS:
        ex = example(DOC, ex_id)
        run = cre99_netting_set(ex)
        res = run["result"]
        want = named(ex["outputs"])
        have = {
            "rc": res.rc,
            "addon_aggregate": res.addon_aggregate,
            "multiplier": res.multiplier,
            "pfe": res.pfe,
            "ead": res.ead,
            "addon_ir": run["addon_ir"],
            "addon_credit": run["addon_credit"],
            "addon_commodity": run["addon_commodity"],
        }
        ir = run["ir"]
        if ir is not None:
            by_ccy = {h.hedging_set: h for h in ir.hedging_sets}
            if "USD" in by_ccy and "EUR" in by_ccy:
                have.update(
                    en_usd=by_ccy["USD"].effective_notional,
                    en_eur=by_ccy["EUR"].effective_notional,
                    addon_usd=by_ccy["USD"].addon,
                    addon_eur=by_ccy["EUR"].addon,
                )
            swaption = [
                t
                for t in ex["trades"]
                if any(x["name"] == "option_type" and x["value"] == "bought-put" for x in t["terms"])
            ]
            if swaption:
                terms = {x["name"]: x["value"] for x in swaption[0]["terms"]}
                have["swaption_delta"] = s.supervisory_delta_option(
                    terms["option_price"],
                    terms["option_strike"],
                    terms["option_volatility"],
                    terms["option_expiry_years"],
                    is_call=False,
                    is_long=True,
                )
        if run["credit_entities"]:
            addons = [
                s.supervisory_parameter(
                    ("credit-index-" if e.is_index else "credit-single-name-") + e.rating, "supervisory_factor"
                ).value
                * e.effective_notional
                for e in run["credit_entities"]
            ]
            rhos = [0.8 if e.is_index else 0.5 for e in run["credit_entities"]]
            have["term_1"] = sum(r * a for r, a in zip(rhos, addons, strict=True)) ** 2
            have["term_2"] = sum((1 - r * r) * a * a for r, a in zip(rhos, addons, strict=True))
            assert run["addon_credit"] == pytest.approx(math.sqrt(have["term_1"] + have["term_2"]), rel=1e-13)
        for name, value in run["commodity_sets"].items():
            have[f"addon_{name}"] = value
        if run["margined"]:
            have["maturity_factor"] = s.maturity_factor(1.0, run["mpor"])
        for name, expected in want.items():
            if name in ("rwa_floored",):
                continue
            assert name in have, (ex_id, name)
            assert have[name] == pytest.approx(expected, rel=1e-9, abs=1e-9), (ex_id, name, have[name], expected)
            checked += 1
    assert checked == sum(len(example(DOC, i)["outputs"]) for i in IDS)  # every intermediate of the fixture


def test_cre99_printed_path_within_one_unit():
    """From the printed intermediates (rounded add-ons per class and the printed RC) the EAD holds to one unit."""
    for ex_id in IDS:
        ex = example(DOC, ex_id)
        printed = named(ex["published"])
        inputs = named(ex["inputs"])
        v, c = float(inputs["v"]), float(inputs["c"])
        addons = s.AddOnByClass(
            interest_rate=float(printed.get("addon_ir", 0.0)),
            credit=float(printed.get("addon_credit", 0.0)),
            commodity=float(printed.get("addon_commodity", 0.0)),
        )
        assert addons.interest_rate + addons.credit + addons.commodity == pytest.approx(
            printed["addon_aggregate"], abs=1.0
        )
        margined = "mpor_business_days" in inputs
        res = s.netting_set_ead(
            v,
            c,
            addons,
            margined=margined,
            th=float(inputs.get("th", 0.0)),
            mta=float(inputs.get("mta", 0.0)),
            nica=float(inputs.get("nica", 0.0)),
        )
        assert abs(res.ead - printed["ead"]) <= 1.0, (ex_id, res.ead, printed["ead"])
        assert res.rc == pytest.approx(printed["rc"], abs=1e-9)
        assert res.multiplier == pytest.approx(printed["multiplier"], abs=5e-4)
    # the one case where the printed add-ons and the printed EAD disagree (CRE99.77): 1.4 x 669 = 936.6, printed 936
    ns4 = example(DOC, "cre99-netting-set-4")
    p4 = named(ns4["published"])
    assert 1.4 * (p4["rc"] + p4["addon_aggregate"]) == pytest.approx(936.6) and p4["ead"] == 936
    assert round(cre99_netting_set(ns4)["result"].ead) == 936


def test_cre99_trade_inputs_are_transcribed_with_paragraphs():
    """Every trade of the fixture names its BIS paragraph: the trade-level path starts from the page, not from the
    printed intermediates."""
    for ex_id in IDS:
        ex = example(DOC, ex_id)
        assert ex["trades"], ex_id
        for trade in ex["trades"]:
            assert trade["paragraph"].startswith("CRE99.")
            assert all(t["basis"] == "transcribed" for t in trade["terms"])
