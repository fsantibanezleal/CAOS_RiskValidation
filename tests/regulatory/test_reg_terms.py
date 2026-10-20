"""RV-926, RV-927: the CRE52.61 systematic and idiosyncratic terms of the credit and equity add-ons
(`credit_addon_terms`, `equity_addon_terms`) and the MAR21.4(5)(b) fallback flag (`cross_bucket_charge_detail`).
The terms are recomputed here by a second route (plain `math`); the existing functions keep their results."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from reg_helpers import cre99_netting_set, example, load_calculators, named

from riskvalidation.regulatory import frtb as f
from riskvalidation.regulatory import sa_ccr as s

DOC = load_calculators()


def _credit_set_2():
    return cre99_netting_set(example(DOC, "cre99-netting-set-2"))


def test_credit_terms_identities_on_cre99_set_2():
    run = _credit_set_2()
    entities = run["credit_entities"]
    assert entities
    terms = s.credit_addon_terms(entities)
    assert isinstance(terms, s.AddonTerms) and dataclasses.is_dataclass(terms)
    addons = [
        s.supervisory_parameter(
            ("credit-index-" if e.is_index else "credit-single-name-") + e.rating, "supervisory_factor"
        ).value
        * e.effective_notional
        for e in entities
    ]
    rhos = [0.8 if e.is_index else 0.5 for e in entities]
    assert terms.systematic == pytest.approx(sum(r * a for r, a in zip(rhos, addons, strict=True)) ** 2, rel=1e-13)
    assert terms.idiosyncratic == pytest.approx(
        sum((1 - r * r) * a * a for r, a in zip(rhos, addons, strict=True)), rel=1e-13
    )
    assert terms.addon == pytest.approx(math.sqrt(terms.systematic + terms.idiosyncratic), rel=1e-14)
    assert terms.addon == s.credit_addon(entities)
    assert terms.addon == pytest.approx(run["addon_credit"], rel=1e-13)


def test_equity_terms_identities():
    entities = [s.EquityEntity(False, 100.0), s.EquityEntity(True, 250.0), s.EquityEntity(False, -40.0)]
    terms = s.equity_addon_terms(entities)
    addons = [(0.20 if e.is_index else 0.32) * e.effective_notional for e in entities]
    rhos = [0.8 if e.is_index else 0.5 for e in entities]
    assert terms.systematic == pytest.approx(sum(r * a for r, a in zip(rhos, addons, strict=True)) ** 2, rel=1e-13)
    assert terms.idiosyncratic == pytest.approx(
        sum((1 - r * r) * a * a for r, a in zip(rhos, addons, strict=True)), rel=1e-13
    )
    assert terms.addon == pytest.approx(math.sqrt(terms.systematic + terms.idiosyncratic), rel=1e-14)
    assert terms.addon == s.equity_addon(entities)


def test_terms_empty_and_refusals_as_the_existing_functions():
    assert s.credit_addon_terms([]) == s.AddonTerms(0.0, 0.0, 0.0)
    assert s.equity_addon_terms([]) == s.AddonTerms(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="rating"):
        s.credit_addon_terms([s.CreditEntity("ZZZ", False, 1.0)])
    with pytest.raises(ValueError, match="effective_notional"):
        s.equity_addon_terms([s.EquityEntity(False, float("nan"))])


def _gamma(n, g):
    m = np.full((n, n), float(g))
    np.fill_diagonal(m, 1.0)
    return m


def test_cross_bucket_detail_flags_the_fallback():
    k = np.array([17.320508075688775] * 2)
    fb = f.cross_bucket_charge_detail(k, np.array([20.0, -20.0]), _gamma(2, 0.8))
    assert isinstance(fb, f.CrossBucket)
    assert fb.fallback_used is True
    assert fb.charge == pytest.approx(10.954451150103322, rel=1e-12)
    assert fb.charge == f.cross_bucket_charge(k, np.array([20.0, -20.0]), _gamma(2, 0.8))
    plain = f.cross_bucket_charge_detail(np.array([3.0, 4.0]), np.array([1.0, 1.0]), _gamma(2, 0.5))
    assert plain.fallback_used is False
    assert plain.charge == f.cross_bucket_charge(np.array([3.0, 4.0]), np.array([1.0, 1.0]), _gamma(2, 0.5))


def test_cross_bucket_detail_matches_the_plain_function_over_scenarios():
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(200):
        k = rng.uniform(0.0, 10.0, 3)
        sb = rng.uniform(-30.0, 30.0, 3)
        gamma = _gamma(3, rng.uniform(0.0, 1.0))
        for scenario in ("low", "medium", "high"):
            d = f.cross_bucket_charge_detail(k, sb, gamma, scenario)
            assert d.charge == f.cross_bucket_charge(k, sb, gamma, scenario)
            g = f.scenario_correlation(gamma, scenario)
            np.fill_diagonal(g, 0.0)
            assert d.fallback_used is bool(float(k @ k + sb @ g @ sb) < 0.0)
            seen.add(d.fallback_used)
    assert seen == {True, False}


def test_cross_bucket_detail_refuses_as_the_plain_function():
    with pytest.raises(ValueError, match="k_b"):
        f.cross_bucket_charge_detail(np.array([-1.0]), np.array([1.0]), np.eye(1))


# RV-928: the CRE52.34 trade effective notionals of credit and commodity trades


def test_credit_trade_effective_notional_formula_and_refusals():
    for mpor in (None, 10, 20):
        got = s.credit_trade_effective_notional(-1.0, 10_000.0, 0.0, 5.0, mpor)
        want = -1.0 * 10_000.0 * s.supervisory_duration(0.0, 5.0) * s.maturity_factor(5.0, mpor)
        assert got == want
    assert s.credit_trade_effective_notional(1.0, 100.0, 1.0, 3.0) == pytest.approx(
        100.0 * (math.exp(-0.05) - math.exp(-0.15)) / 0.05 * 1.0, rel=1e-14
    )
    with pytest.raises(ValueError, match="end"):
        s.credit_trade_effective_notional(1.0, 100.0, 3.0, 1.0)
    with pytest.raises(ValueError, match="mpor_business_days"):
        s.credit_trade_effective_notional(1.0, 100.0, 0.0, 1.0, 0)
    with pytest.raises(ValueError, match="delta"):
        s.credit_trade_effective_notional(float("nan"), 100.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="notional"):
        s.credit_trade_effective_notional(1.0, float("inf"), 0.0, 1.0)


def test_commodity_trade_effective_notional_formula_and_refusals():
    for mpor in (None, 10, 20):
        got = s.commodity_trade_effective_notional(1.0, 2_500.0, 0.5, mpor)
        assert got == 1.0 * 2_500.0 * s.maturity_factor(0.5, mpor)
    assert s.commodity_trade_effective_notional(-1.0, 100.0, 4.0) == -100.0
    with pytest.raises(ValueError, match="maturity_years"):
        s.commodity_trade_effective_notional(1.0, 100.0, -1.0)
    with pytest.raises(ValueError, match="mpor_business_days"):
        s.commodity_trade_effective_notional(1.0, 100.0, 1.0, -5)
    with pytest.raises(ValueError, match="adjusted_notional"):
        s.commodity_trade_effective_notional(1.0, float("nan"), 1.0)


def test_cre99_entity_effective_notionals_reproduce_through_the_new_functions():
    from reg_helpers import terms

    checked = 0
    for ex_id in ("cre99-netting-set-2", "cre99-netting-set-3", "cre99-netting-set-4", "cre99-netting-set-5"):
        ex = example(DOC, ex_id)
        mpor = named(ex["inputs"]).get("mpor_business_days")
        run = cre99_netting_set(ex)
        credit, commodity = [], {}
        for trade in ex["trades"]:
            t = terms(trade)
            if t["asset_class"] == "credit":
                d = 1.0 if t["position"] == "protection-buyer" else -1.0
                en = s.credit_trade_effective_notional(
                    d, t["notional"], float(t["start_years"]), float(t["end_years"]), mpor
                )
                credit.append(s.CreditEntity(t["rating"], t["reference_kind"] == "index", en))
            elif t["asset_class"] == "commodity":
                d = 1.0 if t["position"] == "long" else -1.0
                en = s.commodity_trade_effective_notional(d, t["adjusted_notional"], float(t["maturity_years"]), mpor)
                key = (t["hedging_set"], t["commodity_type"])
                commodity[key] = commodity.get(key, 0.0) + en
        if credit:
            assert credit == run["credit_entities"]
            assert s.credit_addon(credit) == run["addon_credit"]
            checked += 1
        if commodity:
            sf = s.supervisory_parameter("commodity-oil-gas-metals-agricultural-other", "supervisory_factor").value
            rho = s.supervisory_parameter("commodity-oil-gas-metals-agricultural-other", "correlation").value
            sets: dict[str, list] = {}
            for (hs, _k), en in commodity.items():
                sets.setdefault(hs, []).append(s.CommodityEntity(sf, rho, en))
            assert s.commodity_addon(sets) == run["addon_commodity"]
            checked += 1
        assert round(run["result"].ead) == {2: 381, 3: 5406, 4: 936, 5: 1879}[int(ex_id[-1])]
    assert checked >= 2
