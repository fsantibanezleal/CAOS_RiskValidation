"""RV-926, RV-927: the CRE52.61 systematic and idiosyncratic terms of the credit and equity add-ons
(`credit_addon_terms`, `equity_addon_terms`) and the MAR21.4(5)(b) fallback flag (`cross_bucket_charge_detail`).
The terms are recomputed here by a second route (plain `math`); the existing functions keep their results."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from reg_helpers import cre99_netting_set, example, load_calculators

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
