"""RV-924 and RV-925: the regulatory modules import with only NumPy and SciPy, do not mutate their inputs, are
deterministic and
refuse an invalid input naming the field; the documentation pages carry their equations, sources and limits."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "regulatory" / "u9"
NAMES = (
    "01_standardised-credit.md",
    "02_output-floor.md",
    "03_sa-ccr.md",
    "04_ba-cva.md",
    "05_frtb-sbm-and-cascade.md",
)
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib", "QuantLib")
NAN = float("nan")


def test_modules_import_no_dataframe_library():
    code = (
        "import sys, riskvalidation.regulatory.standardised, riskvalidation.regulatory.output_floor,"
        " riskvalidation.regulatory.sa_ccr, riskvalidation.regulatory.ba_cva, riskvalidation.regulatory.frtb;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.regulatory import ba_cva, frtb, output_floor, sa_ccr, standardised

    arr = np.array
    tenors = arr(frtb.girr_parameters().tenors)
    params = frtb.girr_parameters()
    rho3 = arr([[1.0, 0.5, 0.2], [0.5, 1.0, 0.5], [0.2, 0.5, 1.0]])
    gamma2 = arr([[1.0, 0.5], [0.5, 1.0]])
    trades = [sa_ccr.IrTrade("USD", 0.0, 10.0, 1e6, 1.0), sa_ccr.IrTrade("USD", 0.0, 3.0, 5e5, -1.0)]
    zero3 = {"low": 1.0, "medium": 2.0, "high": 3.0}
    return [
        (standardised.sovereign_risk_weight, ("A+ to A-",)),
        (standardised.eca_risk_weight, (3,)),
        (standardised.bank_risk_weight_ecra, ("BBB+ to BBB-", True)),
        (standardised.bank_risk_weight_scra, ("B", False)),
        (standardised.corporate_risk_weight, ("unrated", True)),
        (standardised.retail_risk_weight, ("regulatory",)),
        (standardised.equity_risk_weight, ("equity",)),
        (standardised.residential_re_risk_weight, (0.7, False)),
        (standardised.commercial_re_risk_weight, (0.5, False, 0.3)),
        (standardised.apply_currency_mismatch, (0.75,)),
        (standardised.credit_conversion_factor, ("commitment",)),
        (standardised.defaulted_risk_weight, (0.1,)),
        (standardised.lookup, ("bank-ecra", "A+ to A-")),
        (output_floor.phase_in_factor, (2026,)),
        (output_floor.floored_rwa, (76.0, 140.0, 0.725, 0.25)),
        (sa_ccr.supervisory_duration, (0.0, 5.0)),
        (sa_ccr.supervisory_delta_option, (0.06, 0.05, 0.5, 1.0, False, True)),
        (sa_ccr.supervisory_delta_tranche, (0.0, 0.1, True)),
        (sa_ccr.maturity_factor, (1.0, 14)),
        (sa_ccr.maturity_bucket, (3.0,)),
        (sa_ccr.ir_effective_notional, (100.0, -100.0, 20.0)),
        (sa_ccr.ir_addon, (trades, 10)),
        (sa_ccr.fx_addon, ({"EURUSD": 100.0, "USDJPY": -40.0},)),
        (sa_ccr.entity_aggregate, ([1.0, -2.0], [0.5, 0.8])),
        (sa_ccr.credit_addon, ([sa_ccr.CreditEntity("AA", False, 1e4), sa_ccr.CreditEntity("IG", True, -3e3)],)),
        (sa_ccr.equity_addon, ([sa_ccr.EquityEntity(False, 100.0)],)),
        (sa_ccr.commodity_addon, ({"energy": [sa_ccr.CommodityEntity(0.18, 0.4, 100.0)]},)),
        (sa_ccr.replacement_cost, (80.0, 200.0, 0.0, 5.0, 150.0, True)),
        (sa_ccr.multiplier, (-20.0, 282.0)),
        (sa_ccr.netting_set_ead, (60.0, 0.0, sa_ccr.AddOnByClass(interest_rate=347.0))),
        (ba_cva.materiality_threshold_applies, (5e10,)),
        (ba_cva.sector_risk_weight, ("financials", True)),
        (ba_cva.discount_factor, (5.0,)),
        (ba_cva.scva, (0.05, [(5.0, 100.0)])),
        (ba_cva.k_reduced, ([10.0, 20.0],)),
        (ba_cva.k_hedged, ([8.0, 15.0], 2.0, [0.1, 0.2])),
        (ba_cva.k_full, (30.0, 10.0)),
        (ba_cva.capital, (100.0,)),
        (frtb.girr_parameters, ()),
        (frtb.scenario_correlation, (rho3, "low")),
        (frtb.bucket_charge, (arr([10.0, -4.0, 6.0]), rho3, "high")),
        (frtb.cross_bucket_charge, (arr([10.0, 7.0]), arr([12.0, 5.0]), gamma2, "medium")),
        (frtb.girr_delta_charge, (arr([0, -1e6, 0, 0, 1.16e7, -2.7e7, 8e7, 0, 0, 0.0]), tenors, params)),
        (
            frtb.curvature_charge,
            (arr([[3.0, -2.0], [1.0, 4.0]]), arr([[-1.0, 2.0], [0.5, -3.0]]), arr([[1.0, 0.5], [0.5, 1.0]]), gamma2),
        ),
        (frtb.sbm_capital, (zero3, zero3, zero3)),
        (frtb.liquidity_cascade_es, (10.0, {20: 6.0, 60: 2.0})),
    ]


def _same(r1, r2):
    if isinstance(r1, np.ndarray):
        return np.array_equal(r1, r2, equal_nan=True) if r1.dtype.kind in "fc" else np.array_equal(r1, r2)
    if hasattr(r1, "__dataclass_fields__"):
        return all(_same(getattr(r1, k), getattr(r2, k)) for k in r1.__dataclass_fields__)
    if isinstance(r1, dict):
        return r1.keys() == r2.keys() and all(_same(r1[k], r2[k]) for k in r1)
    if isinstance(r1, (tuple, list)):
        return len(r1) == len(r2) and all(_same(p, q) for p, q in zip(r1, r2, strict=True))
    return r1 == r2


def test_inputs_are_not_mutated_and_results_are_deterministic():
    calls = _calls()
    assert len(calls) >= 40
    for fn, args in calls:
        before = copy.deepcopy(args)
        first = fn(*args)
        second = fn(*args)
        assert _same(before, args), f"{fn.__name__} mutated an input"
        assert _same(first, second), f"{fn.__name__} is not deterministic"


def test_invalid_inputs_name_the_field():
    from riskvalidation.regulatory import ba_cva, frtb, output_floor, sa_ccr, standardised

    rho = np.array([[1.0, 0.5], [0.5, 1.0]])
    cases = [
        (standardised.residential_re_risk_weight, (NAN,), "ltv"),
        (standardised.apply_currency_mismatch, (NAN,), "weight"),
        (standardised.defaulted_risk_weight, (-1.0,), "specific_provision_share"),
        (standardised.eca_risk_weight, (9,), "score"),
        (output_floor.floored_rwa, (NAN, 1.0, 0.5), "rwa_nominated"),
        (output_floor.floored_rwa, (1.0, 1.0, -0.5), "factor"),
        (output_floor.phase_in_factor, (2000,), "year"),
        (sa_ccr.supervisory_duration, (1.0, 0.5), "end"),
        (sa_ccr.supervisory_delta_option, (0.06, 0.05, -0.5, 1.0, True, True), "vol"),
        (sa_ccr.supervisory_delta_tranche, (0.5, 0.2, True), "detachment"),
        (sa_ccr.maturity_factor, (1.0, -1), "mpor_business_days"),
        (sa_ccr.ir_effective_notional, (1.0, 2.0, NAN), "d_b3"),
        (sa_ccr.fx_addon, ({"EURUSD": NAN},), "hedging_sets"),
        (sa_ccr.entity_aggregate, ([1.0], [0.5, 0.5]), "rhos"),
        (sa_ccr.replacement_cost, (1.0, NAN), "c"),
        (sa_ccr.multiplier, (1.0, 1.0, 1.0), "floor"),
        (sa_ccr.netting_set_ead, (1.0, 1.0, sa_ccr.AddOnByClass(), False, 0.0, 0.0, 0.0, -1.4), "alpha"),
        (ba_cva.materiality_threshold_applies, (NAN,), "aggregate_notional_eur"),
        (ba_cva.discount_factor, (0.0,), "m"),
        (ba_cva.scva, (-0.1, [(1.0, 1.0)]), "risk_weight"),
        (ba_cva.k_reduced, ([1.0], 2.0), "rho"),
        (ba_cva.k_hedged, ([1.0], 0.0, []), "hma"),
        (ba_cva.k_full, (1.0, 1.0, 2.0), "beta"),
        (ba_cva.capital, (NAN,), "k"),
        (frtb.scenario_correlation, (rho, "none"), "scenario"),
        (frtb.bucket_charge, (np.array([1.0, NAN]), rho), "ws"),
        (frtb.cross_bucket_charge, (np.array([1.0]), np.array([1.0, 2.0]), np.eye(1)), "s_b"),
        (frtb.girr_delta_charge, (np.zeros(3), np.zeros(3), frtb.girr_parameters()), "sensitivities"),
        (frtb.curvature_charge, (np.zeros((1, 2)), np.zeros((1, 3)), rho, np.eye(1)), "cvr_minus"),
        (frtb.sbm_capital, ({"low": 1.0}, {"low": 1.0}, {"low": 1.0}), "delta"),
        (frtb.liquidity_cascade_es, (NAN, {}), "es_t_full"),
    ]
    for fn, args, field in cases:
        with pytest.raises(ValueError, match=field):
            fn(*args)


def test_u9_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    assert "states no compliance claim" in " ".join(index.split())  # whitespace-insensitive: the page wraps lines
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv:\d{4}\.\d{4,5}|https://", text), f"{name} cites no source"
        assert "Symbols" in text or "symbols" in text, f"{name} defines no symbols"
        assert "## Limits" in text, f"{name} states no limits"
        assert "Golden" in text or "golden" in text, f"{name} states no golden value"
        assert "in force" in text.lower() or "in-force" in text.lower(), f"{name} states no in-force date"
    standardised = (PAGES / "01_standardised-credit.md").read_text(encoding="utf-8")
    for needle in ("CRE20.7", "CRE20.18", "CRE20.43", "CRE20.82", "CRE20.92", "CRE20.94", "CRE20.104", "cre20_grid"):
        assert needle in standardised, needle
    floor = (PAGES / "02_output-floor.md").read_text(encoding="utf-8")
    for needle in ("RBC20.13", "101.5", "95", "RBC90.1", "RBC90.2", "0.725"):
        assert needle in floor, needle
    sa = (PAGES / "03_sa-ccr.md").read_text(encoding="utf-8")
    for needle in ("569", "381", "5,406", "936", "1,879", "CRE99", "UNVERIFIED", "0.965208", "-0.2694", "CRE52.57"):
        assert needle in sa, needle
    cva = (PAGES / "04_ba-cva.md").read_text(encoding="utf-8")
    for needle in ("MAR50.14", "MAR50.16", "UNVERIFIED", "BL-080", "0.65"):
        assert needle in cva, needle
    frtb_page = (PAGES / "05_frtb-sbm-and-cascade.md").read_text(encoding="utf-8")
    for needle in (
        "MAR21.46",
        "MAR21.19",
        "sqrt(2)",
        "MAR21.4",
        "MAR21.101",
        "MAR33.4",
        "MAR33.12",
        "UNVERIFIED",
        "622,253.97",
    ):
        assert needle in frtb_page, needle
