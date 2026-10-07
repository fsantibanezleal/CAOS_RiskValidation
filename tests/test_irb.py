"""U2: the IRB risk-weight functions (RV-201 to RV-204), against BCBS CRE99 Table 1 and the regimes' rules."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from riskvalidation.regulatory import REGIMES, capital_requirement, correlation, el_versus_provisions

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "cre99_table1.json").read_text(encoding="utf-8"))


def _rw_pct(column: dict, pd: float, regime: str = "basel3") -> float:
    res = capital_requirement(
        column["asset_class"], pd, column["lgd"], maturity=column.get("maturity", 2.5),
        sales_eur_m=column.get("sales_eur_m"), regime=regime,
    )
    return 100.0 * res.risk_weight


def test_cre99_table1():
    """RV-201: every one of the 144 cells within one unit of the printed precision (0.01 percentage points)."""
    worst = 0.0
    for row in FIXTURE["rows"]:
        for column, printed in zip(FIXTURE["columns"], row["rw_pct"], strict=True):
            got = _rw_pct(column, row["pd"])
            assert abs(got - printed) <= 0.01 + 1e-9, (row["pd"], column["id"], got, printed)
            worst = max(worst, abs(got - printed))
    assert worst > 0.0  # three cells sit one unit off (dossier 10): the comparison is real, not tautological


def test_regimes():
    """RV-202: Basel II's 1.06 factor and 0.03% floors, CRR3 equal to Basel III final, the Chilean rules."""
    for row in FIXTURE["rows"]:
        for column in FIXTURE["columns"]:
            b3 = _rw_pct(column, row["pd"])
            assert _rw_pct(column, row["pd"], "crr3") == pytest.approx(b3, rel=1e-12)
            assert _rw_pct(column, row["pd"], "basel2") == pytest.approx(1.06 * b3, rel=1e-12)
    low = 0.0001
    assert capital_requirement("corporate", low, 0.45, regime="basel3").pd_applied == 0.0005
    assert capital_requirement("corporate", low, 0.45, regime="basel2").pd_applied == 0.0003
    assert capital_requirement("qrre", low, 0.85, revolver=True, regime="basel3").pd_applied == 0.0010
    assert capital_requirement("qrre", low, 0.85, regime="basel3").pd_applied == 0.0005  # a transactor
    assert capital_requirement("sovereign", low, 0.45, regime="basel3").pd_applied == low  # exempt (CRE32.4)
    # a PD of 0 (a grade that never defaulted) is what the floor is for: it takes the floor wherever one applies ...
    for asset, kwargs, regime, floor in (("corporate", {}, "basel3", 0.0005), ("corporate", {}, "crr3", 0.0005),
                                         ("corporate", {}, "basel2", 0.0003), ("bank", {}, "basel3", 0.0005),
                                         ("qrre", {"revolver": True}, "basel3", 0.0010), ("qrre", {}, "basel3", 0.0005),
                                         ("corporate", {}, "cl_ran21_6", 0.0005), ("sme", {}, "cl_ran21_6", 0.0005)):
        lgd = 0.85 if asset == "qrre" else 0.45
        zero = capital_requirement(asset, 0.0, lgd, regime=regime, **kwargs)
        at_floor = capital_requirement(asset, floor, lgd, regime=regime, **kwargs)
        assert zero.pd_input == 0.0 and zero.pd_applied == floor and zero.k == at_floor.k, (asset, regime)
        assert any("floored" in n for n in zero.notes)
    # ... and is refused where none applies (sovereigns are exempt under Basel III final, CRR3 and Basel II)
    for regime in ("basel3", "crr3", "basel2"):
        with pytest.raises(ValueError, match="only where the regime floors it"):
            capital_requirement("sovereign", 0.0, 0.45, regime=regime)
    for bad in (-1e-9, 1.0, float("nan")):
        with pytest.raises(ValueError, match="pd must lie in"):
            capital_requirement("corporate", bad, 0.45)
    # Chile, RAN 21-6: the corporate curve is Basel's; PYMES have their own curve; NBFI 1.25 times; 61.3% consumer LGD
    for pd in (0.0005, 0.01, 0.2):
        assert capital_requirement("corporate", pd, 0.45, regime="cl_ran21_6").k == pytest.approx(
            capital_requirement("corporate", pd, 0.45).k, rel=1e-12)
        w = (1 - math.exp(-50 * pd)) / (1 - math.exp(-50))
        assert correlation("sme", pd, regime="cl_ran21_6") == pytest.approx(0.08 * w + 0.20 * (1 - w))
        assert correlation("financial_institution", pd, regime="cl_ran21_6") == pytest.approx(
            1.25 * correlation("corporate", pd, regime="cl_ran21_6"))
    consumer = capital_requirement("retail_consumer", 0.02, regime="cl_ran21_6")
    assert consumer.lgd == 0.613 and consumer.maturity is None
    with pytest.raises(ValueError, match="fixes the LGD"):
        capital_requirement("retail_consumer", 0.02, 0.45, regime="cl_ran21_6")
    with pytest.raises(ValueError, match="no asset class 'qrre'"):
        capital_requirement("qrre", 0.02, 0.85, regime="cl_ran21_6")
    for reg in REGIMES.values():
        assert reg.in_force and reg.references["risk_weight"] and reg.references["scaling"]
        assert json.loads(json.dumps(reg.to_dict()))["references"] == dict(reg.references)


def test_adjustments_and_defaulted():
    """RV-203: sales clamped to 5 to 50, the 1.25 multiplier, maturity clamped to 1 to 5, defaulted exposures."""
    pd = 0.01
    assert correlation("corporate", pd, sales_eur_m=1.0) == correlation("corporate", pd, sales_eur_m=5.0)
    assert correlation("corporate", pd, sales_eur_m=60.0) == correlation("corporate", pd)
    assert correlation("corporate", pd, sales_eur_m=5.0) == pytest.approx(correlation("corporate", pd) - 0.04)
    assert correlation("bank", pd, large_fi=True) == pytest.approx(1.25 * correlation("bank", pd))
    short = capital_requirement("corporate", pd, 0.45, maturity=0.2)
    assert short.maturity == 1.0 and any("clamped" in n for n in short.notes)
    assert capital_requirement("corporate", pd, 0.45, maturity=7.0).k == pytest.approx(
        capital_requirement("corporate", pd, 0.45, maturity=5.0).k)
    assert capital_requirement("residential_mortgage", pd, 0.25, maturity=5.0).maturity is None  # no retail MA
    d = capital_requirement("corporate", 1.0, 0.45, defaulted=True, el_best_estimate=0.40, ead=1000.0)
    assert d.k == pytest.approx(0.05) and d.risk_weight == pytest.approx(0.625) and d.rwa == pytest.approx(625.0)
    assert capital_requirement("corporate", 1.0, 0.45, defaulted=True, el_best_estimate=0.50).k == 0.0
    r = capital_requirement("corporate", 0.02, 0.45, ead=1_000_000.0)
    assert r.rwa == pytest.approx(r.risk_weight * 1_000_000.0) and r.references["correlation"] == "CRE31.5"
    # a result carries its own copy of the references; the shared regime tables cannot be changed
    r.references["correlation"] = "edited"
    assert "correlation" not in REGIMES["basel3"].references
    assert capital_requirement("qrre", 0.02, 0.85).references["correlation"] == "CRE31.15"
    with pytest.raises(TypeError):
        REGIMES["basel3"].references["scaling"] = "1.06"  # type: ignore[index]


def test_el_versus_provisions():
    """RV-204: a shortfall is deducted in full; an excess counts in Tier 2 up to 0.6% of IRB credit RWA."""
    short = el_versus_provisions(120.0, 100.0, 10_000.0)
    assert short["cet1_deduction"] == 20.0 and short["tier2_addback"] == 0.0
    small = el_versus_provisions(100.0, 130.0, 10_000.0)
    assert small["cet1_deduction"] == 0.0 and small["tier2_addback"] == 30.0
    capped = el_versus_provisions(100.0, 200.0, 10_000.0)
    assert capped["tier2_addback"] == pytest.approx(60.0)
