"""RV-917 and RV-918: BA-CVA (MAR50.9, MAR50.14 to MAR50.25) function by function (contract `engine-regulatory-api`,
module ba_cva). There is no verified official numeric example (addendum R8): the tests rest on the invariants of
dossier 01 M.1, the
goldens recomputed from the formulas, and the BA-CVA block of the C12 documents."""

from __future__ import annotations

import math

import numpy as np
import pytest
from reg_helpers import load_c12_blocks, load_calculators, named

from riskvalidation.regulatory import ba_cva as b

DOC = load_calculators()


def _df(m: float) -> float:
    return (1 - math.exp(-0.05 * m)) / (0.05 * m)


def test_materiality_threshold():
    assert b.materiality_threshold_applies(1e11) is True
    assert b.materiality_threshold_applies(1.0000001e11) is False
    assert b.materiality_threshold_applies(0.0) is True
    assert b.materiality_threshold_applies(5e10) is True
    assert b.materiality_threshold_applies(2e12) is False
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="aggregate_notional_eur"):
            b.materiality_threshold_applies(bad)


def test_sector_risk_weight():
    cases = {
        "financials": (0.05, 0.12),
        "sovereigns-central-banks-mdbs": (0.005, 0.02),
        "technology-telecoms": (0.02, 0.055),
        "local-government-public": (0.01, 0.04),
        "basic-materials-energy-industrials": (0.03, 0.07),
        "consumer-transport-administrative": (0.03, 0.085),
        "health-care-utilities-professional": (0.015, 0.05),
        "other": (0.05, 0.12),
    }
    for sector, (ig, hy) in cases.items():
        p = b.sector_risk_weight(sector, True)
        assert (
            p.value == ig and p.paragraph == "MAR50.16" and p.table == "MAR50.16 Table 1" and p.in_force == "2023-01-01"
        )
        assert b.sector_risk_weight(sector, False).value == hy
    with pytest.raises(ValueError, match=r"sector.*MAR50\.16|MAR50\.16.*sector"):
        b.sector_risk_weight("crypto", True)


def test_discount_factor_and_scva():
    assert b.discount_factor(1.0) == pytest.approx(0.975412, abs=5e-7)
    assert b.discount_factor(5.0) == pytest.approx(0.884797, abs=5e-7)
    for m in (0.25, 1.0, 3.2, 10.0, 30.0):
        assert b.discount_factor(m) == pytest.approx(_df(m), rel=1e-13)
        assert b.discount_factor(m, imm_bank=True) == 1.0
    assert b.discount_factor(1e-9) == pytest.approx(1.0, abs=1e-9)
    values = [b.discount_factor(m / 4) for m in range(1, 80)]
    assert values == sorted(values, reverse=True)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="m"):
            b.discount_factor(bad)
    assert b.scva(0.05, [(5.0, 100.0)]) == pytest.approx(0.05 * 5 * 100 * 0.884797 / 1.4, abs=5e-6)
    assert b.scva(0.05, [(5.0, 100.0)]) == pytest.approx(15.799944, abs=5e-7)
    two = [(5.0, 100.0), (2.0, 40.0)]
    assert b.scva(0.03, two) == pytest.approx(0.03 / 1.4 * (5 * 100 * _df(5) + 2 * 40 * _df(2)), rel=1e-13)
    assert b.scva(0.03, two, imm_bank=True) == pytest.approx(0.03 / 1.4 * (5 * 100 + 2 * 40), rel=1e-13)
    assert b.scva(0.03, two, alpha=1.0) == pytest.approx(1.4 * b.scva(0.03, two), rel=1e-13)
    # linear in the EAD and in the risk weight; an EAD of 0 gives 0; no netting set gives 0
    assert b.scva(0.05, [(5.0, 300.0)]) == pytest.approx(3 * b.scva(0.05, [(5.0, 100.0)]), rel=1e-13)
    assert b.scva(0.10, [(5.0, 100.0)]) == pytest.approx(2 * b.scva(0.05, [(5.0, 100.0)]), rel=1e-13)
    assert b.scva(0.05, [(5.0, 0.0)]) == 0.0 and b.scva(0.05, []) == 0.0
    with pytest.raises(ValueError, match="risk_weight"):
        b.scva(-0.01, [(1.0, 1.0)])
    with pytest.raises(ValueError, match="netting_sets"):
        b.scva(0.05, [(1.0, -1.0)])
    with pytest.raises(ValueError, match="netting_sets"):
        b.scva(0.05, [(0.0, 1.0)])
    with pytest.raises(ValueError, match="alpha"):
        b.scva(0.05, [(1.0, 1.0)], alpha=0.0)


def test_k_reduced():
    ex = {e["id"]: e for e in DOC["invariant_cases"]}["ba-cva-rho-one"]
    inp = named(ex["inputs"])
    assert b.k_reduced([inp["scva_1"], inp["scva_2"]], rho=inp["rho"]) == pytest.approx(
        named(ex["outputs"])["k_reduced"], rel=1e-12
    )
    assert b.k_reduced([10.0, 20.0], rho=1.0) == pytest.approx(30.0, rel=1e-12)
    assert b.k_reduced([10.0, 20.0], rho=0.0) == pytest.approx(math.sqrt(500.0), rel=1e-13)
    assert b.k_reduced([10.0, 20.0]) == pytest.approx(math.sqrt((0.5 * 30) ** 2 + 0.75 * 500), rel=1e-13)
    assert b.k_reduced([17.5]) == pytest.approx(17.5, rel=1e-13)
    assert b.k_reduced([17.5], rho=0.2) == pytest.approx(17.5, rel=1e-13)
    assert b.k_reduced([]) == 0.0
    rng = np.random.default_rng(7)
    for _ in range(100):
        x = [float(v) for v in rng.uniform(0, 50, size=int(rng.integers(1, 9)))]
        rho = float(rng.uniform(0, 1))
        k = b.k_reduced(x, rho)
        assert math.sqrt(sum(v * v for v in x)) - 1e-9 <= k <= sum(x) + 1e-9
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="rho"):
            b.k_reduced([1.0], rho=bad)
    with pytest.raises(ValueError, match="scvas"):
        b.k_reduced([1.0, -2.0])


def test_k_hedged_and_k_full():
    x = [10.0, 20.0, 5.0]
    # no hedges: SNH = 0, IH = 0, HMA = 0 equals K_reduced
    assert b.k_hedged(x, 0.0, [0.0, 0.0, 0.0]) == pytest.approx(b.k_reduced(x), rel=1e-13)
    assert b.k_hedged(x, 0.0, [0.0, 0.0, 0.0], rho=0.8) == pytest.approx(b.k_reduced(x, 0.8), rel=1e-13)
    # the formula with hedges, recomputed
    scva_minus_snh, ih, hma = [8.0, 15.0, 5.0], 4.0, [0.5, 0.25, 0.0]
    want = math.sqrt((0.5 * sum(scva_minus_snh) - ih) ** 2 + 0.75 * sum(v * v for v in scva_minus_snh) + sum(hma))
    assert b.k_hedged(scva_minus_snh, ih, hma) == pytest.approx(want, rel=1e-13)
    # the radicand is never negative: a hedge larger than the exposure still gives a real result
    assert b.k_hedged([1.0], 1e6, [0.0]) == pytest.approx(1e6 - 0.5, rel=1e-9)
    assert b.k_hedged([0.0], 0.0, [0.0]) == 0.0
    # K_full
    assert b.k_full(30.0, 0.0) == pytest.approx(7.5, rel=1e-15)
    ex = {e["id"]: e for e in DOC["invariant_cases"]}["ba-cva-hedges-floor"]
    inp = named(ex["inputs"])
    assert b.k_full(inp["k_reduced"], inp["k_hedged"], beta=inp["beta"]) == pytest.approx(
        named(ex["outputs"])["k_full"], rel=1e-12
    )
    assert b.k_full(30.0, 12.0) == pytest.approx(0.25 * 30 + 0.75 * 12, rel=1e-15)
    for kh in (0.0, 1.0, 50.0):
        assert b.k_full(30.0, kh) >= 0.25 * 30.0 - 1e-12  # hedges never take it below beta K_reduced
    with pytest.raises(ValueError, match="hma"):
        b.k_hedged([1.0, 2.0], 0.0, [0.0])
    with pytest.raises(ValueError, match="hma"):
        b.k_hedged([1.0], 0.0, [-0.1])
    with pytest.raises(ValueError, match="ih"):
        b.k_hedged([1.0], float("nan"), [0.0])
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="beta"):
            b.k_full(1.0, 1.0, beta=bad)
    with pytest.raises(ValueError, match="k_reduced_value"):
        b.k_full(-1.0, 1.0)


def test_capital_and_the_c12_blocks():
    assert b.capital(100.0) == pytest.approx(65.0, rel=1e-15)
    assert b.capital(100.0, ds=0.5) == pytest.approx(50.0, rel=1e-15)
    assert b.capital(0.0) == 0.0
    with pytest.raises(ValueError, match="k"):
        b.capital(-1.0)
    with pytest.raises(ValueError, match="ds"):
        b.capital(1.0, ds=-0.1)
    for variant, doc in load_c12_blocks().items():
        block = doc["ba_cva"]
        scvas = []
        for cp in block["counterparties"]:
            rw = b.sector_risk_weight(cp["sector"], cp["rating_class"] == "investment-grade")
            assert rw.value == cp["risk_weight"]
            assert b.discount_factor(cp["m_years"], imm_bank=block["imm_bank"]) == pytest.approx(
                cp["discount_factor"], rel=1e-8
            )
            scva = b.scva(
                rw.value, [(cp["m_years"], cp["ead"]["amount"])], imm_bank=block["imm_bank"], alpha=block["alpha"]
            )
            assert scva == pytest.approx(cp["scva"], rel=1e-9), variant
            scvas.append(scva)
        k = b.k_reduced(scvas, rho=block["rho"])
        assert k == pytest.approx(block["k_reduced"], rel=1e-9)
        assert b.capital(k, ds=block["ds"]) == pytest.approx(block["capital"]["amount"], rel=1e-8), variant
