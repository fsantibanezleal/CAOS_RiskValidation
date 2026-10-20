"""RV-908 and RV-909: the RBC90.1 phase-in factors and the RBC20.4 output floor with the RBC90.2 cap (contract
`engine-regulatory-api`, module output_floor). The RBC20.13 values 101.5 and 95 come from the fixture's two official
examples."""

from __future__ import annotations

import math

import pytest
from reg_helpers import example, load_calculators, named

from riskvalidation.regulatory import output_floor as of

DOC = load_calculators()


def test_phase_in_factor_table():
    rows = DOC["output_floor_phase_in"]
    assert [r["year"] for r in rows] == [2023, 2024, 2025, 2026, 2027, 2028]
    for row in rows:
        p = of.phase_in_factor(row["year"])
        assert p.value == row["factor"]
        assert (p.paragraph, p.table, p.in_force) == (row["paragraph"], row["table"], row["in_force_date"])
    assert [of.phase_in_factor(y).value for y in range(2023, 2029)] == [0.5, 0.55, 0.6, 0.65, 0.7, 0.725]
    # 2028 and later stay at the fully phased-in factor
    assert of.phase_in_factor(2035).value == 0.725 and of.phase_in_factor(2100).paragraph == "RBC90.1"
    factors = [of.phase_in_factor(y).value for y in range(2023, 2040)]
    assert factors == sorted(factors)


def test_phase_in_factor_refusals():
    for bad in (2022, 1999, 0, -5):
        with pytest.raises(ValueError, match=r"year.*RBC90\.1|RBC90\.1.*year"):
            of.phase_in_factor(bad)
    for bad in (2025.5, "2025", None, True):
        with pytest.raises(ValueError, match="year"):
            of.phase_in_factor(bad)  # type: ignore[arg-type]


def test_floored_rwa_rbc20_13_goldens():
    plain = example(DOC, "rbc20-13-output-floor")
    capped = example(DOC, "rbc20-13-output-floor-capped")
    a = named(plain["inputs"])
    assert a["rwa_nominated"] == 76 and a["rwa_sa_only"] == 140 and a["floor_factor"] == 0.725
    assert of.floored_rwa(a["rwa_nominated"], a["rwa_sa_only"], a["floor_factor"]) == pytest.approx(
        named(plain["outputs"])["rwa_floored"], rel=1e-12
    )
    b = named(capped["inputs"])
    out = of.floored_rwa(b["rwa_nominated"], b["rwa_sa_only"], b["floor_factor"], cap_share=b["cap_share"])
    assert out == pytest.approx(named(capped["outputs"])["rwa_floored"], rel=1e-12)
    # the contract's own numbers: 140 x 0.725 = 101.5 and 76 x 1.25 = 95
    assert of.floored_rwa(76, 140, 0.725) == pytest.approx(101.5, rel=1e-12)
    assert of.floored_rwa(76, 140, 0.725, cap_share=0.25) == pytest.approx(95.0, rel=1e-12)
    # the phase-in factor of the year feeds the floor
    assert of.floored_rwa(76, 140, of.phase_in_factor(2023).value) == 76  # 0.50 x 140 = 70 does not bind
    assert of.floored_rwa(76, 140, of.phase_in_factor(2025).value) == pytest.approx(84.0)  # 0.60 x 140 binds


def test_floored_rwa_invariants_and_refusals():
    # never below the nominated RWA: the floor binds only when f * RWA_SA exceeds it
    assert of.floored_rwa(120, 140, 0.725) == 120
    assert of.floored_rwa(0, 0, 0.725) == 0
    grid = [
        (rwa, sa, f)
        for rwa in (0.0, 50.0, 101.5, 300.0)
        for sa in (0.0, 100.0, 140.0, 500.0)
        for f in (0.5, 0.725, 1.0)
    ]
    for rwa, sa, f in grid:
        out = of.floored_rwa(rwa, sa, f)
        assert out >= rwa and out == max(rwa, f * sa)
        capped = of.floored_rwa(rwa, sa, f, cap_share=0.25)
        assert rwa <= capped <= out + 1e-12 and capped <= rwa * 1.25 + 1e-12
        assert capped == pytest.approx(min(out, rwa * 1.25))
    # monotone in the factor and in the standardised RWA
    values = [of.floored_rwa(76, 140, f / 100) for f in range(40, 100, 5)]
    assert values == sorted(values)
    # a cap of zero keeps the nominated RWA; an unbinding cap changes nothing
    assert of.floored_rwa(76, 140, 0.725, cap_share=0.0) == 76
    assert of.floored_rwa(76, 140, 0.725, cap_share=10.0) == pytest.approx(101.5, rel=1e-12)
    for name, args in (
        ("rwa_nominated", (-1.0, 140.0, 0.725)),
        ("rwa_sa_only", (76.0, -1.0, 0.725)),
        ("factor", (76.0, 140.0, -0.1)),
        ("rwa_nominated", (math.nan, 140.0, 0.725)),
        ("rwa_sa_only", (76.0, math.inf, 0.725)),
        ("factor", (76.0, 140.0, math.nan)),
    ):
        with pytest.raises(ValueError, match=name):
            of.floored_rwa(*args)
    with pytest.raises(ValueError, match="cap_share"):
        of.floored_rwa(76, 140, 0.725, cap_share=-0.1)
    with pytest.raises(ValueError, match="cap_share"):
        of.floored_rwa(76, 140, 0.725, cap_share=math.nan)
