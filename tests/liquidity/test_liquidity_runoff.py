"""RV-1102 to RV-1105: the constant daily compounding and its inverse, the transcribed parameter sets and the
contractual roll-off (contract `engine-liquidity-api`, module runoff). Every golden is also recomputed here through a
second route: a loop of daily survivals, or the value the ECB note prints in its own tables."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.liquidity import runoff as r

TOL6 = 5e-7  # a golden stated to six decimals holds to 5e-7


def loop_cumulative(q: float, n: int) -> float:
    """The cumulative outflow by multiplying the daily survivals one day at a time (no power function)."""
    survival = 1.0
    for _ in range(n):
        survival *= 1.0 - q
    return 1.0 - survival


def test_cumulative_outflow_constant_goldens_and_invariants():
    f = r.cumulative_outflow_constant
    assert f(0.00103258, 125) == pytest.approx(0.121148, abs=TOL6)
    assert f(0.00103258, 125) == pytest.approx(0.12114837, abs=5e-8)  # the value behind the dossier's 12.11 percent
    assert f(0.00161494, 125) == pytest.approx(0.182930, abs=TOL6)
    assert f(0.00688541, 2) == pytest.approx(0.013723, abs=TOL6)
    assert f(0.00373218, 125) == pytest.approx(0.373368, abs=TOL6)
    # the second route
    for q, n in ((0.00103258, 125), (0.01, 11), (0.3, 3), (0.0, 7), (1.0, 4)):
        assert f(q, n) == pytest.approx(loop_cumulative(q, n), abs=1e-12)
    # invariants: zero at n = 0 and at q = 0, one at q = 1 for n >= 1
    assert f(0.2, 0) == 0.0 and f(1.0, 0) == 0.0 and f(0.0, 50) == 0.0
    assert f(1.0, 1) == 1.0 and f(1.0, 9) == 1.0
    # monotone in n and in q
    n = np.arange(0, 60)
    assert np.all(np.diff(f(0.02, n)) >= 0.0)
    q = np.linspace(0.0, 1.0, 41)
    assert np.all(np.diff(f(q, 10)) >= 0.0)
    # broadcasting follows NumPy; a scalar call returns a float
    assert f(np.array([[0.1], [0.2]]), np.array([1, 2, 3])).shape == (2, 3)
    assert isinstance(f(0.1, 3), float)
    # refusals name the field
    with pytest.raises(ValueError, match="^q: "):
        f(1.5, 3)
    with pytest.raises(ValueError, match="^n: "):
        f(0.1, -1)
    with pytest.raises(ValueError, match="^n: "):
        f(0.1, 2.5)


# the last row of the printed ECB Tables 2.2 (Adverse) and 2.3 (Extreme), percent, and the first two buckets
PRINTED = {
    "adverse": {
        "stable-retail": (0.10, 0.21, 12.11), "other-retail": (0.37, 0.75, 37.34), "operational": (0.51, 1.03, 47.52),
        "non-operational-non-financial-corporates": (0.69, 1.37, 57.84), "non-operational-others": (0.69, 1.37, 57.84),
    },
    "extreme": {
        "stable-retail": (0.16, 0.32, 18.29), "other-retail": (0.44, 0.88, 42.44), "operational": (0.75, 1.50, 61.00),
        "non-operational-non-financial-corporates": (1.07, 2.12, 73.78), "non-operational-others": (1.07, 2.12, 73.78),
    },
}


def test_list_compounding_against_printed_table_2_2_and_2_3():
    """The ECB note prints the cumulated outflow by bucket: the engine's compounding reproduces the printed first
    bucket (one relevant day), second bucket (two) and last bucket (125) to the printed two decimals of a percent."""
    for severity, rows in PRINTED.items():
        for cls, (d1, d2, d125) in rows.items():
            q = r.LIST_PARAMETERS[severity]["open_maturity"][cls]
            assert 100 * r.cumulative_outflow_constant(q, 1) == pytest.approx(d1, abs=5.1e-3), (severity, cls)
            assert 100 * r.cumulative_outflow_constant(q, 2) == pytest.approx(d2, abs=5.1e-3), (severity, cls)
            assert 100 * r.cumulative_outflow_constant(q, 125) == pytest.approx(d125, abs=5.1e-3), (severity, cls)


def test_cumulative_outflow_path_golden_and_invariants():
    p = r.cumulative_outflow_path(np.array([0.01, 0.02, 0.03]))
    assert p == pytest.approx([0.01, 0.0298, 0.058906], abs=1e-12)
    assert p[2] == pytest.approx(1 - 0.99 * 0.98 * 0.97, abs=1e-15)
    # a constant path equals the constant formula on 1..H
    q = 0.00373218
    path = r.cumulative_outflow_path(np.full(125, q))
    assert path == pytest.approx(r.cumulative_outflow_constant(q, np.arange(1, 126)), abs=1e-12)
    assert np.all(np.diff(path) >= 0.0)
    # a rate of 1 on any day makes every later value 1, whatever follows
    after = r.cumulative_outflow_path(np.array([0.1, 1.0, 0.0, 0.5]))
    assert after[1:].tolist() == [1.0, 1.0, 1.0] and after[0] == pytest.approx(0.1)
    # a rate of 0 is legitimate
    assert r.cumulative_outflow_path(np.zeros(4)).tolist() == [0.0, 0.0, 0.0, 0.0]
    x = np.array([0.1, 0.2])
    r.cumulative_outflow_path(x)
    assert x.tolist() == [0.1, 0.2]
    with pytest.raises(ValueError, match="^daily_rates: "):
        r.cumulative_outflow_path(np.array([0.1, 1.1]))
    with pytest.raises(ValueError, match="^daily_rates: "):
        r.cumulative_outflow_path(np.array([]))


A1_GOLDEN = {  # share, days, the recomputed daily rate of the contract
    "svb-2023": (0.85, 2, 0.612702), "first-republic-2023": (0.57, 90, 0.009334), "svb-uk-2023": (0.30, 1, 0.300000),
    "credit-suisse-2023": (0.21, 90, 0.002616), "signature-2023": (0.20, 1, 0.200000),
    "icesave-2008": (0.20, 75, 0.002971), "northern-rock-2007": (0.20, 4, 0.054258),
    "wamu-2008": (0.10, 10, 0.010481), "indymac-2008": (0.08, 14, 0.005938),
}


def test_daily_rate_from_cumulative_goldens_and_round_trip():
    for rid, (share, days, rate) in A1_GOLDEN.items():
        assert r.daily_rate_from_cumulative(share, days) == pytest.approx(rate, abs=TOL6), rid
        # the round trip returns the share to 1e-12, through the independent loop as well
        back = r.cumulative_outflow_constant(r.daily_rate_from_cumulative(share, days), days)
        assert back == pytest.approx(share, abs=1e-12), rid
        assert loop_cumulative(r.daily_rate_from_cumulative(share, days), days) == pytest.approx(share, abs=1e-12)
    assert r.daily_rate_from_cumulative(1.0, 5) == 1.0
    assert r.daily_rate_from_cumulative(0.0, 5) == 0.0
    arr = r.daily_rate_from_cumulative(np.array([0.85, 0.2]), np.array([2, 4]))
    assert arr == pytest.approx([0.612702, 0.054258], abs=TOL6)
    with pytest.raises(ValueError, match="^days: "):
        r.daily_rate_from_cumulative(0.5, 0)
    with pytest.raises(ValueError, match="^share: "):
        r.daily_rate_from_cumulative(1.5, 3)


def test_table_a1_rows_and_implied_rates():
    assert [row.row_id for row in r.TABLE_A1] == list(A1_GOLDEN)
    for row in r.TABLE_A1:
        share, days, rate = A1_GOLDEN[row.row_id]
        assert (row.outflow_share, row.days) == (share, days)
        assert row.implied_daily_rate == pytest.approx(rate, abs=TOL6)
        assert "d555" in row.source and "Table A.1" in row.source
    assert {row.row_id: row.year for row in r.TABLE_A1}["svb-2023"] == 2023
    assert [row.year for row in r.TABLE_A1] == [2023, 2023, 2023, 2023, 2023, 2008, 2007, 2008, 2008]
    assert r.TABLE_A1_LCR_RANGE == (0.03, 0.40, 30)
    assert r.SEGMENTS == ("retail-insured", "retail-uninsured", "operational", "non-operational-corporate", "financial")


# the transcription of ECB LiST 2019 Table 3.3 (open maturity, daily), Table 3.2 and the drawdown, typed from the
# contract and the addendum L1 (an independent source of the module's own constants)
OPEN = {
    "adverse": {"stable-retail": 0.00103258, "other-retail": 0.00373218, "operational": 0.00514464,
                "non-operational-non-financial-corporates": 0.00688541, "non-operational-others": 0.00688541},
    "extreme": {"stable-retail": 0.00161494, "other-retail": 0.00440873, "operational": 0.00750509,
                "non-operational-non-financial-corporates": 0.01065229, "non-operational-others": 0.01065229},
}
FULL_OPEN = ("open-repos", "non-operational-credit-institutions", "non-operational-other-financial",
             "other-transactions")
SIGHT = ("stable-retail", "other-retail", "operational", "non-operational-non-financial-corporates",
         "non-operational-others")
ROLLOFF = {"adverse": {"stable-retail": 0.18, "other-retail": 0.39, "operational": 0.37,
                       "non-operational-non-financial-corporates": 0.52, "non-operational-others": 0.52},
           "extreme": {"stable-retail": 0.27, "other-retail": 0.48, "operational": 0.50,
                       "non-operational-non-financial-corporates": 0.76, "non-operational-others": 0.76}}
CUMULATED_125 = {"adverse": {"stable-retail": 0.121148, "other-retail": 0.373368, "operational": 0.475201,
                             "non-operational-non-financial-corporates": 0.578381},
                 "extreme": {"stable-retail": 0.182930, "other-retail": 0.424381, "operational": 0.610026,
                             "non-operational-non-financial-corporates": 0.737808}}


def test_list_parameters_transcription():
    p = r.LIST_PARAMETERS
    assert set(p) == {"baseline", "adverse", "extreme"}
    assert r.LIST_RELEVANT_DAYS == 125 and r.LIST_FACILITY_DAYS == 63
    for severity in ("baseline", "adverse", "extreme"):
        assert set(p[severity]) == {"open_maturity", "contractual_outflow", "contractual_inflow", "drawdown"}
        om = p[severity]["open_maturity"]
        assert len(om) == 9
        for cls in FULL_OPEN:  # the financial classes and open repos leave in full on the first relevant day
            assert om[cls] == 1.0, (severity, cls)
        for cls in SIGHT:
            expected = 0.0 if severity == "baseline" else OPEN[severity][cls]
            assert om[cls] == expected, (severity, cls)
        co = p[severity]["contractual_outflow"]
        for cls in ("securities-issued", "secured-funding", "non-operational-credit-institutions",
                    "non-operational-other-financial", "fx-swaps", "derivatives", "other-cash-outflows"):
            assert co[cls] == 1.0, (severity, cls)
        for cls in SIGHT:
            expected = 0.0 if severity == "baseline" else ROLLOFF[severity][cls]
            assert co[cls] == expected, (severity, cls)
        ci = p[severity]["contractual_inflow"]
        for cls in ("secured-lending", "loans-credit-institutions", "loans-other-financial", "fx-swaps", "derivatives",
                    "own-paper", "other-inflows"):
            assert ci[cls] == 1.0, (severity, cls)
        for cls in ("loans-retail", "loans-non-financial-corporates", "loans-others"):
            assert ci[cls] == 0.0, (severity, cls)
    assert p["baseline"]["drawdown"] == {"committed-credit-facilities": 0.0, "liquidity-facilities": 0.0}
    assert p["adverse"]["drawdown"] == {"committed-credit-facilities": 0.00202704, "liquidity-facilities": 0.01443904}
    assert p["extreme"]["drawdown"] == {"committed-credit-facilities": 0.00257634, "liquidity-facilities": 0.02176434}
    # the cumulated figures after 125 relevant days (recomputed in the addendum) and after 63 for the facilities
    for severity, rows in CUMULATED_125.items():
        for cls, value in rows.items():
            assert r.cumulative_outflow_constant(p[severity]["open_maturity"][cls], 125) == pytest.approx(
                value, abs=TOL6), (severity, cls)
    for severity, (credit, liquidity) in {"adverse": (0.12, 0.60), "extreme": (0.15, 0.75)}.items():
        dd = p[severity]["drawdown"]
        assert r.cumulative_outflow_constant(dd["committed-credit-facilities"], 63) == pytest.approx(credit, abs=5e-5)
        assert r.cumulative_outflow_constant(dd["liquidity-facilities"], 63) == pytest.approx(liquidity, abs=5e-5)
    # every value is a share
    for severity in p:
        for block in p[severity].values():
            assert all(0.0 <= v <= 1.0 for v in block.values())
    assert "Table 3.3" in r.LIST_SOURCE and "Table 3.2" in r.LIST_SOURCE


def test_contractual_rolloff_golden():
    m = np.array([100.0, 50.0])
    assert r.contractual_rolloff(m, 0.18).tolist() == pytest.approx([18.0, 9.0], abs=1e-12)
    assert r.contractual_rolloff(m, 0.27).tolist() == pytest.approx([27.0, 13.5], abs=1e-12)
    # the rolled-over remainder is not returned and the rate 0 and 1 are legitimate
    assert r.contractual_rolloff(m, 0.0).tolist() == [0.0, 0.0]
    assert r.contractual_rolloff(m, 1.0).tolist() == [100.0, 50.0]
    assert m.tolist() == [100.0, 50.0]  # not mutated
    with pytest.raises(ValueError, match="^outflow_rate: "):
        r.contractual_rolloff(m, 1.2)
    with pytest.raises(ValueError, match="^maturing: "):
        r.contractual_rolloff(np.array([-1.0]), 0.5)


def test_financial_no_rollover_golden():
    m = np.array([100.0, 0.0, 25.0])
    out = r.financial_no_rollover(m)
    assert out.tolist() == [100.0, 0.0, 25.0]
    out[0] = 0.0
    assert m[0] == 100.0  # a new array
    with pytest.raises(ValueError, match="^maturing: "):
        r.financial_no_rollover(np.array([1.0, -2.0]))
