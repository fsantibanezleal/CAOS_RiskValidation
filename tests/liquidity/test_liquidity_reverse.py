"""RV-1115 and RV-1116: the reverse liquidity stress (contract `engine-liquidity-api`, module reverse; dossier 05 E.5
rung 5, line 279)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.liquidity import ladder, reverse
from riskvalidation.liquidity import scenarios as sc


def one_segment(q: float = 0.01, opening: float = 1000.0, horizon: int = 125) -> sc.Scenario:
    """One segment with a constant daily rate q (the fast-run parameter set carries one rate)."""
    fast = sc.fast_run_scenario([(q, 1)])  # a one-day observation of share q is the daily rate q
    assert fast.daily_rate == pytest.approx(q, abs=1e-15)
    return sc.compose_scenario("idiosyncratic", "protracted", fast, {"deposits": (sc.FAST_RUN_CLASS, opening)}, horizon)


def brute_force_sp(opening: float, q: float, cbc: float, horizon: int) -> int | None:
    """The survival period by the day-by-day stock: no function of the engine."""
    stock, cum = opening, 0.0
    for t in range(1, horizon + 1):
        out = stock * q
        stock -= out
        cum += out
        if cbc - cum < 0:
            return t
    return None


def test_survival_vs_multiplier_golden_and_monotone():
    s = one_segment()
    cbc = np.full(125, 100.0)
    res = reverse.survival_vs_multiplier(s, cbc, np.array([1.0, 2.0, 3.0]))
    assert all(isinstance(x, ladder.SurvivalResult) for x in res)
    assert res[0].day == 11 and not res[0].survives  # 0.99^t < 0.9 first at t = 11
    assert brute_force_sp(1000.0, 0.01, 100.0, 125) == 11
    for m, x in zip((1.0, 2.0, 3.0), res, strict=True):
        assert x.day == brute_force_sp(1000.0, min(m * 0.01, 1.0), 100.0, 125)
    days = [x.day for x in res]
    assert days == sorted(days, reverse=True) and days[0] > days[-1]  # non-increasing in the multiplier
    # "survives the horizon" counts as longer than any day: a huge CBC survives, and is never a number
    big = reverse.survival_vs_multiplier(s, np.full(125, 1e9), np.array([1.0, 2.0]))
    assert all(x.survives and x.day is None and x.horizon == 125 for x in big)
    # a stricter monotonicity check over a fine grid, treating None as infinity
    grid = np.linspace(1.0, 20.0, 40)
    sp = [math.inf if x.survives else x.day for x in reverse.survival_vs_multiplier(s, np.full(125, 300.0), grid)]
    assert all(b <= a for a, b in zip(sp, sp[1:], strict=False))
    with pytest.raises(ValueError, match="^multipliers: "):
        reverse.survival_vs_multiplier(s, cbc, np.array([2.0, 1.0]))
    with pytest.raises(ValueError, match="^multipliers: "):
        reverse.survival_vs_multiplier(s, cbc, np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="cbc"):
        reverse.survival_vs_multiplier(s, np.full(10, 100.0), np.array([1.0]))


def test_survival_vs_multiplier_cap():
    s = one_segment(q=0.01)
    cbc = np.full(125, 100.0)
    assert reverse.max_multiplier(s) == pytest.approx(100.0, abs=1e-9)  # 1 / 0.01
    reverse.survival_vs_multiplier(s, cbc, np.array([1.0, 100.0]))  # exactly at the cap: the rate is 1
    with pytest.raises(ValueError, match="cap"):
        reverse.survival_vs_multiplier(s, cbc, np.array([1.0, 100.5]))
    # a rate already at 1 stays, and a segment at 1 does not set the cap
    fin = sc.compose_scenario("idiosyncratic", "acute", sc.list_scenario("adverse"),
                              {"banks": ("non-operational-credit-institutions", 10.0),
                               "retail": ("stable-retail", 100.0)}, 5)
    assert reverse.max_multiplier(fin) == pytest.approx(1.0 / 0.00103258, rel=1e-12)
    # no rate below 1 and above 0: scaling changes nothing, the cap is unbounded
    assert reverse.max_multiplier(sc.compose_scenario(
        "idiosyncratic", "acute", sc.list_scenario("baseline"), {"r": ("stable-retail", 5.0)}, 3)) == math.inf


def test_solve_multiplier_golden_and_round_trip():
    s = one_segment()
    cbc = np.full(125, 100.0)
    res = reverse.solve_multiplier(s, cbc, 5)
    assert isinstance(res, reverse.SolveResult)
    assert res.status == "solved" and res.reason is None
    # 1 - 0.9^(1/5) = 0.0208516, divided by q, rounded up to six decimals
    assert res.multiplier == pytest.approx(2.085164, abs=5e-7)
    exact = (1 - 0.9 ** (1 / 5)) / 0.01
    assert res.multiplier == math.ceil(exact * 1e6) / 1e6
    assert res.survival_at_multiplier == 5
    # the round trip: survival at the solved multiplier is the target, and one day longer just below it
    assert reverse.survival_vs_multiplier(s, cbc, np.array([res.multiplier]))[0].day == 5
    below = reverse.survival_vs_multiplier(s, cbc, np.array([res.multiplier - 1e-6]))[0]
    assert below.day == 6 and brute_force_sp(1000.0, (res.multiplier - 1e-6) * 0.01, 100.0, 125) == 6
    # other targets round trip too, against the closed form 1 - 0.9^(1/T)
    for target in (2, 3, 8, 10):
        got = reverse.solve_multiplier(s, cbc, target)
        assert got.status == "solved"
        assert got.multiplier == pytest.approx((1 - 0.9 ** (1 / target)) / 0.01, abs=2e-6)
        assert got.survival_at_multiplier == target
    # a time-varying CBC: the multiplier is still the smallest one with SP <= target
    cbc_ramp = np.linspace(80.0, 200.0, 125)
    ramp = reverse.solve_multiplier(s, cbc_ramp, 6)
    assert ramp.status == "solved" and ramp.survival_at_multiplier <= 6
    lower = reverse.survival_vs_multiplier(s, cbc_ramp, np.array([max(1.0, ramp.multiplier - 2e-6)]))[0]
    assert lower.survives or lower.day > 6


def test_solve_multiplier_refusals():
    s = one_segment()
    cbc = np.full(125, 100.0)
    # a target not below the horizon is refused, whatever the bank (the horizon here is 125 days)
    for target in (125, 200):
        res = reverse.solve_multiplier(s, cbc, target)
        assert res.status == "refused" and res.multiplier is None and res.survival_at_multiplier is None
        assert "horizon" in res.reason
    # the base survival period is 11: a target at or above it is refused, the base already reaches it
    for target in (100, 12, 11):
        res = reverse.solve_multiplier(s, cbc, target)
        assert res.status == "refused" and res.multiplier is None and res.survival_at_multiplier is None
        assert "base" in res.reason
    assert reverse.solve_multiplier(s, cbc, 10).status == "solved"  # one day below the base survival period
    # a target no multiplier under the cap reaches: even a rate of 1 outflows 50, the CBC is 100
    safe = one_segment(q=0.01, opening=50.0)
    res = reverse.solve_multiplier(safe, cbc, 5)
    assert res.status == "refused" and res.multiplier is None and "reach" in res.reason
    # a bank that cannot be moved by a multiplier (no rate to scale) is refused with the reason
    still = sc.compose_scenario("idiosyncratic", "acute", sc.list_scenario("baseline"),
                                {"r": ("stable-retail", 5.0)}, 20)
    res = reverse.solve_multiplier(still, np.full(20, 1.0), 5)
    assert res.status == "refused" and res.multiplier is None and "reach" in res.reason
    with pytest.raises(ValueError, match="^target_days: "):
        reverse.solve_multiplier(s, cbc, 0)
    with pytest.raises(ValueError, match="cbc"):
        reverse.solve_multiplier(s, np.full(5, 100.0), 3)
