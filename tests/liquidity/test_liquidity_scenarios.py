"""RV-1112 to RV-1114: the scenario families and phases, the fast-run calibration, the anchor and the two sensitivity
grids (contract `engine-liquidity-api`, module scenarios; dossier 05 E.2 and E.5)."""

from __future__ import annotations

import dataclasses
import statistics

import numpy as np
import pytest

from riskvalidation.liquidity import runoff as r
from riskvalidation.liquidity import scenarios as sc


def test_list_scenario_golden():
    adv = sc.list_scenario("adverse")
    assert isinstance(adv, sc.ListScenario)
    assert adv.stable_retail == 0.00103258 and adv.non_operational_non_financial_corporates == 0.00688541
    ext = sc.list_scenario("extreme")
    assert ext.stable_retail == 0.00161494 and ext.non_operational_non_financial_corporates == 0.01065229
    assert r.cumulative_outflow_constant(adv.stable_retail, 125) == pytest.approx(0.121148, abs=5e-7)
    # the values are LIST_PARAMETERS exactly, for every severity (no stable-retail proxy)
    for severity in ("baseline", "adverse", "extreme"):
        s = sc.list_scenario(severity)
        p = r.LIST_PARAMETERS[severity]
        assert s.severity == severity and s.relevant_days == 125
        assert dict(s.open_maturity) == p["open_maturity"]
        assert dict(s.contractual_outflow) == p["contractual_outflow"]
        assert dict(s.contractual_inflow) == p["contractual_inflow"]
        assert dict(s.drawdown) == p["drawdown"]
    # the fastest sight class runs at its own rate, far from the stable retail one
    assert adv.open_maturity["non-operational-non-financial-corporates"] > 6 * adv.open_maturity["stable-retail"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        adv.relevant_days = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="^severity: "):
        sc.list_scenario("severe")
    with pytest.raises(AttributeError):
        adv.no_such_class  # noqa: B018


def test_compose_scenario_golden_and_refusals():
    adv = sc.list_scenario("adverse")
    s = sc.compose_scenario("idiosyncratic", "acute", adv, {"retail": ("other-retail", 100.0)}, 2)
    assert isinstance(s, sc.Scenario)
    assert (s.family, s.phase, s.horizon) == ("idiosyncratic", "acute", 2)
    seg = s.segments[0]
    assert seg.segment_id == "retail" and seg.liquidity_class == "other-retail" and seg.opening == 100.0
    assert seg.daily_rates.tolist() == [0.00373218, 0.00373218]
    out = s.outflows()
    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(0.373218, abs=1e-9)
    assert out[0, 1] == pytest.approx(99.626782 * 0.00373218, abs=1e-9)
    assert out[0, 1] == pytest.approx(0.371825, abs=5e-7)
    # the opening balance is not altered, the arrays are not shared with the caller
    segments = {"a": ("stable-retail", 10.0), "b": ("operational", 20.0)}
    s2 = sc.compose_scenario("combined", "protracted", sc.list_scenario("extreme"), segments, 5)
    assert segments == {"a": ("stable-retail", 10.0), "b": ("operational", 20.0)}
    assert s2.opening.tolist() == [10.0, 20.0] and s2.rates().shape == (2, 5)
    assert s2.rates()[1].tolist() == [0.00750509] * 5
    # a class the set does not hold raises, it is never mapped silently
    with pytest.raises(ValueError, match="xyz"):
        sc.compose_scenario("idiosyncratic", "acute", adv, {"x": ("xyz", 1.0)}, 2)
    with pytest.raises(ValueError, match="^family: "):
        sc.compose_scenario("systemic", "acute", adv, {"x": ("other-retail", 1.0)}, 2)
    with pytest.raises(ValueError, match="^phase: "):
        sc.compose_scenario("idiosyncratic", "chronic", adv, {"x": ("other-retail", 1.0)}, 2)
    with pytest.raises(ValueError, match="^horizon: "):
        sc.compose_scenario("idiosyncratic", "acute", adv, {"x": ("other-retail", 1.0)}, 0)
    with pytest.raises(ValueError, match="segments"):
        sc.compose_scenario("idiosyncratic", "acute", adv, {}, 2)
    with pytest.raises(ValueError, match="^opening"):
        sc.compose_scenario("idiosyncratic", "acute", adv, {"x": ("other-retail", -1.0)}, 2)
    # a fast-run parameter set holds one class only; any other is refused
    fast = sc.fast_run_scenario([(0.2, 4)])
    ok = sc.compose_scenario("idiosyncratic", "acute", fast, {"u": (sc.FAST_RUN_CLASS, 100.0)}, 3)
    assert ok.segments[0].daily_rates.tolist() == [pytest.approx(r.daily_rate_from_cumulative(0.2, 4))] * 3
    with pytest.raises(ValueError, match="other-retail"):
        sc.compose_scenario("idiosyncratic", "acute", fast, {"u": ("other-retail", 100.0)}, 3)


def test_compose_scenario_financial_sight_empties_on_day_one():
    for severity in ("baseline", "adverse", "extreme"):
        s = sc.compose_scenario("idiosyncratic", "acute", sc.list_scenario(severity),
                                {"banks": ("non-operational-credit-institutions", 80.0),
                                 "fin": ("non-operational-other-financial", 20.0)}, 3)
        assert s.rates().tolist() == [[1.0] * 3, [1.0] * 3]
        out = s.outflows()
        assert out[:, 0].tolist() == [80.0, 20.0] and out[:, 1:].sum() == 0.0  # empty on day 1, nothing left


def test_fast_run_scenario_golden_rules_and_scored_refusal():
    rows = [(0.20, 75), (0.20, 4), (0.10, 10), (0.08, 14)]  # Icesave, Northern Rock, WaMu, IndyMac
    f = sc.fast_run_scenario(rows)
    assert isinstance(f, sc.FastRun)
    assert list(f.implied_rates) == pytest.approx([0.002971, 0.054258, 0.010481, 0.005938], abs=5e-7)
    assert f.rule == "maximum-implied-daily-rate"
    assert f.daily_rate == pytest.approx(0.054258, abs=5e-7)
    implied = [r.daily_rate_from_cumulative(s, d) for s, d in rows]
    med = sc.fast_run_scenario(rows, "median-implied-daily-rate")
    mean = sc.fast_run_scenario(rows, "mean-implied-daily-rate")
    assert med.daily_rate == pytest.approx(0.0082095, abs=5e-7)
    assert mean.daily_rate == pytest.approx(0.018412, abs=5e-7)
    # the second route: the statistics through the standard library
    assert f.daily_rate == max(implied) and med.daily_rate == pytest.approx(statistics.median(implied), abs=1e-15)
    assert mean.daily_rate == pytest.approx(statistics.fmean(implied), abs=1e-15)
    assert f.daily_rate == pytest.approx(0.054258, abs=5e-7)
    # calibration on the 2023 rows is refused with a message when the caller marks them scored
    scored = [(0.85, 2)] + rows
    with pytest.raises(ValueError, match="2023.*scored"):
        sc.fast_run_scenario(scored, scored=True)
    sc.fast_run_scenario(scored)  # not scored: allowed (an illustration, never a score)
    sc.fast_run_scenario(rows, scored=True)  # pre-2023 rows only: allowed
    with pytest.raises(ValueError, match="^rule: "):
        sc.fast_run_scenario(rows, "min-implied-daily-rate")
    with pytest.raises(ValueError, match="^rows: "):
        sc.fast_run_scenario([])
    with pytest.raises(ValueError, match="^rows: "):
        sc.fast_run_scenario([(1.5, 3)])
    with pytest.raises(ValueError, match="^rows: "):
        sc.fast_run_scenario([(0.5, 0)])


def test_anchor_from_observation_golden():
    a = sc.anchor_from_observation(0.85, 2)
    assert set(a) == {"outflow_share", "days", "implied_daily_rate", "forecast_claim"}
    assert a["implied_daily_rate"] == pytest.approx(0.612702, abs=5e-7)
    assert a["forecast_claim"] is False and a["outflow_share"] == 0.85 and a["days"] == 2
    assert sc.anchor_from_observation(0.20, 4)["implied_daily_rate"] == pytest.approx(0.054258, abs=5e-7)
    # the rate round-trips through cumulative_outflow_constant, and the observation is never a forecast
    for share, days in ((0.85, 2), (0.57, 90), (0.2, 1)):
        a = sc.anchor_from_observation(share, days)
        assert r.cumulative_outflow_constant(a["implied_daily_rate"], days) == pytest.approx(share, abs=1e-12)
        assert a["forecast_claim"] is False
    with pytest.raises(ValueError, match="^outflow_share: "):
        sc.anchor_from_observation(1.5, 2)
    with pytest.raises(ValueError, match="^days: "):
        sc.anchor_from_observation(0.5, 0)


def test_uninsured_share_grid_golden():
    grid = sc.uninsured_share_grid(40.0, 60.0, [0.5, 0.9])
    assert grid == [(50.0, 50.0), (10.0, 90.0)]
    for share, (ins, un) in zip([0.3, 0.6, 0.95], sc.uninsured_share_grid(125.0, 375.0, [0.3, 0.6, 0.95]), strict=True):
        assert ins + un == pytest.approx(500.0, abs=1e-9)  # the total is preserved
        assert un / (ins + un) == pytest.approx(share, abs=1e-12)
    assert sc.uninsured_share_grid(1.0, 1.0, [0.0, 1.0]) == [(2.0, 0.0), (0.0, 2.0)]
    with pytest.raises(ValueError, match="^shares: "):
        sc.uninsured_share_grid(40.0, 60.0, [0.9, 0.5])
    with pytest.raises(ValueError, match="^shares: "):
        sc.uninsured_share_grid(40.0, 60.0, [0.5, 1.2])
    with pytest.raises(ValueError, match="^insured: "):
        sc.uninsured_share_grid(-1.0, 60.0, [0.5])
    with pytest.raises(ValueError, match="total"):
        sc.uninsured_share_grid(0.0, 0.0, [0.5])


def test_htm_loss_grid_golden():
    assert sc.htm_loss_grid(-15.0, [0.0, 1.0, 2.0]) == [0.0, -15.0, -30.0]
    assert sc.htm_loss_grid(-15.0, [1.0]) == [-15.0]  # 1 is the reported loss
    assert sc.htm_loss_grid(-15.0, [0.5, 1.5]) == pytest.approx([-7.5, -22.5])
    with pytest.raises(ValueError, match="^multiples: "):
        sc.htm_loss_grid(-15.0, [-1.0, 0.0])
    with pytest.raises(ValueError, match="^multiples: "):
        sc.htm_loss_grid(-15.0, [1.0, 1.0])
    with pytest.raises(ValueError, match="^reported_loss: "):
        sc.htm_loss_grid(float("nan"), [1.0])
    assert np.isfinite(sc.htm_loss_grid(-15.0, [0.0])[0])
