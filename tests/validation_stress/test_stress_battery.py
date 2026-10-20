"""RV-1318 to RV-1322: the stress validation tests (contract `engine-stress-api`, module validation.stress). Each
returns one
`TestResult`; the rows serialise into the fields of `common/test-result`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.core.result import Light
from riskvalidation.stress import capital
from riskvalidation.validation import stress as v

ROW_FIELDS = {
    "test_id",
    "model_id",
    "segment",
    "statistic",
    "p_value",
    "metric",
    "h0",
    "alternative",
    "policy_version",
    "alpha_amber",
    "alpha_red",
    "light",
    "n",
    "n_events",
    "inputs_hash",
    "reference",
    "notes",
    "extras",
}


def arr(*x):
    return np.array(x, dtype=float)


def qs(n, start_year=2020):
    ends = ("03-31", "06-30", "09-30", "12-31")
    return [f"{start_year + k // 4}-{ends[k % 4]}" for k in range(n)]


def backcast(pred_peak, real_peak, pred_at=3, real_at=2, **kw):
    n = 8
    pred = np.full(n, 0.1 * pred_peak)
    real = np.full(n, 0.1 * real_peak)
    pred[pred_at] = pred_peak
    real[real_at] = real_peak
    return v.satellite_backcast(pred, real, qs(n), "2019-12-31", **kw), pred, real


# ---------------------------------------------------------------------------------------------------- RV-1318


def test_satellite_backcast_lights_and_leakage():
    r, pred, real = backcast(1.0, 1.2)
    assert r.test_id == "stress.satellite_backcast" and r.light is Light.GREEN
    assert r.statistic == pytest.approx(math.sqrt(float(np.mean((pred - real) ** 2))), abs=1e-15)
    assert r.metric == 1.0  # predicted peak one quarter after the realised one
    assert r.extras["peak_ratio"] == pytest.approx(1.0 / 1.2, abs=1e-15)
    assert r.extras["thresholds"] == {"peak_amber": 0.35, "peak_red": 0.60, "timing_quarters": 2}
    assert r.alpha_amber is None and r.alpha_red is None and r.p_value is None
    # the size gates: deviation 0.5833 is within peak_red 0.60 (amber); 0.6667 and beyond is red; a custom gate moves
    # the light
    r, _, _ = backcast(0.5, 1.2)
    assert abs(1 - r.extras["peak_ratio"]) == pytest.approx(0.58333, abs=5e-6) and r.light is Light.AMBER
    assert backcast(0.5, 1.2, peak_red=0.5)[0].light is Light.RED
    assert backcast(0.4, 1.2)[0].light is Light.RED
    assert backcast(1.0, 1.2, peak_amber=0.1)[0].light is Light.AMBER
    assert backcast(1.8, 1.2)[0].light is Light.AMBER and backcast(2.5, 1.2)[0].light is Light.RED
    # the timing gate: three quarters apart is not green even with an exact size, but is amber while the size is within
    # gate
    t = backcast(1.2, 1.2, pred_at=5, real_at=2)[0]
    assert t.metric == 3.0 and t.light is Light.AMBER
    assert backcast(1.2, 1.2, pred_at=5, real_at=2, timing_quarters=3)[0].light is Light.GREEN
    assert backcast(1.2, 1.2, pred_at=0, real_at=2)[0].metric == -2.0
    # leakage: the fit window must end before the first quarter of the episode
    for fit_end in ("2020-03-31", "2021-12-31"):
        pred, real = np.full(8, 0.1), np.full(8, 0.2)
        pred[2] = real[2] = 0.5
        e = v.satellite_backcast(pred, real, qs(8), fit_end)
        assert e.light is Light.ERROR and "fit_end" in e.notes and e.statistic is None and e.metric is None
    row = v.satellite_backcast(pred, real, qs(8), "2019-12-31", model_id="M1", segment="consumer").to_dict()
    assert row["model_id"] == "M1" and row["segment"] == "consumer"
    # a realised peak that is not positive has no size ratio
    flat = v.satellite_backcast(np.full(8, 0.01), np.full(8, -0.01), qs(8), "2019-12-31")
    assert flat.light is Light.NOT_EVALUATED
    with pytest.raises(ValueError, match="realised"):
        v.satellite_backcast(arr(0.1, 0.2), arr(0.1), qs(2), "2019-12-31")
    with pytest.raises(ValueError, match="peak_amber"):
        backcast(1.0, 1.2, peak_amber=0.7)


# ---------------------------------------------------------------------------------------------------- RV-1319


def test_sensitivity_monotonicity():
    r = v.sensitivity_monotonicity((0.82, -0.045, 0.06), ("positive", "negative", "positive"))
    assert r.test_id == "stress.sensitivity_monotonicity" and r.light is Light.GREEN and r.statistic == 0.0
    assert r.metric is None and r.extras["offending_indices"] == []
    r = v.sensitivity_monotonicity((0.82, 0.045, 0.06), ("positive", "negative", "positive"))
    assert r.light is Light.RED and r.statistic == 1.0 and r.extras["offending_indices"] == [1]
    r = v.sensitivity_monotonicity((-1.0, 5.0, 0.0), ("none", "none", "positive"))
    assert r.statistic == 1.0 and r.extras["offending_indices"] == [2]  # zero is not positive; none is never wrong
    assert v.sensitivity_monotonicity((-1.0, 5.0), ("none", "none")).light is Light.GREEN
    with pytest.raises(ValueError, match="expected_signs"):
        v.sensitivity_monotonicity((1.0,), ("up",))
    with pytest.raises(ValueError, match="expected_signs"):
        v.sensitivity_monotonicity((1.0, 2.0), ("positive",))


def test_static_balance_sheet_rules():
    good = capital.eba_static_balance_sheet(arr(80, 15, 5), arr(2, 3))
    r = v.static_balance_sheet_rules(good)
    assert r.test_id == "stress.static_balance_sheet_rules" and r.light is Light.GREEN and r.statistic == 0.0
    assert r.extras["thresholds"] == {"tolerance": 1e-9} and r.extras["violated"] == []
    cured = good.copy()
    cured[:, 2] = (7.0, 6.0)  # stage 3 falls and the total is no longer constant
    r = v.static_balance_sheet_rules(cured)
    assert r.light is Light.RED and r.statistic == 2.0
    assert set(r.extras["violated"]) == {"constant-total", "no-cure"}
    only_cure = good.copy()
    only_cure[1] = (95.0, 1.0, 4.0)  # total constant, stage 3 falls from 7 to 4
    assert v.static_balance_sheet_rules(only_cure).extras["violated"] == ["no-cure"]
    neg = np.array([[101.0, -2.0, 1.0], [101.0, -2.0, 1.0]])
    r = v.static_balance_sheet_rules(neg)
    assert r.extras["violated"] == ["no-negative-balance"] and r.statistic == 1.0
    assert v.static_balance_sheet_rules(good + 1e-12).light is Light.GREEN
    assert v.static_balance_sheet_rules(good + 1e-6, tolerance=1e-3).light is Light.GREEN
    with pytest.raises(ValueError, match="balances"):
        v.static_balance_sheet_rules(arr(1, 2, 3))
    with pytest.raises(ValueError, match="tolerance"):
        v.static_balance_sheet_rules(good, tolerance=-1.0)


# ---------------------------------------------------------------------------------------------------- RV-1320


def test_bottom_up_vs_top_down():
    r = v.bottom_up_vs_top_down(120.0, 100.0)
    assert r.test_id == "stress.bottom_up_vs_top_down" and r.light is Light.GREEN
    assert r.statistic == pytest.approx(0.2, abs=1e-15) and r.metric == r.statistic
    assert r.extras["thresholds"] == {"amber": 0.25, "red": 0.50}
    # the contract's lights: green to 0.25, amber to 0.50, red above (the contract's own golden pair 160 and 200 is
    # checked in
    # the rule: 160 is 0.6 and therefore red; 140 is 0.4 and amber)
    assert v.bottom_up_vs_top_down(140.0, 100.0).light is Light.AMBER
    assert (
        v.bottom_up_vs_top_down(160.0, 100.0).light is Light.RED
        and v.bottom_up_vs_top_down(200.0, 100.0).light is Light.RED
    )
    assert (
        v.bottom_up_vs_top_down(75.0, 100.0).light is Light.GREEN
        and v.bottom_up_vs_top_down(50.0, 100.0).light is Light.AMBER
    )
    assert v.bottom_up_vs_top_down(25.0, 100.0).light is Light.RED
    assert v.bottom_up_vs_top_down(160.0, 100.0, amber=0.7, red=0.9).light is Light.GREEN
    for bad in (0.0, -5.0):
        e = v.bottom_up_vs_top_down(10.0, bad)
        assert e.light is Light.NOT_EVALUATED and "top_down" in e.notes
    with pytest.raises(ValueError, match="amber"):
        v.bottom_up_vs_top_down(1.0, 1.0, amber=0.6, red=0.5)
    with pytest.raises(ValueError):
        v.bottom_up_vs_top_down(float("nan"), 1.0)


def test_scenario_severity():
    r = v.scenario_severity(5.5, 5.5)
    assert r.test_id == "stress.scenario_severity" and r.light is Light.GREEN and r.statistic == 1.0
    assert r.extras["thresholds"] == {"minimum": 0.75}
    r = v.scenario_severity(3.0, 5.5)
    assert r.light is Light.AMBER and r.statistic == pytest.approx(0.5455, abs=5e-5)
    assert v.scenario_severity(0.0, 5.5).light is Light.AMBER and v.scenario_severity(-1.0, 5.5).light is Light.AMBER
    assert v.scenario_severity(4.125, 5.5).light is Light.GREEN  # exactly at the minimum
    assert v.scenario_severity(3.0, 5.5, minimum=0.5).light is Light.GREEN
    base = v.scenario_severity(5.0, 5.5, is_stress=False)
    assert base.light is Light.NOT_EVALUATED and "not a stress path" in base.notes
    for peak in (0.0, -2.0):
        assert v.scenario_severity(3.0, peak).light is Light.NOT_EVALUATED
    with pytest.raises(ValueError, match="minimum"):
        v.scenario_severity(3.0, 5.5, minimum=-0.1)


# ---------------------------------------------------------------------------------------------------- RV-1321


def test_reverse_solution_check():
    r = v.reverse_solution_check(5.0, 5.0, 2.2361, (2.2361, 2.2361, 2.2362))
    assert r.test_id == "stress.reverse_solution_check" and r.light is Light.GREEN
    assert r.statistic == 0.0 and r.metric == pytest.approx(0.0, abs=1e-15)
    assert r.extras["thresholds"] == {"tolerance": 1e-6}
    assert v.reverse_solution_check(5.0, 5.0, 2.2361, (1.5, 2.2361, 2.2362)).light is Light.RED
    r = v.reverse_solution_check(5.0, 5.0, 2.2361, (1.5, 2.2361, 2.2362))
    assert r.metric == pytest.approx(1.5 / 2.2361 - 1.0, abs=1e-12)
    far = v.reverse_solution_check(5.0, 5.0, 2.0, (2.5, 3.0, 4.0))
    assert far.light is Light.GREEN and far.metric == pytest.approx(0.25, abs=1e-12)
    gap = v.reverse_solution_check(5.01, 5.0, 2.0, (2.0, 2.0, 2.0))
    assert gap.light is Light.RED and gap.statistic == pytest.approx(0.002, abs=1e-12)
    assert v.reverse_solution_check(5.01, 5.0, 2.0, (2.0, 2.0, 2.0), tolerance=0.01).light is Light.GREEN
    e = v.reverse_solution_check(5.0, 5.0, 2.0, (2.0, 2.0))
    assert e.light is Light.NOT_EVALUATED and "starts" in e.notes
    with pytest.raises(ValueError, match="threshold"):
        v.reverse_solution_check(5.0, 0.0, 2.0, (2.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="distance"):
        v.reverse_solution_check(5.0, 5.0, 0.0, (2.0, 2.0, 2.0))


# ---------------------------------------------------------------------------------------------------- RV-1322


def every_result():
    good = capital.eba_static_balance_sheet(arr(80, 15, 5), arr(2, 3))
    return [
        backcast(1.0, 1.2)[0],
        v.sensitivity_monotonicity((0.82, -0.045), ("positive", "negative")),
        v.static_balance_sheet_rules(good),
        v.bottom_up_vs_top_down(120.0, 100.0),
        v.scenario_severity(5.5, 5.5),
        v.reverse_solution_check(5.0, 5.0, 2.2361, (2.2361, 2.2361, 2.2362)),
        v.bottom_up_vs_top_down(1.0, 0.0),
        v.scenario_severity(1.0, 1.0, is_stress=False),
        v.reverse_solution_check(5.0, 5.0, 2.0, (2.0,)),
    ]


def test_not_evaluated_carries_no_number():
    cases = [
        v.bottom_up_vs_top_down(1.0, 0.0),
        v.scenario_severity(1.0, 1.0, is_stress=False),
        v.scenario_severity(1.0, -1.0),
        v.reverse_solution_check(5.0, 5.0, 2.0, (2.0,)),
        v.satellite_backcast(np.full(8, 0.01), np.full(8, -0.01), qs(8), "2019-12-31"),
    ]
    for r in cases:
        assert r.light is Light.NOT_EVALUATED and r.notes.startswith("not evaluated")
        row = r.to_dict()
        assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None
        assert row["light"] == "not_evaluated" and row["reference"] and row["h0"] and row["policy_version"]
        assert "thresholds" in row["extras"]


def test_every_result_serialises_with_its_reference():
    for r in every_result():
        row = r.to_dict()
        assert set(row) == ROW_FIELDS, r.test_id
        assert row["test_id"].startswith("stress.") and row["reference"] and row["h0"] and row["policy_version"]
        assert row["light"] in {"green", "amber", "red", "not_evaluated"}
        assert "thresholds" in row["extras"] and row["alpha_amber"] is None and row["alpha_red"] is None
        assert row["inputs_hash"] != "" or row["light"] == "not_evaluated"
        assert "compliance" not in row["notes"].lower() or "no compliance" in row["notes"].lower()
    ids = {r.test_id for r in every_result()}
    assert ids == {
        "stress.satellite_backcast",
        "stress.sensitivity_monotonicity",
        "stress.static_balance_sheet_rules",
        "stress.bottom_up_vs_top_down",
        "stress.scenario_severity",
        "stress.reverse_solution_check",
    }


def test_non_finite_inputs_raise():
    nan, inf = float("nan"), float("inf")
    pred = np.full(8, 0.1)
    with pytest.raises(ValueError):
        v.satellite_backcast(np.concatenate([pred[:-1], [nan]]), pred, qs(8), "2019-12-31")
    with pytest.raises(ValueError):
        v.sensitivity_monotonicity((nan, 1.0), ("positive", "positive"))
    with pytest.raises(ValueError):
        v.static_balance_sheet_rules(np.array([[1.0, 2.0, nan]]))
    with pytest.raises(ValueError):
        v.bottom_up_vs_top_down(inf, 1.0)
    with pytest.raises(ValueError):
        v.scenario_severity(nan, 1.0)
    with pytest.raises(ValueError):
        v.reverse_solution_check(nan, 1.0, 1.0, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        v.reverse_solution_check(1.0, 1.0, 1.0, (1.0, inf, 1.0))


def test_not_evaluated_rows_carry_model_id_and_segment():
    rows = [
        v.bottom_up_vs_top_down(1.0, 0.0, model_id="M1", segment="S1"),
        v.scenario_severity(1.0, 1.0, is_stress=False, model_id="M1", segment="S1"),
        v.scenario_severity(1.0, -1.0, model_id="M1", segment="S1"),
        v.reverse_solution_check(5.0, 5.0, 2.0, (2.0,), model_id="M1", segment="S1"),
        v.satellite_backcast(
            np.full(8, 0.01), np.full(8, -0.01), qs(8), "2019-12-31", model_id="M1", segment="S1"
        ),
    ]
    for r in rows:
        assert r.light is Light.NOT_EVALUATED
        assert r.model_id == "M1" and r.segment == "S1"
