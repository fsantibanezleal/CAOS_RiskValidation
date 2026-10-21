"""RV-1222 and RV-1223: the operational risk validation tests (dossier 05 ladder and case O1), each one `TestResult`.

The tests are descriptive: p_value, alpha_amber and alpha_red are null and the lights compare a metric against
tolerances the caller supplies in `extras["thresholds"]`.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from riskvalidation.core.result import Light, TestResult
from riskvalidation.oprisk import sma
from riskvalidation.validation import oprisk as v


def arr(*x):
    return np.array(x, dtype=float)


def serialised(r: TestResult) -> dict:
    row = r.to_dict()
    json.dumps(row, allow_nan=False)  # a row never carries NaN or Infinity
    return row


# ----------------------------------------------------------------------------------------------------------------
# benchmark_reproduction


def test_benchmark_reproduction_golden():
    r = v.benchmark_reproduction((738.9056, 298095.7987, 40.3428),
                                 {"panjer": 5849.0, "fft": 5850.0, "monte-carlo": 5900.0}, 1e-6, 0.01)
    assert r.test_id == "oprisk.benchmark_reproduction"
    assert r.light is Light.GREEN
    assert r.metric == pytest.approx(0.801285, abs=5e-7) and r.statistic == r.metric
    assert r.extras["max_quantile_rel_dev"] == pytest.approx(0.008013, abs=5e-7)
    dev = r.extras["deviations"]
    assert dev["mean"] == 0.0 and dev["variance"] == 0.0 and dev["skewness"] == 0.0
    # Panjer and FFT are measured against the discretised range [5849, 5850], Monte Carlo against the continuous 5853.1
    assert dev["panjer"] == 0.0 and dev["fft"] == 0.0 and dev["monte-carlo"] == pytest.approx(46.9 / 5853.1)
    assert r.p_value is None and r.alpha_amber is None and r.alpha_red is None
    assert r.extras["thresholds"] == {"moment_rel_tol": 1e-6, "quantile_rel_tol": 0.01}
    assert "Shevchenko" in r.reference and "738.9056" in r.h0 and r.h0
    row = serialised(r)
    assert row["light"] == "green" and row["test_id"] == "oprisk.benchmark_reproduction"


def test_benchmark_reproduction_lights_and_quantile_deviation_rule():
    ok = {"panjer": 5849.0, "fft": 5849.5, "monte-carlo": 5853.1}
    base = v.benchmark_reproduction((738.9056, 298095.7987, 40.3428), ok, 1e-6, 0.01)
    assert base.metric == 0.0 and base.light is Light.GREEN                 # inside [5849, 5850] and at 5853.1 it is 0
    # the Monte Carlo quantile is measured against the continuous quantile, not the discretised range
    mc = v.benchmark_reproduction((738.9056, 298095.7987, 40.3428), {**ok, "monte-carlo": 5849.0}, 1e-6, 0.01)
    assert mc.extras["deviations"]["monte-carlo"] == pytest.approx(4.1 / 5853.1)
    assert mc.extras["deviations"]["monte-carlo"] > 0.0
    # the moment tolerance: a relative deviation of 1.5e-6 against 1e-6 is amber, 2.5e-6 red
    mean = 738.9056 * (1 + 1.5e-6)
    amber = v.benchmark_reproduction((mean, 298095.7987, 40.3428), ok, 1e-6, 0.01)
    assert amber.metric == pytest.approx(1.5, rel=1e-3) and amber.light is Light.AMBER
    red = v.benchmark_reproduction((738.9056 * (1 + 2.5e-6), 298095.7987, 40.3428), ok, 1e-6, 0.01)
    assert red.light is Light.RED
    # a quantile below 5849 is measured against 5849
    low = v.benchmark_reproduction((738.9056, 298095.7987, 40.3428), {**ok, "panjer": 5849.0 - 100.0}, 1e-6, 0.01)
    assert low.extras["max_quantile_rel_dev"] == pytest.approx(100.0 / 5849.5)
    assert low.metric == pytest.approx(100.0 / 5849.5 / 0.01) and low.light is Light.AMBER
    # a metric exactly on the gate stays on its side of it
    edge = v.benchmark_reproduction((738.9056, 298095.7987, 40.3428), {**ok, "fft": 5850.0 + 58.495}, 1e-6, 0.01)
    assert edge.metric == pytest.approx(1.0) and edge.light is Light.GREEN


def test_benchmark_reproduction_missing_method_is_not_evaluated():
    r = v.benchmark_reproduction((738.9056, 298095.7987, 40.3428), {"panjer": 5849.0, "fft": 5849.0}, 1e-6, 0.01)
    assert r.light is Light.NOT_EVALUATED and "monte-carlo" in r.notes
    row = serialised(r)
    assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None
    with pytest.raises(ValueError):
        v.benchmark_reproduction((738.9056, 298095.7987), {"panjer": 1.0}, 1e-6, 0.01)
    with pytest.raises(ValueError):
        v.benchmark_reproduction((738.9056, 298095.7987, 40.3428), {}, 0.0, 0.01)


# ----------------------------------------------------------------------------------------------------------------
# threshold_stability


def test_threshold_stability_golden():
    r = v.threshold_stability(arr(1, 2, 3, 4), arr(0.90, 0.95, 1.00, 0.97), tol_green=0.05, tol_amber=0.15)
    assert r.test_id == "oprisk.threshold_stability"
    assert r.metric == pytest.approx(0.10, abs=1e-12) and r.statistic == r.metric
    assert r.light is Light.AMBER and r.extras["n_infinite_mean"] == 1 and r.n == 4
    assert r.extras["thresholds"] == {"tol_green": 0.05, "tol_amber": 0.15}
    assert r.p_value is None and r.alpha_amber is None


def test_threshold_stability_lights_nan_and_refusals():
    th = arr(1, 2, 3, 4)
    assert v.threshold_stability(th, arr(0.6, 0.62, 0.65, 0.61)).light is Light.GREEN
    assert v.threshold_stability(th, arr(0.6, 0.7, 0.75, 0.61)).light is Light.AMBER
    assert v.threshold_stability(th, arr(0.6, 0.9, 0.75, 0.61)).light is Light.RED
    with_nan = v.threshold_stability(th, arr(0.6, math.nan, 0.65, 0.61))
    assert with_nan.n == 3 and with_nan.metric == pytest.approx(0.05) and with_nan.light is Light.GREEN
    for xi in (arr(math.nan, 0.5, math.nan, math.nan), arr(math.nan, math.nan, math.nan, math.nan)):
        r = v.threshold_stability(th, xi)
        assert r.light is Light.NOT_EVALUATED
        row = serialised(r)
        assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None and row["notes"]
    for bad_th in (arr(1, 1, 2, 3), arr(3, 2, 1, 0)):
        with pytest.raises(ValueError):
            v.threshold_stability(bad_th, arr(0.5, 0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        v.threshold_stability(th, arr(0.5, 0.5))
    with pytest.raises(ValueError):
        v.threshold_stability(th, arr(0.5, 0.5, 0.5, 0.5), tol_green=0.3, tol_amber=0.1)


# ----------------------------------------------------------------------------------------------------------------
# capital_stability


def test_capital_stability_golden():
    r = v.capital_stability(arr(90, 100, 110, 120), 100.0)
    assert r.test_id == "oprisk.capital_stability"
    assert r.metric == pytest.approx(0.15, abs=1e-12) and r.statistic == pytest.approx(0.05, abs=1e-12)
    assert r.light is Light.GREEN
    e = r.extras
    assert e["ratio_p25"] == pytest.approx(0.975) and e["ratio_median"] == pytest.approx(1.05)
    assert e["ratio_p75"] == pytest.approx(1.125)
    assert e["ratio_p05"] == pytest.approx(np.percentile(arr(90, 100, 110, 120) / 100.0, 5))
    assert e["ratio_p95"] == pytest.approx(np.percentile(arr(90, 100, 110, 120) / 100.0, 95))
    assert e["share_underestimating"] == pytest.approx(0.25)   # strictly below the truth: 90 only
    assert e["thresholds"] == {"dispersion_green": 0.25, "dispersion_amber": 0.50, "bias_green": 0.10,
                               "bias_amber": 0.25}
    assert r.n == 4 and r.p_value is None


def test_capital_stability_light_is_the_worse_of_dispersion_and_bias():
    tight_biased = v.capital_stability(arr(60, 61, 62, 63), 100.0)           # tiny dispersion, bias -0.385
    assert tight_biased.light is Light.RED
    unbiased_wide = v.capital_stability(arr(40, 80, 120, 160), 100.0)        # median 1.0, (p75 - p25) = 0.6
    assert unbiased_wide.light is Light.RED
    amber = v.capital_stability(arr(60, 100, 100, 170), 100.0)               # dispersion 0.275, bias 0.0
    assert amber.light is Light.AMBER and amber.metric == pytest.approx(0.275)
    amber_bias = v.capital_stability(arr(120, 121, 122, 123), 100.0)         # bias 0.215
    assert amber_bias.light is Light.AMBER
    # an underestimating method: every capital below the truth
    assert v.capital_stability(arr(40, 45, 50, 55), 100.0).extras["share_underestimating"] == 1.0
    few = v.capital_stability(arr(90, 100, 110), 100.0)
    assert few.light is Light.NOT_EVALUATED
    row = serialised(few)
    assert row["statistic"] is None and row["metric"] is None and row["notes"]
    for bad in (0.0, -5.0, math.nan):
        with pytest.raises(ValueError):
            v.capital_stability(arr(90, 100, 110, 120), bad)
    with pytest.raises(ValueError):
        v.capital_stability(arr(90, 100, 110, math.nan), 100.0)


# ----------------------------------------------------------------------------------------------------------------
# sma_ilm_response


def test_sma_ilm_response_golden():
    r = v.sma_ilm_response(150.0, np.full(10, 10.0), 90.0)
    assert r.test_id == "oprisk.sma_ilm_response"
    e = r.extras
    assert e["lc_before"] == pytest.approx(150.0) and e["ilm_before"] == pytest.approx(1.0, abs=1e-15)
    assert e["lc_after"] == pytest.approx(285.0)
    assert r.statistic == pytest.approx(1.220648, abs=5e-7) and r.metric == pytest.approx(0.220648, abs=5e-7)
    assert r.light is Light.RED
    assert e["orc_before"] == pytest.approx(150.0) and e["orc_after"] == pytest.approx(150.0 * sma.ilm(285.0, 150.0))
    assert e["thresholds"] == {"jump_green": 0.05, "jump_amber": 0.15}
    assert r.n == 10 and r.p_value is None


def test_sma_ilm_response_lights_added_loss_goes_to_the_latest_year_and_short_history():
    base = np.full(10, 10.0)
    zero = v.sma_ilm_response(150.0, base, 0.0)
    assert zero.light is Light.GREEN and zero.metric == 0.0
    small = v.sma_ilm_response(150.0, base, 5.0)
    assert small.light is Light.GREEN and 0.0 < small.metric <= 0.05
    mid = v.sma_ilm_response(150.0, base, 30.0)
    assert mid.light is Light.AMBER
    # the window is the last ten years: the added loss lands in the latest of them
    long = np.concatenate([np.full(3, 500.0), base])
    assert v.sma_ilm_response(150.0, long, 90.0).extras["lc_after"] == pytest.approx(285.0)
    # under 5 years of data OPE25.15 gives no ILM: not evaluated, with the reason, and no number
    r = v.sma_ilm_response(150.0, np.full(4, 10.0), 90.0)
    assert r.light is Light.NOT_EVALUATED and "5" in r.notes
    row = serialised(r)
    assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None
    assert v.sma_ilm_response(150.0, np.full(5, 10.0), 90.0).light is not Light.NOT_EVALUATED
    for bad in ((0.0, base, 1.0), (150.0, base, -1.0), (150.0, arr(10, -1, 10, 10, 10), 1.0)):
        with pytest.raises(ValueError):
            v.sma_ilm_response(*bad)


# ----------------------------------------------------------------------------------------------------------------
# scenario_influence


def test_scenario_influence_test_golden_and_cases():
    r = v.scenario_influence(arr(0.05, 0.0, 0.2))
    assert r.test_id == "oprisk.scenario_influence"
    assert r.metric == pytest.approx(0.2) and r.statistic == r.metric and r.light is Light.AMBER
    assert r.extras["influences"] == [0.05, 0.0, 0.2] and r.extras["argmax"] == 2
    assert r.extras["thresholds"] == {"influence_amber": 0.10, "influence_red": 0.25}
    assert v.scenario_influence(arr(0.05, -0.02)).light is Light.GREEN
    assert v.scenario_influence(arr(0.05, -0.3)).light is Light.RED                     # a signed influence: |.|
    assert v.scenario_influence(arr(0.05, -0.3)).extras["argmax"] == 1
    empty = v.scenario_influence(arr())
    assert empty.light is Light.NOT_EVALUATED
    row = serialised(empty)
    assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None
    with pytest.raises(ValueError):
        v.scenario_influence(arr(0.1, math.nan))
    with pytest.raises(ValueError):
        v.scenario_influence(arr(0.1), influence_amber=0.3, influence_red=0.1)


# ----------------------------------------------------------------------------------------------------------------
# every result


def test_every_result_serialises_with_its_reference_and_the_common_fields():
    results = [
        v.benchmark_reproduction((738.9056, 298095.7987, 40.3428),
                                 {"panjer": 5849.0, "fft": 5849.0, "monte-carlo": 5849.0}, 1e-6, 0.01, model_id="m1"),
        v.threshold_stability(arr(1, 2, 3), arr(0.5, 0.55, 0.6), model_id="m1"),
        v.capital_stability(arr(90, 100, 110, 120), 100.0, model_id="m1"),
        v.sma_ilm_response(150.0, np.full(10, 10.0), 90.0, model_id="m1"),
        v.scenario_influence(arr(0.05, 0.0, 0.2), model_id="m1"),
    ]
    fields = {"test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
              "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras"}
    for r in results:
        row = serialised(r)
        assert set(row) == fields and row["model_id"] == "m1"
        assert row["reference"] and row["h0"] and row["policy_version"].startswith("contraste-oprisk-battery 1.0.0")
        assert row["p_value"] is None and row["alpha_amber"] is None and row["alpha_red"] is None
        assert row["test_id"].startswith("oprisk.") and "thresholds" in row["extras"]
        assert "compliance" not in row["h0"].lower()


def test_non_finite_inputs_raise():
    with pytest.raises(ValueError):
        v.capital_stability(arr(1, 2, 3, math.inf), 1.0)
    with pytest.raises(ValueError):
        v.sma_ilm_response(150.0, arr(10, 10, 10, 10, math.nan), 1.0)
    with pytest.raises(ValueError):
        v.benchmark_reproduction((math.nan, 1.0, 1.0), {"panjer": 1.0, "fft": 1.0, "monte-carlo": 1.0}, 1e-6, 0.01)
