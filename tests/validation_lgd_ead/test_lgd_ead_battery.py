"""RV-618 to RV-621: the LGD and EAD validation tests, each a `TestResult` at its golden value, its light rule and its
refusal of a test that cannot be evaluated (contract `engine-lgd-ead-api`, module validation.lgd_ead)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.core.result import Light, TestResult
from riskvalidation.validation import lgd_ead as v

CONTRACT_KEYS = {
    "test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
    "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras",
}


def a(*x):
    return np.array(x, dtype=float)


def t_two_sided_df2(t):
    return 1.0 - abs(t) / math.sqrt(t * t + 2.0)


# ---------------------------------------------------------------- RV-618


def test_lgd_calibration_by_segment():
    r = v.lgd_calibration_by_segment(a(0.4, 0.5, 0.6), a(0.3, 0.3, 0.3), segment="mortgage", model_id="m1")
    assert isinstance(r, TestResult) and r.test_id == "lgd_ead.lgd_calibration_by_segment"
    assert r.statistic == pytest.approx(3.464102, abs=5e-7)
    assert r.p_value == pytest.approx(0.074180, abs=5e-7)
    assert r.p_value == pytest.approx(t_two_sided_df2(3.464102), abs=5e-7)
    assert r.metric == pytest.approx(0.2, abs=1e-12)
    assert r.light is Light.GREEN and r.segment == "mortgage" and r.model_id == "m1" and r.n == 3
    assert (r.alpha_amber, r.alpha_red) == (0.05, 0.01)
    assert r.extras["mean_realised"] == pytest.approx(0.5) and r.extras["mean_predicted"] == pytest.approx(0.3)
    assert r.extras["ci_low"] < 0.2 < r.extras["ci_high"]
    tcrit = stats.t.ppf(0.975, 2)
    se = 0.1 / math.sqrt(3)
    assert r.extras["ci_low"] == pytest.approx(0.2 - tcrit * se, abs=1e-9)
    assert r.extras["ci_high"] == pytest.approx(0.2 + tcrit * se, abs=1e-9)
    # an independent computation of the paired t
    ref = stats.ttest_rel(a(0.4, 0.5, 0.6), a(0.3, 0.3, 0.3))
    assert r.statistic == pytest.approx(ref.statistic, abs=1e-9) and r.p_value == pytest.approx(ref.pvalue, abs=1e-9)
    assert r.reference and r.h0 and r.policy_version == "contraste-lgd-ead-battery 1.0.0" and r.alpha_amber == 0.05
    # realised values outside [0, 1] are kept (not clipped) and counted
    pred = a(0.3, 0.3, 0.3, 0.3)
    real = a(-0.2, 0.5, 0.6, 1.3)
    o = v.lgd_calibration_by_segment(real, pred)
    assert o.extras["n_below_zero"] == 1 and o.extras["n_above_one"] == 1
    assert o.extras["mean_realised"] == pytest.approx(real.mean())
    # the light follows the p-value: amber between alpha_red and alpha_amber, red below
    d = np.linspace(0.05, 0.15, 12)
    assert v.lgd_calibration_by_segment(0.3 + d, np.full(12, 0.3)).light is Light.RED
    p_amber = v.lgd_calibration_by_segment(a(0.4, 0.5, 0.6), a(0.3, 0.3, 0.3), alpha_amber=0.10,
                                           alpha_red=0.01)
    assert p_amber.light is Light.AMBER
    assert v.lgd_calibration_by_segment(a(0.4, 0.5, 0.6), a(0.3, 0.3, 0.3), alpha_amber=0.10,
                                        alpha_red=0.08).light is Light.RED
    # no spread in the gaps: a certain gap is red with null statistic and p_value and the gap in the metric
    certain = v.lgd_calibration_by_segment(a(0.4, 0.5, 0.6), a(0.3, 0.4, 0.5))
    assert certain.light is Light.RED and certain.metric == pytest.approx(0.1, abs=1e-12)
    row = certain.to_dict()
    assert row["statistic"] is None and row["p_value"] is None and row["metric"] == pytest.approx(0.1, abs=1e-12)
    none = v.lgd_calibration_by_segment(a(0.3, 0.5, 0.6), a(0.3, 0.5, 0.6))
    assert none.light is Light.GREEN and none.metric == 0.0 and none.to_dict()["statistic"] is None
    with pytest.raises(ValueError, match="alpha"):
        v.lgd_calibration_by_segment(a(0.4, 0.5), a(0.3, 0.3), alpha_amber=0.01, alpha_red=0.05)
    with pytest.raises(ValueError, match="same"):
        v.lgd_calibration_by_segment(a(0.4, 0.5), a(0.3))


def test_lgd_calibration_over_time():
    realised = a(0.4, 0.5, 0.6, 0.3, 0.4, 0.5)
    predicted = a(0.3, 0.3, 0.3, 0.4, 0.4, 0.4)
    period = np.array([2008, 2008, 2008, 2009, 2009, 2009])
    r = v.lgd_calibration_over_time(realised, predicted, period)
    assert r.test_id == "lgd_ead.lgd_calibration_over_time"
    assert r.statistic == pytest.approx(3.464102, abs=5e-7)
    assert r.p_value == pytest.approx(0.148360, abs=5e-7)
    assert r.p_value == pytest.approx(2 * 0.074180, abs=1e-6)
    assert r.metric == pytest.approx(0.2, abs=1e-12) and r.light is Light.GREEN
    rows = r.extras["periods"]
    assert [p["period"] for p in rows] == [2008, 2009] and [p["n"] for p in rows] == [3, 3]
    assert rows[0]["t"] == pytest.approx(3.464102, abs=5e-7) and rows[0]["p"] == pytest.approx(0.074180, abs=5e-7)
    assert rows[1]["t"] == pytest.approx(0.0, abs=1e-12) and rows[1]["p"] == pytest.approx(1.0)
    assert rows[0]["mean_gap"] == pytest.approx(0.2) and r.extras["excluded"] == []
    # periods with fewer than 2 observations or no spread are not tested; the Bonferroni K counts only those tested
    r2 = v.lgd_calibration_over_time(a(0.4, 0.5, 0.6, 0.3), a(0.3, 0.3, 0.3, 0.1), np.array([1, 1, 1, 2]))
    assert r2.p_value == pytest.approx(0.074180, abs=5e-7)  # K = 1
    # the period not tested is listed with null t and p and its reason, and its gap enters the metric
    assert r2.extras["excluded"] == [{"period": 2, "n": 1, "reason": "fewer than 2 observations"}]
    lone = [p for p in r2.extras["periods"] if p["period"] == 2][0]
    assert lone["t"] is None and lone["p"] is None and lone["mean_gap"] == pytest.approx(0.2)
    big_lone = v.lgd_calibration_over_time(a(0.4, 0.5, 0.6, 0.9), a(0.3, 0.3, 0.3, 0.1), np.array([1, 1, 1, 2]))
    assert big_lone.metric == pytest.approx(0.8)
    # the Bonferroni p-value is capped at 1
    r3 = v.lgd_calibration_over_time(a(0.4, 0.45, 0.5, 0.4, 0.45, 0.5, 0.4, 0.45, 0.5), np.full(9, 0.42),
                                     np.repeat([1, 2, 3], 3))
    assert 0.0 <= r3.p_value <= 1.0
    # red when the adjusted p-value is below alpha_red
    big = v.lgd_calibration_over_time(0.3 + np.linspace(0.05, 0.15, 20), np.full(20, 0.3), np.zeros(20, dtype=int))
    assert big.light is Light.RED
    none = v.lgd_calibration_over_time(a(0.4, 0.5), a(0.3, 0.3), np.array([1, 2]))
    assert none.light is Light.NOT_EVALUATED and none.to_dict()["metric"] is None
    # a period with zero spread and a nonzero gap makes the light red, even with other periods green
    cert = v.lgd_calibration_over_time(a(0.4, 0.5, 0.6, 0.3, 0.4, 0.5), a(0.3, 0.4, 0.5, 0.4, 0.4, 0.4),
                                       np.array([1, 1, 1, 2, 2, 2]))
    assert cert.light is Light.RED and cert.extras["excluded"][0]["reason"] == "zero spread"
    only = v.lgd_calibration_over_time(a(0.4, 0.5, 0.6), a(0.3, 0.4, 0.5), np.array([1, 1, 1]))
    assert only.light is Light.RED and only.to_dict()["statistic"] is None and only.to_dict()["p_value"] is None
    assert only.metric == pytest.approx(0.1, abs=1e-12)


# ---------------------------------------------------------------- RV-619


def test_lgd_discrimination():
    r = v.lgd_discrimination(a(0.1, 0.4, 0.6, 0.9), a(0.2, 0.3, 0.7, 0.8))
    assert r.test_id == "lgd_ead.lgd_discrimination"
    assert r.statistic == pytest.approx(1.0) and r.metric == pytest.approx(1.0) and r.light is Light.GREEN
    assert 0.0 <= r.p_value <= 1.0
    assert r.extras["thresholds"] == {"rho_green": 0.30, "rho_amber": 0.10}
    neg = v.lgd_discrimination(a(0.1, 0.4, 0.6, 0.9), a(0.8, 0.7, 0.3, 0.2))
    assert neg.statistic == pytest.approx(-1.0) and neg.light is Light.RED
    # ties by average rank, equal to scipy
    x = a(0.1, 0.1, 0.5, 0.5, 0.9, 0.2, 0.7)
    y = a(0.3, 0.1, 0.5, 0.6, 0.8, 0.2, 0.2)
    ref = stats.spearmanr(x, y)
    got = v.lgd_discrimination(x, y)
    assert got.statistic == pytest.approx(ref.statistic, abs=1e-12) and got.p_value == pytest.approx(ref.pvalue)
    # lights: green >= 0.30, amber [0.10, 0.30), red below
    def light_for(rho_target):
        # a pair of 12 values with rank correlation close to the target (via mixing)
        rng = np.random.default_rng(2)
        base = np.arange(40.0)
        noise = rng.normal(size=40)
        mix = rho_target * base / base.std() + math.sqrt(max(0.0, 1 - rho_target**2)) * noise / noise.std()
        res = v.lgd_discrimination(base, mix)
        return res.statistic, res.light
    s_hi, l_hi = light_for(0.9)
    assert s_hi >= 0.3 and l_hi is Light.GREEN
    amber = v.lgd_discrimination(a(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), a(2, 1, 4, 3, 6, 5, 8, 7, 10, 9), rho_green=0.99,
                                 rho_amber=0.5)
    assert amber.light is Light.AMBER
    assert v.lgd_discrimination(a(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), a(2, 1, 4, 3, 6, 5, 8, 7, 10, 9), rho_green=0.99,
                                rho_amber=0.98).light is Light.RED
    with pytest.raises(ValueError, match="rho"):
        v.lgd_discrimination(a(1, 2, 3), a(1, 2, 3), rho_green=0.1, rho_amber=0.3)


def test_downturn_coverage():
    r = v.downturn_coverage(0.55, a(0.5, 0.6, 0.7))
    assert r.test_id == "lgd_ead.downturn_coverage"
    assert r.statistic == pytest.approx(0.866025, abs=5e-7)
    assert r.p_value == pytest.approx(0.238884, abs=5e-7)
    assert r.p_value == pytest.approx(0.5 - 0.866025 / (2 * math.sqrt(0.866025**2 + 2)), abs=5e-7)
    assert r.metric == pytest.approx(0.05, abs=1e-12) and r.light is Light.GREEN
    assert r.alternative and "exceed" in r.alternative
    # a clear shortfall is red; an estimate above the realised losses is green with a zero shortfall
    red = v.downturn_coverage(0.30, 0.6 + np.linspace(-0.05, 0.05, 15))
    assert red.light is Light.RED and red.metric == pytest.approx(0.30, abs=1e-9)
    ok = v.downturn_coverage(0.80, a(0.5, 0.6, 0.7))
    assert ok.light is Light.GREEN and ok.metric == 0.0 and ok.statistic < 0
    # the shortfall equals the coverage_shortfall of the downturn module
    from riskvalidation.lgd_ead.downturn import coverage_shortfall
    assert coverage_shortfall(0.55, 0.6)[1] == pytest.approx(r.metric)
    # no spread: a shortfall is red (null statistic and p_value, the shortfall in the metric), no shortfall is green
    flat = v.downturn_coverage(0.5, a(0.6, 0.6, 0.6))
    assert flat.light is Light.RED and flat.metric == pytest.approx(0.1, abs=1e-12)
    assert flat.to_dict()["statistic"] is None and flat.to_dict()["p_value"] is None
    ok_flat = v.downturn_coverage(0.7, a(0.6, 0.6, 0.6))
    assert ok_flat.light is Light.GREEN and ok_flat.metric == 0.0


# ---------------------------------------------------------------- RV-620


def test_ead_backtest_by_band():
    r = v.ead_backtest_by_band(a(900, 500), a(800, 600), a(0.2, 0.8), a(0, 0.5, 1.0))
    assert r.test_id == "lgd_ead.ead_backtest_by_band"
    assert r.statistic == pytest.approx(0.0, abs=1e-12)
    assert r.metric == pytest.approx(0.166667, abs=5e-7) and r.light is Light.AMBER
    assert r.p_value is None
    bands = r.extras["bands"]
    assert bands[0]["relative_gap"] == pytest.approx(0.125) and bands[1]["relative_gap"] == pytest.approx(-1 / 6)
    assert r.extras["thresholds"] == {"gap_amber": 0.10, "gap_red": 0.25}
    assert bands[0]["gap"] == pytest.approx(100.0) and bands[1]["predicted_ead"] == 500.0
    assert (bands[0]["lower"], bands[0]["upper"], bands[0]["n"]) == (0.0, 0.5, 1)
    # lights: green <= 0.10, amber <= 0.25, red above
    def light(p):
        return v.ead_backtest_by_band(a(p), a(100.0), a(0.3), a(0, 1.0)).light
    assert light(105.0) is Light.GREEN and light(110.0) is Light.GREEN and light(120.0) is Light.AMBER
    assert light(125.0) is Light.AMBER and light(130.0) is Light.RED
    # the last band is closed, an empty band is listed with null values and stays out of the metric
    e = v.ead_backtest_by_band(a(110, 90), a(100, 100), a(0.1, 1.0), a(0, 0.4, 0.8, 1.0))
    assert e.extras["bands"][1]["n"] == 0 and e.extras["bands"][1]["relative_gap"] is None
    assert e.extras["bands"][1]["predicted_ead"] is None and e.extras["bands"][2]["n"] == 1
    assert e.metric == pytest.approx(0.10)
    # a band whose realised EAD is 0: its gap is listed, its relative gap is null and it stays out of the metric
    z = v.ead_backtest_by_band(a(50, 110), a(0, 100), a(0.1, 0.9), a(0, 0.5, 1.0))
    assert z.extras["bands"][0]["gap"] == pytest.approx(50.0) and z.extras["bands"][0]["relative_gap"] is None
    assert z.metric == pytest.approx(0.10) and z.light is Light.GREEN
    assert z.statistic == pytest.approx((160 - 100) / 100)
    # review F1: a utilisation-change prediction may be negative (the contract does not clip it): it must not crash
    neg = v.ead_backtest_by_band(a(-200, 300), a(50, 400), a(0.1, 0.6), a(0, 0.5, 1.0))
    assert neg.extras["bands"][0]["predicted_ead"] == -200.0 and neg.extras["bands"][0]["relative_gap"] == -5.0
    assert neg.metric == pytest.approx(5.0) and neg.light is Light.RED
    with pytest.raises(ValueError, match="realised_ead"):
        v.ead_backtest_by_band(a(1), a(-1), a(0.2), a(0, 1.0))
    with pytest.raises(ValueError, match="utilisation"):
        v.ead_backtest_by_band(a(1), a(1), a(1.2), a(0, 0.5, 1.0))
    with pytest.raises(ValueError, match="band_edges"):
        v.ead_backtest_by_band(a(1), a(1), a(0.2), a(0, 1.0, 0.5))
    with pytest.raises(ValueError, match="gap"):
        v.ead_backtest_by_band(a(1), a(1), a(0.2), a(0, 1.0), gap_amber=0.3, gap_red=0.1)


def test_ccf_estimator_bias():
    r = v.ccf_estimator_bias(a(0.61, 0.63, 0.65), 0.60, 0.05)
    assert r.test_id == "lgd_ead.ccf_estimator_bias"
    assert r.metric == pytest.approx(0.03, abs=1e-12)
    assert r.statistic == pytest.approx(2.598076, abs=5e-7)
    assert r.p_value == pytest.approx(0.121690, abs=5e-7)
    assert r.p_value == pytest.approx(t_two_sided_df2(2.598076), abs=5e-7)
    assert r.light is Light.GREEN and r.extras["within_tolerance"] is True
    assert r.extras["rmse"] == pytest.approx(math.sqrt(np.mean(a(0.01, 0.03, 0.05) ** 2)))
    from riskvalidation.lgd_ead.ccf import ccf_bias
    assert r.metric == pytest.approx(ccf_bias(0.63, 0.60))
    # a significant bias inside the tolerance stays green; light by the tolerance, not the p-value
    sig = v.ccf_estimator_bias(0.60 + np.array([0.0099, 0.0101, 0.0100, 0.0100]), 0.60, 0.05)
    assert sig.p_value < 0.001 and sig.light is Light.GREEN
    assert v.ccf_estimator_bias(a(0.69, 0.70, 0.71), 0.60, 0.05).light is Light.AMBER  # 0.10 <= 2 * tolerance
    assert v.ccf_estimator_bias(a(0.79, 0.80, 0.81), 0.60, 0.05).light is Light.RED
    assert v.ccf_estimator_bias(a(0.79, 0.80, 0.81), 0.60, 0.05).extras["within_tolerance"] is False
    # review F5: the amber/red boundary is 2 x the tolerance: 2.4 x is red, 1.9 x is amber, exactly 2 x is amber
    assert v.ccf_estimator_bias(a(0.71, 0.72, 0.73), 0.60, 0.05).light is Light.RED
    assert v.ccf_estimator_bias(a(0.694, 0.695, 0.696), 0.60, 0.05).light is Light.AMBER
    assert v.ccf_estimator_bias(a(0.69, 0.70, 0.71), 0.60, 0.05).light is Light.AMBER
    # R < 2 or no spread: the light is kept (it depends on the bias only), the statistic and p_value are null
    one = v.ccf_estimator_bias(a(0.64), 0.60, 0.05)
    assert one.light is Light.GREEN and one.metric == pytest.approx(0.04) and one.n == 1
    assert one.to_dict()["statistic"] is None and one.to_dict()["p_value"] is None
    flat = v.ccf_estimator_bias(a(0.8, 0.8, 0.8), 0.60, 0.05)
    assert flat.light is Light.RED and flat.to_dict()["statistic"] is None and flat.to_dict()["p_value"] is None
    # the parameters of the light travel in extras["thresholds"]
    assert r.extras["thresholds"] == {"tolerance": 0.05}
    with pytest.raises(ValueError, match="tolerance"):
        v.ccf_estimator_bias(a(0.6, 0.7), 0.6, -0.1)


# ---------------------------------------------------------------- RV-621


def _assert_not_evaluated(r, why):
    assert isinstance(r, TestResult) and r.light is Light.NOT_EVALUATED, (r.test_id, r.light)
    row = r.to_dict()
    assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None, r.test_id
    assert r.notes.startswith("not evaluated") and why in r.notes, r.notes
    assert r.reference and r.h0 and r.policy_version


def test_degenerate_inputs_are_not_evaluated():
    z = a()
    _assert_not_evaluated(v.lgd_calibration_by_segment(z, z), "observation")
    _assert_not_evaluated(v.lgd_calibration_by_segment(a(0.4), a(0.3)), "observation")
    _assert_not_evaluated(v.lgd_calibration_over_time(z, z, np.array([], dtype=int)), "period")
    _assert_not_evaluated(v.lgd_calibration_over_time(a(0.4, 0.5), a(0.3, 0.3), np.array([1, 2])), "period")
    _assert_not_evaluated(v.lgd_discrimination(a(0.1, 0.2), a(0.1, 0.2)), "observation")
    _assert_not_evaluated(v.lgd_discrimination(a(0.1, 0.2, 0.3), a(0.5, 0.5, 0.5)), "constant")
    _assert_not_evaluated(v.lgd_discrimination(z, z), "observation")
    _assert_not_evaluated(v.downturn_coverage(0.5, a(0.6)), "observation")
    _assert_not_evaluated(v.downturn_coverage(0.5, z), "observation")
    _assert_not_evaluated(v.ead_backtest_by_band(z, z, z, a(0, 1.0)), "observation")
    _assert_not_evaluated(v.ead_backtest_by_band(a(1.0), a(0.0), a(0.5), a(0, 1.0)), "realised")
    _assert_not_evaluated(v.ccf_estimator_bias(z, 0.5, 0.05), "replication")
    _assert_not_evaluated(v.ead_backtest_by_band(a(1.0, 2.0), a(0.0, 0.0), a(0.2, 0.7), a(0, 0.5, 1.0)), "realised")


def _all_results():
    return [
        v.lgd_calibration_by_segment(a(0.4, 0.5, 0.6), a(0.3, 0.3, 0.3)),
        v.lgd_calibration_over_time(a(0.4, 0.5, 0.6, 0.3, 0.4, 0.5), a(0.3, 0.3, 0.3, 0.4, 0.4, 0.4),
                                    np.array([1, 1, 1, 2, 2, 2])),
        v.lgd_discrimination(a(0.1, 0.4, 0.6, 0.9), a(0.2, 0.3, 0.7, 0.8)),
        v.downturn_coverage(0.55, a(0.5, 0.6, 0.7)),
        v.ead_backtest_by_band(a(900, 500), a(800, 600), a(0.2, 0.8), a(0, 0.5, 1.0)),
        v.ccf_estimator_bias(a(0.61, 0.63, 0.65), 0.60, 0.05),
        v.lgd_calibration_by_segment(a(0.4), a(0.3)),
    ]


def test_every_result_serialises_with_its_reference():
    import json
    ids = set()
    for r in _all_results():
        row = r.to_dict()
        assert set(row) == CONTRACT_KEYS
        assert row["reference"] and row["h0"] and row["policy_version"] and row["test_id"].startswith("lgd_ead.")
        assert row["light"] in {"green", "amber", "red", "not_evaluated", "error"}
        json.dumps(row, allow_nan=False)  # JSON-safe: no NaN leaks into a document
        assert r.inputs_hash or r.light is Light.NOT_EVALUATED
        ids.add(row["test_id"])
    assert len(ids) == 6
    # the hash depends on the data
    r1 = v.lgd_calibration_by_segment(a(0.4, 0.5, 0.6), a(0.3, 0.3, 0.3))
    r2 = v.lgd_calibration_by_segment(a(0.4, 0.5, 0.7), a(0.3, 0.3, 0.3))
    assert r1.inputs_hash and r1.inputs_hash != r2.inputs_hash


def test_non_finite_inputs_raise():
    bad = a(0.4, np.nan, 0.6)
    ok = a(0.3, 0.3, 0.3)
    with pytest.raises(ValueError, match="non-finite"):
        v.lgd_calibration_by_segment(bad, ok)
    with pytest.raises(ValueError, match="non-finite"):
        v.lgd_calibration_by_segment(ok, a(0.3, np.inf, 0.3))
    with pytest.raises(ValueError, match="non-finite"):
        v.lgd_calibration_over_time(bad, ok, np.array([1, 1, 2]))
    with pytest.raises(ValueError, match="non-finite"):
        v.lgd_discrimination(bad, ok)
    with pytest.raises(ValueError, match="non-finite"):
        v.downturn_coverage(0.5, bad)
    with pytest.raises(ValueError, match="non-finite"):
        v.downturn_coverage(float("nan"), ok)
    with pytest.raises(ValueError, match="non-finite"):
        v.ead_backtest_by_band(a(1, np.nan), a(1, 1), a(0.1, 0.2), a(0, 1.0))
    with pytest.raises(ValueError, match="non-finite"):
        v.ccf_estimator_bias(bad, 0.5, 0.05)
    with pytest.raises(ValueError, match="non-finite"):
        v.ccf_estimator_bias(ok, float("nan"), 0.05)
