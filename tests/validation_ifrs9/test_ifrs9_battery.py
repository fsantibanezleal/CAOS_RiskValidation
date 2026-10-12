"""RV-514 to RV-518: the IFRS 9 validation battery of dossier 02 I.2, each test a `TestResult` at its golden value,
its light rule and its refusal of a test that cannot be evaluated (contract `engine-provisions-api`, module ifrs9)."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from riskvalidation.core.result import Light, TestResult
from riskvalidation.provisions import reconciliation as rc
from riskvalidation.provisions.overlays import Overlay
from riskvalidation.validation import ifrs9

FIXTURE = Path(__file__).resolve().parents[1] / "provisions" / "fixtures" / "c06_reporting_date_reconciliation.json"
CONTRACT_KEYS = {
    "test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
    "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras",
}


def a(*v, dtype=float):
    return np.array(v, dtype=dtype)


# ---------------------------------------------------------------- lifetime PD backtest

M3 = a(12, 24, 36, dtype=np.int64)
PRED = a(0.01, 0.02, 0.03)
NAR = a(10000, 9000, 8000)
NEV = a(100, 180, 240)


def test_lifetime_pd_backtest():
    r = ifrs9.lifetime_pd_backtest(M3, PRED, PRED.copy(), NAR, NEV)
    assert isinstance(r, TestResult)
    assert r.test_id == "ifrs9.lifetime_pd_backtest" and r.metric == 0.0 and r.light is Light.GREEN
    assert r.reference and r.h0 and r.policy_version
    assert r.extras["estimator"] == "kaplan-meier"
    np.testing.assert_allclose(r.extras["gap"], [0.0, 0.0, 0.0])
    brier = r.extras["brier"]
    assert brier[0] == pytest.approx(0.01 * 0.99**2 + 0.99 * 0.01**2, abs=1e-15)
    assert ifrs9.lifetime_pd_backtest(M3, PRED, PRED, NAR, NEV, competing_prepayment=True).extras["estimator"] == \
        "aalen-johansen"
    # metric is the largest absolute gap
    obs = a(0.012, 0.02, 0.027)
    r = ifrs9.lifetime_pd_backtest(M3, PRED, obs, NAR, NEV)
    assert r.metric == pytest.approx(0.003, abs=1e-15)
    # without rho: relative gap, green below 0.25, amber below 0.50, red otherwise; the worst horizon decides
    g = lambda o: ifrs9.lifetime_pd_backtest(a(12), a(0.02), a(o), a(1000), a(20)).light  # noqa: E731
    assert g(0.0245) is Light.GREEN and g(0.0252) is Light.AMBER and g(0.0299) is Light.AMBER
    assert g(0.031) is Light.RED and g(0.011) is Light.AMBER and g(0.0099) is Light.RED
    assert ifrs9.lifetime_pd_backtest(M3, PRED, a(0.0101, 0.02, 0.0451), NAR, NEV).light is Light.RED
    # unobserved horizons are excluded, none observed gives not_evaluated
    part = ifrs9.lifetime_pd_backtest(M3, PRED, a(0.01, 0.02, np.nan), NAR, NEV)
    assert part.metric == 0.0 and part.n_events is not None
    none = ifrs9.lifetime_pd_backtest(M3, PRED, a(np.nan, np.nan, np.nan), NAR, NEV)
    assert none.light is Light.NOT_EVALUATED and none.notes and none.to_dict()["metric"] is None
    # the Vasicek band
    rho = 0.1
    inside = ifrs9.lifetime_pd_backtest(M3, PRED, PRED, NAR, NEV, rho=rho)
    assert inside.light is Light.GREEN and len(inside.extras["band_low"]) == 3
    lows, highs = inside.extras["band_low"], inside.extras["band_high"]
    assert all(lo <= p <= hi for lo, p, hi in zip(lows, PRED, highs, strict=True))
    assert inside.alpha_amber == 0.05 and inside.alpha_red == 0.01
    hi_amber = np.array(inside.extras["band_high"])
    hi_red = np.array(inside.extras["band_high_red"])
    assert np.all(hi_red > hi_amber)
    mid = (hi_amber + hi_red) / 2
    assert ifrs9.lifetime_pd_backtest(M3, PRED, mid, NAR, NEV, rho=rho).light is Light.AMBER
    assert ifrs9.lifetime_pd_backtest(M3, PRED, hi_red + 0.01, NAR, NEV, rho=rho).light is Light.RED
    lo_red = np.array(inside.extras["band_low_red"])
    assert ifrs9.lifetime_pd_backtest(M3, PRED, lo_red / 2, NAR, NEV, rho=rho).light is Light.RED
    # the band widens with the asset correlation
    wide = ifrs9.lifetime_pd_backtest(M3, PRED, PRED, NAR, NEV, rho=0.3)
    assert wide.extras["band_high"][0] > inside.extras["band_high"][0]
    # refusals
    for kw in ({"predicted_cumulative": a(0.01, 0.02, 1.0)}, {"predicted_cumulative": a(0.01, 0.02)},
               {"observed_cumulative": a(0.01, 0.02, 1.5)}, {"n_at_risk": a(10, 10, -1)}):
        args = {"months": M3, "predicted_cumulative": PRED, "observed_cumulative": PRED, "n_at_risk": NAR,
                "n_events": NEV}
        args.update(kw)
        with pytest.raises(ValueError):
            ifrs9.lifetime_pd_backtest(**args)
    with pytest.raises(ValueError, match="rho"):
        ifrs9.lifetime_pd_backtest(M3, PRED, PRED, NAR, NEV, rho=1.0)
    with pytest.raises(ValueError, match="months"):
        ifrs9.lifetime_pd_backtest(a(24, 12, 36, dtype=np.int64), PRED, PRED, NAR, NEV)


# ---------------------------------------------------------------- SICR effectiveness


def sicr_inputs(s1_n=1000, s1_d=10, s2_n=100, s2_d=30, direct=None):
    stage = np.concatenate([np.ones(s1_n), np.full(s2_n, 2)])
    defaulted = np.zeros(s1_n + s2_n, dtype=bool)
    defaulted[:s1_d] = True
    defaulted[s1_n:s1_n + s2_d] = True
    months = np.zeros(s1_n + s2_n)
    months[s1_n:s1_n + s2_d] = 6.0
    from_s1 = np.zeros(s1_n + s2_n, dtype=bool)
    from_s1[:s1_d] = True if direct is None else direct
    backstop = np.zeros(s1_n + s2_n, dtype=bool)
    backstop[s1_n:s1_n + 20] = True
    return stage, defaulted, 12, months, from_s1, backstop


def test_sicr_effectiveness():
    r = ifrs9.sicr_effectiveness(*sicr_inputs())
    assert r.test_id == "ifrs9.sicr_effectiveness" and r.metric == pytest.approx(30.0) and r.light is Light.GREEN
    assert r.extras["roc_auc"] == pytest.approx(35700 / 42400, abs=1e-12)
    assert r.extras["hit_rate"] == pytest.approx(0.75)
    assert r.extras["share_stage3_direct_from_stage1"] == pytest.approx(0.25)
    assert r.extras["mean_months_in_stage2"] == pytest.approx(6.0)
    assert r.extras["share_stage2_backstop_only"] == pytest.approx(0.2)
    for key in ("hit_rate", "share_stage3_direct_from_stage1", "share_stage2_backstop_only", "metric", "roc_auc"):
        lo, hi = r.extras[f"{key}_interval"]
        assert lo <= (r.metric if key == "metric" else r.extras[key]) <= hi
    assert r.n == 1100 and r.n_events == 40
    # amber between 1 and 2, red below 1
    assert ifrs9.sicr_effectiveness(*sicr_inputs(s1_d=20, s2_d=3)).light is Light.AMBER
    low = ifrs9.sicr_effectiveness(*sicr_inputs(s1_d=20, s2_d=1))
    assert low.light is Light.RED and low.metric == pytest.approx(0.5)
    # amber at least when more than half of the Stage 3 arrivals came directly from Stage 1
    over = ifrs9.sicr_effectiveness(*sicr_inputs(s1_d=30, s2_d=10))
    assert over.metric > 2.0 and over.extras["share_stage3_direct_from_stage1"] == pytest.approx(0.75)
    assert over.light is Light.AMBER
    # not evaluated: no Stage 1 defaults, no Stage 2 accounts, nothing in Stage 1
    for args in (sicr_inputs(s1_d=0), sicr_inputs(s2_n=0, s2_d=0), sicr_inputs(s1_n=0, s1_d=0)):
        out = ifrs9.sicr_effectiveness(*args)
        assert out.light is Light.NOT_EVALUATED and out.notes and out.to_dict()["metric"] is None
    no2 = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError, match="stage"):
        ifrs9.sicr_effectiveness(a(1, 3), np.array([True, False]), 12, a(0, 0), no2, no2)
    with pytest.raises(ValueError, match="same length"):
        ifrs9.sicr_effectiveness(a(1, 2), np.array([True]), 12, a(0, 0), no2, no2)
    with pytest.raises(ValueError, match="k_months"):
        ifrs9.sicr_effectiveness(*sicr_inputs()[:2], 0, *sicr_inputs()[3:])


def test_sicr_threshold_stability():
    r = ifrs9.sicr_threshold_stability(a(2.0, 2.5), a(0.0, 1.0))
    assert r.test_id == "ifrs9.sicr_threshold_stability" and r.metric == 1.0 and r.light is not Light.GREEN
    assert ifrs9.sicr_threshold_stability(a(2.0, 2.0, 1.5, 1.4), a(0.0, 1.0, 2.0, 3.0)).light is Light.GREEN
    # a multiplier rising while conditions improve is not a loosening under worsening conditions
    assert ifrs9.sicr_threshold_stability(a(2.0, 2.5), a(1.0, 0.0)).metric == 0.0
    # equal conditions are not worsening
    assert ifrs9.sicr_threshold_stability(a(2.0, 2.5), a(1.0, 1.0)).metric == 0.0
    # amber for one period of four, red from half of the periods
    one = ifrs9.sicr_threshold_stability(a(2.0, 2.5, 2.5, 2.0), a(0.0, 1.0, 1.0, 2.0))
    assert one.metric == 1.0 and one.light is Light.AMBER
    two = ifrs9.sicr_threshold_stability(a(2.0, 2.5, 3.0, 3.0), a(0.0, 1.0, 2.0, 3.0))
    assert two.metric == 2.0 and two.light is Light.RED
    assert two.extras["loosening_periods"] == [1, 2]
    assert ifrs9.sicr_threshold_stability(a(2.0), a(0.0)).light is Light.NOT_EVALUATED
    with pytest.raises(ValueError, match="same length"):
        ifrs9.sicr_threshold_stability(a(2.0, 2.5), a(0.0))
    with pytest.raises(ValueError, match="implied_multiplier"):
        ifrs9.sicr_threshold_stability(a(2.0, 0.5), a(0.0, 1.0))


# ---------------------------------------------------------------- stage migration and ECL backtest

P3 = np.array([[0.90, 0.08, 0.02], [0.20, 0.70, 0.10], [0.05, 0.15, 0.80]])


def test_stage_migration_backtest():
    r = ifrs9.stage_migration_backtest(P3, np.round(P3 * 1000).astype(np.int64))
    assert r.test_id == "ifrs9.stage_migration_backtest"
    assert r.statistic == pytest.approx(0.0, abs=1e-12) and r.p_value == pytest.approx(1.0) and r.light is Light.GREEN
    assert r.extras["rows_skipped"] == 0 and r.extras["degrees_of_freedom"] == 6
    counts = np.array([[880, 90, 30], [210, 650, 140], [40, 160, 800]])
    r = ifrs9.stage_migration_backtest(P3, counts)
    expected = sum(((counts[i] - counts[i].sum() * P3[i]) ** 2 / (counts[i].sum() * P3[i])).sum() for i in range(3))
    assert r.statistic == pytest.approx(expected, rel=1e-12)
    assert r.p_value == pytest.approx(stats.chi2.sf(expected, 6), rel=1e-9)
    assert r.light is Light.RED
    # light thresholds on the p-value: green at 0.05 and above, amber from 0.01, red below
    base = np.array([[900, 80, 20], [0, 0, 0], [0, 0, 0]])
    assert ifrs9.stage_migration_backtest(P3, base).light is Light.GREEN
    # rows with no observations are skipped and counted
    skipped = ifrs9.stage_migration_backtest(P3, base)
    assert skipped.extras["rows_skipped"] == 2 and skipped.extras["degrees_of_freedom"] == 2
    nothing = ifrs9.stage_migration_backtest(P3, np.zeros((3, 3), dtype=np.int64))
    assert nothing.light is Light.NOT_EVALUATED and nothing.notes
    # a cell with zero predicted probability and an observation is impossible under H0
    z = ifrs9.stage_migration_backtest(np.array([[1.0, 0.0, 0.0], [0.2, 0.7, 0.1], [0.05, 0.15, 0.8]]),
                                       np.array([[5, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert z.p_value == 0.0 and z.light is Light.RED
    for pm, oc in ((P3 * 1.1, counts), (P3, counts * 0.5 + 0.25), (P3, -counts), (P3[:2], counts)):
        with pytest.raises(ValueError):
            ifrs9.stage_migration_backtest(pm, oc)


def test_stage_migration_light_boundaries():
    # a one-row test with two cells: statistic chosen so the p-value falls in each band
    p = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    for dev, light in ((10, Light.GREEN), (14, Light.AMBER), (19, Light.RED)):
        counts = np.array([[100 + dev, 100 - dev, 0], [0, 0, 0], [0, 0, 0]])
        out = ifrs9.stage_migration_backtest(p, counts)
        chi = 2 * dev**2 / 100
        assert out.statistic == pytest.approx(chi) and out.light is light, (dev, out.p_value)


def test_ecl_backtest():
    r = ifrs9.ecl_backtest(a(30, 60), a(40, 60), 12)
    assert r.test_id == "ifrs9.ecl_backtest" and r.metric == pytest.approx(0.9) and r.light is Light.GREEN
    assert r.extras["horizon_months"] == 12 and r.n == 2
    cases = [(80, Light.GREEN), (125, Light.GREEN), (79, Light.AMBER), (126, Light.AMBER), (200, Light.AMBER),
             (50, Light.AMBER), (49, Light.RED), (201, Light.RED), (0, Light.RED)]
    for realised, light in cases:
        assert ifrs9.ecl_backtest(a(realised), a(100), 12).light is light, realised
    z = ifrs9.ecl_backtest(a(1, 2), a(0, 0), 12)
    assert z.light is Light.NOT_EVALUATED and z.to_dict()["metric"] is None and z.notes
    with pytest.raises(ValueError, match="same shape"):
        ifrs9.ecl_backtest(a(1, 2), a(1), 12)
    with pytest.raises(ValueError, match="horizon_months"):
        ifrs9.ecl_backtest(a(1), a(1), 0)
    with pytest.raises(ValueError, match="realised_pv_losses"):
        ifrs9.ecl_backtest(a(-1), a(1), 12)


# ---------------------------------------------------------------- scenarios, non-linearity, overlays

PITS = a(0.005255059, 0.014287387, 0.073424082)
W3 = a(0.25, 0.5, 0.25)


def test_scenario_sensitivity():
    r = ifrs9.scenario_sensitivity(0.4 * PITS, W3, 0.4 * 0.017997448)
    assert r.test_id == "ifrs9.scenario_sensitivity"
    assert r.metric == pytest.approx(0.4898489, abs=1e-6) and r.light is Light.GREEN
    np.testing.assert_allclose(r.extras["single_scenario_ecl"], 0.4 * PITS)
    sens = np.array(r.extras["weight_sensitivity"])
    assert sens.shape == (3, 3)
    np.testing.assert_allclose(sens[1], 0.4 * PITS - 0.4 * PITS[1])
    assert "never the IFRS 9 measure" in r.notes
    e, w = a(10, 20, 60), a(0.3, 0.5, 0.2)  # weighted ECL 25
    assert ifrs9.scenario_sensitivity(e, w, 25.0).light is Light.GREEN
    assert ifrs9.scenario_sensitivity(e, w, 25.0).metric == 0.0
    assert ifrs9.scenario_sensitivity(e, w, 10.0).light is Light.AMBER  # NL = 1.5
    assert ifrs9.scenario_sensitivity(e, w, 5.0).light is Light.RED  # NL = 4
    assert ifrs9.scenario_sensitivity(e, w, 30.0).light is Light.AMBER  # concave
    z = ifrs9.scenario_sensitivity(e, w, 0.0)
    assert z.light is Light.NOT_EVALUATED and z.notes
    with pytest.raises(ValueError):
        ifrs9.scenario_sensitivity(e, a(0.5, 0.6, 0.0), 10.0)


def test_nonlinearity_test(monkeypatch):
    z = a(1.0, 0.0, -2.0)
    r = ifrs9.nonlinearity_test(0.02, 0.12, z, W3)
    assert r.test_id == "ifrs9.nonlinearity" and r.light is Light.GREEN
    assert r.extras["pd_weighted"] == pytest.approx(0.026813479, abs=1e-8)
    assert r.extras["pd_at_mean_z"] == pytest.approx(0.017997448, abs=1e-8)
    assert r.metric == pytest.approx(0.4898489, abs=1e-6)
    assert r.extras["gauss_hermite_error"] <= 1e-9
    # a failed numerical cross check is red, a numerical failure and not a model finding
    monkeypatch.setattr(ifrs9, "gauss_hermite_pit_pd", lambda pd, rho, n_nodes=64: pd + 1e-6)
    bad = ifrs9.nonlinearity_test(0.02, 0.12, z, W3)
    assert bad.light is Light.RED and bad.extras["gauss_hermite_error"] == pytest.approx(1e-6)


def ovl(oid="O1", amount=5.0, stage="stage1", segment="s1", target="pd"):
    return Overlay(id=oid, rationale_en="r", rationale_es="r", method="expert", target=target, segment_id=segment,
                   stage=stage, amount=amount, staging_effect="none", review_date="2020-12-31", currency="USD")


STAGES = {"stage1": 10.0, "stage2": 40.0, "stage3": 25.0, "poci": 0.0}


def test_overlay_diagnostics():
    r = ifrs9.overlay_diagnostics([ovl()], STAGES, 0)
    assert r.test_id == "ifrs9.overlay_diagnostics"
    assert r.metric == pytest.approx(0.0625) and r.light is Light.GREEN and r.extras["direction_consistent"] is True
    assert r.extras["share_by_stage"]["stage1"] == pytest.approx(5 / 15)
    assert r.extras["share_by_segment"]["s1"] == pytest.approx(0.0625)
    assert r.extras["by_target"]["pd"] == 5.0 and r.extras["parameter_level"] == 5.0
    # no overlays: zero share, green
    none = ifrs9.overlay_diagnostics([], STAGES, 0)
    assert none.metric == 0.0 and none.light is Light.GREEN
    # a positive overlay in an improving scenario is flagged
    inc = ifrs9.overlay_diagnostics([ovl()], STAGES, -1)
    assert inc.extras["direction_consistent"] is False and inc.light is Light.AMBER
    assert ifrs9.overlay_diagnostics([ovl()], STAGES, 1).light is Light.GREEN
    # share bands: green to 0.10, amber to 0.25, red above
    assert ifrs9.overlay_diagnostics([ovl(amount=15.0)], STAGES, 0).light is Light.AMBER  # 15/90
    assert ifrs9.overlay_diagnostics([ovl(amount=11.0)], STAGES, 0).light is Light.AMBER  # 11/86 = 0.128
    assert ifrs9.overlay_diagnostics([ovl(amount=75.0)], STAGES, 0).light is Light.RED  # 0.5
    # portfolio-level overlays are reported under the portfolio keys
    p = ifrs9.overlay_diagnostics([ovl(stage=None, segment=None)], STAGES, 0)
    assert p.extras["share_by_stage"]["portfolio"] == 1.0
    assert p.extras["share_by_segment"]["portfolio"] == pytest.approx(0.0625)
    assert ifrs9.overlay_diagnostics([ovl()], {"stage1": 0.0}, 0).light is Light.RED  # share 1.0
    zero = ifrs9.overlay_diagnostics([], {"stage1": 0.0}, 0)
    assert zero.light is Light.NOT_EVALUATED and zero.notes
    with pytest.raises(ValueError, match="scenario_direction"):
        ifrs9.overlay_diagnostics([ovl()], STAGES, 2)
    with pytest.raises(ValueError, match="unique"):
        ifrs9.overlay_diagnostics([ovl(), ovl()], STAGES, 0)


def load_fixture():
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    make = {"allowance": rc.allowance_rollforward, "gross_carrying_amount": rc.gca_rollforward}
    rows = {k: [make[k](r["stage"], r["opening"], {m["line_id"]: m["amount"] for m in r["movements"]}, r["closing"])
                for r in d["reconciliation"][k]] for k in make}
    ecl_by_stage = {r["stage"]: r["ecl"] for r in d["stage_summary"]["rows"]}
    return d, rows["allowance"], rows["gross_carrying_amount"], d["waterfall"]["lines"], ecl_by_stage


def test_reconciliation_identities():
    d, al, gca, lines, e = load_fixture()
    r = ifrs9.reconciliation_identities(al, gca, lines, e)
    assert r.test_id == "ifrs9.reconciliation_identities"
    assert r.metric is not None and abs(r.metric) <= 1e-9 and r.light is Light.GREEN
    assert r.extras["failing_identities"] == [] and r.extras["tolerance"] == pytest.approx(1e-6 * 230000)
    # the stage 2 closing differs from its ECL by 0.0025, reported and inside the tolerance
    assert r.extras["stage_ecl_gap_max"] == pytest.approx(0.0025, abs=1e-9)
    bad = copy.deepcopy(d)
    bad["reconciliation"]["allowance"][1]["movements"][0]["amount"] += 1.0
    al2 = [rc.allowance_rollforward(x["stage"], x["opening"], {m["line_id"]: m["amount"] for m in x["movements"]},
                                    x["closing"]) for x in bad["reconciliation"]["allowance"]]
    b = ifrs9.reconciliation_identities(al2, gca, lines, e)
    assert b.light is Light.RED and b.metric == pytest.approx(1.0, abs=1e-9)
    assert "allowance-rollforward:stage2" in b.extras["failing_identities"]
    # a stage ECL that does not match the allowance is red even though the roll-forwards close
    e2 = dict(e)
    e2["stage3"] += 5.0
    c = ifrs9.reconciliation_identities(al, gca, lines, e2)
    assert c.light is Light.RED and c.extras["failing_identities"] == ["stage-closing-equals-ecl:stage3"]
    empty = ifrs9.reconciliation_identities([], [], [], {})
    assert empty.light is Light.NOT_EVALUATED and empty.notes


def test_every_result_serialises_with_its_reference():
    d, al, gca, lines, e = load_fixture()
    results = [
        ifrs9.lifetime_pd_backtest(M3, PRED, PRED, NAR, NEV, rho=0.1, model_id="m1", segment="retail"),
        ifrs9.sicr_effectiveness(*sicr_inputs()),
        ifrs9.sicr_threshold_stability(a(2.0, 2.5), a(0.0, 1.0)),
        ifrs9.stage_migration_backtest(P3, np.round(P3 * 100).astype(np.int64)),
        ifrs9.ecl_backtest(a(90), a(100), 12),
        ifrs9.scenario_sensitivity(0.4 * PITS, W3, 0.4 * 0.017997448),
        ifrs9.nonlinearity_test(0.02, 0.12, a(1.0, 0.0, -2.0), W3),
        ifrs9.overlay_diagnostics([ovl()], STAGES, 0),
        ifrs9.reconciliation_identities(al, gca, lines, e),
        ifrs9.ecl_backtest(a(1), a(0), 12),
    ]
    ids = set()
    for r in results:
        row = dataclasses.asdict(r)
        assert set(row) == CONTRACT_KEYS
        out = r.to_dict()
        assert set(out) == CONTRACT_KEYS
        json.dumps(out, allow_nan=False)
        for field in ("reference", "h0", "policy_version", "alternative"):
            assert out[field].strip(), field
        assert out["light"] in {"green", "amber", "red", "not_evaluated"}
        assert out["test_id"].startswith("ifrs9.")
        ids.add(out["test_id"])
        if out["light"] != "not_evaluated":
            assert out["inputs_hash"], out["test_id"]
    assert len(ids) == 9
    first = results[0].to_dict()
    assert first["model_id"] == "m1" and first["segment"] == "retail"
    # deterministic
    again = ifrs9.ecl_backtest(a(90), a(100), 12)
    assert again.inputs_hash == results[4].inputs_hash and math.isfinite(again.metric)
