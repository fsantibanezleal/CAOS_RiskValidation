"""RV-1123 to RV-1127: the liquidity validation tests (contract `engine-liquidity-api`, module validation.liquidity).
Each returns one `TestResult`; the rows serialise into the fields of `common/test-result`."""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest

from riskvalidation.core.result import Light
from riskvalidation.liquidity import cbc as cbc_module
from riskvalidation.validation import liquidity as v

ROW_FIELDS = {"test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
              "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras"}
POLICY = "contraste-liquidity-battery 1.0.0"


def arr(*x):
    return np.array(x, dtype=float)


def test_list_compounding_golden():
    r = v.list_compounding_golden()
    assert r.test_id == "liq.list_compounding_golden" and r.light is Light.GREEN
    assert r.metric == pytest.approx(0.1211484, abs=1e-7)
    assert r.statistic == pytest.approx(3.7e-7, abs=1e-8) and r.p_value is None
    # the statistic is the distance from the printed value, recomputed here by a loop
    s = 1.0
    for _ in range(125):
        s *= 1.0 - 0.00103258
    assert r.metric == pytest.approx(1.0 - s, abs=1e-12)
    assert r.statistic == pytest.approx(abs(1.0 - s - 0.121148), abs=1e-12)
    assert r.extras["thresholds"] == {"tolerance": 5e-7}
    assert v.list_compounding_golden(expected=0.12).light is Light.RED
    assert v.list_compounding_golden(tol=1e-8).light is Light.RED
    assert v.list_compounding_golden(q=0.00161494, expected=0.182930).light is Light.GREEN
    assert POLICY in r.policy_version and "25 and 37" not in r.reference and "237" in r.reference


def test_survival_monotone_in_severity():
    r = v.survival_monotone_in_severity([None, 176, 122])
    assert r.test_id == "liq.survival_monotone_in_severity" and r.light is Light.GREEN and r.statistic == 0.0
    r = v.survival_monotone_in_severity([100, 120])
    assert r.light is Light.RED and r.statistic == 20.0
    assert r.extras["thresholds"] == {"tolerance_days": 0.0}
    # a tolerance accepts a small lengthening
    assert v.survival_monotone_in_severity([100, 120], tol_days=20.0).light is Light.GREEN
    assert v.survival_monotone_in_severity([100, 120], tol_days=19.9).light is Light.RED
    # None is longer than any day: finite to None lengthens the survival period without bound
    r = v.survival_monotone_in_severity([100, None])
    assert r.light is Light.RED and r.statistic == math.inf and r.to_dict()["statistic"] is None
    assert v.survival_monotone_in_severity([None, None, 5]).light is Light.GREEN
    # only the largest increase counts, whatever else falls
    assert v.survival_monotone_in_severity([200, 100, 130, 90]).statistic == 30.0
    r = v.survival_monotone_in_severity([100])
    assert r.light is Light.NOT_EVALUATED


def test_cbc_haircut_rules(monkeypatch):
    r = v.cbc_haircut_rules()
    assert r.test_id == "liq.cbc_haircut_rules" and r.light is Light.GREEN and r.statistic == 0.0 and r.n == 9
    assert r.extras["thresholds"] == {"tolerance": 1e-12}
    # a haircut table with 0.25 for BBB+-BBB- is red
    altered = MappingProxyType({**cbc_module.ECB_EQUITY_HAIRCUTS, "BBB+-BBB-": 0.25})
    monkeypatch.setattr(cbc_module, "ECB_EQUITY_HAIRCUTS", altered)
    bad = v.cbc_haircut_rules()
    assert bad.light is Light.RED and bad.statistic == pytest.approx(0.10, abs=1e-12)
    monkeypatch.undo()
    assert v.cbc_haircut_rules().light is Light.GREEN
    # a wrong label with the right haircut is red too
    original = cbc_module.haircut_rule

    def relabelled(kind, **kw):
        h, label = original(kind, **kw)
        return h, "nominal-value" if label == "ecb-non-equity-hqla" else label

    monkeypatch.setattr(cbc_module, "haircut_rule", relabelled)
    relabel = v.cbc_haircut_rules()
    assert relabel.light is Light.RED and relabel.statistic == 0.0


def test_nlp_identity():
    cbc, out, inf = np.full(4, 100.0), arr(30, 40, 50, 60), arr(0, 10, 0, 0)
    nlp = arr(70.0, 40.0, -10.0, -70.0)
    r = v.nlp_identity(cbc, out, inf, nlp)
    assert r.test_id == "liq.nlp_identity" and r.light is Light.GREEN and r.statistic == 0.0
    assert r.extras["thresholds"] == {"tolerance": 1e-6} and r.n == 4
    bad = nlp.copy()
    bad[0] = 71.0
    r = v.nlp_identity(cbc, out, inf, bad)
    assert r.light is Light.RED and r.statistic == 1.0
    assert v.nlp_identity(cbc, out, inf, bad, tol=1.0).light is Light.GREEN
    with pytest.raises(ValueError, match="same length"):
        v.nlp_identity(cbc, out[:3], inf, nlp)


def test_runoff_adequacy_back_test():
    r = v.runoff_adequacy_back_test(0.013723, 0.85, denominator="DEPUNA")
    assert r.test_id == "liq.runoff_adequacy_back_test" and r.light is Light.RED
    assert r.statistic == pytest.approx(0.016145, abs=5e-7) and r.metric == pytest.approx(0.836277, abs=5e-7)
    assert r.p_value is None and r.segment == "DEPUNA"
    assert v.runoff_adequacy_back_test(0.85, 0.85).light is Light.GREEN
    assert v.runoff_adequacy_back_test(0.85, 0.85).statistic == 1.0
    r = v.runoff_adequacy_back_test(0.5, 0.85)
    assert r.light is Light.AMBER and r.statistic == pytest.approx(0.588235, abs=5e-7)
    # the edges: a ratio exactly at a limit takes the milder light
    assert v.runoff_adequacy_back_test(0.8, 1.0).light is Light.GREEN
    assert v.runoff_adequacy_back_test(0.5, 1.0).light is Light.AMBER
    assert v.runoff_adequacy_back_test(0.49, 1.0).light is Light.RED
    th = v.runoff_adequacy_back_test(0.5, 0.85, compared_at="model-horizon").extras["thresholds"]
    assert th == {"ratio_green": 0.8, "ratio_amber": 0.5, "compared_at": "model-horizon"}
    r = v.runoff_adequacy_back_test(0.3, 0.0)
    assert r.light is Light.NOT_EVALUATED and "realised" in r.notes
    with pytest.raises(ValueError, match="compared_at"):
        v.runoff_adequacy_back_test(0.5, 0.85, compared_at="year")
    with pytest.raises(ValueError, match="ratio_amber"):
        v.runoff_adequacy_back_test(0.5, 0.85, ratio_green=0.4, ratio_amber=0.6)


def test_runoff_vs_generator_truth():
    truth = arr(0.1, 0.2, 0.3)
    se = arr(0.01, 0.01, 0.01)
    r = v.runoff_vs_generator_truth(truth.copy(), truth, se)
    assert r.test_id == "liq.runoff_vs_generator_truth" and r.light is Light.GREEN and r.statistic == 0.0 and r.n == 3
    est = truth.copy()
    est[1] += 2.5 * 0.01
    r = v.runoff_vs_generator_truth(est, truth, se)
    assert r.light is Light.AMBER and r.statistic == pytest.approx(2.5, abs=1e-9)
    est[1] = truth[1] - 3.5 * 0.01
    r = v.runoff_vs_generator_truth(est, truth, se)
    assert r.light is Light.RED and r.statistic == pytest.approx(3.5, abs=1e-9)
    assert r.extras["thresholds"] == {"se_green": 2.0, "se_red": 3.0}
    with pytest.raises(ValueError, match="mc_se"):
        v.runoff_vs_generator_truth(truth, truth, arr(0.01, 0.0, 0.01))
    with pytest.raises(ValueError, match="mc_se"):
        v.runoff_vs_generator_truth(truth, truth, arr(0.01, -0.01, 0.01))
    with pytest.raises(ValueError, match="same length"):
        v.runoff_vs_generator_truth(truth, truth[:2], se)


def ladder_with(near_value: float, far_value: float, horizon: int = 40) -> np.ndarray:
    nlp = np.zeros(horizon)
    nlp[29], nlp[34] = near_value, far_value
    return nlp


def test_cliff_effect_diagnostic():
    r = v.cliff_effect_diagnostic(ladder_with(100.0, 40.0), 1000.0)
    assert r.test_id == "liq.cliff_effect_diagnostic" and r.light is Light.RED
    assert r.statistic == pytest.approx(-0.06, abs=1e-12) and r.metric == pytest.approx(-60.0, abs=1e-12)
    r = v.cliff_effect_diagnostic(ladder_with(100.0, 95.0), 1000.0)
    assert r.light is Light.GREEN and r.statistic == pytest.approx(-0.005, abs=1e-12)
    r = v.cliff_effect_diagnostic(ladder_with(100.0, 80.0), 1000.0)
    assert r.light is Light.AMBER and r.statistic == pytest.approx(-0.02, abs=1e-12)
    assert r.extras["thresholds"] == {"amber": -0.01, "red": -0.03}
    r = v.cliff_effect_diagnostic(np.zeros(30), 1000.0)
    assert r.light is Light.NOT_EVALUATED and "far" in r.notes
    assert v.cliff_effect_diagnostic(ladder_with(100.0, 90.0), 1000.0).light is Light.GREEN  # at -0.01: at least amber
    assert v.cliff_effect_diagnostic(ladder_with(100.0, 70.0), 1000.0).light is Light.AMBER  # exactly -0.03
    # the lights are conventions: a caller's thresholds move them
    assert v.cliff_effect_diagnostic(ladder_with(100.0, 40.0), 1000.0, amber=-0.1, red=-0.2).light is Light.GREEN
    with pytest.raises(ValueError, match="amber"):
        v.cliff_effect_diagnostic(ladder_with(100.0, 40.0), 1000.0, amber=-0.05, red=-0.01)


def test_htm_loss_effect():
    r = v.htm_loss_effect(72.0, 58.5, None, 48)
    assert r.test_id == "liq.htm_loss_effect" and r.light is Light.GREEN
    assert r.statistic == pytest.approx(13.5, abs=1e-12) and r.metric == pytest.approx(0.1875, abs=1e-12)
    assert r.extras["thresholds"] == {"tolerance": 0.0}
    r = v.htm_loss_effect(72.0, 80.0, 40, 60)
    assert r.light is Light.RED and r.statistic == -8.0
    # the CBC falls but the survival period lengthens: red
    assert v.htm_loss_effect(72.0, 58.5, 40, 60).light is Light.RED
    assert v.htm_loss_effect(72.0, 58.5, 40, None).light is Light.RED
    assert v.htm_loss_effect(72.0, 58.5, None, None).light is Light.GREEN
    assert v.htm_loss_effect(72.0, 72.0, 40, 40).light is Light.GREEN
    assert v.htm_loss_effect(0.0, 0.0, None, None).light is Light.NOT_EVALUATED


def test_uninsured_share_sensitivity():
    r = v.uninsured_share_sensitivity([0.5, 0.9], [None, 67])
    assert r.test_id == "liq.uninsured_share_sensitivity" and r.light is Light.GREEN and r.statistic == 0.0
    r = v.uninsured_share_sensitivity([0.5, 0.9], [40, 67])
    assert r.light is Light.RED and r.statistic == 27.0 and r.extras["thresholds"] == {"tolerance_days": 0.0}
    assert v.uninsured_share_sensitivity([0.5, 0.9], [40, 67], tol_days=27.0).light is Light.GREEN
    with pytest.raises(ValueError, match="shares"):
        v.uninsured_share_sensitivity([0.9, 0.5], [40, 20])
    with pytest.raises(ValueError, match="survival_days"):
        v.uninsured_share_sensitivity([0.5, 0.9], [40])
    assert v.uninsured_share_sensitivity([0.5], [40]).light is Light.NOT_EVALUATED


def test_reverse_stress_round_trip():
    r = v.reverse_stress_round_trip(2.085164, 5, 5)
    assert r.test_id == "liq.reverse_stress_round_trip" and r.light is Light.GREEN and r.statistic == 0.0
    assert r.metric == 2.085164
    r = v.reverse_stress_round_trip(2.0, 7, 5)
    assert r.light is Light.RED and r.statistic == 2.0
    assert v.reverse_stress_round_trip(2.0, 7, 5, tol_days=2.0).light is Light.GREEN
    r = v.reverse_stress_round_trip(None, None, 30)
    assert r.light is Light.NOT_EVALUATED and "refused" in r.notes
    assert r.extras["thresholds"] == {"tolerance_days": 0.0}


def test_htm_revaluation_vs_reported():
    for model, stat, light in ((81.0, 0.05, Light.GREEN), (82.0, 0.10, Light.AMBER), (85.0, 0.25, Light.RED)):
        r = v.htm_revaluation_vs_reported(model, 80.0, 100.0)
        assert r.test_id == "liq.htm_revaluation_vs_reported"
        assert r.statistic == pytest.approx(stat, abs=1e-12) and r.light is light, model
    assert v.htm_revaluation_vs_reported(81.0, 80.0, 100.0).extras["thresholds"] == {"green": 0.05, "amber": 0.15}
    r = v.htm_revaluation_vs_reported(None, 80.0, 100.0)
    assert r.light is Light.NOT_EVALUATED and "model_fair_value" in r.notes
    r = v.htm_revaluation_vs_reported(81.0, 100.0, 100.0)
    assert r.light is Light.NOT_EVALUATED and "zero" in r.notes


def _every_result():
    nlp = arr(70.0, 40.0, -10.0, -70.0)
    return [
        v.list_compounding_golden(), v.survival_monotone_in_severity([None, 176, 122]), v.cbc_haircut_rules(),
        v.nlp_identity(np.full(4, 100.0), arr(30, 40, 50, 60), arr(0, 10, 0, 0), nlp),
        v.runoff_adequacy_back_test(0.5, 0.85), v.runoff_vs_generator_truth(arr(1, 2), arr(1, 2), arr(0.1, 0.1)),
        v.cliff_effect_diagnostic(ladder_with(100.0, 80.0), 1000.0), v.htm_loss_effect(72.0, 58.5, None, 48),
        v.uninsured_share_sensitivity([0.5, 0.9], [None, 67]), v.reverse_stress_round_trip(2.085164, 5, 5),
        v.htm_revaluation_vs_reported(82.0, 80.0, 100.0),
    ]


def _not_evaluated_results():
    return [
        v.runoff_adequacy_back_test(0.3, 0.0), v.cliff_effect_diagnostic(np.zeros(30), 1000.0),
        v.reverse_stress_round_trip(None, None, 30), v.htm_revaluation_vs_reported(None, 80.0, 100.0),
        v.htm_revaluation_vs_reported(81.0, 100.0, 100.0), v.htm_loss_effect(0.0, 0.0, None, None),
        v.survival_monotone_in_severity([5]), v.uninsured_share_sensitivity([0.5], [5]),
    ]


def test_not_evaluated_carries_no_number():
    for r in _not_evaluated_results():
        assert r.light is Light.NOT_EVALUATED, r.test_id
        row = r.to_dict()
        assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None, r.test_id
        assert r.notes.startswith("not evaluated: ") and len(r.notes) > len("not evaluated: "), r.test_id
        assert r.p_value is None and r.metric is None and not np.isfinite(r.statistic), r.test_id
        assert "thresholds" in row["extras"], r.test_id


def test_every_result_serialises_with_its_reference():
    results = _every_result() + _not_evaluated_results()
    assert len(results) >= 19
    for r in results:
        row = r.to_dict()
        assert set(row) == ROW_FIELDS, r.test_id
        assert row["test_id"].startswith("liq.") and row["reference"].strip() and row["h0"].strip()
        assert row["alternative"].strip()
        assert row["policy_version"].startswith(POLICY), r.test_id
        assert isinstance(row["extras"]["thresholds"], dict), r.test_id
        assert row["light"] in {"green", "amber", "red", "not_evaluated"}
    ids = {r.test_id for r in results}
    assert ids == {f"liq.{n}" for n in (
        "list_compounding_golden", "survival_monotone_in_severity", "cbc_haircut_rules", "nlp_identity",
        "runoff_adequacy_back_test", "runoff_vs_generator_truth", "cliff_effect_diagnostic", "htm_loss_effect",
        "uninsured_share_sensitivity", "reverse_stress_round_trip", "htm_revaluation_vs_reported")}
    # the label is the packaged policy; the thresholds travel in the row's extras, so that a view can explain a light
    assert v.list_compounding_golden().policy_version == POLICY
    assert v.list_compounding_golden().extras["thresholds"]["tolerance"] == 5e-07
    assert v.runoff_adequacy_back_test(0.5, 0.85).extras["thresholds"]["ratio_green"] == 0.8
    # a non-finite input raises, never a number
    with pytest.raises(ValueError):
        v.list_compounding_golden(q=float("nan"))
    with pytest.raises(ValueError):
        v.runoff_adequacy_back_test(float("inf"), 0.85)
    with pytest.raises(ValueError):
        v.htm_loss_effect(float("nan"), 1.0, None, None)


def test_negative_tolerance_is_refused():
    """F7: a negative tolerance would turn a shortening of the survival period into a failure."""
    with pytest.raises(ValueError, match="^tol_days: "):
        v.survival_monotone_in_severity([100, 90], tol_days=-5)
    with pytest.raises(ValueError, match="^tol_days: "):
        v.uninsured_share_sensitivity([0.5, 0.9], [100, 90], tol_days=-5)
    with pytest.raises(ValueError, match="^tol_days: "):
        v.reverse_stress_round_trip(2.0, 5, 5, tol_days=-1.0)
    assert v.survival_monotone_in_severity([100, 90], tol_days=0.0).light is Light.GREEN
