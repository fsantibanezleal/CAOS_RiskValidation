"""RV-1024 to RV-1028: the IRRBB validation tests (contract `engine-irrbb-api`, module validation.irrbb). Each returns
one
`TestResult`; the rows serialise into the fields of `common/test-result`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.core.result import Light
from riskvalidation.irrbb import eve_nii as e
from riskvalidation.irrbb import shocks as sh
from riskvalidation.validation import irrbb as v

ROW_FIELDS = {"test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
              "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras"}
MIDS = np.array([0.0028, 0.0417, 0.1667, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 12.5,
                 17.5, 25.0])


def arr(*x):
    return np.array(x, dtype=float)


def sheet_items():
    items = np.zeros((3, 19))
    items[0, 3], items[1, 8], items[2, 15] = 100.0, -30.0, -70.0
    return items


def whole_eve():
    """The EVE of the aggregated flows, computed here by a plain loop (the identity is exact)."""
    items = sheet_items()
    flows = items.sum(axis=0)
    return sum(flows[k] * math.exp(-0.02 * MIDS[k]) for k in range(19))


# ---------------------------------------------------------------------------------------------------- RV-1024


def test_srp31_92_goldens():
    r = v.srp31_92_goldens()
    assert r.test_id == "irrbb.srp31_92_goldens" and r.light is Light.GREEN
    assert r.statistic == pytest.approx(0.0393, abs=5e-5) and r.p_value is None
    # the statistic is the flattener's deviation: -1.6393 against the printed -1.6
    a_s = math.exp(-3.5 / 4)
    flattener = 0.8 * 100 * a_s - 0.6 * 100 * (1 - a_s)
    assert r.statistic == pytest.approx(abs(flattener - (-1.6)), abs=1e-12)
    assert r.extras["thresholds"] == {"tolerance_scalar": 5e-4, "tolerance_bp": 5e-2}
    # a tolerance tighter than the printed precision turns the rule red
    assert v.srp31_92_goldens(tolerance_bp=0.03).light is Light.RED
    assert v.srp31_92_goldens(tolerance_scalar=1e-4).light is Light.RED
    assert v.srp31_92_goldens().to_dict()["reference"]
    with pytest.raises(ValueError):
        v.srp31_92_goldens(tolerance_bp=-1.0)


def test_cmf_annex1_goldens(monkeypatch):
    r = v.cmf_annex1_goldens()
    assert r.test_id == "irrbb.cmf_annex1_goldens" and r.light is Light.GREEN
    assert r.statistic == 0.0 and r.p_value is None
    assert r.n == 8 + 5 + 1 + 3 + 2  # the pairs, Hoja 9, the charge, the Annex 1 EVE golden, the Hoja 10 example
    # a perturbed formula is counted as a failed case and turns the light red
    real = e.cmf_pair
    def perturbed(clp, clf, offset=True, share=0.5):
        return real(clp, clf, offset, share) + (1.0 if clp == 30 else 0.0)

    monkeypatch.setattr(e, "cmf_pair", perturbed)
    bad = v.cmf_annex1_goldens()
    assert bad.light is Light.RED and bad.statistic >= 1.0
    monkeypatch.undo()
    real_table = sh.shock_table
    monkeypatch.setattr(
        sh, "shock_table",
        lambda cal: {**real_table(cal), "CLP": (301, 400, 200)} if cal == "cmf-annex1" else real_table(cal))
    assert v.cmf_annex1_goldens().light is Light.RED
    monkeypatch.undo()
    # the two Hoja 10 values are pinned at their printed precision: 0.1 bp off, or 1e-3 off the scalar, is red
    real_curve, real_scalars = sh.shock_curve, sh.shock_scalars
    monkeypatch.setattr(sh, "shock_curve", lambda *a, **k: real_curve(*a, **k) + 0.1)
    off_bp = v.cmf_annex1_goldens()
    assert off_bp.light is Light.RED and off_bp.statistic >= 1.0
    monkeypatch.undo()
    monkeypatch.setattr(sh, "shock_scalars", lambda *a, **k: tuple(x + 1e-3 for x in real_scalars(*a, **k)))
    assert v.cmf_annex1_goldens().light is Light.RED
    monkeypatch.undo()
    # a function that raises counts as a failed case instead of escaping (annex1_delta_eve included)
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(e, "annex1_delta_eve", boom)
    raised = v.cmf_annex1_goldens()
    assert raised.light is Light.RED and raised.statistic == 3.0
    monkeypatch.undo()
    # the row states its own rule: the exact tolerance and the printed precision of the two Hoja 10 values
    ok = v.cmf_annex1_goldens()
    assert ok.extras["thresholds"] == {"tolerance": 1e-12, "tolerance_printed": {"alpha_short": 5e-4, "bp": 5e-2}}
    assert "tolerance_printed" in ok.policy_version


# ---------------------------------------------------------------------------------------------------- RV-1025


def test_eve_sum_of_parts():
    items = sheet_items()
    flat, whole = np.full(19, 0.02), whole_eve()
    r = v.eve_sum_of_parts(items, flat, MIDS, whole)
    assert r.test_id == "irrbb.eve_sum_of_parts" and r.light is Light.GREEN
    assert r.statistic < 1e-6 and r.metric < 1e-9 and r.extras["thresholds"] == {"rel_tol": 1e-9}
    assert r.statistic == pytest.approx(abs(sum(e.eve(row, flat, MIDS) for row in items) - whole), abs=1e-15)
    bad = v.eve_sum_of_parts(items, flat, MIDS, 12.9)
    assert bad.light is Light.RED and bad.statistic == pytest.approx(abs(whole - 12.9), abs=1e-9)
    assert bad.metric == pytest.approx(abs(whole - 12.9) / abs(12.9), rel=1e-6)
    # the relative tolerance decides: a six-decimal whole EVE is red at 1e-9 and green at 1e-6
    rounded = 12.828783
    assert v.eve_sum_of_parts(items, flat, MIDS, rounded).light is Light.RED
    assert v.eve_sum_of_parts(items, flat, MIDS, rounded, rel_tol=1e-6).light is Light.GREEN
    # a zero EVE reconciled with zero parts is green, not a division by zero
    z = v.eve_sum_of_parts(np.zeros((2, 19)), flat, MIDS, 0.0)
    assert z.light is Light.GREEN and z.metric == 0.0
    with pytest.raises(ValueError):
        v.eve_sum_of_parts(items[:, :5], flat, MIDS, whole)
    with pytest.raises(ValueError):
        v.eve_sum_of_parts(items, flat, MIDS, whole, rel_tol=-1.0)


# ---------------------------------------------------------------------------------------------------- RV-1026


def test_nmd_parameter_stability():
    m = arr(1, 5)
    same = v.nmd_parameter_stability(arr(0.5, 0.5).reshape(1, 2).repeat(2, axis=0), m)
    assert same.light is Light.GREEN and same.statistic == 0.0 and same.metric == 0.0
    red = v.nmd_parameter_stability(np.array([[0.5, 0.5], [0.8, 0.2]]), m)
    assert red.light is Light.RED and red.statistic == pytest.approx(0.30, abs=1e-12)
    amber = v.nmd_parameter_stability(np.array([[0.5, 0.5], [0.65, 0.35]]), m)
    assert amber.light is Light.AMBER and amber.statistic == pytest.approx(0.15, abs=1e-12)
    assert amber.extras["thresholds"] == {"gap_amber": 0.10, "gap_red": 0.25}
    # the metric is the largest change of the weighted average maturity: 0.5 (5 - 1) / 2 -> |dW| 0.30 * 4 / 2
    assert red.metric == pytest.approx(0.30 * 4 / 2, abs=1e-12)
    # consecutive windows: the largest distance over the three pairs, not the first against the last
    w = np.array([[0.5, 0.5], [0.52, 0.48], [0.9, 0.1], [0.88, 0.12]])
    r = v.nmd_parameter_stability(w, m)
    assert r.statistic == pytest.approx(0.38, abs=1e-12) and r.light is Light.RED and r.n == 4
    # exactly on a gate stays in the milder colour
    on = np.array([[0.5, 0.5], [0.6, 0.4]])
    assert v.nmd_parameter_stability(on, m, gap_amber=0.10).light is Light.GREEN
    ne = v.nmd_parameter_stability(np.array([[0.5, 0.5]]), m)
    assert ne.light is Light.NOT_EVALUATED
    with pytest.raises(ValueError):
        v.nmd_parameter_stability(np.array([[0.5, 0.5], [0.8, 0.2]]), arr(1, 5, 7))
    with pytest.raises(ValueError):
        v.nmd_parameter_stability(np.array([[0.5, 0.5], [0.8, 0.2]]), m, gap_amber=0.3, gap_red=0.2)


def test_eve_with_without_behavioural():
    hid = v.eve_with_without_behavioural(10.0, 20.0, 100.0)
    assert hid.test_id == "irrbb.eve_with_without_behavioural" and hid.light is Light.RED
    assert hid.statistic == pytest.approx(0.10, abs=1e-12) and hid.metric == pytest.approx(10.0, abs=1e-12)
    th = hid.extras["thresholds"]
    assert (th["diff_amber"], th["diff_red"], th["threshold"]) == (0.05, 0.15, 0.15)
    assert th["ratio_behavioural"] == pytest.approx(0.10) and th["ratio_contractual"] == pytest.approx(0.20)
    # red because a breach is hidden, whatever the size: the same 0.10 difference without a hidden breach is amber
    amber = v.eve_with_without_behavioural(5.0, 15.0, 100.0)
    assert amber.statistic == pytest.approx(0.10) and amber.light is Light.AMBER
    ok = v.eve_with_without_behavioural(10.0, 12.0, 100.0)
    assert ok.light is Light.GREEN and ok.statistic == pytest.approx(0.02, abs=1e-12)
    # a hidden breach is red even for a tiny difference (behavioural 0.15 passes, contractual 0.151 breaches)
    assert v.eve_with_without_behavioural(15.0, 15.1, 100.0).light is Light.RED
    # both breach: not hidden, judged by the size of the difference
    assert v.eve_with_without_behavioural(20.0, 21.0, 100.0).light is Light.GREEN
    assert v.eve_with_without_behavioural(20.0, 40.0, 100.0).light is Light.RED
    # the behavioural measure is the larger one: the sign is reported, the size judged
    assert v.eve_with_without_behavioural(12.0, 10.0, 100.0).statistic == pytest.approx(-0.02)
    with pytest.raises(ValueError):
        v.eve_with_without_behavioural(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        v.eve_with_without_behavioural(1.0, 1.0, 100.0, diff_amber=0.2, diff_red=0.1)


# ---------------------------------------------------------------------------------------------------- RV-1027


def test_behavioural_change_at_breach():
    d2 = np.array(["2022-03-31", "2022-06-30"], dtype="datetime64[D]")
    red = v.behavioural_change_at_breach(arr(0.65, 0.75), d2, np.array([False, True]))
    assert red.test_id == "irrbb.behavioural_change_at_breach" and red.light is Light.RED
    assert red.statistic == 1.0 and red.n == 2
    green = v.behavioural_change_at_breach(arr(3.0, 3.0), d2, np.array([True, True]))
    assert green.light is Light.GREEN and green.statistic == 0.0
    assert v.behavioural_change_at_breach(arr(0.35, 0.40), d2, np.array([False, False])).light is Light.GREEN
    # a breach at the earlier date does not coincide with a change at the later one
    assert v.behavioural_change_at_breach(arr(0.35, 0.40), d2, np.array([True, False])).statistic == 0.0
    d5 = np.array(["2021-12-31", "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"], dtype="datetime64[D]")
    many = v.behavioural_change_at_breach(
        arr(0.65, 0.65, 0.70, 0.75, 0.75), d5, np.array([True, True, True, False, True]))
    assert many.statistic == 1.0 and many.light is Light.RED  # only the change at 2022-06-30 coincides with a breach
    two = v.behavioural_change_at_breach(
        arr(0.65, 0.70, 0.75, 0.80, 0.80), d5, np.array([False, True, True, False, True]))
    assert two.statistic == 2.0
    assert v.behavioural_change_at_breach(arr(0.5), d5[:1], np.array([True])).light is Light.NOT_EVALUATED
    with pytest.raises(ValueError):
        v.behavioural_change_at_breach(arr(0.65, 0.75), d2[::-1], np.array([False, True]))
    with pytest.raises(ValueError):
        v.behavioural_change_at_breach(arr(0.65, 0.75), np.array([d2[0], d2[0]]), np.array([False, True]))
    with pytest.raises(ValueError):
        v.behavioural_change_at_breach(arr(0.65, 0.75, 0.8), d2, np.array([False, True]))


def test_nii_out_of_sample():
    g = v.nii_out_of_sample(90.0, 100.0)
    assert g.test_id == "irrbb.nii_out_of_sample" and g.light is Light.GREEN
    assert g.statistic == pytest.approx(0.10, abs=1e-12) and g.metric == pytest.approx(-10.0)
    assert g.extras["thresholds"] == {"err_green": 0.10, "err_amber": 0.25}
    r = v.nii_out_of_sample(70.0, 100.0)
    assert r.light is Light.RED and r.statistic == pytest.approx(0.30, abs=1e-12)
    assert v.nii_out_of_sample(80.0, 100.0).light is Light.AMBER
    assert v.nii_out_of_sample(-20.0, -100.0).statistic == pytest.approx(0.80)
    for realised in (None, 0.0):
        ne = v.nii_out_of_sample(50.0, realised)
        assert ne.light is Light.NOT_EVALUATED and "realised" in ne.notes
    with pytest.raises(ValueError):
        v.nii_out_of_sample(1.0, 2.0, err_green=0.3, err_amber=0.2)
    with pytest.raises(ValueError):
        v.nii_out_of_sample(float("nan"), 2.0)


# ---------------------------------------------------------------------------------------------------- RV-1028


def every_result():
    items = sheet_items()
    flat = np.full(19, 0.02)
    d2 = np.array(["2022-03-31", "2022-06-30"], dtype="datetime64[D]")
    return [
        v.srp31_92_goldens(), v.cmf_annex1_goldens(),
        v.eve_sum_of_parts(items, flat, MIDS, whole_eve()),
        v.nmd_parameter_stability(np.array([[0.5, 0.5], [0.65, 0.35]]), arr(1, 5)),
        v.eve_with_without_behavioural(10.0, 20.0, 100.0),
        v.behavioural_change_at_breach(arr(0.65, 0.75), d2, np.array([False, True])),
        v.nii_out_of_sample(90.0, 100.0),
    ]


def test_not_evaluated_carries_no_number():
    cases = [
        v.nmd_parameter_stability(np.array([[0.5, 0.5]]), arr(1, 5)),
        v.nii_out_of_sample(1.0, None),
        v.nii_out_of_sample(1.0, 0.0),
        v.behavioural_change_at_breach(arr(0.5), np.array(["2022-03-31"], dtype="datetime64[D]"), np.array([True])),
    ]
    for r in cases:
        assert r.light is Light.NOT_EVALUATED and r.notes.startswith("not evaluated")
        row = r.to_dict()
        assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None
        assert row["light"] == "not_evaluated" and row["reference"] and row["h0"] and row["policy_version"]


def test_every_result_serialises_with_its_reference():
    for r in every_result():
        row = r.to_dict()
        assert set(row) == ROW_FIELDS, r.test_id
        assert row["test_id"].startswith("irrbb.") and row["reference"] and row["h0"] and row["policy_version"]
        assert row["light"] in {"green", "amber", "red", "not_evaluated"}
        assert "thresholds" in row["extras"]
        assert row["inputs_hash"] != "" or r.test_id in {"irrbb.srp31_92_goldens", "irrbb.cmf_annex1_goldens"}
        assert "compliance" not in row["notes"].lower() or "no compliance" in row["notes"].lower()


def test_non_finite_inputs_raise():
    items, flat = sheet_items(), np.full(19, 0.02)
    bad = items.copy()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        v.eve_sum_of_parts(bad, flat, MIDS, 1.0)
    with pytest.raises(ValueError):
        v.eve_sum_of_parts(items, flat, MIDS, float("inf"))
    with pytest.raises(ValueError):
        v.nmd_parameter_stability(np.array([[0.5, float("nan")], [0.5, 0.5]]), arr(1, 5))
    with pytest.raises(ValueError):
        v.eve_with_without_behavioural(float("nan"), 1.0, 100.0)
    with pytest.raises(ValueError):
        dates = np.array(["2022-03-31", "2022-06-30"], dtype="datetime64[D]")
        v.behavioural_change_at_breach(arr(0.5, float("nan")), dates, np.array([True, True]))
    with pytest.raises(ValueError):
        v.nii_out_of_sample(float("inf"), 1.0)
