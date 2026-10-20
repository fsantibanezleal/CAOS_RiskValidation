"""RV-1308 to RV-1312: the capital engine (contract `engine-stress-api`, module stress.capital)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from riskvalidation.stress import capital as c

# Banco Santander-Chile Form 20-F 2025, Ch$ million (dossier 03 l.1324-1328)
RWA = 41_715_467.0
CET1 = 4_601_923.0
EFFECTIVE_EQUITY = 7_047_321.0
PILLAR2 = 52_144.0

ANNEX3_IDS = [
    "1",
    "1.1",
    "1.2",
    "1.3",
    "2",
    "2.T2",
    "2.AT1",
    "2.CET1",
    "3",
    "4",
    "4.1",
    "4.1.1",
    "4.1.2",
    "4.2",
    "4.3",
    "4.4",
    "4.4.1",
    "4.4.2",
    "4.4.3",
    "4.5",
    "4.6",
    "4.7",
    "4.8",
    "5",
    "6",
    "7",
    "7.1",
    "7.2",
    "8",
    "9",
    "10.1",
    "10.2",
    "11",
    "12",
]
# the labels as the CMF prints them (addendum F7, Circular 2.365, Annex 3), typed here from the addendum text
ANNEX3_LABELS = {
    "1": "Ratios de capital",
    "1.1": "PE / APR",
    "1.2": "T1 / APR",
    "1.3": "CET1 / APR",
    "2": "Patrimonio efectivo",
    "2.T2": "T2",
    "2.AT1": "AT1",
    "2.CET1": "CET1",
    "3": "APR",
    "4": "Cuantificacion del capital necesario",
    "4.1": "Riesgo de credito",
    "4.1.1": "Metodo estandar",
    "4.1.2": "Modelo interno",
    "4.2": "Riesgo de mercado (ME)",
    "4.3": "Riesgo operacional (ME)",
    "4.4": "Riesgo de concentracion",
    "4.4.1": "Individual",
    "4.4.2": "Sectorial",
    "4.4.3": "Otro tipo",
    "4.5": "RMLB",
    "4.6": "Riesgo estrategico",
    "4.7": "Riesgo reputacional",
    "4.8": "Otros riesgos materiales",
    "5": "OI",
    "6": "Cuantificacion del capital necesario por riesgos no cubiertos en pilar 1 (4.4+4.5+4.6+4.7+4.8)",
    "7": "Exigencia minima de PE, en periodo de planificacion (8%+7.1+7.2)",
    "7.1": "Exigencia adicional de PE por art.66 quater (banco sistemico)",
    "7.2": "Exigencia adicional de PE por art.66 quinquies",
    "8": "Cuantificacion del capital necesario por riesgos no cubiertos en pilar 1, "
    "y no exigidos por art.66 quinquies (6-7.2)",
    "9": "Colchones de conservacion y contra ciclico, en periodo de planificacion (art.66 bis y 66 ter)",
    "10.1": "Impacto en indice de capital basico de las pruebas de tension severas, sin mitigacion",
    "10.2": "Impacto en indice de capital basico de las pruebas de tension severas, con mitigacion",
    "11": "Componente d) del OI",
    "12": "Holgura de indice de patrimonio efectivo respecto del OI (1.1 - 5)",
}


def arr(*x):
    return np.array(x, dtype=float)


# ---------------------------------------------------------------------------------------------------- RV-1308


def test_cet1_path_loss_only_golden_and_invariants():
    out = c.cet1_path_loss_only(11.0, (1.0, 2.0, 1.5))
    assert out == pytest.approx([10.0, 8.0, 6.5], abs=1e-15)
    assert out / 100.0 == pytest.approx([0.10, 0.08, 0.065], abs=1e-15)
    rng = np.random.default_rng(5)
    losses = rng.uniform(0, 3, 6)
    path = c.cet1_path_loss_only(20.0, losses)
    assert np.all(np.diff(path) <= 0.0) and path[-1] == pytest.approx(20.0 - losses.sum(), abs=1e-12)
    assert np.all(c.cet1_path_loss_only(5.0, (0.0, 0.0)) == 5.0)
    with pytest.raises(ValueError, match="losses"):
        c.cet1_path_loss_only(11.0, (1.0, -0.1))
    with pytest.raises(ValueError, match="cet1_0"):
        c.cet1_path_loss_only(float("nan"), (1.0,))
    with pytest.raises(ValueError, match="losses"):
        c.cet1_path_loss_only(11.0, ())


def test_ppnr_row_golden_and_refusals():
    r = c.ppnr_row(660000.0, 0.25, 0.55)
    assert (r.nii, r.fee_income, r.operating_cost, r.ppnr) == pytest.approx((660000.0, 165000.0, 453750.0, 371250.0))
    for nii, f, k in ((100.0, 0.1, 0.3), (5e6, 0.0, 1.0), (0.0, 0.5, 0.5)):
        row = c.ppnr_row(nii, f, k)
        assert row.fee_income == pytest.approx(f * nii) and row.operating_cost == pytest.approx(
            k * (nii + row.fee_income)
        )
        assert row.ppnr == pytest.approx(nii + row.fee_income - row.operating_cost)
    for bad in ((1.0, -0.1, 0.5), (1.0, 1.1, 0.5), (1.0, 0.5, -0.1), (1.0, 0.5, 1.1)):
        with pytest.raises(ValueError, match="ratio"):
            c.ppnr_row(*bad)
    with pytest.raises(ValueError, match="nii"):
        c.ppnr_row(float("inf"), 0.1, 0.1)
    with pytest.raises(AttributeError):
        r.ppnr = 0.0  # frozen


# ---------------------------------------------------------------------------------------------------- RV-1309


def test_internal_capital_target_santander_golden_and_cap():
    t = c.internal_capital_target(c_dsib=0.015, c_p2=0.00125, c_ccob=0.025, c_ccyb=0.005, c_d=0.0, p2r=PILLAR2 / RWA)
    assert t.min_pe == 0.08 and t.oi == pytest.approx(0.12625, abs=1e-12)
    assert t.oi == pytest.approx(t.min_pe + t.c_dsib + t.c_p2 + t.c_ccob + t.c_ccyb + t.c_d, abs=1e-15)
    # the reported ratios: Pillar 2 over RWA is 0.125 percent, CET1 over RWA 0.110317, effective equity 0.168938
    assert PILLAR2 / RWA == pytest.approx(0.00125, abs=1e-5)
    assert CET1 / RWA == pytest.approx(0.110317, abs=5e-7)
    assert EFFECTIVE_EQUITY / RWA == pytest.approx(0.168938, abs=5e-7)
    assert EFFECTIVE_EQUITY / RWA - t.oi == pytest.approx(0.042688, abs=5e-7)
    # the 4 percent cap bounds p2r, never c_p2
    assert 0.04 * RWA == pytest.approx(1_668_618.68, abs=1e-6)
    assert PILLAR2 < 0.04 * RWA
    c.internal_capital_target(0.0, 0.05, 0.0, 0.0, 0.0, p2r=0.04)  # p2r exactly at the cap is allowed
    big = c.internal_capital_target(0.015, 0.09, 0.025, 0.005, 0.0)  # c_p2 has no cap
    assert big.c_p2 == 0.09 and big.oi == pytest.approx(0.08 + 0.015 + 0.09 + 0.03)
    with pytest.raises(ValueError, match="p2r"):
        c.internal_capital_target(0.0, 0.05, 0.0, 0.0, 0.0, p2r=0.0400001)
    with pytest.raises(ValueError, match="c_p2"):
        c.internal_capital_target(0.0, 0.001, 0.0, 0.0, 0.0, p2r=0.002)
    for field_index, name in enumerate(("c_dsib", "c_p2", "c_ccob", "c_ccyb", "c_d")):
        args = [0.01] * 5
        args[field_index] = -0.001
        with pytest.raises(ValueError, match=name):
            c.internal_capital_target(*args)
    with pytest.raises(ValueError, match="p2r"):
        c.internal_capital_target(0.0, 0.0, 0.0, 0.0, 0.0, p2r=-0.01)
    with pytest.raises(ValueError, match="c_d"):
        c.internal_capital_target(0.0, 0.0, 0.0, 0.0, float("nan"))


def test_chilean_cet1_stack_golden_and_invariants():
    s0 = c.chilean_cet1_stack(0.005, 0.015, 0.0)
    assert s0 == pytest.approx(0.09, abs=1e-12)
    assert CET1 / RWA - s0 == pytest.approx(0.020317, abs=5e-7)
    assert c.chilean_cet1_stack(0.005, 0.015, 0.00125) == pytest.approx(0.09125, abs=1e-12)
    # 4.5 + 2.5 percent + the three components, and increasing in each
    assert c.chilean_cet1_stack(0.0, 0.0, 0.0) == pytest.approx(0.07, abs=1e-15)
    assert c.chilean_cet1_stack(0.01, 0.0, 0.0) > c.chilean_cet1_stack(0.0, 0.0, 0.0)
    assert c.chilean_cet1_stack(0.0, 0.01, 0.0) > c.chilean_cet1_stack(0.0, 0.0, 0.0)
    assert c.chilean_cet1_stack(0.0, 0.0, 0.01) > c.chilean_cet1_stack(0.0, 0.0, 0.0)
    assert c.chilean_cet1_stack(0.025, 0.0, 0.0) == pytest.approx(0.095)
    for bad in ((-0.001, 0.0, 0.0), (0.0, -0.001, 0.0), (0.0, 0.0, -0.001), (0.0251, 0.0, 0.0)):
        with pytest.raises(ValueError):
            c.chilean_cet1_stack(*bad)


# ---------------------------------------------------------------------------------------------------- RV-1310


def test_capital_plan_golden_and_invariants():
    plan = c.capital_plan(
        rwa=100.0,
        cet1_0=11.0,
        at1=2.0,
        t2=3.0,
        base_cet1=(11.5, 12.0, 12.6),
        stress_losses=(1.0, 2.0, 1.5),
        mitigation_effects=(0.0, 1.0, 1.0),
        oi=0.145,
        legal_minimum=0.09625,
    )
    assert set(plan) == {"base", "stress-without-mitigation", "stress-with-mitigation"}
    no = plan["stress-without-mitigation"]
    assert [y.year for y in no] == [1, 2, 3]
    assert [y.cet1 for y in no] == pytest.approx([10.0, 8.0, 6.5])
    assert [y.pe for y in no] == pytest.approx([15.0, 13.0, 11.5])
    assert [y.pe_pct for y in no] == pytest.approx([0.15, 0.13, 0.115])
    assert [y.headroom_pct for y in no] == pytest.approx([0.05375, 0.03375, 0.01875], abs=1e-12)
    assert [y.deficit for y in no] == [0.0, 0.0, 0.0]
    assert all((y.rwa, y.at1, y.t2, y.requirement_pct) == (100.0, 2.0, 3.0, 0.09625) for y in no)
    mit = plan["stress-with-mitigation"]
    assert [y.cet1 for y in mit] == pytest.approx([10.0, 9.0, 8.5])
    assert all(m.pe >= n.pe for m, n in zip(mit, no, strict=False))
    base = plan["base"]
    assert [y.cet1 for y in base] == pytest.approx([11.5, 12.0, 12.6]) and all(y.requirement_pct == 0.145 for y in base)
    assert [y.pe_pct for y in base] == pytest.approx([0.165, 0.17, 0.176])
    # a deficit when PE falls under the requirement: requirement 0.13 on RWA 100 against PE 11.5 in year 3
    harsh = c.capital_plan(100.0, 11.0, 2.0, 3.0, (11.0, 11.0, 11.0), (1.0, 2.0, 1.5), (0.0, 0.0, 0.0), 0.145, 0.13)
    assert [y.deficit for y in harsh["stress-without-mitigation"]] == pytest.approx([0.0, 0.0, 1.5])
    assert all(
        y.pe_pct - y.requirement_pct == pytest.approx(y.headroom_pct) for y in harsh["stress-without-mitigation"]
    )
    with pytest.raises(ValueError, match="length"):
        c.capital_plan(100.0, 11.0, 2.0, 3.0, (1.0, 1.0), (1.0, 2.0, 1.5), (0.0, 0.0, 0.0), 0.1, 0.1)
    with pytest.raises(ValueError, match="length"):
        c.capital_plan(100.0, 11.0, 2.0, 3.0, (1.0, 1.0, 1.0), (1.0, 2.0, 1.5), (0.0, 0.0), 0.1, 0.1)
    with pytest.raises(ValueError, match="mitigation_effects"):
        c.capital_plan(100.0, 11.0, 2.0, 3.0, (1.0, 1.0, 1.0), (1.0, 2.0, 1.5), (0.0, -1.0, 0.0), 0.1, 0.1)
    with pytest.raises(ValueError, match="rwa"):
        c.capital_plan(0.0, 11.0, 2.0, 3.0, (1.0, 1.0, 1.0), (1.0, 2.0, 1.5), (0.0, 0.0, 0.0), 0.1, 0.1)
    with pytest.raises(AttributeError):
        no[0].cet1 = 0.0  # frozen


def test_p2g_proxy_golden_and_invariants():
    assert c.p2g_proxy(0.11, (0.10, 0.08, 0.065), 0.02) == pytest.approx(0.025, abs=1e-15)
    assert c.p2g_proxy(0.11, (0.10, 0.08, 0.065), 0.0) == pytest.approx(0.045, abs=1e-15)
    assert c.p2g_proxy(0.11, (0.12, 0.13, 0.115), 0.0) == 0.0  # never below the start: zero
    assert c.p2g_proxy(0.11, (0.10, 0.08, 0.065), 0.5) == 0.0  # never negative
    values = [c.p2g_proxy(0.11, (0.10, 0.08, 0.065), u) for u in np.linspace(0, 0.06, 13)]
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:], strict=False))
    # the peak is the maximum over the years, not the last one
    assert c.p2g_proxy(0.11, (0.05, 0.10, 0.09), 0.0) == pytest.approx(0.06)
    with pytest.raises(ValueError, match="usable"):
        c.p2g_proxy(0.11, (0.1,), -0.01)
    with pytest.raises(ValueError, match="cet1_ratios_severe"):
        c.p2g_proxy(0.11, (), 0.0)


# ---------------------------------------------------------------------------------------------------- RV-1311


def _annex_inputs(mitigation: bool = True):
    capital = {"cet1": 11.0, "at1": 1.25, "t2": 2.5}
    target = c.internal_capital_target(c_dsib=0.015, c_p2=0.00125, c_ccob=0.025, c_ccyb=0.005, c_d=0.01875)
    rwa_by_risk = {"credit": 70.0, "market": 20.0, "operational": 10.0}
    internal = {
        "credit": 6.0,
        "concentration_individual": None,
        "concentration_sectoral": None,
        "concentration_other": None,
        "irrbb": 0.5,
        "strategic": None,
        "reputational": None,
        "other": 0.25,
    }
    plan = c.capital_plan(
        100.0, 11.0, 1.25, 2.5, (11.5, 12.0, 12.6), (1.0, 2.0, 1.5), (0.0, 1.0, 1.0), target.oi, 0.09625
    )
    return capital, target, rwa_by_risk, internal, plan, mitigation


def _cell(row, col):
    return row[col]["status"], row[col]["value"]


def test_annex3_summary_rows_labels_and_identities():
    rows = c.annex3_summary(*_annex_inputs())
    assert [r["row_id"] for r in rows] == ANNEX3_IDS and len(rows) == 34
    assert {r["row_id"]: r["label_cmf"] for r in rows} == ANNEX3_LABELS
    for r in rows:
        assert set(r) == {"row_id", "label_cmf", "a", "b", "c", "d"}
        for col in "abcd":
            assert r[col]["status"] in {"value", "RI", "not-to-be-filled", "not-applicable"}
            assert (r[col]["value"] is None) == (r[col]["status"] != "value")
    by = {r["row_id"]: r for r in rows}
    target = _annex_inputs()[1]
    rwa = 100.0
    assert by["3"]["b"]["value"] == rwa and by["3"]["a"]["value"] == 1.0
    assert by["5"]["a"]["value"] == target.oi
    assert by["7"]["a"]["value"] == pytest.approx(0.08 + by["7.1"]["a"]["value"] + by["7.2"]["a"]["value"], abs=1e-15)
    assert by["12"]["a"]["value"] == pytest.approx(by["1.1"]["a"]["value"] - by["5"]["a"]["value"], abs=1e-15)
    assert by["8"]["a"]["value"] == pytest.approx(by["6"]["a"]["value"] - by["7.2"]["a"]["value"], abs=1e-15)
    assert by["8"]["b"]["value"] == pytest.approx(by["6"]["b"]["value"] - by["7.2"]["b"]["value"], abs=1e-12)
    assert by["9"]["a"]["value"] == pytest.approx(0.025 + 0.005, abs=1e-15)
    assert by["11"]["a"]["value"] == pytest.approx(0.01875) and by["11"]["b"]["value"] == pytest.approx(1.875)


def test_annex3_summary_golden_and_cell_statuses():
    rows = c.annex3_summary(*_annex_inputs())
    by = {r["row_id"]: r for r in rows}
    assert by["1.1"]["a"]["value"] == pytest.approx(0.1475, abs=1e-12) and by["1.1"]["b"]["value"] == pytest.approx(
        14.75
    )
    assert by["1.2"]["a"]["value"] == pytest.approx(0.1225) and by["1.3"]["a"]["value"] == pytest.approx(0.11)
    assert by["2.T2"]["a"]["value"] == pytest.approx(0.025) and by["2.AT1"]["a"]["value"] == pytest.approx(0.0125)
    assert by["12"]["a"]["value"] == pytest.approx(0.0025, abs=1e-12)
    assert by["7"]["a"]["value"] == pytest.approx(0.09625, abs=1e-12)
    # the legal requirement per risk: 8 percent of the risk's RWA; market and operational under the standardised method
    assert _cell(by["4.1"], "c") == ("value", pytest.approx(5.6)) and _cell(by["4.2"], "d") == (
        "not-to-be-filled",
        None,
    )
    assert _cell(by["4.3"], "d") == ("not-to-be-filled", None) and by["4.3"]["c"]["value"] == pytest.approx(0.8)
    assert _cell(by["4.1.2"], "d") == ("value", 6.0) and _cell(by["4.5"], "d") == ("value", 0.5)
    for rid in ("4.4.1", "4.4.2", "4.4.3", "4.6", "4.7"):
        assert _cell(by[rid], "d") == ("RI", None), rid
    assert _cell(by["4.8"], "d") == ("value", 0.25)
    # row 6 sums the internal capital of rows 4.4.x, 4.5, 4.6, 4.7 and 4.8 that are quantified
    assert by["6"]["b"]["value"] == pytest.approx(0.75) and by["6"]["a"]["value"] == pytest.approx(0.0075)
    assert by["1"]["a"]["status"] == "not-applicable"
    # row 10.1: the largest fall of CET1/RWA in the stress without mitigation, row 10.2 with mitigation, both by the
    # plan
    assert by["10.1"]["a"]["value"] == pytest.approx(0.11 - 0.065, abs=1e-12)
    assert by["10.2"]["a"]["value"] == pytest.approx(0.11 - 0.085, abs=1e-12)
    no_mit = {r["row_id"]: r for r in c.annex3_summary(*_annex_inputs(mitigation=False))}
    assert _cell(no_mit["10.2"], "a") == ("not-applicable", None)
    assert no_mit["10.1"]["a"]["value"] == pytest.approx(0.045, abs=1e-12)
    # inputs are never mutated and every refusal names the field
    snapshot = copy.deepcopy(_annex_inputs()[0]), copy.deepcopy(_annex_inputs()[3])
    inputs = _annex_inputs()
    c.annex3_summary(*inputs)
    assert inputs[0] == snapshot[0] and inputs[3] == snapshot[1]
    capital, target, rwa_by, internal, plan, m = _annex_inputs()
    with pytest.raises(ValueError, match="cet1"):
        c.annex3_summary({"at1": 1.0, "t2": 1.0}, target, rwa_by, internal, plan, m)
    with pytest.raises(ValueError, match="market"):
        c.annex3_summary(capital, target, {"credit": 1.0, "operational": 1.0}, internal, plan, m)
    with pytest.raises(ValueError, match="irrbb"):
        c.annex3_summary(capital, target, rwa_by, {k: v for k, v in internal.items() if k != "irrbb"}, plan, m)
    with pytest.raises(ValueError, match="stress-without-mitigation"):
        c.annex3_summary(capital, target, rwa_by, internal, {"base": plan["base"]}, m)


# ---------------------------------------------------------------------------------------------------- RV-1312


def test_eba_static_balance_sheet_golden_and_invariants():
    stages = arr(80, 15, 5)
    out = c.eba_static_balance_sheet(stages, arr(2, 3))
    assert out.shape == (2, 3)
    assert out[:, 2] == pytest.approx([7.0, 10.0]) and out.sum(axis=1) == pytest.approx([100.0, 100.0])
    assert (out[:, 0] + out[:, 1]) == pytest.approx([93.0, 90.0])
    # pro rata: stage 1 and stage 2 fall in proportion to their balances
    assert out[0, 0] == pytest.approx(80 - 2 * 80 / 95) and out[0, 1] == pytest.approx(15 - 2 * 15 / 95)
    assert np.array_equal(stages, arr(80, 15, 5))
    rng = np.random.default_rng(2)
    flows = rng.uniform(0, 3, 12)
    path = c.eba_static_balance_sheet(arr(500, 100, 20), flows)
    assert (
        np.allclose(path.sum(axis=1), 620.0, atol=1e-9) and np.all(np.diff(path[:, 2]) >= 0.0) and np.all(path >= 0.0)
    )
    assert np.allclose(np.diff(np.concatenate([[20.0], path[:, 2]])), flows)
    assert np.allclose(c.eba_static_balance_sheet(arr(80, 15, 5), arr(0, 0))[:, :2].sum(axis=1), 95.0)
    with pytest.raises(ValueError, match="new_s3_flows"):
        c.eba_static_balance_sheet(arr(80, 15, 5), arr(2, 94))
    with pytest.raises(ValueError, match="new_s3_flows"):
        c.eba_static_balance_sheet(arr(80, 15, 5), arr(-1, 2))
    with pytest.raises(ValueError, match="stage_balances"):
        c.eba_static_balance_sheet(arr(80, -15, 5), arr(1))
    with pytest.raises(ValueError, match="stage_balances"):
        c.eba_static_balance_sheet(arr(80, 15), arr(1))
    assert math.isfinite(float(out.sum()))
