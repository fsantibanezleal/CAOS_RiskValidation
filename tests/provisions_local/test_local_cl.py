"""RV-550 to RV-561: CMF CNCB Chapter B-1 (provisions.local_cl), every transcribed cell asserted against the fixture."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from riskvalidation.provisions import local_cl as cl

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "provisions_local" / "b1_tables.json")
                     .read_text(encoding="utf-8"))
BUCKETS = FIXTURE["mortgage"]["buckets"]  # "0", "1-29", "30-59", "60-89", "default"


def _close(a, b, tol=1e-12):
    assert abs(a - b) <= tol, (a, b)


def _check(cell, pi_pct, pdi_pct, pe_tol=1e-12, pe_pct=None):
    pi, pdi, pe = cell
    _close(pi, pi_pct / 100)
    _close(pdi, pdi_pct / 100)
    if pe_pct is None:
        _close(pe, pi * pdi, 1e-15)
    else:
        _close(pe, pe_pct / 100, pe_tol)
        assert abs(pe - pi * pdi) <= 5e-7


# RV-550 ---------------------------------------------------------------------------------------------------------

def test_individual_table_every_cell():
    rows = FIXTURE["individual"]["rows"]
    assert list(rows) == ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4"]
    previous = -1.0
    for cat, (pi, pdi, pe) in rows.items():
        got = cl.b1_individual_category(cat)
        _close(got[0], pi / 100)
        _close(got[1], pdi / 100)
        _close(got[2], pe / 100)
        assert abs(got[2] - got[0] * got[1]) <= 1e-9
        assert got[0] > previous
        previous = got[0]
    assert cl.b1_individual_category("A1") == pytest.approx((0.0004, 0.9, 0.00036), abs=1e-12)
    assert cl.b1_individual_category("A4") == pytest.approx((0.02, 0.875, 0.0175), abs=1e-12)
    assert cl.b1_individual_category("B4") == pytest.approx((0.45, 0.975, 0.43875), abs=1e-12)
    for bad in ("C1", "a1", "A7", "", "normal"):
        with pytest.raises(KeyError, match="B-1"):
            cl.b1_individual_category(bad)


# RV-551 ---------------------------------------------------------------------------------------------------------

def test_individual_provision():
    assert cl.b1_individual_provision(1000.0, 0.0, "A4") == pytest.approx(17.5, abs=1e-9)
    assert cl.b1_individual_provision(1000.0, 400.0, "A4", "A1") == pytest.approx(10.644, abs=1e-9)
    for cat in FIXTURE["individual"]["rows"]:
        _, _, pe = cl.b1_individual_category(cat)
        assert cl.b1_individual_provision(1234.5, 0.0, cat) == pytest.approx(1234.5 * pe, abs=1e-9)
    # full guarantee: the whole exposure takes the guarantor's parameters
    assert cl.b1_individual_provision(1000.0, 1000.0, "B4", "A1") == pytest.approx(0.36, abs=1e-9)
    assert cl.b1_individual_provision(0.0, 0.0, "A1") == 0.0  # empty exposure
    with pytest.raises(ValueError):
        cl.b1_individual_provision(1000.0, 400.0, "A4")  # guarantee with no guarantor category
    with pytest.raises(ValueError):
        cl.b1_individual_provision(1000.0, 1001.0, "A4", "A1")
    with pytest.raises(ValueError):
        cl.b1_individual_provision(1000.0, -1.0, "A4", "A1")
    with pytest.raises(ValueError):
        cl.b1_individual_provision(-1.0, 0.0, "A4")
    with pytest.raises(KeyError):
        cl.b1_individual_provision(1000.0, 0.0, "C3")


# RV-552 ---------------------------------------------------------------------------------------------------------

def test_default_provision_buckets():
    assert cl.b1_default_provision(1000.0, 700.0) == pytest.approx((0.30, "C3", 250.0), abs=1e-12)
    assert cl.b1_default_provision(1000.0, 970.0)[1] == "C1"  # TPE 0.03 inclusive
    assert cl.b1_default_provision(100.0, 20.0)[1] == "C5"  # TPE 0.80 inclusive
    assert cl.b1_default_provision(100.0, 19.0)[1] == "C6"  # TPE 0.81
    rows = FIXTURE["default"]["rows"]
    # each bucket: its upper bound inclusive, just above it the next category
    names = list(rows)
    for i, name in enumerate(names[:-1]):
        upper = rows[name]["upper_tpe_pct"]
        tpe, cat, prov = cl.b1_default_provision(100.0, 100.0 - upper)
        assert cat == name and prov == pytest.approx(100.0 * rows[name]["pp_pct"] / 100)
        _, cat2, prov2 = cl.b1_default_provision(1000.0, 1000.0 - (upper * 10 + 1))
        assert cat2 == names[i + 1] and prov2 == pytest.approx(1000.0 * rows[names[i + 1]]["pp_pct"] / 100)
    # float TPE at a boundary lands in the inclusive bucket: (0.3 - 0.1 ...) style sums
    assert cl.b1_default_provision(3.0, 3.0 - 0.9)[1] == "C3"  # TPE 0.9/3 = 0.30000000000000004 or similar
    assert cl.b1_default_provision(10.0, 0.0)[1] == "C6"
    assert cl.b1_default_provision(10.0, 10.0)[1] == "C1"  # TPE 0
    # nondecreasing in E - R
    prev = -1.0
    for r in range(1000, -1, -10):
        prov = cl.b1_default_provision(1000.0, float(r))[2]
        assert prov >= prev
        prev = prov
    for e, r in ((1000.0, 1001.0), (1000.0, -1.0)):
        with pytest.raises(ValueError):
            cl.b1_default_provision(e, r)
    for e in (0.0, -5.0):
        with pytest.raises(ValueError):
            cl.b1_default_provision(e, 0.0)


# RV-553 ---------------------------------------------------------------------------------------------------------

_MORT_PVG = {"<=40": 0.30, "<=80": 0.60, "<=90": 0.85, ">90": 0.95}


def test_mortgage_table_every_cell():
    n = 0
    for key, pvg in _MORT_PVG.items():
        row = FIXTURE["mortgage"]["rows"][key]
        for j, bucket in enumerate(BUCKETS):
            cell = cl.b1_mortgage_pe(pvg, bucket)
            _check(cell, row["PI"][j], row["PDI"][j], pe_tol=1e-12, pe_pct=row["PE"][j])
            n += 1
        pis = row["PI"]
        assert all(a < b for a, b in zip(pis, pis[1:]))
        assert pis[-1] == 100
    assert n == 20
    # boundaries inclusive: 40, 80, 90 per cent
    assert cl.b1_mortgage_pe(0.40, "0")[0] == pytest.approx(0.010916)
    assert cl.b1_mortgage_pe(0.4000001, "0")[0] == pytest.approx(0.019158)
    assert cl.b1_mortgage_pe(0.80, "0")[0] == pytest.approx(0.019158)
    assert cl.b1_mortgage_pe(0.90, "0")[0] == pytest.approx(0.02515)
    assert cl.b1_mortgage_pe(1.30, "0")[0] == pytest.approx(0.0274)
    assert cl.b1_mortgage_pe(0.0, "0")[0] == pytest.approx(0.010916)
    assert cl.b1_mortgage_pe(0.75, "0") == pytest.approx((0.019158, 0.021955, 0.000421), abs=1e-12)
    assert cl.b1_mortgage_pe(0.95, "default") == pytest.approx((1.0, 0.302436, 0.302436), abs=1e-12)
    with pytest.raises(ValueError, match="mortgage"):
        cl.b1_mortgage_pe(0.5, "90+")
    with pytest.raises(ValueError, match="mortgage"):
        cl.b1_mortgage_pe(0.5, "not-due")
    with pytest.raises(ValueError):
        cl.b1_mortgage_pe(-0.1, "0")


# RV-554 ---------------------------------------------------------------------------------------------------------

def test_leasing_table_every_cell():
    t = FIXTURE["leasing"]
    pvbs = [0.30, 0.45, 0.60, 0.85, 0.95]
    n = 0
    for real_estate, pi_key, pdi_key in ((True, "PI_real_estate", "PDI_real_estate"),
                                         (False, "PI_non_real_estate", "PDI_non_real_estate")):
        for i, pvb in enumerate(pvbs):
            for j, bucket in enumerate(t["buckets"]):
                _check(cl.b1_leasing_pe(pvb, bucket, real_estate), t[pi_key][j], t[pdi_key][i])
                n += 1
    assert n == 50
    assert cl.b1_leasing_pe(0.45, "0", True) == pytest.approx((0.0079, 0.0005, 3.95e-6), abs=1e-15)
    # inclusive bounds 40, 50, 80, 90
    for pvb, i in ((0.40, 0), (0.50, 1), (0.80, 2), (0.90, 3)):
        assert cl.b1_leasing_pe(pvb, "0", False)[1] == pytest.approx(t["PDI_non_real_estate"][i] / 100)
        assert cl.b1_leasing_pe(pvb + 1e-6, "0", False)[1] == pytest.approx(t["PDI_non_real_estate"][i + 1] / 100)
    assert cl.b1_leasing_pe(0.7, "default", False)[0] == 1.0
    with pytest.raises(ValueError, match="leasing"):
        cl.b1_leasing_pe(0.5, "bad", True)
    with pytest.raises(ValueError):
        cl.b1_leasing_pe(-0.5, "0", True)


# RV-555 ---------------------------------------------------------------------------------------------------------

def test_student_loan_table_every_cell():
    t = FIXTURE["student"]
    for scheme in ("CAE", "CORFO"):
        for j, bucket in enumerate(t["buckets"]):
            _check(cl.b1_student_loan_pe(scheme, bucket), t["PI_due"][scheme][j], t["PDI_due"])
        _check(cl.b1_student_loan_pe(scheme, "not-due"), t["PI_not_due"][scheme], t["PDI_not_due"][scheme])
    assert cl.b1_student_loan_pe("CAE", "0") == pytest.approx((0.052, 0.709, 0.036868), abs=1e-12)
    assert cl.b1_student_loan_pe("CAE", "not-due") == pytest.approx((0.416, 0.503, 0.209248), abs=1e-12)
    assert cl.b1_student_loan_pe("CORFO", "default")[0] == 1.0
    with pytest.raises(ValueError, match="student"):
        cl.b1_student_loan_pe("OTHER", "0")
    with pytest.raises(ValueError, match="student"):
        cl.b1_student_loan_pe("CAE", "90+")


# RV-556 ---------------------------------------------------------------------------------------------------------

def test_commercial_table_every_cell():
    t = FIXTURE["commercial"]
    ptvgs = [0.50, 0.70, 0.80, 0.95]
    n = 0
    for i, ptvg in enumerate(ptvgs):
        for j, bucket in enumerate(t["buckets"]):
            # PTVG above 100 per cent only exists in the last PDI bucket
            pi_col = t["PI_guarantee_ptvg_le_100"] if ptvg <= 1.0 else t["PI_guarantee_ptvg_gt_100"]
            _check(cl.b1_commercial_pe(bucket, ptvg), pi_col[j], t["PDI_with_guarantee"][i])
            _check(cl.b1_commercial_pe(bucket, ptvg, True), pi_col[j], t["PDI_with_guarantee_recourse"][i])
            n += 2
    for j, bucket in enumerate(t["buckets"]):
        _check(cl.b1_commercial_pe(bucket, None), t["PI_no_guarantee"][j], t["PDI_no_guarantee"])
        _check(cl.b1_commercial_pe(bucket, None, True), t["PI_no_guarantee"][j], t["PDI_no_guarantee_recourse"])
        _check(cl.b1_commercial_pe(bucket, 1.2), t["PI_guarantee_ptvg_gt_100"][j], t["PDI_with_guarantee"][3])
        _check(cl.b1_commercial_pe(bucket, 1.0), t["PI_guarantee_ptvg_le_100"][j], t["PDI_with_guarantee"][3])
        n += 4
    assert n == 60
    assert cl.b1_commercial_pe("0", None) == pytest.approx((0.0491, 0.569, 0.0279379), abs=1e-12)
    assert cl.b1_commercial_pe("0", 0.5) == pytest.approx((0.0186, 0.05, 0.00093), abs=1e-12)
    assert cl.b1_commercial_pe("default", 0.5)[0] == 1.0
    # inclusive bounds 60, 75, 90 and PDI increasing with PTVG
    for ptvg, i in ((0.60, 0), (0.75, 1), (0.90, 2)):
        assert cl.b1_commercial_pe("0", ptvg)[1] == pytest.approx(t["PDI_with_guarantee"][i] / 100)
        assert cl.b1_commercial_pe("0", ptvg + 1e-6)[1] == pytest.approx(t["PDI_with_guarantee"][i + 1] / 100)
    pdis = [cl.b1_commercial_pe("0", p)[1] for p in (0.3, 0.7, 0.8, 0.95)]
    assert pdis == sorted(pdis) and len(set(pdis)) == 4
    with pytest.raises(ValueError, match="commercial"):
        cl.b1_commercial_pe("bad", None)
    with pytest.raises(ValueError):
        cl.b1_commercial_pe("0", -0.1)


# RV-557 ---------------------------------------------------------------------------------------------------------

_DPD_REP = [(0, 7), (8, 30), (31, 60), (61, 89)]
_COLS = [(True, False), (True, True), (False, False), (False, True)]  # (mortgage, arrears)


def test_consumer_matrix_every_cell():
    t = FIXTURE["consumer"]
    n = 0
    for r, (lo, hi) in enumerate(_DPD_REP):
        for c, (mortgage, arrears) in enumerate(_COLS):
            pi_pct = t["PI"][r][c]
            for p, product in enumerate(t["products"]):
                pdi_pct = (t["PDI_with_mortgage"] if mortgage else t["PDI_without_mortgage"])[p]
                for dpd in (lo, hi):  # both ends of the row are inclusive
                    _check(cl.b1_consumer_pe(dpd, mortgage, arrears, product), pi_pct, pdi_pct)
                    n += 1
    assert n == 16 * 3 * 2
    for mortgage in (True, False):
        for arrears in (True, False):
            for p, product in enumerate(t["products"]):
                pdi_pct = (t["PDI_with_mortgage"] if mortgage else t["PDI_without_mortgage"])[p]
                for dpd in (90, 120, 1000):
                    _check(cl.b1_consumer_pe(dpd, mortgage, arrears, product), 100, pdi_pct)
    # monotonicity: down each column and from no-arrears to arrears within a mortgage status
    for c in range(4):
        col = [t["PI"][r][c] for r in range(4)]
        assert col == sorted(col)
    for r in range(4):
        assert t["PI"][r][0] < t["PI"][r][1] and t["PI"][r][2] < t["PI"][r][3]
    assert cl.b1_consumer_pe(3, True, False, "instalment") == pytest.approx((0.033, 0.477, 0.015741), abs=1e-12)
    assert cl.b1_consumer_pe(70, False, True, "cards-lines-other") == pytest.approx((0.869, 0.603, 0.524007), abs=1e-12)
    assert [cl.b1_consumer_pe(d, True, False, "instalment")[0] for d in (0, 8, 31, 61)] == [0.033, 0.204, 0.502, 0.626]
    with pytest.raises(ValueError, match="consumer"):
        cl.b1_consumer_pe(-1, True, False, "instalment")
    with pytest.raises(ValueError, match="consumer"):
        cl.b1_consumer_pe(3, True, False, "mortgage")


# RV-558 ---------------------------------------------------------------------------------------------------------

def test_group_provision():
    assert cl.b1_group_provision(1000.0, 0.02, 0.5, 0.0004) == pytest.approx(10.2, abs=1e-12)
    assert cl.b1_group_provision(1000.0, 0.02) == pytest.approx(20.0, abs=1e-12)
    assert cl.b1_group_provision(0.0, 0.02, 0.5, 0.0004) == 0.0  # empty segment
    for share in (0.0, 0.25, 0.5, 1.0):
        assert cl.b1_group_provision(1000.0, 0.02, share, 0.0004) <= 1000.0 * 0.02 + 1e-12
    with pytest.raises(ValueError):
        cl.b1_group_provision(1000.0, 0.02, 1.1, 0.0)
    with pytest.raises(ValueError):
        cl.b1_group_provision(1000.0, 0.02, -0.1, 0.0)
    with pytest.raises(ValueError):
        cl.b1_group_provision(-1.0, 0.02)


# RV-559 ---------------------------------------------------------------------------------------------------------

def test_minimum_applies_only_to_individual_normal_portfolio():
    c = cl.b1_minimum_normal_check(25e9, 100e6)
    assert (c.minimum_rate, c.minimum_amount, c.binding, c.top_up) == (0.005, pytest.approx(125e6), True, pytest.approx(25e6))
    c = cl.b1_minimum_normal_check(25e9, 150e6)
    assert (c.binding, c.top_up) == (False, 0.0)
    c = cl.b1_minimum_normal_check(25e9, 125e6)  # exactly at the minimum: not binding
    assert (c.binding, c.top_up) == (False, 0.0)
    c = cl.b1_minimum_normal_check(0.0, 0.0)  # empty segment
    assert (c.minimum_amount, c.binding, c.top_up) == (0.0, False, 0.0)
    with pytest.raises(ValueError):
        cl.b1_minimum_normal_check(-1.0, 0.0)
    with pytest.raises(ValueError):
        cl.b1_minimum_normal_check(1.0, -1.0)
    # the minimum is a property of the normal portfolio check alone: no other function carries it
    assert cl.b1_group_provision(1_000_000.0, 0.0) == 0.0
    assert cl.b1_group_provision(1_000_000.0, 0.0, 0.5, 0.0) == 0.0
    assert cl.b1_individual_provision(1_000_000.0, 0.0, "A1") == pytest.approx(360.0)  # 0.036 per cent, below 0.5
    for fn in (cl.b1_group_provision, cl.b1_individual_provision, cl.b1_default_provision, cl.b1_group_max_rule):
        assert "minimum" not in inspect.signature(fn).parameters
        assert "0.005" not in inspect.getsource(fn)


# RV-560 ---------------------------------------------------------------------------------------------------------

def test_group_max_rule():
    assert cl.b1_group_max_rule(100.0, 90.0, False) == (100.0, "max")
    assert cl.b1_group_max_rule(100.0, 110.0, False) == (110.0, "max")
    assert cl.b1_group_max_rule(100.0, 90.0, True) == (90.0, "internal")
    assert cl.b1_group_max_rule(100.0, None, False) == (100.0, "standard")
    assert cl.b1_group_max_rule(100.0, None, True) == (100.0, "standard")
    assert cl.b1_group_max_rule(0.0, None) == (0.0, "standard")  # empty segment
    for std, internal in ((50.0, 10.0), (50.0, 50.0), (0.0, 5.0)):
        assert cl.b1_group_max_rule(std, internal)[0] >= std
    with pytest.raises(ValueError):
        cl.b1_group_max_rule(-1.0, None)


# RV-561 ---------------------------------------------------------------------------------------------------------

def test_additional_provisions_own_line():
    assert cl.b1_additional_provisions(60e6, 0.0, 120e6) == 60e6
    assert cl.b1_additional_provisions(0.0, 0.0, 120e6) == 0.0
    assert cl.b1_additional_provisions(120e6, 0.0, 120e6) == 120e6
    with pytest.raises(ValueError):
        cl.b1_additional_provisions(130e6, 0.0, 120e6)
    with pytest.raises(ValueError):
        cl.b1_additional_provisions(-1.0, 0.0, 120e6)
    with pytest.raises(ValueError):
        cl.b1_additional_provisions(10.0, 20.0, 5.0)  # inverted limits
    # no comparable-total function takes the additional provisions; no IFRS 9 ECL lives in this module
    public = [n for n in dir(cl) if n.startswith("b1_")]
    assert not [n for n in public if "total" in n]
    for n in public:
        if n != "b1_additional_provisions":
            assert "additional" not in inspect.signature(getattr(cl, n)).parameters
    source = inspect.getsource(cl)
    for forbidden in ("provisions.ecl", "provisions.staging", "probability_weighted", "stage3_ecl", "discount_factors"):
        assert forbidden not in source, forbidden
