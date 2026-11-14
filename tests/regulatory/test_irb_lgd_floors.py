"""BL-053: the IRB LGD input floors in the capital requirement (`capital_requirement(..., lgd_floor=...)`).

Without the option the result is byte for byte the 1.1.0 one (a SHA-256 of the repr and of the sorted JSON of each of
1,891 calls, generated on 1.1.0 before the option existed). With it, the LGD of the risk-weight and of the expected-loss
formulas is the floored one; golden values are hand-computed from CRE99 Table 1 (K is linear in the LGD) and from the
CRE32.17 weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from riskvalidation.lgd_ead.floors import LGDFloorInput
from riskvalidation.regulatory import IRBResult, capital_requirement

HERE = Path(__file__).parent
BASELINE = json.loads((HERE / "fixtures" / "irb_no_floor_baseline_1_1_0.json").read_text(encoding="utf-8"))
CRE99 = json.loads((HERE.parent / "fixtures" / "cre99_table1.json").read_text(encoding="utf-8"))


def _sha(r: IRBResult) -> str:
    return hashlib.sha256((repr(r) + "\n" + json.dumps(r.to_dict(), sort_keys=True)).encode()).hexdigest()[:24]


def test_without_the_option_the_result_is_the_1_1_0_one_byte_for_byte():
    assert len(BASELINE) == 1891
    checked = 0
    for case in BASELINE:
        call = dict(case["call"])
        ac, pd, lgd, regime = call.pop("asset_class"), call.pop("pd"), call.pop("lgd"), call.pop("regime")
        try:
            r = capital_requirement(ac, pd, lgd, regime=regime, **call)
        except ValueError as e:
            assert case.get("error") == str(e), case["call"]
            continue
        assert case["sha"] == _sha(r), case["call"]
        assert r.lgd_floor_applied is None and "lgd_floor_applied" not in r.to_dict()
        checked += 1
    assert checked > 1500


def _rw_pct(col_id: str, pd: float) -> float:
    i = [c["id"] for c in CRE99["columns"]].index(col_id)
    row = next(r for r in CRE99["rows"] if r["pd"] == pd)
    return row["rw_pct"][i]


PDS = (0.0025, 0.01, 0.05)


@pytest.mark.parametrize("pd", PDS)
def test_golden_qrre_below_the_floor_equals_the_printed_50_percent_column(pd):
    """A QRRE LGD of 30% is floored to 50%: its risk weight is the printed CRE99 QRRE column of LGD 50%."""
    r = capital_requirement("qrre", pd, 0.30, lgd_floor=LGDFloorInput(), regime="basel3")
    assert r.lgd == 0.50 and r.lgd_floor_applied.lgd_input == 0.30 and r.lgd_floor_applied.binding
    assert abs(100.0 * r.risk_weight - _rw_pct("qrre_50", pd)) <= 0.01 + 1e-9


@pytest.mark.parametrize("pd", PDS)
def test_golden_mortgage_at_2_percent_is_floored_to_5_percent(pd):
    """K is linear in the LGD: the mortgage risk weight at 5% is the printed LGD 45% column times 5/45."""
    r = capital_requirement("residential_mortgage", pd, 0.02, lgd_floor=LGDFloorInput(), regime="basel3")
    assert r.lgd == 0.05
    assert 100.0 * r.risk_weight == pytest.approx(_rw_pct("mortgage_45", pd) * 5.0 / 45.0, abs=0.01 * 5 / 45 + 1e-9)
    # whatever the collateral and the cover (CRE32.59)
    spec = LGDFloorInput(collateral="real-estate", e=100.0, e_s=100.0)
    assert capital_requirement("residential_mortgage", pd, 0.02, lgd_floor=spec).lgd == 0.05


@pytest.mark.parametrize("pd", PDS)
def test_golden_corporate_unsecured_below_25_percent(pd):
    """Corporate (sales 50, so no size adjustment) at LGD 10% floored to 25%: the printed LGD 40% column times 25/40."""
    r = capital_requirement("corporate", pd, 0.10, sales_eur_m=50.0, lgd_floor=LGDFloorInput())
    assert r.lgd == 0.25 and r.lgd_floor_applied.source == "CRE32.16"
    assert 100.0 * r.risk_weight == pytest.approx(_rw_pct("corp_s50", pd) * 25.0 / 40.0, abs=0.01 * 25 / 40 + 1e-9)


def test_golden_partly_secured_other_retail_uses_the_cre32_17_blend():
    spec = LGDFloorInput(collateral="real-estate", e=100.0, e_s=60.0)
    r = capital_requirement("other_retail", 0.02, 0.12, lgd_floor=spec)
    assert r.lgd == pytest.approx(0.30 * 0.4 + 0.10 * 0.6, abs=1e-15)
    assert r.lgd_floor_applied.floor == pytest.approx(0.18, abs=1e-15) and r.lgd_floor_applied.secured_share == 0.6
    plain = capital_requirement("other_retail", 0.02, 0.18)
    assert r.k == pytest.approx(plain.k, rel=1e-12) and r.risk_weight == pytest.approx(plain.risk_weight, rel=1e-12)
    assert "CRE32.17" in r.lgd_floor_applied.source
    # an LGD above the blend is kept
    assert capital_requirement("other_retail", 0.02, 0.25, lgd_floor=spec).lgd == 0.25


def test_golden_crr3_cells():
    spec = LGDFloorInput(portfolio="public-sector")
    pub = capital_requirement("corporate", 0.01, 0.02, regime="crr3", lgd_floor=spec)
    assert pub.lgd == 0.05 and pub.lgd_floor_applied.source == "CRR3 Article 161(5)" and pub.regime == "crr3"
    corp = capital_requirement("corporate", 0.01, 0.10, regime="crr3", lgd_floor=LGDFloorInput())
    assert corp.lgd == 0.25 and corp.lgd_floor_applied.source == "CRR3 Article 161(4)"
    q = capital_requirement("qrre", 0.01, 0.30, regime="crr3", lgd_floor=LGDFloorInput())
    assert q.lgd == 0.50 and q.lgd_floor_applied.source == "CRR3 Article 164(4)"
    m = capital_requirement("residential_mortgage", 0.01, 0.02, regime="crr3", lgd_floor=LGDFloorInput())
    assert m.lgd == 0.05 and m.lgd_floor_applied.source == "CRR3 Article 164(4a)(d)"


def test_floor_feeds_the_expected_loss_and_the_risk_weight_and_is_reported():
    r = capital_requirement("corporate", 0.02, 0.10, ead=1_000_000.0, lgd_floor=LGDFloorInput())
    info = r.lgd_floor_applied
    assert (info.lgd_input, info.floor, info.lgd_used, info.binding, info.applied) == (0.10, 0.25, 0.25, True, True)
    assert r.lgd == info.lgd_used and any("LGD floored at 25.00%" in n for n in r.notes)
    plain = capital_requirement("corporate", 0.02, 0.25, ead=1_000_000.0)
    assert r.k == plain.k and r.rwa == plain.rwa
    assert "lgd_floor" in r.references and "CRE32.16" in r.references["lgd_floor"]
    d = r.to_dict()
    assert d["lgd_floor_applied"]["lgd_input"] == 0.10
    assert json.loads(json.dumps(d))["lgd_floor_applied"] == d["lgd_floor_applied"]
    assert r.k > capital_requirement("corporate", 0.02, 0.10).k


def test_exactly_at_the_floor_and_above_every_floor_change_nothing():
    for ac, lgd, spec in (("corporate", 0.25, LGDFloorInput()), ("qrre", 0.50, LGDFloorInput()),
                          ("other_retail", 0.30, LGDFloorInput()),
                          ("residential_mortgage", 0.05, LGDFloorInput()), ("corporate", 0.90, LGDFloorInput())):
        base = capital_requirement(ac, 0.02, lgd)
        r = capital_requirement(ac, 0.02, lgd, lgd_floor=spec)
        assert (r.k, r.risk_weight, r.lgd, r.notes) == (base.k, base.risk_weight, base.lgd, base.notes)
        assert r.lgd_floor_applied.binding is False and r.lgd_floor_applied.applied is True
    # a floor of 0 (corporate fully secured by financial collateral) binds nowhere
    zero = capital_requirement("corporate", 0.02, 0.0, lgd_floor=LGDFloorInput(collateral="financial", e=1.0, e_s=2.0))
    assert zero.lgd == 0.0 and zero.lgd_floor_applied.floor == 0.0 and not zero.lgd_floor_applied.binding


def test_a_defaulted_exposure_and_a_sovereign_get_no_floor_and_say_why():
    d = capital_requirement("corporate", 1.0, 0.01, defaulted=True, el_best_estimate=0.0, lgd_floor=LGDFloorInput())
    plain = capital_requirement("corporate", 1.0, 0.01, defaulted=True, el_best_estimate=0.0)
    assert (d.k, d.risk_weight, d.lgd) == (plain.k, plain.risk_weight, 0.01)
    assert d.lgd_floor_applied.applied is False and d.lgd_floor_applied.floor is None
    assert "CRE31.7" in d.lgd_floor_applied.source
    s = capital_requirement("sovereign", 0.02, 0.05, lgd_floor=LGDFloorInput())
    sp = capital_requirement("sovereign", 0.02, 0.05)
    assert (s.k, s.lgd) == (sp.k, 0.05) and s.lgd_floor_applied.applied is False
    assert "CRE32.16" in s.lgd_floor_applied.source


def test_the_option_is_refused_where_there_is_no_floor():
    for regime in ("basel2", "cl_ran21_6"):
        with pytest.raises(ValueError, match="regime"):
            capital_requirement("corporate", 0.02, 0.2, regime=regime, lgd_floor=LGDFloorInput())
    with pytest.raises(ValueError, match="UNVERIFIED"):
        capital_requirement("bank", 0.02, 0.2, lgd_floor=LGDFloorInput())
    with pytest.raises(TypeError):
        capital_requirement("corporate", 0.02, 0.2, lgd_floor=True)  # type: ignore[arg-type]


def test_k_is_monotone_in_the_floor():
    ks = [capital_requirement("corporate", 0.02, 0.0, lgd_floor=LGDFloorInput(collateral=c, e=100.0, e_s=s)).k
          for c, s in (("financial", 200.0), ("receivables", 200.0), ("other-physical", 200.0), ("unsecured", 0.0))]
    assert ks == sorted(ks) and ks[0] == 0.0 and ks[-1] > 0.0
