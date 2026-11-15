"""BL-053: the CRE32.17 partly secured floor, the CRR3 floor cells (Articles 161(4), 161(5), 164(4)) and the resolution
of the floor of one exposure. Every expected value is typed from the paragraph text read on 2026-11-11, not from the
module."""

from __future__ import annotations

import dataclasses

import pytest

from riskvalidation.lgd_ead import floors as f


def test_partly_secured_floor_hand_cases():
    # CRE32.17: Floor = U E_U / (E (1 + H_E)) + S E_S / (E (1 + H_E)), E_U = E (1 + H_E) - E_S, E_S capped at E (1 +
    # H_E)
    assert f.partly_secured_lgd_floor(0.30, 0.10, e=100.0, e_s=60.0) == pytest.approx(0.30 * 0.4 + 0.10 * 0.6)
    # a haircut on the exposure: E (1 + H_E) = 110, E_S = 44, E_U = 66
    assert f.partly_secured_lgd_floor(0.30, 0.10, e=100.0, e_s=44.0, h_e=0.10) == pytest.approx(0.22, abs=1e-15)
    # E_S = 0: the unsecured floor; E_S exactly E (1 + H_E) and above it: the secured floor (E_S is capped)
    assert f.partly_secured_lgd_floor(0.25, 0.15, e=100.0, e_s=0.0) == 0.25
    assert f.partly_secured_lgd_floor(0.25, 0.15, e=100.0, e_s=110.0, h_e=0.10) == pytest.approx(0.15, abs=1e-15)
    assert f.partly_secured_lgd_floor(0.25, 0.15, e=100.0, e_s=500.0, h_e=0.10) == pytest.approx(0.15, abs=1e-15)
    assert f.partly_secured_lgd_floor(0.25, 0.15, e=100.0, e_s=109.9999, h_e=0.10) > 0.15
    # the weighted average lies between the two floors and moves monotonically from one to the other
    prev = 0.25
    for es in (0.0, 10.0, 30.0, 60.0, 90.0, 100.0):
        v = f.partly_secured_lgd_floor(0.25, 0.10, e=100.0, e_s=es)
        assert 0.10 - 1e-15 <= v <= 0.25 + 1e-15 and v <= prev + 1e-15
        prev = v


def test_partly_secured_floor_refuses_bad_inputs():
    for kw in (dict(e=0.0, e_s=0.0), dict(e=-1.0, e_s=0.0), dict(e=100.0, e_s=-1.0), dict(e=100.0, e_s=1.0, h_e=-0.1),
               dict(e=float("nan"), e_s=1.0), dict(e=100.0, e_s=float("inf"))):
        with pytest.raises(ValueError):
            f.partly_secured_lgd_floor(0.25, 0.10, **kw)
    for u, s in ((-0.1, 0.1), (0.25, 1.5), (float("nan"), 0.1)):
        with pytest.raises(ValueError, match="floor"):
            f.partly_secured_lgd_floor(u, s, e=1.0, e_s=0.5)


def test_crr3_cells_equal_the_regulation():
    # Article 161(4) Table 1, Article 161(5), Article 164(4) Table 1 (Regulation (EU) 2024/1623)
    want = {
        ("corporate", "unsecured"): 0.25, ("corporate", "financial"): 0.00, ("corporate", "receivables"): 0.10,
        ("corporate", "real-estate"): 0.10, ("corporate", "other-physical"): 0.15,
        ("public-sector", "unsecured"): 0.05,
        ("retail", "mortgage"): 0.05, ("retail", "qrre"): 0.50, ("retail", "unsecured"): 0.30,
        ("retail", "financial"): 0.00, ("retail", "receivables"): 0.10, ("retail", "real-estate"): 0.10,
        ("retail", "other-physical"): 0.15,
    }
    assert dict(f.LGD_FLOOR_CELLS_CRR3) == want
    for (p, c), v in want.items():
        assert f.lgd_floor(p, c, regime="crr3") == pytest.approx(v, abs=1e-15)
    # the Basel table has no public-sector row, and the call without a regime is the Basel one
    with pytest.raises(ValueError, match="cell"):
        f.lgd_floor("public-sector", "unsecured")
    with pytest.raises(ValueError, match="cell"):
        f.lgd_floor("public-sector", "financial", regime="crr3")
    assert f.lgd_floor("corporate", "unsecured") == 0.25
    with pytest.raises(ValueError, match="regime"):
        f.lgd_floor("corporate", "unsecured", regime="basel2")


def test_every_cell_names_its_paragraph():
    for regime, cells in (("basel3", f.LGD_FLOOR_CELLS), ("crr3", f.LGD_FLOOR_CELLS_CRR3)):
        assert set(f.LGD_FLOOR_SOURCES[regime]) == set(cells)
    assert f.LGD_FLOOR_SOURCES["basel3"][("corporate", "unsecured")] == "CRE32.16"
    assert f.LGD_FLOOR_SOURCES["basel3"][("retail", "qrre")] == "CRE32.58"
    assert f.LGD_FLOOR_SOURCES["basel3"][("retail", "mortgage")] == "CRE32.59"
    assert f.LGD_FLOOR_SOURCES["crr3"][("corporate", "real-estate")] == "CRR3 Article 161(4)"
    assert f.LGD_FLOOR_SOURCES["crr3"][("public-sector", "unsecured")] == "CRR3 Article 161(5)"
    assert f.LGD_FLOOR_SOURCES["crr3"][("retail", "qrre")] == "CRR3 Article 164(4)"
    assert f.LGD_FLOOR_SOURCES["crr3"][("retail", "mortgage")] == "CRR3 Article 164(4a)(d)"


def test_input_record_is_frozen_and_validated():
    spec = f.LGDFloorInput()
    assert (spec.collateral, spec.e, spec.e_s, spec.h_e, spec.portfolio) == ("unsecured", 1.0, 0.0, 0.0, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.e = 2.0  # type: ignore[misc]
    for kw in (dict(collateral="gold"), dict(e=0.0), dict(e_s=-1.0), dict(h_e=-0.1), dict(portfolio="sovereign")):
        with pytest.raises(ValueError):
            f.LGDFloorInput(**kw)


def test_resolve_corporate_unsecured_and_binding_edges():
    r = f.resolve_lgd_floor("corporate", 0.10, "basel3", f.LGDFloorInput())
    assert (r.applied, r.floor, r.lgd_input, r.lgd_used, r.binding) == (True, 0.25, 0.10, 0.25, True)
    assert r.source == "CRE32.16" and r.portfolio == "corporate" and r.regime == "basel3"
    at = f.resolve_lgd_floor("corporate", 0.25, "basel3", f.LGDFloorInput())  # exactly at the floor: kept, not binding
    assert at.lgd_used == 0.25 and at.binding is False and at.applied is True
    above = f.resolve_lgd_floor("corporate", 0.45, "basel3", f.LGDFloorInput())
    assert above.lgd_used == 0.45 and above.binding is False
    # every class that maps to the corporate row
    for ac in ("sme", "hvcre"):
        assert f.resolve_lgd_floor(ac, 0.0, "crr3", f.LGDFloorInput()).source == "CRR3 Article 161(4)"


def test_resolve_retail_cells():
    q = f.resolve_lgd_floor("qrre", 0.30, "basel3", f.LGDFloorInput())
    assert (q.floor, q.lgd_used, q.source) == (0.50, 0.50, "CRE32.58")
    m = f.resolve_lgd_floor("residential_mortgage", 0.02, "basel3", f.LGDFloorInput())
    assert (m.floor, m.lgd_used, m.source) == (0.05, 0.05, "CRE32.59")
    # CRE32.59 and Article 164(4a)(d): 5% whatever the collateral and whatever the cover
    for spec in (f.LGDFloorInput(collateral="real-estate", e=100.0, e_s=40.0),
                 f.LGDFloorInput(collateral="financial", e=100.0, e_s=0.0), f.LGDFloorInput(collateral="mortgage")):
        for regime in ("basel3", "crr3"):
            assert f.resolve_lgd_floor("residential_mortgage", 0.0, regime, spec).floor == 0.05
    o = f.resolve_lgd_floor("other_retail", 0.12, "basel3",
                            f.LGDFloorInput(collateral="real-estate", e=100.0, e_s=60.0))
    assert o.floor == pytest.approx(0.18, abs=1e-15) and o.lgd_used == pytest.approx(0.18, abs=1e-15)
    assert o.secured_share == pytest.approx(0.6) and o.source == "CRE32.17 and CRE32.59"
    u = f.resolve_lgd_floor("other_retail", 0.0, "basel3", f.LGDFloorInput())
    assert (u.floor, u.source) == (0.30, "CRE32.58")
    # a partly secured corporate exposure takes the same blend (CRE32.17, Article 161(6))
    c = f.resolve_lgd_floor("corporate", 0.0, "crr3", f.LGDFloorInput(collateral="receivables", e=100.0, e_s=50.0))
    assert c.floor == pytest.approx(0.175, abs=1e-15) and c.source == "CRR3 Article 161(4) and 161(6)"


def test_resolve_partly_secured_edges():
    spec0 = f.LGDFloorInput(collateral="real-estate", e=100.0, e_s=0.0, h_e=0.05)
    assert f.resolve_lgd_floor("other_retail", 0.0, "basel3", spec0).floor == pytest.approx(0.30)
    full = f.LGDFloorInput(collateral="real-estate", e=100.0, e_s=105.0, h_e=0.05)
    assert f.resolve_lgd_floor("other_retail", 0.0, "basel3", full).floor == pytest.approx(0.10)
    over = f.LGDFloorInput(collateral="real-estate", e=100.0, e_s=1000.0, h_e=0.05)
    r = f.resolve_lgd_floor("other_retail", 0.0, "basel3", over)
    assert r.floor == pytest.approx(0.10) and r.secured_share == 1.0


def test_resolve_crr3_public_sector():
    spec = f.LGDFloorInput(portfolio="public-sector")
    r = f.resolve_lgd_floor("corporate", 0.02, "crr3", spec)
    assert (r.floor, r.lgd_used, r.source, r.portfolio) == (0.05, 0.05, "CRR3 Article 161(5)", "public-sector")
    assert f.resolve_lgd_floor("corporate", 0.30, "crr3", spec).binding is False
    with pytest.raises(ValueError, match="public-sector"):
        f.resolve_lgd_floor("corporate", 0.02, "basel3", spec)
    with pytest.raises(ValueError, match="cell"):
        f.resolve_lgd_floor("corporate", 0.02, "crr3", f.LGDFloorInput(portfolio="public-sector",
                                                                       collateral="financial", e=2.0, e_s=1.0))
    with pytest.raises(ValueError, match="retail"):
        f.resolve_lgd_floor("qrre", 0.02, "crr3", spec)


def test_resolve_exemptions_name_their_source():
    s = f.resolve_lgd_floor("sovereign", 0.05, "basel3", f.LGDFloorInput())
    assert (s.applied, s.floor, s.lgd_used, s.binding) == (False, None, 0.05, False)
    assert "CRE32.16" in s.source and "sovereign" in s.reason
    d = f.resolve_lgd_floor("corporate", 0.01, "basel3", f.LGDFloorInput(), defaulted=True)
    assert (d.applied, d.floor, d.lgd_used) == (False, None, 0.01) and "CRE31.7" in d.source and "defaulted" in d.reason
    d3 = f.resolve_lgd_floor("other_retail", 0.01, "crr3", f.LGDFloorInput(), defaulted=True)
    assert "Article 158(5)" in d3.source


def test_resolve_refuses_what_it_cannot_place():
    with pytest.raises(ValueError, match="UNVERIFIED"):
        f.resolve_lgd_floor("bank", 0.1, "basel3", f.LGDFloorInput())
    with pytest.raises(ValueError, match="UNVERIFIED"):
        f.resolve_lgd_floor("financial_institution", 0.1, "crr3", f.LGDFloorInput())
    for regime in ("basel2", "cl_ran21_6", "nope"):
        with pytest.raises(ValueError, match="regime"):
            f.resolve_lgd_floor("corporate", 0.1, regime, f.LGDFloorInput())
    with pytest.raises(ValueError, match="cell"):  # QRRE has no secured cell
        f.resolve_lgd_floor("qrre", 0.1, "basel3", f.LGDFloorInput(collateral="financial", e=2.0, e_s=1.0))
    with pytest.raises(ValueError, match="e_s"):  # an unsecured exposure with collateral value is contradictory
        f.resolve_lgd_floor("corporate", 0.1, "basel3", f.LGDFloorInput(collateral="unsecured", e=2.0, e_s=1.0))
    with pytest.raises(ValueError, match="lgd"):
        f.resolve_lgd_floor("corporate", 1.5, "basel3", f.LGDFloorInput())
    with pytest.raises(ValueError, match="residential_mortgage"):
        f.resolve_lgd_floor("corporate", 0.1, "basel3", f.LGDFloorInput(collateral="mortgage"))
