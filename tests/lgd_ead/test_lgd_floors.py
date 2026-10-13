"""RV-616 and RV-617: the IRB LGD and EAD floors and the supervisory CCFs of dossier 01 section A.5 (contract
`engine-lgd-ead-api`, module floors). The expected cells below are typed from the table of dossier 01, not read from the
module."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from riskvalidation.lgd_ead import floors as f


def a(*v):
    return np.array(v, dtype=float)


# dossier 01 A.5 "A-IRB corporate LGD floors" (CRE32.16) and "Retail LGD floors" (CRE32.58 to CRE32.59)
DOSSIER_01 = {
    ("corporate", "unsecured"): 0.25, ("corporate", "financial"): 0.00, ("corporate", "receivables"): 0.10,
    ("corporate", "real-estate"): 0.10, ("corporate", "other-physical"): 0.15,
    ("retail", "mortgage"): 0.05, ("retail", "qrre"): 0.50, ("retail", "unsecured"): 0.30,
    ("retail", "financial"): 0.00, ("retail", "receivables"): 0.10, ("retail", "real-estate"): 0.10,
    ("retail", "other-physical"): 0.15,
}
PORTFOLIOS = ("corporate", "retail")
COLLATERALS = ("unsecured", "financial", "receivables", "real-estate", "other-physical", "qrre", "mortgage")


def test_lgd_floor_cells_equal_dossier_01():
    for (p, col), want in DOSSIER_01.items():
        assert f.lgd_floor(p, col) == pytest.approx(want, abs=1e-15), (p, col)
    # every other combination has no cell and raises (a corporate qrre, a corporate mortgage)
    for p in PORTFOLIOS:
        for col in COLLATERALS:
            if (p, col) not in DOSSIER_01:
                with pytest.raises(ValueError, match="cell"):
                    f.lgd_floor(p, col)
    with pytest.raises(ValueError, match="portfolio"):
        f.lgd_floor("sovereign", "unsecured")
    with pytest.raises(ValueError, match="collateral"):
        f.lgd_floor("corporate", "gold")
    assert len(f.LGD_FLOOR_CELLS) == 12


def test_apply_lgd_floor_golden():
    r = f.apply_lgd_floor(a(0.20, 0.30, 0.60), 0.25)
    assert isinstance(r, f.FloorBite) and dataclasses.is_dataclass(r)
    np.testing.assert_allclose(r.applied, [0.25, 0.30, 0.60])
    assert r.binding.tolist() == [True, False, False]
    assert r.share_binding == pytest.approx(1 / 3)
    assert r.mean_before == pytest.approx(0.366667, abs=5e-7)
    assert r.mean_after == pytest.approx(0.383333, abs=5e-7)
    assert r.uplift == pytest.approx(0.016667, abs=5e-7)
    assert r.uplift == r.mean_after - r.mean_before and r.uplift >= 0
    zero = f.apply_lgd_floor(a(0.0, 0.3), 0.0)  # a floor of 0 binds nowhere
    assert not zero.binding.any() and zero.uplift == 0.0
    x = a(0.1, 0.9, 0.25)
    out = f.apply_lgd_floor(x, 0.25)
    assert np.all(out.applied >= x) and out.binding.tolist() == (out.applied > x).tolist()
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="floor"):
            f.apply_lgd_floor(a(0.2), bad)
    with pytest.raises(ValueError):
        f.apply_lgd_floor(a(), 0.25)
    with pytest.raises(ValueError, match="non-finite"):
        f.apply_lgd_floor(a(np.nan), 0.25)


def test_supervisory_ccf():
    # dossier 01 A.5 and B.2: CRE32.33 equal to the standardised CCFs of CRE20.94 to CRE20.101
    want = {"direct-credit-substitutes": 1.0, "nifs-rufs-transaction-related-contingents": 0.5, "commitments": 0.4,
            "short-term-self-liquidating-trade-letters-of-credit": 0.2, "unconditionally-cancellable": 0.1}
    for item, v in want.items():
        assert f.supervisory_ccf(item) == pytest.approx(v, abs=1e-15), item
    assert f.supervisory_ccf("commitments") != 0.75  # Basel II is not implemented
    with pytest.raises(ValueError, match="item"):
        f.supervisory_ccf("guarantee")


def test_ead_floor_golden_and_bounds():
    assert f.ead_floor(a(600), a(1000), a(0.40))[0] == pytest.approx(680.0, abs=1e-12)
    assert f.ead_floor(a(600), a(1000), a(0.40), off_balance_share=1.0)[0] == pytest.approx(760.0)
    rng = np.random.default_rng(1)
    drawn = rng.uniform(0, 500, 50)
    limit = drawn + rng.uniform(0, 500, 50)
    ccf = rng.uniform(0, 1, 50)
    fl = f.ead_floor(drawn, limit, ccf)
    assert np.all(fl >= drawn - 1e-12) and np.all(fl <= limit + 1e-12)
    assert np.all(f.ead_floor(drawn, limit, np.minimum(ccf + 0.1, 1.0)) >= fl - 1e-12)
    assert np.all(f.ead_floor(drawn, limit + 10, ccf) >= fl)
    with pytest.raises(ValueError, match="sa_ccf"):
        f.ead_floor(a(1), a(2), a(1.2))
    # an overdrawn line (drawn above limit) has no off-balance amount: its floor is the drawn amount, not an error
    over = f.ead_floor(a(1200, 600), a(1000, 1000), a(0.4, 0.4))
    np.testing.assert_allclose(over, [1200.0, 680.0])
    assert np.all(f.ead_floor(a(1200, 100), a(1000, 1000), a(1.0, 1.0)) <= np.maximum(a(1200, 100), a(1000, 1000)))
    with pytest.raises(ValueError, match="off_balance_share"):
        f.ead_floor(a(1), a(2), a(0.4), off_balance_share=1.5)


def test_apply_ead_floor_golden():
    r = f.apply_ead_floor(a(650, 700), a(680, 680))
    np.testing.assert_allclose(r.applied, [680, 700])
    assert r.binding.tolist() == [True, False]
    assert r.share_binding == 0.5 and r.mean_before == 675 and r.mean_after == 690 and r.uplift == 15
    with pytest.raises(ValueError, match="same"):
        f.apply_ead_floor(a(1, 2), a(1))
