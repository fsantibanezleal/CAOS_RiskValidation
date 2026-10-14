"""RV-1007 to RV-1009: the six SRP31 shock shapes, the size tables, the CMF Hoja 9 table, the floors (contract
`engine-irrbb-api`, module shocks). The tables below are the rows of dossier 01 section H.3 and dossier 03 section
F.2 as printed there; the module data is compared with them cell by cell."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from riskvalidation.irrbb import shocks as s

MIDPOINTS = np.array([0.0028, 0.0417, 0.1667, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
                      12.5, 17.5, 25.0])

CURRENCIES = ("ARS", "AUD", "BRL", "CAD", "CHF", "CNY", "EUR", "GBP", "HKD", "IDR", "INR", "JPY", "KRW", "MXN", "RUB",
              "SAR", "SEK", "SGD", "TRY", "USD", "ZAR")
# dossier 01 H.3: the rows Parallel, Short and Long of the two tables as printed there, one entry per currency above
T2024 = (
    (400, 350, 400, 200, 175, 225, 225, 275, 225, 400, 325, 100, 225, 400, 400, 275, 275, 175, 400, 200, 325),
    (500, 425, 500, 275, 250, 300, 350, 425, 375, 500, 475, 100, 350, 500, 500, 375, 425, 250, 500, 300, 500),
    (300, 300, 300, 175, 200, 150, 200, 250, 200, 300, 225, 100, 225, 200, 300, 250, 200, 225, 300, 225, 300),
)
T2016 = (
    (400, 300, 400, 200, 100, 250, 200, 250, 200, 400, 400, 100, 300, 400, 400, 200, 200, 150, 400, 200, 400),
    (500, 450, 500, 300, 150, 300, 250, 300, 250, 500, 500, 100, 400, 500, 500, 300, 300, 200, 500, 300, 500),
    (300, 200, 300, 150, 100, 150, 100, 150, 100, 350, 300, 100, 200, 300, 300, 150, 150, 100, 300, 150, 300),
)
# dossier 03 section F.2, Hoja 9 (RAN 21-13 Annex 1), as printed
HOJA9 = {
    "canasta-1": (200, 300, 150),
    "canasta-2": (400, 500, 300),
    "CLP": (300, 400, 200),
    "UR-nii": (300, None, None),
    "UR-eve": (200, 250, 150),
}


def parse(rows):
    return {cur: (rows[0][k], rows[1][k], rows[2][k]) for k, cur in enumerate(CURRENCIES)}


def alphas(t, x=4.0):
    return math.exp(-t / x), 1 - math.exp(-t / x)


# ---------------------------------------------------------------------------------------------------- RV-1007


def test_shock_scalars_srp31_92():
    a_s, a_l = s.shock_scalars(np.array([3.5]), 4.0)
    assert a_s[0] == pytest.approx(0.417, abs=5e-4) and a_s[0] == pytest.approx(math.exp(-3.5 / 4), abs=1e-15)
    assert a_s[0] == pytest.approx(0.416862, abs=5e-7) and a_l[0] == pytest.approx(0.583138, abs=5e-7)
    a_s, a_l = s.shock_scalars(MIDPOINTS)
    assert np.allclose(a_s + a_l, 1.0, atol=1e-15)
    assert np.all(np.diff(a_s) < 0) and np.all((a_s > 0) & (a_s <= 1)) and np.all((a_l > 0) & (a_l <= 1))
    a_s8, _ = s.shock_scalars(MIDPOINTS, x=8.0)
    assert np.all(a_s8 > a_s)
    with pytest.raises(ValueError):
        s.shock_scalars(MIDPOINTS, x=0.0)


def test_shock_curve_goldens_and_invariants():
    t = np.array([3.5])
    up = s.shock_curve("short-up", 100, 100, 100, t)[0]
    ste = s.shock_curve("steepener", 100, 100, 100, t)[0]
    fla = s.shock_curve("flattener", 100, 100, 100, t)[0]
    assert (up, ste, fla) == pytest.approx((41.7, 25.4, -1.6), abs=0.05)
    a_s, a_l = alphas(3.5)
    assert up == pytest.approx(100 * a_s, abs=1e-12)
    assert ste == pytest.approx(-0.65 * 100 * a_s + 0.9 * 100 * a_l, abs=1e-12)
    assert fla == pytest.approx(0.8 * 100 * a_s - 0.6 * 100 * a_l, abs=1e-12)
    assert (up, ste, fla) == pytest.approx((41.686, 25.386, -1.639), abs=5e-4)
    # USD 2024: parallel-up is 200 at every one of the 19 midpoints
    par, sh, lo = s.shock_table("2024")["USD"]
    assert np.all(s.shock_curve("parallel-up", par, sh, lo, MIDPOINTS) == 200.0)
    assert np.all(s.shock_curve("parallel-down", par, sh, lo, MIDPOINTS) == -200.0)
    assert np.array_equal(s.shock_curve("short-up", par, sh, lo, MIDPOINTS),
                          -s.shock_curve("short-down", par, sh, lo, MIDPOINTS))
    # the long size enters only inside the two rotations
    for name in ("parallel-up", "parallel-down", "short-up", "short-down"):
        assert np.array_equal(s.shock_curve(name, par, sh, 100, MIDPOINTS),
                              s.shock_curve(name, par, sh, 900, MIDPOINTS))
    for name in ("steepener", "flattener"):
        assert not np.array_equal(s.shock_curve(name, par, sh, 100, MIDPOINTS),
                                  s.shock_curve(name, par, sh, 900, MIDPOINTS))
    # the shapes at all midpoints against the formulas, in the order of SHOCKS
    a_s, a_l = np.exp(-MIDPOINTS / 4), 1 - np.exp(-MIDPOINTS / 4)
    expect = -0.65 * sh * a_s + 0.9 * lo * a_l
    assert np.allclose(s.shock_curve("steepener", par, sh, lo, MIDPOINTS), expect, atol=1e-12)
    assert s.SHOCKS == ("parallel-up", "parallel-down", "steepener", "flattener", "short-up", "short-down")
    with pytest.raises(ValueError):
        s.shock_curve("twist", 100, 100, 100, MIDPOINTS)
    with pytest.raises(ValueError):
        s.shock_curve("steepener", -1, 100, 100, MIDPOINTS)
    with pytest.raises(ValueError):
        s.shock_curve("steepener", 100, 100, float("nan"), MIDPOINTS)


# ---------------------------------------------------------------------------------------------------- RV-1008


def test_shock_tables_cells_and_order():
    t24, t16 = parse(T2024), parse(T2016)
    got24, got16 = s.shock_table("2024"), s.shock_table("2016")
    assert list(got24) == list(t24) == list(got16) and len(got24) == 21 and "CLP" not in got24
    for cur in t24:
        assert tuple(got24[cur]) == t24[cur], cur
        assert tuple(got16[cur]) == t16[cur], cur
    assert got24["USD"] == (200, 300, 225) and got16["USD"] == (200, 300, 150) and got16["EUR"] == (200, 250, 100)
    cmf = s.shock_table("cmf-annex1")
    assert list(cmf) == list(HOJA9) and all(tuple(cmf[k]) == HOJA9[k] for k in HOJA9)
    assert cmf["UR-nii"] == (300, None, None)
    with pytest.raises(ValueError):
        s.shock_table("2030")
    # the returned mapping is a copy: editing it does not change the data
    got24["USD"] = (1, 1, 1)
    assert s.shock_table("2024")["USD"] == (200, 300, 225)


def test_shock_size_origin_and_refusals():
    usd = s.shock_size("USD", "2024")
    assert (usd.currency, usd.parallel_bp, usd.short_bp, usd.long_bp) == ("USD", 200, 300, 225)
    assert usd.origin == "srp31-table-2024" and usd.label is None and usd.method is None
    assert s.shock_size("USD", "2016").origin == "srp31-table-2016"
    assert s.shock_size("USD", "2016").long_bp == 150
    clp = s.shock_size("CLP", "cmf-annex1")
    assert (clp.parallel_bp, clp.short_bp, clp.long_bp, clp.origin) == (300, 400, 200, "cmf-annex1")
    clf = s.shock_size("CLF", "cmf-annex1")
    assert (clf.parallel_bp, clf.short_bp, clf.long_bp, clf.origin) == (200, 250, 150, "cmf-annex1")
    assert dataclasses.is_dataclass(usd)
    with pytest.raises(dataclasses.FrozenInstanceError):
        usd.parallel_bp = 1  # type: ignore[misc]
    # a CLP size is never invented
    for cal in ("2024", "2016"):
        with pytest.raises(ValueError, match="CLP"):
            s.shock_size("CLP", cal)
    with pytest.raises(ValueError):
        s.shock_size("EUR", "cmf-annex1")
    with pytest.raises(ValueError):
        s.shock_size("CLF", "2024")
    # the Annex 1 NII row has no short and long size: it cannot become a three-size ShockSize
    with pytest.raises(ValueError):
        s.shock_size("UR-nii", "cmf-annex1")
    label = {"en": "Contraste estimate on BCCh yields", "es": "Estimacion de Contraste con rendimientos del BCCh"}
    est = s.shock_size("CLP", "2024", estimate=(275.0, 350.0, 200.0, label, "SRP98.56-63 on BCCh F022 yields"))
    assert est.origin == "contraste-estimate" and est.label == label and est.method.startswith("SRP98")
    assert (est.parallel_bp, est.short_bp, est.long_bp) == (275.0, 350.0, 200.0)
    for bad_label in ({}, {"en": "x"}, {"en": "", "es": "y"}, None):
        with pytest.raises(ValueError):
            s.shock_size("CLP", "2024", estimate=(275.0, 350.0, 200.0, bad_label, "m"))
    for bad_method in ("", "  ", None):
        with pytest.raises(ValueError):
            s.shock_size("CLP", "2024", estimate=(275.0, 350.0, 200.0, label, bad_method))
    with pytest.raises(ValueError):
        s.shock_size("CLP", "2024", estimate=(-1.0, 350.0, 200.0, label, "m"))
    # an estimate for a currency that has a tabled size is refused, never silently preferred
    with pytest.raises(ValueError):
        s.shock_size("USD", "2024", estimate=(275.0, 350.0, 200.0, label, "m"))


# ---------------------------------------------------------------------------------------------------- RV-1009


def test_apply_floor_goldens_and_invariants():
    cases = [(0.004, -300, 0.0, 0.0), (-0.001, -300, 0.0, -0.001), (0.03, -300, 0.0, 0.0), (0.004, -300, -0.02, -0.02)]
    for r0, shock, floor, expect in cases:
        assert float(s.apply_floor(np.array([r0]), np.array([shock]), floor)[0]) == pytest.approx(expect, abs=1e-15)
    r0 = np.linspace(-0.01, 0.05, 25)
    for shock in (-400.0, -50.0, 0.0, 120.0):
        out = s.apply_floor(r0, np.full(25, shock), 0.0)
        assert np.all(out >= np.minimum(0.0, r0) - 1e-15)
        # a floor never lifts a rate above its pre-shock value
        assert np.all(out <= np.maximum(r0 + shock / 1e4, np.minimum(0.0, r0)) + 1e-15)
    assert np.allclose(s.apply_floor(r0, np.full(25, 75.0), None), r0 + 0.0075, atol=1e-15)
    # a rate already below the floor is left at its pre-shock value by a downward shock
    assert float(s.apply_floor(np.array([-0.002]), np.array([-100.0]), 0.0)[0]) == pytest.approx(-0.002, abs=1e-15)
    # an array floor (a maturity-dependent floor) is applied bucket by bucket
    fl = s.eu_floor(MIDPOINTS)
    base = np.full(19, 0.0)
    out = s.apply_floor(base, np.full(19, -500.0), fl)
    assert np.allclose(out, np.minimum(fl, 0.0), atol=1e-15)
    with pytest.raises(ValueError):
        s.apply_floor(np.zeros(3), np.zeros(2), 0.0)
    with pytest.raises(ValueError):
        s.apply_floor(np.zeros(3), np.zeros(3), np.zeros(2))


def test_eu_floor():
    out = s.eu_floor(np.array([0.0, 10.0, 50.0, 60.0]))
    assert out == pytest.approx([-0.015, -0.012, 0.0, 0.0], abs=1e-15)
    m = np.linspace(0, 80, 161)
    f = s.eu_floor(m)
    assert np.all(np.diff(f) >= -1e-18) and np.all(f[m >= 50] == 0.0) and np.all(f <= 0.0)
    assert np.allclose(f[m <= 50], -0.015 + 0.0003 * m[m <= 50], atol=1e-15)
    with pytest.raises(ValueError):
        s.eu_floor(np.array([-1.0]))
