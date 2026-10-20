"""RV-1305 to RV-1307: the scenario path handling (contract `engine-stress-api`, module stress.paths)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from riskvalidation.stress import paths as p

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "c06_rd_2020_06_macro_paths.json"


def arr(*x):
    return np.array(x, dtype=float)


def test_transpose_path_golden_and_invariants():
    out = p.transpose_path(arr(6, 8, 10), 5.0, 3.0, 3)
    assert out == pytest.approx([4.0, 6.0, 8.0], abs=1e-15)
    # the first difference is that of the source, periods aligned by their distance from the jump-off
    levels = np.array([5.5, 5.9, 7.2, 6.1, 4.4])
    out = p.transpose_path(levels, 5.0, -2.0, 4)
    assert np.allclose(np.diff(out), np.diff(levels[:4]), atol=1e-15) and out.size == 4
    assert out == pytest.approx([-1.5, -1.1, 0.2, -0.9], abs=1e-12)
    assert p.transpose_path(levels, 5.0, 5.0, 5) == pytest.approx(levels, abs=1e-15)
    with pytest.raises(ValueError, match="n"):
        p.transpose_path(levels, 5.0, 3.0, 6)
    with pytest.raises(ValueError, match="n"):
        p.transpose_path(levels, 5.0, 3.0, 0)
    before = levels.copy()
    p.transpose_path(levels, 5.0, 3.0, 3)
    assert np.array_equal(levels, before)


def test_revert_to_baseline_golden_and_invariants():
    out = p.revert_to_baseline(arr(8.0), 5.0, 14, 12)
    # k - len(path) = 1, 6, 12, 13 are the indices len + j - 1 of a path of one value
    assert [out[1 + j - 1] for j in (1, 6, 12)] == pytest.approx([7.75, 6.5, 5.0], abs=1e-12)
    assert out[13] == 5.0 and out[0] == 8.0 and out.size == 14
    assert np.all(out[12:] == 5.0) and np.all(np.diff(out) <= 0.0)
    # the published values are unchanged and a longer published path reverts from its last value
    path = arr(1.0, 2.0, 4.0)
    out = p.revert_to_baseline(path, 1.0, 9, 6)
    assert np.array_equal(out[:3], path)
    for k in range(4, 10):
        w = min(1.0, (k - 3) / 6)
        assert out[k - 1] == pytest.approx(4.0 + w * (1.0 - 4.0), abs=1e-12)
    assert out[8] == 1.0
    assert np.array_equal(p.revert_to_baseline(path, 1.0, 3, 6), path)
    with pytest.raises(ValueError, match="reversion_quarters"):
        p.revert_to_baseline(path, 1.0, 9, 0)
    with pytest.raises(ValueError, match="n"):
        p.revert_to_baseline(path, 1.0, 2, 6)


def test_annual_to_quarterly_golden_and_invariants():
    out = p.annual_to_quarterly(arr(6, 8), 4.0)
    assert out == pytest.approx([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0], abs=1e-15)
    rng = np.random.default_rng(1)
    years = rng.normal(size=5)
    jump = 0.3
    q = p.annual_to_quarterly(years, jump)
    assert q.size == 20 and np.array_equal(q[3::4], years)
    # piecewise linear: constant second difference inside each year
    for y in range(5):
        seg = np.concatenate([[jump if y == 0 else years[y - 1]], q[4 * y : 4 * y + 4]])
        assert np.allclose(np.diff(seg, 2), 0.0, atol=1e-12)


def test_level_from_deviation_goldens():
    assert float(p.level_from_deviation(arr(4.442), arr(0.106), "plus")[0]) == pytest.approx(4.548, abs=1e-12)
    assert float(p.level_from_deviation(arr(26000.0), arr(-4.239), "percent")[0]) == pytest.approx(24897.86, abs=1e-9)
    assert p.level_from_deviation(arr(1, 2), arr(0, 0), "percent") == pytest.approx([1, 2], abs=1e-15)
    with pytest.raises(ValueError, match="kind"):
        p.level_from_deviation(arr(1), arr(1), "ratio")
    with pytest.raises(ValueError, match="shape"):
        p.level_from_deviation(arr(1, 2), arr(1), "plus")


def test_check_vintage():
    assert p.check_vintage("2026-02-04", "2026-08-31") is None
    assert p.check_vintage("2026-08-31", "2026-08-31") is None
    with pytest.raises(ValueError, match="vintage"):
        p.check_vintage("2027-01-01", "2026-08-31")
    with pytest.raises(ValueError, match="vintage_date"):
        p.check_vintage("not-a-date", "2026-08-31")
    with pytest.raises(ValueError, match="observation_date"):
        p.check_vintage("2026-01-01", "2026-13-31")


def test_paths_reproduce_the_c06_committed_output():
    """The engine form of `cases/c06/macro.py` (R14): the committed paths of C06's reporting date 2020-06 (Contraste
    `data/derived/C06/models-rd-2020-06.json`, `fit.scenario_definitions`, 40 quarters from the quarter after the
    reporting
    date) are rebuilt from their first 13 quarters (the Federal Reserve shape on the last known value) by the
    reversion rule, with the product's REVERSION_QUARTERS = 12."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["reversion_quarters"] == 12 and data["horizon"] == 13
    paths = {
        s["id"]: {m["variable"]: np.array(m["values"], dtype=float) for m in s["macro"]} for s in data["scenarios"]
    }
    for var in ("unemployment_rate", "real_gdp_growth", "prime_rate"):
        base = paths["baseline"][var]
        tail = base[12]
        rebuilt_base = p.revert_to_baseline(base[:13], tail, 40, 12)
        assert np.max(np.abs(rebuilt_base - base)) <= 1e-6  # the committed values are rounded to 9 digits
        for name in ("adverse", "severely-adverse"):
            published = paths[name][var]
            if var == "prime_rate" and name == "severely-adverse":
                continue  # the product floors this path at 0 after building it, so its unfloored 13th value is unknown
            rebuilt = p.revert_to_baseline(published[:13], tail, 40, 12)
            assert np.max(np.abs(rebuilt - published)) <= 1e-6, (name, var)
    # the adverse shape is the midpoint of the two Fed shapes: the same transposition gives the mean of the paths
    for var in ("unemployment_rate", "real_gdp_growth"):
        mid = 0.5 * (paths["baseline"][var][:13] + paths["severely-adverse"][var][:13])
        assert np.max(np.abs(mid - paths["adverse"][var][:13])) <= 1e-9
    # the transposition: shapes re-anchored at another jump-off keep their first difference and the C06 values at 0
    # shift
    for var in ("unemployment_rate", "real_gdp_growth"):
        sev = paths["severely-adverse"][var][:13]
        moved = p.transpose_path(sev, sev[0], sev[0] + 1.5, 13)
        assert np.allclose(moved - sev, 1.5) and np.allclose(np.diff(moved), np.diff(sev))
