"""RV-622 and RV-623: the LGD and EAD modules import with only NumPy and SciPy, do not mutate their inputs and are
deterministic; the documentation pages carry their equations, sources and limits."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "lgd_ead"
NAMES = ("01_workout-lgd.md", "02_lgd-models.md", "03_downturn.md", "04_ccf-and-ead.md", "05_floors.md",
         "06_validation-tests.md")
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib")


def test_modules_import_no_dataframe_library():
    code = (
        "import sys, riskvalidation.lgd_ead.workout, riskvalidation.lgd_ead.lgd_models,"
        " riskvalidation.lgd_ead.downturn,"
        " riskvalidation.lgd_ead.ccf, riskvalidation.lgd_ead.floors, riskvalidation.validation.lgd_ead;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.lgd_ead import ccf, downturn, floors, lgd_models, workout
    from riskvalidation.validation import lgd_ead as val

    def arr(*x):
        return np.array(x, dtype=float)

    X = np.array([[1, 0], [1, 0], [1, 1], [1, 1], [1, 0.5], [1, 0.2], [1, 0.9], [1, 0.4]], dtype=float)
    y = arr(0, 0.5, 0.5, 1.0, 0.3, 0.0, 0.8, 0.45)
    t = np.array([[1.0, 2.0]])
    drawn, limit = np.full(22, 100.0), np.full(22, 1000.0)
    return [
        (workout.workout_lgd, (arr(100), t, np.array([[40.0, 30.0]]), np.array([[10.0, 0.0]]), np.zeros((1, 2)), 0.1)),
        (workout.range_flag, (arr(-0.1, 0.5, 1.2),)),
        (workout.resolve_incomplete, (arr(0.2, 0.6, 0.9), np.array([True, True, False]), "excluded")),
        (workout.lgd_distribution, (arr(-0.05, 0, 0.5, 1.2), arr(0, 0.5, 1.0))),
        (lgd_models.endpoint_transform, (y, 0.01)),
        (lgd_models.fit_ols, (X, y)),
        (lgd_models.fit_fractional_logit, (X, y)),
        (lgd_models.fit_beta, (X, y, 0.01)),
        (lgd_models.fit_hurdle, (X, y)),
        (lgd_models.fit_tobit, (X, y)),
        (downturn.downturn_add_on, (arr(0.3, 0.6, 0.4), arr(100, 50, 50), np.array([False, True, False]))),
        (ccf.ccf_observations, (drawn, limit, 900.0, np.array([9, 12]))),
        (ccf.treat_ccf, (arr(-0.1, 0.2, 1.3), "floor-cap")),
        (ccf.fit_direct_ead, (X, y * 1000, np.full(8, 1000.0))),
        (floors.apply_lgd_floor, (arr(0.2, 0.3, 0.6), 0.25)),
        (floors.ead_floor, (arr(600), arr(1000), arr(0.4))),
        (val.lgd_calibration_by_segment, (arr(0.4, 0.5, 0.6), arr(0.3, 0.3, 0.35))),
        (val.lgd_discrimination, (arr(0.1, 0.4, 0.6, 0.9), arr(0.2, 0.3, 0.7, 0.8))),
        (val.ead_backtest_by_band, (arr(900, 500), arr(800, 600), arr(0.2, 0.8), arr(0, 0.5, 1.0))),
    ]


def _same(r1, r2):
    if isinstance(r1, np.ndarray):
        return np.array_equal(r1, r2, equal_nan=True) if r1.dtype.kind in "fc" else np.array_equal(r1, r2)
    if hasattr(r1, "to_dict"):
        d1, d2 = r1.to_dict(), r2.to_dict()
        return d1 == d2
    if hasattr(r1, "__dataclass_fields__"):
        return all(_same(getattr(r1, k), getattr(r2, k)) for k in r1.__dataclass_fields__)
    if isinstance(r1, dict):
        return r1.keys() == r2.keys() and all(_same(r1[k], r2[k]) for k in r1)
    if isinstance(r1, tuple):
        return all(_same(p, q) for p, q in zip(r1, r2, strict=True))
    return r1 == r2


def test_inputs_are_not_mutated_and_results_are_deterministic():
    for fn, args in _calls():
        before = copy.deepcopy(args)
        first = fn(*args)
        second = fn(*args)
        for b, aft in zip(before, args, strict=True):
            if isinstance(b, np.ndarray):
                assert np.array_equal(b, aft, equal_nan=True), f"{fn.__name__} mutated an input"
        assert _same(first, second), f"{fn.__name__} is not deterministic"


def test_lgd_ead_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv:\d{4}\.\d{4,5}|https://", text), f"{name} cites no source"
        assert "Symbols" in text or "symbols" in text, f"{name} defines no symbols"
        assert "UNVERIFIED" in text or "Limits" in text or "limits" in text, f"{name} states no limit"
    ccf_page = (PAGES / "04_ccf-and-ead.md").read_text(encoding="utf-8")
    assert "eu-undrawn" in ccf_page and "loan equivalent" in ccf_page and "0.827586" in ccf_page
    floors_page = (PAGES / "05_floors.md").read_text(encoding="utf-8")
    assert "CRE32.16" in floors_page and "CRE32.17" in floors_page and "CRE32.58" in floors_page
    assert "0.479339" in (PAGES / "01_workout-lgd.md").read_text(encoding="utf-8")
    val = (PAGES / "06_validation-tests.md").read_text(encoding="utf-8")
    assert "Contraste internal gate" in val and "no compliance claim" in (val + index)
