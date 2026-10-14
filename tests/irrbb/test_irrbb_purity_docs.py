"""RV-1029 and RV-1030: the IRRBB modules import with only NumPy and SciPy, do not mutate their inputs and are
deterministic;
the documentation pages carry their equations, sources and limits."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "irrbb"
NAMES = ("01_curves.md", "02_shocks.md", "03_slotting.md", "04_eve-and-nii.md", "05_annex1.md", "06_behavioural.md",
         "07_simulation.md", "08_validation-tests.md")
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib", "QuantLib")


def test_modules_import_no_dataframe_library():
    code = (
        "import sys, riskvalidation.irrbb.curves, riskvalidation.irrbb.shocks, riskvalidation.irrbb.slotting,"
        " riskvalidation.irrbb.eve_nii, riskvalidation.irrbb.behavioural, riskvalidation.irrbb.simulation,"
        " riskvalidation.validation.irrbb;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.irrbb import behavioural, curves, eve_nii, shocks, simulation, slotting
    from riskvalidation.validation import irrbb as val

    def arr(*x):
        return np.array(x, dtype=float)

    mids = slotting.bucket_grid().midpoints
    cf = np.zeros(19)
    cf[3], cf[8], cf[15] = 100.0, -30.0, -70.0
    flat = np.full(19, 0.02)
    rng = np.random.default_rng(0)
    hist = np.cumsum(rng.normal(0, 0.002, (40, 3)), axis=0) + 0.03
    yields = np.cumsum(rng.normal(0, 0.002, (60, 3)), axis=0) + 0.03
    m = arr(0.5, 1, 2, 3, 5, 7, 10, 20)
    inc = rng.uniform(-0.1, 0.1, 400)
    sea = rng.uniform(0, 300, 400)
    prepaid = rng.random(400) < 0.3
    path = np.full(12, 0.03)
    return [
        (curves.zero_rates_at, (arr(1, 2), arr(0.01, 0.03), arr(1.5))),
        (curves.discount_factors, (arr(0.02), arr(2.5))),
        (curves.fit_nelson_siegel, (m, curves.nelson_siegel_yield(m, 0.05, -0.02, 0.01, 2.0))),
        (curves.fit_svensson, (m, curves.svensson_yield(m, 0.05, -0.02, 0.01, 0.005, 2.0, 6.0))),
        (curves.fit_dns_factors, (arr(3, 12, 24, 60), curves.dns_yield(arr(3, 12, 24, 60), 0.05, -0.02, 0.01)[None])),
        (curves.forecast_dns_factors, (hist, 6, 20, 3)),
        (shocks.shock_curve, ("steepener", 200.0, 300.0, 225.0, mids)),
        (shocks.apply_floor, (flat, np.full(19, -300.0), 0.0)),
        (slotting.slot_cash_flows, (arr(0.5, 2.5, 9.5), arr(100, -30, -70))),
        (slotting.nmd_split, (100.0, "retail-transactional", 0.9, 5.0)),
        (eve_nii.eve, (cf, flat, mids)),
        (eve_nii.delta_eve, (cf, cf, flat, flat + 0.01, mids)),
        (eve_nii.annex1_delta_nii, (arr(30, -10), arr(-20, 5), -200.0)),
        (eve_nii.annex1_delta_eve, (arr(10, -20, 0, 5, -30, 40), arr(20, 30, -10, -2, 50, -15))),
        (eve_nii.prioritised_bank, (6.0, 10.0, 100.0, 40.0)),
        (behavioural.replicating_portfolio, (yields[:, 1], yields, arr(1, 3, 5))),
        (behavioural.fit_prepayment_logit, (inc, sea, prepaid)),
        (simulation.simulate_nii, (arr(100, 100), arr(1, -1), arr(0.05, 0.015), arr(0.02, 0.0), arr(1.0, 0.5),
                                   np.array([0, 0]), path, 0.0)),
        (simulation.hw1f_scenarios, (arr(0.02, 0.03), arr(1, 5), 0.1, 0.01, 12, 20, 4)),
        (simulation.dns_scenarios, (hist, arr(3, 12, 36), 6, 20, 5)),
        (simulation.ear_evar, (np.arange(1.0, 101.0), np.arange(1.0, 101.0), 60.0, 60.0)),
        (val.srp31_92_goldens, ()),
        (val.cmf_annex1_goldens, ()),
        (val.eve_sum_of_parts, (np.vstack([cf, cf]), flat, mids, 2 * float(eve_nii.eve(cf, flat, mids)))),
        (val.nmd_parameter_stability, (np.array([[0.5, 0.5], [0.65, 0.35]]), arr(1, 5))),
        (val.eve_with_without_behavioural, (10.0, 20.0, 100.0)),
        (val.nii_out_of_sample, (90.0, 100.0)),
    ]


def _same(r1, r2):
    if isinstance(r1, np.ndarray):
        return np.array_equal(r1, r2, equal_nan=True) if r1.dtype.kind in "fc" else np.array_equal(r1, r2)
    if hasattr(r1, "to_dict"):
        return r1.to_dict() == r2.to_dict()
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


def test_irrbb_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv:\d{4}\.\d{4,5}|https://", text), f"{name} cites no source"
        assert "Symbols" in text or "symbols" in text, f"{name} defines no symbols"
        assert "UNVERIFIED" in text or "Limits" in text or "limits" in text, f"{name} states no limit"
        assert "Golden" in text or "golden" in text, f"{name} states no golden value"
    shocks_page = (PAGES / "02_shocks.md").read_text(encoding="utf-8")
    for needle in ("SRP31.92", "41.7", "CLP", "Contraste estimate"):
        assert needle in shocks_page, needle
    slot = (PAGES / "03_slotting.md").read_text(encoding="utf-8")
    assert "SRP31.96" in slot and "0.0028" in slot and "UNVERIFIED" in slot
    annex = (PAGES / "05_annex1.md").read_text(encoding="utf-8")
    assert "RAN 21-13" in annex and "0.025" in annex and "offset" in annex and "Hoja 9" in annex
    curves_page = (PAGES / "01_curves.md").read_text(encoding="utf-8")
    assert "29.446" in curves_page and "0.0609" in curves_page
    eve_page = (PAGES / "04_eve-and-nii.md").read_text(encoding="utf-8")
    assert "SRP31.129" in eve_page and "loss" in eve_page
    val = (PAGES / "08_validation-tests.md").read_text(encoding="utf-8")
    assert "Contraste internal gate" in val and "no compliance claim" in (val + index)
