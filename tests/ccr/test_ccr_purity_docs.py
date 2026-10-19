"""RV-983 and RV-985: the ccr modules import with only NumPy and SciPy, do not mutate their inputs and are deterministic
for a seed; the documentation pages carry their equations, sources and limits."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "ccr"
NAMES = ("01_hw1f.md", "02_swaps.md", "03_exposure.md", "04_collateral.md", "05_cva-and-wrong-way-risk.md",
         "06_validation-tests.md")
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib", "QuantLib", "arch")


def test_modules_import_numpy_scipy_only():
    code = (
        "import sys, riskvalidation.ccr.hw1f, riskvalidation.ccr.swaps, riskvalidation.ccr.exposure,"
        " riskvalidation.ccr.collateral, riskvalidation.ccr.cva, riskvalidation.validation.ccr;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.ccr import collateral as co
    from riskvalidation.ccr import cva as cv
    from riskvalidation.ccr import exposure as ex
    from riskvalidation.ccr import hw1f, swaps
    from riskvalidation.validation import ccr as val

    def arr(*x):
        return np.array(x, dtype=float)

    p0, fwd = hw1f.curve_functions(arr(0.03, 0.032, 0.035, 0.04, 0.042), arr(0.5, 1, 2, 5, 10))
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 2.0, 5)
    sw = swaps.Swap("s", 1.0, 0.04, True, 0.0, 2.0, 2)
    sim_grid = swaps.simulation_grid(grid, [sw])
    short = hw1f.simulate_short_rate(sim_grid, 0.1, 0.01, fwd, 40, 1)
    values = rng.normal(0.0, 1.0, (40, 5))
    by_trade = rng.normal(0.0, 1.0, (3, 40, 5))
    w = np.cumsum(rng.normal(0.0, 1.0, (60, 4)), axis=1)
    tgt = np.exp(-0.02 * np.arange(1, 5) * 0.5)
    dts = np.full(4, 0.5)
    expo = np.maximum(w, 0.0)
    eff = ex.effective_ee(arr(1, 3, 2))
    dates = [f"{2000 + i // 12}-{i % 12 + 1:02d}-15" for i in range(60)]
    series = 0.03 + hw1f.simulate_state(np.arange(60) / 12.0, 0.4, 0.01, 2, 1)[0]
    realised = rng.normal(0.0, 1.0, 20)
    simulated = rng.normal(0.0, 1.0, (20, 50))
    return [
        (hw1f.phi, (arr(1, 2), 0.1, 0.01, fwd)),
        (hw1f.bond_b, (arr(0, 1), arr(2, 3), 0.1)),
        (hw1f.bond_a, (arr(0, 1), arr(2, 3), 0.1, 0.01, p0, fwd)),
        (hw1f.bond_price, (arr(0.01, 0.02), 1.0, 3.0, 0.1, 0.01, p0, fwd)),
        (hw1f.simulate_state, (grid, 0.1, 0.01, 10, 3)),
        (hw1f.simulate_short_rate, (grid, 0.1, 0.01, fwd, 10, 3)),
        (hw1f.bank_account, (grid, values)),
        (hw1f.bond_option, (True, 0.9, 1.0, 3.0, 0.1, 0.01, p0)),
        (hw1f.swaption_jamshidian, (True, 1.0, arr(1.5, 2.0), arr(0.02, 1.02), 0.1, 0.01, p0, fwd)),
        (hw1f.calibrate_ar1, (dates, series, "2010-01-01")),
        (swaps.cash_flow_times, (sw,)),
        (swaps.swap_value, (sw, 1.0, arr(0.02, 0.03), 0.1, 0.01, p0, fwd)),
        (swaps.par_fixed_rate, (2.0, 2, p0)),
        (swaps.swap_values_on_grid, ([sw], sim_grid, short, 0.1, 0.01, p0, fwd)),
        (ex.expected_exposure, (values, np.exp(0.03 * grid)[None, :] * np.ones((40, 1)))),
        (ex.effective_ee, (arr(1, 3, 2), 0.5)),
        (ex.effective_epe, (arr(0.25, 0.5, 1.0), eff)),
        (ex.epe, (arr(0.25, 0.5, 1.0), arr(1, 3, 2))),
        (ex.pfe, (values, 0.95, 4, 20)),
        (ex.net_values, (by_trade,)),
        (ex.netting_benefit, (arr(1, 2) * np.ones((2, 2)), arr(1.5, 3.0))),
        (ex.effective_maturity, (arr(0.5, 1.0, 2.0), arr(1, 2, 3), arr(1, 2, 3), arr(0.98, 0.97, 0.94))),
        (ex.analytic_ee_single_swap, (sw, arr(0.0, 0.5, 1.0, 2.0), 0.1, 0.01, p0, fwd)),
        (co.credit_support_amount, (arr(-5, 0, 7), 1.0, 2.0, 3.0)),
        (co.delivery_amount, (arr(10, 5), arr(8, 6), arr(0, 0), 1.0)),
        (co.collateralised_exposure, (values, values * 0.9)),
        (co.variation_margin_exposure, (values, values * 1.1, 0.0, 1.0, 1.0, 0.5)),
        (co.dynamic_initial_margin, (values[:, 0], values[:, 1], values[:, 2], 0.99, 14, 14)),
        (co.apply_initial_margin, (np.abs(values), np.abs(values) * 0.5)),
        (cv.survival_from_spread, (arr(0.01, 0.02), arr(1, 2), 0.6)),
        (cv.default_probabilities, (arr(1.0, 0.9, 0.8),)),
        (cv.cva, (arr(10, 20), arr(0.1, 0.1), 0.6)),
        (cv.dva, (arr(10, 20), arr(0.1, 0.1), 0.6)),
        (cv.wwr_calibrate_a, (w, tgt, dts, 0.05)),
        (cv.wwr_cva, (w, expo, tgt, dts, 0.6, 0.05)),
        (val.ee_analytic_vs_simulated, (arr(1, 2), arr(1, 2.1), arr(0.1, 0.1))),
        (val.eepe_recomputed, (arr(0.25, 0.5, 1.0), arr(1, 3, 2), 2.5)),
        (val.cva_recomputed, (arr(10, 20), arr(0.1, 0.1), 0.6, 1.8)),
        (val.path_convergence, ({100: arr(1, 2), 400: arr(1, 2.1)}, {100: 1.0, 400: 1.1})),
        (val.reconcile_ee_profile, (arr(1, 2), arr(0.1, 0.1), arr(1, 2.1), arr(0.1, 0.1))),
        (val.pfe_exceedance_backtest, (realised, np.full(20, 1.5), 0.95)),
        (val.pit_exposure_backtest, (realised, simulated, {"p05": 2.5, "p01": 3.9})),
        (val.effective_sample_size, (norm_cdf(rng.standard_normal(30)),)),
    ]


def norm_cdf(z):
    from scipy.stats import norm
    return norm.cdf(z)


def _same(a, b):
    if isinstance(a, tuple):
        return all(_same(x, y) for x, y in zip(a, b, strict=False)) and len(a) == len(b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if hasattr(a, "to_dict"):
        return str(a.to_dict()) == str(b.to_dict())
    return a == b


def test_functions_do_not_mutate_and_are_deterministic():
    for fn, args in _calls():
        before = copy.deepcopy(args)
        first = fn(*args)
        second = fn(*args)
        for x, y in zip(args, before, strict=True):
            if isinstance(x, np.ndarray):
                assert np.array_equal(x, y), f"{fn.__module__}.{fn.__name__} mutated an input"
            elif isinstance(x, dict):
                assert x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) for k in x)
        assert _same(first, second), f"{fn.__module__}.{fn.__name__} is not deterministic"


def test_docs_pages():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv \d{4}\.\d{4,5}|QuantLib|CRE53|ORE|BCBS 185|dossier 05", text), \
            f"{name} cites no source"
        assert re.search(r"^##\s+(Limits|Conventions and limits)", text, re.M), f"{name} states no limits"
    # the pages state the limits the contract names
    joined = "\n".join((PAGES / n).read_text(encoding="utf-8") for n in NAMES)
    for phrase in ("SIMM", "Andersen", "UNVERIFIED", "regression proxy", "Hull and White", "sticky date",
                   "effective sample size", "0.968546", "0.040773", "-6.91283"):
        assert phrase in joined, phrase
