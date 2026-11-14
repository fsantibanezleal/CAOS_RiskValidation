"""RV-728 and RV-729: the portfolio modules import with only NumPy and SciPy, do not mutate their inputs and are
deterministic for a seed; the documentation pages carry their equations, sources and limits."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from _portfolio_helpers import identical

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "portfolio"
NAMES = ("01_portfolio-and-measures.md", "02_copulas.md", "03_creditrisk-plus.md", "04_granularity-and-importance.md",
         "05_euler-and-concentration.md", "06_vasicek-fit.md", "07_validation-tests.md", "08_panel-fit.md")
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib")
MODULES = ("portfolio", "copula", "creditrisk_plus", "granularity", "importance", "euler", "asrf_fit", "panel_fit")


def test_modules_import_no_dataframe_library():
    imports = ", ".join(f"riskvalidation.portfolio.{m}" for m in MODULES)
    code = (
        f"import sys, {imports}, riskvalidation.validation.portfolio;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.portfolio import (
        asrf_fit,
        copula,
        creditrisk_plus,
        euler,
        granularity,
        importance,
        panel_fit,
        portfolio,
    )
    from riskvalidation.validation import portfolio as val

    p = identical(20, pd=0.03, rho=0.2, lgd=0.5, ead=2.0, n_groups=2)
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 1.0, size=2_000)
    parts = rng.gamma(2.0, 1.0, size=(2_000, 3))
    total = parts.sum(axis=1)
    rates = np.array([0.011, 0.02, 0.035, 0.008, 0.05, 0.015, 0.027, 0.012])
    nu = np.array([1, 2, 3])
    panel_d = np.array([[1, 4, 9], [0, 2, 6], [2, 5, 13], [0, 1, 4], [1, 3, 8], [0, 2, 7]])
    panel_n = np.tile(np.array([300, 200, 100]), (6, 1))
    crp = creditrisk_plus.CreditRiskPlusInput(nu=nu, pd=np.array([0.1, 0.2, 0.3]), w0=np.array([0.2, 0.5, 1.0]),
                                              weights=np.array([[0.8], [0.5], [0.0]]), sigma2=np.array([0.5]),
                                              loss_unit=1.0, currency="USD")
    return [
        (portfolio.expected_loss, (p,)),
        (portfolio.lhp_measures, (10.0, 0.5, 0.03, 0.2, (0.99, 0.999))),
        (portfolio.var_from_sample, (x, 0.99)),
        (portfolio.es_from_sample, (x, 0.99)),
        (portfolio.measures_from_sample, (x, (0.99,), float(x.mean()))),
        (copula.simulate_gaussian, (p, 2_000, 4, 500, True)),
        (copula.simulate_student_t, (p, 6.0, 2_000, 4, 500, True)),
        (copula.mc_standard_errors, (x, 0.95)),
        (creditrisk_plus.pmf_stable, (crp, 50)),
        (creditrisk_plus.pmf_panjer, (crp, 50)),
        (creditrisk_plus.measures_from_pmf, (creditrisk_plus.pmf_stable(crp, 60), 1.0, (0.9,))),
        (importance.twisted_pd, (np.array([0.1, 0.2]), np.array([1.0, 2.0]), 0.5)),
        (importance.solve_theta, (np.array([0.1, 0.2]), np.array([1.0, 2.0]), 1.5)),
        (importance.importance_sample_gaussian, (p, 12.0, 2_000, 4)),
        (granularity.granularity_adjustment, (np.array([0.5, 0.5]), np.array([0.04, 0.03]), np.array([0.01, 0.02]),
                                              np.array([0.45, 0.4]), 0.25, 0.25, 0.999)),
        (granularity.k_creditrisk_plus, (np.array([0.01]), np.array([0.45]), np.array([0.5]), 17.5)),
        (euler.es_contributions, (parts, total, 0.95)),
        (euler.var_contributions_kernel, (parts, total, 0.95)),
        (euler.asrf_var_contributions, (np.array([60.0, 40.0]), np.array([0.01, 0.05]), 0.45, 0.15, 0.999)),
        (euler.hhi, (np.array([50.0, 30.0, 20.0]),)),
        (asrf_fit.log_likelihood, (rates, 0.02, 0.15)),
        (asrf_fit.fit_ml, (rates,)),
        (asrf_fit.fit_moments, (rates,)),
        (asrf_fit.profile_likelihood, (rates, np.array([0.05, 0.1, 0.2]))),
        (asrf_fit.implied_factors, (rates, 0.02, 0.15)),
        (panel_fit.fit_panel_asrf, (panel_d, panel_n)),
        (panel_fit.fit_panel_moments, (panel_d, panel_n)),
        (panel_fit.default_correlation, (np.array([0.01, 0.05]), 0.2)),
        (val.tail_backtest, (rates, 0.02, 0.15)),
        (val.mc_convergence, (0.101, 0.001, 0.1)),
        (val.euler_additivity, (np.array([0.5, 0.5]), 1.0)),
        (val.crp_recursion_vs_mc, (np.array([0.5, 0.3, 0.2]), np.repeat([0, 1, 2], [5, 3, 2]))),
        (val.correlation_size_power, (rng.random(200) < 0.06, 0.1, 0.1)),
    ]


def _flat(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return [_flat(getattr(obj, f)) for f in obj.__dataclass_fields__]
    if hasattr(obj, "to_dict"):
        return _flat(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _flat(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_flat(v) for v in obj]
    return obj


def test_inputs_are_not_mutated_and_results_are_deterministic():
    for fn, args in _calls():
        before = copy.deepcopy(args)
        first = fn(*args)
        second = fn(*args)
        for a, b in zip(args, before, strict=True):
            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b), fn.__name__
        assert repr(_flat(first)) == repr(_flat(second)), fn.__name__


def test_portfolio_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    assert "RV-701" in index and "RV-729" in index and "UNVERIFIED" in index
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv \d{4}\.\d{4,5}|https://", text), f"{name} cites no primary source"
        assert re.search(r"^#+ .*\b(Limits|Limitations)\b", text, re.M | re.I), f"{name} has no limits section"
        assert re.search(r"golden|Golden", text), f"{name} states no golden value"
    # the items the dossiers mark UNVERIFIED are marked here
    assert "UNVERIFIED" in (PAGES / "04_granularity-and-importance.md").read_text(encoding="utf-8")
    assert "UNVERIFIED" in (PAGES / "05_euler-and-concentration.md").read_text(encoding="utf-8")
