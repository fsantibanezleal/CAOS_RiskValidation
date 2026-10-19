"""RV-1325 and RV-1326: the aggregation modules import with only NumPy and SciPy, do not mutate their inputs and are
deterministic; the documentation pages carry their equations, sources and limits."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "aggregation"
NAMES = ("01_simple.md", "02_copulas.md", "03_bounds.md", "04_band.md", "05_model-risk.md", "06_validation-tests.md")
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib", "QuantLib")


def test_modules_import_no_dataframe_library():
    code = (
        "import sys, riskvalidation.aggregation.simple, riskvalidation.aggregation.copulas,"
        " riskvalidation.aggregation.bounds, riskvalidation.aggregation.band, riskvalidation.aggregation.model_risk,"
        " riskvalidation.validation.aggregation;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _ident(u):
    return np.asarray(u, dtype=float)


def _pareto(u):
    return (1.0 - np.asarray(u, dtype=float)) ** -0.5 - 1.0


def _calls():
    from riskvalidation.aggregation import band, bounds, copulas, model_risk, simple
    from riskvalidation.validation import aggregation as val

    def arr(*x):
        return np.array(x, dtype=float)

    r5 = np.array([[1.0, 0.5], [0.5, 1.0]])
    rng = np.random.default_rng(0)
    samples = np.exp(rng.normal(size=(500, 3)))
    ppf = copulas.marginal_ppf_sample(np.sort(samples[:, 0]))
    pmf = copulas.marginal_ppf_pmf(arr(0, 1, 2), arr(0.5, 0.3, 0.2))
    m = np.sort(rng.lognormal(size=(40, 3)), axis=0)
    return [
        (simple.simple_sum, (arr(3, 4),)),
        (simple.check_correlation, (r5,)),
        (simple.variance_covariance, (arr(3, 4), r5)),
        (simple.hybrid_quantile, (samples, 0.99)),
        (simple.diversification_factor, (6.0, [3.0, 4.0])),
        (simple.euler_diversification_benefit, (6.0, 8.0)),
        (simple.std_contributions, (np.array([[4.0, 2.0], [2.0, 9.0]]),)),
        (lambda u: ppf(u), (arr(0.2, 0.5, 0.9),)),
        (lambda u: pmf(u), (arr(0.2, 0.5, 0.9),)),
        (copulas.simulate_copula_uniforms, (r5, "t", 5.0, 200, 1)),
        (copulas.simulate_component_losses, ([_ident, _pareto], r5, "gaussian", None, 300, 2, 100)),
        (copulas.t_tail_dependence, (0.5, 4.0)),
        (copulas.clayton_copula, (0.5, 0.5, 1.0)),
        (copulas.clayton_lower_tail_dependence, (2.0,)),
        (copulas.copula_aggregate, ([_ident, _pareto], r5, "t", 5.0, 0.0, 0.9, 500, 3, 200)),
        (bounds.comonotonic_var, ([_pareto, _pareto], 0.99)),
        (bounds.rearrangement_algorithm, (m, "max-min-row-sum", 1e-6, 50, 4)),
        (bounds.worst_var_bounds, ([_pareto] * 3, 0.9, 200, 1e-3, 50, 5)),
        (band.dependence_band, ([_pareto, _pareto], arr(3, 4), arr(0, 0), r5, [4.0], 0.99, 1, 1000, 6, 500, 100, 1e-3)),
        (model_risk.barrieu_scandolo, ((100.0, 130.0, 90.0), 0)),
        (model_risk.glasserman_xu_worst_case, (samples[:, 0], 0.1)),
        (val.simple_sum_and_variance_covariance, (arr(3, 4), r5, 6.0)),
        (val.copula_choice_sensitivity, (100.0, ((4, 107.0), (8, 103.5)))),
        (val.worst_case_bracket, (4.0, (5.0, 7.0), 8.0)),
        (val.correlation_stress, (100.0, 105.0)),
        (val.diversification_sign, ((0.87, 0.9),)),
        (val.horizon_confidence_consistency, ((1, 1), (0.999, 0.999))),
        (val.elliptical_mc_agreement, (100.4, 0.2, 100.0)),
        (val.rosenberg_schuermann_reference, (1.0, 0.80624)),
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
                assert np.array_equal(b, aft, equal_nan=True), f"{getattr(fn, '__name__', fn)} mutated an input"
        assert _same(first, second), f"{getattr(fn, '__name__', fn)} is not deterministic"


def test_aggregation_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv:\d{4}\.\d{4,5}|https://", text), f"{name} cites no source"
        assert "Symbols" in text or "symbols" in text, f"{name} defines no symbols"
        assert "UNVERIFIED" in text or "Limits" in text or "limits" in text, f"{name} states no limit"
        assert "Golden" in text or "golden" in text, f"{name} states no golden value"
    bounds_page = (PAGES / "03_bounds.md").read_text(encoding="utf-8")
    for needle in ("851.72901", "44.7671", "141.67", "Embrechts", "10.1016/j.jbankfin.2013.03.014", "UNVERIFIED"):
        assert needle in bounds_page, needle
    simple_page = (PAGES / "01_simple.md").read_text(encoding="utf-8")
    assert "VaR - EL" in simple_page and "6.0828" in simple_page
    cop = (PAGES / "02_copulas.md").read_text(encoding="utf-8")
    for needle in ("Demarta", "Frey", "0.2532", "Clayton", "BL-078"):
        assert needle in cop, needle
    band_page = (PAGES / "04_band.md").read_text(encoding="utf-8")
    assert "information-not-capital" in band_page and "headline-capital" in band_page
    mr_page = (PAGES / "05_model-risk.md").read_text(encoding="utf-8")
    assert "Barrieu" in mr_page and "Glasserman" in mr_page and "0.7311" in mr_page
    val = (PAGES / "06_validation-tests.md").read_text(encoding="utf-8")
    assert "Contraste internal gate" in val and "no compliance claim" in (val + index)
    assert "reference-not-a-gate" in val and "0.2403" in val
