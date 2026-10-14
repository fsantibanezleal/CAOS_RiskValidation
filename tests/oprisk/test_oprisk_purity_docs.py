"""RV-1224 and RV-1225: the operational risk modules import with only NumPy and SciPy, do not mutate their inputs and
are deterministic; the documentation pages carry their equations, sources and limits."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "oprisk"
NAMES = ("01_distributions-and-truncation.md", "02_aggregate-loss.md", "03_sla-and-scenarios.md", "04_sma-ope25.md",
         "05_validation-tests.md")
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib")


def test_modules_import_no_dataframe_library():
    code = (
        "import sys, riskvalidation.oprisk.distributions, riskvalidation.oprisk.aggregate, riskvalidation.oprisk.sla,"
        " riskvalidation.oprisk.scenarios, riskvalidation.oprisk.sma, riskvalidation.validation.oprisk;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.oprisk import aggregate, distributions, scenarios, sla, sma
    from riskvalidation.validation import oprisk as val

    def arr(*x):
        return np.array(x, dtype=float)

    rng = np.random.default_rng(3)
    losses = 20_000.0 * np.exp(rng.exponential(1.0, size=300))
    f = aggregate.discretise_severity("lognormal", (0.0, 1.0), 1.0, 64)
    h = aggregate.panjer_recursion("poisson", (3.0,), f)
    sp = scenarios.ScenarioPoint
    pts = [sp("a", "percentile", 200.0, 2.5), sp("b", "percentile", 1800.0, 0.1)]
    return [
        (distributions.frequency_pmf, ("poisson", (2.0,), 5)),
        (distributions.severity_cdf, ("gpd", (0.5, 100.0, 0.0), arr(50.0, 500.0))),
        (distributions.severity_ppf, ("lognormal", (0.0, 1.0), arr(0.1, 0.9))),
        (distributions.truncated_severity_cdf, ("lognormal", (0.0, 1.0), arr(2.0, 5.0), 1.0)),
        (distributions.fit_frequency, (np.array([3, 5, 4, 6]), "poisson")),
        (distributions.fit_severity, ("lognormal", losses, 20_000.0, True)),
        (distributions.fit_severity, ("gpd", losses, 20_000.0, True)),
        (distributions.fit_splice, (losses, 20_000.0, 60_000.0, 10)),
        (distributions.tail_index_profile, (losses, arr(20_000.0, 40_000.0), 10)),
        (aggregate.discretise_severity, ("lognormal", (0.0, 1.0), 1.0, 32)),
        (aggregate.panjer_recursion, ("poisson", (3.0,), f)),
        (aggregate.fft_aggregate, ("poisson", (3.0,), f)),
        (aggregate.simulate_annual_losses, ("poisson", (3.0,), "lognormal", (0.0, 1.0), 500, 7)),
        (aggregate.mc_quantile, (losses, 0.99)),
        (aggregate.quantile_from_pmf, (h, 1.0, 0.9)),
        (aggregate.exceedance_from_pmf, (h, 1.0)),
        (aggregate.compound_poisson_moments, (3.0, "lognormal", (0.0, 1.0))),
        (sla.sla_var, (10.0, "gpd", (0.5, 100.0, 0.0), 0.99)),
        (sla.sla_var_mean_corrected, (10.0, "gpd", (0.5, 100.0, 0.0), 0.99)),
        (scenarios.fit_tail_to_scenarios, (pts, 0.0, 10.0)),
        (sma.sma_capital, (35e9, np.full(10, 3e8))),
        (val.capital_stability, (arr(90, 100, 110, 120), 100.0)),
        (val.threshold_stability, (arr(1, 2, 3), arr(0.5, 0.6, 0.55))),
        (val.sma_ilm_response, (150.0, np.full(10, 10.0), 90.0)),
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
        return len(r1) == len(r2) and all(_same(p, q) for p, q in zip(r1, r2, strict=True))
    if isinstance(r1, float) and r1 != r1:
        return r2 != r2
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


def test_oprisk_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv:?\s?\d{4}\.\d{4,5}|https://", text), f"{name} cites no source"
        assert "Symbols" in text or "symbols" in text, f"{name} defines no symbols"
        assert "UNVERIFIED" in text or "Limits" in text or "limits" in text, f"{name} states no limit"
    agg = (PAGES / "02_aggregate-loss.md").read_text(encoding="utf-8")
    assert "0.998999329" in agg and "0.999000217" in agg and "5849" in agg and "738.9056" in agg
    assert "Table 2" in agg and "arXiv 1008.1108" in agg
    sla_page = (PAGES / "03_sla-and-scenarios.md").read_text(encoding="utf-8")
    assert "UNVERIFIED" in sla_page and "6124.555320" in sla_page and "infinite mean" in sla_page
    sma_page = (PAGES / "04_sma-ope25.md").read_text(encoding="utf-8")
    assert "OPE25.7" in sma_page and "5.37" in sma_page and "OPE25.15" in sma_page and "OPE25.9" in sma_page
    dist = (PAGES / "01_distributions-and-truncation.md").read_text(encoding="utf-8")
    assert "22.5" in dist and "1 - F(H)" in dist.replace("\\", "")
    val = (PAGES / "05_validation-tests.md").read_text(encoding="utf-8")
    assert "descriptive" in val and "no compliance claim" in (val + index)
