"""RV-828 to RV-830: no forecast reads later data, results are deterministic and inputs are not mutated, the modules
import with only NumPy and SciPy, the optional extra names `arch`, and the documentation pages carry equations,
sources and limits."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from market_sim import simulate_garch

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "market"
NAMES = ("01_historical-simulation.md", "02_parametric-and-ewma.md", "03_garch-and-evt.md", "04_caviar-and-gas-fz.md",
         "05_scoring-and-comparison.md", "06_frtb-tests.md", "07_backtests.md")
HEAVY = ("pandas", "statsmodels", "sklearn", "torch", "matplotlib", "arch")


def test_modules_import_no_dataframe_library():
    code = (
        "import sys, riskvalidation.market.conventions, riskvalidation.market.historical,"
        " riskvalidation.market.parametric, riskvalidation.market.garch, riskvalidation.market.evt,"
        " riskvalidation.market.caviar, riskvalidation.market.gas_fz, riskvalidation.market.scoring,"
        " riskvalidation.market.frtb, riskvalidation.validation.market;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def test_market_extra_names_arch():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^market\s*=\s*\[(.*?)\]", text, re.M)
    assert m and "arch" in m.group(1), "pyproject.toml must define the optional extra `market` with arch"
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.M).group(1)
    assert "arch" not in deps, "the engine core stays NumPy and SciPy"


def test_ci_installs_the_market_extra():
    """Review M2: with only `.[dev]` the arch tests skip in CI."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r'pip install -e "\.\[dev,market\]"', ci), "ci.yml must install .[dev,market]"
    assert 'pip install -e ".[dev]"' not in ci


def _estimators(series):
    from riskvalidation.market import caviar, conventions, gas_fz, historical, parametric, scoring

    def cav(y):
        fit = caviar.fit_caviar(y, 0.05, n_starts=2, seed=3)
        return fit.params.tolist(), caviar.caviar_var_forecast(fit, y), caviar.caviar_es_forecast(fit, y)

    def gas(y):
        fit = gas_fz.fit_gas_fz(y, 0.05, n_starts=2, seed=3)
        return fit.params.tolist(), fit.mean_fz0, gas_fz.gas_fz_forecast(fit, y)

    return {
        "hs": lambda y: historical.historical_var_es(y, 0.025, 500),
        "awhs": lambda y: historical.age_weighted_var_es(y, 0.025, 0.98, 500),
        "ewma": lambda y: parametric.ewma_var_es(y, 0.94, 0.025, float(np.var(y[:100]))),
        "sample": lambda y: conventions.var_es_from_sample(y, 0.025),
        "caviar": cav,
        "gas": gas,
        "fz0": lambda y: scoring.mean_scores(y, np.full(y.size, -0.02), np.full(y.size, -0.03), 0.025),
    }


def test_forecasts_do_not_depend_on_later_data():
    series = simulate_garch(900, nu=6, seed=91)
    t = 700
    buffer = series.copy()
    head = buffer[:t]  # a view: what the caller holds after day t is not part of the argument
    est = _estimators(series)
    first = {k: f(head.copy()) for k, f in est.items() if k != "fz0"}
    buffer[t:] = 99.0  # the future, changed after the fact
    second = {k: f(head) for k, f in est.items() if k != "fz0"}
    assert first == second
    # and the forecast does read the last day it is given
    moved = head.copy()
    moved[-1] = -0.2
    assert est["hs"](moved) != first["hs"]
    assert est["ewma"](moved) != first["ewma"]
    # the leakage-safe rolling scheme: the forecast of day t from returns[:t] equals the forecast from a longer array
    # that was sliced to the same days
    for k in ("hs", "awhs", "ewma"):
        assert est[k](series[:t]) == est[k](series.copy()[:t])


def test_inputs_are_not_mutated_and_results_are_deterministic():
    series = simulate_garch(700, nu=6, seed=92)
    for name, fn in _estimators(series).items():
        before = series.copy()
        r1, r2 = fn(series), fn(series)
        assert np.array_equal(series, before), f"{name} mutated its input"
        assert r1 == r2, f"{name} is not deterministic"


def test_market_pages_carry_equations_sources_and_limits():
    for name in NAMES:
        page = PAGES / name
        assert page.is_file(), name
        text = page.read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"https?://|doi:", text), f"{name} cites no source with a DOI or a URL"
        assert re.search(r"(?i)\blimits?\b", text), f"{name} states no limits"
        assert "golden" in text.lower(), f"{name} states no golden value"
        assert chr(0x2014) not in text
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert name in index
    decisions = "\n".join((PAGES / n).read_text(encoding="utf-8") for n in NAMES)
    for d in range(1, 11):
        assert f"D-{d}" in decisions, f"decision D-{d} is not recorded"
    assert "return convention" in decisions and "loss numbers" in decisions
