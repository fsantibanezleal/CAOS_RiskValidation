"""RV-1323 and RV-1324: the stress modules import with only NumPy and SciPy, do not mutate their inputs and are
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
PAGES = ROOT / "docs" / "stress"
NAMES = (
    "01_satellites.md",
    "02_paths.md",
    "03_capital.md",
    "04_reverse-stress.md",
    "05_scenario-models.md",
    "06_validation-tests.md",
)
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib", "QuantLib")


def test_modules_import_no_heavy_library():
    code = (
        "import sys, riskvalidation.stress.satellites, riskvalidation.stress.paths, riskvalidation.stress.capital,"
        " riskvalidation.stress.reverse, riskvalidation.stress.scenario_models, riskvalidation.validation.stress;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.stress import capital, paths, reverse, satellites, scenario_models
    from riskvalidation.validation import stress as val

    def arr(*x):
        return np.array(x, dtype=float)

    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(size=(n, 2))
    g = rng.normal(size=n)
    q = [f"{2000 + k // 4}-{('03-31', '06-30', '09-30', '12-31')[k % 4]}" for k in range(n)]
    fit = satellites.fit_satellite(
        satellites.inverse_link("probit", g), x, ["real_gdp_growth", "treasury_3m"], [0, 1], 1, "probit", q, q[0], q[-1]
    )
    cov = np.array([[1.0, 0.3], [0.3, 4.0]])
    y = np.cumsum(rng.normal(size=(60, 2)), axis=0)
    plan = capital.capital_plan(
        100.0, 11.0, 2.0, 3.0, (11.0, 11.5, 12.0), (1.0, 2.0, 1.5), (0.0, 1.0, 1.0), 0.145, 0.09625
    )
    target = capital.internal_capital_target(0.015, 0.00125, 0.025, 0.005, 0.01875)
    internal = {
        "credit": 6.0,
        "concentration_individual": None,
        "concentration_sectoral": None,
        "concentration_other": None,
        "irrbb": 0.5,
        "strategic": None,
        "reputational": None,
        "other": 0.25,
    }
    return [
        (satellites.link, ("probit", arr(0.1, 0.5))),
        (satellites.inverse_link, ("logit", arr(-1.0, 1.0))),
        (
            satellites.fit_satellite,
            (
                satellites.inverse_link("probit", g),
                x,
                ["real_gdp_growth", "treasury_3m"],
                [0, 1],
                1,
                "probit",
                q,
                q[0],
                q[-1],
            ),
        ),
        (satellites.project_scenario, (fit, arr(0.1), x[:1], x[1:5])),
        (paths.transpose_path, (arr(6, 8, 10), 5.0, 3.0, 3)),
        (paths.revert_to_baseline, (arr(8.0, 7.0), 5.0, 14, 12)),
        (paths.annual_to_quarterly, (arr(6, 8), 4.0)),
        (paths.level_from_deviation, (arr(1, 2), arr(0.1, 0.2), "percent")),
        (capital.cet1_path_loss_only, (11.0, arr(1, 2, 1.5))),
        (
            capital.capital_plan,
            (100.0, 11.0, 2.0, 3.0, arr(11, 11.5, 12), arr(1, 2, 1.5), arr(0, 1, 1), 0.145, 0.09625),
        ),
        (capital.p2g_proxy, (0.11, arr(0.10, 0.08, 0.065), 0.02)),
        (
            capital.annex3_summary,
            (
                {"cet1": 11.0, "at1": 1.25, "t2": 2.5},
                target,
                {"credit": 70.0, "market": 20.0, "operational": 10.0},
                internal,
                plan,
                True,
            ),
        ),
        (capital.eba_static_balance_sheet, (arr(80, 15, 5), arr(2, 3))),
        (reverse.reverse_stress_linear, (arr(1, 1), cov, 5.0)),
        (reverse.reverse_stress, (lambda v: float(v[0] + v[1]), cov, 5.0, [arr(3, 1), arr(-2, 6), arr(0.5, 0.5)])),
        (reverse.plausibility, (2.0, 2)),
        (scenario_models.fit_var, (y, 2)),
        (scenario_models.skewed_t_pdf, (arr(-1, 0, 1), 0.0, 1.0, 1.0, 5.0)),
        (scenario_models.fit_skewed_t_to_quantiles, (arr(-2.0, -0.7, 0.7, 2.0),)),
        (
            val.satellite_backcast,
            (
                arr(0.1, 0.5, 0.2, 0.1),
                arr(0.1, 0.4, 0.3, 0.1),
                ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"],
                "2019-12-31",
            ),
        ),
        (val.bottom_up_vs_top_down, (120.0, 100.0)),
        (val.scenario_severity, (3.0, 5.5)),
        (val.reverse_solution_check, (5.0, 5.0, 2.0, (2.0, 2.1, 2.2))),
        (val.static_balance_sheet_rules, (capital.eba_static_balance_sheet(arr(80, 15, 5), arr(2, 3)),)),
        (val.sensitivity_monotonicity, ((0.8, -0.1), ("positive", "negative"))),
    ]


def _same(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if hasattr(a, "__dataclass_fields__"):
        return type(a) is type(b) and all(_same(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=False))
    if isinstance(a, float) and a != a:
        return b != b
    return a == b


def test_inputs_are_not_mutated_and_results_are_deterministic():
    for fn, args in _calls():
        snapshot = copy.deepcopy(args)
        first = fn(*args)
        unchanged = all(_same(a, b) for a, b in zip(args, snapshot, strict=False) if not callable(a))
        assert unchanged, f"{fn.__name__} mutated an input"
        second = fn(*copy.deepcopy(snapshot))
        if hasattr(first, "to_dict"):
            assert first.to_dict() == second.to_dict(), fn.__name__
        else:
            assert _same(first, second), fn.__name__


def test_stress_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|DOI 10\.\d{4,}|https://", text), f"{name} cites no primary source"
        assert re.search(r"^## (Limits|What the engine does not claim)", text, re.M | re.I), f"{name} states no limits"
        assert "Golden" in text or "golden" in text, f"{name} gives no golden value"
    assert "engine-stress-api" in index
    assert "UNVERIFIED" in "".join((PAGES / n).read_text(encoding="utf-8") for n in NAMES)
    assert not [p for p in PAGES.glob("*.md") if p.name != "README.md" and p.name not in NAMES]
