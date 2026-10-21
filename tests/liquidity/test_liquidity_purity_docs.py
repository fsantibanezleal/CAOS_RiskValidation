"""RV-1128 and RV-1129: the liquidity modules import with only NumPy and SciPy, do not mutate their inputs and are
deterministic; the documentation pages carry their equations, sources and limits."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "liquidity"
NAMES = ("01_runoff.md", "02_counterbalancing-capacity.md", "03_ladders.md", "04_scenarios.md", "05_reverse-stress.md",
         "06_depositor-generator.md", "07_lcr-and-nsfr.md", "08_validation-tests.md")
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib", "QuantLib")


def test_modules_import_no_dataframe_library():
    code = (
        "import sys, riskvalidation.liquidity._checks, riskvalidation.liquidity.runoff, riskvalidation.liquidity.cbc,"
        " riskvalidation.liquidity.ladder, riskvalidation.liquidity.scenarios, riskvalidation.liquidity.reverse,"
        " riskvalidation.liquidity.deposits, riskvalidation.regulatory.lcr_nsfr, riskvalidation.validation.liquidity;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.liquidity import cbc, deposits, ladder, reverse, runoff, scenarios
    from riskvalidation.regulatory import lcr_nsfr as lcr
    from riskvalidation.validation import liquidity as val

    def arr(*x):
        return np.array(x, dtype=float)

    fast = scenarios.fast_run_scenario([(0.01, 1)])
    scenario = scenarios.compose_scenario("idiosyncratic", "protracted", fast,
                                          {"d": (scenarios.FAST_RUN_CLASS, 1000.0)}, 125)
    depositors = deposits.DepositorGenerator(deposits.DepositorConfig(
        (deposits.DepositorSegment("retail-uninsured", 1.0, 0.001, 0.01),), 50, 10, 0.2, 0.3))
    nlp40 = np.zeros(40)
    nlp40[29], nlp40[34] = 100.0, 80.0
    pts = [(1, 0.5), (2, 1.0)]
    return [
        (runoff.cumulative_outflow_constant, (arr(0.01, 0.02), arr(3, 4))),
        (runoff.cumulative_outflow_path, (arr(0.01, 0.02, 0.03),)),
        (runoff.daily_rate_from_cumulative, (arr(0.1, 0.5), arr(2, 3))),
        (runoff.contractual_rolloff, (arr(100, 50), 0.18)),
        (runoff.financial_no_rollover, (arr(100, 0, 25),)),
        (cbc.haircut_rule, ("equity",), {"equity_rating": "AAA-A-"}),
        (cbc.liquidity_value, (arr(100, 50), arr(0.1, 0.2), arr(20, 0))),
        (cbc.monetisation_capacity, (100.0, pts, 4)),
        (cbc.cbc_stack, ({"A": (100.0, 20.0, 0.1)}, pts, 3)),
        (cbc.cbc_with_htm_loss, (100.0, 85.0, 20.0, 0.1)),
        (ladder.contractual_gaps, (arr(10, 20), arr(5, 30))),
        (ladder.behavioural_outflows, (arr(100, 50), np.array([[0.1, 0.1], [0.2, 0.3]]))),
        (ladder.net_liquidity_position, (arr(100, 100), arr(30, 40), arr(0, 10))),
        (ladder.survival_period, (arr(5, -1),)),
        (ladder.minimum_nlp, (arr(5, -1),)),
        (ladder.cliff_effect, (nlp40, 1000.0)),
        (scenarios.list_scenario, ("adverse",)),
        (scenarios.compose_scenario, ("combined", "acute", scenarios.list_scenario("extreme"),
                                      {"r": ("other-retail", 10.0)}, 4)),
        (scenarios.fast_run_scenario, ([(0.2, 4), (0.1, 10)],)),
        (scenarios.anchor_from_observation, (0.85, 2)),
        (scenarios.uninsured_share_grid, (40.0, 60.0, [0.5, 0.9])),
        (scenarios.htm_loss_grid, (-15.0, [0.0, 1.0])),
        (reverse.survival_vs_multiplier, (scenario, np.full(125, 100.0), arr(1, 2, 3))),
        (reverse.solve_multiplier, (scenario, np.full(125, 100.0), 5)),
        (depositors.draw, (np.random.default_rng(5),)),
        (depositors.truth, ()),
        (lcr.hqla_stock, (100.0, 20.0, 30.0)),
        (lcr.lcr_outflows, ({"retail_stable": 1000.0},)),
        (lcr.lcr_inflows, ({"secured_lending_level2a": 100.0},)),
        (lcr.net_cash_outflows, (100.0, 90.0)),
        (lcr.lcr_ratio, (100.0, 50.0)),
        (lcr.nsfr_asf, ({"regulatory_capital": 10.0},)),
        (lcr.nsfr_rsf, ({"unencumbered_level_2b": 100.0},)),
        (lcr.nsfr_ratio, (105.0, 50.0)),
        (val.list_compounding_golden, ()),
        (val.survival_monotone_in_severity, ([None, 176, 122],)),
        (val.cbc_haircut_rules, ()),
        (val.nlp_identity, (arr(100, 100), arr(30, 40), arr(0, 10), arr(70, 40))),
        (val.runoff_adequacy_back_test, (0.5, 0.85)),
        (val.runoff_vs_generator_truth, (arr(1, 2), arr(1, 2), arr(0.1, 0.1))),
        (val.cliff_effect_diagnostic, (nlp40, 1000.0)),
        (val.htm_loss_effect, (72.0, 58.5, None, 48)),
        (val.uninsured_share_sensitivity, ([0.5, 0.9], [None, 67])),
        (val.reverse_stress_round_trip, (2.085164, 5, 5)),
        (val.htm_revaluation_vs_reported, (81.0, 80.0, 100.0)),
    ]


def _same(a, b):
    if isinstance(a, np.random.Generator):  # the stream is advanced by a call; determinism is judged on the result
        return isinstance(b, np.random.Generator)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if hasattr(a, "__dataclass_fields__"):
        return type(a) is type(b) and all(_same(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__)
    if hasattr(a, "to_dict") and hasattr(b, "to_dict"):
        return _same(a.to_dict(), b.to_dict())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and a != a:
        return b != b
    return a == b


def test_inputs_are_not_mutated_and_results_are_deterministic():
    for entry in _calls():
        fn, args = entry[0], entry[1]
        kwargs = entry[2] if len(entry) > 2 else {}
        before = copy.deepcopy(args)
        # bound methods of a generator and a fresh Generator per call: the seed is the only state
        first = fn(*copy.deepcopy(args), **kwargs)
        second = fn(*copy.deepcopy(args), **kwargs)
        assert _same(first, second), fn.__qualname__
        fn(*args, **kwargs)
        assert _same(args, before), f"{fn.__qualname__} mutated an input"


def test_liquidity_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"({name})" in index, f"README does not link {name}"
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"https://\S+", text), f"{name} cites no source with a URL"
        assert re.search(r"^## .*Limits", text, re.M), f"{name} has no Limits section"
        assert re.search(r"[Gg]olden", text), f"{name} states no golden value"
        assert "Sources" in text, f"{name} has no sources"
    # every item the contract marks UNVERIFIED is named where it applies
    must_say = {
        "01_runoff.md": ("TARGET2", "UNVERIFIED"),
        "02_counterbalancing-capacity.md": ("UNVERIFIED", "pro rata"),
        "03_ladders.md": ("UNVERIFIED", "cliff"),
        "04_scenarios.md": ("UNVERIFIED", "collateral"),
        "06_depositor-generator.md": ("UNVERIFIED", "convention"),
        "07_lcr-and-nsfr.md": ("never the bank's reported", "no official"),
        "08_validation-tests.md": ("Contraste internal", "conventions"),
    }
    for name, words in must_say.items():
        text = (PAGES / name).read_text(encoding="utf-8")
        for w in words:
            assert w.lower() in text.lower(), f"{name} does not say {w!r}"
    # the formulas of the pages are the engine's: the one key equation of each part appears
    runoff_page = (PAGES / "01_runoff.md").read_text(encoding="utf-8")
    assert "1-(1-q)^n" in re.sub(r"[ {}]", "", runoff_page)
    assert "NLP(t)" in (PAGES / "03_ladders.md").read_text(encoding="utf-8")
    assert "Adj_{15" in (PAGES / "07_lcr-and-nsfr.md").read_text(encoding="utf-8")
