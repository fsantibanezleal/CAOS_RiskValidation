"""RV-1479 and RV-1480: the fairness modules import with only NumPy and SciPy, do not mutate their inputs and are
deterministic; the documentation pages carry their equations, sources and limits."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "fairness"
NAMES = ("01_metrics.md", "02_inference.md", "03_conditional-and-replication.md", "04_situation-testing.md",
         "05_planted-bias.md", "06_validation-tests.md")
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib", "fairlearn", "shap",
         "QuantLib")
TEST_IDS = ("fair.demographic_parity", "fair.adverse_impact", "fair.two_proportion_z", "fair.fisher_exact",
            "fair.equalized_odds", "fair.equal_opportunity", "fair.calibration_within_groups", "fair.auc_by_group",
            "fair.conditional_disparity", "fair.rate_spread_disparity", "fair.replication", "fair.situation_testing")


def test_modules_import_no_dataframe_or_learning_library():
    code = (
        "import sys, riskvalidation.fairness.metrics, riskvalidation.fairness.inference,"
        " riskvalidation.fairness.conditional, riskvalidation.fairness.replication, riskvalidation.fairness.situation,"
        " riskvalidation.fairness.plants, riskvalidation.fairness.policy, riskvalidation.validation.fairness;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.fairness import conditional, inference, metrics, plants, replication, situation
    from riskvalidation.validation import fairness as fv

    rng = np.random.default_rng(0)
    n = 300
    group = np.where(rng.random(n) < 0.4, "a", "b")
    y = (rng.random(n) < 0.5).astype(int)
    decision = (rng.random(n) < 0.6).astype(int)
    score = rng.normal(size=n) + y
    pd = rng.uniform(0.1, 0.9, n)
    protected = (group == "a").astype(float)[:, None]
    controls = rng.normal(size=(n, 2))
    spread = controls[:, 0] * 0.3 + rng.normal(size=n)
    rates = metrics.group_selection_rates(decision, group)
    model = conditional.fit_decision_model(decision, protected, controls)
    eff = conditional.protected_effect(model, 1)
    rep = replication.replicate_effect(eff, eff)
    xs = rng.uniform(0, 10, (n, 2))
    g01 = (group == "a").astype(int)
    res = situation.situation_test(xs, g01, decision, k=5)
    cf = xs + 0.1
    return [
        (metrics.group_selection_rates, (decision, group)),
        (metrics.demographic_parity_difference, (rates, "b")),
        (metrics.demographic_parity_ratio, (rates, "b")),
        (metrics.adverse_impact_ratio, (rates, "b")),
        (metrics.four_fifths_flag, (rates,)),
        (metrics.equalized_odds_difference, (y, decision, group)),
        (metrics.equal_opportunity_difference, (y, decision, group)),
        (metrics.calibration_within_groups, (pd, y, group)),
        (metrics.auc_by_group, (score, y, group)),
        (inference.two_proportion_z_test, (30, 100, 45, 120)),
        (inference.fisher_exact_test, (30, 100, 45, 120)),
        (conditional.fit_decision_model, (decision, protected, controls)),
        (conditional.protected_effect, (model, 1)),
        (conditional.fit_rate_spread_model, (spread, protected, controls)),
        (replication.replicate_effect, (eff, eff)),
        (situation.gower_distance, (xs[0], xs[1], np.array([False, False]), np.array([10.0, 10.0]))),
        (situation.k_nearest, (xs[0], xs[1:], 5, np.array([False, False]), np.array([10.0, 10.0]))),
        (situation.situation_statistic, (decision[:10], decision[10:20])),
        (situation.wald_interval_one_sided, (0.5, 0.5, 0.0, 2)),
        (situation.situation_test, (xs, g01, decision, 5)),
        (situation.situation_test, (xs, g01, decision, 5, 0.0, 0.05, "mutatis-mutandis", cf, "standard")),
        (situation.bonferroni_alpha, (0.05, 3)),
        (lambda: plants.ThresholdBiasPlant(200, 0.3, 1).draw(np.random.default_rng(3)), ()),
        (lambda: plants.LabelNoiseBiasPlant(200, 0.2, 1).draw(np.random.default_rng(3)), ()),
        (lambda: plants.ProxyFeaturePlant(200, 0.5, 1).draw(np.random.default_rng(3)), ()),
        (fv.demographic_parity, ("m", rates, "b")),
        (fv.adverse_impact, ("m", rates, "b")),
        (fv.two_proportion_z, ("m", 30, 100, 45, 120)),
        (fv.fisher_exact, ("m", 30, 100, 45, 120)),
        (fv.equalized_odds, ("m", y, decision, group)),
        (fv.equal_opportunity, ("m", y, decision, group)),
        (fv.calibration_within_groups, ("m", pd, y, group)),
        (fv.auc_by_group, ("m", score, y, group)),
        (fv.conditional_disparity, ("m", eff, ["a"], ["b"])),
        (fv.rate_spread_disparity, ("m", eff, ["a"], ["b"])),
        (fv.replication, ("m", rep)),
        (fv.situation_testing, ("m", res)),
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
    if isinstance(r1, (tuple, list)):
        return len(r1) == len(r2) and all(_same(p, q) for p, q in zip(r1, r2, strict=True))
    return r1 == r2


def test_inputs_are_not_mutated_and_results_are_deterministic():
    for fn, args in _calls():
        before = copy.deepcopy(args)
        first = fn(*args)
        second = fn(*args)
        for b, aft in zip(before, args, strict=True):
            if isinstance(b, np.ndarray):
                same = np.array_equal(b, aft, equal_nan=True) if b.dtype.kind == "f" else np.array_equal(b, aft)
                assert same, f"{getattr(fn, '__name__', fn)} mutated an input"
        assert _same(first, second), f"{getattr(fn, '__name__', fn)} is not deterministic"


def test_fairness_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv:\d{4}\.\d{4,5}|https://", text), f"{name} cites no source"
        assert "Symbols" in text or "symbols" in text, f"{name} defines no symbols"
        assert "UNVERIFIED" in text or "Limits" in text or "limits" in text, f"{name} states no limit"
        assert "Golden" in text or "golden" in text, f"{name} states no golden value"
        assert "no legal claim" in text.lower() or "states no legal claim" in text.lower() or "legal" in text.lower()
    metrics = (PAGES / "01_metrics.md").read_text(encoding="utf-8")
    assert "employment benchmark" in metrics and "29 CFR 1607.4(D)" in metrics
    assert "5 x_g n_h < 4 x_h n_g" in metrics or "5 x_g n_h" in metrics
    assert "0.7999999999999999" in metrics  # the float ratio the exact rule is built to avoid
    situation = (PAGES / "04_situation-testing.md").read_text(encoding="utf-8")
    for needle in ("0.58154", "-0.0815", "range", "standard", "arXiv:2502.01267", "arXiv:2405.13693", "55", "44"):
        assert needle in situation, needle
    validation = (PAGES / "06_validation-tests.md").read_text(encoding="utf-8")
    for test_id in TEST_IDS:
        assert f"`{test_id}`" in validation, test_id
    assert "4,000" in validation and "Monte Carlo" in validation
    for page in (PAGES / n for n in NAMES):
        assert chr(0x2014) not in page.read_text(encoding="utf-8"), f"{page.name} holds an em-dash"
