"""RV-1436 and RV-1438: the ML modules import with only NumPy and SciPy, inputs are not mutated, results are
deterministic; the documentation pages carry their formulas, sources and limits (`docs/ml/`)."""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "ml"
NAMES = ("01_robustness.md", "02_seeds-and-explanations.md", "03_effects.md", "04_generalisation.md", "05_conformal.md",
         "06_leakage-and-data-quality.md", "07_cbpe-and-challenger.md", "08_size-and-power.md", "09_validation-tests.md")
HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib", "shap", "mapie", "interpret")
MODULES = ("robustness", "seeds", "explanations", "effects", "generalisation", "conformal", "leakage", "dataquality",
           "cbpe", "challenger", "plants", "policy")


def test_modules_import_no_learning_or_dataframe_library():
    code = (
        "import sys, " + ", ".join(f"riskvalidation.ml.{m}" for m in MODULES) + ", riskvalidation.validation.ml;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def _calls():
    from riskvalidation.ml import (
        cbpe,
        challenger,
        conformal,
        dataquality,
        effects,
        explanations,
        generalisation,
        leakage,
        robustness,
        seeds,
    )

    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 3))
    y = (rng.random(200) < 1.0 / (1.0 + np.exp(-x[:, 0]))).astype(int)
    p = rng.beta(2.0, 5.0, 200)
    edges = np.array([0.1, 0.3])

    def predict(a):
        return 1.0 / (1.0 + np.exp(-a[:, 0]))

    def train(seed):
        noise = np.random.default_rng(seed).standard_normal(200)
        return lambda a: 1.0 / (1.0 + np.exp(-(a[:, 0] + noise)))

    attr = rng.standard_normal((3, 50, 6))
    dates = np.array(["2020-01-01"] * 200, dtype="datetime64[D]")
    return [
        (robustness.perturb_features, (x, 0.3, 1)),
        (robustness.robustness_curve, (predict, x, y, [0.0, 0.5], 2)),
        (robustness.shift_score_rank, (x, x + 0.2, 3)),
        (robustness.worst_subsample_performance, (predict, x, y, rng.random(200), 0.5)),
        (robustness.oot_window_performance, (predict, x, y, np.repeat([1, 2], 100))),
        (seeds.seed_auc_dispersion, (train, x, y, [1, 2, 3])),
        (seeds.grade_assignment_stability, (np.vstack([p, p[::-1]]), edges)),
        (seeds.attribution_rank_stability, (rng.random((3, 5)),)),
        (explanations.reason_code_overlap, (attr, 3)),
        (explanations.rank_agreement, (rng.random(6), rng.random(6))),
        (explanations.shapley_additivity_gap, (rng.random(50), 0.1, attr[0])),
        (explanations.sign_check, (rng.random(5), 1)),
        (explanations.monotonicity_check, (rng.random(5), 1, 0.01)),
        (effects.partial_dependence, (predict, x, 0, np.linspace(-1, 1, 5))),
        (effects.ice_curves, (predict, x, 0, np.linspace(-1, 1, 5))),
        (effects.accumulated_local_effects, (predict, x, 0, 5)),
        (generalisation.overfit_gap, (0.9, 0.8, 0.05)),
        (generalisation.weak_spots, (rng.random(200), np.arange(200) % 4, 1.2)),
        (conformal.split_conformal_quantile, (rng.random(99), 0.1)),
        (conformal.conformal_sets, (p, 0.5)),
        (conformal.conformal_coverage, (conformal.conformal_sets(p, 0.5), y, 0.1)),
        (leakage.temporal_order_violations, (np.repeat(dates[:, None], 2, axis=1), dates)),
        (leakage.single_feature_auc, (x, y)),
        (leakage.entity_overlap, (["a", "b"], ["b", "c"])),
        (leakage.resemblance_auc, (x, x + 0.1, 4)),
        (dataquality.dq_completeness, (x,)),
        (dataquality.dq_validity, (x, np.full(3, -5.0), np.full(3, 5.0))),
        (dataquality.dq_label_consistency, (x, y)),
        (cbpe.cbpe_confusion, (p, 0.2)),
        (cbpe.cbpe_auc, (p,)),
        (challenger.paired_brier_bootstrap, (p, p[::-1].copy(), y, 100, 5)),
        (challenger.grade_disagreement, (p, p[::-1].copy(), edges)),
    ]


def test_inputs_are_not_mutated_and_results_are_deterministic():
    for fn, args in _calls():
        before = copy.deepcopy([a for a in args if isinstance(a, np.ndarray)])
        first = fn(*args)
        second = fn(*args)
        after = [a for a in args if isinstance(a, np.ndarray)]
        for b, a in zip(before, after, strict=True):
            np.testing.assert_array_equal(a, b, err_msg=f"{fn.__name__} mutated an input")
        np.testing.assert_equal(first, second)  # frozen dataclasses, arrays and floats compare exactly


def test_ml_pages_carry_formulas_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    assert "engine-ml-api" in index
    for name in NAMES:
        text = (PAGES / name).read_text(encoding="utf-8")
        assert f"]({name})" in index, f"{name} is not linked from the index"
        assert "$$" in text or "$" in text, f"{name} has no formula"
        assert re.search(r"doi:10\.\d{4,}|arXiv \d{4}\.\d{4,5}|https?://", text), f"{name} cites no source"
        assert "## Limits" in text or "## What these" in text, f"{name} states no limits"
        assert "dossier" in text, f"{name} does not name the dossier it transcribes"
    text = "\n".join((PAGES / n).read_text(encoding="utf-8") for n in NAMES)
    for item in ("MAPIE", "UNVERIFIED", "ceil((n + 1)", "numpy.sort(scores)[k - 1]", "perturbation_tolerance"):
        assert item in text, item
    assert "no compliance claim" in text.lower() or "makes no compliance claim" in text.lower()
