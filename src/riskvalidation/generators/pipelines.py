"""The two pipelines that feed the p-value tests of the ML family in their size and power studies (U16-E2 follow-up).

The tests ``ml.shift_resilience`` and ``ml.conformal_coverage`` read the outputs of engine functions, not raw arrays:
each pipeline below is that glue (the worst-subsample performance and its permutation reference for the shift test;
the split-conformal quantile and the empirical coverage for the conformal test), written once so that the study
registry (`riskvalidation.generators.size_studies`) and the suites share it.
"""

from __future__ import annotations

import numpy as np

from riskvalidation.ml import conformal, robustness
from riskvalidation.ml.plants import CovariateShiftPlant
from riskvalidation.validation import ml as v

SHIFT_PERMUTATIONS = 99  # R of the size study (the engine default is 199)
SHIFT_N = 2000  # development rows of the shift plant; the evaluation set is its first SHIFT_N // 2 rows (docs/ml/08)


def run_shift(plant: CovariateShiftPlant, share: float, shift_permutations: int = SHIFT_PERMUTATIONS):
    def run(X_dev, y_dev, X_eval, y_eval, seed):
        X_eval, y_eval = X_eval[: len(X_dev) // 2], y_eval[: len(X_dev) // 2]  # dev >= 2 x eval
        score = robustness.shift_score_rank(X_dev, X_eval, seed)
        res = robustness.worst_subsample_performance(plant.predict, X_eval, y_eval, score, share)
        ref = robustness.shift_development_reference(
            plant.predict, X_dev, y_dev, X_eval, y_eval, score, share, shift_permutations, seed
        )
        return v.shift_resilience("m", res, ref)

    return run


def run_conformal(n_alpha: float = 0.10, n_cal: int | None = None):
    """The pipeline of the conformal coverage test; `n_cal` keeps only the first `n_cal` calibration points (the plant
    draws ten times the test window), so the three split ratios of the size study share one generator."""

    def run(p_cal, y_cal, p_test, y_test):
        if n_cal is not None:
            p_cal, y_cal = p_cal[:n_cal], y_cal[:n_cal]
        scores = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
        q = conformal.split_conformal_quantile(scores, n_alpha)
        cov = conformal.conformal_coverage(conformal.conformal_sets(p_test, q), y_test, n_alpha, n_cal=len(p_cal))
        return v.conformal_coverage("m", cov)

    return run


__all__ = ["SHIFT_N", "SHIFT_PERMUTATIONS", "run_conformal", "run_shift"]
