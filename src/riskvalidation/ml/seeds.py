"""Seed stability: the dispersion of the AUC, of the grade assigned to each obligor and of the attribution ranks across
retraining seeds (dossier 06 I.1 l.1191; ECB guide paragraph 43 on the randomness of the training process and paragraph
59 on stored seeds, A.3 l.290-299 and B.9 l.647). The training function is a callable the pipeline supplies. Contract:
``engine-ml-api`` 1.1.0, module ``riskvalidation/ml/seeds.py``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from riskvalidation.ml._checks import binary, mat, predictions, same_rows
from riskvalidation.validation.discrimination import auc_with_variance

Predict = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SeedDispersion:
    """The AUC across ``n_seeds`` retraining seeds: mean, sample standard deviation (ddof 1), minimum and maximum."""

    n_seeds: int
    mean: float
    sd: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.sd < 0.0:
            raise ValueError(f"sd must be nonnegative, got {self.sd}")
        if self.minimum > self.mean or self.mean > self.maximum:
            raise ValueError("minimum must not exceed mean, nor mean the maximum")


def seed_auc_dispersion(
    train: Callable[[int], Predict], X: np.ndarray, y: np.ndarray, seeds: Sequence[int]
) -> SeedDispersion:
    """Mean, sample standard deviation (ddof 1), minimum and maximum of the AUC of ``train(seed)`` on ``(X, y)`` over at
    least two distinct seeds. Pure given ``train``; a train function that ignores its seed has a zero
    standard deviation."""
    x = mat(X, "X")
    yy = binary(y, "y")
    same_rows(X=x, y=yy)
    ids = [int(s) for s in seeds]
    if any(isinstance(s, bool) for s in seeds):
        raise ValueError("seeds must be integers")
    if len(ids) < 2:
        raise ValueError("seeds must hold at least two seeds")
    if len(set(ids)) != len(ids):
        raise ValueError("seeds must be distinct")
    aucs = np.array([auc_with_variance(predictions(train(s), x), yy)[0] for s in ids])
    lo, hi = float(aucs.min()), float(aucs.max())
    mean = min(max(float(aucs.mean()), lo), hi)  # a mean can sit one ulp outside identical values
    sd = math.sqrt(float(np.sum((aucs - mean) ** 2)) / (len(aucs) - 1))
    return SeedDispersion(n_seeds=len(ids), mean=mean, sd=sd, minimum=lo, maximum=hi)


def grade_assignment_stability(pd_by_seed: np.ndarray, edges: np.ndarray) -> float:
    """The share of obligors whose grade differs between at least two seeds, in [0, 1].

    ``pd_by_seed`` has shape (n_seeds, n) with PDs in (0, 1); ``edges`` are the g - 1 strictly increasing PD cut points.
    The grade of a PD is the number of edges at or below it (the bins are [e_k, e_k+1): an edge value belongs to the
    upper grade). An obligor counts once; one seed returns 0."""
    pd = np.array(pd_by_seed, dtype=float)
    if pd.ndim != 2 or pd.size == 0:
        raise ValueError("pd_by_seed must be a non-empty two-dimensional array of shape (n_seeds, n)")
    if not np.all(np.isfinite(pd)) or np.any(pd <= 0.0) or np.any(pd >= 1.0):
        raise ValueError("pd_by_seed must lie strictly between 0 and 1")
    e = np.array(edges, dtype=float)
    if e.ndim != 1 or e.size == 0 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0.0):
        raise ValueError("edges must be a non-empty, strictly increasing vector of finite PD cut points")
    grade = np.searchsorted(e, pd, side="right")
    return float(np.mean(grade.min(axis=0) != grade.max(axis=0)))


@dataclass(frozen=True)
class RankStability:
    """The mean and the minimum pairwise Kendall tau-b of the feature ranks across seeds, in [-1, 1]."""

    mean_tau: float
    min_tau: float

    def __post_init__(self) -> None:
        if self.min_tau > self.mean_tau:
            raise ValueError("min_tau must not exceed mean_tau")


def attribution_rank_stability(attributions_by_seed: np.ndarray) -> RankStability:
    """The mean and the minimum, over seed pairs, of ``scipy.stats.kendalltau`` (tau-b, so ties are handled) between the
    per-feature mean absolute attributions of two seeds. ``attributions_by_seed`` has shape (n_seeds, p), n_seeds at
    least two; a constant row has no rank and raises."""
    a = np.array(attributions_by_seed, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError("attributions_by_seed must have shape (n_seeds, p) with at least two seeds and two features")
    if not np.all(np.isfinite(a)):
        raise ValueError("attributions_by_seed contains non-finite values")
    if np.any(a.max(axis=1) == a.min(axis=1)):
        raise ValueError("attributions_by_seed has a constant row: its ranks are undefined")
    taus = np.array([stats.kendalltau(a[i], a[j]).statistic for i, j in itertools.combinations(range(len(a)), 2)])
    lo, hi = float(taus.min()), float(taus.max())
    return RankStability(mean_tau=min(max(float(taus.mean()), lo), hi), min_tau=lo)
