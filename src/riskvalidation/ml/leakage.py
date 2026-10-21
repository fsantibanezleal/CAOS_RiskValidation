"""The leakage screen: temporal order, single-feature AUC, train and test entity overlap and the resemblance AUC between
development and out-of-time samples (dossier 06 C.7 l.1048-1051; I.1 l.1197; dossier 07 D pattern 3, l.597-600).

These are conventions: no regulatory text prescribes them (06 C.7). The AUC is the engine's own (``disc_auc`` and
``auc_with_variance``). Contract: ``engine-ml-api`` 1.1.0, module ``riskvalidation/ml/leakage.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from riskvalidation.ml import _logistic
from riskvalidation.ml._checks import binary, mat, same_rows, scalar
from riskvalidation.validation.discrimination import auc_with_variance, disc_auc


def temporal_order_violations(feature_dates: np.ndarray, observation_date: np.ndarray) -> int:
    """The number of feature values dated after the observation date of their row. ``feature_dates`` has shape (n, p)
    and ``observation_date`` shape (n,), both ``datetime64``. NaT is never a violation; a date equal to the observation
    date is not one either (06 C.7 l.1048-1051: no feature timestamped after the observation date)."""
    f, o = np.asarray(feature_dates), np.asarray(observation_date)
    if f.ndim != 2 or f.dtype.kind != "M":
        raise ValueError("feature_dates must be a two-dimensional datetime64 array of shape (n, p)")
    if o.ndim != 1 or o.dtype.kind != "M":
        raise ValueError("observation_date must be a one-dimensional datetime64 array of shape (n,)")
    if o.shape[0] != f.shape[0]:
        raise ValueError(f"observation_date must have {f.shape[0]} rows, got {o.shape[0]}")
    late = (f > o[:, None]) & ~np.isnat(f) & ~np.isnat(o)[:, None]
    return int(late.sum())


@dataclass(frozen=True)
class SingleFeatureAuc:
    """The largest ``max(AUC, 1 - AUC)`` over columns, its column and whether it exceeds the near-perfect threshold."""

    max_auc: float
    feature: int
    suspect: bool

    def __post_init__(self) -> None:
        if self.max_auc < 0.5 or self.max_auc > 1.0:
            raise ValueError(f"max_auc must lie in [0.5, 1], got {self.max_auc}")


def single_feature_auc(X: np.ndarray, y: np.ndarray, threshold: float = 0.95) -> SingleFeatureAuc:
    """The largest ``max(AUC, 1 - AUC)`` over the columns of ``X`` (``disc_auc`` of each column against ``y``), its
    column (the first on a tie) and whether it exceeds ``threshold`` strictly: a lone near-perfect feature is a leakage
    suspect (a convention, 06 C.7 l.1048-1051)."""
    x = mat(X, "X")
    yy = binary(y, "y")
    same_rows(X=x, y=yy)
    t = scalar(threshold, "threshold")
    if not 0.5 <= t <= 1.0:
        raise ValueError(f"threshold must lie in [0.5, 1], got {t}")
    best = [max(a, 1.0 - a) for a in (float(disc_auc(x[:, j], yy).metric) for j in range(x.shape[1]))]
    j = int(np.argmax(best))
    return SingleFeatureAuc(max_auc=best[j], feature=j, suspect=bool(best[j] > t))


def entity_overlap(train_ids: Sequence[str], test_ids: Sequence[str]) -> float:
    """The share of distinct test entities present in training, in [0, 1]; identifiers are compared as strings. An empty
    test set has no share and raises."""
    test = {str(i) for i in test_ids}
    if not test:
        raise ValueError("test_ids is empty: the overlap share is undefined")
    train = {str(i) for i in train_ids}
    return len(test & train) / len(test)


@dataclass(frozen=True)
class AucResult:
    """An AUC with its variance over ``n`` rows."""

    auc: float
    variance: float
    n: int

    def __post_init__(self) -> None:
        if self.variance < 0.0:
            raise ValueError(f"variance must be nonnegative, got {self.variance}")


def resemblance_auc(X_dev: np.ndarray, X_oot: np.ndarray, seed: int) -> AucResult:
    """The cross-fitted AUC, with its ECB variance (``auc_with_variance``), of a classifier separating development rows
    (class 0) from out-of-time rows (class 1): near 0.5 when the samples agree (probatus ``sample_similarity``, dossier
    06 C.7 l.1048-1051; dossier 07 D pattern 3). The classifier is the engine core's ridge logistic fit, scored on folds
    that did not train on the row, deterministic for a ``seed``."""
    dev, oot = mat(X_dev, "X_dev"), mat(X_oot, "X_oot")
    if dev.shape[1] != oot.shape[1]:
        raise ValueError(f"X_oot must have the {dev.shape[1]} columns of X_dev, got {oot.shape[1]}")
    if min(len(dev), len(oot)) < _logistic.FOLDS:
        raise ValueError(f"X_dev and X_oot need at least {_logistic.FOLDS} rows each")
    prob = _logistic.crossfit_resemblance(dev, oot, seed)
    label = np.r_[np.zeros(len(dev), dtype=np.int64), np.ones(len(oot), dtype=np.int64)]
    auc, var, _, _ = auc_with_variance(prob, label)
    return AucResult(auc=auc, variance=var, n=len(label))

__all__ = [
    "AucResult",
    "SingleFeatureAuc",
    "entity_overlap",
    "resemblance_auc",
    "single_feature_auc",
    "temporal_order_violations",
]
