"""The overfit gap and the weak spots (PiML "Overfit" and "WeakSpot"; dossier 06 C.7 l.1016-1018; I.1 l.1187-1188; ECB guide
paragraph 43(a); EBA recommendation (i)). Contract: ``engine-ml-api`` 1.0.0, module ``riskvalidation/ml/generalisation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.ml._checks import nonneg, same_rows, scalar, vec


@dataclass(frozen=True)
class OverfitResult:
    """The train-test gap in metric units, the tolerance and the flag ``gap > tolerance``."""

    gap: float
    tolerance: float
    flag: bool

    def __post_init__(self) -> None:
        if bool(self.flag) != bool(self.gap > self.tolerance):
            raise ValueError("flag must equal (gap > tolerance)")


def overfit_gap(metric_train: float, metric_test: float, tolerance: float,
                higher_is_better: bool = True) -> OverfitResult:
    """``gap = metric_train - metric_test`` when higher is better (AUC), ``metric_test - metric_train`` otherwise (Brier);
    the flag is raised when the gap exceeds the tolerance (a gap equal to it is not flagged)."""
    train, test = scalar(metric_train, "metric_train"), scalar(metric_test, "metric_test")
    tol = nonneg(tolerance, "tolerance")
    gap = train - test if higher_is_better else test - train
    return OverfitResult(gap=gap, tolerance=tol, flag=bool(gap > tol))


@dataclass(frozen=True)
class WeakSpot:
    """A slice (``label``) of ``n`` rows whose mean error is ``ratio`` times the overall mean error."""

    label: int
    n: int
    mean_error: float
    ratio: float


def weak_spots(error: np.ndarray, bins: np.ndarray, k: float, min_n: int = 30) -> list[WeakSpot]:
    """The slices whose mean error exceeds ``k`` times the overall mean error and that hold at least ``min_n`` rows, sorted
    by decreasing ratio (ties by label). ``error`` is the per-row loss (Brier terms) and ``bins`` the integer slice label
    of each row. A slice below ``min_n`` is never reported; an overall mean of zero leaves nothing to exceed."""
    e = vec(error, "error")
    b = np.asarray(bins)
    if b.ndim != 1 or not np.issubdtype(b.dtype, np.integer):
        raise ValueError("bins must be a one-dimensional array of integer slice labels")
    same_rows(error=e, bins=b)
    kk = scalar(k, "k")
    if kk <= 0.0:
        raise ValueError(f"k must be positive, got {kk}")
    if isinstance(min_n, bool) or int(min_n) != min_n or min_n < 1:
        raise ValueError(f"min_n must be an integer of at least 1, got {min_n}")
    overall = float(e.mean())
    if overall <= 0.0:
        return []
    spots = []
    for label in np.unique(b):
        m = b == label
        n = int(m.sum())
        if n < min_n:
            continue
        mean = float(e[m].mean())
        ratio = mean / overall
        if ratio > kk:
            spots.append(WeakSpot(label=int(label), n=n, mean_error=mean, ratio=ratio))
    return sorted(spots, key=lambda w: (-w.ratio, w.label))
