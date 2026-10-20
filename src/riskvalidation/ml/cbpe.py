"""Label-free performance estimation, confidence-based (NannyML CBPE; dossier 06 C.7 l.1052-1058; I.1 l.1199).

For a calibrated probability ``p_j`` and the predicted label ``yhat_j``, ``P(yhat = y)_j = 1 - |yhat_j - p_j|``; the
expected true positives and true negatives are summed over observations, and the ROC AUC is estimated by repeating the
confusion over thresholds (https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html).
Assumptions: well-calibrated probabilities, no concept drift, no extrapolation into unseen regions, adequate sample
size. An early-warning estimate of the AUC while defaults mature: it complements, never replaces, back-testing, and it
is always flagged as an estimate. Contract: ``engine-ml-api`` 1.1.0, module ``riskvalidation/ml/cbpe.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.ml._checks import closed_unit, index, scalar, vec

ASSUMPTIONS = ("calibrated-probabilities", "no-concept-drift", "no-extrapolation", "adequate-sample")


@dataclass(frozen=True)
class CbpeConfusion:
    """The expected true positives, false positives, true negatives and false negatives (they sum to n)."""

    tp: float
    fp: float
    tn: float
    fn: float


def _probabilities(p: np.ndarray) -> np.ndarray:
    a = vec(p, "p")
    closed_unit(a, "p")
    return a


def _confusion(p: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    yhat = p >= threshold
    # P(yhat = y) = 1 - |yhat - p|: p when the label is predicted 1, 1 - p when it is predicted 0
    return (
        float(np.sum(p[yhat])),
        float(np.sum(1.0 - p[yhat])),
        float(np.sum(1.0 - p[~yhat])),
        float(np.sum(p[~yhat])),
    )


def cbpe_confusion(p: np.ndarray, threshold: float) -> CbpeConfusion:
    """The expected confusion counts of calibrated PDs ``p`` at ``threshold`` (``yhat = 1`` when ``p >= threshold``):
    ``tp = sum p`` and ``fp = sum (1 - p)`` over the predicted positives, ``tn = sum (1 - p)`` and ``fn = sum p`` over
    the predicted negatives."""
    a = _probabilities(p)
    t = scalar(threshold, "threshold")
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {t}")
    return CbpeConfusion(*_confusion(a, t))


@dataclass(frozen=True)
class CbpeAuc:
    """The estimated AUC, always labelled ``estimate``, with its four assumptions."""

    estimated_auc: float
    label: str
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.label != "estimate":
            raise ValueError("label must be 'estimate': a label-free AUC is never a measurement")
        if len(self.assumptions) != 4:
            raise ValueError("assumptions must name the four assumptions of the estimate")


def cbpe_auc(p: np.ndarray, n_thresholds: int = 101) -> CbpeAuc:
    """The ROC AUC estimated without labels: the expected confusion at ``n_thresholds`` thresholds evenly spaced on [0,
    1] gives the expected rates ``tpr = TP / (TP + FN)`` and ``fpr = FP / (FP + TN)``, and the AUC is the trapezoidal
    area under them (the curve runs from (1, 1) at threshold 0 to (0, 0) above the largest probability). No label is
    read."""
    a = _probabilities(p)
    m = index(n_thresholds, "n_thresholds", 10**9, lo=2)
    if a.sum() <= 0.0 or (1.0 - a).sum() <= 0.0:
        raise ValueError("p holds no expected positive or no expected negative: the ROC is undefined")
    tpr, fpr = np.empty(m), np.empty(m)
    for i, t in enumerate(np.linspace(0.0, 1.0, m)):
        tp, fp, tn, fn = _confusion(a, float(t))
        tpr[i], fpr[i] = tp / (tp + fn), fp / (fp + tn)
    order = np.argsort(fpr, kind="stable")
    x, y = fpr[order], tpr[order]
    auc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return CbpeAuc(estimated_auc=min(1.0, max(0.0, auc)), label="estimate", assumptions=ASSUMPTIONS)
