"""Robustness to input perturbation and resilience to distribution shift (dossier 06 C.7 l.1009-1015; I.1 l.1189-1190).

Perturbation (PiML "Robustness"; SR 11-7 sensitivity analysis): Gaussian noise of scale eps times each feature's
standard deviation is added to the **evaluation** features only, never to the training slice, and AUC and Brier are read
against eps. Resilience (PiML "Resilience"): performance on the worst subsample by a shift score (the worst 10 percent
is the dossier's example) and on out-of-time windows. A model enters only as a predict callable on arrays; the engine
imports no learning library. Contract: ``engine-ml-api`` 1.0.0, module ``riskvalidation/ml/robustness.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from riskvalidation.ml import _logistic
from riskvalidation.ml._checks import binary, decimal, mat, nonneg, predictions, same_rows, scalar, vec
from riskvalidation.validation.discrimination import auc_with_variance

Predict = Callable[[np.ndarray], np.ndarray]


def perturb_features(X: np.ndarray, eps: float, seed: int, sd: np.ndarray | None = None) -> np.ndarray:
    """``X`` plus Gaussian noise of standard deviation ``eps * sd[j]`` in column j, a new array of shape (n, p).

    ``eps`` is a multiple of each feature's standard deviation (dimensionless); ``sd`` has shape (p,) in feature units
    and defaults to the sample standard deviation (ddof 1) of the columns of ``X``. ``eps = 0`` returns ``X`` unchanged.
    The noise is a function of ``seed`` alone: the same seed gives the same standard normal draws at every ``eps``, so
    the perturbed samples of a grid are nested."""
    x = mat(X, "X")
    e = nonneg(eps, "eps")
    if sd is None:
        if x.shape[0] < 2:
            raise ValueError("X needs at least two rows to estimate sd")
        s = x.std(axis=0, ddof=1)
    else:
        s = vec(sd, "sd")
        if s.shape != (x.shape[1],):
            raise ValueError(f"sd must have shape ({x.shape[1]},), got {s.shape}")
        if np.any(s < 0.0):
            raise ValueError("sd must be nonnegative")
    if e == 0.0:
        return x.copy()
    z = np.random.default_rng(seed).standard_normal(x.shape)
    return x + z * (e * s)


@dataclass(frozen=True)
class RobustnessCurve:
    """AUC and Brier per noise scale and the AUC delta against scale 0 (``metric(eps) - metric(0)``).

    Tuples of equal length with ``grid[0] == 0.0``; AUC and Brier in [0, 1] and ``auc_delta`` in [-1, 1]. A value that
    is NaN (a metric that could not be computed) is carried, never replaced."""

    grid: tuple[float, ...]
    auc: tuple[float, ...]
    brier: tuple[float, ...]
    auc_delta: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("grid", "auc", "brier", "auc_delta"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        if len({len(self.grid), len(self.auc), len(self.brier), len(self.auc_delta)}) != 1 or not self.grid:
            raise ValueError("grid, auc, brier and auc_delta must be non-empty tuples of equal length")
        if self.grid[0] != 0.0:
            raise ValueError("grid must start at scale 0.0")
        for name, lo, hi in (("auc", 0.0, 1.0), ("brier", 0.0, 1.0), ("auc_delta", -1.0, 1.0)):
            if any(v < lo or v > hi for v in getattr(self, name)):
                raise ValueError(f"{name} must lie in [{lo}, {hi}]")


def _metrics(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return auc_with_variance(p, y)[0], float(np.mean((p - y) ** 2))


def robustness_curve(
    predict: Predict, X: np.ndarray, y: np.ndarray, grid: Sequence[float], seed: int
) -> RobustnessCurve:
    """AUC and Brier of ``predict`` on ``X`` perturbed at each scale of ``grid`` (the noise of ``perturb_features`` with
    the same seed at every scale), and the AUC delta against scale 0.

    The first point is always scale 0 (prepended when ``grid`` lacks it; a 0 inside ``grid`` is not repeated), so the
    deltas are ``metric(eps) - metric(0)``. Both classes must be present, and ``predict`` must map (n, p) to
    probabilities of shape (n,)."""
    x = mat(X, "X")
    yy = binary(y, "y")
    same_rows(X=x, y=yy)
    g = vec(grid, "grid")
    if np.any(g < 0.0):
        raise ValueError("grid must hold nonnegative noise scales")
    scales = [0.0] + [float(s) for s in g if s != 0.0]
    auc0, brier0 = _metrics(predictions(predict, x), yy)
    auc, brier = [auc0], [brier0]
    for eps in scales[1:]:
        a, b = _metrics(predictions(predict, perturb_features(x, eps, seed)), yy)
        auc.append(a)
        brier.append(b)
    return RobustnessCurve(
        grid=tuple(scales), auc=tuple(auc), brier=tuple(brier), auc_delta=tuple(a - auc0 for a in auc)
    )


def robustness_finding(curve: RobustnessCurve, tolerance: float, preset_eps: float) -> bool:
    """True exactly when the AUC loss ``-auc_delta`` at ``preset_eps`` exceeds ``tolerance`` (a loss equal to the
    tolerance is not a finding). ``preset_eps`` must be a point of the grid."""
    tol = nonneg(tolerance, "tolerance")
    eps = scalar(preset_eps, "preset_eps")
    hits = [i for i, g in enumerate(curve.grid) if math.isclose(g, eps, rel_tol=0.0, abs_tol=1e-12)]
    if not hits:
        raise ValueError(f"preset_eps {eps} is not a point of the grid {curve.grid}")
    return bool(-curve.auc_delta[hits[0]] > tol)


def shift_score_rank(X_dev: np.ndarray, X_eval: np.ndarray, seed: int) -> np.ndarray:
    """The shift score of each evaluation row, shape (n_eval,) in [0, 1]: the cross-fitted probability that a
    resemblance classifier (the engine core's ridge logistic fit, ``riskvalidation.ml._logistic``) gives to
    "evaluation".

    Development rows are class 0 and evaluation rows class 1 with balanced weights, so two samples that agree score 0.5
    on average; the probability of each row comes from folds that did not train on it. Deterministic for a ``seed``."""
    dev, ev = mat(X_dev, "X_dev"), mat(X_eval, "X_eval")
    if dev.shape[1] != ev.shape[1]:
        raise ValueError(f"X_eval must have the {dev.shape[1]} columns of X_dev, got {ev.shape[1]}")
    if min(len(dev), len(ev)) < _logistic.FOLDS:
        raise ValueError(f"X_dev and X_eval need at least {_logistic.FOLDS} rows each")
    return _logistic.crossfit_resemblance(dev, ev, seed)[len(dev) :]


@dataclass(frozen=True)
class SubsampleResult:
    """AUC and Brier on the worst subsample (``n`` rows) beside the full-sample metrics."""

    n: int
    auc: float
    brier: float
    auc_full: float
    brier_full: float

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or int(self.n) != self.n or self.n < 1:
            raise ValueError(f"n must be an integer of at least 1, got {self.n}")
        for name in ("auc", "brier", "auc_full", "brier_full"):
            v = getattr(self, name)
            if v < 0.0 or v > 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


def worst_subsample_performance(
    predict: Predict, X: np.ndarray, y: np.ndarray, score: np.ndarray, share: float
) -> SubsampleResult:
    """AUC and Brier on the ``ceil(share * n)`` rows with the highest shift ``score`` (ties broken by row order), beside
    the full-sample metrics. The size is computed on the exact decimal ``share`` (0.1 * 30 is 3.0000000000000004 in
    floats). ``share = 1`` returns the full-sample metrics exactly. A subsample holding a single class has no AUC: it
    raises."""
    x = mat(X, "X")
    yy = binary(y, "y")
    s = vec(score, "score")
    same_rows(X=x, y=yy, score=s)
    sh = decimal(share, "share")
    if not 0 < sh <= 1:
        raise ValueError(f"share must lie in (0, 1], got {share}")
    n = len(yy)
    m = max(1, math.ceil(sh * n))
    p = predictions(predict, x)
    auc_full, brier_full = _metrics(p, yy)
    if m == n:
        return SubsampleResult(n=n, auc=auc_full, brier=brier_full, auc_full=auc_full, brier_full=brier_full)
    idx = np.argsort(-s, kind="stable")[:m]
    if yy[idx].min() == yy[idx].max():
        raise ValueError(f"the worst subsample of {m} rows holds a single class: no AUC (raise share)")
    auc, brier = _metrics(p[idx], yy[idx])
    return SubsampleResult(n=m, auc=auc, brier=brier, auc_full=auc_full, brier_full=brier_full)


def oot_window_performance(predict: Predict, X: np.ndarray, y: np.ndarray, window: np.ndarray) -> dict[str, float]:
    """The AUC of each out-of-time window, keyed by its integer label as a string, plus the key ``omitted``: the number
    of windows left out because they hold a single class (a window with fewer than two classes never gets a number)."""
    x = mat(X, "X")
    yy = binary(y, "y", both=False)
    w = np.asarray(window)
    if w.ndim != 1 or not np.issubdtype(w.dtype, np.integer):
        raise ValueError("window must be a one-dimensional array of integer labels")
    same_rows(X=x, y=yy, window=w)
    p = predictions(predict, x)
    out: dict[str, float] = {}
    omitted = 0
    for label in np.unique(w):
        m = w == label
        if yy[m].min() == yy[m].max():
            omitted += 1
            continue
        out[str(int(label))] = auc_with_variance(p[m], yy[m])[0]
    out["omitted"] = float(omitted)
    return out
