"""The FZ0 and tick losses and the comparison tests (dossier 05 A.1 and B.6).

All forecasts are in the return convention (VaR and ES negative decimals, `e <= v`). The FZ0 loss is a score: it can be
negative and only differences between models are meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from riskvalidation.market.conventions import check_alpha, vec


@dataclass(frozen=True)
class Scores:
    n: int                       # days scored for the tick loss
    n_fz0: int                   # days with both VaR and ES defined
    fz0_mean: float | None       # None when n_fz0 = 0
    tick_mean: float


@dataclass(frozen=True)
class DmResult:
    mean_diff: float             # mean of loss_a - loss_b
    statistic: float
    p_value: float               # two-sided, normal
    n: int
    lags: int


@dataclass(frozen=True)
class McsResult:
    members: list[int]           # indices of the models in the set
    p_values: np.ndarray         # (M,) MCS p-values
    confidence: float
    statistic: str               # "Tmax" or "TR"


def _same_shape(**arrays: np.ndarray) -> None:
    shapes = {k: a.shape for k, a in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"the inputs must have the same shape, got {shapes}")


def fz0_loss(y: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float) -> np.ndarray:
    """`L = -(1/(alpha e)) 1{y <= v}(v - y) + v/e + log(-e) - 1` (dossier 05 A.1, arXiv 1707.05108 eq. 6), per day.

    The Fissler-Ziegel loss of eq. 4 with `G1 = 0`, `G2(x) = -1/x` and `calG2(x) = -log(-x)`. ValueError unless `v < 0`,
    `e < 0` and `e <= v` elementwise.
    """
    alpha = check_alpha(alpha)
    y, v, e = (np.asarray(a, dtype=float) for a in (y, v, e))
    _same_shape(y=y, v=v, e=e)
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(v)) and np.all(np.isfinite(e))):
        raise ValueError("y, v and e must be finite")
    if np.any(v >= 0.0) or np.any(e >= 0.0) or np.any(e > v):
        raise ValueError("the FZ0 loss needs v < 0, e < 0 and e <= v elementwise")
    hit = (y <= v).astype(float)
    return -(1.0 / (alpha * e)) * hit * (v - y) + v / e + np.log(-e) - 1.0


def tick_loss(y: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    """`(alpha - 1{y < v})(y - v)` per day (dossier 05 A.2 row 11): nonnegative, zero at `y = v`."""
    alpha = check_alpha(alpha)
    y, v = np.asarray(y, dtype=float), np.asarray(v, dtype=float)
    _same_shape(y=y, v=v)
    return (alpha - (y < v)) * (y - v)


def mean_scores(y: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float) -> Scores:
    """Mean tick loss over all `n` days and mean FZ0 over the days with a finite `e` (NaN where the ES is undefined)."""
    alpha = check_alpha(alpha)
    y, v = vec(y, "y"), vec(v, "v")
    e = np.array(e, dtype=float)
    _same_shape(y=y, v=v, e=e)
    ok = np.isfinite(e)
    n_fz0 = int(ok.sum())
    fz0 = float(np.mean(fz0_loss(y[ok], v[ok], e[ok], alpha))) if n_fz0 else None
    return Scores(n=int(y.size), n_fz0=n_fz0, fz0_mean=fz0, tick_mean=float(np.mean(tick_loss(y, v, alpha))))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int) -> DmResult:
    """Diebold-Mariano (1995): `d_t = loss_a_t - loss_b_t`, `DM = mean(d) / sqrt(LRV / T)` with the Newey-West
    (Bartlett) long-run variance `LRV = g0 + 2 sum_{j=1}^{lags} (1 - j/(lags+1)) g_j`, `g_j = (1/T) sum_t (d_t -
    mean)(d_{t-j} - mean)` (decision D-2) and `p_value = 2 (1 - Phi(|DM|))`. A negative `mean_diff` means model A has
    the lower loss. One-step forecasts use `lags = 0`. ValueError if `d` has no spread or `T <= lags + 1`."""
    a, b = vec(loss_a, "loss_a"), vec(loss_b, "loss_b")
    _same_shape(loss_a=a, loss_b=b)
    if int(lags) != lags or lags < 0:
        raise ValueError(f"lags must be a non-negative integer, got {lags}")
    d = a - b
    t = d.size
    if t <= lags + 1:
        raise ValueError(f"need more than lags + 1 = {lags + 1} observations, got {t}")
    mean = float(d.mean())
    c = d - mean
    if float(np.max(np.abs(c))) <= 1e-14 * max(1.0, abs(mean)):
        raise ValueError("the loss differential has no spread: the statistic is undefined")
    lrv = float(c @ c) / t
    for j in range(1, lags + 1):
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * float(c[j:] @ c[:-j]) / t
    if lrv <= 0.0:
        raise ValueError("the long-run variance is not positive: the statistic is undefined")
    z = mean / math.sqrt(lrv / t)
    return DmResult(mean_diff=mean, statistic=float(z), p_value=float(2.0 * stats.norm.sf(abs(z))), n=int(t),
                    lags=int(lags))


def model_confidence_set(losses: np.ndarray, confidence: float = 0.90, statistic: str = "TR", n_bootstrap: int = 1000,
                         block_length: int = 10, seed: int = 0) -> McsResult:
    """The Model Confidence Set of Hansen, Lunde and Nason (2011) through `arch.bootstrap.MCS` (stationary bootstrap).

    `losses` is `(T, M)`; a model is in the set iff its MCS p-value is at least `1 - confidence`; the set is never empty
    (the last survivor has p-value 1); `M = 1` gives that model with p-value 1. Same inputs and seed, same result."""
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if statistic not in ("Tmax", "TR"):
        raise ValueError(f"statistic must be 'Tmax' or 'TR', got {statistic!r}")
    x = np.array(losses, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"losses must be a (T, M) array with T >= 2, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("losses contain non-finite values")
    m = x.shape[1]
    if m == 1:
        return McsResult(members=[0], p_values=np.array([1.0]), confidence=float(confidence), statistic=statistic)
    try:
        import pandas as pd  # noqa: PLC0415
        from arch.bootstrap import MCS  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "model_confidence_set needs the optional package `arch`: pip install 'riskvalidation[market]'"
        ) from exc
    mcs = MCS(pd.DataFrame(x), size=1.0 - confidence, reps=int(n_bootstrap), block_size=int(block_length),
              method="R" if statistic == "TR" else "max", seed=int(seed))
    mcs.compute()
    pv = np.empty(m)
    for name, row in mcs.pvalues.iterrows():
        pv[int(name)] = float(row["Pvalue"])
    members = [i for i in range(m) if pv[i] >= 1.0 - confidence]
    return McsResult(members=members, p_values=pv, confidence=float(confidence), statistic=statistic)

__all__ = [
    "DmResult",
    "McsResult",
    "Scores",
    "diebold_mariano",
    "fz0_loss",
    "mean_scores",
    "model_confidence_set",
    "tick_loss",
]
