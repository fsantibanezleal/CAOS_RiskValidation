"""Partial dependence (Friedman 2001), individual conditional expectation (Goldstein et al. 2015) and accumulated local
effects (Apley and Zhu 2020, the estimator centred to mean zero) on a predict callable (dossier 06 C.7 l.994-1003).

Contract: ``engine-ml-api`` 1.1.0, module ``riskvalidation/ml/effects.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from riskvalidation.ml._checks import index, mat, predictions, vec

Predict = Callable[[np.ndarray], np.ndarray]


def ice_curves(predict: Predict, X: np.ndarray, feature: int, grid: np.ndarray) -> np.ndarray:
    """One curve per row, shape (n, m): the prediction of row i with ``feature`` set to each value of ``grid`` and the
    other columns at their observed values (Goldstein, Kapelner, Bleich and Pitkin 2015,
    doi:10.1080/10618600.2014.907095)."""
    x = mat(X, "X")
    j = index(feature, "feature", x.shape[1])
    g = vec(grid, "grid")
    out = np.empty((x.shape[0], g.size))
    for k, value in enumerate(g):
        xv = x.copy()
        xv[:, j] = value
        out[:, k] = predictions(predict, xv, unit=False)
    return out


def partial_dependence(predict: Predict, X: np.ndarray, feature: int, grid: np.ndarray) -> np.ndarray:
    """The average prediction at each grid value, shape (m,): ``(1/n) sum_i f(x_S, x_i,C)`` (Friedman 2001,
    doi:10.1214/aos/1013203451). It is the mean of ``ice_curves`` over rows, exactly."""
    return ice_curves(predict, X, feature, grid).mean(axis=0)


def accumulated_local_effects(
    predict: Predict, X: np.ndarray, feature: int, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """The Apley and Zhu (2020, doi:10.1111/rssb.12377) accumulated local effect of ``feature``, centred to mean zero.

    Returns the ``n_bins + 1`` quantile bin edges (the ``inverted_cdf`` sample quantiles of the feature, so every edge
    is a data value and every bin (z_{k-1}, z_k] holds at least one row; the first bin also holds the minimum) and the
    effect at the upper edge of each bin: ``g_k = sum_{l <= k} mean over the rows of bin l of [f(z_l, x_-j) - f(z_{l-1},
    x_-j)]`` minus the count-weighted mean of ``g``, so the count-weighted mean of the effect is 0 within 1e-12. A
    feature whose quantile edges coincide (ties that would empty a bin) raises."""
    x = mat(X, "X")
    j = index(feature, "feature", x.shape[1])
    nb = index(n_bins, "n_bins", x.shape[0] + 1, lo=1)
    col = x[:, j]
    edges = np.quantile(col, np.linspace(0.0, 1.0, nb + 1), method="inverted_cdf")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError(
            f"n_bins={nb} gives coinciding quantile edges for feature {j}: fewer distinct values than bins"
        )
    which = np.searchsorted(edges[1:-1], col, side="left")
    counts = np.bincount(which, minlength=nb)
    local = np.empty(nb)
    for k in range(nb):
        rows = x[which == k]
        upper, lower = rows.copy(), rows.copy()
        upper[:, j], lower[:, j] = edges[k + 1], edges[k]
        local[k] = np.mean(predictions(predict, upper, unit=False) - predictions(predict, lower, unit=False))
    accumulated = np.cumsum(local)
    return edges, accumulated - float(np.sum(counts * accumulated) / x.shape[0])

__all__ = [
    "accumulated_local_effects",
    "ice_curves",
    "partial_dependence",
]
