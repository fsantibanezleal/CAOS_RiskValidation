"""The engine core's default logistic fit (NumPy and SciPy only) and the cross-fitted resemblance score built on it.

The fit is ridge-penalised Newton on standardised features with an unpenalised intercept and balanced class weights, so
that two samples that agree score 0.5 on average whatever their sizes. Two conventions are fixed here and recorded in
``docs/ml/01_robustness.md``: the ridge strength ``RIDGE`` (1, the default of scikit-learn's ``C = 1``) and the number
of cross-fitting folds ``FOLDS`` (5). Neither is a regulatory value.
"""

from __future__ import annotations

import numpy as np
from scipy import special

RIDGE = 1.0
FOLDS = 5
MAX_ITER = 50
GRAD_TOL = 1e-8


def fit_logistic(x: np.ndarray, y: np.ndarray, weight: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Coefficients (intercept first) minimising ``sum w (log(1 + e^z) - y z) + ridge / 2 |beta|^2``; the penalised
    objective is strictly convex, so damped Newton converges."""
    n, p = x.shape
    a = np.column_stack([np.ones(n), x])
    pen = np.r_[0.0, np.full(p, ridge)]
    beta = np.zeros(p + 1)

    def objective(b: np.ndarray) -> float:
        z = a @ b
        return float(np.sum(weight * (np.logaddexp(0.0, z) - y * z)) + 0.5 * np.sum(pen * b * b))

    f = objective(beta)
    for _ in range(MAX_ITER):
        mu = special.expit(a @ beta)
        grad = a.T @ (weight * (mu - y)) + pen * beta
        if np.max(np.abs(grad)) <= GRAD_TOL:
            break
        hess = (a * (weight * mu * (1.0 - mu))[:, None]).T @ a + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-12:
            cand = beta - t * step
            fc = objective(cand)
            if fc <= f:
                break
            t *= 0.5
        else:
            break
        beta, f = cand, fc
    return beta


def predict_proba(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return special.expit(beta[0] + x @ beta[1:])


def crossfit_resemblance(x_a: np.ndarray, x_b: np.ndarray, seed: int, folds: int = FOLDS) -> np.ndarray:
    """Out-of-fold probability of membership in sample ``b`` for every row of ``a`` stacked on ``b`` (length n_a + n_b).

    Rows with an identical feature vector form one group and a group is never split across folds: a row whose exact twin
    sits in the other sample would otherwise be scored by a model that saw the twin with the opposite label, which
    pushes the AUC of two identical samples below 0.5. The groups are dealt to the folds by a seeded permutation; the
    features are standardised on the training folds and the classes carry the weight n / (2 n_class)."""
    x = np.vstack([x_a, x_b])
    y = np.r_[np.zeros(len(x_a)), np.ones(len(x_b))]
    n = len(y)
    rng = np.random.default_rng(seed)
    _, group = np.unique(x + 0.0, axis=0, return_inverse=True)
    group = group.reshape(-1)
    fold = (rng.permutation(group.max() + 1) % folds)[group]
    out = np.empty(n)
    for k in range(folds):
        tr, te = fold != k, fold == k
        mean = x[tr].mean(axis=0)
        sd = x[tr].std(axis=0)
        sd[sd == 0.0] = 1.0
        if y[tr].min() == y[tr].max():
            raise ValueError("a training fold holds a single class: the samples have too few distinct rows")
        w = np.where(y[tr] == 1.0, n / (2.0 * y[tr].sum()), n / (2.0 * (len(y[tr]) - y[tr].sum())))
        beta = fit_logistic((x[tr] - mean) / sd, y[tr], w)
        out[te] = predict_proba(beta, (x[te] - mean) / sd)
    return out
