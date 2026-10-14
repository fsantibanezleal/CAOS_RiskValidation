"""Behavioural models: the static replicating portfolio of non-maturity deposits, the weighted average maturity, the
deposit beta with asymmetric pass-through, and the prepayment logit with its maximum-likelihood fit (dossier 05 D.3
and D.4).

Contract: ``engine-irrbb-api`` 1.1.0, module behavioural.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from riskvalidation.lgd_ead._checks import mat, same_length, scalar, vec

_ENUMERATION_LIMIT = 10
"""Up to this many tranches the replicating portfolio is the exact global optimum by enumeration of the supports."""


@dataclass(frozen=True)
class ReplicatingPortfolio:
    maturities: np.ndarray       # (J,) years
    weights: np.ndarray          # (J,)
    margin_volatility: float | None


def _solve_support(cov: np.ndarray, c: np.ndarray, support: list[int]) -> np.ndarray:
    """The minimiser of ``w'Cw - 2 w'c`` over weights supported on ``support`` and summing to one (KKT system)."""
    r = len(support)
    kkt = np.zeros((r + 1, r + 1))
    kkt[:r, :r] = 2.0 * cov[np.ix_(support, support)]
    kkt[:r, r] = 1.0
    kkt[r, :r] = 1.0
    sol = np.linalg.lstsq(kkt, np.r_[2.0 * c[support], 1.0], rcond=None)[0]
    w = np.zeros(cov.shape[0])
    w[support] = sol[:r]
    return w


def _objective(w: np.ndarray, cov: np.ndarray, c: np.ndarray) -> float:
    return float(w @ cov @ w - 2.0 * w @ c)


def replicating_portfolio(client_rate: np.ndarray, tranche_yields: np.ndarray,
                          maturities: np.ndarray) -> ReplicatingPortfolio:
    """The static replicating portfolio (dossier 05 D.3): the weights ``w_j >= 0``, ``sum w_j = 1`` that minimise the
    variance of the margin ``m_t = sum_j w_j y_jt - i_t``. Up to 10 tranches the answer is the global optimum by
    exhaustive search over the supports (each an equality-constrained quadratic problem solved exactly); above that a
    sequential quadratic programme polished on its support. ``margin_volatility`` is the sample standard deviation of
    the margin, or
    None with fewer than ``J + 2`` observations."""
    i = vec(client_rate, "client_rate")
    y = mat(tranche_yields, "tranche_yields")
    m = vec(maturities, "maturities")
    if y.shape[0] != i.size:
        raise ValueError(f"tranche_yields has {y.shape[0]} rows for {i.size} observations of the client rate")
    if y.shape[1] != m.size:
        raise ValueError(f"tranche_yields has {y.shape[1]} columns for {m.size} maturities")
    if np.any(m <= 0.0):
        raise ValueError("maturities must be positive")
    t, j = y.shape
    if t < 2:
        raise ValueError("at least 2 observations are needed")
    cov = np.cov(y, rowvar=False).reshape(j, j)
    c = np.array([np.cov(y[:, k], i)[0, 1] for k in range(j)])
    if j <= _ENUMERATION_LIMIT:
        best_val, best_w = np.inf, None
        for size in range(1, j + 1):
            for sup in itertools.combinations(range(j), size):
                w = _solve_support(cov, c, list(sup))
                if np.all(w >= -1e-12):
                    w = np.clip(w, 0.0, None)
                    w /= w.sum()
                    val = _objective(w, cov, c)
                    if val < best_val - 1e-18:
                        best_val, best_w = val, w
        assert best_w is not None
        w = best_w
    else:
        start = np.full(j, 1.0 / j)
        res = minimize(lambda x: _objective(x, cov, c), start, jac=lambda x: 2.0 * (cov @ x - c), method="SLSQP",
                       bounds=[(0.0, 1.0)] * j, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
                       options={"ftol": 1e-15, "maxiter": 500})
        w = np.clip(res.x, 0.0, None)
        w /= w.sum()
        sup = [k for k in range(j) if w[k] > 1e-7]
        polished = _solve_support(cov, c, sup)
        if np.all(polished >= -1e-12):
            polished = np.clip(polished, 0.0, None)
            polished /= polished.sum()
            if _objective(polished, cov, c) <= _objective(w, cov, c):
                w = polished
    vol = float(np.std(y @ w - i, ddof=1)) if t >= j + 2 else None
    return ReplicatingPortfolio(m.copy(), w, vol)


def weighted_average_maturity(maturities: np.ndarray, weights: np.ndarray) -> float:
    """``sum_j w_j m_j / 2``: a rolling tranche of maturity ``m`` reinvested evenly over its life has a weighted
    average maturity of ``m / 2`` (BCBS 368 footnote 38 as read in dossier 05 D.3). The weights must sum to one to
    1e-9."""
    m, w = vec(maturities, "maturities"), vec(weights, "weights")
    same_length(maturities=m, weights=w)
    if abs(float(w.sum()) - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to one, got {float(w.sum())}")
    return float(np.sum(w * m) / 2.0)


def deposit_beta_passthrough(client_rate: float, market_change: float, beta_up: float, beta_down: float) -> float:
    """The new client rate ``client_rate + beta * market_change`` with ``beta_up`` when ``market_change >= 0`` and
    ``beta_down`` otherwise (Drechsler, Savov and Schnabl 2021; dossier 05 D.3). Not floored: the caller floors."""
    r, dm = scalar(client_rate, "client_rate"), scalar(market_change, "market_change")
    up, down = scalar(beta_up, "beta_up"), scalar(beta_down, "beta_down")
    if up < 0.0 or down < 0.0:
        raise ValueError("betas must be nonnegative")
    return r + (up if dm >= 0.0 else down) * dm


def _coef(coef: np.ndarray) -> np.ndarray:
    c = vec(coef, "coef")
    if c.size != 3:
        raise ValueError("coef must hold the intercept, the incentive coefficient and the seasoning coefficient")
    return c


def prepayment_logit(incentive: np.ndarray, seasoning_months: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """The SMM ``1 / (1 + exp(-(b0 + b1 incentive + b2 seasoning)))`` (dossier 05 D.4; burnout and seasonality are not
    in this version). The incentive is the contract rate minus the market rate, in decimals."""
    x, s = vec(incentive, "incentive", allow_empty=True), vec(seasoning_months, "seasoning_months", allow_empty=True)
    same_length(incentive=x, seasoning_months=s)
    c = _coef(coef)
    return expit(c[0] + c[1] * x + c[2] * s)


@dataclass(frozen=True)
class LogitFit:
    coef: np.ndarray             # (3,); NaN when converged is False
    std_err: np.ndarray          # (3,) standard errors from the inverse observed information; NaN when not converged
    n: int
    converged: bool


def fit_prepayment_logit(incentive: np.ndarray, seasoning_months: np.ndarray, prepaid: np.ndarray) -> LogitFit:
    """Maximum likelihood by Newton's method with step halving from a fixed start (zero) and tolerance, on the
    standardised regressors (the incentive is in decimals and the seasoning in months). ``prepaid`` is a boolean or
    0/1 array. Not converged (``converged`` False, coefficients NaN, no exception) when a class is empty, a regressor
    is constant, the classes are completely separated (a standardised coefficient beyond 50) or the iteration does not
    settle in 100 steps; the bake then uses the PSA benchmark and says so. ``std_err`` are the standard errors from the
    inverse observed information at the solution, in the original units."""
    x1, x2 = vec(incentive, "incentive"), vec(seasoning_months, "seasoning_months")
    y_raw = np.asarray(prepaid)
    if y_raw.ndim != 1 or y_raw.size == 0 or not np.all(np.isin(y_raw, (0, 1))):
        raise ValueError("prepaid must be a non-empty one-dimensional boolean or 0/1 array")
    y = y_raw.astype(float)
    same_length(incentive=x1, seasoning_months=x2, prepaid=y)
    n = int(y.size)
    failed = LogitFit(np.full(3, np.nan), np.full(3, np.nan), n, False)
    sd = np.array([x1.std(), x2.std()])
    if y.sum() in (0.0, float(n)) or np.any(sd <= 1e-12 * np.maximum(1.0, np.abs([x1.mean(), x2.mean()]))):
        return failed
    mu = np.array([x1.mean(), x2.mean()])
    x = np.column_stack([np.ones(n), (x1 - mu[0]) / sd[0], (x2 - mu[1]) / sd[1]])

    def loglik(b: np.ndarray) -> float:
        z = x @ b
        return float(np.sum(y * z - np.logaddexp(0.0, z)))

    beta = np.zeros(3)
    ll = loglik(beta)
    converged = False
    for _ in range(100):
        p = expit(x @ beta)
        grad = x.T @ (y - p)
        hess = x.T @ (x * (p * (1.0 - p))[:, None])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return failed
        scale, new = 1.0, ll
        for _ in range(40):
            cand = beta + scale * step
            new = loglik(cand)
            if new >= ll - 1e-15:
                break
            scale *= 0.5
        else:
            return failed
        beta, ll = cand, new
        if np.max(np.abs(beta)) > 50.0:
            return failed
        if np.max(np.abs(scale * step)) < 1e-10:
            converged = True
            break
    if not converged:
        return failed
    # back to the original units: coef = A beta with A = [[1, -mu1 / sd1, -mu2 / sd2], [0, 1 / sd1, 0], [0, 0, 1 / sd2]]
    a = np.array([[1.0, -mu[0] / sd[0], -mu[1] / sd[1]], [0.0, 1.0 / sd[0], 0.0], [0.0, 0.0, 1.0 / sd[1]]])
    p = expit(x @ beta)
    info = x.T @ (x * (p * (1.0 - p))[:, None])
    try:
        cov = a @ np.linalg.inv(info) @ a.T
    except np.linalg.LinAlgError:
        return failed
    return LogitFit(a @ beta, np.sqrt(np.diag(cov)), n, True)
