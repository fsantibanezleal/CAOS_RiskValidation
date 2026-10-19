"""The disparity that persists after the permitted controls (the C21 question, dossier 08 l.839; decision Q4): a
logistic decision model of the approval on the protected indicators and the controls, and a linear model of the rate
spread on the same terms, each with the Wald interval of one protected effect (contract `engine-fairness-api` 1.0.0).

The design of both models is, in this order, the intercept, the `q` protected indicator columns and the `c` control
columns; `coefficients[0]` is the intercept, `coefficients[1 + j]` the effect of protected indicator `j`, and
`protected_effect(model, index)` reads the coefficient at `index`. Coefficients of the decision model are log-odds,
those of the rate-spread model are in the units of the rate spread (percentage points). The columns are standardised
internally for conditioning and the results are returned in the original units. The model is an approximation of
comparative evidence: the controls the caller could not supply (public HMDA has no credit score) are the caller's
statement, never imputed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import expit

from riskvalidation.fairness._checks import binary, finite_matrix, unit_interval_open
from riskvalidation.lgd_ead._checks import vec

MAX_ITER = 100
STEP_TOL = 1e-11
DIVERGED = 50.0  # a standardised coefficient beyond this is complete or quasi-complete separation


@dataclass(frozen=True)
class DecisionModel:
    """Coefficients and standard errors (equal lengths; intercept first), the rows used and the convergence flag."""

    coefficients: tuple[float, ...]
    standard_errors: tuple[float, ...]
    n: int
    converged: bool

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(self.standard_errors):
            raise ValueError("coefficients and standard_errors must have the same length")


@dataclass(frozen=True)
class Effect:
    """One coefficient with its odds ratio exp(coefficient) and its Wald interval at `level`."""

    coefficient: float
    odds_ratio: float
    lo: float
    hi: float
    level: float


def _design(protected: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The standardised non-intercept columns, their means and standard deviations; a constant column is refused."""
    cols = np.hstack([protected, controls])
    names = [f"protected[{j}]" for j in range(protected.shape[1])]
    names += [f"controls[{j}]" for j in range(controls.shape[1])]
    mu, sd = cols.mean(axis=0), cols.std(axis=0)
    for name, s in zip(names, sd, strict=True):
        if not s > 0.0:
            raise ValueError(f"{name} is constant: the design matrix is rank deficient (collinear with the intercept)")
    z = (cols - mu) / sd
    x = np.hstack([np.ones((cols.shape[0], 1)), z])
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("the design matrix of protected and controls is rank deficient: a column is collinear")
    return x, mu, sd


def _to_original(beta: np.ndarray, cov: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and standard errors of the standardised design in the units of the original columns."""
    k = beta.size
    a = np.zeros((k, k))
    a[0, 0] = 1.0
    a[0, 1:] = -mu / sd
    a[1:, 1:] = np.diag(1.0 / sd)
    return a @ beta, np.sqrt(np.diag(a @ cov @ a.T))


def _failed(k: int, n: int) -> DecisionModel:
    nan = (math.nan,) * k
    return DecisionModel(nan, nan, n, False)


def _check_inputs(
    protected: np.ndarray, controls: np.ndarray, n: int, rows_of: str
) -> tuple[np.ndarray, np.ndarray]:
    p = finite_matrix(protected, "protected", rows=n, rows_of=rows_of)
    c = finite_matrix(controls, "controls", rows=n, rows_of=rows_of, allow_no_columns=True)
    return p, c


def fit_decision_model(decision: np.ndarray, protected: np.ndarray, controls: np.ndarray) -> DecisionModel:
    """Logistic model of the decision (1 favourable: approval by the C21 convention) on the intercept, the protected
    indicators (n, q) and the controls (n, c), by Newton's method (iteratively reweighted least squares) with step
    halving from the zero start. Converged False, with NaN coefficients and no exception, when the iteration does not
    settle in 100 steps or the classes are completely separated. Raises `ValueError` naming the field for a decision
    that is not 0 and 1 with both classes present, non-finite entries, rows that do not match and a rank-deficient
    design."""
    y = binary(decision, "decision")
    if y.min() == y.max():
        raise ValueError("decision must contain both 0 and 1")
    n = y.shape[0]
    prot, ctrl = _check_inputs(protected, controls, n, "decision")
    x, mu, sd = _design(prot, ctrl)
    k = x.shape[1]
    yf = y.astype(float)

    def loglik(b: np.ndarray) -> float:
        z = x @ b
        return float(np.sum(yf * z - np.logaddexp(0.0, z)))

    beta = np.zeros(k)
    ll = loglik(beta)
    converged = False
    for _ in range(MAX_ITER):
        mu_hat = expit(x @ beta)
        grad = x.T @ (yf - mu_hat)
        info = x.T @ (x * (mu_hat * (1.0 - mu_hat))[:, None])
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            return _failed(k, n)
        scale, cand, new = 1.0, beta, ll
        for _ in range(40):
            cand = beta + scale * step
            new = loglik(cand)
            if new >= ll - 1e-14:
                break
            scale *= 0.5
        else:
            return _failed(k, n)
        beta, ll = cand, new
        if not np.all(np.isfinite(beta)) or np.max(np.abs(beta[1:])) > DIVERGED:
            return _failed(k, n)
        if np.max(np.abs(scale * step)) < STEP_TOL:
            converged = True
            break
    if not converged:
        return _failed(k, n)
    mu_hat = expit(x @ beta)
    info = x.T @ (x * (mu_hat * (1.0 - mu_hat))[:, None])
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return _failed(k, n)
    coef, se = _to_original(beta, cov, mu, sd)
    if not (np.all(np.isfinite(coef)) and np.all(np.isfinite(se))):
        return _failed(k, n)
    return DecisionModel(tuple(float(v) for v in coef), tuple(float(v) for v in se), n, True)


def fit_rate_spread_model(rate_spread: np.ndarray, protected: np.ndarray, controls: np.ndarray) -> DecisionModel:
    """Linear least squares of the rate spread (percentage points, for the loans that carry it) on the same terms.
    Rows with a missing (NaN) rate spread are dropped and never imputed; the number of rows dropped is the rows given
    minus `n` of the result. Infinite values, a rank-deficient design, non-finite controls and fewer rows than
    coefficients plus one raise `ValueError` naming the field. Standard errors use the residual variance
    RSS / (n - k)."""
    rs = vec(rate_spread, "rate_spread", allow_nan=True)
    prot, ctrl = _check_inputs(protected, controls, rs.shape[0], "rate_spread")
    keep = ~np.isnan(rs)
    if not keep.any():
        raise ValueError("rate_spread has no non-missing rows")
    rs, prot, ctrl = rs[keep], prot[keep], ctrl[keep]
    n = rs.shape[0]
    x, mu, sd = _design(prot, ctrl)
    k = x.shape[1]
    if n <= k:
        raise ValueError(f"rate_spread has {n} usable rows for {k} coefficients: at least {k + 1} are needed")
    beta, *_ = np.linalg.lstsq(x, rs, rcond=None)
    resid = rs - x @ beta
    sigma2 = float(resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    coef, se = _to_original(beta, cov, mu, sd)
    return DecisionModel(tuple(float(v) for v in coef), tuple(float(v) for v in se), n, True)


def protected_effect(model: DecisionModel, index: int, level: float = 0.95) -> Effect:
    """The coefficient at `index` of the model, its odds ratio exp(coefficient) and the Wald interval coefficient plus
    or minus z se at `level` (z the normal quantile of 1 - (1 - level) / 2). The odds ratio is meaningful for the
    decision model only; for the linear rate-spread model read the coefficient."""
    k = len(model.coefficients)
    if not isinstance(index, (int, np.integer)) or not 0 <= index < k:
        raise ValueError(f"index must lie in [0, {k - 1}] (0 is the intercept), got {index!r}")
    lv = unit_interval_open(level, "level")
    coef, se = model.coefficients[index], model.standard_errors[index]
    if not (math.isfinite(coef) and math.isfinite(se) and se >= 0.0):
        raise ValueError(f"the standard error of coefficient {index} is not finite: the model did not converge")
    z = float(stats.norm.ppf(1.0 - (1.0 - lv) / 2.0))
    with np.errstate(over="ignore"):
        odds = float(np.exp(coef))
    return Effect(float(coef), odds, float(coef - z * se), float(coef + z * se), lv)


__all__ = ["DecisionModel", "Effect", "fit_decision_model", "fit_rate_spread_model", "protected_effect"]
