"""The LGD rungs of dossier 04 F.3: OLS (clipped at prediction), fractional logit, beta regression, hurdle, Tobit,
and the
cure decomposition. Pure NumPy and SciPy; every fit is deterministic (a fixed start, optimiser and tolerance).

A design matrix X (N, K) includes its intercept column; no function adds one. Contract: `engine-lgd-ead-api` 1.0.0,
module lgd_models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, special, stats

from riskvalidation.lgd_ead._checks import closed_unit, mat, same_length, vec

KINDS = ("ols", "fractional-logit", "beta", "hurdle", "tobit")
_MAX_ABS_LINEAR = 40.0   # a linear predictor beyond this has saturated the logistic: no finite maximiser (separation)
_SCORE_TOL = 1e-11


@dataclass(frozen=True)
class LgdFit:
    kind: str                  # "ols", "fractional-logit", "beta", "hurdle", "tobit"
    names: tuple[str, ...]     # one per column of X
    coefficients: np.ndarray   # (K,) of the mean model (for a hurdle: the severity stage)
    # beta: phi, epsilon; hurdle: stage1_coefficients, severity_columns; tobit: sigma
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- shared checks and solvers


def _design(X: Any, y: Any, names: tuple[str, ...] | None) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    x = mat(X, "X")
    yy = vec(y, "y")
    if x.shape[0] != yy.shape[0]:
        raise ValueError(f"X and y must have the same number of rows, got {x.shape[0]} and {yy.shape[0]}")
    return x, yy, _names(x, names)


def _names(x: np.ndarray, names: tuple[str, ...] | None) -> tuple[str, ...]:
    if names is None:
        return tuple(f"x{i}" for i in range(x.shape[1]))
    if len(names) != x.shape[1]:
        raise ValueError(f"names must have one entry per column of X ({x.shape[1]}), got {len(names)}")
    return tuple(str(n) for n in names)


def _check_rank(x: np.ndarray) -> None:
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        raise ValueError(f"X is rank deficient (rank {rank} < {x.shape[1]} columns)")


def _check_unit_response(y: np.ndarray) -> None:
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("y must lie in [0, 1]")


def lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least squares coefficients; a rank-deficient design raises."""
    _check_rank(x)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def logit_quasi_mle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The Bernoulli quasi-maximum-likelihood estimator of Papke and Wooldridge (1996) by Newton iterations with step
    halving, from a zero start: maximises sum(y x'b - log(1 + exp(x'b))). The score X'(y - mu) is below 1e-11 at
    the end."""
    _check_rank(x)
    b = np.zeros(x.shape[1])

    def ll(beta: np.ndarray) -> float:
        z = x @ beta
        return float(np.sum(y * z - np.logaddexp(0.0, z)))

    cur = ll(b)
    for _ in range(200):
        mu = special.expit(x @ b)
        score = x.T @ (y - mu)
        if np.max(np.abs(score)) < _SCORE_TOL:
            return b
        w = mu * (1.0 - mu)
        hess = (x * w[:, None]).T @ x
        try:
            step = np.linalg.solve(hess, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the fractional logit did not converge (singular information matrix)") from exc
        scale = 1.0
        while scale > 1e-12:
            cand = b + scale * step
            new = ll(cand)
            if new >= cur - 1e-14:
                break
            scale *= 0.5
        b, cur = cand, new
        if np.max(np.abs(x @ b)) > _MAX_ABS_LINEAR:
            # judged on the linear predictor x'b, not on the coefficients, so that the unit of a covariate (a rate in
            # decimals or in percent) does not decide whether a legitimate fit is refused (review F2)
            raise ValueError("the fractional logit did not converge (the linear predictor diverges: perfect "
                             "separation or a group at the endpoints)")
    raise ValueError("the fractional logit did not converge within 200 iterations")


# ---------------------------------------------------------------- the rungs


def endpoint_transform(y: np.ndarray, epsilon: float) -> np.ndarray:
    """min(max(y, epsilon), 1 - epsilon): the small adjustment the beta rungs need at the endpoints and to which the
    results are very sensitive (Qi and Zhao 2011, dossier 04 F.3). The epsilon used is recorded by the caller."""
    _check_epsilon(epsilon)
    v = vec(y, "y")
    closed_unit(v, "y")
    return np.minimum(np.maximum(v, epsilon), 1.0 - epsilon)


def _check_epsilon(epsilon: float) -> None:
    eps = float(epsilon)
    if not (0.0 < eps < 0.5):
        raise ValueError(f"epsilon must lie in (0, 0.5), got {epsilon!r}")


def fit_ols(X: np.ndarray, y: np.ndarray, names: tuple[str, ...] | None = None) -> LgdFit:
    """The baseline of dossier 04 F.3: least squares of y on X; the clipping to [0, 1] happens in `predict_lgd`."""
    x, yy, nm = _design(X, y, names)
    _check_unit_response(yy)
    return LgdFit(kind="ols", names=nm, coefficients=lstsq(x, yy), extras={})


def fit_fractional_logit(X: np.ndarray, y: np.ndarray, names: tuple[str, ...] | None = None) -> LgdFit:
    """Fractional logit, E[y | x] = 1 / (1 + exp(-x b)), by the Bernoulli quasi-ML of Papke and Wooldridge (1996),
    consistent for the endpoints (dossier 04 F.3)."""
    x, yy, nm = _design(X, y, names)
    _check_unit_response(yy)
    return LgdFit(kind="fractional-logit", names=nm, coefficients=logit_quasi_mle(x, yy), extras={})


def fit_beta(X: np.ndarray, y: np.ndarray, epsilon: float, names: tuple[str, ...] | None = None) -> LgdFit:
    """Beta regression with a logit mean link and a log precision link (Ferrari and Cribari-Neto 2004) on
    `endpoint_transform(y, epsilon)`, by maximum likelihood (BFGS, analytic gradient, fixed start). The precision is
    constant, so `extras["phi"]` is the precision at the mean of X."""
    _check_epsilon(epsilon)
    x, yy, nm = _design(X, y, names)
    _check_unit_response(yy)
    _check_rank(x)
    yt = np.minimum(np.maximum(yy, epsilon), 1.0 - epsilon)
    ly, l1y = np.log(yt), np.log1p(-yt)
    k = x.shape[1]
    b0 = np.linalg.lstsq(x, special.logit(yt), rcond=None)[0]
    mu0 = special.expit(x @ b0)
    resid = float(np.mean((yt - mu0) ** 2))
    phi0 = max(float(np.mean(mu0 * (1.0 - mu0))) / max(resid, 1e-12) - 1.0, 0.5)

    def negll(theta: np.ndarray) -> tuple[float, np.ndarray]:
        mu = special.expit(x @ theta[:k])
        phi = float(np.exp(theta[k]))
        a, b = mu * phi, (1.0 - mu) * phi
        ll = (special.gammaln(phi) - special.gammaln(a) - special.gammaln(b) + (a - 1.0) * ly + (b - 1.0) * l1y).sum()
        dmu = phi * (special.digamma(b) - special.digamma(a) + ly - l1y)
        g_beta = x.T @ (dmu * mu * (1.0 - mu))
        dphi = (special.digamma(phi) - mu * special.digamma(a) - (1.0 - mu) * special.digamma(b) + mu * ly
                + (1.0 - mu) * l1y).sum()
        return -float(ll), -np.append(g_beta, dphi * phi)

    res = optimize.minimize(negll, np.append(b0, np.log(phi0)), jac=True, method="BFGS",
                            options={"gtol": 1e-9, "maxiter": 500})
    if not np.all(np.isfinite(res.x)) or float(np.max(np.abs(res.jac))) > 1e-4 * max(1.0, x.shape[0]):
        raise ValueError("the beta regression did not converge")
    return LgdFit(kind="beta", names=nm, coefficients=res.x[:k].copy(),
                  extras={"phi": float(np.exp(res.x[k])), "epsilon": float(epsilon)})


def fit_hurdle(X: np.ndarray, y: np.ndarray, X_severity: np.ndarray | None = None,
               names: tuple[str, ...] | None = None, *, min_positive: int = 5) -> LgdFit:
    """Two-stage model (Leow and Mues 2012; Tong, Mues and Thomas 2013): stage 1 is a logit of y > 0 on X, stage 2 a
    fractional logit of y on `X_severity` (default X) among the exposures with y > 0.

    `coefficients` are the stage 2 ones, `extras["stage1_coefficients"]` the stage 1 ones. A separate severity
    design must
    be a set of columns of X (matched by value); their indices are kept in `extras["severity_columns"]` so that
    `predict_lgd` needs only X."""
    x, yy, nm = _design(X, y, names)
    _check_unit_response(yy)
    pos = yy > 0.0
    n_pos = int(pos.sum())
    if n_pos < int(min_positive):
        raise ValueError(f"fewer than min_positive={min_positive} exposures with a positive loss ({n_pos})")
    if n_pos == yy.size:
        raise ValueError("every y is positive: the hurdle stage has no zero losses to model")
    if X_severity is None:
        cols = tuple(range(x.shape[1]))
        xs = x
    else:
        xs = mat(X_severity, "X_severity")
        if xs.shape[0] != x.shape[0]:
            raise ValueError("X_severity must have the same number of rows as X")
        found = []
        for j in range(xs.shape[1]):
            match = [i for i in range(x.shape[1]) if np.array_equal(xs[:, j], x[:, i])]
            if not match:
                raise ValueError(f"column {j} of X_severity is not a column of X: the severity design must be a set "
                                 "of columns of X so that predict_lgd needs only X")
            found.append(match[0])
        cols = tuple(found)
    stage1 = logit_quasi_mle(x, pos.astype(float))
    stage2 = logit_quasi_mle(xs[pos], yy[pos])
    return LgdFit(kind="hurdle", names=nm, coefficients=stage2,
                  extras={"stage1_coefficients": stage1, "severity_columns": cols})


def _tobit_negll(theta: np.ndarray, x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 mid: np.ndarray) -> tuple[float, np.ndarray]:
    k = x.shape[1]
    b, s = theta[:k], float(np.exp(theta[k]))
    xb = x @ b
    ll = 0.0
    g_b = np.zeros(k)
    g_s = 0.0
    if mid.any():
        z = (y[mid] - xb[mid]) / s
        ll += float(np.sum(-np.log(s) - 0.5 * z * z - 0.5 * np.log(2.0 * np.pi)))
        g_b += x[mid].T @ (z / s)
        g_s += float(np.sum(-1.0 + z * z))
    if lo.any():
        a = (0.0 - xb[lo]) / s
        ll += float(np.sum(stats.norm.logcdf(a)))
        lam = np.exp(stats.norm.logpdf(a) - stats.norm.logcdf(a))
        g_b += x[lo].T @ (-lam / s)
        g_s += float(np.sum(-lam * a))
    if hi.any():
        q = (1.0 - xb[hi]) / s
        ll += float(np.sum(stats.norm.logsf(q)))
        lam = np.exp(stats.norm.logpdf(q) - stats.norm.logsf(q))
        g_b += x[hi].T @ (lam / s)
        g_s += float(np.sum(lam * q))
    return -ll, -np.append(g_b, g_s)


def fit_tobit(X: np.ndarray, y: np.ndarray, names: tuple[str, ...] | None = None) -> LgdFit:
    """Two-limit censored normal, y = min(1, max(0, y*)), y* = x b + e, e ~ N(0, sigma^2) (Tobin 1958), by maximum
    likelihood (BFGS, analytic gradient) from the OLS start. With no y at 0 or 1 the coefficients are the OLS ones and
    sigma is sqrt(RSS / N)."""
    x, yy, nm = _design(X, y, names)
    _check_unit_response(yy)
    _check_rank(x)
    b0 = np.linalg.lstsq(x, yy, rcond=None)[0]
    s0 = max(float(np.sqrt(np.mean((yy - x @ b0) ** 2))), 1e-3)
    lo, hi = yy <= 0.0, yy >= 1.0
    mid = ~(lo | hi)
    start = np.append(b0, np.log(s0))
    res = optimize.minimize(_tobit_negll, start, args=(x, yy, lo, hi, mid), jac=True, method="BFGS",
                            options={"gtol": 1e-10, "maxiter": 1000})
    if not np.all(np.isfinite(res.x)) or float(np.max(np.abs(res.jac))) > 1e-6 * max(1.0, x.shape[0]):
        raise ValueError("the Tobit likelihood did not converge")
    return LgdFit(kind="tobit", names=nm, coefficients=res.x[:-1].copy(), extras={"sigma": float(np.exp(res.x[-1]))})


def cure_decomposition(p_cure: np.ndarray, lgd_cure: np.ndarray, lgd_liquidation: np.ndarray) -> np.ndarray:
    """LGD = p_cure LGD_cure + (1 - p_cure) LGD_liq (the practitioner decomposition of dossier 04 F.3; the single
    academic source was not pinned there: attribution UNVERIFIED)."""
    p, c, liq = vec(p_cure, "p_cure"), vec(lgd_cure, "lgd_cure"), vec(lgd_liquidation, "lgd_liquidation")
    same_length(p_cure=p, lgd_cure=c, lgd_liquidation=liq)
    for arr, name in ((p, "p_cure"), (c, "lgd_cure"), (liq, "lgd_liquidation")):
        closed_unit(arr, name)
    return p * c + (1.0 - p) * liq


def predict_lgd(fit: LgdFit, X: np.ndarray) -> np.ndarray:
    """The LGD estimate (M,) in [0, 1]. Only the OLS prediction is clipped; every other kind is in [0, 1] by
    construction."""
    x = mat(X, "X")
    k = len(fit.names)
    if x.shape[1] != k:
        raise ValueError(f"X must have {k} columns (those of the fit), got {x.shape[1]}")
    if fit.kind == "ols":
        return np.clip(x @ fit.coefficients, 0.0, 1.0)
    if fit.kind in ("fractional-logit", "beta"):
        return special.expit(x @ fit.coefficients)
    if fit.kind == "hurdle":
        cols = list(fit.extras["severity_columns"])
        p_loss = special.expit(x @ fit.extras["stage1_coefficients"])
        return p_loss * special.expit(x[:, cols] @ fit.coefficients)
    if fit.kind == "tobit":
        s = float(fit.extras["sigma"])
        m = x @ fit.coefficients
        a, b = (0.0 - m) / s, (1.0 - m) / s
        out = m * (stats.norm.cdf(b) - stats.norm.cdf(a)) + s * (stats.norm.pdf(a) - stats.norm.pdf(b)) \
            + stats.norm.sf(b)
        return np.clip(out, 0.0, 1.0)   # only removes float noise: out is in [0, 1] by construction
    raise ValueError(f"unknown kind {fit.kind!r}")


__all__ = ["KINDS", "LgdFit", "cure_decomposition", "endpoint_transform", "fit_beta", "fit_fractional_logit",
           "fit_hurdle", "fit_ols", "fit_tobit", "logit_quasi_mle", "lstsq", "predict_lgd"]
