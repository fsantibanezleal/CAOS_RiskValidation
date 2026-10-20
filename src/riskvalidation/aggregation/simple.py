"""The first rungs of the aggregation ladder and the closed-form allocation and diversification quantities (dossier 05
G.1 items 1, 2 and 4, and G.4).

Capital is economic capital on the unexpected-loss basis, `EC = VaR_level(L) - EL` (dossier 04 l.652-653; BCBS 2005
explanatory note, section 4.4). Money is a float in the caller's currency, correlations are decimals. The simple sum
is the
headline capital of the case; every other figure of the aggregation modules is information, not capital (dossier 05
l.519). The engine computes and states no compliance claim.

Contract: `engine-aggregation-api` 1.0.0, module aggregation.simple.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from riskvalidation.portfolio.portfolio import check_level, float_vector, var_from_sample

CORR_TOL = 1e-9


def _matrix(x: object, name: str) -> np.ndarray:
    a = np.array(x, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _nonnegative_vector(x: object, name: str) -> np.ndarray:
    a = float_vector(x, name)
    if np.any(a < 0.0):
        raise ValueError(f"{name} must not contain a negative element")
    return a


def simple_sum(ec: np.ndarray) -> float:
    """The sum of the marginal economic capitals (dossier 05 G.1 item 1), the headline capital of the case.

    `ec` (K,) money, each >= 0. Equals `variance_covariance(ec, R)` for R all ones. Golden: (3, 4) gives 7.0.
    """
    return float(np.sum(_nonnegative_vector(ec, "ec")))


def check_correlation(corr: np.ndarray, tol: float = CORR_TOL) -> None:
    """Raises ValueError unless `corr` is a square, symmetric (to `tol`) matrix with a unit diagonal (to `tol`), every
    entry in [-1, 1] and a smallest eigenvalue >= -tol (positive semi-definite); returns None otherwise."""
    a = _matrix(corr, "corr")
    if not (math.isfinite(tol) and tol >= 0.0):
        raise ValueError("tol must be nonnegative")
    if np.max(np.abs(a - a.T)) > tol:
        raise ValueError("corr must be symmetric")
    if np.max(np.abs(np.diag(a) - 1.0)) > tol:
        raise ValueError("corr must have a unit diagonal")
    if np.any(np.abs(a) > 1.0 + tol):
        raise ValueError("corr entries must lie in [-1, 1]")
    smallest = float(np.linalg.eigvalsh(0.5 * (a + a.T))[0])
    if smallest < -tol:
        raise ValueError(f"corr must be positive semi-definite (smallest eigenvalue {smallest:.6g})")


def variance_covariance(ec: np.ndarray, corr: np.ndarray) -> float:
    """EC_tot = sqrt(EC' R EC) (dossier 05 G.1 item 2): exact for elliptical losses with a common quantile scaling, an
    approximation otherwise. `ec` (K,) money, `corr` (K, K). Goldens: (3, 4) with R the identity 5.0, R all ones 7.0,
    R = [[1, 0.5], [0.5, 1]] sqrt(37) = 6.0828."""
    e = _nonnegative_vector(ec, "ec")
    r = _matrix(corr, "corr")
    check_correlation(r)
    if r.shape[0] != e.size:
        raise ValueError(f"ec and corr must have the same length, got {e.size} and {r.shape[0]}")
    return math.sqrt(max(float(e @ r @ e), 0.0))


def hybrid_quantile(samples: np.ndarray, q: float) -> float:
    """The hybrid VaR of Rosenberg and Schuermann (dossier 05 G.1 item 4, "H-VaR"): the quantile of each column combined
    with the empirical correlation of the columns, sqrt(v' R_emp v) with v_k = var_from_sample(samples[:, k], q).

    `samples` (S, K) money, a loss sample per risk on a common row index. The dossier names the approach and not the
    estimator: the Pearson correlation of the columns is the contract's definition (UNVERIFIED against the paper).
    """
    x = np.array(samples, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"samples must be two-dimensional, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples contains non-finite values")
    if x.shape[0] < 2:
        raise ValueError("samples needs at least 2 rows: a correlation is undefined")
    level = check_level(q)
    if np.any(np.ptp(x, axis=0) == 0.0):
        raise ValueError("samples has a constant column: its correlation is undefined")
    v = np.array([var_from_sample(x[:, k], level) for k in range(x.shape[1])])
    if np.any(v < 0.0):
        raise ValueError(f"the square-root formula needs nonnegative quantiles, got {v.tolist()}")
    r = np.corrcoef(x, rowvar=False).reshape(x.shape[1], x.shape[1])
    return math.sqrt(max(float(v @ r @ v), 0.0))


def diversification_factor(rho_total: float, rho_marginals: Sequence[float]) -> float:
    """DF = rho(L) / sum rho(L_i) (dossier 05 G.4). Not forced to be at most 1: VaR is not subadditive (dossier 05 G.3),
    the sign of every benefit is reported by the validation test and never repaired here. Golden: (6.0828, [3, 4]) gives
    0.8690."""
    total = float(rho_total)
    if not math.isfinite(total):
        raise ValueError("rho_total must be finite")
    parts = float_vector(rho_marginals, "rho_marginals")
    denom = float(parts.sum())
    if denom <= 0.0:
        raise ValueError(f"the sum of rho_marginals must be positive, got {denom}")
    return total / denom


def euler_diversification_benefit(rho_contribution: float, rho_standalone: float) -> float:
    """1 - rho(L_i | L) / rho(L_i) (dossier 05 G.4); zero when the contribution equals the standalone measure.
    Golden: (6, 8) gives 0.25."""
    contribution, standalone = float(rho_contribution), float(rho_standalone)
    if not math.isfinite(contribution):
        raise ValueError("rho_contribution must be finite")
    if not (math.isfinite(standalone) and standalone > 0.0):
        raise ValueError(f"rho_standalone must be positive, got {rho_standalone}")
    return 1.0 - contribution / standalone


def std_contributions(cov: np.ndarray, c: float = 1.0) -> np.ndarray:
    """The Euler contributions of the standard deviation measure, c cov[L_i, L] / sqrt(var L) (Tasche, eq. 3.4;
    dossier 05
    G.4), so that the variance-covariance total allocates exactly. `cov` (K, K) the covariance of (L_1, ..., L_K) in
    money
    squared. Golden: [[4, 2], [2, 9]] gives (1.4552, 2.6679), summing to sqrt(17) = 4.1231."""
    a = _matrix(cov, "cov")
    mult = float(c)
    if not math.isfinite(mult):
        raise ValueError("c must be finite")
    scale = max(1.0, float(np.max(np.abs(a))))
    if np.max(np.abs(a - a.T)) > CORR_TOL * scale:
        raise ValueError("cov must be symmetric")
    if float(np.linalg.eigvalsh(0.5 * (a + a.T))[0]) < -CORR_TOL * scale:
        raise ValueError("cov must be positive semi-definite")
    var_total = float(a.sum())
    if not var_total > 0.0:
        raise ValueError("the variance of the total is zero: the standard deviation contributions are undefined")
    return mult * a.sum(axis=1) / math.sqrt(var_total)
