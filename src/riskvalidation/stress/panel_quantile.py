"""The panel quantile satellite (BL-078): quantile regression with an individual location effect per entity (a bank), by
Canay's two-step estimator or Koenker's penalized estimator, with bootstrap intervals and the projection of a quantile
grid per entity and period under a scenario (Covas, Rump and Zakrajsek 2014).

Model (Koenker 2004, equation (2.2); Canay 2011, equation (2.3)): Q(tau | x_it) = alpha_i + x_it' beta(tau), i = 1..n
entities, t = 1..T_i periods; the entity effect alpha_i is a pure location shift common to all quantiles and only the
slopes depend on tau.

- Two-step (Canay 2011, doi:10.1111/j.1368-423X.2011.00349.x, equations (4.1) and (4.2), A.17): step 1 the within
  estimator gamma_hat of the mean slopes and alpha_hat_i = the time mean of Y_it - X_it' gamma_hat (centred to a zero
  mean over entities, Canay's E(alpha_i) = 0); step 2 the pooled quantile regression of Y_it - alpha_hat_i on X_it per
  tau, a linear program. A column of the design that does not vary within entities (the constant) has no within
  estimate: it is left out of step 1 and kept in step 2.
- Penalized (Koenker 2004, doi:10.1016/j.jmva.2004.05.006, equation (2.4)): the linear program minimising the sum over
  the quantiles k, the entities and the periods of w_k rho_tau_k(y_it - alpha_i - x_it' beta(tau_k)) + lam sum_i
  |alpha_i|, solved by `scipy.optimize.linprog` (HiGHS) on sparse matrices. lam = 0 is the fixed-effects estimator
  (give no constant column: the effects absorb it), a very large lam the pooled quantile regression with zero effects.
  The weights default to 1 / q for the q quantiles; Covas et al. use lam = 1.

No analytic standard error is offered, by decision. Besstremyannaya and Golovan (2019) show that the two-step slopes
carry a bias of order 1 / T that does not fall with n and that the constant converges at sqrt(n) only; Chen and Huo
(2021, arXiv:1911.04729) find a first-step term missing from the covariance (4.7) of Canay. The accompanying intervals
come only from `panel_quantile_bootstrap` (entities resampled whole, Canay's appendix B.5 and B.6, or stationary blocks
of mean 4 periods over time, Covas et al.) and are labelled heuristic in a wide panel (n / T of 1 or more) and always
conjecture-based: the validity of the bootstrap is not proved.

Randomness: `numpy.random.Generator(numpy.random.PCG64(seed))`. Inputs are never mutated. Contract: `engine-stress-api`
(the satellite of 1.3.0 is recorded when Contraste consumes it), module stress.panel_quantile.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, sparse

METHODS = ("two_step", "penalized")
SCHEMES = ("entity", "stationary")
_CONST_TOL = 1e-12


@dataclass(frozen=True, eq=False)
class PanelQuantileFit:
    """A fitted panel quantile model. `taus` (q,) ascending; `coef` (q, p) the slopes beta(tau); `alpha` (n,) the entity
    effects in the order of `entities`; `method`; `n` entities, `T` the mean periods per entity, `n_obs`; `converged`
    True when every linear program solved to optimality; `n_over_t`, `n2_over_t` and `wide_panel` (True at n / T of 1 or
    more) the regime in which the two-step theory (n / T^s tending to 0) is doubtful; `gamma` the within slopes of step
    1 (NaN on a column without within variation; None for the penalized method); `lam` and `weights` the penalized
    settings (None for the two-step). There is no standard error field, by design (module docstring)."""

    taus: np.ndarray
    coef: np.ndarray
    alpha: np.ndarray
    entities: np.ndarray
    method: str
    n: int
    T: float
    n_obs: int
    converged: bool
    n_over_t: float
    n2_over_t: float
    wide_panel: bool
    gamma: np.ndarray | None
    lam: float | None
    weights: np.ndarray | None


@dataclass(frozen=True, eq=False)
class PanelQuantileBootstrap:
    """Bootstrap replicates of a panel quantile fit: `coef_draws` (B, q, p); `alpha_draws` (B, n) for the stationary
    scheme (None for the entity scheme, whose replicates hold other entities); `lower` and `upper` (q, p) the percentile
    limits at `level`; `label` states what the interval is: always conjecture-based, heuristic when `heuristic` (a wide
    panel)."""

    taus: np.ndarray
    scheme: str
    n_boot: int
    mean_block: float
    level: float
    seed: int
    coef_draws: np.ndarray
    alpha_draws: np.ndarray | None
    lower: np.ndarray
    upper: np.ndarray
    heuristic: bool
    label: str


@dataclass(frozen=True, eq=False)
class PanelQuantileProjection:
    """The quantile grid per entity and period: `quantiles` (n, H, q) ascending in tau (rearranged), `taus`, `crossings`
    the number of (entity, period) cells whose quantiles crossed before the rearrangement; with a bootstrap `lower` and
    `upper` (n, H, q) the percentile limits of the rearranged grids and `label` the bootstrap's label, else None."""

    taus: np.ndarray
    quantiles: np.ndarray
    crossings: int
    lower: np.ndarray | None
    upper: np.ndarray | None
    label: str | None


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def _check_taus(taus: Sequence[float]) -> np.ndarray:
    t = np.array(list(taus), dtype=float)
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)) or np.any((t <= 0.0) | (t >= 1.0)):
        raise ValueError(f"taus must be a non-empty vector with each tau in (0, 1), got {list(taus)}")
    t = np.sort(t)
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("taus must be distinct")
    return t


def _check_panel(y: object, x: object, entity: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy = np.array(y, dtype=float)
    xx = np.array(x, dtype=float)
    ent = np.asarray(entity)
    if yy.ndim != 1 or xx.ndim != 2 or xx.shape[0] != yy.size or ent.shape != yy.shape:
        raise ValueError(f"y (N,), x (N, p) and entity (N,) must agree, got {yy.shape}, {xx.shape} and {ent.shape}")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("y or x contains non-finite values")
    if yy.size <= xx.shape[1] or xx.shape[1] < 1:
        raise ValueError(f"N = {yy.size} observations are not above the {xx.shape[1]} regressors")
    if np.linalg.matrix_rank(xx) < xx.shape[1]:
        raise ValueError("x does not have full column rank")
    labels, idx = np.unique(ent, return_inverse=True)
    return yy, xx, labels, idx.astype(int)


def _check_method(method: str, lam: float, weights: Sequence[float] | None, q: int) -> tuple[float, np.ndarray]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    lv = float(lam)
    if not (math.isfinite(lv) and lv >= 0.0):
        raise ValueError(f"lam must be a finite number of 0 or more, got {lam!r}")
    if weights is None:
        w = np.full(q, 1.0 / q)
    else:
        w = np.array(list(weights), dtype=float)
        if w.shape != (q,) or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError(f"weights must be {q} positive numbers, one per tau, got {list(weights)}")
    return lv, w


# ----------------------------------------------------------------------------------------------------------------
# The linear programs


def _linprog(c: np.ndarray, a_eq: sparse.spmatrix, b_eq: np.ndarray, free: int) -> tuple[np.ndarray, bool]:
    """min c'z subject to A z = b with the first `free` variables free and the rest non-negative (HiGHS)."""
    nvar = c.size
    bounds = np.zeros((nvar, 2))
    bounds[:, 1] = np.inf
    bounds[:free, 0] = -np.inf
    res = optimize.linprog(c, A_eq=a_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    if res.x is None:
        return np.full(nvar, np.nan), False
    return np.asarray(res.x, dtype=float), bool(res.status == 0)


def _pooled_quantile(y: np.ndarray, x: np.ndarray, tau: float) -> tuple[np.ndarray, bool]:
    """The pooled quantile regression min_theta sum rho_tau(y - x theta) (Koenker 2004, equation (2.1)), as a linear
    program in (theta, u+, u-)."""
    n_obs, p = x.shape
    eye = sparse.identity(n_obs, format="csr")
    a = sparse.hstack([sparse.csr_matrix(x), eye, -eye], format="csr")
    c = np.concatenate([np.zeros(p), np.full(n_obs, tau), np.full(n_obs, 1.0 - tau)])
    z, ok = _linprog(c, a, y, p)
    return z[:p], ok


def _penalized(y: np.ndarray, x: np.ndarray, idx: np.ndarray, n: int, taus: np.ndarray, lam: float,
               w: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Koenker's (2.4): variables (beta_1..beta_q free, alpha+ and alpha- >= 0, u+ and u- >= 0 per quantile and
    observation)."""
    n_obs, p = x.shape
    q = taus.size
    m = n_obs * q
    z_ind = sparse.csr_matrix((np.ones(n_obs), (np.arange(n_obs), idx)), shape=(n_obs, n))
    x_blocks = sparse.kron(sparse.identity(q, format="csr"), sparse.csr_matrix(x), format="csr")
    a_blocks = sparse.vstack([z_ind] * q, format="csr")
    eye = sparse.identity(m, format="csr")
    a = sparse.hstack([x_blocks, a_blocks, -a_blocks, eye, -eye], format="csr")
    c = np.concatenate([
        np.zeros(q * p), np.full(n, lam), np.full(n, lam),
        np.repeat(w * taus, n_obs), np.repeat(w * (1.0 - taus), n_obs),
    ])
    z, ok = _linprog(c, a, np.tile(y, q), q * p)
    beta = z[: q * p].reshape(q, p)
    alpha = z[q * p : q * p + n] - z[q * p + n : q * p + 2 * n]
    return beta, alpha, ok


def _group_mean(v: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(idx, minlength=n).astype(float)
    if v.ndim == 1:
        return np.bincount(idx, weights=v, minlength=n) / counts
    sums = np.column_stack([np.bincount(idx, weights=v[:, j], minlength=n) for j in range(v.shape[1])])
    return sums / counts[:, None]


def _two_step(y: np.ndarray, x: np.ndarray, idx: np.ndarray, n: int,
              taus: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    xd = x - _group_mean(x, idx, n)[idx]
    yd = y - _group_mean(y, idx, n)[idx]
    scale = 1.0 + np.max(np.abs(x), axis=0)
    keep = np.max(np.abs(xd), axis=0) > _CONST_TOL * scale
    gamma = np.full(x.shape[1], np.nan)
    if np.any(keep):
        gamma[keep] = np.linalg.lstsq(xd[:, keep], yd, rcond=None)[0]
    resid = y - x[:, keep] @ gamma[keep] if np.any(keep) else y.copy()
    alpha = _group_mean(resid, idx, n)
    alpha = alpha - alpha.mean()
    corrected = y - alpha[idx]
    coef = np.empty((taus.size, x.shape[1]))
    ok = True
    for k, tau in enumerate(taus):
        coef[k], good = _pooled_quantile(corrected, x, float(tau))
        ok = ok and good
    return coef, alpha, gamma, ok


def _fit_arrays(y: np.ndarray, x: np.ndarray, idx: np.ndarray, n: int, taus: np.ndarray, method: str, lam: float,
                w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    if method == "two_step":
        return _two_step(y, x, idx, n, taus)
    beta, alpha, ok = _penalized(y, x, idx, n, taus, lam, w)
    return beta, alpha, None, ok


def fit_panel_quantile(y: np.ndarray, x: np.ndarray, entity: np.ndarray, taus: Sequence[float],
                       method: str = "two_step", lam: float = 1.0,
                       weights: Sequence[float] | None = None) -> PanelQuantileFit:
    """Panel quantile regression with an individual location effect per entity (module docstring). `y` (N,), `x` (N, p)
    the covariates (a constant column is allowed, except for `method="penalized"` with `lam` 0), `entity` (N,) the
    entity label of each row, `taus` the quantiles (each in (0, 1), sorted ascending in the fit). `method="two_step"` is
    Canay's estimator, `"penalized"` Koenker's (2.4) with the penalty `lam` and the quantile weights `weights`.

    The fit offers no analytic standard error: Besstremyannaya and Golovan (2019) find an O(1 / T) bias of the slopes
    that does not fall with n and a constant that converges at sqrt(n); Chen and Huo (2021) find a first-step term
    missing from Canay's covariance. Use `panel_quantile_bootstrap`, whose intervals are heuristic in a wide panel.
    `ValueError` for inconsistent shapes, a non-finite input, a rank-deficient `x`, an entity with fewer than 2 rows for
    the two-step estimator, `taus` outside (0, 1) and a negative `lam`."""
    yy, xx, labels, idx = _check_panel(y, x, entity)
    t = _check_taus(taus)
    lv, w = _check_method(method, lam, weights, t.size)
    n = labels.size
    counts = np.bincount(idx, minlength=n)
    if method == "two_step" and np.any(counts < 2):
        raise ValueError("the two-step estimator needs at least 2 periods for every entity")
    coef, alpha, gamma, ok = _fit_arrays(yy, xx, idx, n, t, method, lv, w)
    mean_t = yy.size / n
    return PanelQuantileFit(
        taus=t, coef=coef, alpha=alpha, entities=labels, method=method, n=int(n), T=float(mean_t), n_obs=int(yy.size),
        converged=bool(ok and np.all(np.isfinite(coef))), n_over_t=float(n / mean_t), n2_over_t=float(n * n / mean_t),
        wide_panel=bool(n / mean_t >= 1.0), gamma=gamma, lam=lv if method == "penalized" else None,
        weights=w if method == "penalized" else None,
    )


# ----------------------------------------------------------------------------------------------------------------
# Bootstrap


def _time_layout(idx: np.ndarray, n: int, time: np.ndarray | None) -> np.ndarray:
    """(n, T) row indices of a balanced panel ordered by period within each entity."""
    counts = np.bincount(idx, minlength=n)
    if np.any(counts != counts[0]):
        raise ValueError("the stationary bootstrap needs a balanced panel (the same periods for every entity)")
    order = np.arange(idx.size) if time is None else np.argsort(np.asarray(time), kind="stable")
    layout = np.empty((n, int(counts[0])), dtype=int)
    fill = np.zeros(n, dtype=int)
    for row in order:
        i = idx[row]
        layout[i, fill[i]] = row
        fill[i] += 1
    return layout


def _stationary_indices(rng: np.random.Generator, t: int, mean_block: float) -> np.ndarray:
    """Politis and Romano (1994): a start uniform on 0..T-1, then the next period with probability 1 - 1 / mean_block
    and a new uniform start otherwise, wrapping around, for T draws."""
    out = np.empty(t, dtype=int)
    out[0] = rng.integers(t)
    restart = rng.random(t) < 1.0 / mean_block
    starts = rng.integers(t, size=t)
    for j in range(1, t):
        out[j] = starts[j] if restart[j] else (out[j - 1] + 1) % t
    return out


def panel_quantile_bootstrap(y: np.ndarray, x: np.ndarray, entity: np.ndarray, taus: Sequence[float],
                             method: str = "two_step", lam: float = 1.0, weights: Sequence[float] | None = None,
                             scheme: str = "entity", n_boot: int = 200, mean_block: float = 4.0, seed: int = 0,
                             level: float = 0.95, time: np.ndarray | None = None) -> PanelQuantileBootstrap:
    """Bootstrap replicates and percentile intervals of the slopes of `fit_panel_quantile`, the only intervals the panel
    satellite offers. `scheme="entity"` resamples whole entities with replacement (Canay 2011, appendix B.5 and B.6; the
    validity is a conjecture there); `"stationary"` resamples the periods with the stationary bootstrap of Politis and
    Romano (1994), geometric blocks of mean `mean_block` periods (Covas, Rump and Zakrajsek 2014: mean 4, 5,000
    replications), keeping every entity's row at a resampled period; it needs a balanced panel and `time` (N,), the
    period of each row (the order of the rows within an entity by default). The interval is labelled conjecture-based
    and heuristic in a wide panel."""
    yy, xx, labels, idx = _check_panel(y, x, entity)
    t = _check_taus(taus)
    lv, w = _check_method(method, lam, weights, t.size)
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    mb = float(mean_block)
    if not (math.isfinite(mb) and mb >= 1.0):
        raise ValueError(f"mean_block must be a number of 1 or more, got {mean_block!r}")
    lvl = float(level)
    if not (math.isfinite(lvl) and 0.0 < lvl < 1.0):
        raise ValueError(f"level must lie in (0, 1), got {level!r}")
    n = labels.size
    counts = np.bincount(idx, minlength=n)
    if method == "two_step" and np.any(counts < 2):
        raise ValueError("the two-step estimator needs at least 2 periods for every entity")
    if time is not None and np.asarray(time).shape != yy.shape:
        raise ValueError("time must have one period per row")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    b_count = int(n_boot)
    q, p = t.size, xx.shape[1]
    coefs = np.empty((b_count, q, p))
    alphas = np.empty((b_count, n)) if scheme == "stationary" else None
    rows_of = [np.flatnonzero(idx == i) for i in range(n)]
    layout = _time_layout(idx, n, None if time is None else np.asarray(time)) if scheme == "stationary" else None
    for b in range(b_count):
        if scheme == "entity":
            pick = rng.integers(n, size=n)
            rows = np.concatenate([rows_of[i] for i in pick])
            new_idx = np.concatenate([np.full(rows_of[i].size, j) for j, i in enumerate(pick)])
        else:
            assert layout is not None
            cols = _stationary_indices(rng, layout.shape[1], mb)
            rows = layout[:, cols].ravel()
            new_idx = np.repeat(np.arange(n), cols.size)
        coef, alpha, _, _ = _fit_arrays(yy[rows], xx[rows], new_idx, n, t, method, lv, w)
        coefs[b] = coef
        if alphas is not None:
            alphas[b] = alpha
    lo, hi = np.nanquantile(coefs, [(1.0 - lvl) / 2.0, 1.0 - (1.0 - lvl) / 2.0], axis=0)
    mean_t = yy.size / n
    wide = bool(n / mean_t >= 1.0)
    label = (f"{int(round(100 * lvl))}% percentile {scheme} bootstrap interval, conjecture-based (the bootstrap's "
             "validity for the panel quantile estimators is not proved)")
    if wide:
        label += "; heuristic: wide panel (n / T of 1 or more), where the first-step bias does not vanish"
    return PanelQuantileBootstrap(taus=t, scheme=scheme, n_boot=b_count, mean_block=mb, level=lvl, seed=int(seed),
                                  coef_draws=coefs, alpha_draws=alphas, lower=lo, upper=hi, heuristic=wide, label=label)


# ----------------------------------------------------------------------------------------------------------------
# Projection


def project_panel_quantiles(fit: PanelQuantileFit, x_future: np.ndarray,
                            bootstrap: PanelQuantileBootstrap | None = None) -> PanelQuantileProjection:
    """The quantile grid of every entity and period under future covariates: alpha_i + x_it' beta(tau_k), rearranged by
    sorting along tau when quantiles cross (Chernozhukov, Fernandez-Val and Galichon 2010; Covas et al. 2014), so the
    grid is monotone in tau. `x_future` is (H, p) (the same covariates for every entity, a scenario path) or (n, H, p)
    (per entity, in the order of `fit.entities`); its width must be the fit's p. `PanelQuantileProjection.quantiles` is
    (n, H, q). With a `bootstrap` of the same fit the limits are the percentiles of the rearranged grids of the
    replicates (the entity effects of the fit for the entity scheme, the replicate's for the stationary one): there is
    no analytic interval."""
    p = fit.coef.shape[1]
    xf = np.array(x_future, dtype=float)
    if xf.ndim == 2:
        xf = np.broadcast_to(xf[None, :, :], (fit.n,) + xf.shape)
    if xf.ndim != 3 or xf.shape[2] != p:
        raise ValueError(f"x_future must have {p} columns (the fit's covariates) as (H, {p}) or (n, H, {p}), "
                         f"got shape {np.shape(x_future)}")
    if xf.shape[0] != fit.n:
        raise ValueError(f"x_future must hold one path per entity ({fit.n}), got {xf.shape[0]}")
    if not np.all(np.isfinite(xf)):
        raise ValueError("x_future contains non-finite values")

    def grid(coef: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return alpha[:, None, None] + np.einsum("nhp,qp->nhq", xf, coef)

    raw = grid(fit.coef, fit.alpha)
    crossings = int(np.sum(np.any(np.diff(raw, axis=2) < 0.0, axis=2)))
    quantiles = np.sort(raw, axis=2)
    lower = upper = None
    label = None
    if bootstrap is not None:
        if bootstrap.coef_draws.shape[1:] != fit.coef.shape or not np.array_equal(bootstrap.taus, fit.taus):
            raise ValueError("bootstrap does not belong to this fit (taus or coefficient shape differ)")
        draws = np.empty((bootstrap.n_boot,) + quantiles.shape)
        for b in range(bootstrap.n_boot):
            a = fit.alpha if bootstrap.alpha_draws is None else bootstrap.alpha_draws[b]
            draws[b] = np.sort(grid(bootstrap.coef_draws[b], a), axis=2)
        lo_q, hi_q = (1.0 - bootstrap.level) / 2.0, 1.0 - (1.0 - bootstrap.level) / 2.0
        lower, upper = np.nanquantile(draws, [lo_q, hi_q], axis=0)
        label = bootstrap.label
    return PanelQuantileProjection(taus=fit.taus.copy(), quantiles=quantiles, crossings=crossings, lower=lower,
                                   upper=upper, label=label)


__all__ = [
    "PanelQuantileBootstrap",
    "PanelQuantileFit",
    "PanelQuantileProjection",
    "fit_panel_quantile",
    "panel_quantile_bootstrap",
    "project_panel_quantiles",
]
