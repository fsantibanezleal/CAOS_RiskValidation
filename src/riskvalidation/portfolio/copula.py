"""Default-mode Monte Carlo of the Gaussian and Student-t copulas (dossier 04 H.3, ladder rungs C2 and C3).

Gaussian copula (Li 2000, doi:10.3905/jfi.2000.319253): the asset return of name i is A_i = beta_i' Z + sqrt(1 -
beta_i' beta_i) e_i with Z ~ N(0, factor_corr) and independent N(0, 1) e_i; the name defaults iff A_i < G(PD_i) and
then loses EAD_i LGD_i.

Student-t copula (Frey and McNeil 2003, doi:10.21314/jor.2003.089; Demarta and McNeil 2005,
doi:10.1111/j.1751-5823.2005.tb00254.x): A_i = sqrt(W) (beta_i' Z + sqrt(1 - beta_i' beta_i) e_i) with W = nu /
chi2_nu independent of Z and e; the name defaults iff A_i < t_nu^{-1}(PD_i). The marginal default probability is PD_i
in both, and the t copula adds tail dependence: the joint default probability exceeds the Gaussian one for the same
asset correlation.

Randomness: `numpy.random.Generator(numpy.random.PCG64(seed))`; the result is reproducible for the same (seed,
n_scenarios, chunk_scenarios), and a different chunk size gives a different but equally valid sample. The scenarios
are simulated in row blocks of chunk_scenarios // 3, so that the float64 working arrays (two of them) and the default
flags never exceed chunk_scenarios * N * 8 bytes. Nothing reads global state, the clock or the environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import ndtri

from riskvalidation.portfolio.portfolio import (
    Portfolio,
    check_level,
    es_from_sample,
    float_vector,
    idiosyncratic_scale,
    positive_int,
    var_from_sample,
)

DEFAULT_CHUNK = 100_000
MIN_PER_BATCH = 10


@dataclass(frozen=True)
class LossSample:
    """Simulated losses (money) and how they were drawn; maps onto `rungs[*].simulation` of the C10 document."""

    losses: np.ndarray
    group_losses: np.ndarray | None
    n_scenarios: int
    seed: int
    chunk_scenarios: int

    def __post_init__(self) -> None:
        if self.losses.ndim != 1 or self.losses.size != self.n_scenarios:
            raise ValueError("n_scenarios must equal the number of losses")
        if self.group_losses is not None and self.group_losses.shape[0] != self.n_scenarios:
            raise ValueError("group_losses must have one row per scenario")


def _simulate(p: Portfolio, nu: float | None, n_scenarios: int, seed: int, chunk_scenarios: int,
              track_groups: bool) -> LossSample:
    s = positive_int(n_scenarios, "n_scenarios")
    chunk = positive_int(chunk_scenarios, "chunk_scenarios")
    seed = int(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    n, f = p.loadings.shape
    chol_t = np.linalg.cholesky(p.factor_corr).T
    load_t = np.ascontiguousarray(p.loadings.T)
    idio = idiosyncratic_scale(p)
    threshold = ndtri(p.pd) if nu is None else stats.t.ppf(p.pd, nu)
    unit_loss = p.ead * p.lgd
    g = len(p.group_ids)
    onehot = None
    if track_groups:
        onehot = np.zeros((n, g))
        onehot[np.arange(n), p.group] = 1.0
    losses = np.empty(s)
    group_losses = np.empty((s, g)) if track_groups else None
    block = max(1, chunk // 3)
    for start in range(0, s, block):
        r = min(block, s - start)
        z = rng.standard_normal((r, f)) @ chol_t
        a = rng.standard_normal((r, n))
        a *= idio
        a += z @ load_t
        if nu is not None:
            a *= np.sqrt(nu / rng.chisquare(nu, size=r))[:, None]
        defaulted = a < threshold
        np.multiply(defaulted, unit_loss, out=a)  # the loss of each name, written over the asset returns
        losses[start:start + r] = a.sum(axis=1)
        if group_losses is not None and onehot is not None:
            group_losses[start:start + r] = a @ onehot
    return LossSample(losses=losses, group_losses=group_losses, n_scenarios=s, seed=seed, chunk_scenarios=chunk)


def simulate_gaussian(p: Portfolio, n_scenarios: int, seed: int, chunk_scenarios: int = DEFAULT_CHUNK,
                      track_groups: bool = False) -> LossSample:
    """The one-factor or multi-factor Gaussian copula default-mode loss sample (rung C2)."""
    return _simulate(p, None, n_scenarios, seed, chunk_scenarios, track_groups)


def simulate_student_t(p: Portfolio, nu: float, n_scenarios: int, seed: int, chunk_scenarios: int = DEFAULT_CHUNK,
                       track_groups: bool = False) -> LossSample:
    """The Student-t copula default-mode loss sample (rung C3); nu > 2 degrees of freedom."""
    v = float(nu)
    if not (math.isfinite(v) and v > 2.0):
        raise ValueError(f"nu must be a finite number above 2, got {nu}")
    return _simulate(p, v, n_scenarios, seed, chunk_scenarios, track_groups)


def mc_standard_errors(losses: np.ndarray, q: float, n_batches: int = 20) -> tuple[float, float]:
    """The batch-means standard errors (se_var, se_es) of the sample VaR and ES at level q.

    The sample is split in `n_batches` consecutive equal batches (the remainder is dropped), VaR and ES are computed
    per batch with the sample estimators, and the standard error is sd(batch estimates, ddof = 1) / sqrt(n_batches).
    The estimator is the contract's choice; the dossier names none. At least 2 batches of 10 losses are needed.
    """
    q = check_level(q)
    x = float_vector(losses, "losses")
    nb = positive_int(n_batches, "n_batches")
    if nb < 2:
        raise ValueError("n_batches must be at least 2")
    per = x.size // nb
    if per < MIN_PER_BATCH:
        raise ValueError(f"{x.size} losses give {per} per batch for {nb} batches; at least {MIN_PER_BATCH} are needed")
    batches = x[: per * nb].reshape(nb, per)
    v = np.array([var_from_sample(b, q) for b in batches])
    e = np.array([es_from_sample(b, q) for b in batches])
    root = math.sqrt(nb)
    return float(np.std(v, ddof=1) / root), float(np.std(e, ddof=1) / root)

__all__ = [
    "LossSample",
    "mc_standard_errors",
    "simulate_gaussian",
    "simulate_student_t",
]
