"""Euler risk contributions and the concentration measures (dossier 04 H.2 and H.6; dossier 05 G.4).

Tasche (2008), arXiv 0708.2542: for a positively homogeneous risk measure the Euler contributions add up to the total,
and

    RC_i^VaR = E[L_i | L = VaR_q(L)],   RC_i^ES = E[L_i | L >= VaR_q(L)].

The ES contributions are taken from the sample with the Acerbi-Tasche weights of `portfolio.es_from_sample`, so that
they add up to the sample ES exactly. The VaR contributions condition on a single loss level and are noisy; the kernel
estimator of Tasche (2009, doi:10.1080/14697680802620599) smooths them: a Nadaraya-Watson estimator with a Gaussian
kernel at L = VaR_q and Silverman's bandwidth 1.06 sd(L) S^(-1/5) (the dossier names the paper, not the bandwidth: the
choice is this contract's, UNVERIFIED). The kernel estimator is biased at the boundary: its sum agrees with the VaR
within a tolerance the caller states, not exactly.

The closed-form one-factor anchor is the conditional-mean contribution that the monotonicity of the loss in the factor
gives, EAD_i LGD_i p_i(G(1 - q)); it is the leading (asymptotic, portfolio-invariant) term of Emmer and Tasche (2005,
arXiv cond-mat/0302402, eq. 2.17) and not their full contribution, which adds granularity terms. Numbering of the
equations of Tasche (2008) is UNVERIFIED in dossier 04 H.2.

Concentration (H.6): HHI = sum s_i^2 with s_i = EAD_i / sum EAD, and the effective number of names 1 / HHI.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

from riskvalidation.engines.vasicek import conditional_pd
from riskvalidation.portfolio.portfolio import check_level, float_vector, quantile_rank, var_from_sample

ADD_UP_RTOL = 1e-9


def _check_groups(group_losses: np.ndarray, losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gl = np.array(group_losses, dtype=float)
    if gl.ndim != 2 or gl.size == 0:
        raise ValueError("group_losses must be a non-empty two-dimensional array (S, G)")
    if not np.all(np.isfinite(gl)):
        raise ValueError("group_losses contains non-finite values")
    x = float_vector(losses, "losses")
    if gl.shape[0] != x.size:
        raise ValueError(f"group_losses has {gl.shape[0]} rows but losses has {x.size}")
    if not np.allclose(gl.sum(axis=1), x, rtol=ADD_UP_RTOL, atol=ADD_UP_RTOL):
        raise ValueError("the group losses do not add up to the losses in every scenario")
    return gl, x


def es_contributions(group_losses: np.ndarray, losses: np.ndarray, q: float) -> np.ndarray:
    """RC_g = E[L_g | L >= VaR_q] with the Acerbi-Tasche weights; sum_g RC_g = es_from_sample(losses, q) exactly."""
    q = check_level(q)
    gl, x = _check_groups(group_losses, losses)
    s = x.size
    order = np.argsort(x, kind="stable")  # ties at the boundary share its fractional mass in input order
    k = quantile_rank(q, s)
    w = np.zeros(s)
    w[order[k:]] = 1.0 / s
    w[order[k - 1]] = max(0.0, k / s - q)
    return (w @ gl) / (1.0 - q)


def var_contributions_kernel(group_losses: np.ndarray, losses: np.ndarray, q: float,
                             bandwidth: float | None = None) -> np.ndarray:
    """RC_g = E[L_g | L = VaR_q] by the Nadaraya-Watson estimator with a Gaussian kernel at the sample VaR.

    The default bandwidth is Silverman's 1.06 sd(L) S^(-1/5); a zero spread without a given bandwidth raises
    ValueError. The sum of the contributions agrees with the VaR within the caller's tolerance, not exactly.
    """
    q = check_level(q)
    gl, x = _check_groups(group_losses, losses)
    if bandwidth is None:
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        h = 1.06 * sd * x.size ** (-0.2)
        if not h > 0.0:
            raise ValueError("the losses have no spread: the default bandwidth is zero")
    else:
        h = float(bandwidth)
        if not (math.isfinite(h) and h > 0.0):
            raise ValueError("bandwidth must be positive")
    u2 = ((x - var_from_sample(x, q)) / h) ** 2
    k = np.exp(-0.5 * (u2 - u2.min()))  # shifted so that the largest kernel weight is 1: no underflow of the sum
    return (k @ gl) / k.sum()


def asrf_var_contributions(ead: np.ndarray, pd: np.ndarray, lgd: np.ndarray | float, rho: np.ndarray | float,
                           q: float) -> np.ndarray:
    """RC_i = EAD_i LGD_i p_i(G(1 - q)) with p_i the Vasicek conditional PD (`engines.vasicek.conditional_pd`).

    The loss is monotone in the factor, so the loss quantile is the conditional mean at the factor quantile z = G(1 -
    q) (dossier 04 C.4). The contributions add up to the infinitely granular (ASRF) VaR. `lgd` and `rho` are scalars
    or per name.
    """
    q = check_level(q)
    e = float_vector(ead, "ead")
    p = float_vector(pd, "pd")
    n = e.size
    lg = np.broadcast_to(np.array(lgd, dtype=float), (n,)).copy() if np.ndim(lgd) == 0 else float_vector(lgd, "lgd")
    r = np.broadcast_to(np.array(rho, dtype=float), (n,)).copy() if np.ndim(rho) == 0 else float_vector(rho, "rho")
    if not (p.size == lg.size == r.size == n):
        raise ValueError("ead, pd, lgd and rho must have the same length")
    if np.any(e <= 0.0):
        raise ValueError("ead must be strictly positive")
    if np.any(lg < 0.0) or np.any(lg > 1.0):
        raise ValueError("lgd must lie in [0, 1]")
    z = float(ndtri(1.0 - q))
    cond = np.empty(n)
    pairs, inverse = np.unique(np.column_stack([p, r]), axis=0, return_inverse=True)
    # the engine validates pd and rho in (0, 1)
    values = np.array([float(conditional_pd(float(a), float(b), z)) for a, b in pairs])
    cond[:] = values[np.ravel(inverse)]
    return e * lg * cond


def hhi(ead: np.ndarray) -> float:
    """sum_i s_i^2 with s_i = EAD_i / sum_j EAD_j; 1 / N for equal exposures, 1 for one name."""
    e = float_vector(ead, "ead")
    if np.any(e < 0.0) or not float(e.sum()) > 0.0:
        raise ValueError("ead must be nonnegative with a positive total")
    s = e / e.sum()
    return float(np.sum(s * s))


def effective_names(ead: np.ndarray) -> float:
    """1 / HHI: between 1 and N."""
    return 1.0 / hhi(ead)

__all__ = [
    "asrf_var_contributions",
    "effective_names",
    "es_contributions",
    "hhi",
    "var_contributions_kernel",
]
