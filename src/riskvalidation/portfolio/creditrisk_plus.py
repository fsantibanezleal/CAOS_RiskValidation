"""CreditRisk+ with K Gamma sectors (dossier 04 H.4; ladder rung C4).

Exposures are integer multiples nu_i of a loss unit, X = sum_i nu_i Y_i with Y_i Poisson of stochastic intensity R_i =
p_i (w0_i + sum_k w_ki S_k), the sector factors S_k independent Gamma with E[S_k] = 1 and var(S_k) = sigma_k^2, and
the probability generating function (Haaf, Reiss and Schoenmakers 2004, doi:10.21314/jor.2004.097)

    G(z) = exp( sum_i w0_i p_i (z^nu_i - 1) - sum_k (1 / sigma_k^2) ln[1 - sigma_k^2 sum_i w_ki p_i (z^nu_i - 1)] ).

The stable recursion. With P_k(z) = sum_i w_ki p_i z^nu_i, mu_k = P_k(1), Q_k = P_k / mu_k and t_k = sigma_k^2 mu_k /
(1 + sigma_k^2 mu_k),

    1 - sigma_k^2 (P_k - mu_k) = (1 + sigma_k^2 mu_k) (1 - t_k Q_k),   -ln(1 - t_k Q_k) = sum_{j>=1} b_kj z^j,

so ln G = sum_j a_j z^j with a_0 = -sum_i w0_i p_i - sum_k ln(1 + sigma_k^2 mu_k) / sigma_k^2 and, for j >= 1, a_j =
sum_{i: nu_i = j} w0_i p_i + sum_k b_kj / sigma_k^2. The log series has the nonnegative coefficients b_j = t q_j + (t
/ j) sum_{l<j} (j - l) b_{j-l} q_l (from the derivative of the series), and G' = (ln G)' G gives p_0 = exp(a_0), n p_n
= sum_{j=1}^n j a_j p_{n-j}. Every term is nonnegative: no subtraction of terms of opposite sign, so no probability is
negative whatever the grid length.

The Panjer-type recursion of the original CreditRisk+ document clears the denominators prod_k D_k, D_k = 1 + sigma_k^2
mu_k - sigma_k^2 P_k, and so subtracts terms of opposite sign; it is numerically unstable (dossier 04 H.4) and is kept
only as the shown comparison, never as a result.

Limits (dossier 04 H.4): default mode only, deterministic recoveries in the basic model, and the Poisson approximation
can give several defaults of one obligor for a large PD. The original CSFB document is not accessed (UNVERIFIED
primary).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import polynomial as npoly

from riskvalidation.portfolio.portfolio import LossMeasures, check_levels, float_vector, positive_int

_CURRENCY = re.compile(r"^[A-Z]{3}$")
UNDERFLOW = 1e-300


@dataclass(frozen=True)
class CreditRiskPlusInput:
    """nu (N,) integer loss units, pd (N,), w0 (N,), weights (N, K), sigma2 (K,), loss_unit money per unit."""

    nu: np.ndarray
    pd: np.ndarray
    w0: np.ndarray
    weights: np.ndarray
    sigma2: np.ndarray
    loss_unit: float
    currency: str

    def __post_init__(self) -> None:
        nu = np.array(self.nu)
        if nu.ndim != 1 or nu.size == 0:
            raise ValueError("nu must be a non-empty one-dimensional array")
        if nu.dtype.kind not in "iuf" or not np.all(np.isfinite(nu)) or np.any(np.mod(nu, 1) != 0) or np.any(nu < 1):
            raise ValueError("nu must hold integers of at least 1 loss unit")
        nu = nu.astype(np.int64)
        n = nu.size
        pd = float_vector(self.pd, "pd")
        w0 = float_vector(self.w0, "w0")
        if pd.size != n or w0.size != n:
            raise ValueError("nu, pd and w0 must have the same length")
        weights = np.array(self.weights, dtype=float)
        sigma2 = float_vector(self.sigma2, "sigma2", allow_empty=True)
        if weights.ndim != 2 or weights.shape[0] != n:
            raise ValueError(f"weights must have shape (N, K), got {weights.shape}")
        if weights.shape[1] != sigma2.size:
            raise ValueError(f"weights have {weights.shape[1]} sectors but sigma2 has {sigma2.size}")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contains non-finite values")
        if np.any(pd <= 0.0) or np.any(pd >= 1.0):
            raise ValueError("pd must lie strictly between 0 and 1")
        if np.any(w0 < 0.0) or np.any(weights < 0.0):
            raise ValueError("the weights must be nonnegative")
        if np.any(sigma2 <= 0.0):
            raise ValueError("sigma2 must be positive")
        if not np.allclose(w0 + weights.sum(axis=1), 1.0, atol=1e-12, rtol=0):
            raise ValueError("w0 plus the sector weights must add up to 1 for every obligor")
        unit = float(self.loss_unit)
        if not (math.isfinite(unit) and unit > 0.0):
            raise ValueError("loss_unit must be positive")
        if not isinstance(self.currency, str) or not _CURRENCY.match(self.currency):
            raise ValueError(f"currency must be three capital letters (ISO 4217), got {self.currency!r}")
        for name, value in (("nu", nu), ("pd", pd), ("w0", w0), ("weights", weights), ("sigma2", sigma2)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "loss_unit", unit)


def _sector_terms(inp: CreditRiskPlusInput) -> list[tuple[np.ndarray, float, float]]:
    """(w_k p by obligor, mu_k, sigma_k^2) of every sector with some weight."""
    out = []
    for k in range(inp.sigma2.size):
        wp = inp.weights[:, k] * inp.pd
        mu = float(wp.sum())
        if mu > 0.0:
            out.append((wp, mu, float(inp.sigma2[k])))
    return out


def _band(nu: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """sum of `values` by exposure band: coefficient c of the polynomial sum_i values_i z^nu_i, length `size`."""
    keep = nu < size
    return np.bincount(nu[keep], weights=values[keep], minlength=size).astype(float)


def _log_pgf_coefficients(inp: CreditRiskPlusInput, n_max: int) -> tuple[float, np.ndarray]:
    """a_0 and a_1 .. a_{n_max} of ln G(z) (index 0 of the array is unused and zero)."""
    a = np.zeros(n_max + 1)
    a0 = -float((inp.w0 * inp.pd).sum())
    a += _band(inp.nu, inp.w0 * inp.pd, n_max + 1)
    a[0] = 0.0
    for wp, mu, s2 in _sector_terms(inp):
        t = s2 * mu / (1.0 + s2 * mu)
        a0 -= math.log1p(s2 * mu) / s2
        q = _band(inp.nu, wp, n_max + 1) / mu
        q[0] = 0.0
        bands = np.flatnonzero(q)
        qb = q[bands]
        b = np.zeros(n_max + 1)
        for j in range(1, n_max + 1):
            below = bands < j
            acc = t * q[j]
            if below.any():
                idx = bands[below]
                acc += (t / j) * float(np.dot(qb[below] * (j - idx), b[j - idx]))
            b[j] = acc
        a += b / s2
    return a0, a


def pmf_stable(inp: CreditRiskPlusInput, n_max: int) -> np.ndarray:
    """P(X = 0 .. n_max) loss units by the stable recursion of Haaf, Reiss and Schoenmakers (2004).

    The tail mass 1 - sum is left out, never added to the last entry. Every entry is nonnegative. Raises ValueError
    when P(X = 0) underflows (a portfolio so large that exp(a_0) is below 1e-300: use a larger loss unit or split the
    portfolio).
    """
    m = positive_int(n_max, "n_max")
    a0, a = _log_pgf_coefficients(inp, m)
    p = np.zeros(m + 1)
    p0 = math.exp(a0) if a0 > -745.0 else 0.0
    if p0 < UNDERFLOW:
        raise ValueError(f"P(X = 0) = exp({a0:.1f}) underflows: the recursion cannot start; use a larger loss unit")
    p[0] = p0
    ja = np.arange(m + 1) * a
    for n in range(1, m + 1):
        p[n] = float(np.dot(ja[1:n + 1], p[n - 1::-1])) / n
    return p


def pmf_panjer(inp: CreditRiskPlusInput, n_max: int) -> np.ndarray:
    """The Panjer-type recursion of the original CreditRisk+ document, kept only as the shown unstable comparison.

    Same signature as `pmf_stable`. The entries are returned as computed and never clipped: on a portfolio with many
    exposure bands and a long grid it may carry round-off and negative values.
    """
    m = positive_int(n_max, "n_max")
    a0 = -float((inp.w0 * inp.pd).sum())
    size = int(inp.nu.max()) + 1
    d_poly = np.array([1.0])
    sectors = []
    for wp, mu, s2 in _sector_terms(inp):
        a0 -= math.log1p(s2 * mu) / s2
        p_k = _band(inp.nu, wp, size)
        d_k = -s2 * p_k
        d_k[0] += 1.0 + s2 * mu
        sectors.append((p_k, d_k))
        d_poly = npoly.polymul(d_poly, d_k)
    # N(z) = A0'(z) D(z) + sum_k P_k'(z) prod_{l != k} D_l(z)
    a0_prime = np.zeros(size)
    np.add.at(a0_prime, inp.nu - 1, inp.w0 * inp.pd * inp.nu)
    n_poly = npoly.polymul(a0_prime, d_poly)
    for k, (p_k, _) in enumerate(sectors):
        others = np.array([1.0])
        for other, (_, d_other) in enumerate(sectors):
            if other != k:
                others = npoly.polymul(others, d_other)
        n_poly = npoly.polyadd(n_poly, npoly.polymul(p_k[1:] * np.arange(1, size), others))
    p0 = math.exp(a0) if a0 > -745.0 else 0.0
    if p0 < UNDERFLOW:
        raise ValueError(f"P(X = 0) = exp({a0:.1f}) underflows: the recursion cannot start; use a larger loss unit")
    p = np.zeros(m + 1)
    p[0] = p0
    d_deg, n_deg = d_poly.size - 1, n_poly.size - 1
    for k in range(1, m + 1):
        jn = min(n_deg, k - 1)
        num = float(np.dot(n_poly[: jn + 1], p[k - 1 - np.arange(jn + 1)]))
        jd = min(d_deg, k)
        if jd >= 1:
            idx = np.arange(1, jd + 1)
            num -= float(np.dot(d_poly[1: jd + 1] * (k - idx), p[k - idx]))
        p[k] = num / (k * d_poly[0])
    return p


def measures_from_pmf(pmf: np.ndarray, loss_unit: float, levels: Sequence[float], el: float | None = None
                      ) -> tuple[LossMeasures, ...]:
    """VaR and ES on the unit grid: k = min{n : F(n) >= q - 1e-12}, VaR = k loss_unit and

    ES = (1 / (1 - q)) [sum_{n>k} n pmf(n) + (F(k) - q) k] loss_unit (the discrete Acerbi-Tasche estimator). `el`
    defaults to loss_unit sum n pmf(n). Raises ValueError when the grid holds less mass than the level (the reason
    names the missing mass) and when the VaR falls below the expected loss. No Monte Carlo standard error.
    """
    f = float_vector(pmf, "pmf")
    unit = float(loss_unit)
    if not (math.isfinite(unit) and unit > 0.0):
        raise ValueError("loss_unit must be positive")
    lv = check_levels(levels)
    n = np.arange(f.size)
    cdf = np.cumsum(f)
    mean = unit * float((n * f).sum())
    e = mean if el is None else float(el)
    out = []
    for q in lv:
        hit = np.flatnonzero(cdf >= q - 1e-12)
        if hit.size == 0:
            raise ValueError(f"the grid holds mass {cdf[-1]:.12f} < {q}: the missing mass {q - cdf[-1]:.3e} lies "
                             "beyond n_max")
        k = int(hit[0])
        var = k * unit
        es = ((n[k + 1:] * f[k + 1:]).sum() + max(0.0, cdf[k] - q) * k) / (1.0 - q) * unit
        if var < e - 1e-9 * max(1.0, abs(e)):
            raise ValueError(f"the VaR {var} at level {q} is below the expected loss {e}: no capital to report")
        out.append(LossMeasures(level=q, el=e, var=var, es=float(es), ec=var - e, var_se=None, es_se=None))
    return tuple(out)

__all__ = [
    "CreditRiskPlusInput",
    "measures_from_pmf",
    "pmf_panjer",
    "pmf_stable",
]
