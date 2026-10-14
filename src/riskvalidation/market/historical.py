"""The historical-simulation family of dossier 05 A.2 rows 1 to 3.

Every function takes the returns dated before the forecast day, most recent last, and returns the one-step forecast in
the return convention. The age-weighted rule is that of Boudoukh, Richardson and Whitelaw (1998), section 3.
"""

from __future__ import annotations

import numpy as np

from riskvalidation.market.conventions import check_alpha, var_es_from_sample, vec


def age_weights(lam: float, k: int) -> np.ndarray:
    """`w_i = (1 - lam) / (1 - lam^k) lam^(i-1)`, `i = 1..k`: index 0 is the return observed one day ago."""
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must lie in (0, 1), got {lam}")
    if int(k) != k or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    i = np.arange(k, dtype=float)
    return (1.0 - lam) / (1.0 - lam**k) * lam**i


def historical_var_es(returns: np.ndarray, alpha: float, window: int) -> tuple[float, float]:
    """`var_es_from_sample(returns[-window:], alpha)`; ValueError if there are fewer returns than the window."""
    r = vec(returns, "returns")
    if window < 1 or r.size < window:
        raise ValueError(f"need at least window = {window} returns, got {r.size}")
    return var_es_from_sample(r[-window:], alpha)


def age_weighted_var_es(returns: np.ndarray, alpha: float, lam: float, window: int) -> tuple[float, float]:
    """Age-weighted (hybrid) historical simulation of Boudoukh, Richardson and Whitelaw (1998).

    The return observed `i` days ago has the weight `age_weights(lam, window)[i-1]`. The returns are sorted ascending
    and the weights cumulated (`C_j`). Section 3 of the paper puts half of a return's weight on each side of it, up
    to the point halfway to the neighbouring return, and interpolates linearly "between adjacent points". The
    quantile function therefore has two knots per return: `(C_j - w_j / 2, r_j)` (the return itself) and `(C_j, (r_j
    + r_{j+1}) / 2)` (the point halfway to the next return, at the cumulated weight). The worked example of pages 6
    and 7 (5% VaR 2.63% and 2.34%) uses exactly these knots: `2.70% - (2.70% - 2.60%) (0.05 - 0.0478) / (0.0511 -
    0.0478)`, with 2.60% halfway between 2.70% and 2.50%. The VaR is that function evaluated at `alpha` (the smallest
    return below the first knot, the largest above the last).

    Decision D-9 (the paper gives no ES rule): the ES is the tail integral `(1/alpha) int_0^alpha Q(u) du` of that same
    piecewise linear function `Q` (constant at the smallest return on `(0, p_1)`), the definition of dossier 05 A.0
    applied to the distribution the VaR is read from; the integral is exact (trapezoids). `es <= var` because `Q` is
    nondecreasing.
    """
    alpha = check_alpha(alpha)
    r = vec(returns, "returns")
    if window < 1 or r.size < window:
        raise ValueError(f"need at least window = {window} returns, got {r.size}")
    recent = r[-window:][::-1]                  # index 0: one day ago
    w = age_weights(lam, window)
    order = np.argsort(recent, kind="stable")
    x, ws = recent[order], w[order]
    cum = np.cumsum(ws)
    p_ret = cum - 0.5 * ws                       # the level of each sorted return
    # interleave (C_j - w_j/2, r_j) and (C_j, (r_j + r_{j+1})/2), the last return having no right neighbour
    knots_p = np.empty(2 * x.size - 1)
    knots_x = np.empty(2 * x.size - 1)
    knots_p[0::2], knots_x[0::2] = p_ret, x
    knots_p[1::2], knots_x[1::2] = cum[:-1], 0.5 * (x[:-1] + x[1:])
    var = float(np.interp(alpha, knots_p, knots_x))
    if alpha <= knots_p[0]:
        return var, float(x[0])
    inside = knots_p < alpha
    grid = np.concatenate([[0.0], knots_p[inside], [alpha]])
    vals = np.concatenate([[x[0]], knots_x[inside], [var]])
    area = float(np.sum(0.5 * (vals[1:] + vals[:-1]) * np.diff(grid)))
    return var, area / alpha


def filtered_hs_var_es(std_residuals: np.ndarray, mu_next: float, sigma_next: float,
                       alpha: float) -> tuple[float, float]:
    """Filtered historical simulation (Barone-Adesi, Giannopoulos and Vosper 1999) without resampling (decision D-4).

    The scenarios are `mu_next + sigma_next z_s` for every standardised residual (the conditional mean removed in `z`,
    so it is not counted twice); the result is `var_es_from_sample(scenarios, alpha)`.
    """
    z = vec(std_residuals, "std_residuals")
    if not (np.isfinite(sigma_next) and sigma_next > 0.0):
        raise ValueError(f"sigma_next must be positive, got {sigma_next}")
    return var_es_from_sample(float(mu_next) + float(sigma_next) * z, alpha)


def volatility_updated_hs_var_es(returns: np.ndarray, sigma_hist: np.ndarray, sigma_next: float,
                                 alpha: float) -> tuple[float, float]:
    """Volatility-updated HS of Hull and White (1998): `r*_i = r_i sigma_next / sigma_hist_i`."""
    r = vec(returns, "returns")
    s = vec(sigma_hist, "sigma_hist")
    if r.shape != s.shape:
        raise ValueError(f"returns and sigma_hist must have the same length, got {r.size} and {s.size}")
    if np.any(s <= 0.0) or not (np.isfinite(sigma_next) and sigma_next > 0.0):
        raise ValueError("sigma_hist and sigma_next must be positive")
    return var_es_from_sample(r * (float(sigma_next) / s), alpha)
