"""Rung 7: conditional EVT of McNeil and Frey (2000) (dossier 05 A.2 row 10).

The formulas of row 10 are in the loss convention for the standardised residuals of the filter, `z_loss = -z`, with
`q` near 1; `fit_gpd` and `evt_var_es` take and return that convention (standardised units), and
`conditional_evt_forecast` converts
at the boundary to the return convention of the engine.

Decision of this unit (the contract says "AR-GARCH filter fitted by QML"): the filter has a constant conditional mean
(AR order 0) and normal innovations, which is the pseudo-maximum-likelihood of McNeil and Frey; the AR(1) term is not
used.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

from riskvalidation.market.conventions import check_alpha, from_loss_numbers, to_loss_numbers, vec

MIN_EXCEEDANCES = 10


@dataclass(frozen=True)
class GpdFit:
    threshold: float     # u, the loss-residual threshold
    xi: float            # shape
    beta: float          # scale, > 0
    n: int               # residuals used
    n_exceed: int        # N_u, residuals above the threshold
    converged: bool


@dataclass(frozen=True)
class EvtForecast:
    var: float                 # return convention
    es: float | None           # None when xi >= 1
    xi: float
    n_exceed: int


def fit_gpd(z_loss: np.ndarray, threshold_fraction: float) -> GpdFit:
    """Peaks over threshold: `u` is the empirical `(1 - threshold_fraction)` quantile of `z_loss`, `N_u` counts the
    residuals strictly above it, and the excesses `z - u` are fitted by maximum likelihood with `scipy.stats.genpareto`
    (location fixed at 0). ValueError if `N_u < 10`.
    """
    z = vec(z_loss, "z_loss")
    f = float(threshold_fraction)
    if not (0.0 < f < 0.5):
        raise ValueError(f"threshold_fraction must lie in (0, 0.5), got {threshold_fraction}")
    u = float(np.quantile(z, 1.0 - f))
    excess = z[z > u] - u
    if excess.size < MIN_EXCEEDANCES:
        raise ValueError(f"only {excess.size} residuals above the threshold; at least {MIN_EXCEEDANCES} are needed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi, _, beta = stats.genpareto.fit(excess, floc=0.0)
    ok = bool(np.isfinite(xi) and np.isfinite(beta) and beta > 0.0)
    return GpdFit(threshold=u, xi=float(xi), beta=float(beta), n=int(z.size), n_exceed=int(excess.size), converged=ok)


def evt_var_es(fit: GpdFit, q: float) -> tuple[float, float | None]:
    """`z_q = u + (beta/xi)((n/N_u (1-q))^(-xi) - 1)` and `es_q = z_q/(1-xi) + (beta - xi u)/(1-xi)` for `xi < 1`.

    `q` is the confidence level near 1 (`q = 1 - alpha`); `es_q` is None when `xi >= 1` (the caller records the day in
    `es_undefined_days`, never a number). The limit `xi = 0` is `z_q = u - beta ln(n (1-q)/N_u)`, `es_q = z_q + beta`.
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must lie in (0, 1), got {q}")
    log_ratio = float(np.log(fit.n * (1.0 - q) / fit.n_exceed))
    if abs(fit.xi) < 1e-12:
        z_q = fit.threshold - fit.beta * log_ratio
        return z_q, (z_q + fit.beta if fit.xi < 1.0 else None)
    z_q = fit.threshold + (fit.beta / fit.xi) * float(np.expm1(-fit.xi * log_ratio))
    if fit.xi >= 1.0:
        return z_q, None
    return z_q, z_q / (1.0 - fit.xi) + (fit.beta - fit.xi * fit.threshold) / (1.0 - fit.xi)


def conditional_evt_forecast(returns: np.ndarray, alpha: float, threshold_fraction: float,
                             kind: str = "garch") -> EvtForecast:
    """GARCH filter by QML, GPD tail of the standardised loss residuals, then (return convention)
    `var = mu_{t+1} - sigma_{t+1} z_q` and `es = mu_{t+1} - sigma_{t+1} es_q`, `es = None` exactly when `xi >= 1`.

    `returns` are the days before the forecast day; nothing after the last element is read. Needs the optional `arch`.
    """
    from riskvalidation.market.garch import fit_garch, garch_filter  # noqa: PLC0415

    alpha = check_alpha(alpha)
    y = vec(returns, "returns")
    fit = fit_garch(y, kind=kind, innovation="normal", mean="constant")
    flt = garch_filter(fit, y)
    gpd = fit_gpd(to_loss_numbers(flt.std_residuals), threshold_fraction)
    z_q, es_q = evt_var_es(gpd, 1.0 - alpha)
    # the formulas of A.2 row 10 are in the loss convention (mu_loss = -mu_next); the sign is flipped back once, here
    var = float(from_loss_numbers(-flt.mu_next + flt.sigma_next * z_q))
    es = None if es_q is None else float(from_loss_numbers(-flt.mu_next + flt.sigma_next * es_q))
    return EvtForecast(var=var, es=es, xi=gpd.xi, n_exceed=gpd.n_exceed)


def threshold_stability(z_loss: np.ndarray, fractions: np.ndarray) -> list[tuple[float, float, int]]:
    """One `(fraction, xi, n_exceed)` per fraction, ascending in `fraction` (the model-risk table of A.2 row 10)."""
    z = vec(z_loss, "z_loss")
    fr = np.sort(vec(fractions, "fractions"))
    out = []
    for f in fr:
        gpd = fit_gpd(z, float(f))
        out.append((float(f), gpd.xi, gpd.n_exceed))
    return out
