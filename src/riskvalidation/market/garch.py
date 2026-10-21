"""Rung 5: GARCH(1,1), GJR-GARCH and EGARCH with normal, Student-t or skew-t innovations, through `arch` (dossier 05 A.2
rows 7 to 9 and the library mapping).

`arch` is the optional extra `market`; it is imported inside the functions that need it (never at import time), with a
clear error when it is absent. Returns reach `arch` scaled to percent (the data scale it is built for) and the
parameters are reported back in decimals: `mu` in decimals, `omega` in decimals squared for GARCH and GJR, and for
EGARCH `omega` of the log-variance equation in decimals (`omega_dec = omega_pct - ln(1e4) (1 - beta)`). The
log-likelihood is that of the returns in decimals (`ll_dec = ll_pct + N ln 100`).

Parameter names are those of `arch`: `mu`, `omega`, `alpha[1]`, `gamma[1]` (GJR leverage; for EGARCH the sign term, the
`theta` of dossier 05 A.2 row 9, with `alpha[1]` the magnitude term, the `gamma` of row 9), `beta[1]`, `nu`, `lambda`
(skew-t).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from riskvalidation.market.conventions import check_alpha, var_es_from_distribution, vec
from riskvalidation.market.parametric import delta_normal_var_es

KINDS = ("garch", "gjr", "egarch")
INNOVATIONS = ("student-t", "skew-t", "normal")
MEANS = ("zero", "constant")
MIN_OBS = 250
PCT = 100.0
LOG_SCALE = math.log(PCT**2)    # ln(1e4): the log-variance shift of the scaling to percent


@dataclass(frozen=True)
class GarchFit:
    kind: str                       # "garch", "gjr" or "egarch"
    innovation: str                 # "student-t", "skew-t" or "normal"
    params: dict[str, float]        # omega, alpha[1], gamma[1] (gjr, egarch), beta[1], nu, lambda (skew-t), mu
    log_likelihood: float
    converged: bool
    n_obs: int


@dataclass(frozen=True)
class GarchFilter:
    """The fitted filter run through a series: per-day conditional mean and volatility, the standardised residuals
    `z_t = (r_t - mu_t) / sigma_t` and the one-step forecasts for the next day (decimals)."""

    mu: np.ndarray
    sigma: np.ndarray
    std_residuals: np.ndarray
    mu_next: float
    sigma_next: float


def _arch() -> Any:
    try:
        import arch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised through a subprocess in the tests
        raise ImportError(
            "riskvalidation.market.garch needs the optional package `arch`; install the extra: "
            "pip install 'riskvalidation[market]'"
        ) from exc
    return arch


def _check_spec(kind: str, innovation: str, mean: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if innovation not in INNOVATIONS:
        raise ValueError(f"innovation must be one of {INNOVATIONS}, got {innovation!r}")
    if mean not in MEANS:
        raise ValueError(f"mean must be one of {MEANS}, got {mean!r}")


def _model(arch: Any, y_pct: np.ndarray, kind: str, innovation: str, mean: str) -> Any:
    dist = {"student-t": "t", "skew-t": "skewt", "normal": "normal"}[innovation]
    mean_name = "Constant" if mean == "constant" else "Zero"
    if kind == "egarch":
        return arch.arch_model(y_pct, mean=mean_name, vol="EGARCH", p=1, o=1, q=1, dist=dist, rescale=False)
    return arch.arch_model(y_pct, mean=mean_name, vol="GARCH", p=1, o=1 if kind == "gjr" else 0, q=1, dist=dist,
                           rescale=False)


def _to_decimals(params: dict[str, float], kind: str) -> dict[str, float]:
    out = dict(params)
    if "mu" in out:
        out["mu"] = out["mu"] / PCT
    if kind == "egarch":
        out["omega"] = out["omega"] - LOG_SCALE * (1.0 - out["beta[1]"])
    else:
        out["omega"] = out["omega"] / PCT**2
    return out


def _to_percent_vector(fit: GarchFit, names: list[str]) -> np.ndarray:
    p = fit.params
    values = []
    for n in names:
        v = float(p["nu" if n == "eta" else n])
        if n == "mu":
            v *= PCT
        elif n == "omega":
            v = v + LOG_SCALE * (1.0 - float(p["beta[1]"])) if fit.kind == "egarch" else v * PCT**2
        values.append(v)
    return np.array(values, dtype=float)


def _stationary(params: dict[str, float], kind: str, innovation: str) -> bool:
    a, b = params["alpha[1]"], params["beta[1]"]
    if kind == "garch":
        ok = params["omega"] > 0.0 and a >= 0.0 and b >= 0.0 and a + b < 1.0
    elif kind == "gjr":
        g = params["gamma[1]"]
        ok = params["omega"] > 0.0 and a >= 0.0 and b >= 0.0 and g + a >= 0.0 and a + g / 2.0 + b < 1.0
    else:
        ok = abs(b) < 1.0
    if innovation != "normal":
        ok = ok and params["nu"] > 2.0
    return bool(ok)


def fit_garch(returns: np.ndarray, kind: str = "garch", innovation: str = "student-t",
              mean: str = "constant") -> GarchFit:
    """Quasi-maximum-likelihood fit through `arch.arch_model` on `returns` (decimals, before the forecast day).

    Deterministic (the optimiser, its start and its tolerance are those of `arch`, fixed). A failed optimisation or a
    fit outside the stationarity domain gives `converged = False` and never a made-up parameter. `N >= 250`
    (ValueError otherwise); the fit never reads beyond the last element of `returns`.
    """
    _check_spec(kind, innovation, mean)
    y = vec(returns, "returns")
    if y.size < MIN_OBS:
        raise ValueError(f"fit_garch needs at least {MIN_OBS} returns, got {y.size}")
    arch = _arch()
    am = _model(arch, y * PCT, kind, innovation, mean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False, update_freq=0)
    # arch names the skew-t degrees of freedom eta
    raw = {("nu" if str(k) == "eta" else str(k)): float(v) for k, v in res.params.items()}
    params = _to_decimals(raw, kind)
    ll = float(res.loglikelihood) + y.size * math.log(PCT)
    ok = bool(res.convergence_flag == 0 and np.isfinite(ll) and _stationary(params, kind, innovation))
    return GarchFit(kind=kind, innovation=innovation, params=params, log_likelihood=ll, converged=ok,
                    n_obs=int(y.size))


def garch_filter(fit: GarchFit, returns: np.ndarray) -> GarchFilter:
    """Run the fitted filter through `returns` (the series the fit used) and forecast the next day."""
    y = vec(returns, "returns")
    arch = _arch()
    am = _model(arch, y * PCT, fit.kind, fit.innovation, "constant" if "mu" in fit.params else "zero")
    names = am.parameter_names() + am.volatility.parameter_names() + am.distribution.parameter_names()
    fixed = am.fix(_to_percent_vector(fit, names))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = fixed.forecast(horizon=1, reindex=False)
    mu_next = float(fc.mean.values[-1, 0]) / PCT
    sigma_next = math.sqrt(float(fc.variance.values[-1, 0])) / PCT
    sigma = np.asarray(fixed.conditional_volatility, dtype=float) / PCT
    resid = np.asarray(fixed.resid, dtype=float) / PCT
    return GarchFilter(mu=y - resid, sigma=sigma, std_residuals=resid / sigma, mu_next=mu_next, sigma_next=sigma_next)


def _student_t_unit_var(alpha: float, nu: float) -> tuple[float, float]:
    """VaR and ES quantities (return convention) of the unit-variance Student-t: `q_alpha` and the tail mean."""
    if nu <= 2.0:
        raise ValueError(f"nu must exceed 2 for unit-variance innovations, got {nu}")
    scale = math.sqrt((nu - 2.0) / nu)
    t = float(stats.t.ppf(alpha, nu))
    es = -float(stats.t.pdf(t, nu)) / alpha * (nu + t * t) / (nu - 1.0) * scale
    return t * scale, es


def innovation_var_es(fit: GarchFit, alpha: float) -> tuple[float, float]:
    """`q_alpha` and `(1/alpha) int_0^alpha q_u du` of the unit-variance innovation of the fit."""
    alpha = check_alpha(alpha)
    if fit.innovation == "normal":
        return delta_normal_var_es(0.0, 1.0, alpha)
    nu = float(fit.params["nu"])
    if fit.innovation == "student-t":
        return _student_t_unit_var(alpha, nu)
    arch = _arch()
    dist = arch.univariate.SkewStudent()
    par = np.array([nu, float(fit.params["lambda"])])
    return var_es_from_distribution(lambda u: np.asarray(dist.ppf(np.asarray(u, dtype=float), par), dtype=float),
                                    alpha, n_nodes=400)


def garch_var_es_forecast(fit: GarchFit, returns: np.ndarray, alpha: float) -> tuple[float, float]:
    """`var = mu + sigma_{t+1} q_alpha`, `es = mu + sigma_{t+1} (1/alpha) int_0^alpha q_u du` for the next day.

    `returns` is the series the fit used; the filter is run through it to the last day. For normal innovations the
    result is `delta_normal_var_es(mu, sigma_{t+1}, alpha)`.
    """
    alpha = check_alpha(alpha)
    flt = garch_filter(fit, returns)
    if fit.innovation == "normal":
        return delta_normal_var_es(flt.mu_next, flt.sigma_next, alpha)
    q, es_q = innovation_var_es(fit, alpha)
    return flt.mu_next + flt.sigma_next * q, flt.mu_next + flt.sigma_next * es_q

__all__ = [
    "GarchFilter",
    "GarchFit",
    "fit_garch",
    "garch_filter",
    "garch_var_es_forecast",
    "innovation_var_es",
]
