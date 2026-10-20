"""The satellite of a default or loss rate (dossier 04 I.1, l.785-795; dossier 05 H.1 layer 2, l.399).

    g(DR_t) = alpha + sum_k phi_k g(DR_(t-k)) + sum_l beta_l' X_(t-l) + eps

with `g` the probit (the Vasicek-consistent link: regressing Phi^-1(DR_t) on macro variables is regressing the latent
credit cycle Z_t on them), the logit, or the identity (the linear form of the Federal Reserve's own PD regression, for
the segments whose rate is zero or negative in some quarters: net recoveries). It is fitted by OLS in NumPy with
classical standard errors on an explicit sample window. Regressors are the agency series only (decision Q7 of the unit
sizing).

The conditional PD under a scenario and the implied systematic factor are the engine's existing
`engines.vasicek.conditional_pd` and `engines.vasicek.implied_factor`; they are reused, not rewritten here.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, ndtr, ndtri

LINKS = ("probit", "logit", "identity")
# : the agency series a satellite may read (decision Q7: the BBB corporate yield and the third-party index levels are
# not read)
AGENCY_REGRESSORS = (
    "real_gdp_growth",
    "unemployment_rate",
    "cpi_inflation",
    "treasury_3m",
    "treasury_5y",
    "treasury_10y",
    "mortgage_rate",
    "prime_rate",
)
NON_POSITIVE_REASON = (
    "a probit or logit fit on a non-positive rate: the link needs a rate in (0, 1); use the identity link"
)


def _name(name: str) -> str:
    if name not in LINKS:
        raise ValueError(f"unknown link {name!r}: expected one of {', '.join(LINKS)}")
    return name


def link(name: str, p: np.ndarray) -> np.ndarray:
    """The link value of a rate: `probit` Phi^-1(p), `logit` ln(p / (1 - p)), `identity` p."""
    _name(name)
    a = np.array(p, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError("p contains non-finite values")
    if name == "identity":
        return a
    if np.any(a <= 0.0) or np.any(a >= 1.0):
        raise ValueError(f"a {name} link needs a rate in (0, 1); a non-positive rate: use the identity link")
    return ndtri(a) if name == "probit" else np.log(a / (1.0 - a))


def inverse_link(name: str, x: np.ndarray) -> np.ndarray:
    """The rate of a link value: in (0, 1) for the probit and the logit, the value itself (signed) for the identity."""
    _name(name)
    a = np.array(x, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError("x contains non-finite values")
    if name == "identity":
        return a
    return ndtr(a) if name == "probit" else expit(a)


@dataclass(frozen=True)
class SatelliteFit:
    """A fitted satellite; maps onto the models document `fit.satellites[*]`."""

    link: str
    window_start: str  # ISO quarter end
    window_end: str
    n_obs: int
    ar_order: int
    regressor_names: tuple[str, ...]
    lags: tuple[int, ...]
    coefficient_names: tuple[str, ...]  # "intercept", "ar_1".., "<regressor>_lag<k>"
    estimates: tuple[float, ...]
    std_errors: tuple[float, ...]
    r_squared: float
    residual_sd: float


@dataclass(frozen=True)
class SatelliteProjection:
    """A projected rate path: the link values and the rates (signed on the identity link, read as net recoveries)."""

    link_values: np.ndarray  # (H,)
    rates: np.ndarray  # (H,)


def _quarter_index(iso: str, name: str) -> int:
    """The running quarter number of an ISO quarter-end date; refuses a date that is not a quarter end."""
    try:
        d = _dt.date.fromisoformat(str(iso))
    except ValueError as exc:
        raise ValueError(f"{name} {iso!r} is not an ISO date") from exc
    last_day = (d + _dt.timedelta(days=1)).day == 1
    if d.month not in (3, 6, 9, 12) or not last_day:
        raise ValueError(f"{name} {iso!r} is not a quarter end (an ISO date such as 2019-12-31)")
    return d.year * 4 + d.month // 3 - 1


def _lags(lags: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(k) for k in lags)
    if not out or any(k < 0 for k in out) or len(set(out)) != len(out) or any(int(k) != k for k in lags):
        raise ValueError("lags must be a non-empty set of distinct non-negative integers")
    return out


def fit_satellite(
    rates: np.ndarray,
    regressors: np.ndarray,
    regressor_names: Sequence[str],
    lags: Sequence[int],
    ar_order: int,
    link_name: str,
    quarters: Sequence[str],
    window_start: str,
    window_end: str,
) -> SatelliteFit:
    """OLS by `numpy.linalg.lstsq` of `g(rate_t)` on an intercept, `ar_order` lags of itself and every regressor at
    every lag of `lags`, over the quarters of `[window_start, window_end]` only; classical standard errors
    `sqrt(diag(s^2 (X'X)^-1))`.

    `rates` is (T,), `regressors` (T, P), `quarters` (T,) ISO quarter ends, ascending and contiguous inside the window.
    Only the quarters inside the window enter, and a lag never reaches before the window start: `n_obs` is the number of
    quarters of the window less the largest lag. Coefficient order: intercept, `ar_1..ar_p`, then for each regressor in
    the given order each lag in the given order. A probit or logit fit on a rate outside (0, 1) is refused (use the
    identity link for net
    recoveries)."""
    if link_name not in LINKS:
        raise ValueError(f"unknown link_name {link_name!r}: expected one of {', '.join(LINKS)}")
    if not isinstance(ar_order, (int, np.integer)) or ar_order < 0:
        raise ValueError(f"ar_order must be a non-negative integer, got {ar_order!r}")
    ar_order = int(ar_order)
    lag_set = _lags(lags)
    names = tuple(str(n) for n in regressor_names)
    bad = [n for n in names if n not in AGENCY_REGRESSORS]
    if bad:
        raise ValueError(
            f"regressor {bad[0]} is outside the agency list ({', '.join(AGENCY_REGRESSORS)}): the BBB corporate "
            f"yield and the third-party index levels are not read"
        )
    y_all = np.array(rates, dtype=float)
    x_all = np.array(regressors, dtype=float)
    q = [str(x) for x in quarters]
    if y_all.ndim != 1 or x_all.ndim != 2 or len(q) != y_all.size or x_all.shape[0] != y_all.size:
        raise ValueError(
            f"rates (T,), regressors (T, P) and quarters (T,) must agree, got {y_all.shape}, {x_all.shape} and {len(q)}"
        )
    if x_all.shape[1] != len(names) or len(names) == 0:
        raise ValueError(f"regressor_names has {len(names)} names for {x_all.shape[1]} regressor columns")
    qi = [_quarter_index(x, "quarters") for x in q]
    lo_q, hi_q = _quarter_index(window_start, "window_start"), _quarter_index(window_end, "window_end")
    if lo_q > hi_q:
        raise ValueError(f"window_start {window_start} is after window_end {window_end}")
    if any(b <= a for a, b in zip(qi, qi[1:], strict=False)):
        raise ValueError("quarters must be ascending")
    if lo_q < qi[0]:
        raise ValueError(f"window_start {window_start} is before the first quarter {q[0]}")
    if hi_q > qi[-1]:
        raise ValueError(f"window_end {window_end} is after the last quarter {q[-1]}")
    inside = [k for k, v in enumerate(qi) if lo_q <= v <= hi_q]
    if len(inside) != hi_q - lo_q + 1 or any(qi[inside[k + 1]] - qi[inside[k]] != 1 for k in range(len(inside) - 1)):
        raise ValueError("quarters have a gap inside the window: a missing quarter is never filled")
    sl = slice(inside[0], inside[-1] + 1)
    y, x = y_all[sl], x_all[sl]
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise ValueError("rates or regressors contain non-finite values in the window")
    if link_name != "identity":
        if np.any(y <= 0.0):
            raise ValueError(NON_POSITIVE_REASON)
        if np.any(y >= 1.0):
            raise ValueError("a probit or logit fit on a rate of 1 or more: the link needs a rate in (0, 1)")
    g = link(link_name, y)
    big = max([ar_order, *lag_set])
    n = g.size - big
    cols = (
        ["intercept"]
        + [f"ar_{k}" for k in range(1, ar_order + 1)]
        + [f"{r}_lag{lag}" for r in names for lag in lag_set]
    )
    if n < len(cols) + 1:
        raise ValueError(
            f"fewer observations ({n}) than coefficients ({len(cols)}) plus one residual degree of freedom"
        )
    rows = np.arange(big, g.size)
    design = [np.ones(n)]
    design += [g[rows - k] for k in range(1, ar_order + 1)]
    design += [x[rows - lag, j] for j in range(len(names)) for lag in lag_set]
    xm = np.column_stack(design)
    target = g[rows]
    tss = float(((target - target.mean()) ** 2).sum())
    if np.ptp(target) == 0.0:
        raise ValueError("the link values are constant in the window: R squared is undefined (a constant rate)")
    beta, _, rank, _ = np.linalg.lstsq(xm, target, rcond=None)
    if rank < xm.shape[1]:
        raise ValueError(
            f"the design matrix is rank deficient (rank {rank} of {xm.shape[1]}): a regressor is collinear"
        )
    resid = target - xm @ beta
    rss = float(resid @ resid)
    s2 = rss / (n - xm.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(xm.T @ xm)))
    r2 = min(1.0, max(0.0, 1.0 - rss / tss))
    return SatelliteFit(
        link=link_name,
        window_start=str(window_start),
        window_end=str(window_end),
        n_obs=int(n),
        ar_order=ar_order,
        regressor_names=names,
        lags=lag_set,
        coefficient_names=tuple(cols),
        estimates=tuple(float(b) for b in beta),
        std_errors=tuple(float(e) for e in se),
        r_squared=float(r2),
        residual_sd=float(np.sqrt(s2)),
    )


def project_scenario(
    fit: SatelliteFit, last_link_values: np.ndarray, regressor_history: np.ndarray, regressor_path: np.ndarray
) -> SatelliteProjection:
    """Plug a regressor path into the recursion of the fit and recover the rate path (dossier 04 l.785-795).

    `last_link_values` is (ar_order,) the last observed link values, most recent last; `regressor_history` is (max lag,
    P) the last observed regressors, most recent last; `regressor_path` is (H, P). The link value of projected quarter t
    is `alpha + sum_k phi_k g_(t-k) + sum_(j, l) beta_(j,l) X_(j, t-l)`; a lag of a quarter before the first projected
    one reads the
    history."""
    p = len(fit.regressor_names)
    lag_set = fit.lags
    big_lag = max(lag_set)
    last = np.array(last_link_values, dtype=float)
    hist = np.array(regressor_history, dtype=float)
    path = np.array(regressor_path, dtype=float)
    if last.shape != (fit.ar_order,):
        raise ValueError(f"last_link_values must have shape ({fit.ar_order},), got {last.shape}")
    if hist.ndim != 2 or hist.shape != (big_lag, p):
        raise ValueError(f"regressor_history must have shape ({big_lag}, {p}), got {hist.shape}")
    if path.ndim != 2 or path.shape[1] != p or path.shape[0] < 1:
        raise ValueError(f"regressor_path must have shape (H >= 1, {p}), got {path.shape}")
    if not (np.all(np.isfinite(last)) and np.all(np.isfinite(hist)) and np.all(np.isfinite(path))):
        raise ValueError("last_link_values, regressor_history and regressor_path must be finite")
    est = np.array(fit.estimates, dtype=float)
    alpha, phi = est[0], est[1 : 1 + fit.ar_order]
    beta = est[1 + fit.ar_order :].reshape(p, len(lag_set))
    xs = np.vstack([hist, path]) if big_lag else path  # row big_lag + h is the regressor of projected quarter h + 1
    values: list[float] = []
    for h in range(path.shape[0]):
        value = alpha
        for k in range(1, fit.ar_order + 1):
            prev = values[h - k] if h - k >= 0 else last[last.size - (k - h)]
            value += phi[k - 1] * prev
        for j in range(p):
            for m, lag in enumerate(lag_set):
                value += beta[j, m] * xs[big_lag + h - lag, j]
        values.append(float(value))
    lv = np.array(values)
    return SatelliteProjection(link_values=lv, rates=np.array(inverse_link(fit.link, lv)))
