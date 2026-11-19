"""The e-backtests of VaR and of expected shortfall of Wang, Wang and Ziegel (2022): backtest e-statistics, betting
processes and the e-process that multiplies the bets.

A forecaster reports, each day, a VaR (and for an expected shortfall test also an ES) for the loss of the next day.
The backtest e-statistic of a day is a function of the realised loss and the forecast whose expectation is at most 1
when the forecast is not an under-forecast and more than 1 when it is (Definitions 1 and 2). The e-process bets on it:
`M_t = prod_s (1 - lambda_s + lambda_s X_s)` with the bet `lambda_s` in `[0, gamma]` chosen from the past only
(equation (7)). Under the null `M_t` is a non-negative supermartingale with `M_0 = 1`, so the monitor that rejects
when `M_t` reaches `1 / alpha_level` has size at most `alpha_level` at any stopping time (Theorem 2, Ville).

Conventions of the engine, not of the paper: losses are positive numbers, a hit is `loss > forecast`, `alpha` is the
exceedance probability (the paper's level `p` is `1 - alpha`: `alpha = 0.025` is a 97.5% VaR or ES), `alpha_level` is
the size of the monitor (the paper's alert thresholds 2, 5 and 10 are `alpha_level` 0.5, 0.2 and 0.1) and times are
1-based day counts.

- `var_backtest_e_statistic`: equation (1), `1{loss > var} / alpha`.
- `es_backtest_e_statistic`: equation (3), `(loss - var)+ / (alpha (es - var))`, with 0/0 = 1 and 1/0 = infinity.
- `betting_process`, `e_backtest_var`, `e_backtest_es`: the betting processes GREE, GREL and GREM of Section 5.1
  (`method`), solved by the closed form (15) the paper uses in its simulations or by the growth-rate maximisation of
  (13) and (14) (`solver`), on all past days or a rolling `window` (Remark 8 and Section 8).

Choices the paper leaves open, each fixed here: the first day, which has no past sample, bets nothing; a past sample
whose mean e-statistic is at most 1 bets nothing (Proposition 2 (i)); a past e-statistic equal to infinity makes the
empirical growth rate infinite for every positive bet, so the bet is the cap `gamma`; the default is GREM, "recommended
as a default choice" (Sections 5.2 and 8), with equal weights; an ES forecast below its VaR forecast, which the
paper sets to infinity as "a case of no relevance", is refused.

Sources: Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991; Ville (1939); Ramdas, Grunwald, Vovk and
Shafer (2023), Statistical Science 38(4):576-601, arXiv 2210.01948.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy import special

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.validation.sequential._core import check_alpha_level, check_open_unit, frozen
from riskvalidation.validation.sequential.monitor import EProcessResult

REFERENCE = (
    "Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991; Ville (1939); Ramdas, Grunwald, Vovk and "
    "Shafer (2023), Statistical Science 38(4):576-601, arXiv 2210.01948"
)
H0_VAR = ("the VaR forecast is not below the true conditional VaR at level 1 - alpha on any day (it does not "
          "under-forecast the risk), equation (8)")
H0_ES = ("the ES forecast is not below the true conditional ES and the VaR forecast equals the true conditional VaR "
         "at level 1 - alpha on every day, equation (9)")
METHODS = ("GREE", "GREL", "GREM")
SOLVERS = ("taylor", "exact")
_BLOCK = 2_000_000  # matrix cells per block of the quadratic GREL and exact computations
_BISECTIONS = 56  # halvings of [0, gamma] of the exact solver: 0.5 / 2^56 is below 1e-17


# ---------------------------------------------------------------------------------------------------- inputs

def _vector(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} holds a NaN or an infinite value")
    return arr


def _same_length(*named: tuple[str, np.ndarray]) -> None:
    if len({a.size for _, a in named}) != 1:
        raise ValueError(f"{', '.join(n for n, _ in named)} must have the same length")


def _check_window(window: int | None) -> int | None:
    if window is None:
        return None
    if isinstance(window, bool) or int(window) != window or window < 1:
        raise ValueError(f"window must be None or a positive whole number of days, got {window!r}")
    return int(window)


def _check_options(method: str, solver: str, gamma: float, window: int | None) -> tuple[float, int | None]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    return check_open_unit("gamma", gamma), _check_window(window)


# ------------------------------------------------------------------------------------------ e-statistics

def _e_var(x: np.ndarray, r: np.ndarray, alpha: float) -> np.ndarray:
    """Equation (1), `1{x > r} / (1 - p)` with `1 - p = alpha`, broadcast over its arguments."""
    return (x > r).astype(float) / alpha


def _e_es(x: np.ndarray, r: np.ndarray, z: np.ndarray, alpha: float) -> np.ndarray:
    """Equation (3) with the conventions 0/0 = 1 and 1/0 = infinity, broadcast over its arguments."""
    num = np.maximum(x - z, 0.0)
    den = alpha * (r - z)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
    return np.where(den == 0.0, np.where(num == 0.0, 1.0, np.inf), ratio)


def var_backtest_e_statistic(loss: object, var: object, alpha: float) -> np.ndarray:
    """The backtest e-statistic of the VaR, the paper's equation (1) (Example 3): `1{loss_t > var_t} / alpha` per day.

    `loss` and `var` are finite one-dimensional arrays of the same length (positive losses, a hit is a loss strictly
    above the VaR forecast); `alpha` is the exceedance probability of the VaR (the paper's `1 - p`). Its expectation is
    at most 1 when the VaR forecast is at least the true conditional VaR, equal to 1 at the true value, and more than 1
    when the forecast is below it. By Theorem 4 every other one-sided e-statistic for the VaR is dominated by one of
    the form `1 - h + h e`, which the bet of the e-process supplies.
    """
    a = check_open_unit("alpha", alpha)
    x, r = _vector(loss, "loss"), _vector(var, "var")
    _same_length(("loss", x), ("var", r))
    return _e_var(x, r, a)


def es_backtest_e_statistic(loss: object, var: object, es: object, alpha: float) -> np.ndarray:
    """The backtest e-statistic of the expected shortfall, the paper's equation (3) and Theorem 1:
    `(loss_t - var_t)+ / (alpha (es_t - var_t))` per day, with the paper's conventions 0/0 = 1 and 1/0 = infinity when
    the ES forecast equals the VaR forecast.

    `var` is the VaR forecast at level `1 - alpha` (the auxiliary functional the test needs to be exact: the null
    fixes it) and `es` the ES forecast; an ES forecast below the VaR forecast is refused (the paper sets the
    e-statistic to infinity there, "a case of no relevance since ES >= VaR"). Its expectation is 1 at the true VaR and
    the true ES (equality in Theorem 1), below 1 when the ES forecast is higher and above 1 when it is lower.
    """
    a = check_open_unit("alpha", alpha)
    x, z, r = _vector(loss, "loss"), _vector(var, "var"), _vector(es, "es")
    _same_length(("loss", x), ("var", z), ("es", r))
    if np.any(r < z):
        raise ValueError("es must not be below var: an ES forecast below the VaR forecast of the same level is invalid")
    return _e_es(x, r, z, a)


# ------------------------------------------------------------------------------------------ betting processes

def _constant(a: np.ndarray) -> bool:
    return bool(a.size == 0 or np.all(a == a[0]))


def _taylor_from_sums(s1: np.ndarray, s2: np.ndarray, gamma: float) -> np.ndarray:
    """Equation (15): `0 v (sum (e - 1) / sum (e - 1)^2) ^ gamma`; no bet when the squares sum to 0 (every e is 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s2 > 0.0, s1 / s2, 0.0)
    return np.clip(ratio, 0.0, gamma)


def _lambda_from_matrix(e: np.ndarray, mask: np.ndarray, gamma: float, solver: str) -> np.ndarray:
    """One bet per row from the e-statistics `e[row, s]` of the masked past days of that row."""
    is_inf = np.isinf(e) & mask
    d = np.where(mask & ~is_inf, e - 1.0, 0.0)
    if solver == "taylor":
        lam = _taylor_from_sums(d.sum(axis=1), (d * d).sum(axis=1), gamma)
    else:  # maximise the empirical growth rate mean log(1 + lam d), concave: bisect its decreasing derivative
        def slope(lam_: np.ndarray) -> np.ndarray:
            return (d / (1.0 + lam_[:, None] * d) * mask).sum(axis=1)

        lo, hi = np.zeros(d.shape[0]), np.full(d.shape[0], gamma)
        for _ in range(_BISECTIONS):
            mid = 0.5 * (lo + hi)
            positive = slope(mid) > 0.0
            lo, hi = np.where(positive, mid, lo), np.where(positive, hi, mid)
        lam = 0.5 * (lo + hi)
        lam = np.where(slope(np.full(d.shape[0], gamma)) >= 0.0, gamma, lam)  # the optimum is at or beyond the cap
        lam = np.where(slope(np.zeros(d.shape[0])) <= 0.0, 0.0, lam)  # a sample that does not favour a bet: none
    return np.where(is_inf.any(axis=1), gamma, lam)


def _window_start(rows: np.ndarray, window: int | None) -> np.ndarray:
    return np.zeros(rows.size, dtype=int) if window is None else np.maximum(0, rows - window)


def _lambda_gree(e: np.ndarray, gamma: float, solver: str, window: int | None) -> np.ndarray:
    """GREE, equation (13): the sample is the e-statistics of the past days, each on its own day's forecast."""
    n = e.size
    t = np.arange(n)
    if solver == "taylor":
        inf = np.isinf(e)
        d = np.where(inf, 0.0, e - 1.0)
        c1, c2, ci = (np.concatenate([[0.0], np.cumsum(v)]) for v in (d, d * d, inf.astype(float)))
        lo = _window_start(t, window)
        lam = _taylor_from_sums(c1[t] - c1[lo], c2[t] - c2[lo], gamma)
        return np.where(ci[t] - ci[lo] > 0.0, gamma, lam)
    lam = np.zeros(n)
    block = max(1, _BLOCK // max(n, 1))
    for start in range(0, n, block):
        rows = t[start:start + block]
        mask = (t[None, :] < rows[:, None]) & (t[None, :] >= _window_start(rows, window)[:, None])
        lam[rows] = _lambda_from_matrix(np.broadcast_to(e, (rows.size, n)), mask, gamma, solver)
    return lam


def _lambda_grel(x: np.ndarray, r: np.ndarray, z: np.ndarray, ev: Callable[..., np.ndarray], e: np.ndarray,
                 gamma: float, solver: str, window: int | None) -> np.ndarray:
    """GREL, equation (14): the sample is the past losses, each valued at the forecast of the day being bet on. With
    constant forecasts the sample equals GREE's, which the paper states and which is then computed in linear time."""
    if _constant(r) and _constant(z):
        return _lambda_gree(e, gamma, solver, window)
    n = x.size
    t = np.arange(n)
    lam = np.zeros(n)
    block = max(1, _BLOCK // max(n, 1))
    for start in range(0, n, block):
        rows = t[start:start + block]
        mask = (t[None, :] < rows[:, None]) & (t[None, :] >= _window_start(rows, window)[:, None])
        sample = ev(x[None, :], r[rows, None], z[rows, None])
        lam[rows] = _lambda_from_matrix(sample, mask, gamma, solver)
    return lam


def _log_path(e: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Equation (7) in logs; a day with no bet contributes 1 whatever the e-statistic (0 times infinity is no bet)."""
    safe = np.where(lam > 0.0, e, 1.0)
    return np.cumsum(np.log1p(lam * (safe - 1.0)))


def _mixture_weight(prev_a: np.ndarray, prev_b: np.ndarray) -> np.ndarray:
    """`M_a / (M_a + M_b)` from the logs of two wealths (both infinite: one half each)."""
    with np.errstate(invalid="ignore"):
        w = special.expit(prev_a - prev_b)
    return np.where(np.isnan(w), 0.5, w)


@dataclass(frozen=True, eq=False)
class _Run:
    e: np.ndarray
    lam: np.ndarray
    log_e: np.ndarray
    components: dict[str, np.ndarray]


def _run(x: np.ndarray, r: np.ndarray, z: np.ndarray, alpha: float, measure: str, method: str, solver: str,
         gamma: float, window: int | None) -> _Run:
    def ev(xx: np.ndarray, rr: np.ndarray, zz: np.ndarray) -> np.ndarray:
        return _e_var(xx, rr, alpha) if measure == "var" else _e_es(xx, rr, zz, alpha)

    e = ev(x, r, z)
    if x.size == 0:
        return _Run(e, np.empty(0), np.empty(0), {})
    lam_e = _lambda_gree(e, gamma, solver, window) if method in ("GREE", "GREM") else None
    lam_l = _lambda_grel(x, r, z, ev, e, gamma, solver, window) if method in ("GREL", "GREM") else None
    if method == "GREE":
        return _Run(e, lam_e, _log_path(e, lam_e), {})  # type: ignore[arg-type]
    if method == "GREL":
        return _Run(e, lam_l, _log_path(e, lam_l), {})  # type: ignore[arg-type]
    assert lam_e is not None and lam_l is not None
    path_e, path_l = _log_path(e, lam_e), _log_path(e, lam_l)
    prev_e, prev_l = np.concatenate([[0.0], path_e[:-1]]), np.concatenate([[0.0], path_l[:-1]])
    w = _mixture_weight(prev_e, prev_l)
    lam = w * lam_e + (1.0 - w) * lam_l  # the betting process of the mixture, Section 5.1
    log_e = np.logaddexp(path_e, path_l) - math.log(2.0)
    return _Run(e, lam, log_e, {"GREE": path_e, "GREL": path_l})


def _inputs(loss: object, var: object, es: object | None,
            alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    a = check_open_unit("alpha", alpha)
    x, z = _vector(loss, "loss"), _vector(var, "var")
    if es is None:
        _same_length(("loss", x), ("var", z))
        return x, z, z, a, "var"
    r = _vector(es, "es")
    _same_length(("loss", x), ("var", z), ("es", r))
    if np.any(r < z):
        raise ValueError("es must not be below var: an ES forecast below the VaR forecast of the same level is invalid")
    return x, r, z, a, "es"


def betting_process(loss: object, var: object, alpha: float, *, es: object | None = None, method: str = "GREM",
                    solver: str = "taylor", gamma: float = 0.5, window: int | None = None) -> np.ndarray:
    """The bets `lambda_t`, one per day, of a betting process of Section 5.1 of the paper (the VaR backtest when `es`
    is `None`, the ES backtest of the pair (`es`, `var`) otherwise).

    `method` is `"GREE"` (equation (13): the sample is the e-statistics of the past days on their own forecasts),
    `"GREL"` (equation (14): the past losses on the current forecasts) or `"GREM"` (the default, the paper's
    recommendation: the equal mixture of the wealth of the two, with the bet `(M^GREE lambda^GREE + M^GREL
    lambda^GREL) / (M^GREE + M^GREL)` of the day before). `solver` is `"taylor"` (the closed form (15), used in the
    paper's simulations and in Appendix A) or `"exact"` (the maximiser of the empirical growth rate over `[0, gamma]`
    by bisection of its derivative). `gamma` in (0, 1) caps the bet (default 1/2, Remark 7); `window` keeps the last
    days only (Remark 8, Section 8). The first day bets 0. Every bet is chosen from the days before the one it is
    paid on, so the e-process is valid whatever the choice (Theorem 2).
    """
    g, w = _check_options(method, solver, gamma, window)
    x, r, z, a, measure = _inputs(loss, var, es, alpha)
    return _run(x, r, z, a, measure, method, solver, g, w).lam


# -------------------------------------------------------------------------------------------------- the record

@dataclass(frozen=True, eq=False)
class EBacktestResult(EProcessResult):
    """An `EProcessResult` of an e-backtest: `log_e` is `log M_t` of equation (7), `e_statistic` the backtest
    e-statistic of each day (infinite where the paper's conventions say so), `betting` the bet `lambda_t` of each day
    and, under GREM, `component_log_e` the logs of the wealth of GREE and of GREL whose mean is `M_t`.

    `extras` carries the parameters (`alpha`, `measure` `"var"` or `"es"`, `method`, `solver`, `gamma`, `window`) and
    `n_hits`, the number of days the loss exceeded the VaR forecast.
    """

    e_statistic: np.ndarray = field(default_factory=lambda: np.empty(0))
    betting: np.ndarray = field(default_factory=lambda: np.empty(0))
    component_log_e: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "e_statistic", frozen(self.e_statistic))
        object.__setattr__(self, "betting", frozen(self.betting))
        object.__setattr__(self, "component_log_e",
                           MappingProxyType({k: frozen(v) for k, v in self.component_log_e.items()}))


def _result(monitor_id: str, h0: str, run: _Run, hash_value: str, level: float, extras: dict) -> EBacktestResult:
    return EBacktestResult(
        monitor_id=monitor_id, log_e=run.log_e, alpha_level=level, h0=h0, reference=REFERENCE, inputs_hash=hash_value,
        notes="anytime-valid and one-sided (it tests under-forecasting only); read it every day",
        extras=extras, e_statistic=run.e, betting=run.lam, component_log_e=run.components)


@register(
    "sequential.e_backtest_var",
    family="sequential",
    name="VaR e-backtest",
    h0=H0_VAR,
    reference=REFERENCE,
    decision="metric",
)
def e_backtest_var(loss: object, var: object, alpha: float, *, alpha_level: float = 0.05, method: str = "GREM",
                   solver: str = "taylor", gamma: float = 0.5, window: int | None = None) -> EBacktestResult:
    """The e-backtest of a VaR forecast (Wang, Wang and Ziegel 2022, Section 4.2): the e-process (7) of the backtest
    e-statistic (1) with the betting process `method`.

    `loss` and `var` are finite arrays of the same length in time order (positive losses; the VaR forecast of day `t`
    made before the loss of day `t`); `alpha` is the exceedance probability of the VaR; `alpha_level` the size of the
    monitor (default 0.05, threshold 20; the paper's alert thresholds 2, 5 and 10 are 0.5, 0.2, 0.1). For `method`,
    `solver`, `gamma` and `window` see `betting_process`. The test is one-sided: a VaR above the truth is not
    rejected (for the other side the paper, Remark 5, uses the negated loss). Returns an `EBacktestResult` (an empty
    series: `n = 0`, not evaluated).
    """
    level = check_alpha_level(alpha_level)
    g, w = _check_options(method, solver, gamma, window)
    x, r, z, a, _ = _inputs(loss, var, None, alpha)
    run = _run(x, r, z, a, "var", method, solver, g, w)
    extras = {"alpha": a, "measure": "var", "method": method, "solver": solver, "gamma": g, "window": w,
              "n_hits": int((x > r).sum())}
    return _result("sequential.e_backtest_var", H0_VAR, run, inputs_hash(x, r, a, level, method, solver, g, w), level,
                   extras)


@register(
    "sequential.e_backtest_es",
    family="sequential",
    name="ES e-backtest",
    h0=H0_ES,
    reference=REFERENCE,
    decision="metric",
)
def e_backtest_es(loss: object, var: object, es: object, alpha: float, *, alpha_level: float = 0.05,
                  method: str = "GREM", solver: str = "taylor", gamma: float = 0.5,
                  window: int | None = None) -> EBacktestResult:
    """The e-backtest of an expected-shortfall forecast and its VaR forecast (Wang, Wang and Ziegel 2022, Theorem 1 and
    Section 4.2): the e-process (7) of the backtest e-statistic (3) with the betting process `method`.

    `loss`, `var` and `es` are finite arrays of the same length in time order; `var` is the VaR forecast at level
    `1 - alpha` and `es` the ES forecast at the same level, never below `var`. The null (9) fixes the VaR forecast to
    the truth, so the evidence read is against the ES forecast given that VaR; Proposition 1 shows the e-process stays
    valid when both forecasts are above the truth by an amount (or a factor) that keeps their difference above the true
    ES minus VaR. `alpha_level`, `method`, `solver`, `gamma` and `window` as in `e_backtest_var`. Returns an
    `EBacktestResult` (an empty series: `n = 0`, not evaluated).
    """
    level = check_alpha_level(alpha_level)
    g, w = _check_options(method, solver, gamma, window)
    x, r, z, a, _ = _inputs(loss, var, es, alpha)
    run = _run(x, r, z, a, "es", method, solver, g, w)
    extras = {"alpha": a, "measure": "es", "method": method, "solver": solver, "gamma": g, "window": w,
              "n_hits": int((x > z).sum())}
    return _result("sequential.e_backtest_es", H0_ES, run, inputs_hash(x, z, r, a, level, method, solver, g, w), level,
                   extras)


__all__ = [
    "EBacktestResult",
    "betting_process",
    "e_backtest_es",
    "e_backtest_var",
    "es_backtest_e_statistic",
    "var_backtest_e_statistic",
]
