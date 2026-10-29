"""Adaptive conformal recalibration of VaR and ES forecasts (BS-16; dossier 05 A.0 return convention).

A model forecasts the one-day VaR and ES at a tail level `alpha` (for example 0.01 or 0.025). The wrapper keeps a
working level `a_t` and asks the model for its forecast at that level; after the day it moves the level with the
adaptive conformal inference (ACI) rule of Gibbs and Candes (2021, "Adaptive Conformal Inference Under Distribution
Shift", NeurIPS 34, arXiv 2106.00170):

    a_{t+1} = a_t + gamma (alpha - err_t),     err_t = 1{y_t < VaR_t(a_t)},

so a hit lowers the level (a deeper VaR) by `gamma (1 - alpha)` and a quiet day raises it by `gamma alpha`. The
guarantee of Gibbs and Candes concerns the long-run hit frequency, `|mean(err) - alpha| <= (max(a_1, 1 - a_1) + gamma)
/ (gamma T)` for the unclipped recursion, and holds for any forecaster and any data order; it says nothing about the
independence of the hits. Clustered hits are not repaired by a level shift (Christoffersen 1998): the independence and
the conditional-coverage tests are reported beside the unconditional one so the wrapper is judged on all of them.

Differences from the paper, stated (decision D-1): the working level is clipped to `[level_min, level_max]` after
every update (the paper lets it leave (0, 1), where the set is empty or the whole line, which is not a VaR); the
guarantee then holds only while the level stays inside the band, and `AciResult.n_clipped` counts the days it did not.
The ES is the model's own tail mean at the working level (decision D-2): no conformal guarantee is claimed for it.

Everything is in the return convention (VaR and ES negative decimals, `es <= var`) and uses the one-day horizon. The
model is any callable `forecaster(t, level) -> (var, es)` that uses only the information before day `t`; the wrapper
cannot check that, so the contract is tested on the caller side (`tests/market/test_conformal.py`). Deterministic: no
random number is drawn.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from riskvalidation.market.conventions import check_alpha, vec
from riskvalidation.market.historical import historical_var_es

GIBBS_CANDES_2021 = "Gibbs and Candes (2021), NeurIPS 34, arXiv 2106.00170"
LEVEL_MIN = 1e-3
LEVEL_MAX = 0.25

#: a model as the wrapper sees it: `forecaster(t, level) -> (var, es)`, using only the returns before day `t`
Forecaster = Callable[[int, float], tuple[float, float]]


@dataclass(frozen=True)
class AciResult:
    """Wrapped forecasts for the days `start..T-1` of a return series (all arrays have `T - start` entries)."""

    alpha: float                 # the target tail level
    gamma: float                 # the ACI step
    start: int                   # index of the first forecast day in the series
    level_min: float
    level_max: float
    levels: np.ndarray           # working level used on each day (decimal in (0, 0.5))
    var: np.ndarray              # wrapped VaR, return convention (negative decimal)
    es: np.ndarray               # wrapped ES, return convention, `es <= var`
    hits: np.ndarray             # bool: `y_t < var_t`
    clipped: np.ndarray          # bool: the update after this day was cut by the band
    n_clipped: int


@dataclass(frozen=True)
class ConformalScores:
    """Back-test and score summary of one VaR and ES sequence (p-values are None where the test is not evaluated)."""

    n: int
    n_hits: int
    hit_rate: float
    kupiec_p: float | None
    independence_p: float | None
    cc_p: float | None
    fz0_mean: float | None       # mean FZ0 loss (a score: only differences between models are meaningful)
    z2: float | None             # Acerbi-Szekely Z2 statistic (0 under a correct ES, negative for an understated one)
    reasons: tuple[str, ...]     # why a None is None


@dataclass(frozen=True)
class AciComparison:
    wrapped: ConformalScores
    unwrapped: ConformalScores
    result: AciResult
    fz0_change: float | None     # (wrapped FZ0 - unwrapped FZ0) / |unwrapped FZ0|; positive is worse


def _check_band(alpha: float, gamma: float, level_min: float, level_max: float) -> float:
    a = check_alpha(alpha)
    if not (math.isfinite(gamma) and gamma >= 0.0):
        raise ValueError(f"gamma must be a finite non-negative number, got {gamma}")
    if not (0.0 < level_min < a < level_max < 0.5 + 1e-12 and level_max <= 0.5):
        raise ValueError(f"need 0 < level_min < alpha < level_max <= 0.5, got {level_min}, {a}, {level_max}")
    return a


def aci_update(level: float, hit: bool, alpha: float, gamma: float, level_min: float = LEVEL_MIN,
               level_max: float = LEVEL_MAX) -> float:
    """One ACI step: `clip(level + gamma (alpha - 1{hit}), level_min, level_max)` (Gibbs and Candes 2021, eq. 2).

    `level`, `alpha`, `level_min` and `level_max` are tail probabilities (decimals); `gamma >= 0` is the step. A hit
    lowers the level by `gamma (1 - alpha)`, a day without a hit raises it by `gamma alpha`. ValueError unless
    `0 < level_min < alpha < level_max <= 0.5` and `gamma` is finite and non-negative.
    """
    a = _check_band(alpha, gamma, level_min, level_max)
    new = float(level) + gamma * (a - float(bool(hit)))
    return float(min(max(new, level_min), level_max))


def historical_forecaster(returns: np.ndarray, window: int) -> Forecaster:
    """The HS model as a forecaster: `historical_var_es(returns[t - window:t], level, window)` (no day `t` or later)."""
    r = vec(returns, "returns")
    if int(window) != window or window < 1:
        raise ValueError(f"window must be a positive integer, got {window}")

    def forecast(t: int, level: float) -> tuple[float, float]:
        if t < window or t > r.size:
            raise ValueError(f"day {t} needs {window} earlier returns and a series of {r.size} days")
        return historical_var_es(r[t - window:t], level, window)

    return forecast


def aci_var_es(returns: np.ndarray, forecaster: Forecaster, alpha: float, gamma: float = 0.005, start: int = 1,
               level_min: float = LEVEL_MIN, level_max: float = LEVEL_MAX) -> AciResult:
    """Wrap a VaR and ES model with the ACI level update over the days `start..len(returns)-1`.

    `returns` are the realised one-day returns (decimals, one per day, finite). On day `t` the model gives
    `(var, es) = forecaster(t, level_t)`; the day is a hit when `returns[t] < var`; then `level_{t+1} =
    aci_update(level_t, hit, ...)`. The first working level is `alpha`. With `gamma = 0` the result is the unwrapped
    model at `alpha`. `start >= 1`. ValueError for a non-finite return, a non-finite forecast or `es > var`
    (a model that cannot forecast a day is the caller's to handle: the wrapper never fills a missing forecast).
    """
    a = _check_band(alpha, gamma, level_min, level_max)
    y = vec(returns, "returns")
    if int(start) != start or start < 1:
        raise ValueError(f"start must be an integer >= 1, got {start}")
    if start >= y.size:
        raise ValueError(f"no forecast day: start = {start} for a series of {y.size} returns")
    m = y.size - int(start)
    levels, var, es = np.empty(m), np.empty(m), np.empty(m)
    hits, clipped = np.zeros(m, dtype=bool), np.zeros(m, dtype=bool)
    level = a
    for k in range(m):
        t = int(start) + k
        v, e = forecaster(t, level)
        if not (math.isfinite(v) and math.isfinite(e)):
            raise ValueError(f"the forecaster returned a non-finite forecast on day {t}; var and es must be finite")
        if e > v + 1e-12 * max(1.0, abs(v)):
            raise ValueError(f"the forecaster returned es > var on day {t} (es <= var in the return convention)")
        e = min(e, v)       # a rounding excess of one unit in the last place (es == var when alpha N < 1) is repaired
        levels[k], var[k], es[k] = level, v, e
        hits[k] = y[t] < v
        raw = level + gamma * (a - float(hits[k]))
        level = min(max(raw, level_min), level_max)
        clipped[k] = level != raw
    return AciResult(alpha=a, gamma=float(gamma), start=int(start), level_min=float(level_min),
                     level_max=float(level_max), levels=levels, var=var, es=es, hits=hits, clipped=clipped,
                     n_clipped=int(clipped.sum()))


def forecast_scores(returns: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> ConformalScores:
    """Kupiec, independence and conditional coverage p-values, mean FZ0 and Z2 of aligned forecasts.

    `returns`, `var`, `es` are decimals in the return convention, one per forecast day. The tests are those of
    `riskvalidation.validation.market` (asymptotic p-values, `alpha_amber`/`alpha_red` irrelevant here); FZ0 is
    `market.scoring.fz0_loss`; `z2 = 1 - sum(y_t 1{hit_t} / es_t) / (T alpha)` (Acerbi and Szekely 2014 Z2, written
    in the return convention; zero in expectation under a correct ES, negative when the ES is understated).
    A p-value the tests cannot evaluate (for example the
    independence with no hit) is None and `reasons` says why; a non-finite ES gives None for FZ0 and Z2.
    """
    from riskvalidation.market.scoring import fz0_loss  # noqa: PLC0415
    from riskvalidation.validation import market as vm  # noqa: PLC0415

    a = check_alpha(alpha)
    y, v = vec(returns, "returns"), vec(var, "var")
    e = np.array(es, dtype=float)
    if not (y.shape == v.shape == e.shape):
        raise ValueError("returns, var and es must have the same length")
    hit = y < v
    n, x = int(y.size), int(hit.sum())
    reasons: list[str] = []
    kup = vm.kupiec_pof(hit, a)
    ind = vm.christoffersen_independence(hit, a)
    cc = vm.christoffersen_conditional_coverage(hit, a)

    def p_of(res, name: str) -> float | None:
        if res.p_value is None or res.p_value != res.p_value:
            reasons.append(f"{name} not evaluated: {res.notes.removeprefix('not evaluated: ')}")
            return None
        return float(res.p_value)

    kupiec_p, ind_p, cc_p = p_of(kup, "kupiec"), p_of(ind, "independence"), p_of(cc, "conditional coverage")
    fz0 = z2 = None
    if np.all(np.isfinite(e)):
        fz0 = float(np.mean(fz0_loss(y, v, e, a)))
        z2 = float(1.0 - np.sum(y[hit] / e[hit]) / (n * a))
    else:
        reasons.append("fz0 and z2 not evaluated: an ES forecast is not finite")
    return ConformalScores(n=n, n_hits=x, hit_rate=x / n, kupiec_p=kupiec_p, independence_p=ind_p, cc_p=cc_p,
                           fz0_mean=fz0, z2=z2, reasons=tuple(reasons))


def aci_compare(returns: np.ndarray, forecaster: Forecaster, alpha: float, gamma: float = 0.005, start: int = 1,
                level_min: float = LEVEL_MIN, level_max: float = LEVEL_MAX) -> AciComparison:
    """The model at `alpha` (unwrapped) against its ACI wrapper over the same days, scored by `forecast_scores`.

    `fz0_change = (FZ0_wrapped - FZ0_unwrapped) / |FZ0_unwrapped|` (positive: the wrapper scores worse; the plan
    tolerates at most 0.05); None when either FZ0 is undefined.
    """
    y = vec(returns, "returns")
    wrapped = aci_var_es(y, forecaster, alpha, gamma, start, level_min, level_max)
    plain = aci_var_es(y, forecaster, alpha, 0.0, start, level_min, level_max)
    tail = y[int(start):]
    sw = forecast_scores(tail, wrapped.var, wrapped.es, alpha)
    su = forecast_scores(tail, plain.var, plain.es, alpha)
    change = None
    if sw.fz0_mean is not None and su.fz0_mean is not None and su.fz0_mean != 0.0:
        change = (sw.fz0_mean - su.fz0_mean) / abs(su.fz0_mean)
    return AciComparison(wrapped=sw, unwrapped=su, result=wrapped, fz0_change=change)


__all__ = [
    "AciComparison",
    "AciResult",
    "ConformalScores",
    "aci_compare",
    "aci_update",
    "aci_var_es",
    "forecast_scores",
    "historical_forecaster",
]
