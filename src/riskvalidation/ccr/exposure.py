"""Exposure measures of a netting set (dossier 05 C.1; CRE53.11 to CRE53.20 from the addendum R9): EE and ENE, the
effective EE and EPE, PFE, netting, the effective maturity, the analytic EE of a single swap (ladder rung 2) and ladder
rung 1.

EE is the positive exposure to the counterparty, ENE the bank's own reported positive. Paths are rows, grid points
columns. Contract: ``engine-ccr-api`` 1.0.0, module exposure.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable

import numpy as np

from riskvalidation.ccr.hw1f import swaption_jamshidian
from riskvalidation.ccr.swaps import TIME_TOL, Swap, _accruals, cash_flow_times, reset_times
from riskvalidation.lgd_ead._checks import mat, same_length, scalar, vec

HORIZON_TOL = 1e-9
"""Tolerance, in years, when a grid point is compared with the one-year horizon."""


def _grid(grid: np.ndarray) -> np.ndarray:
    g = vec(grid, "grid")
    if g[0] < 0.0 or np.any(np.diff(g) <= 0.0):
        raise ValueError("grid must be nonnegative and strictly increasing")
    return g


def expected_exposure(values: np.ndarray, numeraire: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(ee, mc_se)`` each (K,): ``ee = mean(max(values, 0) / N)`` and its Monte Carlo
    standard error ``std / sqrt(P)`` (sample standard deviation, ``ddof = 1``). ``numeraire`` (P, K)
    is ``N(t)`` for the discounted measure, ``None`` for the undiscounted Basel style. ENE is
    ``expected_exposure(-values)``."""
    v = mat(values, "values")
    if v.shape[0] < 2:
        raise ValueError("values needs at least 2 paths")
    if numeraire is None:
        pos = np.maximum(v, 0.0)
    else:
        n = mat(numeraire, "numeraire")
        if n.shape != v.shape:
            raise ValueError(f"numeraire must have the shape of values {v.shape}, got {n.shape}")
        if np.any(n <= 0.0):
            raise ValueError("numeraire must be positive")
        pos = np.maximum(v, 0.0) / n
    return pos.mean(axis=0), pos.std(axis=0, ddof=1) / math.sqrt(v.shape[0])


def effective_ee(ee: np.ndarray, current_exposure: float = 0.0) -> np.ndarray:
    """``Effective EE_{t_k} = max(Effective EE_{t_{k-1}}, EE_{t_k})`` with
    ``Effective EE_{t_0}`` the current exposure (CRE53.12 equation 2); non-decreasing and at least ``ee``."""
    e = vec(ee, "ee")
    c = scalar(current_exposure, "current_exposure")
    return np.maximum.accumulate(np.maximum(e, c))


def _time_average(grid: np.ndarray, profile: np.ndarray, horizon: float, name: str) -> float:
    g = _grid(grid)
    p = vec(profile, name)
    same_length(grid=g, **{name: p})
    h = scalar(horizon, "horizon")
    if h <= 0.0:
        raise ValueError("horizon must be positive")
    mask = g <= h + HORIZON_TOL
    if not np.any(mask):
        raise ValueError(f"horizon: no grid point at or before {h} (the first is {g[0]})")
    dt = np.diff(np.concatenate([[0.0], g]))
    covered = float(g[mask][-1])
    if covered <= 0.0:
        raise ValueError("horizon: the covered time is zero")
    return float(np.sum((p * dt)[mask]) / covered)


def effective_epe(grid: np.ndarray, eff_ee: np.ndarray, horizon: float = 1.0) -> float:
    """The effective EPE (EEPE): ``sum_{t_k <= min(1, maturity)} Effective EE_k dt_k``
    with ``dt_k = t_k - t_{k-1}``, ``t_0 = 0`` (CRE53.13 equation 3), normalised by the
    covered time so that a constant profile returns its value."""
    return _time_average(grid, eff_ee, horizon, "eff_ee")


def epe(grid: np.ndarray, ee: np.ndarray, horizon: float = 1.0) -> float:
    """The plain time-weighted average of EE over the first year (dossier 05 C.1, ``EPE_B``)."""
    return _time_average(grid, ee, horizon, "ee")


def pfe(values: np.ndarray, quantile: float, seed: int = 0, n_bootstrap: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """``(pfe, mc_se)`` each (K,):
    ``PFE_alpha(t) = (inf {x : F_t(x) >= alpha})^+``, the positive part of the
    inverse empirical quantile. The standard error is the order-statistic one,
    ``sqrt(q (1 - q) / P) / density``, estimated by the standard deviation of the
    quantile over ``n_bootstrap`` seeded resamples of the paths."""
    v = mat(values, "values")
    q = scalar(quantile, "quantile")
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    nb = int(n_bootstrap)
    if nb != n_bootstrap or nb < 2:
        raise ValueError("n_bootstrap must be an integer of at least 2")
    p = v.shape[0]
    point = np.maximum(np.quantile(v, q, axis=0, method="inverted_cdf"), 0.0)
    rng = np.random.default_rng(seed)
    boot = np.empty((nb, v.shape[1]))
    for b in range(nb):
        boot[b] = np.quantile(v[rng.integers(0, p, size=p)], q, axis=0, method="inverted_cdf")
    return point, boot.std(axis=0, ddof=1)


def net_values(values_by_trade: np.ndarray) -> np.ndarray:
    """(P, K) the netting-set value: the sum over trades of the (S, P, K) values. The netted EE is at most the sum of
    the single-trade EE; the difference is the netting benefit."""
    c = np.array(values_by_trade, dtype=float)
    if c.ndim != 3:
        raise ValueError(f"values_by_trade must be three-dimensional (S, P, K), got shape {c.shape}")
    if c.size == 0 or not np.all(np.isfinite(c)):
        raise ValueError("values_by_trade is empty or contains non-finite values")
    return c.sum(axis=0)


def netting_benefit(ee_by_trade: np.ndarray, ee_netted: np.ndarray) -> np.ndarray:
    """(K,) ``sum_s EE_s - EE_netted``; non-negative when the EEs come from the same paths."""
    e = mat(ee_by_trade, "ee_by_trade")
    n = vec(ee_netted, "ee_netted")
    if e.shape[1] != n.size:
        raise ValueError(f"ee_netted has {n.size} points for ee_by_trade with {e.shape[1]} columns")
    return e.sum(axis=0) - n


def effective_maturity(grid: np.ndarray, ee: np.ndarray, eff_ee: np.ndarray, df: np.ndarray, cap: float = 5.0) -> float:
    """CRE53.20:
    ``M = (sum_{t_k <= 1} EffEE dt df + sum_{t_k > 1} EE dt df) / (sum_{t_k <= 1} EffEE dt df)``,
    capped at ``cap`` (5 years); 1 exactly when every point is within one year. Raises when the effective EE is zero
    over the first year (the ratio is undefined)."""
    g = _grid(grid)
    e, f, d = vec(ee, "ee"), vec(eff_ee, "eff_ee"), vec(df, "df")
    same_length(grid=g, ee=e, eff_ee=f, df=d)
    c = scalar(cap, "cap")
    if c < 1.0:
        raise ValueError("cap must be at least 1")
    if np.any(d <= 0.0):
        raise ValueError("df must be positive")
    dt = np.diff(np.concatenate([[0.0], g]))
    first = g <= 1.0 + HORIZON_TOL
    s1 = float(np.sum((f * dt * d)[first]))
    if s1 <= 0.0:
        raise ValueError("eff_ee is zero over the first year: the effective maturity is undefined")
    s2 = float(np.sum((e * dt * d)[~first]))
    return min((s1 + s2) / s1, c)


def analytic_ee_single_swap(swap: Swap, grid: np.ndarray, a: float, sigma: float, p0: Callable,
                            forward: Callable) -> np.ndarray:
    """(K,) the undiscounted EE of a single swap, ladder rung 2 (addendum R4, derived):
    ``EE_B(t) = Swaption(0; t) / P(0, t)``, the payer swaption for a payer swap and the
    receiver one for a receiver swap, expiring at ``t`` on the remaining swap (flows at or before ``t`` excluded) from
    ``swaption_jamshidian``. The identity holds on the reset dates only, so a grid point that is neither a reset date
    nor at or after the maturity raises ``ValueError``; EE is 0 from the maturity.

    The identity is a statement of the risk-neutral measure: compare the discounted simulated EE with ``P(0, t)``
    times this profile, a check of the engine and not of a real-world exposure."""
    g = _grid(grid)
    times, resets, accr = cash_flow_times(swap), reset_times(swap), _accruals(swap)
    out = np.zeros(g.size)
    for k, t in enumerate(g):
        if t >= swap.maturity - TIME_TOL:
            continue
        if np.min(np.abs(resets - t)) > TIME_TOL:
            raise ValueError(f"grid point t={t} is neither a reset date of the swap nor at or after its maturity: "
                             "the analytic EE holds on the reset dates only")
        after = times > t + TIME_TOL
        amounts = swap.notional * swap.fixed_rate * accr[after]
        amounts[-1] += swap.notional
        price = swaption_jamshidian(swap.is_payer, float(t), times[after], amounts, a, sigma, p0, forward,
                                    nominal=swap.notional)
        out[k] = price / float(np.asarray(p0(np.array(float(t)))))
    return out


def rung1_ead(v: float, c: float, addon_aggregate: float, margined: bool = False, th: float = 0.0, mta: float = 0.0,
              nica: float = 0.0) -> float:
    """Ladder rung 1 (decision D3): SA-CCR's ``1.4 (RC + PFE)``, a thin wrapper of
    ``riskvalidation.regulatory.sa_ccr.netting_set_ead`` that only ties the ladder numbering to a public name; the
    result equals its ``.ead`` exactly. The aggregate add-on is the sum of the five asset classes (CRE52.24), so it is
    carried whole in one class and the rest are zero. The module is resolved when called (it ships with unit U9-E1)."""
    agg = scalar(addon_aggregate, "addon_aggregate")
    if agg < 0.0:
        raise ValueError("addon_aggregate must be nonnegative")
    sa = importlib.import_module("riskvalidation.regulatory.sa_ccr")
    result = sa.netting_set_ead(v, c, sa.AddOnByClass(interest_rate=agg), margined, th, mta, nica)
    return result.ead

__all__ = [
    "analytic_ee_single_swap",
    "effective_ee",
    "effective_epe",
    "effective_maturity",
    "epe",
    "expected_exposure",
    "net_values",
    "netting_benefit",
    "pfe",
    "rung1_ead",
]
