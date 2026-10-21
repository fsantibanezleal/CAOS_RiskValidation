"""Fixed-float interest rate swaps priced along Hull-White paths on the single curve of the case (no spread): cash flows
by date, so a close-out window sees the flows it contains (the sticky-date approach of addendum R3).

The fixed leg pays ``N * fixed_rate * accrual`` at each fixed time. The floating leg resets on the
fixed schedule: at a reset date ``T_j`` the coupon ``N tau L_j`` with
``L_j = (1 / P(T_j, T_{j+1}) - 1) / tau`` is fixed and paid at ``T_{j+1}``.
Contract: ``engine-ccr-api`` 1.0.0, module swaps.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from riskvalidation.ccr.hw1f import bond_price
from riskvalidation.lgd_ead._checks import mat, scalar, vec

TIME_TOL = 1e-9
"""Tolerance, in years, to decide that a time is a reset or a payment date (grids built by arithmetic differ
by 1e-16)."""


@dataclass(frozen=True)
class Swap:
    trade_id: str
    notional: float          # money
    fixed_rate: float        # decimal, annual-equivalent coupon rate paid per fixed period
    is_payer: bool           # the bank pays the fixed rate
    start: float             # years
    maturity: float          # years
    fixed_frequency: int     # fixed payments per year

    def __post_init__(self) -> None:
        if not isinstance(self.trade_id, str) or not self.trade_id:
            raise ValueError("trade_id must be a non-empty string")
        if not (math.isfinite(self.notional) and self.notional > 0.0):
            raise ValueError(f"notional must be positive, got {self.notional}")
        if not math.isfinite(self.fixed_rate):
            raise ValueError("fixed_rate is non-finite")
        if not (math.isfinite(self.start) and self.start >= 0.0):
            raise ValueError(f"start must be nonnegative, got {self.start}")
        if not (math.isfinite(self.maturity) and self.maturity > self.start):
            raise ValueError(f"maturity must exceed start, got maturity={self.maturity} and start={self.start}")
        f = self.fixed_frequency
        if isinstance(f, bool) or int(f) != f or f < 1:
            raise ValueError(f"fixed_frequency must be a positive integer, got {f}")


def cash_flow_times(swap: Swap) -> np.ndarray:
    """(n,) the fixed payment times ``start + k / frequency`` up to the maturity, the maturity last
    (a last period shorter than ``1 / frequency`` ends at the maturity)."""
    f = int(swap.fixed_frequency)
    n = max(int(math.ceil((swap.maturity - swap.start) * f - 1e-9)), 1)
    times = swap.start + np.arange(1, n + 1) / f
    times[-1] = swap.maturity
    return times


def reset_times(swap: Swap) -> np.ndarray:
    """(n,) the reset dates of the floating leg: the start and every fixed payment time but the last."""
    times = cash_flow_times(swap)
    return np.concatenate([[swap.start], times[:-1]])


def _accruals(swap: Swap) -> np.ndarray:
    return cash_flow_times(swap) - reset_times(swap)


def swap_value(swap: Swap, t: float, r: np.ndarray, a: float, sigma: float, p0: Callable, forward: Callable,
               last_fixing: np.ndarray | None = None) -> np.ndarray:
    """(P,) the value to the bank at ``t`` given the short rates ``r`` at ``t``; positive is an asset, a payer value is
    minus the receiver value of the same terms. Only the flows after ``t`` are valued (sticky date).

    Fixed leg: ``sum c_i P(t, T_i)``. Floating leg: with ``t`` on a reset date or before the first one,
    ``N (P(t, T_0') - P(t, T_n))`` with ``T_0'`` the next reset at or after ``t``; strictly between two resets
    ``T_j < t < T_{j+1}`` the coupon fixed at ``T_j`` is known and the leg is
    ``N tau L_j P(t, T_{j+1}) + N (P(t, T_{j+1}) - P(t, T_n))``, which needs ``last_fixing`` (the simple rate ``L_j``
    of each path); its absence there raises ``ValueError`` (never silently approximated)."""
    tt = scalar(t, "t")
    if tt < 0.0:
        raise ValueError("t must be nonnegative")
    rr = vec(r, "r")
    times, resets, accr = cash_flow_times(swap), reset_times(swap), _accruals(swap)
    after = times > tt + TIME_TOL
    if not np.any(after):
        return np.zeros_like(rr)
    n = swap.notional

    def p(big_t: float) -> np.ndarray:
        return bond_price(rr, tt, float(big_t), a, sigma, p0, forward)

    fixed = np.zeros_like(rr)
    for big_t, tau in zip(times[after], accr[after], strict=True):
        fixed += n * swap.fixed_rate * tau * p(big_t)
    j = int(np.searchsorted(resets, tt + TIME_TOL, side="right")) - 1       # the last reset on or before t
    on_reset = j >= 0 and abs(resets[j] - tt) <= TIME_TOL
    p_end = p(times[-1])
    if j < 0 or on_reset:
        nxt = resets[0] if j < 0 else resets[j]
        head = np.ones_like(rr) if abs(nxt - tt) <= TIME_TOL else p(nxt)
        floating = n * (head - p_end)
    else:
        if last_fixing is None:
            raise ValueError(f"last_fixing is required strictly between two resets (t={tt}); the valuation is never "
                             "approximated without the path's fixing")
        lf = np.array(last_fixing, dtype=float)
        if lf.shape != rr.shape or not np.all(np.isfinite(lf)):
            raise ValueError(f"last_fixing must be a finite array of the shape of r {rr.shape}, got {lf.shape}")
        p_next = p(times[j])
        floating = n * accr[j] * lf * p_next + n * (p_next - p_end)
    return floating - fixed if swap.is_payer else fixed - floating


def par_fixed_rate(maturity: float, frequency: int, p0: Callable, start: float = 0.0) -> float:
    """``(P(0, start) - P(0, maturity)) / sum (1 / frequency) P(0, T_i)``:
    the fixed rate at which the value of a swap is 0."""
    s = Swap("par", 1.0, 0.0, False, scalar(start, "start"), scalar(maturity, "maturity"), frequency)
    times = cash_flow_times(s)
    df = np.asarray(p0(np.concatenate([[s.start], times])), dtype=float)
    return float((df[0] - df[-1]) / np.sum(_accruals(s) * df[1:]))


def simulation_grid(grid: np.ndarray, swaps: list[Swap]) -> np.ndarray:
    """The union of ``grid`` and every reset date of every swap that is not after the last grid point: the grid to
    simulate on, so that the fixing at the last reset is read from the same path (``swap_values_on_grid``)."""
    g = vec(grid, "grid")
    if np.any(np.diff(g) <= 0.0) or g[0] < 0.0:
        raise ValueError("grid must be nonnegative and strictly increasing")
    parts = [g] + [reset_times(s)[reset_times(s) <= g[-1] + TIME_TOL] for s in swaps]
    merged = np.sort(np.concatenate(parts))
    keep = np.concatenate([[True], np.diff(merged) > TIME_TOL])
    return merged[keep]


def swap_values_on_grid(swaps: list[Swap], grid: np.ndarray, short_rate: np.ndarray, a: float, sigma: float,
                        p0: Callable, forward: Callable) -> np.ndarray:
    """(S, P, K) the value of each swap along each path at each grid point; the sum over swaps is the netting-set value
    (netting is on the sum, not on the sum of positive parts).

    ``short_rate`` (P, K) comes from ``simulate_short_rate`` on ``grid``, which must therefore hold every reset date the
    valuation reads (use ``simulation_grid``); between two resets the fixing is the simple rate of the path at the last
    reset, ``L_j = (1 / P(T_j, T_{j+1}; r(T_j)) - 1) / tau``, and a
    reset date missing from ``grid`` raises ``ValueError``."""
    if len(swaps) == 0:
        raise ValueError("swaps must not be empty")
    g = vec(grid, "grid")
    if np.any(np.diff(g) <= 0.0) or g[0] < 0.0:
        raise ValueError("grid must be nonnegative and strictly increasing")
    sr = mat(short_rate, "short_rate")
    if sr.shape[1] != g.size:
        raise ValueError(f"short_rate has {sr.shape[1]} columns for a grid of {g.size} points")
    out = np.zeros((len(swaps), sr.shape[0], g.size))
    for s_i, sw in enumerate(swaps):
        times, resets = cash_flow_times(sw), reset_times(sw)
        for k, t in enumerate(g):
            if t >= sw.maturity - TIME_TOL:
                continue
            j = int(np.searchsorted(resets, t + TIME_TOL, side="right")) - 1
            fixing = None
            if j >= 0 and abs(resets[j] - t) > TIME_TOL:                       # strictly between two resets
                idx = int(np.argmin(np.abs(g - resets[j])))
                if abs(g[idx] - resets[j]) > TIME_TOL:
                    raise ValueError(f"grid must contain the reset date {resets[j]} of swap {sw.trade_id!r} "
                                     "(simulate on simulation_grid)")
                tau = float(times[j] - resets[j])
                p_fix = bond_price(sr[:, idx], float(resets[j]), float(times[j]), a, sigma, p0, forward)
                fixing = (1.0 / p_fix - 1.0) / tau
            out[s_i, :, k] = swap_value(sw, float(t), sr[:, k], a, sigma, p0, forward, fixing)
    return out

__all__ = [
    "Swap",
    "cash_flow_times",
    "par_fixed_rate",
    "reset_times",
    "simulation_grid",
    "swap_value",
    "swap_values_on_grid",
]
