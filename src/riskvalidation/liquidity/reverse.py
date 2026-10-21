"""Reverse liquidity stress: the multiplier on the run-off rates at which the survival period first falls to a target.

Source: dossier 05 E.5 rung 5, line 279 ("the run-off multiplier at which the survival period falls below a target, for
example 30 days", the named parameter `reverse_target_days`). Contract `engine-liquidity-api` 2.0.0, module reverse.

The multiplier is applied to every segment of a base scenario and capped so that each rate stays at or below 1 (decision
Q9): the cap is `1 / max(q below 1)`. The survival period `SP(m)` is non-increasing in `m`, and "survives the horizon"
counts as longer than any day. A target that is not bracketed is refused with its reason, never a number. The
survival period is an integer day, so `SP(m)` can step past the target when the CBC grows over time (a monetisation
schedule); the solve then returns the threshold multiplier (decision F2, contract 1.2.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from riskvalidation.liquidity import ladder as _ladder
from riskvalidation.liquidity._checks import check_increasing_grid
from riskvalidation.liquidity.ladder import SurvivalResult
from riskvalidation.liquidity.scenarios import Scenario

ROUNDING = 1e-6  # the multiplier is rounded up to six decimals


@dataclass(frozen=True)
class SolveResult:
    """`status` is `"solved"` or `"refused"`; a refusal carries its `reason` and no number. `multiplier` is the
    threshold `m*`; `survival_at_multiplier` is `SP(m*)`, which is below the target when the curve steps;
    `survival_just_below` is the `SurvivalResult` at `m* - 1e-6` (`survives` true and `day` None when the bank
    survives the horizon there), None for a refusal."""

    status: str
    multiplier: float | None
    survival_at_multiplier: int | None
    reason: str | None
    survival_just_below: SurvivalResult | None = None


def max_multiplier(scenario: Scenario) -> float:
    """The cap `1 / max(q < 1)` over every segment and day (decision Q9); infinite when no rate lies in `(0, 1)`, as
    scaling then changes nothing."""
    rates = scenario.rates()
    below = rates[rates < 1.0]
    top = float(below.max()) if below.size else 0.0
    return math.inf if top <= 0.0 else 1.0 / top


def _cbc(scenario: Scenario, cbc: np.ndarray) -> np.ndarray:
    c = np.array(cbc, dtype=float)
    if c.ndim != 1 or c.shape[0] != scenario.horizon or not np.all(np.isfinite(c)):
        raise ValueError(f"cbc: must be a finite array of {scenario.horizon} daily values (the horizon), "
                         f"got shape {c.shape}")
    return c


def _nlp(scenario: Scenario, cbc: np.ndarray, multiplier: float) -> np.ndarray:
    """`NLP(t)` with every rate scaled to `min(m q, 1)` (a rate already at 1 stays)."""
    rates = np.minimum(multiplier * scenario.rates(), 1.0)
    out = _ladder.behavioural_outflows(scenario.opening, rates).sum(axis=0)
    return _ladder.net_liquidity_position(cbc, out, np.zeros_like(out))


def survival_vs_multiplier(scenario: Scenario, cbc: np.ndarray, multipliers: np.ndarray) -> list[SurvivalResult]:
    """The survival period under each multiplier of the base rates: one `SurvivalResult` per multiplier, each segment's
    rate `min(m x q, 1)`. `multipliers` increase and are at least 1; `SP(m)` is non-increasing in `m`. A multiplier
    beyond
    the cap `1 / max(q < 1)` is refused (the document omits that point). `cbc` is `(H,)`, the horizon of the scenario.
    Golden: one segment, opening 1000, `q = 0.01`, constant `cbc = 100`, `H = 125`: `m = 1` gives `SP = 11`."""
    c = _cbc(scenario, cbc)
    grid = check_increasing_grid(multipliers, "multipliers", minimum=1.0)
    cap = max_multiplier(scenario)
    if grid[-1] > cap * (1.0 + 1e-12):
        raise ValueError(f"multipliers: {float(grid[-1])} is beyond the cap {cap} (1 / the largest rate below 1), so "
                         "that point is refused and the document omits it")
    return [_ladder.survival_period(_nlp(scenario, c, float(m))) for m in grid]


def solve_multiplier(scenario: Scenario, cbc: np.ndarray, target_days: int) -> SolveResult:
    """The threshold multiplier `m*`: the smallest `m` (rounded up to six decimals, never above the cap) at which
    `SP(m) <= target_days`.

    `SP(m) <= target` holds exactly when the minimum of `NLP(t)` over `t <= target` is negative, a continuous,
    non-increasing function of `m`, so the root is bracketed with `scipy.optimize.brentq` between 1 and the cap. The
    survival period is an integer day, so when the CBC grows over time `SP(m)` can step past the target (for example
    from "survives the horizon" straight to day 1): `survival_at_multiplier` is then `SP(m*)`, below the target, and no
    multiplier gives exactly the target. The round trip is the threshold property `SP(m*) <= target` and
    `SP(m* - 1e-6) > target`, whose second term is `survival_just_below`; with a CBC that does not grow it also holds
    with `SP(m*) == target`. A target that is not bracketed (the base survival period already at or below it, no
    multiplier under the cap reaches it, or a target not below the horizon) is refused with the reason, never a number.
    Golden: the one-segment bank of `survival_vs_multiplier` and `target_days = 5` gives `2.085164` (`1 - 0.9^(1/5) =
    0.0208516` over `q`, rounded up)."""
    c = _cbc(scenario, cbc)
    if target_days != int(target_days) or int(target_days) < 1:
        raise ValueError(f"target_days: must be a positive integer, got {target_days!r}")
    target = int(target_days)

    def refused(reason: str) -> SolveResult:
        return SolveResult(status="refused", multiplier=None, survival_at_multiplier=None, reason=reason)

    if target >= scenario.horizon:
        return refused(f"the target of {target} days is not below the horizon of {scenario.horizon} days")
    base = _ladder.survival_period(_nlp(scenario, c, 1.0))
    if not base.survives and base.day is not None and base.day <= target:
        return refused(f"the base survival period of {base.day} days is already at or below the target of "
                       f"{target} days")

    def worst(m: float) -> float:
        return float(_nlp(scenario, c, m)[:target].min())

    def reached(m: float) -> bool:
        return worst(m) < 0.0

    cap = max_multiplier(scenario)
    top = cap if math.isfinite(cap) else 1.0
    if not reached(top):
        where = f"the cap of {cap}" if math.isfinite(cap) else "the cap (no rate below 1 to scale)"
        return refused(f"no multiplier up to {where} reaches a survival period of {target} days or less")
    low = 1.0
    if worst(low) <= 0.0:  # a base position that touches zero inside the target window is crossed at once
        root = low
    else:
        root = float(optimize.brentq(worst, low, top, xtol=1e-14, rtol=4.0 * np.finfo(float).eps, maxiter=500))
    candidate = min(math.ceil(root / ROUNDING) * ROUNDING, top)
    candidate = min(round(candidate, 6), top)
    while not reached(candidate) and candidate < top:
        candidate = min(round(candidate + ROUNDING, 6), top)
    while candidate - ROUNDING >= low and reached(round(candidate - ROUNDING, 6)):
        candidate = round(candidate - ROUNDING, 6)
    sp = _ladder.survival_period(_nlp(scenario, c, candidate))
    if sp.day is None:  # pragma: no cover - guarded by reached(top) and the loops above
        return refused("no multiplier under the cap reaches the target")
    below = _ladder.survival_period(_nlp(scenario, c, round(candidate - ROUNDING, 6)))
    return SolveResult(status="solved", multiplier=candidate, survival_at_multiplier=int(sp.day), reason=None,
                       survival_just_below=below)


__all__ = ["ROUNDING", "SolveResult", "SurvivalResult", "max_multiplier", "solve_multiplier", "survival_vs_multiplier"]
