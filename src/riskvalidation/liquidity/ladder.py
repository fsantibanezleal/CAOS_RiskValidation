"""The contractual and behavioural ladders, the net liquidity position and the survival period.

Sources: dossier 05 E.1 lines 234 to 236 (ECB LiST 2019: `NLP(t) = CBC(t) - sum over s <= t of (O(s) - I(s))`, the
survival
period as the first day the NLP turns negative, the cliff effect `(NLP(35) - NLP(30)) / TA`), E.5 rungs 1 and 2; dossier
01
G.3 lines 642 and 643 (the SRP50 contractual maturity mismatch tool); the U11 research addendum L1.6. Contract
`engine-liquidity-api` 1.2.0, module ladder.

Money is USD thousands, days are integers counted from 1 (day 1 is the first relevant day after the valuation date).
A balance that never turns the net liquidity position negative has no survival day: `survival_period` returns
"survives the horizon" with the horizon, never a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.liquidity._checks import check_nonnegative_stock, check_unit_share
from riskvalidation.validation._common import as_float_array


@dataclass(frozen=True)
class SurvivalResult:
    """The survival period: `day` is the first day the NLP is negative, `None` with `survives` true otherwise."""

    survives: bool
    day: int | None
    horizon: int


def _vec(x: np.ndarray, name: str) -> np.ndarray:
    return as_float_array(np.array(x, dtype=float), name)


def _same(**arrays: np.ndarray) -> None:
    sizes = {k: v.shape[0] for k, v in arrays.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"{', '.join(sizes)} must have the same length, got {sizes}")


def contractual_gaps(inflows: np.ndarray, outflows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The contractual maturity mismatch: `gap = inflows - outflows` per bucket and `cumulative_gap = cumsum(gap)`
    (dossier 05 E.5 rung 1; SRP50). The last cumulative gap is the sum of all gaps. Money per bucket, shape `(B,)`.
    Golden: inflows `[10, 20, 30]`, outflows `[40, 10, 5]` give `[-30, 10, 25]` and `[-30, -20, 5]`."""
    i, o = _vec(inflows, "inflows"), _vec(outflows, "outflows")
    _same(inflows=i, outflows=o)
    gap = i - o
    return gap, np.cumsum(gap)


def behavioural_outflows(opening: np.ndarray, daily_rates: np.ndarray) -> np.ndarray:
    """Outflows by segment and day: each row is `opening x diff(cumulative_outflow_path(rates), prepend=0)`, computed as
    `opening x stock share of the day before x rate` (the same value without a subtraction of nearly equal numbers).
    The row sum is `opening x cumulative_outflow_path(rates)[-1]`; the stock never goes negative (a rate of 1 empties
    it).
    `opening` `(S,)` money, `daily_rates` `(S, H)` decimals per relevant day. Golden: `[100]` with `[[0.1, 0.1]]`
    gives `[[10, 9]]`."""
    op = check_nonnegative_stock(opening, "opening")
    if op.ndim != 1:
        raise ValueError(f"opening: must be one-dimensional, got shape {op.shape}")
    q = check_unit_share(daily_rates, "daily_rates")
    if q.ndim != 2:
        raise ValueError(f"daily_rates: must be two-dimensional (S, H), got shape {q.shape}")
    if q.shape[0] != op.shape[0]:
        raise ValueError(f"opening has {op.shape[0]} segments but daily_rates has {q.shape[0]} rows")
    before = np.cumprod(1.0 - q, axis=1)
    before = np.concatenate([np.ones((q.shape[0], 1)), before[:, :-1]], axis=1)
    return op[:, None] * before * q


def net_liquidity_position(cbc: np.ndarray, outflows: np.ndarray, inflows: np.ndarray) -> np.ndarray:
    """`NLP(t) = CBC(t) - sum over s <= t of (O(s) - I(s))` (dossier 05 E.1 line 234), `(H,)` money by day: the CBC
    available on day t and the stressed outflows and inflows of day s. Equal length required; inputs are not mutated.
    Golden: `cbc = [100]*4`, outflows `[30, 40, 50, 60]`, inflows `[0, 10, 0, 0]` give `[70, 40, -10, -70]`."""
    c, o, i = _vec(cbc, "cbc"), _vec(outflows, "outflows"), _vec(inflows, "inflows")
    _same(cbc=c, outflows=o, inflows=i)
    return c - np.cumsum(o - i)


def survival_period(nlp: np.ndarray) -> SurvivalResult:
    """`SP = min{t : NLP(t) < 0}` counted from day 1 (dossier 05 E.1 line 235). A zero NLP is not negative. A position
    that never turns negative returns `survives = True`, `day = None` and `horizon = H`: "survives the horizon" with the
    horizon, never a number."""
    x = _vec(nlp, "nlp")
    negative = np.flatnonzero(x < 0.0)
    if negative.size == 0:
        return SurvivalResult(survives=True, day=None, horizon=int(x.size))
    return SurvivalResult(survives=False, day=int(negative[0]) + 1, horizon=int(x.size))


def minimum_nlp(nlp: np.ndarray) -> tuple[float, int]:
    """The minimum NLP and the first day it is reached (day 1 is the first)."""
    x = _vec(nlp, "nlp")
    return float(x.min()), int(np.argmin(x)) + 1


def cliff_effect(nlp: np.ndarray, total_assets: float, near: int = 30, far: int = 35) -> float:
    """`(NLP(far) - NLP(near)) / TA` (dossier 05 E.1 line 236), a diagnostic of LCR window optimisation: a share of
    total
    assets, negative when the position drops. `near` and `far` are the named parameter `cliff_days`; the LiST note does
    not say whether they count calendar or relevant days (UNVERIFIED). A horizon shorter than `far` raises naming it.
    Golden: `NLP(30) = 100`, `NLP(35) = 40`, `TA = 1000` gives `-0.06`."""
    x = _vec(nlp, "nlp")
    ta = float(total_assets)
    if not np.isfinite(ta) or ta <= 0.0:
        raise ValueError(f"total_assets: must be positive, got {total_assets}")
    if int(near) != near or int(far) != far or not 1 <= near < far:
        raise ValueError(f"near and far: must be integers with 1 <= near < far, got {near} and {far}")
    if x.size < far:
        raise ValueError(f"far: the horizon is {x.size} days, shorter than far = {far}, "
                         "so the cliff effect is undefined")
    return float((x[far - 1] - x[near - 1]) / ta)


__all__ = ["SurvivalResult", "behavioural_outflows", "cliff_effect", "contractual_gaps", "minimum_nlp",
           "net_liquidity_position", "survival_period"]
