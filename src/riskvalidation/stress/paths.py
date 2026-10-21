"""Scenario path handling: the shape of a published path (its change from its jump-off) transposed onto the last
observed value, reversion to the baseline after the published horizon, annual to quarterly conversion, deviation paths,
and the check that a path's vintage does not postdate the observation date (SDD l.169).

The engine form of the rule `cases/c06/macro.py` applies in the product (R14 of the unit sizing); `REVERSION_QUARTERS`
is that module's product convention (a convention, not a source value). A path keeps the unit of its source (percent for
the Federal Reserve and EBA paths).
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

#: the product convention of `cases/c06/macro.py`; the models field `capital_engine.reversion_quarters`
REVERSION_QUARTERS = 12


def _vec(x: object, name: str) -> np.ndarray:
    a = np.array(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _num(x: float, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} is non-finite")
    return v


def transpose_path(levels: np.ndarray, source_jump_off: float, target_value: float, n: int) -> np.ndarray:
    """`target_value + (levels - source_jump_off)` for the first `n` periods, aligned by their distance from the jump-
    off
    (decision Q6): the shape of the published path on the last observed value."""
    lv = _vec(levels, "levels")
    jump, target = _num(source_jump_off, "source_jump_off"), _num(target_value, "target_value")
    if int(n) != n or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if n > lv.size:
        raise ValueError(f"n = {n} is above the {lv.size} periods of levels (extend with revert_to_baseline)")
    return target + (lv[: int(n)] - jump)


def revert_to_baseline(path: np.ndarray, baseline_tail: float, n: int, reversion_quarters: int) -> np.ndarray:
    """Past the published horizon the path reverts linearly to the baseline tail: for `k > len(path)`,
    `w = min(1, (k - len(path)) / reversion_quarters)` and `value = path[-1] + w (baseline_tail - path[-1])`; the first
    `len(path)` values are unchanged. `n` is the number of periods returned (`n >= len(path)`)."""
    p = _vec(path, "path")
    tail = _num(baseline_tail, "baseline_tail")
    if int(reversion_quarters) != reversion_quarters or reversion_quarters < 1:
        raise ValueError(f"reversion_quarters must be an integer of at least 1, got {reversion_quarters!r}")
    if int(n) != n or n < p.size:
        raise ValueError(f"n = {n} is below the {p.size} periods of path")
    out = np.empty(int(n))
    out[: p.size] = p
    k = np.arange(p.size + 1, int(n) + 1)
    w = np.minimum(1.0, (k - p.size) / float(reversion_quarters))
    out[p.size :] = p[-1] + w * (tail - p[-1])
    return out


def annual_to_quarterly(year_end_levels: np.ndarray, jump_off_level: float) -> np.ndarray:
    """Quarter `k` (1 to 4) of year `y` is `prev + (cur - prev) k / 4`: the year-end levels joined linearly from the
    jump-off level; the fourth quarter equals the year-end level. Whether the EBA workbook values are year averages or
    year-end values is
    UNVERIFIED in the research record (F6); they are treated as year-end levels."""
    ye = _vec(year_end_levels, "year_end_levels")
    prev = np.concatenate([[_num(jump_off_level, "jump_off_level")], ye[:-1]])
    k = np.arange(1, 5) / 4.0
    out = prev[:, None] + (ye - prev)[:, None] * k[None, :]
    out[:, 3] = ye  # the fourth quarter is the year-end level itself, with no rounding of the join
    return out.reshape(-1)


def level_from_deviation(baseline: np.ndarray, deviation: np.ndarray, kind: str) -> np.ndarray:
    """A level from a deviation of the baseline run: `plus` is `baseline + deviation` (rates, unemployment), `percent`
    is
    `baseline * (1 + deviation / 100)` (GDP, a percent difference). NGFS scenario runs hold deviations from the Baseline
    run (F8)."""
    b, d = np.array(baseline, dtype=float), np.array(deviation, dtype=float)
    if kind not in ("plus", "percent"):
        raise ValueError(f"unknown kind {kind!r}: expected 'plus' or 'percent'")
    if b.shape != d.shape:
        raise ValueError(f"baseline and deviation must have the same shape, got {b.shape} and {d.shape}")
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(d))):
        raise ValueError("baseline or deviation contains non-finite values")
    return b + d if kind == "plus" else b * (1.0 + d / 100.0)


def _date(s: str, name: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(str(s))
    except ValueError as exc:
        raise ValueError(f"{name} {s!r} is not an ISO date") from exc


def check_vintage(vintage_date: str, observation_date: str) -> None:
    """Reject a path whose vintage postdates the model's observation date (SDD l.169, the `macro_path` family rule);
    equal
    dates pass."""
    v, o = _date(vintage_date, "vintage_date"), _date(observation_date, "observation_date")
    if v > o:
        raise ValueError(f"the vintage {vintage_date} postdates the observation date {observation_date}")

__all__ = [
    "annual_to_quarterly",
    "check_vintage",
    "level_from_deviation",
    "revert_to_baseline",
    "transpose_path",
]
