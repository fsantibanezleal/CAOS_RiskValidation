"""Private numerics shared by the sequential monitors: input checks and the log-space mixture of likelihood ratios."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import special

VERSION_LABEL = "contraste-sequential-monitors 1.0.0"


def check_alpha_level(alpha_level: float) -> float:
    """The size of the monitor: a number strictly between 0 and 1."""
    a = float(alpha_level)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha_level must lie strictly between 0 and 1, got {alpha_level}")
    return a


def check_open_unit(name: str, value: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and 0.0 < v < 1.0):
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")
    return v


def flags(values: object, name: str) -> np.ndarray:
    """A one-dimensional 0/1 sequence as a boolean array; NaN, other values and other shapes raise."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.dtype == bool:
        return arr
    try:
        num = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must hold flags (0, 1, False or True)") from exc
    if not np.all(np.isfinite(num)):
        raise ValueError(f"{name} holds a NaN or an infinite value: a missing observation is not a non-hit")
    if not np.all((num == 0.0) | (num == 1.0)):
        raise ValueError(f"{name} must hold flags (0, 1, False or True)")
    return num.astype(bool)


def counts(values: object, name: str) -> np.ndarray:
    """A two-dimensional array of non-negative whole numbers (as floats); NaN, fractions and negatives raise."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional (steps by grades), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} holds a NaN or an infinite value")
    if np.any(arr < 0) or np.any(arr != np.round(arr)):
        raise ValueError(f"{name} must hold non-negative whole numbers")
    return arr


def multiplier_grid(multipliers: Sequence[float] | None, direction: str, default_low: Sequence[float],
                    default_high: Sequence[float]) -> np.ndarray:
    """The alternatives as multiplicative factors of the null parameter: explicit, or the default grid of one side
    or of both."""
    if multipliers is not None:
        m = np.asarray(multipliers, dtype=float)
        if m.ndim != 1 or m.size == 0 or not np.all(np.isfinite(m)) or np.any(m <= 0.0) or np.any(m == 1.0):
            raise ValueError("multipliers must be a non-empty list of positive numbers different from 1")
        return np.unique(m)
    if direction == "upper":
        return np.asarray(sorted(default_high), dtype=float)
    if direction == "lower":
        return np.asarray(sorted(default_low), dtype=float)
    if direction == "two-sided":
        return np.asarray(sorted([*default_low, *default_high]), dtype=float)
    raise ValueError(f"direction must be 'upper', 'lower' or 'two-sided', got {direction!r}")


def log_mixture(cum_log_lr: np.ndarray, log_weights: np.ndarray) -> np.ndarray:
    """`log sum_k w_k exp(cum_log_lr[k, t])` along the last axis; the weights sum to one, so the value at t = 0 is 0."""
    return special.logsumexp(cum_log_lr + log_weights[:, None], axis=0)


def frozen(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    out.setflags(write=False)
    return out
