"""Input checks of the liquidity modules, each naming the offending field.

They add only what the per-package checks of `oprisk`, `lgd_ead` and `provisions` do not hold: a stock or a share of
any shape (a scalar becomes a one-element array), weights that sum to one and a strictly increasing day or multiplier
grid. Each returns the validated `float64` array (a new array: the input is never aliased) and raises
`ValueError(f"{name}: {reason}, got {value}")`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _array(x: Any, name: str) -> np.ndarray:
    """A new finite, non-empty `float64` array of any shape; a scalar becomes a one-element array."""
    try:
        a = np.array(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: must be numeric, got {x!r}") from exc
    a = np.atleast_1d(a)
    if a.size == 0:
        raise ValueError(f"{name}: must not be empty, got {x!r}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be finite, got {x!r}")
    return a


def check_nonnegative_stock(x: np.ndarray | float, name: str) -> np.ndarray:
    """Money (USD thousands) of any shape: finite, non-empty and not negative (zero is allowed)."""
    a = _array(x, name)
    if np.any(a < 0.0):
        raise ValueError(f"{name}: must be non-negative, got {np.min(a)}")
    return a


def check_unit_share(x: np.ndarray | float, name: str, closed: bool = True) -> np.ndarray:
    """Shares or rates (decimals) in the closed interval `[0, 1]` when `closed`, else in the open interval `(0, 1)`.

    A daily rate of exactly 0 or 1 is legitimate for a segment (a financial deposit leaves in full on day 1), so it
    passes when `closed`."""
    a = _array(x, name)
    if closed:
        bad = (a < 0.0) | (a > 1.0)
        interval = "[0, 1]"
    else:
        bad = (a <= 0.0) | (a >= 1.0)
        interval = "(0, 1)"
    if np.any(bad):
        raise ValueError(f"{name}: must lie in {interval}, got {a[bad][0]}")
    return a


def check_weights_sum_to_one(w: np.ndarray, name: str, tol: float = 1e-9) -> np.ndarray:
    """Segment weights (shares) of shape `(S,)`: each in `[0, 1]` and the sum within `tol` of 1."""
    a = check_unit_share(w, name)
    if a.ndim != 1:
        raise ValueError(f"{name}: must be one-dimensional, got shape {a.shape}")
    total = float(a.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"{name}: shares must sum to 1 (within {tol}), got a sum of {total}")
    return a


def check_increasing_grid(g: np.ndarray, name: str, minimum: float | None = None) -> np.ndarray:
    """A day grid or a multiplier grid: strictly increasing, optionally with every entry at least `minimum`."""
    a = _array(g, name)
    if a.ndim != 1:
        raise ValueError(f"{name}: must be one-dimensional, got shape {a.shape}")
    if minimum is not None and np.any(a < minimum):
        index = int(np.argmax(a < minimum))
        raise ValueError(f"{name}: every entry must be at least {minimum}, got {a[index]} at index {index}")
    steps = np.diff(a)
    if np.any(steps <= 0.0):
        index = int(np.argmax(steps <= 0.0)) + 1
        raise ValueError(f"{name}: must be strictly increasing, got {a[index]} after {a[index - 1]} at index {index}")
    return a
