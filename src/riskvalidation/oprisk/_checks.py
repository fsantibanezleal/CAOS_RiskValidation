"""Input checks shared by the operational risk modules, each naming the offending field."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

ZERO_SPREAD = 1e-12  # a spread at most this (relative to max(1, the largest value)) is zero


def vec(x: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    """A new one-dimensional finite float64 array."""
    a = np.array(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def scalar(x: Any, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} is non-finite")
    return v


def positive(x: Any, name: str) -> float:
    v = scalar(x, name)
    if v <= 0.0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


def level_in_open_unit(level: Any, name: str = "level") -> float:
    v = scalar(level, name)
    if not 0.0 < v < 1.0:
        raise ValueError(f"{name} must lie in (0, 1), got {v}")
    return v


def params_of(params: Sequence[float], size: int, law: str) -> tuple[float, ...]:
    try:
        p = tuple(float(v) for v in params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the parameters of {law!r} must be numbers") from exc
    if len(p) != size:
        raise ValueError(f"{law!r} takes {size} parameters, got {len(p)}")
    if not all(math.isfinite(v) for v in p):
        raise ValueError(f"the parameters of {law!r} must be finite")
    return p


def spread_is_zero(a: np.ndarray) -> bool:
    return float(np.std(a)) <= ZERO_SPREAD * max(1.0, float(np.max(np.abs(a))))
