"""Input checks shared by the provisions modules.

Every check names the field in its message, so that a refusal says which input broke which rule of the contract
(`engine-provisions-api`, conventions).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

WEIGHT_TOLERANCE = 1e-9


def float_array(x: Any, name: str, ndim: int | None = None) -> np.ndarray:
    """A new float64 array, finite, optionally of a fixed number of dimensions. Empty arrays are allowed."""
    a = np.array(x, dtype=float)  # a copy: inputs are never aliased
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimension(s), got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def open_unit(x: Any, name: str) -> np.ndarray:
    """A PD at an estimate site: strictly between 0 and 1."""
    a = np.array(x, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0) or np.any(a >= 1.0):
        raise ValueError(f"{name} must lie strictly between 0 and 1 (open interval)")
    return a


def closed_unit(x: Any, name: str) -> np.ndarray:
    """A rate in [0, 1]: LGD, CCF, a coverage rate, a share."""
    a = np.array(x, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0.0) or np.any(a > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return a


def non_negative(x: Any, name: str) -> np.ndarray:
    a = np.array(x, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return a


def scalar_open_unit(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or not 0.0 < v < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1 (open interval), got {x!r}")
    return v


def months_array(months: Any, name: str = "months") -> np.ndarray:
    """Strictly increasing positive integer end months, as int64."""
    raw = np.asarray(months)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array of integer months")
    if raw.dtype.kind not in "iu":
        f = raw.astype(float)
        if not np.all(np.isfinite(f)) or np.any(f != np.floor(f)):
            raise ValueError(f"{name} must hold integer months")
        raw = f
    m = raw.astype(np.int64)
    if m[0] <= 0 or np.any(np.diff(m) <= 0):
        raise ValueError(f"{name} must be positive and strictly increasing")
    return m


def same_length(arrays: dict[str, np.ndarray]) -> int:
    sizes = {k: v.shape[0] for k, v in arrays.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"{', '.join(sizes)} must have the same length, got {sizes}")
    return next(iter(sizes.values()))


def weights_array(weights: Any, n: int | None = None, name: str = "weights") -> np.ndarray:
    """Probability weights: in [0, 1], summing to 1 within 1e-9, at least two scenarios (IFRS 9 B5.5.42)."""
    w = np.array(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {w.shape}")
    if w.size < 2:
        raise ValueError(f"{name}: at least two scenarios are required when scenarios are used (IFRS 9 B5.5.42)")
    if n is not None and w.size != n:
        raise ValueError(f"{name} has {w.size} entries for {n} scenarios")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or np.any(w > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    if abs(float(np.sum(w)) - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"{name} must sum to 1 within {WEIGHT_TOLERANCE:g}, got {float(np.sum(w))!r}")
    return w


def bool_array(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype != bool:
        raise ValueError(f"{name} must be a boolean array, got dtype {a.dtype}")
    return a.copy()
