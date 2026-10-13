"""Input checks shared by the LGD and EAD modules, each naming the offending field."""

from __future__ import annotations

from typing import Any

import numpy as np


def vec(x: Any, name: str, *, allow_nan: bool = False, allow_empty: bool = False) -> np.ndarray:
    """A new one-dimensional float64 array; non-finite values raise (NaN may be allowed, infinities never)."""
    a = np.array(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty")
    bad = np.isinf(a) if allow_nan else ~np.isfinite(a)
    if np.any(bad):
        raise ValueError(f"{name} contains non-finite values")
    return a


def mat(x: Any, name: str) -> np.ndarray:
    """A new two-dimensional finite float64 array."""
    a = np.array(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def same_length(**arrays: np.ndarray) -> None:
    sizes = {k: v.shape[0] for k, v in arrays.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"{', '.join(sizes)} must have the same length, got {sizes}")


def scalar(x: Any, name: str) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{name} is non-finite")
    return v


def closed_unit(a: np.ndarray, name: str) -> None:
    if np.any(a < 0.0) or np.any(a > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
