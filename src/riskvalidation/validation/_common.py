"""Input checks shared by the validation tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def as_float_array(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def as_binary(y: Any, name: str = "defaults") -> np.ndarray:
    a = np.asarray(y)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    vals = np.unique(a)
    if not np.all(np.isin(vals, (0, 1))):
        raise ValueError(f"{name} must contain only 0 and 1, got values {vals[:5].tolist()}")
    return a.astype(int)


def check_pd(pd: Any, name: str = "pd") -> np.ndarray:
    a = np.atleast_1d(np.asarray(pd, dtype=float))
    if np.any(~np.isfinite(a)) or np.any(a <= 0.0) or np.any(a >= 1.0):
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return a


def check_counts(n: Any, d: Any) -> tuple[np.ndarray, np.ndarray]:
    n_a = np.atleast_1d(np.asarray(n))
    d_a = np.atleast_1d(np.asarray(d))
    if n_a.shape != d_a.shape:
        raise ValueError("obligor and default counts must have the same shape")
    for arr, name in ((n_a, "obligor counts"), (d_a, "default counts")):
        if not np.all(np.equal(np.mod(arr, 1), 0)) or np.any(arr < 0):
            raise ValueError(f"{name} must be non-negative integers")
    if np.any(d_a > n_a):
        raise ValueError("default counts cannot exceed obligor counts")
    return n_a.astype(np.int64), d_a.astype(np.int64)


def labels(k: int, given: Sequence[str] | None) -> list[str]:
    if given is None:
        return [f"grade {i + 1}" for i in range(k)]
    if len(given) != k:
        raise ValueError(f"expected {k} labels, got {len(given)}")
    return [str(g) for g in given]
