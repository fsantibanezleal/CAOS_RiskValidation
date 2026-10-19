"""Input checks of the ML modules, each naming the offending field (the shared array checks are the LGD and EAD ones)."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np

from riskvalidation.lgd_ead._checks import closed_unit, mat, scalar, vec

__all__ = ["binary", "closed_unit", "decimal", "index", "mat", "nonneg", "predictions", "same_rows", "scalar", "vec"]


def binary(y: Any, name: str = "y", *, both: bool = True) -> np.ndarray:
    """A new int64 vector of 0 and 1; with ``both`` each class must be present."""
    a = np.asarray(y)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isin(a, (0, 1))):
        raise ValueError(f"{name} must contain only 0 and 1")
    out = a.astype(np.int64)
    if both and (out.min() == out.max()):
        raise ValueError(f"{name} must contain both classes")
    return out


def predictions(predict: Callable[[np.ndarray], Any], x: np.ndarray, name: str = "predict", *, unit: bool = True) -> np.ndarray:
    """The predict callable's output on ``x``: shape (n,), finite and, for a probability, in [0, 1]."""
    out = np.array(predict(x), dtype=float)
    if out.shape != (x.shape[0],):
        raise ValueError(f"{name} must return an array of shape ({x.shape[0]},), got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} returned non-finite values")
    if unit and (out.min() < 0.0 or out.max() > 1.0):
        raise ValueError(f"{name} must return probabilities in [0, 1]")
    return out


def same_rows(**arrays: np.ndarray) -> None:
    sizes = {k: v.shape[0] for k, v in arrays.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"{', '.join(sizes)} must have the same number of rows, got {sizes}")


def nonneg(x: Any, name: str) -> float:
    v = scalar(x, name)
    if v < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {v}")
    return v


def index(x: Any, name: str, hi: int, *, lo: int = 0) -> int:
    """An integer in [lo, hi)."""
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {x!r}")
    if not lo <= int(x) < hi:
        raise ValueError(f"{name} must lie in [{lo}, {hi}), got {x}")
    return int(x)


def decimal(x: float, name: str) -> Fraction:
    """The exact decimal a float was written as (``Fraction(str(x))``), so that 101 * 0.9 cannot round the wrong way."""
    return Fraction(str(float(scalar(x, name))))
