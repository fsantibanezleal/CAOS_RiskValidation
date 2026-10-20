"""Input checks shared by the fairness modules, each naming the offending field.

Two kinds of refusal. A value that is malformed (a decision that is not 0 or 1, rows of unequal length, a non-finite
entry) raises `ValueError` naming the field. A value that is well formed but too thin to compute from (a single group,
a group without both outcome classes, a group of no rows) raises `InsufficientData`, a `ValueError` subclass, so the
`TestResult` functions can return the `not_evaluated` light with the reason instead of failing.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class InsufficientData(ValueError):
    """Valid data too thin for the quantity asked: a validation test returns `not_evaluated` with this reason."""


def binary(x: Any, name: str) -> np.ndarray:
    """A new one-dimensional int64 array holding only 0 and 1."""
    a = np.asarray(x)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array of 0 and 1")
    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values; it must hold only 0 and 1")
    if a.dtype.kind not in "biuf" or not np.all(np.isin(a, (0, 1))):
        raise ValueError(f"{name} must contain only 0 and 1")
    return a.astype(np.int64)


def outcome(y: Any, name: str = "y") -> np.ndarray:
    """The outcome of the outcome-conditioned tests; a missing outcome is refused with the reason."""
    if y is None:
        raise ValueError(
            f"{name} is missing: the dataset declares no outcome (outcome: none), and the outcome-conditioned tests "
            "(equalized odds, equal opportunity, calibration within groups, AUC by group) never run on it"
        )
    return binary(y, name)


def group_labels(group: Any, n: int, name: str = "group") -> np.ndarray:
    """The group of each row, length `n`: numeric and boolean labels are kept as given (and sort numerically), other
    labels become strings. A group is named by `str(label)` in every result."""
    g = np.asarray(group)
    if g.ndim != 1 or g.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array of labels")
    if g.shape[0] != n:
        raise ValueError(f"{name} has {g.shape[0]} rows but {n} were expected")
    if g.dtype.kind == "f" and not np.all(np.isfinite(g)):
        raise ValueError(f"{name} contains a missing label")
    if g.dtype.kind in "biuf":
        return g
    if g.dtype.kind == "O":
        if any(v is None or (isinstance(v, float) and v != v) for v in g):
            raise ValueError(f"{name} contains a missing label")
        return np.array([str(v) for v in g])
    return g.astype(str)


def group_names(g: np.ndarray) -> list[str]:
    """The names of the groups present, in label order."""
    return [str(v) for v in np.unique(g)]


def same_rows(a: np.ndarray, name_a: str, b: np.ndarray, name_b: str) -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"{name_b} has {b.shape[0]} rows but {name_a} has {a.shape[0]}")


def count(v: Any, name: str) -> int:
    """An integer count (a Python or NumPy integer, or a float with an integral value)."""
    if isinstance(v, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer count, got a boolean")
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v).is_integer():
        return int(v)
    raise ValueError(f"{name} must be an integer count, got {v!r}")


def count_pair(x: Any, n: Any, x_name: str, n_name: str) -> tuple[int, int]:
    """A favourable count and a group size with 0 <= x <= n; a group of no rows is `InsufficientData`."""
    xi, ni = count(x, x_name), count(n, n_name)
    if ni < 1:
        raise InsufficientData(f"{n_name} must be at least 1: the group has no rows")
    if not 0 <= xi <= ni:
        raise ValueError(f"{x_name} must lie in [0, {n_name}] = [0, {ni}], got {xi}")
    return xi, ni


def finite_matrix(
    x: Any, name: str, *, rows: int | None = None, rows_of: str = "the outcome", allow_no_columns: bool = False
) -> np.ndarray:
    """A new two-dimensional finite float64 array; a one-dimensional input is read as one column."""
    a = np.array(x, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional, got shape {a.shape}")
    if a.shape[1] == 0 and not allow_no_columns:
        raise ValueError(f"{name} has no columns")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    if rows is not None and a.shape[0] != rows:
        raise ValueError(f"{name} has {a.shape[0]} rows but {rows_of} has {rows}")
    return a


def unit_interval_open(v: float, name: str) -> float:
    """A float strictly inside (0, 1)."""
    f = float(v)
    if not (np.isfinite(f) and 0.0 < f < 1.0):
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {v!r}")
    return f
