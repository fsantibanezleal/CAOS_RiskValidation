"""Data-quality checks as counted checks feeding a ``TestResult``: completeness, validity ranges, uniqueness,
consistency, timeliness and label consistency (dossier 06 C.7 l.1045-1047; I.1 l.1198; ECB guide paragraph 52(h)(vii)
and Overarching paragraph 11). Contract: ``engine-ml-api`` 1.1.0, module ``riskvalidation/ml/dataquality.py``.
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from riskvalidation.ml._checks import binary, same_rows, vec


@dataclass(frozen=True)
class DataQualityCount:
    """A counted check: ``n_failed`` of ``n_checked`` failed and ``share_ok = 1 - n_failed / n_checked``."""

    check: str
    n_checked: int
    n_failed: int
    share_ok: float

    def __post_init__(self) -> None:
        if not isinstance(self.check, str) or not self.check.strip():
            raise ValueError("check must be a non-empty string")
        if self.n_checked < 1:
            raise ValueError(f"n_checked must be at least 1, got {self.n_checked}")
        if not 0 <= self.n_failed <= self.n_checked:
            raise ValueError(f"n_failed must lie in [0, n_checked], got {self.n_failed} of {self.n_checked}")
        if abs(self.share_ok - (1.0 - self.n_failed / self.n_checked)) > 1e-15:
            raise ValueError("share_ok must equal 1 - n_failed / n_checked")


@dataclass(frozen=True)
class CompletenessCount(DataQualityCount):
    """A ``DataQualityCount`` that also carries the number of NaN cells of each column (the contract's counts per
    column)."""

    by_column: tuple[int, ...] = ()


def _count(check: str, n_checked: int, n_failed: int) -> DataQualityCount:
    return DataQualityCount(check=check, n_checked=n_checked, n_failed=n_failed, share_ok=1.0 - n_failed / n_checked)


def _table(x: Any, name: str) -> np.ndarray:
    a = np.array(x, dtype=float)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if np.any(np.isinf(a)):
        raise ValueError(f"{name} contains infinite values")
    return a


def dq_completeness(X: np.ndarray) -> DataQualityCount:
    """The cells that are not NaN: ``n_checked`` is the number of cells, ``n_failed`` the NaN cells, and ``by_column``
    the NaN count of each column (ECB 52(h)(vii); maps to SGCD shock F1)."""
    x = _table(X, "X")
    miss = np.isnan(x)
    base = _count("completeness", int(x.size), int(miss.sum()))
    return CompletenessCount(
        base.check, base.n_checked, base.n_failed, base.share_ok, by_column=tuple(int(c) for c in miss.sum(axis=0))
    )


def dq_validity(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> DataQualityCount:
    """The cells outside ``[lo, hi]`` (inclusive bounds, per column, in feature units) among the cells that are not NaN:
    a NaN is not a validity violation and is not checked here (``dq_completeness`` counts it)."""
    x = _table(X, "X")
    lo_a, hi_a = vec(lo, "lo"), vec(hi, "hi")
    for name, arr in (("lo", lo_a), ("hi", hi_a)):
        if arr.shape != (x.shape[1],):
            raise ValueError(f"{name} must have shape ({x.shape[1]},), got {arr.shape}")
    if np.any(lo_a > hi_a):
        raise ValueError("hi must not be below lo")
    seen = ~np.isnan(x)
    if not seen.any():
        raise ValueError("X holds no non-NaN cell to check")
    out = seen & ((x < lo_a) | (x > hi_a))
    return _count("validity", int(seen.sum()), int(out.sum()))


def dq_uniqueness(ids: Sequence[str]) -> DataQualityCount:
    """The repeated identifiers: the first occurrence is kept, every later one is a failure."""
    keys = [str(i) for i in ids]
    if not keys:
        raise ValueError("ids is empty")
    return _count("uniqueness", len(keys), len(keys) - len(Counter(keys)))


def dq_consistency(a: Sequence[object], b: Sequence[object]) -> DataQualityCount:
    """The rows where two views of the same field disagree (SGCD F4 reads both views); equal-length sequences."""
    if len(a) != len(b):
        raise ValueError(f"b must have the length {len(a)} of a, got {len(b)}")
    if len(a) == 0:
        raise ValueError("a is empty")
    return _count("consistency", len(a), sum(1 for u, w in zip(a, b, strict=True) if u != w))


def dq_timeliness(dates: Sequence[str], as_of: str, max_age_days: int) -> DataQualityCount:
    """The rows older than ``max_age_days`` at ``as_of`` (ISO dates; an age equal to the maximum, or a date after
    ``as_of``, is not stale; ECB 52(h)(vii): the age of the financial statements)."""
    if isinstance(max_age_days, bool) or int(max_age_days) != max_age_days or max_age_days < 0:
        raise ValueError(f"max_age_days must be a nonnegative integer, got {max_age_days}")
    if len(dates) == 0:
        raise ValueError("dates is empty")
    try:
        ref = _dt.date.fromisoformat(as_of)
        ages = [(ref - _dt.date.fromisoformat(d)).days for d in dates]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dates and as_of must be ISO dates (YYYY-MM-DD): {exc}") from exc
    return _count("timeliness", len(ages), sum(1 for age in ages if age > max_age_days))


def dq_label_consistency(X: np.ndarray, y: np.ndarray) -> DataQualityCount:
    """The rows whose feature vector is identical (exact equality) to that of a row with a different label: a planted
    contradictory pair counts both rows."""
    x = _table(X, "X")
    yy = binary(y, "y", both=False)
    same_rows(X=x, y=yy)
    labels: dict[bytes, set[int]] = {}
    keys = [(r + 0.0).tobytes() for r in x]  # + 0.0 maps -0.0 to 0.0
    for key, label in zip(keys, yy.tolist(), strict=True):
        labels.setdefault(key, set()).add(label)
    return _count("label_consistency", len(yy), sum(1 for key in keys if len(labels[key]) > 1))
