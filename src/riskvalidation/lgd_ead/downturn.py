"""Downturn LGD (dossier 04 F.4): the downturn period identified from a supplied macro path and Contraste's cohort-based
add-on. The EBA/RTS/2018/04 and EBA/GL/2019/03 approaches are cross-references only (the wording of the three
calibration approaches is not in the dossier: UNVERIFIED), so this module claims no regulatory approach. Contract
`engine-lgd-ead-api` 1.0.0, module downturn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.lgd_ead._checks import same_length, scalar, vec


def identify_downturn(years: np.ndarray, indicator: np.ndarray, threshold: float,
                      higher_is_worse: bool = True) -> np.ndarray:
    """A year is a downturn year when the indicator is strictly above the threshold (or strictly below it when
    `higher_is_worse` is false); a tie is not a downturn. A NaN indicator raises."""
    y = np.array(years)
    if y.ndim != 1 or y.size == 0 or not np.all(np.equal(np.mod(y.astype(float), 1), 0)):
        raise ValueError("years must be a non-empty one-dimensional array of integers")
    y = y.astype(np.int64)
    if np.any(np.diff(y) <= 0):
        raise ValueError("years must be strictly increasing")
    ind = np.array(indicator, dtype=float)
    if ind.ndim != 1:
        raise ValueError("indicator must be one-dimensional")
    if np.any(np.isnan(ind)):
        raise ValueError("indicator contains NaN")
    if ind.shape[0] != y.shape[0]:
        raise ValueError(f"years and indicator must have the same length, got {y.shape[0]} and {ind.shape[0]}")
    thr = scalar(threshold, "threshold")
    return ind > thr if higher_is_worse else ind < thr


@dataclass(frozen=True)
class DownturnResult:
    long_run: float     # default-weighted mean LGD over all cohorts
    downturn: float     # default-weighted mean LGD over the downturn cohorts
    add_on: float       # downturn - long_run


def downturn_add_on(cohort_lgd: np.ndarray, cohort_weights: np.ndarray, downturn_mask: np.ndarray) -> DownturnResult:
    """Default-weighted mean over the downturn cohorts minus the default-weighted mean over all cohorts."""
    lgd = vec(cohort_lgd, "cohort_lgd")
    w = vec(cohort_weights, "cohort_weights")
    mask = np.array(downturn_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("downturn_mask must be one-dimensional")
    same_length(cohort_lgd=lgd, cohort_weights=w, downturn_mask=mask)
    if np.any(w <= 0.0):
        raise ValueError("cohort_weights must be strictly positive (the number of defaults)")
    if not mask.any():
        raise ValueError("downturn_mask has no true entry: no downturn cohort")
    long_run = float(np.sum(w * lgd) / np.sum(w))
    down = float(np.sum(w[mask] * lgd[mask]) / np.sum(w[mask]))
    return DownturnResult(long_run=long_run, downturn=down, add_on=down - long_run)


def coverage_shortfall(estimate: float, realised: float) -> tuple[bool, float]:
    """(covered, shortfall) with covered = estimate >= realised - 1e-12 and shortfall = max(0, realised - estimate)."""
    e, r = scalar(estimate, "estimate"), scalar(realised, "realised")
    return bool(e >= r - 1e-12), max(0.0, r - e)
