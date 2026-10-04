"""Rating-system stability: migrations between grades and concentration in grades (ECB 2019, section 2.5.5).

Grades are numbered 1..K from best to worst quality (non-default grades only); a migration matrix holds the
counts N_ij of obligors in grade i at the start of the period and grade j at its end.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

from riskvalidation.core.build import make_result
from riskvalidation.core.policy import Policy
from riskvalidation.core.registry import register
from riskvalidation.core.result import TestResult

ECB_2019 = (
    "ECB (2019), Instructions for reporting the validation results of internal models: IRB Pillar I models for "
    "credit risk"
)


def _matrix(counts: Sequence[Sequence[float]]) -> np.ndarray:
    m = np.asarray(counts, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
        raise ValueError("the migration matrix must be square with at least two grades")
    if np.any(m < 0) or not np.all(np.isfinite(m)):
        raise ValueError("migration counts must be finite and non-negative")
    return m


def matrix_weighted_bandwidth(counts: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Upper and lower MWB (ECB 2019, section 2.5.5.1); 0 when nobody migrates, 1 at the extreme."""
    m = _matrix(counts)
    k = m.shape[0]
    n_i = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n_i[:, None] > 0, m / n_i[:, None], 0.0)
    grade = np.arange(1, k + 1)
    span = np.maximum(np.abs(grade - k), np.abs(grade - 1))
    num_u = num_l = den_u = den_l = 0.0
    for i in range(k):
        for j in range(k):
            w = n_i[i] * p[i, j]
            if j > i:
                num_u += abs(i - j) * w
                den_u += span[i] * w
            elif j < i:
                num_l += abs(i - j) * w
                den_l += span[i] * w
    upper = num_u / den_u if den_u > 0 else 0.0
    lower = num_l / den_l if den_l > 0 else 0.0
    return float(upper), float(lower)


@register(
    "rating.mwb",
    family="rating-system",
    name="Matrix weighted bandwidth",
    h0="descriptive: size of migrations away from the diagonal, upper (downgrades) and lower (upgrades)",
    reference=f"{ECB_2019}, section 2.5.5.1",
    decision="metric",
)
def rating_mwb(
    counts: Sequence[Sequence[float]],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """The metric is the larger of the upper and lower MWB; both are reported."""
    upper, lower = matrix_weighted_bandwidth(counts)
    m = _matrix(counts)
    return make_result(
        "rating.mwb",
        metric=max(upper, lower),
        policy=policy,
        n=int(m.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(m,),
        extras={"mwb_upper": upper, "mwb_lower": lower},
    )


def _holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = pvals.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adj[idx] = running
    return adj


@register(
    "rating.migration_ztests",
    family="rating-system",
    name="Migration matrix monotonicity z-tests",
    h0="transition frequencies do not increase moving away from the diagonal",
    reference=f"{ECB_2019}, section 2.5.5.2; Holm (1979) adjustment across cells added by riskvalidation",
    decision="p_value",
)
def rating_migration_ztests(
    counts: Sequence[Sequence[float]],
    *,
    alpha: float = 0.05,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """One z-test per off-diagonal cell against its neighbour closer to the diagonal; p = Phi(z).

    The ECB does not specify multiple-testing control; the result's p-value is the smallest Holm-adjusted p
    across cells (family-wise), and every cell's z, raw p and adjusted p are in ``extras``.
    """
    m = _matrix(counts)
    k = m.shape[0]
    n_i = m.sum(axis=1)
    cells, zs, ps = [], [], []
    for i in range(k):
        if n_i[i] == 0:
            continue
        p = m[i] / n_i[i]
        for j in range(k):
            if j == i:
                continue
            nb = j + 1 if j < i else j - 1  # neighbour closer to the diagonal
            var = (p[j] * (1 - p[j]) + p[nb] * (1 - p[nb]) + 2 * p[j] * p[nb]) / n_i[i]
            if var <= 0:
                continue  # "missing" in the ECB template
            z = (p[nb] - p[j]) / math.sqrt(var)
            cells.append((i + 1, j + 1))
            zs.append(z)
            ps.append(float(stats.norm.cdf(z)))
    if not cells:
        return make_result(
            "rating.migration_ztests",
            statistic=float("nan"),
            policy=policy,
            n=int(m.sum()),
            model_id=model_id,
            segment=segment,
            notes="no off-diagonal cell has a defined statistic",
        )
    pv = np.asarray(ps)
    adj = _holm(pv)
    return make_result(
        "rating.migration_ztests",
        statistic=float(np.sum(pv < alpha)),
        p_value=float(adj.min()),
        policy=policy,
        alternative="a cell farther from the diagonal is more populated than its closer neighbour",
        n=int(m.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(m, alpha),
        notes=f"statistic = number of cells with raw p below {alpha:g}",
        extras={
            "cells": [
                {"from": c[0], "to": c[1], "z": z, "p": p, "p_holm": a}
                for c, z, p, a in zip(cells, zs, ps, adj.tolist(), strict=True)
            ],
            "alpha": alpha,
        },
    )


def concentration(freq: Sequence[float]) -> tuple[float, float]:
    """Coefficient of variation and Herfindahl index of grade frequencies (ECB 2019, section 2.5.5.3).

    HI = 1 + log((CV^2 + 1) / K) / log K: 0 for a uniform distribution over K grades, 1 when all mass sits in one.
    """
    r = np.asarray(freq, dtype=float)
    if r.ndim != 1 or r.size < 2 or np.any(r < 0) or r.sum() <= 0:
        raise ValueError("grade frequencies must be a non-negative vector over at least two grades")
    r = r / r.sum()
    k = r.size
    cv = math.sqrt(k * float(np.sum((r - 1.0 / k) ** 2)))
    hi = 1.0 + math.log((cv**2 + 1.0) / k) / math.log(k)
    return cv, hi


@register(
    "rating.hhi",
    family="rating-system",
    name="Concentration in rating grades (Herfindahl index)",
    h0="the current Herfindahl index is lower than the Herfindahl index at development",
    reference=f"{ECB_2019}, section 2.5.5.3; Miller (1991), Communications in Statistics, Theory and Methods "
    "20(10)",
    decision="p_value",
)
def rating_hhi(
    freq_current: Sequence[float],
    freq_initial: Sequence[float],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """p = 1 - Phi( sqrt(K-1) (CV_curr - CV_init) / sqrt(CV_curr^2 (0.5 + CV_curr^2)) ).

    Frequencies may be obligor counts or exposures; the ECB reports both for HI.
    """
    cv_c, hi_c = concentration(freq_current)
    cv_i, hi_i = concentration(freq_initial)
    k = len(freq_current)
    if len(freq_initial) != k:
        raise ValueError("current and initial frequencies must cover the same grades")
    extras = {"cv_current": cv_c, "cv_initial": cv_i, "hi_current": hi_c, "hi_initial": hi_i, "grades": k}
    if cv_c == 0.0:
        return make_result(
            "rating.hhi",
            statistic=float("nan"),
            policy=policy,
            model_id=model_id,
            segment=segment,
            notes="current distribution is uniform (CV = 0); the ECB statistic is undefined",
            extras=extras,
        )
    s = math.sqrt(k - 1) * (cv_c - cv_i) / math.sqrt(cv_c**2 * (0.5 + cv_c**2))
    return make_result(
        "rating.hhi",
        statistic=s,
        p_value=float(stats.norm.sf(s)),
        policy=policy,
        alternative="concentration in grades increased since development",
        model_id=model_id,
        segment=segment,
        inputs=(np.asarray(freq_current, float), np.asarray(freq_initial, float)),
        extras=extras,
    )


__all__ = [
    "concentration",
    "matrix_weighted_bandwidth",
    "rating_hhi",
    "rating_migration_ztests",
    "rating_mwb",
]
