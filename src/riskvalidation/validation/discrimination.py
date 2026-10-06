"""Discriminatory power: does the model rank defaulters as riskier than non-defaulters?

Inputs are risk-oriented: a higher value means a riskier obligor (a PD, a score oriented to risk, or a rating
grade number where higher is worse). Ties get half credit, as in the Mann-Whitney statistic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats

from riskvalidation.core.build import make_result
from riskvalidation.core.policy import Policy
from riskvalidation.core.registry import register
from riskvalidation.core.result import TestResult
from riskvalidation.validation._common import as_binary, as_float_array

ECB_2019 = (
    "ECB (2019), Instructions for reporting the validation results of internal models: IRB Pillar I models for "
    "credit risk"
)


def _split(risk: Sequence[float], defaults: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    r = as_float_array(risk, "risk")
    y = as_binary(defaults)
    if r.shape != y.shape:
        raise ValueError("risk and defaults must have the same length")
    a, b = r[y == 1], r[y == 0]
    if a.size == 0 or b.size == 0:
        raise ValueError("discrimination needs at least one defaulter and one non-defaulter")
    return a, b


def structural_components(defaulters: np.ndarray, survivors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DeLong placements: V10[a] = share of non-defaulters ranked safer than defaulter a (ties count 1/2);
    V01[b] = share of defaulters ranked riskier than non-defaulter b. Both average to the AUC."""
    sb = np.sort(survivors)
    sa = np.sort(defaulters)
    below = np.searchsorted(sb, defaulters, side="left")
    ties_a = np.searchsorted(sb, defaulters, side="right") - below
    v10 = (below + 0.5 * ties_a) / sb.size
    above = sa.size - np.searchsorted(sa, survivors, side="right")
    ties_b = np.searchsorted(sa, survivors, side="right") - np.searchsorted(sa, survivors, side="left")
    v01 = (above + 0.5 * ties_b) / sa.size
    return v10, v01


def auc_with_variance(risk: Sequence[float], defaults: Sequence[int]) -> tuple[float, float, int, int]:
    """AUC and the ECB variance s^2 = var(V10)/|A| + var(V01)/|B| (unbiased sample variances)."""
    a, b = _split(risk, defaults)
    v10, v01 = structural_components(a, b)
    auc = float(v10.mean())
    va = float(np.var(v10, ddof=1)) if a.size > 1 else 0.0
    vb = float(np.var(v01, ddof=1)) if b.size > 1 else 0.0
    return auc, va / a.size + vb / b.size, int(a.size), int(b.size)


def cap_accuracy_ratio(risk: Sequence[float], defaults: Sequence[int]) -> float:
    """Accuracy ratio from the cumulative accuracy profile, tied groups interpolated linearly.

    AR = (area under the model CAP - 1/2) / (area under the perfect CAP - 1/2); equals 2 AUC - 1 with the same
    tie convention (BCBS WP14).
    """
    r = as_float_array(risk, "risk")
    y = as_binary(defaults)
    n, nd = r.size, int(y.sum())
    if nd == 0 or nd == n:
        raise ValueError("the CAP needs defaulters and non-defaulters")
    vals, inv = np.unique(-r, return_inverse=True)  # riskiest first
    cnt = np.bincount(inv)
    dft = np.bincount(inv, weights=y)
    x = np.concatenate([[0.0], np.cumsum(cnt) / n])
    h = np.concatenate([[0.0], np.cumsum(dft) / nd])
    area = float(np.sum((x[1:] - x[:-1]) * (h[1:] + h[:-1]) / 2.0))
    pi = nd / n
    perfect = 1.0 - pi / 2.0
    return (area - 0.5) / (perfect - 0.5)


@register(
    "disc.auc",
    family="discrimination",
    name="Area under the ROC curve",
    h0="descriptive: probability that a defaulter is ranked riskier than a non-defaulter (ties count one half)",
    reference=f"{ECB_2019}, Annex 3.1; Hanley and McNeil (1982), doi:10.1148/radiology.143.1.7063747",
    decision="metric",
)
def disc_auc(
    risk: Sequence[float],
    defaults: Sequence[int],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """AUC with the ECB standard error, the Gini/accuracy ratio 2 AUC - 1 and a normal 95% interval."""
    auc, s2, na, nb = auc_with_variance(risk, defaults)
    se = math.sqrt(s2)
    return make_result(
        "disc.auc",
        metric=auc,
        policy=policy,
        n=na + nb,
        n_events=na,
        model_id=model_id,
        segment=segment,
        inputs=(np.asarray(risk, float), np.asarray(defaults)),
        extras={
            "gini": 2.0 * auc - 1.0,
            "s2": s2,
            "se": se,
            "ci95_low": max(0.0, auc - 1.959963984540054 * se),
            "ci95_high": min(1.0, auc + 1.959963984540054 * se),
        },
    )


@register(
    "disc.auc_vs_initial",
    family="discrimination",
    name="AUC against the AUC at initial validation",
    h0="the AUC at development is smaller than the AUC in the observation period",
    reference=f"{ECB_2019}, section 2.5.4.1 and Annex 3.1",
    decision="p_value",
)
def disc_auc_vs_initial(
    auc_initial: float,
    risk: Sequence[float],
    defaults: Sequence[int],
    *,
    auc_initial_variance: float = 0.0,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """S = (AUC_init - AUC_curr) / s, p = 1 - Phi(S); the ECB treats the development AUC as a known number.

    When the development AUC was itself estimated on a sample independent of the current one, pass its variance (the
    ``s2`` of :func:`auc_with_variance` on the development sample) as ``auc_initial_variance``: the statistic then uses
    sqrt(s^2 + s_init^2), the variance of the difference of two independent estimates. This extends the ECB formula
    and the result says so. Measured with known-truth generators (riskvalidation 0.03.000), the ECB form rejects about
    11% of the time at the 5% level when the development sample is as large as the current one and the AUC has not
    changed; the extended form holds the size."""
    if not 0.0 <= auc_initial <= 1.0:
        raise ValueError("the initial AUC must lie in [0, 1]")
    if not (math.isfinite(auc_initial_variance) and auc_initial_variance >= 0.0):
        raise ValueError("the variance of the initial AUC must be a non-negative number")
    auc, s2_cur, na, nb = auc_with_variance(risk, defaults)
    s2 = s2_cur + auc_initial_variance
    if s2 <= 0.0:
        return make_result(
            "disc.auc_vs_initial",
            statistic=float("nan"),
            policy=policy,
            n=na + nb,
            n_events=na,
            model_id=model_id,
            segment=segment,
            notes="the variance of the current AUC is zero; the statistic is undefined",
            extras={"auc_initial": auc_initial, "auc_current": auc},
        )
    s = (auc_initial - auc) / math.sqrt(s2)
    extended = auc_initial_variance > 0.0
    return make_result(
        "disc.auc_vs_initial",
        statistic=s,
        p_value=float(stats.norm.sf(s)),
        policy=policy,
        alternative="discrimination deteriorated since development",
        n=na + nb,
        n_events=na,
        model_id=model_id,
        segment=segment,
        inputs=(auc_initial, np.asarray(risk, float), np.asarray(defaults), auc_initial_variance),
        notes="the development AUC's variance is added to the ECB statistic (an extension of the ECB 2019 formula): "
        "both AUCs are estimates on independent samples" if extended else "",
        extras={"auc_initial": auc_initial, "auc_current": auc, "s2": s2_cur,
                "auc_initial_variance": auc_initial_variance, "s2_total": s2},
    )


@register(
    "disc.delong",
    family="discrimination",
    name="DeLong test of two correlated AUCs",
    h0="the two models have the same AUC on the same obligors",
    reference="DeLong, DeLong and Clarke-Pearson (1988), Biometrics 44(3), doi:10.2307/2531595",
    decision="p_value",
)
def disc_delong(
    risk_1: Sequence[float],
    risk_2: Sequence[float],
    defaults: Sequence[int],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """Paired test for a champion and a challenger scored on the same sample; two-sided."""
    y = as_binary(defaults)
    r1 = as_float_array(risk_1, "risk_1")
    r2 = as_float_array(risk_2, "risk_2")
    if not (r1.shape == r2.shape == y.shape):
        raise ValueError("both risk vectors and defaults must have the same length")
    comps = [structural_components(r[y == 1], r[y == 0]) for r in (r1, r2)]
    na, nb = int(y.sum()), int(y.size - y.sum())
    if na < 2 or nb < 2:
        raise ValueError("the DeLong test needs at least two defaulters and two non-defaulters")
    v10 = np.vstack([c[0] for c in comps])
    v01 = np.vstack([c[1] for c in comps])
    s = np.cov(v10, ddof=1) / na + np.cov(v01, ddof=1) / nb
    auc1, auc2 = float(v10[0].mean()), float(v10[1].mean())
    var = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    extras = {"auc_1": auc1, "auc_2": auc2, "difference": auc1 - auc2, "variance": var}
    if var <= 0.0:
        return make_result(
            "disc.delong",
            statistic=float("nan"),
            policy=policy,
            n=int(y.size),
            n_events=na,
            model_id=model_id,
            segment=segment,
            notes="the variance of the AUC difference is not positive (identical rankings); undefined",
            extras=extras,
        )
    z = (auc1 - auc2) / math.sqrt(var)
    return make_result(
        "disc.delong",
        statistic=z,
        p_value=float(2.0 * stats.norm.sf(abs(z))),
        policy=policy,
        alternative="the AUCs differ",
        n=int(y.size),
        n_events=na,
        model_id=model_id,
        segment=segment,
        inputs=(r1, r2, y),
        extras=extras,
    )


@register(
    "disc.ks",
    family="discrimination",
    name="Kolmogorov-Smirnov separation",
    h0="descriptive: maximum distance between the risk distributions of defaulters and non-defaulters",
    reference="BCBS (2005), Working Paper 14; two-sample Kolmogorov-Smirnov statistic (SciPy ks_2samp)",
    decision="metric",
)
def disc_ks(
    risk: Sequence[float],
    defaults: Sequence[int],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """KS separation; its p-value tests identical distributions, which is rarely the question in validation."""
    a, b = _split(risk, defaults)
    res = stats.ks_2samp(a, b)
    return make_result(
        "disc.ks",
        metric=float(res.statistic),
        p_value=float(res.pvalue),
        policy=policy,
        n=int(a.size + b.size),
        n_events=int(a.size),
        model_id=model_id,
        segment=segment,
        inputs=(a, b),
        notes="the p-value tests H0 of identical distributions (no discrimination at all)",
    )


def bootstrap_ci(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    risk: Sequence[float],
    defaults: Sequence[int],
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval from a bootstrap stratified by default status (defaulters and non-defaulters are
    resampled separately, so every replicate keeps the observed default count)."""
    r = as_float_array(risk, "risk")
    y = as_binary(defaults)
    ia, ib = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if ia.size == 0 or ib.size == 0:
        raise ValueError("the stratified bootstrap needs both classes")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.concatenate([rng.choice(ia, ia.size), rng.choice(ib, ib.size)])
        vals[k] = statistic(r[idx], y[idx])
    lo, hi = np.quantile(vals, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    return float(lo), float(hi)


__all__ = [
    "auc_with_variance",
    "bootstrap_ci",
    "cap_accuracy_ratio",
    "disc_auc",
    "disc_auc_vs_initial",
    "disc_delong",
    "disc_ks",
    "structural_components",
]
