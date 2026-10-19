"""Inference on the selection rates of two groups: the pooled two-proportion z-test and Fisher's exact test (dossier 06
C.7 l.1036-1039: smaller differences can matter when statistically significant; contract `engine-fairness-api` 1.0.0).

Group 1 minus group 2 throughout. The interval of a controlled effect comes from the decision model of
`fairness.conditional`, not from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from riskvalidation.fairness._checks import count_pair


@dataclass(frozen=True)
class ZResult:
    """Pooled z statistic (group 1 minus group 2), two-sided p-value, the group sizes and a note (None unless the
    pooled rate is 0 or 1)."""

    statistic: float
    p_value: float
    n1: int
    n2: int
    note: str | None


@dataclass(frozen=True)
class FisherResult:
    """The sample odds ratio (None when undefined) and the two-sided exact p-value."""

    odds_ratio: float | None
    p_value: float


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> ZResult:
    """z = (x1/n1 - x2/n2) / sqrt(p (1 - p) (1/n1 + 1/n2)) with the pooled rate p = (x1 + x2) / (n1 + n2); the p-value
    is 2 (1 - Phi(|z|)). A pooled rate of 0 or 1 has no variance: statistic 0, p-value 1 and a note."""
    a, na = count_pair(x1, n1, "x1", "n1")
    b, nb = count_pair(x2, n2, "x2", "n2")
    pooled_num, pooled_den = a + b, na + nb
    if pooled_num in (0, pooled_den):
        return ZResult(0.0, 1.0, na, nb, "the pooled rate is 0 or 1: the statistic is 0 and the p-value 1")
    p = pooled_num / pooled_den
    se = math.sqrt(p * (1.0 - p) * (1.0 / na + 1.0 / nb))
    z = (a / na - b / nb) / se
    return ZResult(float(z), float(2.0 * stats.norm.sf(abs(z))), na, nb, None)


def fisher_exact_test(x1: int, n1: int, x2: int, n2: int) -> FisherResult:
    """Fisher's exact test on [[x1, n1 - x1], [x2, n2 - x2]]: the two-sided p-value of scipy.stats.fisher_exact and the
    sample odds ratio x1 (n2 - x2) / ((n1 - x1) x2) (not the conditional maximum likelihood estimate); None when the
    denominator is 0, where the ratio is infinite or 0/0."""
    a, na = count_pair(x1, n1, "x1", "n1")
    b, nb = count_pair(x2, n2, "x2", "n2")
    res = stats.fisher_exact([[a, na - a], [b, nb - b]], alternative="two-sided")
    denominator = (na - a) * b
    odds = None if denominator == 0 else float(a * (nb - b) / denominator)
    return FisherResult(odds, float(min(1.0, max(0.0, res.pvalue))))


__all__ = ["FisherResult", "ZResult", "fisher_exact_test", "two_proportion_z_test"]
