"""Split conformal prediction (dossier 06 C.7 l.1019-1028; I.1 l.1194).

Scores ``s_i = s(x_i, y_i)`` on n calibration points (``1 - p_hat_{y_i}(x_i)`` for classes); ``q_hat`` is the k-th
smallest score with ``k = ceil((n + 1)(1 - alpha))``; the set ``C(x) = {y : s(x, y) <= q_hat}``. Under exchangeability
``P(Y in C(X)) >= 1 - alpha`` (Angelopoulos and Bates, arXiv 2107.07511; Vovk, Gammerman and Shafer 2022,
doi:10.1007/978-3-031-06649-8). The validation test is the empirical coverage on out-of-time or shifted data against
``1 - alpha`` with a binomial test on the coverage indicators; a coverage loss is read as an exchangeability violation,
never as proof. Contract: ``engine-ml-api`` 1.0.0, module ``riskvalidation/ml/conformal.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import stats

from riskvalidation.ml._checks import binary, closed_unit, decimal, vec


def _alpha(alpha: float) -> Fraction:
    a = decimal(alpha, "alpha")
    if not 0 < a < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return a


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The k-th smallest score, ``k = ceil((n + 1)(1 - alpha))``, with k computed on the exact decimal ``alpha``
    (``fractions.Fraction(str(alpha))``) so that ``101 * 0.9`` cannot round the wrong way.

    Equal to ``numpy.sort(scores)[k - 1]`` and to ``numpy.quantile(scores, k / n, method="inverted_cdf")``; the
    ``higher`` method of ``numpy.quantile`` is not the reference (it returns the (k + 1)-th order statistic). ``k > n``
    returns ``+inf`` (the set is the whole label space); the quantile is monotone in ``alpha``."""
    s = vec(scores, "scores")
    a = _alpha(alpha)
    n = s.size
    k = math.ceil((n + 1) * (1 - a))
    if k > n:
        return math.inf
    return float(np.sort(s)[k - 1])


def conformal_sets(p1: np.ndarray, qhat: float) -> np.ndarray:
    """The prediction sets as a boolean array (n, 2): column y is true when the score of label y, ``1 - p(y)``, is at
    most ``qhat``. A set may be empty or hold both labels; ``qhat = +inf`` gives both labels everywhere."""
    p = vec(p1, "p1")
    closed_unit(p, "p1")
    q = float(qhat)
    if math.isnan(q):
        raise ValueError("qhat is NaN")
    return np.column_stack([p <= q, 1.0 - p <= q])  # the score of label 0 is 1 - (1 - p1) = p1


@dataclass(frozen=True)
class CoverageResult:
    """The empirical coverage of ``n`` labels, the target ``1 - alpha`` and the two-sided binomial p-value."""

    n: int
    coverage: float
    target: float
    p_value: float


def conformal_coverage(sets: np.ndarray, y: np.ndarray, alpha: float) -> CoverageResult:
    """The empirical coverage (the share of rows whose set holds the true label), the target ``1 - alpha`` and the
    two-sided ``scipy.stats.binomtest`` p-value of the coverage indicators against ``1 - alpha``."""
    st = np.asarray(sets)
    if st.ndim != 2 or st.shape[1] != 2 or st.dtype != bool or st.shape[0] == 0:
        raise ValueError("sets must be a non-empty boolean array of shape (n, 2)")
    yy = binary(y, "y", both=False)
    if yy.shape[0] != st.shape[0]:
        raise ValueError("y must have one label per row of sets")
    a = _alpha(alpha)
    hit = st[np.arange(len(yy)), yy]
    k, n = int(hit.sum()), int(len(yy))
    target = float(1 - a)
    p = float(stats.binomtest(k, n, target, alternative="two-sided").pvalue)
    return CoverageResult(n=n, coverage=k / n, target=target, p_value=min(1.0, max(0.0, p)))
