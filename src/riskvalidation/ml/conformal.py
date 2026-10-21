"""Split conformal prediction (dossier 06 C.7 l.1019-1028; I.1 l.1194).

Scores ``s_i = s(x_i, y_i)`` on n calibration points (``1 - p_hat_{y_i}(x_i)`` for classes); ``q_hat`` is the k-th
smallest score with ``k = ceil((n + 1)(1 - alpha))``; the set ``C(x) = {y : s(x, y) <= q_hat}``. Under exchangeability
``P(Y in C(X)) >= 1 - alpha`` (Angelopoulos and Bates, arXiv 2107.07511; Vovk, Gammerman and Shafer 2022,
doi:10.1007/978-3-031-06649-8). The validation test is the empirical coverage on out-of-time or shifted data against
``1 - alpha``; given a calibration of ``n_cal`` points the coverage of the set is Beta(k, n_cal + 1 - k) distributed
under exchangeability, so the covered count of ``m`` test points is Beta-Binomial(m, k, n_cal + 1 - k) and the test
reads the observed count against that law (the binomial p-value of the dossier stays beside it and never drives the
light); a coverage loss is read as an exchangeability violation, never as proof. Contract: ``engine-ml-api``
1.1.0, module ``riskvalidation/ml/conformal.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import stats

from riskvalidation.ml._checks import binary, closed_unit, decimal, vec

REL_ERR = 1.0 + 1e-7  # the tolerance of scipy.stats.binomtest on "no more likely than the observed count"


def _alpha(alpha: float) -> Fraction:
    a = decimal(alpha, "alpha")
    if not 0 < a < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return a


def _rank(n: int, a: Fraction) -> int:
    return math.ceil((n + 1) * (1 - a))


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The k-th smallest score, ``k = ceil((n + 1)(1 - alpha))``, with k computed on the exact decimal ``alpha``
    (``fractions.Fraction(str(alpha))``) so that ``101 * 0.9`` cannot round the wrong way.

    Equal to ``numpy.sort(scores)[k - 1]`` and to ``numpy.quantile(scores, k / n, method="inverted_cdf")``; the
    ``higher`` method of ``numpy.quantile`` is not the reference (it returns the (k + 1)-th order statistic). ``k > n``
    returns ``+inf`` (the set is the whole label space); the quantile is monotone in ``alpha``."""
    s = vec(scores, "scores")
    a = _alpha(alpha)
    n = s.size
    k = _rank(n, a)
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
    """The empirical coverage of ``n`` labels and the target ``1 - alpha``, the size ``n_cal`` of the calibration window
    and its rank ``k = ceil((n_cal + 1)(1 - alpha))``, the two-sided Beta-Binomial p-value (``p_value``, the one the
    light reads; NaN when ``k > n_cal``: the set is every label and the coverage is 1 by construction) and the two-sided
    binomial p-value of the dossier (``p_value_binomial``), reported beside it and never driving the light."""

    n: int
    coverage: float
    target: float
    p_value: float
    n_cal: int
    k: int
    p_value_binomial: float


def conformal_coverage(sets: np.ndarray, y: np.ndarray, alpha: float, *, n_cal: int) -> CoverageResult:
    """The empirical coverage (the share of rows whose set holds the true label), the target ``1 - alpha`` and two
    two-sided p-values of the covered count ``c`` of the ``m`` rows.

    ``n_cal`` is the number of calibration scores the quantile came from. Under exchangeability the coverage of the set
    given the calibration is Beta(k, n_cal + 1 - k) with ``k = ceil((n_cal + 1)(1 - alpha))`` (the exact decimal
    ``alpha``), so ``c`` is Beta-Binomial(m, k, n_cal + 1 - k): ``p_value`` is the sum of the probabilities of every
    count no more likely than ``c`` (the minimum-likelihood rule of ``scipy.stats.binomtest``,
    ``scipy.stats.betabinom`` for the law). ``p_value_binomial`` is the two-sided ``scipy.stats.binomtest`` of the
    dossier (06 C.7 l.1019-1028), which ignores the randomness of the calibration quantile and is oversized when
    ``m`` is not small against ``n_cal``. When ``k > n_cal`` the set is every label, the coverage is 1 by construction
    and ``p_value`` is NaN."""
    st = np.asarray(sets)
    if st.ndim != 2 or st.shape[1] != 2 or st.dtype != bool or st.shape[0] == 0:
        raise ValueError("sets must be a non-empty boolean array of shape (n, 2)")
    yy = binary(y, "y", both=False)
    if yy.shape[0] != st.shape[0]:
        raise ValueError("y must have one label per row of sets")
    if isinstance(n_cal, bool) or not isinstance(n_cal, (int, np.integer)) or n_cal < 1:
        raise ValueError(f"n_cal must be an integer of at least 1, got {n_cal!r}")
    nc = int(n_cal)
    a = _alpha(alpha)
    hit = st[np.arange(len(yy)), yy]
    c, m = int(hit.sum()), int(len(yy))
    target = float(1 - a)
    k = _rank(nc, a)
    binom = float(stats.binomtest(c, m, target, alternative="two-sided").pvalue)
    p = math.nan
    if k <= nc:
        pmf = stats.betabinom.pmf(np.arange(m + 1), m, k, nc + 1 - k)
        p = float(pmf[pmf <= pmf[c] * REL_ERR].sum())
        p = min(1.0, max(0.0, p))
    return CoverageResult(
        n=m, coverage=c / m, target=target, p_value=p, n_cal=nc, k=k, p_value_binomial=min(1.0, max(0.0, binom))
    )

__all__ = [
    "CoverageResult",
    "conformal_coverage",
    "conformal_sets",
    "split_conformal_quantile",
]
