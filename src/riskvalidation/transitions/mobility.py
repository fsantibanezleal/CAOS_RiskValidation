r"""Mobility of a migration matrix: how much of the probability mass leaves the diagonal.

Jafry and Schuermann (2004), Measurement, estimation and comparison of credit migration matrices, Journal of Banking
and Finance 28(11), 2603-2639 (closed access; read at the abstract of their Wharton working paper 03-09, "Metrics for
comparing credit migration matrices"): subtract the identity, take the singular values of the result and average
them, "a metric which has an intuitively-appealing 'size' related to the average probability of migration", chosen
to be "distribution discriminatory, i.e. sensitive to the distribution of off-diagonal probability mass",
$M_{SVD}(P) = \frac{1}{n}\sum_{i=1}^n \sigma_i(P - I)$. The difference of two matrices' $M_{SVD}$ compares their
mobility.

The trace index $(n - \operatorname{tr} P)/(n - 1)$ is the classical comparator the literature attributes to
Shorrocks (1978), Econometrica 46(5); that paper was not read here (UNVERIFIED). It sees only the diagonal: with
seven grades that each keep 0.9, a matrix moving the other 0.1 one notch and one spreading it evenly over the other
six grades have the same trace index (0.1167) and different $M_{SVD}$ (0.1206 against 0.1000,
``tests/test_transitions.py::test_mobility``). $M_{SVD}$ is not a distance in notches either: moving the 0.1 one
notch or five notches along a cycle gives the same singular values.

Both are zero on the identity. Whether the default row (absorbing, a zero row of $P - I$) belongs in the matrix is
the caller's choice; C04 reports both on the full published matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np

JAFRY_SCHUERMANN_2004 = (
    "Jafry and Schuermann (2004), Measurement, estimation and comparison of credit migration matrices, Journal of "
    "Banking and Finance 28(11), 2603-2639, doi:10.1016/j.jbankfin.2004.06.004"
)
SHORROCKS_1978 = "Shorrocks (1978), The measurement of mobility, Econometrica 46(5), 1013-1024 (not read: UNVERIFIED)"


def _square(p: Any) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError(f"a migration matrix must be square with at least two states, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("the matrix must be finite")
    return a


def svd_mobility(p: Any) -> float:
    """Jafry and Schuermann's M_SVD: the mean singular value of P - I."""
    a = _square(p)
    return float(np.linalg.svd(a - np.eye(a.shape[0]), compute_uv=False).mean())


def trace_mobility(p: Any) -> float:
    """The trace index (n - tr P)/(n - 1)."""
    a = _square(p)
    n = a.shape[0]
    return float((n - np.trace(a)) / (n - 1))


__all__ = ["svd_mobility", "trace_mobility"]
