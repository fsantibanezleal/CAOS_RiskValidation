"""RV-110, RV-111: ECB rating-system statistics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.validation import rating_hhi, rating_migration_ztests, rating_mwb
from riskvalidation.validation.rating_system import concentration, matrix_weighted_bandwidth


def test_mwb_and_migration_ztests():
    assert matrix_weighted_bandwidth(np.diag([50, 80, 40])) == (0.0, 0.0)
    extreme = np.array([[0, 0, 30], [0, 10, 0], [25, 0, 0]])
    assert matrix_weighted_bandwidth(extreme) == pytest.approx((1.0, 1.0))
    # hand computation on a 3x3 matrix (grades 1 best, 3 worst)
    m = np.array([[80, 15, 5], [10, 70, 20], [2, 18, 80]], dtype=float)
    n = m.sum(axis=1)
    p = m / n[:, None]
    num_u = 1 * n[0] * p[0, 1] + 2 * n[0] * p[0, 2] + 1 * n[1] * p[1, 2]
    den_u = 2 * n[0] * (p[0, 1] + p[0, 2]) + 1 * n[1] * p[1, 2]
    num_l = 1 * n[1] * p[1, 0] + 2 * n[2] * p[2, 0] + 1 * n[2] * p[2, 1]
    den_l = 1 * n[1] * p[1, 0] + 2 * n[2] * (p[2, 0] + p[2, 1])
    up, lo = matrix_weighted_bandwidth(m)
    assert up == pytest.approx(num_u / den_u) and lo == pytest.approx(num_l / den_l)
    assert rating_mwb(m).metric == pytest.approx(max(up, lo))

    # z-test for cell (1 -> 3) against its neighbour (1 -> 2), as in ECB section 2.5.5.2
    r = rating_migration_ztests(m)
    cell = next(c for c in r.extras["cells"] if (c["from"], c["to"]) == (1, 3))
    var = (p[0, 2] * (1 - p[0, 2]) + p[0, 1] * (1 - p[0, 1]) + 2 * p[0, 2] * p[0, 1]) / n[0]
    z = (p[0, 1] - p[0, 2]) / math.sqrt(var)
    assert cell["z"] == pytest.approx(z) and cell["p"] == pytest.approx(stats.norm.cdf(z))
    # a matrix where a far cell is more populated than its neighbour is flagged
    bad = np.array([[60, 5, 35], [10, 80, 10], [5, 15, 80]], dtype=float)
    rb = rating_migration_ztests(bad)
    assert rb.p_value < 0.01
    cells = rb.extras["cells"]
    raw = np.array([c["p"] for c in cells])
    assert min(c["p_holm"] for c in cells) == pytest.approx(min(1.0, raw.size * raw.min()))


def test_hhi_cv_ecb():
    cv, hi = concentration([25, 25, 25, 25])
    assert cv == pytest.approx(0.0) and hi == pytest.approx(0.0)
    cv1, hi1 = concentration([0, 100, 0, 0])
    assert hi1 == pytest.approx(1.0) and cv1 == pytest.approx(math.sqrt(3))
    init = [20, 30, 30, 20]
    cur = [5, 15, 60, 20]
    r = rating_hhi(cur, init)
    k = 4
    cv_i, _ = concentration(init)
    cv_c, _ = concentration(cur)
    s = math.sqrt(k - 1) * (cv_c - cv_i) / math.sqrt(cv_c**2 * (0.5 + cv_c**2))
    assert r.statistic == pytest.approx(s)
    assert r.p_value == pytest.approx(1 - stats.norm.cdf(s))
    assert r.extras["hi_current"] > r.extras["hi_initial"]
