"""RV-1457 and RV-1458: the pooled z-test and the Fisher exact test against scipy and statsmodels."""

from __future__ import annotations

import dataclasses
import math

import pytest
from scipy import stats

from riskvalidation.fairness.inference import FisherResult, ZResult, fisher_exact_test, two_proportion_z_test

GRID = [(30, 100, 45, 120), (5, 50, 15, 60), (0, 40, 7, 40), (60, 100, 75, 100), (1, 3, 2, 3), (500, 1000, 450, 1000),
        (12, 12, 3, 20), (2, 2000, 10, 1900)]


def test_two_proportion_z_reference_scipy_and_statsmodels():
    for x1, n1, x2, n2 in GRID:
        r = two_proportion_z_test(x1, n1, x2, n2)
        pooled = (x1 + x2) / (n1 + n2)
        if pooled in (0.0, 1.0):
            continue
        z = (x1 / n1 - x2 / n2) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        assert r.statistic == pytest.approx(z, abs=1e-12)
        assert r.p_value == pytest.approx(2.0 * stats.norm.sf(abs(z)), abs=1e-12)
        assert (r.n1, r.n2) == (n1, n2)
        assert r.note is None
        assert isinstance(r, ZResult)
    sm = pytest.importorskip("statsmodels.stats.proportion", reason="statsmodels is the optional reference")
    for x1, n1, x2, n2 in GRID:
        if (x1 + x2) in (0, n1 + n2):
            continue
        z_sm, p_sm = sm.proportions_ztest([x1, x2], [n1, n2])
        r = two_proportion_z_test(x1, n1, x2, n2)
        assert r.statistic == pytest.approx(z_sm, abs=1e-10)
        assert r.p_value == pytest.approx(p_sm, abs=1e-10)


def test_two_proportion_z_degenerate_pooled_rate():
    for args in ((0, 10, 0, 20), (10, 10, 20, 20)):
        r = two_proportion_z_test(*args)
        assert r.statistic == 0.0 and r.p_value == 1.0
        assert r.note is not None and "pooled" in r.note
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.statistic = 1.0  # type: ignore[misc]
    # a symmetric swap flips the sign of the statistic and keeps the p-value
    a, b = two_proportion_z_test(30, 100, 45, 120), two_proportion_z_test(45, 120, 30, 100)
    assert a.statistic == -b.statistic and a.p_value == pytest.approx(b.p_value, abs=1e-15)


def test_two_proportion_z_refuses_invalid_counts():
    for bad, field in (((-1, 10, 3, 10), "x1"), ((11, 10, 3, 10), "x1"), ((3, 10, 11, 10), "x2"),
                       ((3, 10, -2, 10), "x2"), ((1.5, 10, 3, 10), "x1"), ((3, 10.5, 3, 10), "n1"),
                       ((3, 10, 3, 2.5), "n2")):
        with pytest.raises(ValueError, match=field):
            two_proportion_z_test(*bad)
        with pytest.raises(ValueError, match=field):
            fisher_exact_test(*bad)
    for bad, field in (((0, 0, 3, 10), "n1"), ((3, 10, 0, 0), "n2")):
        with pytest.raises(ValueError, match=field):
            two_proportion_z_test(*bad)
        with pytest.raises(ValueError, match=field):
            fisher_exact_test(*bad)


def test_fisher_exact_equals_scipy_on_a_grid():
    for x1, n1, x2, n2 in GRID:
        r = fisher_exact_test(x1, n1, x2, n2)
        odds, p = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]], alternative="two-sided")
        assert r.p_value == pytest.approx(p, abs=1e-12)
        if math.isfinite(odds):
            assert r.odds_ratio == pytest.approx(odds, abs=1e-12)
        assert 0.0 <= r.p_value <= 1.0
        assert isinstance(r, FisherResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.p_value = 0.5  # type: ignore[misc]


def test_fisher_odds_ratio_undefined_is_none():
    # a zero cell in the denominator (n1 - x1 = 0 or x2 = 0) makes the sample odds ratio infinite or 0/0: None
    assert fisher_exact_test(12, 12, 3, 20).odds_ratio is None
    assert fisher_exact_test(5, 10, 0, 10).odds_ratio is None
    assert fisher_exact_test(0, 10, 0, 10).odds_ratio is None
    assert fisher_exact_test(0, 10, 5, 10).odds_ratio == 0.0
    r = fisher_exact_test(30, 100, 45, 120)
    assert r.odds_ratio == pytest.approx((30 * 75) / (70 * 45), abs=1e-12)
