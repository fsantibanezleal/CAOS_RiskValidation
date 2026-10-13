"""RV-611: downturn LGD (dossier 04 F.4): the period from a supplied macro path, the cohort add-on and the coverage
shortfall (contract `engine-lgd-ead-api`, module downturn). The macro path is constructed, not BLS data."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from riskvalidation.lgd_ead import downturn as d


def a(*v):
    return np.array(v, dtype=float)


def test_identify_downturn():
    years = np.arange(2006, 2012)
    ind = a(4.6, 4.6, 5.8, 9.3, 9.6, 8.9)
    got = d.identify_downturn(years, ind, 7.0)
    assert got.tolist() == [False, False, False, True, True, True] and got.dtype == bool
    # strict: a tie is not a downturn year
    assert d.identify_downturn(years, a(7.0, 7.0, 7.1, 6.9, 7.0, 8.0), 7.0).tolist() == [False, False, True, False,
                                                                                         False, True]
    # lower is worse (for example a house price index)
    low = d.identify_downturn(a(2006, 2007, 2008), a(100.0, 90.0, 80.0), 90.0, higher_is_worse=False)
    assert low.tolist() == [False, False, True]
    with pytest.raises(ValueError, match="NaN"):
        d.identify_downturn(years, a(4.6, np.nan, 5.8, 9.3, 9.6, 8.9), 7.0)
    with pytest.raises(ValueError, match="increasing"):
        d.identify_downturn(np.array([2006, 2006, 2008]), a(1.0, 2.0, 3.0), 1.5)
    with pytest.raises(ValueError, match="same length"):
        d.identify_downturn(years, a(1.0, 2.0), 1.5)
    with pytest.raises(ValueError):
        d.identify_downturn(years, ind, float("nan"))


def test_downturn_add_on():
    lgd = a(0.30, 0.60, 0.40)
    w = a(100, 50, 50)
    mask = np.array([False, True, False])
    r = d.downturn_add_on(lgd, w, mask)
    assert isinstance(r, d.DownturnResult) and dataclasses.is_dataclass(r)
    assert r.long_run == pytest.approx(0.40, abs=1e-12)
    assert r.downturn == pytest.approx(0.60, abs=1e-12)
    assert r.add_on == pytest.approx(0.20, abs=1e-12)
    assert r.add_on == r.downturn - r.long_run  # exactly
    # independent of the order of the cohorts
    for perm in itertools.permutations(range(3)):
        p = list(perm)
        q = d.downturn_add_on(lgd[p], w[p], mask[p])
        assert (q.long_run, q.downturn) == pytest.approx((0.40, 0.60), abs=1e-12)
    # the add-on may be negative and is reported as is
    neg = d.downturn_add_on(a(0.5, 0.3), a(10, 10), np.array([False, True]))
    assert neg.add_on == pytest.approx(-0.1, abs=1e-12)
    with pytest.raises(ValueError, match="mask"):
        d.downturn_add_on(lgd, w, np.array([False, False, False]))
    with pytest.raises(ValueError, match="cohort_weights"):
        d.downturn_add_on(lgd, a(100, 0, 50), mask)
    with pytest.raises(ValueError, match="same length"):
        d.downturn_add_on(lgd, a(1, 1), mask)
    with pytest.raises(ValueError):
        d.downturn_add_on(a(0.3, np.nan, 0.4), w, mask)


def test_coverage_shortfall():
    covered, short = d.coverage_shortfall(0.55, 0.60)
    assert (covered, short) == (False, pytest.approx(0.05, abs=1e-12))
    assert d.coverage_shortfall(0.70, 0.68) == (True, 0.0)
    assert d.coverage_shortfall(0.60, 0.60) == (True, 0.0)
    assert d.coverage_shortfall(0.60 - 1e-13, 0.60)[0] is True  # the 1e-12 tolerance
    assert d.coverage_shortfall(0.60 - 1e-9, 0.60)[0] is False
    with pytest.raises(ValueError):
        d.coverage_shortfall(float("nan"), 0.5)
