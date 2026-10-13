"""RV-601 to RV-604: the workout LGD of dossier 04 F.1, its range flags, the incomplete-workout treatment, the training
cut-off and the LGD distribution (contract `engine-lgd-ead-api`, module workout)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from riskvalidation.lgd_ead import workout as w


def a(*v, dtype=float):
    return np.array(v, dtype=dtype)


# ---------------------------------------------------------------- RV-601


def test_eba_discount_rate_golden():
    assert w.eba_discount_rate(0.0325) == pytest.approx(0.0825, abs=1e-15)
    assert w.eba_discount_rate(0.02, add_on=0.03) == pytest.approx(0.05, abs=1e-15)
    assert w.eba_discount_rate(-0.005) == pytest.approx(0.045, abs=1e-15)  # a negative interbank rate is legal
    with pytest.raises(ValueError, match="add_on"):
        w.eba_discount_rate(0.03, add_on=-0.01)
    with pytest.raises(ValueError):
        w.eba_discount_rate(float("nan"))


def _one(recov, costs, draws=(0.0, 0.0), rate=0.10, ead=100.0, times=(1.0, 2.0)):
    return w.workout_lgd(a(ead), np.array([times]), np.array([recov]), np.array([costs]), np.array([draws]), rate)


def test_workout_lgd_goldens_and_invariants():
    # the golden of the contract: PV = 30 / 1.1 + 30 / 1.21 = 52.066116
    got = _one((40.0, 30.0), (10.0, 0.0))
    assert got.shape == (1,)
    assert got[0] == pytest.approx(0.479339, abs=5e-7)
    assert got[0] == pytest.approx(1.0 - (30 / 1.1 + 30 / 1.21) / 100.0, abs=1e-15)
    # outside [0, 1]: returned as is, never clipped
    assert _one((5.0, 0.0), (20.0, 0.0))[0] == pytest.approx(1.136364, abs=5e-7)
    assert _one((120.0, 0.0), (0.0, 0.0))[0] == pytest.approx(-0.090909, abs=5e-7)
    # zero rate: 1 - sum(R - C - Add) / EAD
    assert _one((40.0, 30.0), (10.0, 0.0), (5.0, 0.0), rate=0.0)[0] == pytest.approx(1 - 55.0 / 100.0, abs=1e-15)
    # monotone in recoveries (down), costs, drawings (up) and, with positive net flows, in the rate (up)
    base = _one((40.0, 30.0), (10.0, 0.0))[0]
    assert _one((50.0, 30.0), (10.0, 0.0))[0] < base
    assert _one((40.0, 30.0), (12.0, 0.0))[0] > base
    assert _one((40.0, 30.0), (10.0, 0.0), (3.0, 0.0))[0] > base
    assert _one((40.0, 30.0), (10.0, 0.0), rate=0.12)[0] > base
    # NOT nondecreasing in the rate in general: a net cash outflow (costs above recoveries) is discounted too
    assert _one((0.0, 0.0), (30.0, 30.0), rate=0.12)[0] < _one((0.0, 0.0), (30.0, 30.0), rate=0.10)[0]
    # the contract golden for it: recoveries 5 and costs 20 give 1.15 at a zero rate and 1.136364 at 10%
    assert _one((5.0, 0.0), (20.0, 0.0), rate=0.0)[0] == pytest.approx(1.15, abs=1e-15)
    assert _one((5.0, 0.0), (20.0, 0.0), rate=0.10)[0] < _one((5.0, 0.0), (20.0, 0.0), rate=0.0)[0]
    # one rate per exposure, zero-padded flows, several exposures
    ead = a(100.0, 200.0)
    t = np.array([[1.0, 2.0], [1.0, 1.5]])
    rec = np.array([[40.0, 30.0], [100.0, 0.0]])
    zero = np.zeros((2, 2))
    out = w.workout_lgd(ead, t, rec, zero, zero, a(0.10, 0.0))
    assert out[0] == pytest.approx(1 - (40 / 1.1 + 30 / 1.21) / 100, abs=1e-15)
    assert out[1] == pytest.approx(0.5, abs=1e-15)


def test_workout_lgd_refusals():
    t = np.array([[1.0, 2.0]])
    z = np.zeros((1, 2))
    r = np.array([[10.0, 10.0]])
    with pytest.raises(ValueError, match="ead"):
        w.workout_lgd(a(0.0), t, r, z, z, 0.1)
    with pytest.raises(ValueError, match="times"):
        w.workout_lgd(a(100.0), np.array([[0.0, 2.0]]), r, z, z, 0.1)
    with pytest.raises(ValueError, match="recoveries"):
        w.workout_lgd(a(100.0), t, np.array([[-1.0, 2.0]]), z, z, 0.1)
    with pytest.raises(ValueError, match="costs"):
        w.workout_lgd(a(100.0), t, r, np.array([[-1.0, 0.0]]), z, 0.1)
    with pytest.raises(ValueError, match="drawings"):
        w.workout_lgd(a(100.0), t, r, z, np.array([[0.0, -1.0]]), 0.1)
    with pytest.raises(ValueError, match="same shape"):
        w.workout_lgd(a(100.0), t, np.array([[1.0]]), z, z, 0.1)
    with pytest.raises(ValueError, match="rate"):
        w.workout_lgd(a(100.0), t, r, z, z, -1.0)
    with pytest.raises(ValueError, match="rate"):
        w.workout_lgd(a(100.0), t, r, z, z, a(0.1, 0.1))
    with pytest.raises(ValueError, match="non-finite"):
        w.workout_lgd(a(100.0), t, np.array([[np.nan, 1.0]]), z, z, 0.1)
    with pytest.raises(ValueError, match="non-finite"):
        w.workout_lgd(a(np.inf), t, r, z, z, 0.1)


# ---------------------------------------------------------------- RV-602


def test_range_flag_golden_and_nan():
    got = w.range_flag(a(-0.090909, 0.0, 0.5, 1.0, 1.136364))
    assert list(got) == ["below-zero", "in-range", "in-range", "in-range", "above-one"]
    assert got.dtype.kind == "U"
    with pytest.raises(ValueError, match="NaN"):
        w.range_flag(a(0.1, np.nan))
    assert list(w.range_flag(a(np.inf, -np.inf))) == ["above-one", "below-zero"]
    assert w.range_flag(np.array([])).shape == (0,)


# ---------------------------------------------------------------- RV-603


def test_resolve_incomplete():
    lgd = a(0.2, 0.6, 0.9)
    complete = np.array([True, True, False])
    ex = w.resolve_incomplete(lgd, complete, "excluded")
    assert isinstance(ex, w.ResolvedWorkouts) and dataclasses.is_dataclass(ex)
    np.testing.assert_array_equal(ex.lgd, [0.2, 0.6])
    assert (ex.n_excluded, ex.n_projected, ex.treatment) == (1, 0, "excluded")
    assert ex.lgd.mean() == pytest.approx(0.4)
    pr = w.resolve_incomplete(lgd, complete, "projected", a(np.nan, np.nan, 0.7))
    np.testing.assert_array_equal(pr.lgd, [0.2, 0.6, 0.7])
    assert (pr.n_excluded, pr.n_projected, pr.treatment) == (0, 1, "projected")
    assert pr.lgd.mean() == pytest.approx(0.5)
    # the values of an incomplete workout are ignored, so a NaN there is allowed
    assert w.resolve_incomplete(a(0.2, np.nan), np.array([True, False]), "excluded").lgd.tolist() == [0.2]
    # counts add up
    c = np.array([False, True, False, False])
    out = w.resolve_incomplete(a(0.1, 0.2, 0.3, 0.4), c, "excluded")
    assert out.n_excluded + out.n_projected == int((~c).sum()) and out.lgd.size == 1
    with pytest.raises(ValueError, match="treatment"):
        w.resolve_incomplete(lgd, complete, "keep")
    with pytest.raises(ValueError, match="projected"):
        w.resolve_incomplete(lgd, complete, "projected")
    with pytest.raises(ValueError, match="projected"):
        w.resolve_incomplete(lgd, complete, "projected", a(np.nan, np.nan, np.nan))
    with pytest.raises(ValueError, match="same length"):
        w.resolve_incomplete(lgd, np.array([True, False]), "excluded")
    # all complete: nothing excluded
    full = w.resolve_incomplete(lgd, np.array([True, True, True]), "excluded")
    assert full.n_excluded == 0 and full.lgd.size == 3
    # projection also accepted when every workout is complete, even without it for "excluded"
    assert w.resolve_incomplete(lgd, np.array([True] * 3), "projected").n_projected == 0
    # everything incomplete and excluded: an empty result, with the count
    none = w.resolve_incomplete(lgd, np.array([False] * 3), "excluded")
    assert none.lgd.size == 0 and none.n_excluded == 3


def test_closed_before():
    d = np.array(["2008-06-30", "2008-12-31", "2009-03-31", "NaT"], dtype="datetime64[D]")
    got = w.closed_before(d, "2008-12-31")
    assert got.tolist() == [True, False, False, False]
    assert got.dtype == bool
    assert w.closed_before(d, "2009-01-01").tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="cutoff"):
        w.closed_before(d, "not a date")
    with pytest.raises(ValueError, match="datetime64"):
        w.closed_before(a(1.0, 2.0), "2008-12-31")


# ---------------------------------------------------------------- RV-604


def test_lgd_distribution():
    d = w.lgd_distribution(a(-0.05, 0, 0, 0.25, 0.5, 1, 1, 1.2), a(0, 0.5, 1.0))
    assert isinstance(d, w.LgdDistribution)
    assert d.counts.tolist() == [3, 3] and d.counts.dtype == np.int64
    assert (d.n, d.n_below_zero, d.n_above_one, d.mass_zero, d.mass_one) == (8, 1, 1, 2, 2)
    assert int(d.counts.sum()) + d.n_below_zero + d.n_above_one == d.n
    assert d.mass_zero <= d.counts[0] and d.mass_one <= d.counts[-1]
    # bin i is [e_i, e_i+1), the last bin is closed
    four = w.lgd_distribution(a(0.0, 0.25, 0.5, 0.75, 1.0), a(0, 0.25, 0.5, 0.75, 1.0))
    assert four.counts.tolist() == [1, 1, 1, 2]
    with pytest.raises(ValueError, match="NaN"):
        w.lgd_distribution(a(0.1, np.nan), a(0, 1.0))
    for bad in (a(0.0, 0.5), a(0.1, 0.5, 1.0), a(0.0, 0.5, 0.5, 1.0), a(0.0, 1.0, 0.5), a(1.0)):
        with pytest.raises(ValueError, match="edges"):
            w.lgd_distribution(a(0.1), bad)
    with pytest.raises(ValueError):
        w.lgd_distribution(a(), a(0, 1.0))
