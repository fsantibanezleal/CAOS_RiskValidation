"""RV-1414 to RV-1417: partial dependence, ICE, accumulated local effects, the overfit gap and the weak spots
(`riskvalidation.ml.effects` and `riskvalidation.ml.generalisation`, contract engine-ml-api 1.0.0)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.ml.effects import accumulated_local_effects, ice_curves, partial_dependence
from riskvalidation.ml.generalisation import OverfitResult, WeakSpot, overfit_gap, weak_spots


def _additive(x):
    return 0.3 * x[:, 0] + np.sin(x[:, 1]) + x[:, 2] ** 2


def test_partial_dependence_of_an_additive_model_is_the_component_plus_a_constant():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 3))
    keep = x.copy()
    grid = np.linspace(-2.0, 2.0, 9)
    pd1 = partial_dependence(_additive, x, 1, grid)
    constant = float(np.mean(0.3 * x[:, 0]) + np.mean(x[:, 2] ** 2))
    np.testing.assert_allclose(pd1, np.sin(grid) + constant, atol=1e-12)
    assert pd1.shape == (9,)
    np.testing.assert_array_equal(x, keep)  # the other columns keep their observed values; the input is not mutated
    assert np.all(pd1 >= np.sin(grid).min() + constant - 1e-12) and np.all(pd1 <= np.sin(grid).max() + constant + 1e-12)


def test_ice_curves_mean_is_the_partial_dependence_exactly():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((120, 3))
    grid = np.linspace(-1.0, 1.0, 5)
    ice = ice_curves(_additive, x, 0, grid)
    assert ice.shape == (120, 5)
    np.testing.assert_array_equal(ice.mean(axis=0), partial_dependence(_additive, x, 0, grid))
    # one curve per row: row i varies only the feature, the other columns are that row's own
    np.testing.assert_allclose(ice[7], 0.3 * grid + np.sin(x[7, 1]) + x[7, 2] ** 2, atol=1e-12)


def test_effects_refuse_invalid_feature_and_grid():
    x = np.zeros((10, 2))
    with pytest.raises(ValueError, match="feature"):
        partial_dependence(lambda a: a[:, 0], x, 2, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="grid"):
        ice_curves(lambda a: a[:, 0], x, 0, np.array([]))
    with pytest.raises(ValueError, match="predict"):
        partial_dependence(lambda a: a[:5, 0], x, 0, np.array([0.0, 1.0]))


def test_ale_of_an_additive_model_is_the_centred_component():
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.standard_normal((n, 3))
    x[:, 2] = 0.7 * x[:, 1] + 0.3 * x[:, 2]  # correlated features: ALE is still the component
    edges, eff = accumulated_local_effects(_additive, x, 1, 10)
    assert edges.shape == (11,) and eff.shape == (10,)
    counts = np.bincount(np.searchsorted(edges[1:-1], x[:, 1], side="left"), minlength=10)
    assert counts.min() >= 1 and counts.sum() == n  # quantile bins hold at least one row
    h = np.sin(edges[1:])  # the component at the upper edge of each bin
    ref = h - float(np.sum(counts * h) / n)  # the component centred with the same count weights
    # the model also has terms in the other columns, which a column-1-only change does not move: exact identity
    np.testing.assert_allclose(eff, ref, atol=1e-12)
    assert abs(float(np.sum(counts * eff) / n)) < 1e-12  # centred to mean zero (count-weighted)


def test_ale_is_centred_and_bins_are_quantiles_for_a_non_additive_model():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1500, 3))

    def f(a):
        return a[:, 0] * a[:, 1] + a[:, 2]

    edges, eff = accumulated_local_effects(f, x, 0, 7)
    counts = np.bincount(np.searchsorted(edges[1:-1], x[:, 0], side="left"), minlength=7)
    assert abs(float(np.sum(counts * eff) / len(x))) < 1e-12
    assert np.all(np.diff(edges) > 0) and edges[0] == x[:, 0].min() and edges[-1] == x[:, 0].max()
    assert np.all(np.abs(counts - len(x) / 7) <= 0.1 * len(x) / 7 + 2)  # near-equal quantile bins


def test_ale_refuses_ties_that_empty_a_bin_and_bad_input():
    x = np.zeros((50, 2))
    x[:, 1] = np.random.default_rng(5).standard_normal(50)
    with pytest.raises(ValueError, match="n_bins"):
        accumulated_local_effects(lambda a: a[:, 1], x, 0, 5)  # constant feature: duplicate quantile edges
    with pytest.raises(ValueError, match="n_bins"):
        accumulated_local_effects(lambda a: a[:, 1], x, 1, 0)
    with pytest.raises(ValueError, match="feature"):
        accumulated_local_effects(lambda a: a[:, 1], x, 5, 4)


def test_overfit_gap_directions_and_boundary():
    r = overfit_gap(0.90, 0.80, 0.05)
    assert isinstance(r, OverfitResult) and r.gap == pytest.approx(0.10) and r.tolerance == 0.05 and r.flag is True
    assert overfit_gap(0.85, 0.80, 0.05).flag is False  # gap 0.05 (0.85 - 0.80 = 0.04999999999999993 in floats)
    assert overfit_gap(0.8, 0.5, 0.3).flag is (0.8 - 0.5 > 0.3)  # flag is exactly (gap > tolerance)
    assert overfit_gap(0.25, 0.20, 0.05, higher_is_better=False).gap == pytest.approx(-0.05)  # Brier: test - train
    b = overfit_gap(0.10, 0.18, 0.05, higher_is_better=False)
    assert b.gap == pytest.approx(0.08) and b.flag is True and b.tolerance == 0.05
    with pytest.raises(ValueError, match="tolerance"):
        overfit_gap(0.9, 0.8, -0.1)
    with pytest.raises(ValueError, match="metric_test"):
        overfit_gap(0.9, float("nan"), 0.1)
    with pytest.raises(ValueError, match="flag"):
        OverfitResult(gap=0.1, tolerance=0.05, flag=False)


def test_weak_spots_planted_slice_three_times_the_rest():
    err = np.ones(500)
    bins = np.repeat(np.arange(5), 100)
    err[bins == 3] = 3.0
    overall = float(err.mean())  # 1.4
    hit = weak_spots(err, bins, 2.0)
    assert [(w.label, w.n) for w in hit] == [(3, 100)]
    assert hit[0].mean_error == 3.0 and hit[0].ratio == pytest.approx(3.0 / overall, abs=1e-15) and hit[0].ratio > 2.0
    assert weak_spots(err, bins, 4.0) == []  # absent for k = 4


def test_weak_spots_min_n_sorting_and_refusals():
    err = np.ones(3160)
    bins = np.r_[np.repeat(0, 3000), np.repeat(1, 100), np.repeat(2, 50), np.repeat(3, 10)]
    err[bins == 1] = 4.0
    err[bins == 2] = 6.0
    err[bins == 3] = 100.0  # a huge error on a slice below min_n: never reported
    spots = weak_spots(err, bins, 1.5)
    assert [w.label for w in spots] == [2, 1]  # sorted by decreasing ratio
    assert all(isinstance(w, WeakSpot) and w.n >= 30 for w in spots)
    assert [w.label for w in weak_spots(err, bins, 1.5, min_n=5)] == [3, 2, 1]
    assert weak_spots(np.zeros(60), np.repeat([0, 1], 30), 2.0) == []  # no error anywhere: nothing exceeds
    with pytest.raises(ValueError, match="bins"):
        weak_spots(err, bins[:-1], 2.0)
    with pytest.raises(ValueError, match="k"):
        weak_spots(err, bins, 0.0)
    with pytest.raises(ValueError, match="min_n"):
        weak_spots(err, bins, 2.0, min_n=0)
