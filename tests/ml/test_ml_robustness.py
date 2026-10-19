"""RV-1401 to RV-1406: perturbation robustness, the shift score, the worst subsample and out-of-time windows
(`riskvalidation.ml.robustness`, contract engine-ml-api 1.0.0)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ml_testkit import logistic_xy

from riskvalidation.ml.robustness import (
    RobustnessCurve,
    SubsampleResult,
    oot_window_performance,
    perturb_features,
    robustness_curve,
    robustness_finding,
    shift_score_rank,
    worst_subsample_performance,
)
from riskvalidation.validation.discrimination import auc_with_variance


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_perturb_features_identity_determinism_and_purity():
    x = np.random.default_rng(1).standard_normal((50, 3))
    keep = x.copy()
    same = perturb_features(x, 0.0, seed=3)
    np.testing.assert_array_equal(same, x)
    a, b = perturb_features(x, 0.5, seed=3), perturb_features(x, 0.5, seed=3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, perturb_features(x, 0.5, seed=4))
    assert a.shape == x.shape
    np.testing.assert_array_equal(x, keep)  # the input is never mutated
    assert same is not x  # a new array even for eps = 0


def test_perturb_features_noise_has_sd_eps_times_sd():
    rng = np.random.default_rng(2)
    n = 100_000
    x = rng.standard_normal((n, 3)) * np.array([1.0, 3.0, 0.5])
    eps = 0.4
    for sd in (None, np.array([2.0, 0.1, 5.0])):
        noise = perturb_features(x, eps, seed=11, sd=sd) - x
        target = eps * (x.std(axis=0, ddof=1) if sd is None else sd)
        measured = noise.std(axis=0, ddof=1)
        se = target / math.sqrt(2.0 * (n - 1))  # standard error of a sample standard deviation of normal data
        assert np.all(np.abs(measured - target) <= 4.0 * se), (measured, target)
        assert np.all(np.abs(noise.mean(axis=0)) <= 4.0 * target / math.sqrt(n))


@pytest.mark.parametrize("kwargs, field", [
    ({"X": np.zeros(5), "eps": 0.1}, "X"),
    ({"X": np.zeros((5, 2)), "eps": -0.1}, "eps"),
    ({"X": np.zeros((5, 2)), "eps": float("nan")}, "eps"),
    ({"X": np.array([[1.0, np.nan]]), "eps": 0.1}, "X"),
    ({"X": np.zeros((5, 2)), "eps": 0.1, "sd": np.ones(3)}, "sd"),
    ({"X": np.zeros((5, 2)), "eps": 0.1, "sd": np.array([1.0, -1.0])}, "sd"),
])
def test_perturb_features_refuses_invalid_input_naming_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        perturb_features(seed=0, **kwargs)


def test_robustness_curve_constant_predictor_and_predictor_ignoring_the_input():
    x, y = logistic_xy(800, 3, 4)
    const = robustness_curve(lambda a: np.full(len(a), 0.3), x, y, [0.0, 0.5, 1.0], seed=0)
    assert const.auc == (0.5, 0.5, 0.5) and const.auc_delta == (0.0, 0.0, 0.0)
    fixed = _expit(x[:, 0] - 0.5)
    ignoring = robustness_curve(lambda a: fixed, x, y, [0.0, 0.5, 2.0], seed=0)
    assert ignoring.auc_delta == (0.0, 0.0, 0.0)
    assert ignoring.brier[0] == ignoring.brier[2]


def test_robustness_curve_degrades_and_is_deterministic():
    x, y = logistic_xy(3000, 3, 5, signal=1.5)

    def predict(a):
        return _expit(a[:, 0] - 0.5)

    c1 = robustness_curve(predict, x, y, [0.0, 0.5, 2.0], seed=7)
    c2 = robustness_curve(predict, x, y, [0.0, 0.5, 2.0], seed=7)
    assert c1 == c2
    assert c1.grid == (0.0, 0.5, 2.0)
    auc0 = auc_with_variance(predict(x), y)[0]
    assert c1.auc[0] == auc0
    assert c1.auc_delta[0] == 0.0
    assert c1.auc_delta[1] == c1.auc[1] - c1.auc[0]
    assert c1.auc_delta[2] < c1.auc_delta[1] < 0.0  # more noise, more loss
    brier0 = float(np.mean((predict(x) - y) ** 2))
    assert c1.brier[0] == pytest.approx(brier0, abs=1e-15)
    assert c1.brier[2] > c1.brier[0]


def test_robustness_curve_starts_at_scale_zero_and_refuses_bad_input():
    x, y = logistic_xy(400, 2, 6)

    def predict(a):
        return _expit(a[:, 0])

    curve = robustness_curve(predict, x, y, [0.25, 0.5], seed=1)
    assert curve.grid == (0.0, 0.25, 0.5)  # scale 0 is always the first point
    assert len(curve.grid) == len(curve.auc) == len(curve.brier) == len(curve.auc_delta)
    assert robustness_curve(predict, x, y, [0.0, 0.0, 0.5], seed=1).grid == (0.0, 0.5)
    with pytest.raises(ValueError, match="grid"):
        robustness_curve(predict, x, y, [-0.1], seed=1)
    with pytest.raises(ValueError, match="y"):
        robustness_curve(predict, x, np.zeros(len(y), dtype=int), [0.5], seed=1)
    with pytest.raises(ValueError, match="predict"):
        robustness_curve(lambda a: np.full(len(a) + 1, 0.5), x, y, [0.5], seed=1)
    with pytest.raises(ValueError, match="predict"):
        robustness_curve(lambda a: np.full(len(a), 1.5), x, y, [0.5], seed=1)


def test_robustness_curve_dataclass_invariants():
    ok = RobustnessCurve(grid=(0.0, 1.0), auc=(0.8, 0.7), brier=(0.1, 0.2), auc_delta=(0.0, -0.1))
    with pytest.raises(AttributeError):
        ok.grid = (0.0,)  # frozen
    with pytest.raises(ValueError, match="equal length"):
        RobustnessCurve(grid=(0.0, 1.0), auc=(0.8,), brier=(0.1, 0.2), auc_delta=(0.0, -0.1))
    with pytest.raises(ValueError, match="grid"):
        RobustnessCurve(grid=(0.5, 1.0), auc=(0.8, 0.7), brier=(0.1, 0.2), auc_delta=(0.0, -0.1))


def test_robustness_finding_boundary_and_grid():
    curve = RobustnessCurve(grid=(0.0, 0.5, 1.0), auc=(0.8, 0.78, 0.7), brier=(0.1, 0.1, 0.1),
                            auc_delta=(0.0, -0.02, -0.1))
    assert robustness_finding(curve, 0.02, 0.5) is False  # a delta equal to the tolerance is not a finding
    assert robustness_finding(curve, 0.0199, 0.5) is True
    assert robustness_finding(curve, 0.02, 1.0) is True
    assert robustness_finding(curve, 0.5, 1.0) is False
    with pytest.raises(ValueError, match="preset_eps"):
        robustness_finding(curve, 0.02, 0.75)
    with pytest.raises(ValueError, match="tolerance"):
        robustness_finding(curve, -0.01, 0.5)


def test_shift_score_rank_identical_samples_score_near_one_half():
    x = np.random.default_rng(8).standard_normal((1500, 4))
    for other in (x.copy(), np.random.default_rng(9).standard_normal((1500, 4))):
        s = shift_score_rank(x, other, seed=2)
        assert s.shape == (1500,) and np.all((s >= 0.0) & (s <= 1.0))
        assert abs(float(s.mean()) - 0.5) <= 0.05


def test_shift_score_rank_ranks_planted_shifted_rows_above_the_rest():
    rng = np.random.default_rng(10)
    dev = rng.standard_normal((1500, 3))
    plain = rng.standard_normal((750, 3))
    shifted = rng.standard_normal((750, 3)) + np.array([1.5, 0.0, 0.0])
    ev = np.vstack([plain, shifted])
    s = shift_score_rank(dev, ev, seed=3)
    flag = np.r_[np.zeros(750, dtype=int), np.ones(750, dtype=int)]
    assert float(s[750:].mean()) > float(s[:750].mean()) + 0.2
    assert auc_with_variance(s, flag)[0] > 0.8  # power: the shifted rows rank above the rest


def test_shift_score_rank_is_deterministic_and_pure():
    rng = np.random.default_rng(11)
    dev, ev = rng.standard_normal((400, 3)), rng.standard_normal((300, 3)) + 0.5
    d0, e0 = dev.copy(), ev.copy()
    a, b = shift_score_rank(dev, ev, seed=5), shift_score_rank(dev, ev, seed=5)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(dev, d0)
    np.testing.assert_array_equal(ev, e0)
    with pytest.raises(ValueError, match="X_eval"):
        shift_score_rank(dev, ev[:, :2], seed=5)


def test_worst_subsample_share_one_is_the_full_sample_and_the_size_is_exact():
    rng = np.random.default_rng(12)
    n = 30
    x = rng.standard_normal((n, 1))
    y = (rng.random(n) < _expit(x[:, 0])).astype(int)
    score = rng.random(n)

    def predict(a):
        return _expit(a[:, 0])

    full = worst_subsample_performance(predict, x, y, score, 1.0)
    auc = auc_with_variance(predict(x), y)[0]
    assert full.n == n and full.auc == full.auc_full == auc
    assert full.brier == full.brier_full == float(np.mean((predict(x) - y) ** 2))
    # ceil(0.5 * 30) = 15; and 0.1 * 30 is 3.0000000000000004 in floats, so the size must come from exact arithmetic
    assert worst_subsample_performance(predict, x, y, score, 0.5).n == 15
    y2 = np.tile([0, 1], 15)
    assert worst_subsample_performance(predict, x, y2, np.arange(n, dtype=float), 0.1).n == 3


def test_worst_subsample_takes_the_highest_scores_and_breaks_ties_by_row_order():
    x = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])

    def predict(a):
        return (a[:, 0] + 1.0) / 10.0

    res = worst_subsample_performance(predict, x, y, score, 0.5)  # rows 1, 2, 4, 6 (scores 0.9, 0.8, 0.7, 0.6)
    p = predict(x)[[1, 2, 4, 6]]
    yy = y[[1, 2, 4, 6]]
    assert res.n == 4
    assert res.auc == auc_with_variance(p, yy)[0]
    assert res.brier == float(np.mean((p - yy) ** 2))
    tied = worst_subsample_performance(predict, x, y, np.zeros(8), 0.5)  # ties: the first rows in row order
    q, qq = predict(x)[:4], y[:4]
    assert tied.n == 4 and tied.auc == auc_with_variance(q, qq)[0]


def test_worst_subsample_refusals():
    x = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 0, 1])

    def predict(a):
        return (a[:, 0] + 1.0) / 10.0

    with pytest.raises(ValueError, match="share"):
        worst_subsample_performance(predict, x, y, np.arange(6.0), 0.0)
    with pytest.raises(ValueError, match="share"):
        worst_subsample_performance(predict, x, y, np.arange(6.0), 1.5)
    with pytest.raises(ValueError, match="score"):
        worst_subsample_performance(predict, x, y, np.arange(5.0), 0.5)
    with pytest.raises(ValueError, match="single class"):
        worst_subsample_performance(predict, x, np.array([0, 0, 0, 1, 1, 1]), np.arange(6.0), 0.5)
    with pytest.raises(ValueError, match="n"):
        SubsampleResult(n=0, auc=0.5, brier=0.2, auc_full=0.5, brier_full=0.2)


def test_oot_window_performance_omits_and_counts_single_class_windows():
    rng = np.random.default_rng(13)
    n = 240
    window = np.repeat([2019, 2020, 2021, 2022], 60)
    x = rng.standard_normal((n, 1))
    y = (rng.random(n) < _expit(x[:, 0])).astype(int)
    y[window == 2021] = 0  # a planted single-class window
    out = oot_window_performance(lambda a: _expit(a[:, 0]), x, y, window)
    assert set(out) == {"2019", "2020", "2022", "omitted"}
    assert out["omitted"] == 1.0
    for w in (2019, 2020, 2022):
        m = window == w
        assert out[str(w)] == auc_with_variance(_expit(x[m, 0]), y[m])[0]
    with pytest.raises(ValueError, match="window"):
        oot_window_performance(lambda a: np.full(len(a), 0.5), x, y, window[:-1])
