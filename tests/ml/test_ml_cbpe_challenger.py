"""RV-1424 to RV-1427: the label-free performance estimate and the champion and challenger comparison
(`riskvalidation.ml.cbpe` and `riskvalidation.ml.challenger`, contract engine-ml-api 1.0.0)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.ml.cbpe import CbpeAuc, CbpeConfusion, cbpe_auc, cbpe_confusion
from riskvalidation.ml.challenger import PairedBrier, grade_disagreement, paired_brier_bootstrap
from riskvalidation.validation.discrimination import auc_with_variance


def test_cbpe_confusion_hand_computed_rows():
    # C.7 l.1054: P(yhat = y)_j = 1 - |yhat_j - p_j|; yhat = 1 when p >= threshold
    p = np.array([0.9, 0.3, 0.6, 0.2])
    c = cbpe_confusion(p, 0.5)  # yhat = 1, 0, 1, 0
    assert isinstance(c, CbpeConfusion)
    assert (c.tp, c.fp, c.tn, c.fn) == pytest.approx((0.9 + 0.6, 0.1 + 0.4, 0.7 + 0.8, 0.3 + 0.2), abs=1e-15)
    assert c.tp + c.fp + c.tn + c.fn == pytest.approx(4.0, abs=1e-12)
    low = cbpe_confusion(p, 0.95)  # everything predicted 0
    assert (
        (low.tp, low.fp) == (0.0, 0.0)
        and low.tn == pytest.approx(0.1 + 0.7 + 0.4 + 0.8)
        and low.fn == pytest.approx(0.9 + 0.3 + 0.6 + 0.2)
    )
    edge = cbpe_confusion(np.array([0.5]), 0.5)  # p equal to the threshold is predicted 1
    assert edge.tp == 0.5 and edge.fp == 0.5
    for bad, field in ((np.array([1.2]), "p"), (np.array([0.5]), "threshold")):
        with pytest.raises(ValueError, match=field):
            cbpe_confusion(bad, 1.5 if field == "threshold" else 0.5)
    with pytest.raises(AttributeError):
        c.tp = 1.0


def test_cbpe_auc_closed_form_on_a_calibrated_uniform_sample_and_assumptions():
    n = 20_000
    p = (np.arange(n) + 0.5) / n  # a uniform grid of calibrated probabilities: the exact AUC is 5/6
    est = cbpe_auc(p)
    assert est.label == "estimate"
    assert est.assumptions == ("calibrated-probabilities", "no-concept-drift", "no-extrapolation", "adequate-sample")
    assert est.estimated_auc == pytest.approx(5.0 / 6.0, abs=2e-4)
    assert isinstance(est, CbpeAuc) and 0.0 <= est.estimated_auc <= 1.0
    assert cbpe_auc(p, n_thresholds=1001).estimated_auc == pytest.approx(5.0 / 6.0, abs=2e-5)


def test_cbpe_auc_of_a_calibrated_simulated_sample_matches_the_labelled_auc():
    rng = np.random.default_rng(1)
    n = 30_000
    p = rng.beta(0.8, 2.5, n)
    y = (rng.random(n) < p).astype(int)  # the probabilities are exactly calibrated
    est = cbpe_auc(p).estimated_auc
    auc, var, _, _ = auc_with_variance(p, y)
    assert abs(est - auc) <= 4.0 * math.sqrt(var)  # within Monte Carlo error of the labelled AUC
    shrunk = cbpe_auc(0.5 * p + 0.25).estimated_auc  # a different (non-calibrated) input is a different estimate
    assert shrunk != est


def test_cbpe_auc_refusals():
    with pytest.raises(ValueError, match="p"):
        cbpe_auc(np.array([0.2, 1.5]))
    with pytest.raises(ValueError, match="n_thresholds"):
        cbpe_auc(np.array([0.2, 0.7]), n_thresholds=1)
    with pytest.raises(ValueError, match="p"):
        cbpe_auc(np.zeros(5))  # no expected positive: the ROC is undefined
    with pytest.raises(ValueError, match="label"):
        CbpeAuc(estimated_auc=0.7, label="measured", assumptions=("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="assumptions"):
        CbpeAuc(estimated_auc=0.7, label="estimate", assumptions=("a",))


def _briers(seed: int, n: int = 1000):
    rng = np.random.default_rng(seed)
    p_true = rng.beta(2.0, 5.0, n)
    y = (rng.random(n) < p_true).astype(int)
    return p_true, y, rng


def test_paired_brier_bootstrap_identical_forecasts_zero_difference_zero_width():
    p, y, _ = _briers(1)
    r = paired_brier_bootstrap(p, p, y, 300, seed=2)
    assert r == PairedBrier(diff=0.0, lo=0.0, hi=0.0, n_boot=300)


def test_paired_brier_bootstrap_point_estimate_interval_width_and_determinism():
    p_true, y, rng = _briers(2, n=1500)
    champ = np.clip(p_true + 0.05 * rng.standard_normal(1500), 0.01, 0.99)
    chall = np.clip(p_true + 0.15 * rng.standard_normal(1500), 0.01, 0.99)  # a noisier challenger: higher Brier
    r = paired_brier_bootstrap(champ, chall, y, 4000, seed=3)
    d = (chall - y) ** 2 - (champ - y) ** 2
    assert r.diff == pytest.approx(float(d.mean()), abs=1e-15) and r.diff > 0.0  # challenger minus champion
    assert r.lo <= r.diff <= r.hi
    half_width = 1.959963984540054 * float(d.std(ddof=1)) / math.sqrt(len(d))  # the normal-theory 95% half width
    assert (r.hi - r.lo) == pytest.approx(2.0 * half_width, rel=0.08)
    assert paired_brier_bootstrap(champ, chall, y, 4000, seed=3) == r
    assert paired_brier_bootstrap(champ, chall, y, 4000, seed=4) != r
    assert r.lo > 0.0  # the interval excludes zero in the champion's favour


def test_paired_brier_bootstrap_interval_contains_the_point_estimate():
    for seed in range(60):
        p_true, y, rng = _briers(seed, n=400)
        a = np.clip(p_true + 0.1 * rng.standard_normal(400), 0.01, 0.99)
        b = np.clip(p_true + 0.1 * rng.standard_normal(400), 0.01, 0.99)
        r = paired_brier_bootstrap(a, b, y, 200, seed=seed)
        assert r.lo <= r.diff <= r.hi, (seed, r)


def test_paired_brier_bootstrap_refusals():
    p, y, _ = _briers(3, 50)
    with pytest.raises(ValueError, match="p_challenger"):
        paired_brier_bootstrap(p, p[:-1], y, 100, seed=0)
    with pytest.raises(ValueError, match="n_boot"):
        paired_brier_bootstrap(p, p, y, 1, seed=0)
    with pytest.raises(ValueError, match="y"):
        paired_brier_bootstrap(p, p, np.full(50, 2), 100, seed=0)
    with pytest.raises(ValueError, match="lo"):
        PairedBrier(diff=0.0, lo=0.1, hi=0.0, n_boot=10)


def test_grade_disagreement_hand_table_symmetry_and_notches():
    edges = np.array([0.02, 0.05, 0.10])  # grades 0 to 3
    a = np.array([0.01, 0.03, 0.06, 0.20, 0.04])  # grades 0 1 2 3 1
    b = np.array([0.04, 0.01, 0.07, 0.03, 0.20])  # grades 1 0 2 1 3
    # grade distances 1 1 0 2 2
    assert grade_disagreement(a, b, edges) == 0.4
    assert grade_disagreement(b, a, edges) == 0.4  # symmetric
    assert grade_disagreement(a, b, edges, notches=1) == 0.8
    assert grade_disagreement(a, b, edges, notches=3) == 0.0
    assert grade_disagreement(a, a, edges) == 0.0
    with pytest.raises(ValueError, match="notches"):
        grade_disagreement(a, b, edges, notches=0)
    with pytest.raises(ValueError, match="pd_b"):
        grade_disagreement(a, b[:3], edges)
    with pytest.raises(ValueError, match="edges"):
        grade_disagreement(a, b, np.array([0.1, 0.05]))
