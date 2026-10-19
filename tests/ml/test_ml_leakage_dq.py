"""RV-1420 to RV-1423: the leakage screen and the data-quality counts (`riskvalidation.ml.leakage` and
`riskvalidation.ml.dataquality`, contract engine-ml-api 1.0.0)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.ml.dataquality import (
    DataQualityCount,
    dq_completeness,
    dq_consistency,
    dq_label_consistency,
    dq_timeliness,
    dq_uniqueness,
    dq_validity,
)
from riskvalidation.ml.leakage import (
    AucResult,
    SingleFeatureAuc,
    entity_overlap,
    resemblance_auc,
    single_feature_auc,
    temporal_order_violations,
)
from riskvalidation.validation.discrimination import auc_with_variance, disc_auc


def test_temporal_order_violations_counts_future_cells_and_never_nat():
    obs = np.array(["2020-01-31", "2020-02-29", "2020-03-31"], dtype="datetime64[D]")
    feat = np.array([["2020-01-01", "2020-01-31"],      # equal to the observation date: not a violation
                     ["2020-03-01", "NaT"],             # a planted future cell; NaT is never a violation
                     ["2020-03-30", "2020-04-01"]], dtype="datetime64[D]")  # one future cell
    assert temporal_order_violations(feat, obs) == 2
    assert temporal_order_violations(feat[:, :1], obs) == 1
    clean = np.broadcast_to(obs[:, None], (3, 2)).copy()
    assert temporal_order_violations(clean, obs) == 0
    with pytest.raises(ValueError, match="observation_date"):
        temporal_order_violations(feat, obs[:2])
    with pytest.raises(ValueError, match="feature_dates"):
        temporal_order_violations(np.zeros((3, 2)), obs)


def test_single_feature_auc_finds_the_planted_leaking_column_in_either_direction():
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random(n) < 0.3).astype(int)
    x = rng.standard_normal((n, 4))
    x[:, 2] += 6.0 * y  # planted leak
    res = single_feature_auc(x, y)
    assert res.feature == 2 and res.suspect is True and 0.95 < res.max_auc <= 1.0
    x[:, 2] = -x[:, 2]  # a leak that ranks the other way: max(AUC, 1 - AUC) still finds it
    res2 = single_feature_auc(x, y)
    assert res2.feature == 2 and res2.max_auc == pytest.approx(res.max_auc, abs=1e-12)
    clean = single_feature_auc(rng.standard_normal((n, 4)), y)
    assert 0.5 <= clean.max_auc < 0.6 and clean.suspect is False
    a = disc_auc(x[:, 2], y).metric
    assert res2.max_auc == max(a, 1.0 - a)


def test_single_feature_auc_threshold_boundary_and_refusals():
    rng = np.random.default_rng(2)
    y = np.tile([0, 1], 200)
    x = rng.standard_normal((400, 3)) + 0.5 * y[:, None]
    base = single_feature_auc(x, y)
    assert single_feature_auc(x, y, threshold=base.max_auc).suspect is False  # suspect == (max_auc > threshold)
    assert single_feature_auc(x, y, threshold=base.max_auc - 1e-9).suspect is True
    with pytest.raises(ValueError, match="X"):
        single_feature_auc(np.array([[1.0, np.nan], [2.0, 3.0]]), np.array([0, 1]))
    with pytest.raises(ValueError, match="threshold"):
        single_feature_auc(x, y, threshold=1.5)
    with pytest.raises(AttributeError):
        base.feature = 1
    with pytest.raises(ValueError, match="max_auc"):
        SingleFeatureAuc(max_auc=0.4, feature=0, suspect=False)


def test_entity_overlap():
    assert entity_overlap(["a", "b"], ["c", "d"]) == 0.0
    assert entity_overlap(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert entity_overlap([1, 2, 3], ["2", "3", "4"]) == pytest.approx(2.0 / 3.0)  # compared as strings
    assert entity_overlap(["a"], ["a", "a", "b", "b"]) == 0.5  # a test entity counts once
    with pytest.raises(ValueError, match="test_ids"):
        entity_overlap(["a"], [])


def test_resemblance_auc_identical_samples_are_near_one_half_and_a_shift_is_found():
    rng = np.random.default_rng(3)
    dev = rng.standard_normal((1500, 3))
    for oot in (dev.copy(), rng.standard_normal((1500, 3))):
        r = resemblance_auc(dev, oot, seed=4)
        assert isinstance(r, AucResult) and r.n == 3000 and r.variance >= 0.0
        assert abs(r.auc - 0.5) <= 2.0 * math.sqrt(r.variance)  # within 2 standard errors of one half
    shifted = resemblance_auc(dev, rng.standard_normal((1500, 3)) + np.array([1.0, 0.0, 0.0]), seed=4)
    assert shifted.auc > 0.65
    assert resemblance_auc(dev, dev + 0.1, seed=4) == resemblance_auc(dev, dev + 0.1, seed=4)  # deterministic
    with pytest.raises(ValueError, match="X_oot"):
        resemblance_auc(dev, dev[:, :2], seed=1)
    with pytest.raises(ValueError, match="variance"):
        AucResult(auc=0.5, variance=-1.0, n=10)


def test_dq_completeness_counts_planted_nan_exactly():
    x = np.ones((4, 3))
    x[0, 0] = x[2, 0] = x[3, 2] = np.nan
    c = dq_completeness(x)
    assert (c.check, c.n_checked, c.n_failed) == ("completeness", 12, 3)
    assert c.share_ok == pytest.approx(1.0 - 3.0 / 12.0, abs=1e-15)
    assert c.by_column == (2, 0, 1)
    assert isinstance(c, DataQualityCount)
    assert dq_completeness(np.ones((5, 2))).share_ok == 1.0
    with pytest.raises(ValueError, match="X"):
        dq_completeness(np.ones(5))


def test_dq_validity_counts_out_of_range_cells_and_not_nan():
    x = np.array([[0.5, 10.0], [1.5, 3.0], [np.nan, -1.0], [0.0, 5.0]])
    c = dq_validity(x, np.array([0.0, 0.0]), np.array([1.0, 5.0]))
    # 7 cells are not NaN; out of range: 1.5, 10.0 and -1.0
    assert (c.check, c.n_checked, c.n_failed) == ("validity", 7, 3)
    assert c.share_ok == pytest.approx(1.0 - 3.0 / 7.0, abs=1e-15)
    assert dq_validity(x[[0, 3]], np.array([0.0, 0.0]), np.array([1.0, 10.0])).n_failed == 0  # bounds are inclusive
    with pytest.raises(ValueError, match="lo"):
        dq_validity(x, np.array([0.0]), np.array([1.0, 5.0]))
    with pytest.raises(ValueError, match="hi"):
        dq_validity(x, np.array([0.0, 0.0]), np.array([1.0, -5.0]))
    with pytest.raises(ValueError, match="X"):
        dq_validity(np.full((2, 2), np.nan), np.zeros(2), np.ones(2))


def test_dq_uniqueness_consistency_timeliness_and_label_consistency():
    u = dq_uniqueness(["a", "b", "a", "c", "a"])
    assert (u.check, u.n_checked, u.n_failed) == ("uniqueness", 5, 2)  # the first occurrence is kept
    assert u.share_ok == pytest.approx(0.6, abs=1e-15)
    c = dq_consistency(["x", "y", "z", "w"], ["x", "q", "z", "r"])
    assert (c.check, c.n_checked, c.n_failed) == ("consistency", 4, 2)
    with pytest.raises(ValueError, match="b"):
        dq_consistency(["x"], ["x", "y"])
    t = dq_timeliness(["2026-01-01", "2025-06-30", "2024-12-31", "2025-01-31"], "2026-01-31", 365)
    # ages in days: 30, 215, 396, 365 (the boundary is not stale): one stale row
    assert (t.check, t.n_checked, t.n_failed) == ("timeliness", 4, 1)
    assert dq_timeliness(["2026-02-15"], "2026-01-31", 0).n_failed == 0  # a future date is not stale
    with pytest.raises(ValueError, match="max_age_days"):
        dq_timeliness(["2026-01-01"], "2026-01-31", -1)
    with pytest.raises(ValueError, match="dates"):
        dq_timeliness(["31/01/2026"], "2026-01-31", 10)
    x = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 1, 1, 0])
    lab = dq_label_consistency(x, y)  # the first pair has different labels, the second pair agrees
    assert (lab.check, lab.n_checked, lab.n_failed) == ("label_consistency", 5, 2)
    assert dq_label_consistency(x, np.array([0, 0, 1, 1, 0])).n_failed == 0
    with pytest.raises(ValueError, match="y"):
        dq_label_consistency(x, y[:3])


def test_data_quality_count_invariants():
    ok = DataQualityCount(check="completeness", n_checked=10, n_failed=2, share_ok=0.8)
    assert ok.share_ok == 1.0 - 2 / 10
    with pytest.raises(ValueError, match="share_ok"):
        DataQualityCount(check="completeness", n_checked=10, n_failed=2, share_ok=0.9)
    with pytest.raises(ValueError, match="n_failed"):
        DataQualityCount(check="completeness", n_checked=10, n_failed=11, share_ok=-0.1)
    with pytest.raises(AttributeError):
        ok.n_failed = 0
    assert auc_with_variance([0.1, 0.9], [0, 1])[0] == 1.0  # the AUC used by the screens is the engine's
