"""RV-1407 to RV-1413: seed stability and the explanation checks (`riskvalidation.ml.seeds` and
`riskvalidation.ml.explanations`, contract engine-ml-api 1.0.0)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from ml_testkit import logistic_xy
from scipy import stats

from riskvalidation.ml.explanations import (
    OverlapResult,
    monotonicity_check,
    rank_agreement,
    reason_code_overlap,
    shapley_additivity_gap,
    sign_check,
)
from riskvalidation.ml.seeds import (
    RankStability,
    SeedDispersion,
    attribution_rank_stability,
    grade_assignment_stability,
    seed_auc_dispersion,
)
from riskvalidation.validation.discrimination import auc_with_variance


def test_seed_dispersion_of_a_seed_blind_train_function_is_zero():
    x, y = logistic_xy(500, 3, 1)

    def train(seed):  # ignores its seed
        return lambda a: 1.0 / (1.0 + np.exp(-a[:, 0]))

    d = seed_auc_dispersion(train, x, y, [1, 2, 3, 4])
    assert d.sd == 0.0 and d.minimum == d.mean == d.maximum and d.n_seeds == 4
    assert d.mean == auc_with_variance(train(0)(x), y)[0]


def test_seed_dispersion_statistics_match_numpy_on_the_seed_aucs():
    x, y = logistic_xy(600, 3, 2)
    rng_noise = {s: np.random.default_rng(s).standard_normal(len(y)) for s in (3, 5, 8, 13)}

    def train(seed):
        noise = rng_noise[seed]
        return lambda a: 1.0 / (1.0 + np.exp(-(a[:, 0] + noise)))

    d = seed_auc_dispersion(train, x, y, [3, 5, 8, 13])
    aucs = np.array([auc_with_variance(train(s)(x), y)[0] for s in (3, 5, 8, 13)])
    assert d.mean == pytest.approx(aucs.mean(), abs=1e-15)
    assert d.sd == pytest.approx(aucs.std(ddof=1), abs=1e-15)  # sample standard deviation, ddof 1
    assert d.minimum == aucs.min() and d.maximum == aucs.max()
    assert d.minimum <= d.mean <= d.maximum and d.sd > 0.0


def test_seed_dispersion_refusals_and_dataclass():
    x, y = logistic_xy(100, 2, 3)
    train = lambda seed: (lambda a: 1.0 / (1.0 + np.exp(-a[:, 0])))  # noqa: E731
    with pytest.raises(ValueError, match="seeds"):
        seed_auc_dispersion(train, x, y, [1])
    with pytest.raises(ValueError, match="seeds"):
        seed_auc_dispersion(train, x, y, [1, 1, 2])
    with pytest.raises(ValueError, match="predict"):
        seed_auc_dispersion(lambda s: (lambda a: np.full(3, 0.5)), x, y, [1, 2])
    with pytest.raises(AttributeError):
        SeedDispersion(n_seeds=2, mean=0.7, sd=0.0, minimum=0.7, maximum=0.7).sd = 1.0
    with pytest.raises(ValueError, match="sd"):
        SeedDispersion(n_seeds=2, mean=0.7, sd=-0.1, minimum=0.6, maximum=0.8)
    with pytest.raises(ValueError, match="minimum"):
        SeedDispersion(n_seeds=2, mean=0.5, sd=0.1, minimum=0.6, maximum=0.8)


def test_grade_assignment_stability_hand_table_and_identity():
    edges = np.array([0.1, 0.3])  # grades 0 (< 0.1), 1 ([0.1, 0.3)), 2 (>= 0.3)
    pd = np.array([[0.05, 0.20, 0.40, 0.09],
                   [0.05, 0.25, 0.35, 0.12],
                   [0.06, 0.20, 0.50, 0.31]])
    # grades: seed 1 [0 1 2 0], seed 2 [0 1 2 1], seed 3 [0 1 2 2]: only the fourth obligor changes grade
    assert grade_assignment_stability(pd, edges) == 0.25
    pd2 = pd.copy()
    pd2[2, 1] = 0.35  # the second obligor now is grade 2 in seed 3
    assert grade_assignment_stability(pd2, edges) == 0.5
    assert grade_assignment_stability(np.tile(pd[0], (4, 1)), edges) == 0.0  # identical seeds
    assert grade_assignment_stability(pd[:1], edges) == 0.0  # one seed
    # an edge value belongs to the upper grade (the bins are [e_k, e_k+1))
    assert grade_assignment_stability(np.array([[0.1], [0.0999999]]), edges) == 1.0


def test_grade_assignment_stability_refuses_bad_edges_and_pd():
    with pytest.raises(ValueError, match="edges"):
        grade_assignment_stability(np.full((2, 3), 0.2), np.array([0.3, 0.1]))
    with pytest.raises(ValueError, match="pd_by_seed"):
        grade_assignment_stability(np.array([[0.2, 1.2]]), np.array([0.1]))
    with pytest.raises(ValueError, match="pd_by_seed"):
        grade_assignment_stability(np.full(3, 0.2), np.array([0.1]))


def test_attribution_rank_stability_identity_and_scipy_reference():
    ident = np.tile(np.array([0.5, 0.2, 0.9, 0.1]), (3, 1))
    r = attribution_rank_stability(ident)
    assert r.mean_tau == 1.0 and r.min_tau == 1.0
    rows = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    r = attribution_rank_stability(rows)
    assert r.mean_tau == pytest.approx(-1.0 / 3.0, abs=1e-15) and r.min_tau == -1.0
    rng = np.random.default_rng(4)
    a = np.round(rng.random((5, 9)), 1)  # ties on purpose: tau-b
    taus = [stats.kendalltau(a[i], a[j]).statistic for i, j in itertools.combinations(range(5), 2)]
    out = attribution_rank_stability(a)
    assert out.mean_tau == pytest.approx(np.mean(taus), abs=1e-15) and out.min_tau == pytest.approx(min(taus), abs=1e-15)
    assert out.min_tau <= out.mean_tau
    with pytest.raises(ValueError, match="attributions_by_seed"):
        attribution_rank_stability(a[:1])
    with pytest.raises(ValueError, match="constant"):
        attribution_rank_stability(np.ones((3, 4)))
    with pytest.raises(ValueError, match="min_tau"):
        RankStability(mean_tau=0.1, min_tau=0.5)


def test_reason_code_overlap_identity_and_hand_computed_permutation():
    rng = np.random.default_rng(5)
    base = rng.standard_normal((40, 8))
    same = reason_code_overlap(np.stack([base, base, base]))
    assert same.mean == 1.0 and same.p05 == 1.0 and same.top_k == 4 and same.n_refits == 3
    # two obligors, six features; the top four by absolute attribution are columns 0 to 3 in refit 1
    a1 = np.array([[6.0, -5.0, 4.0, 3.0, 2.0, 1.0],
                   [6.0, -5.0, 4.0, 3.0, 2.0, 1.0]])
    a2 = a1.copy()
    a2[0, [3, 4]] = a1[0, [4, 3]]  # a planted swap of two columns for the first obligor: top four {0, 1, 2, 4}
    res = reason_code_overlap(np.stack([a1, a2]))
    assert res.mean == pytest.approx((0.75 + 1.0) / 2.0, abs=1e-15)
    assert res.p05 == pytest.approx(np.percentile([0.75, 1.0], 5.0), abs=1e-15) and res.p05 <= res.mean
    two = reason_code_overlap(np.stack([a1, a2]), top_k=2)
    assert two.top_k == 2 and two.mean == 1.0  # the top two are untouched by the swap
    assert OverlapResult(top_k=4, n_refits=2, mean=0.9, p05=0.7).p05 <= 0.9


def test_reason_code_overlap_refusals():
    a = np.ones((2, 5, 3))
    with pytest.raises(ValueError, match="top_k"):
        reason_code_overlap(a, top_k=4)
    with pytest.raises(ValueError, match="attributions_by_refit"):
        reason_code_overlap(a[:1], top_k=2)
    with pytest.raises(ValueError, match="attributions_by_refit"):
        reason_code_overlap(np.ones((2, 3)), top_k=2)
    # finding F1 of the unit: p05 <= mean is not a theorem of the shares (96 obligors at 1 and 4 at 0 give a 5th percentile
    # of 1 above the mean 0.96), so the function reports the exact percentile and the dataclass does not enforce it
    a1 = np.tile(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), (100, 1))
    a2 = a1.copy()
    a2[:4] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # reversed ranking for four obligors
    skew = reason_code_overlap(np.stack([a1, a2]), top_k=2)
    shares = np.where(np.arange(100) < 4, 0.0, 1.0)
    assert skew.mean == pytest.approx(shares.mean(), abs=1e-15)
    assert skew.p05 == pytest.approx(np.percentile(shares, 5.0), abs=1e-15) and skew.p05 > skew.mean
    with pytest.raises(ValueError, match="top_k"):
        OverlapResult(top_k=0, n_refits=2, mean=0.5, p05=0.4)


def test_rank_agreement_equal_reversed_and_scipy_reference():
    v = np.array([0.4, 0.1, 0.7, 0.2, 0.9])
    assert rank_agreement(v, v) == 1.0
    assert rank_agreement(v, -v) == -1.0
    w = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    assert rank_agreement(v, w) == pytest.approx(stats.spearmanr(v, w).statistic, abs=1e-15)
    with pytest.raises(ValueError, match="constant"):
        rank_agreement(np.ones(5), v)
    with pytest.raises(ValueError, match="effect_range"):
        rank_agreement(v, v[:4])


def test_shapley_additivity_gap_identity_on_a_linear_model_and_a_planted_gap():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((500, 5))
    w, b = np.array([0.5, -1.0, 2.0, 0.0, 0.3]), 0.7
    pred = b + x @ w
    mean = x.mean(axis=0)
    phi = (x - mean) * w  # exact interventional Shapley values of a linear model
    phi0 = b + float(mean @ w)
    assert shapley_additivity_gap(pred, phi0, phi) < 1e-12
    bad = phi.copy()
    bad[17, 2] += 0.125
    assert shapley_additivity_gap(pred, phi0, bad) == pytest.approx(0.125, abs=1e-12)
    with pytest.raises(ValueError, match="attributions"):
        shapley_additivity_gap(pred[:-1], phi0, phi)


def test_sign_check_and_monotonicity_check():
    up = np.array([0.0, 0.2, 0.5, 0.9])
    assert sign_check(up, +1) is True and sign_check(up, -1) is False
    assert sign_check(up[::-1], -1) is True
    assert sign_check(up[::-1], +1) is False  # a planted reversed effect
    assert sign_check(np.array([0.0, 1.0, 0.0]), +1) is False  # no net change is not the expected sign
    with pytest.raises(ValueError, match="expected_sign"):
        sign_check(up, 0)
    bump = np.array([0.0, 1.0, 2.0, 1.5, 3.0])  # a planted dip of height 0.5 after the running maximum
    assert monotonicity_check(bump, +1, 0.0) == 0.5
    assert monotonicity_check(bump, +1, 0.5) == 0.0  # within the tolerance
    assert monotonicity_check(bump, +1, 0.4) == 0.5
    assert monotonicity_check(-bump, -1, 0.0) == 0.5
    assert monotonicity_check(up, +1, 0.0) == 0.0
    with pytest.raises(ValueError, match="tolerance"):
        monotonicity_check(up, +1, -1.0)
