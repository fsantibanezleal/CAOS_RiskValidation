"""RV-1463 to RV-1468: situation testing against the paper's equations (arXiv 2502.01267 equations 3 to 10) and the
toy golden of the research addendum 7.6 (4), under both distance normalizations. The oracle below is an independent
pure-Python transcription of the equations; it is never the reference code's report columns."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.fairness.situation import (
    SituationResult,
    bonferroni_alpha,
    gower_distance,
    k_nearest,
    situation_statistic,
    situation_test,
    wald_interval_one_sided,
)

# The toy fixture of addendum 7.6 (4): four protected rows Score 10, 11, 12, 13 with Y 0, 0, 1, 1 and four others
# 10.5, 11.5, 12.5, 13.5 all Y 1; k = 2, alpha 0.05, tau 0, negative value 0.
TOY_X = np.array([[10.0], [11.0], [12.0], [13.0], [10.5], [11.5], [12.5], [13.5]])
TOY_GROUP = np.array([1, 1, 1, 1, 0, 0, 0, 0])
TOY_DECISION = np.array([0, 0, 1, 1, 1, 1, 1, 1])
Z95 = float(stats.norm.ppf(0.95))


# ---------------------------------------------------------------------------------------------------------------
# RV-1463


def test_gower_distance_hand_computed_under_both_normalizations():
    x1, x2 = np.array([1.0, 2.0, 5.0]), np.array([3.0, 7.0, 9.0])  # the categorical codes 2 and 7 differ
    categorical = np.array([False, True, False])
    # 'range': scale holds max - min; d = (|1-3|/4 + overlap 1 + |5-9|/10) / 3 = (0.5 + 1 + 0.4) / 3
    rng_scale = np.array([4.0, 1.0, 10.0])
    d = gower_distance(x1, x2, categorical, rng_scale)
    assert d == pytest.approx((0.5 + 1.0 + 0.4) / 3.0, abs=1e-15)
    assert d == gower_distance(x1, x2, categorical, rng_scale, normalization="range")
    assert 0.0 <= d <= 1.0
    # 'standard': scale holds the population standard deviation; the continuous term is the absolute z-score gap
    std_scale = np.array([2.0, 1.0, 0.5])
    d_std = gower_distance(x1, x2, categorical, std_scale, normalization="standard")
    assert d_std == pytest.approx((1.0 + 1.0 + 8.0) / 3.0, abs=1e-15)
    assert d_std > 1.0  # not bounded by 1
    # identical rows are at distance 0; the categorical term is an overlap (0 when equal)
    assert gower_distance(x1, x1, categorical, rng_scale) == 0.0
    # the overlap distance is 0 or 1 whatever the codes are: equal codes cost nothing, any two different codes cost 1
    assert gower_distance(np.array([1.0, 2.0, 5.0]), np.array([1.0, 2.0, 5.0]), categorical, rng_scale) == 0.0
    one_code_differs = gower_distance(np.array([1.0, 2.0, 5.0]), np.array([1.0, 7.0, 5.0]), categorical, rng_scale)
    assert one_code_differs == pytest.approx(1.0 / 3.0, abs=1e-15)
    # the population standard deviation (ddof 0) is the scale of the 'standard' normalization
    col = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert col.std() == 2.0
    assert gower_distance(np.array([5.0]), np.array([9.0]), np.array([False]), np.array([col.std()]),
                          normalization="standard") == pytest.approx(2.0, abs=1e-15)


def test_gower_distance_refusals():
    x1, x2 = np.array([1.0, 2.0]), np.array([2.0, 4.0])
    cat = np.array([False, False])
    with pytest.raises(ValueError, match="normalization"):
        gower_distance(x1, x2, cat, np.array([1.0, 1.0]), normalization="minmax")
    with pytest.raises(ValueError, match="scale"):
        gower_distance(x1, x2, cat, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="scale"):
        gower_distance(x1, x2, cat, np.array([1.0, -2.0]))
    with pytest.raises(ValueError, match="x2"):
        gower_distance(x1, np.array([1.0]), cat, np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="categorical"):
        gower_distance(x1, x2, np.array([False]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="x1"):
        gower_distance(np.array([1.0, np.nan]), x2, cat, np.array([1.0, 1.0]))
    # a zero scale on a categorical attribute is harmless: the overlap distance does not read it
    assert gower_distance(np.array([1.0, 0.0]), np.array([1.0, 4.0]), np.array([False, True]),
                          np.array([2.0, 0.0])) == pytest.approx(0.5, abs=1e-15)


# ---------------------------------------------------------------------------------------------------------------
# RV-1464


@pytest.mark.parametrize("normalization,scale", [("range", 3.0), ("standard", float(TOY_X[4:].std()))])
def test_k_nearest_toy_fixture_under_both_normalizations(normalization, scale):
    pool = TOY_X[4:]  # the other group: 10.5, 11.5, 12.5, 13.5
    cat, sc = np.array([False]), np.array([scale])
    assert list(k_nearest(np.array([10.0]), pool, 2, cat, sc, normalization)) == [0, 1]
    assert list(k_nearest(np.array([12.0]), pool, 2, cat, sc, normalization)) == [1, 2]  # 11.5 and 12.5 (0.5 each)
    # the protected group around its own row, with the row itself taken out: Score 10 has neighbours 11 and 12
    others = TOY_X[1:4]
    assert list(k_nearest(np.array([10.0]), others, 2, cat, sc, normalization)) == [0, 1]
    assert normalization in ("range", "standard")


def test_k_nearest_ties_by_row_order_and_bounds():
    pool = np.array([[1.0], [3.0], [3.0], [2.0], [1.0]])
    cat, sc = np.array([False]), np.array([2.0])
    # center 2: distances 1, 1, 1, 0, 1 -> the nearest is index 3, then ties broken by row order: 0, 1
    assert list(k_nearest(np.array([2.0]), pool, 3, cat, sc)) == [3, 0, 1]
    assert list(k_nearest(np.array([2.0]), pool, 5, cat, sc)) == [3, 0, 1, 2, 4]
    with pytest.raises(ValueError, match="k"):
        k_nearest(np.array([2.0]), pool, 6, cat, sc)
    with pytest.raises(ValueError, match="k"):
        k_nearest(np.array([2.0]), pool, 0, cat, sc)
    # a categorical attribute enters as an overlap: the exact-match row wins
    pool2 = np.array([[1.0, 0.0], [1.0, 5.0]])
    got = k_nearest(np.array([1.0, 5.0]), pool2, 1, np.array([False, True]), np.array([1.0, 1.0]))
    assert list(got) == [1]
    with pytest.raises(ValueError, match="center"):
        k_nearest(np.array([1.0, 2.0]), pool, 1, cat, sc)


# ---------------------------------------------------------------------------------------------------------------
# RV-1465 and RV-1466


def test_situation_statistic_golden():
    delta, p_c, p_t = situation_statistic(np.array([0, 1]), np.array([1, 1]))
    assert (p_c, p_t, delta) == (0.5, 0.0, 0.5)
    delta2, p_c2, p_t2 = situation_statistic(np.array([1, 1, 1]), np.array([0, 0, 1]), negative=1)
    assert (p_c2, p_t2) == (1.0, pytest.approx(1 / 3)) and delta2 == pytest.approx(2 / 3)
    assert -1.0 <= situation_statistic(np.array([1, 1]), np.array([0, 0]))[0] <= 1.0
    assert situation_statistic(np.array([1, 1]), np.array([0, 0]))[0] == -1.0
    with pytest.raises(ValueError, match="decision_test"):
        situation_statistic(np.array([0, 1]), np.array([1, 1, 1]))
    with pytest.raises(ValueError, match="decision_control"):
        situation_statistic(np.array([0, 2]), np.array([1, 1]))
    with pytest.raises(ValueError, match="negative"):
        situation_statistic(np.array([0, 1]), np.array([1, 1]), negative=2)


def test_wald_interval_one_sided_toy_golden():
    delta, p_c, p_t = situation_statistic(np.array([0, 1]), np.array([1, 1]))
    lower = wald_interval_one_sided(delta, p_c, p_t, k=2, alpha=0.05)
    w = delta - lower
    assert Z95 == pytest.approx(1.644854, abs=1e-6)
    assert w == pytest.approx(0.58154, abs=1e-5)  # the exact quantile, not the reference code's rounded 1.645
    assert lower == pytest.approx(-0.0815, abs=1e-4)
    assert lower < 0.0  # below tau = 0: the toy claim is not significant
    assert w != pytest.approx(0.581595, abs=1e-7)  # the reference code's figure is not claimed


def test_wald_interval_formula_and_refusals():
    rng = np.random.default_rng(4)
    for _ in range(50):
        p_c, p_t = rng.random(2)
        k = int(rng.integers(1, 80))
        alpha = float(rng.choice([0.01, 0.05, 0.10]))
        delta = p_c - p_t
        expect = delta - stats.norm.ppf(1 - alpha) * math.sqrt((p_c * (1 - p_c) + p_t * (1 - p_t)) / k)
        assert wald_interval_one_sided(delta, p_c, p_t, k, alpha) == pytest.approx(expect, abs=1e-12)
    # the plus sign of the numerator: unequal variances add (the EAAMO typo would subtract and could go negative)
    assert wald_interval_one_sided(0.0, 0.5, 0.5, 4) == pytest.approx(-Z95 * math.sqrt(0.5 / 4), abs=1e-12)
    assert wald_interval_one_sided(0.3, 1.0, 0.7, 10, 0.05) < 0.3
    for bad in (dict(k=0), dict(alpha=0.0), dict(alpha=1.0), dict(alpha=-0.1)):
        args = dict(delta=0.5, p_c=0.5, p_t=0.0, k=2, alpha=0.05)
        args.update(bad)
        with pytest.raises(ValueError, match="k|alpha"):
            wald_interval_one_sided(**args)
    with pytest.raises(ValueError, match="p_c"):
        wald_interval_one_sided(0.5, 1.5, 0.0, 2)


# ---------------------------------------------------------------------------------------------------------------
# RV-1467: the oracle


def _scales(x, group, normalization):
    protected, other = x[group == 1], x[group == 0]
    if normalization == "range":
        return protected.max(axis=0) - protected.min(axis=0), other.max(axis=0) - other.min(axis=0)
    sd = x.std(axis=0)
    return sd, sd


def _dist(a, b, scale):
    return sum(abs(float(u) - float(v)) / float(s) for u, v, s in zip(a, b, scale, strict=True)) / len(a)


def _knn(center, rows, idx, k, scale):
    ranked = sorted(idx, key=lambda i: (_dist(center, rows[i], scale), i))
    return ranked[:k]


def oracle(x, group, decision, k, tau=0.0, alpha=0.05, normalization="range", counterfactual=None):
    """Equations (3) to (10) of the companion paper, centers excluded, negative decision 0."""
    z = float(stats.norm.ppf(1 - alpha))
    ctrl_scale, test_scale = _scales(x, group, normalization)
    prot = [i for i in range(len(x)) if group[i] == 1]
    other = [i for i in range(len(x)) if group[i] == 0]
    cases = significant = 0
    for i in prot:
        ctr = _knn(x[i], x, [j for j in prot if j != i], k, ctrl_scale)
        centre = x[i] if counterfactual is None else counterfactual[i]
        tst = _knn(centre, x, other, k, test_scale)
        p_c = sum(decision[j] == 0 for j in ctr) / k
        p_t = sum(decision[j] == 0 for j in tst) / k
        delta = p_c - p_t
        lower = delta - z * math.sqrt((p_c * (1 - p_c) + p_t * (1 - p_t)) / k)
        cases += delta > tau
        significant += lower > tau
    return len(prot), int(cases), int(significant)


@pytest.mark.parametrize("normalization", ["range", "standard"])
def test_situation_test_toy_golden_under_both_normalizations(normalization):
    res = situation_test(TOY_X, TOY_GROUP, TOY_DECISION, k=2, tau=0.0, alpha=0.05, normalization=normalization)
    assert isinstance(res, SituationResult)
    assert (res.n_complainants, res.n_cases, res.n_significant) == (4, 4, 0)  # delta 0.5, lower bound -0.0815
    assert (res.k, res.tau, res.alpha) == (2, 0.0, 0.05)
    assert oracle(TOY_X, TOY_GROUP, TOY_DECISION, 2, normalization=normalization) == (4, 4, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.k = 3  # type: ignore[misc]


def _planted(seed, n=400, bias=0.6, p_other=0.2):
    rng = np.random.default_rng(seed)
    group = (rng.random(n) < 0.4).astype(int)
    x = rng.uniform(0, 10, (n, 2))
    p_neg = np.where(group == 1, p_other + bias, p_other)
    decision = (rng.random(n) >= p_neg).astype(int)  # 0 is the negative decision
    return x, group, decision


def test_situation_test_planted_discrimination_is_significant():
    x, group, decision = _planted(1)
    biased = situation_test(x, group, decision, k=15)
    assert biased.n_significant / biased.n_complainants > 0.8
    assert biased == SituationResult(biased.n_complainants, biased.n_cases, biased.n_significant, 15, 0.0, 0.05)
    assert (biased.n_complainants, biased.n_cases, biased.n_significant) == oracle(x, group, decision, 15)
    # a tolerance tau near the true gap (0.6) leaves almost nothing significant
    tolerant = situation_test(x, group, decision, k=15, tau=0.55)
    assert tolerant.n_significant < biased.n_significant * 0.25
    assert (tolerant.n_complainants, tolerant.n_cases, tolerant.n_significant) == oracle(
        x, group, decision, 15, tau=0.55)
    # the null: no bias, and a one-sided test at 5 percent rarely calls a complainant significant
    x0, g0, d0 = _planted(2, bias=0.0)
    null = situation_test(x0, g0, d0, k=15)
    assert null.n_significant / null.n_complainants < 0.15


def test_situation_test_mutatis_mutandis_uses_the_counterfactual():
    # protected rows decide 0 (negative) everywhere; the other group decides 1 near them and 0 far away (20, 21)
    x = np.array([[10.0], [11.0], [12.0], [13.0], [10.5], [11.5], [12.5], [13.5], [20.0], [21.0]])
    group = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    decision = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    cp = situation_test(x, group, decision, k=2)
    assert (cp.n_complainants, cp.n_cases, cp.n_significant) == (4, 4, 4)  # p_c 1.0, p_t 0.0, w = 0
    cf = x.copy()
    cf[:4] = np.array([[20.0], [20.5], [21.0], [21.5]])  # the generated counterfactual of each complainant
    mm = situation_test(x, group, decision, k=2, comparator="mutatis-mutandis", counterfactual=cf)
    assert (mm.n_cases, mm.n_significant) == (0, 0)  # the comparators decide 0 too: delta 0
    assert (mm.n_complainants, mm.n_cases, mm.n_significant) == oracle(x, group, decision, 2, counterfactual=cf)
    # a counterfactual equal to the factual rows reproduces the ceteris paribus result
    same = situation_test(x, group, decision, k=2, comparator="mutatis-mutandis", counterfactual=x.copy())
    assert same == cp
    # the 'ceteris-paribus' comparator ignores a counterfactual it is given
    assert situation_test(x, group, decision, k=2, counterfactual=cf) == cp
    with pytest.raises(ValueError, match="counterfactual"):
        situation_test(x, group, decision, k=2, comparator="mutatis-mutandis")
    with pytest.raises(ValueError, match="counterfactual"):
        situation_test(x, group, decision, k=2, comparator="mutatis-mutandis", counterfactual=cf[:5])


def test_situation_test_refusals_and_invariants():
    base = dict(X=TOY_X, group=TOY_GROUP, decision=TOY_DECISION, k=2)
    cases = [
        (dict(comparator="paired"), "comparator"),
        (dict(normalization="zscore"), "normalization"),
        (dict(group=np.array([1, 1, 1, 1, 0, 0, 0, 2])), "group"),
        (dict(decision=np.array([0, 0, 1, 1, 1, 1, 1, 7])), "decision"),
        (dict(X=np.where(np.arange(8)[:, None] == 3, np.nan, TOY_X)), "X"),
        (dict(X=TOY_X[:7]), "X"),
        (dict(k=0), "k"),
        (dict(k=4), "k"),  # a protected group of 4 leaves 3 once the complainant is excluded
        (dict(alpha=1.0), "alpha"),
        (dict(tau=2.0), "tau"),
        (dict(group=np.zeros(8, dtype=int)), "group"),
        (dict(X=np.hstack([TOY_X, np.zeros((8, 1))])), "scale"),  # a constant attribute: a zero range
    ]
    for override, field in cases:
        with pytest.raises(ValueError, match=field):
            situation_test(**{**base, **override})
    res = situation_test(TOY_X, TOY_GROUP, TOY_DECISION, k=2)
    assert res.n_significant <= res.n_cases <= res.n_complainants
    x, g, d = _planted(3, n=120)
    x0, g0, d0 = x.copy(), g.copy(), d.copy()
    r = situation_test(x, g, d, k=5)
    assert r.n_significant <= r.n_cases <= r.n_complainants
    assert np.array_equal(x, x0) and np.array_equal(g, g0) and np.array_equal(d, d0)
    with pytest.raises(ValueError, match="n_significant"):
        SituationResult(n_complainants=3, n_cases=2, n_significant=3, k=1, tau=0.0, alpha=0.05)


def test_situation_test_range_and_standard_scalings_differ():
    # the paper's text (range) and the reference code (standard scaler) disagree on heavy-tailed attributes: the
    # published loan-application counts are claimed only under 'standard'
    rng = np.random.default_rng(31)
    n = 300
    group = (rng.random(n) < 0.4).astype(int)
    x = np.column_stack([rng.normal(size=n), rng.standard_cauchy(n).clip(-200, 200)])
    decision = (rng.random(n) >= np.where(group == 1, 0.55, 0.3)).astype(int)
    out = {}
    for normalization in ("range", "standard"):
        got = situation_test(x, group, decision, k=10, normalization=normalization)
        assert (got.n_complainants, got.n_cases, got.n_significant) == oracle(
            x, group, decision, 10, normalization=normalization)
        out[normalization] = (got.n_cases, got.n_significant)
    assert out["range"] != out["standard"]


# ---------------------------------------------------------------------------------------------------------------
# RV-1468


def test_bonferroni_alpha():
    assert bonferroni_alpha(0.05, 1) == 0.05
    assert bonferroni_alpha(0.05, 2) == 0.05 / 2
    assert bonferroni_alpha(0.10, 4) == 0.10 / 4
    for q in (0, -1):
        with pytest.raises(ValueError, match="q"):
            bonferroni_alpha(0.05, q)
    with pytest.raises(ValueError, match="alpha"):
        bonferroni_alpha(0.0, 2)


# ---------------------------------------------------------------------------------------------------------------
# RV-1467: the published counts of the companion paper, claimed only under normalization='standard'

LOAN_CSV = "FAIRNESS_LOAN_APPLICATION_CSV"
LOAN_CF_CSV = "FAIRNESS_LOAN_APPLICATION_CF_CSV"
# addendum 7.6 (2): ST cases and significant, then CST without centers, at k = 15, 30, 50, 100, 250
PUBLISHED_ST = {15: (55, 44), 30: (65, 57), 50: (84, 65), 100: (107, 85), 250: (204, 148)}
PUBLISHED_CST = {15: (288, 272), 30: (313, 306), 50: (342, 331), 100: (395, 383), 250: (534, 519)}


def _read_pipe_csv(path: str) -> list[list[str]]:
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh, delimiter="|"))[1:]


def test_situation_test_reproduces_the_published_loan_application_counts_under_standard():
    """Optional reconciliation: the loan-application file of the companion repository (no licence declared, so it is not
    stored here) read from the path in FAIRNESS_LOAN_APPLICATION_CSV (columns LoanApproval, AnnualSalary,
    AccountBalance, u1, u2, Gender, separated by '|'). Standard situation testing reproduces the published ST counts
    under 'standard'; the counterfactual counts reproduce when the counterfactual file
    (FAIRNESS_LOAN_APPLICATION_CF_CSV) is expressed in the factual attribute space through its own scaler, which is
    what the reference code does (finding F3)."""
    import os

    path = os.environ.get(LOAN_CSV)
    if not path or not os.path.exists(path):
        pytest.skip(f"set {LOAN_CSV} to the companion repository's data/LoanApplication_v2.csv to run this")
    rows = _read_pipe_csv(path)
    decision = np.array([1 if float(r[0]) > 0 else 0 for r in rows])
    x = np.array([[float(r[1]), float(r[2])] for r in rows])
    group = np.array([int(r[5]) for r in rows])
    assert (len(rows), int(group.sum())) == (4993, 1712)
    for k, (cases, significant) in PUBLISHED_ST.items():
        got = situation_test(x, group, decision, k=k, normalization="standard")
        assert (got.n_cases, got.n_significant) == (cases, significant), k
    cf_path = os.environ.get(LOAN_CF_CSV)
    if cf_path and os.path.exists(cf_path):
        cf = np.array([[float(r[1]), float(r[2])] for r in _read_pipe_csv(cf_path)])
        rescaled = (cf - cf.mean(axis=0)) / cf.std(axis=0) * x.std(axis=0) + x.mean(axis=0)
        for k, (cases, significant) in PUBLISHED_CST.items():
            got = situation_test(x, group, decision, k=k, normalization="standard", comparator="mutatis-mutandis",
                                 counterfactual=rescaled)
            assert (got.n_cases, got.n_significant) == (cases, significant), k


# ---------------------------------------------------------------------------------------------------------------
# F3 (1.1.0): the categorical mask of situation_test, the paper's Gower distance (equations 3 to 5)

CAT_X = np.array([[0.0, 2.0], [6.0, 0.0], [9.0, 1.0], [2.0, 1.0], [7.0, 0.0], [6.0, 1.0], [1.0, 2.0]])
CAT_GROUP = np.array([1, 1, 1, 0, 0, 0, 0])
CAT_DECISION = np.array([1, 0, 1, 0, 0, 1, 1])


def test_situation_test_categorical_golden_by_hand():
    """Column 1 is categorical (overlap), column 0 continuous (Manhattan over max - min of the search space), k = 1.
    Control scale (protected col 0) 9, test scale (others col 0) 6. By hand, distance = (md + ol) / 2:
    P0 (0, 2): control P1 5/6 vs P2 1, nearest P1 (negative, p_c 1); test O0 2/3, O1 13/12, O2 1, O3 1/12, nearest O3
    (favourable, p_t 0): delta 1, lower bound 1.
    P1 (6, 0): control P0 5/6 vs P2 2/3, nearest P2 (favourable, p_c 0): delta <= 0.
    P2 (9, 1): control P1 2/3 vs P0 1, nearest P1 (negative, p_c 1); test O2 1/4 is nearest (favourable, p_t 0):
    delta 1. Two cases, both significant. Read as continuous the same table gives one case."""
    cat = np.array([False, True])
    res = situation_test(CAT_X, CAT_GROUP, CAT_DECISION, k=1, categorical=cat)
    assert (res.n_complainants, res.n_cases, res.n_significant) == (3, 2, 2)
    plain = situation_test(CAT_X, CAT_GROUP, CAT_DECISION, k=1)
    assert (plain.n_cases, plain.n_significant) == (1, 1)  # md on the code: the categorical column is mismeasured
    assert situation_test(CAT_X, CAT_GROUP, CAT_DECISION, k=1, categorical=None) == plain
    assert situation_test(CAT_X, CAT_GROUP, CAT_DECISION, k=1, categorical=np.array([False, False])) == plain


def test_situation_test_categorical_under_standard_uses_overlap_and_standardises_the_rest_only():
    cat = np.array([False, True])
    res = situation_test(CAT_X, CAT_GROUP, CAT_DECISION, k=2, categorical=cat, normalization="standard")
    sd0 = CAT_X[:, 0].std()
    prot, other = [0, 1, 2], [3, 4, 5, 6]

    def dist(a, b):
        return (abs(CAT_X[a, 0] - CAT_X[b, 0]) / sd0 + (CAT_X[a, 1] != CAT_X[b, 1])) / 2.0

    cases = sig = 0
    for i in prot:
        c = sorted((j for j in prot if j != i), key=lambda j: (dist(i, j), j))[:2]
        t = sorted(other, key=lambda j: (dist(i, j), j))[:2]
        p_c = sum(CAT_DECISION[j] == 0 for j in c) / 2
        p_t = sum(CAT_DECISION[j] == 0 for j in t) / 2
        delta = p_c - p_t
        cases += delta > 0
        sig += delta - stats.norm.ppf(0.95) * math.sqrt((p_c * (1 - p_c) + p_t * (1 - p_t)) / 2) > 0
    assert (res.n_cases, res.n_significant) == (cases, sig)


def test_situation_test_categorical_refusals():
    with pytest.raises(ValueError, match="categorical"):
        situation_test(CAT_X, CAT_GROUP, CAT_DECISION, k=1, categorical=np.array([False, True, False]))
    # a constant categorical column has no scale to read: it is not a zero-scale error
    x = CAT_X.copy()
    x[:, 1] = 1.0
    situation_test(x, CAT_GROUP, CAT_DECISION, k=1, categorical=np.array([False, True]))
    with pytest.raises(ValueError, match="zero scale"):
        situation_test(x, CAT_GROUP, CAT_DECISION, k=1)
