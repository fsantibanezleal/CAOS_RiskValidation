"""RV-1418, RV-1419 and RV-1439: split conformal prediction (`riskvalidation.ml.conformal`, contract engine-ml-api
1.0.0) and its reconciliation with MAPIE (optional extra, read for this test: see the docstring of the last test)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from riskvalidation.ml.conformal import (
    CoverageResult,
    conformal_coverage,
    conformal_sets,
    split_conformal_quantile,
)


def test_split_conformal_quantile_goldens():
    s = np.arange(1.0, 101.0)
    assert split_conformal_quantile(s, 0.1) == 91.0  # k = ceil(101 * 0.9) = 91, the 91st smallest score
    assert np.quantile(s, 0.91, method="higher") == 92.0  # the `higher` method is NOT the reference
    assert split_conformal_quantile(s, 0.1) == np.quantile(s, 91 / 100, method="inverted_cdf")
    n19 = np.arange(1.0, 20.0)
    assert split_conformal_quantile(n19, 0.05) == 19.0  # k = ceil(20 * 0.95) = 19
    # a real float trap found by search (finding F2: the contract's examples 101 * 0.9 and 20 * 0.95 do not misround in
    # IEEE doubles): n = 149 and alpha = 0.18 give 150 * 0.82 = 123 exactly, and the float product 150 * (1 - 0.18)
    # ceils to 124
    assert math.ceil((149 + 1) * (1 - 0.18)) == 124
    assert split_conformal_quantile(np.arange(1.0, 150.0), 0.18) == 123.0
    assert split_conformal_quantile(np.arange(1.0, 6.0), 0.1) == math.inf  # k = 6 above n = 5: the whole label space


def test_split_conformal_quantile_reference_and_monotone_in_alpha():
    rng = np.random.default_rng(1)
    for n in (7, 19, 50, 101, 333):
        scores = rng.random(n)
        srt = np.sort(scores)
        prev = -math.inf
        for alpha in (0.5, 0.3, 0.2, 0.1, 0.05, 0.01):
            k = math.ceil(Fraction(n + 1) * (1 - Fraction(str(alpha))))
            q = split_conformal_quantile(scores, alpha)
            if k > n:
                assert q == math.inf
            else:
                assert q == srt[k - 1]
                assert q == np.quantile(scores, k / n, method="inverted_cdf")
            assert q >= prev  # monotone: a smaller alpha never gives a smaller quantile
            prev = q


def test_split_conformal_quantile_refusals_and_purity():
    s = np.array([3.0, 1.0, 2.0])
    keep = s.copy()
    split_conformal_quantile(s, 0.5)
    np.testing.assert_array_equal(s, keep)
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            split_conformal_quantile(s, bad)
    with pytest.raises(ValueError, match="scores"):
        split_conformal_quantile(np.array([]), 0.1)
    with pytest.raises(ValueError, match="scores"):
        split_conformal_quantile(np.array([1.0, np.nan]), 0.1)


def test_conformal_sets_hand_table_and_infinite_quantile():
    p1 = np.array([0.05, 0.5, 0.95])
    sets = conformal_sets(p1, 0.1)
    # label 0 has score p1, label 1 has score 1 - p1
    np.testing.assert_array_equal(sets, np.array([[True, False], [False, False], [False, True]]))
    assert sets.shape == (3, 2) and sets.dtype == bool
    assert conformal_sets(p1, math.inf).all()  # qhat = +inf: both labels everywhere
    assert conformal_sets(np.array([0.5]), 0.5).all()  # a score equal to qhat is in the set
    with pytest.raises(ValueError, match="p1"):
        conformal_sets(np.array([0.5, 1.5]), 0.1)
    with pytest.raises(ValueError, match="qhat"):
        conformal_sets(p1, float("nan"))


def test_conformal_coverage_keeps_the_binomial_value_beside_the_beta_binomial_p_value():
    sets = np.array([[True, False]] * 6 + [[False, True]] * 3 + [[False, False]] * 1)
    y = np.array([0] * 6 + [1] * 2 + [0] + [1])
    res = conformal_coverage(sets, y, 0.1, n_cal=99)
    k = math.ceil(100 * 0.9)  # 90
    assert res == CoverageResult(
        n=10, coverage=0.8, target=0.9, p_value=res.p_value, n_cal=99, k=k, p_value_binomial=res.p_value_binomial
    )
    assert res.p_value_binomial == stats.binomtest(8, 10, 0.9, alternative="two-sided").pvalue
    pmf = stats.betabinom.pmf(np.arange(11), 10, k, 99 + 1 - k)
    assert res.p_value == pytest.approx(float(pmf[pmf <= pmf[8] * (1.0 + 1e-7)].sum()), abs=1e-12)
    full = conformal_coverage(np.ones((50, 2), dtype=bool), np.zeros(50, dtype=int), 0.1, n_cal=99)
    assert full.coverage == 1.0 and full.target == 0.9
    assert 0.0 <= full.p_value <= 1.0
    with pytest.raises(ValueError, match="sets"):
        conformal_coverage(np.ones((3, 3), dtype=bool), np.zeros(3, dtype=int), 0.1, n_cal=9)
    with pytest.raises(ValueError, match="y"):
        conformal_coverage(np.ones((3, 2), dtype=bool), np.array([0, 1, 2]), 0.1, n_cal=9)
    with pytest.raises(ValueError, match="alpha"):
        conformal_coverage(np.ones((3, 2), dtype=bool), np.zeros(3, dtype=int), 1.0, n_cal=9)
    with pytest.raises(AttributeError):
        res.n = 3  # frozen


def test_conformal_coverage_is_two_sided_on_a_coverage_above_target():
    """Review minor m1 (mutant M4): every row covered at alpha 0.5 is above the target; the binomial value beside the
    light is two-sided, not the one-sided `greater`."""
    sets, y = np.array([[True, False]] * 10), np.zeros(10, dtype=int)
    res = conformal_coverage(sets, y, 0.5, n_cal=19)
    assert res.k == 10 and res.coverage == 1.0
    assert res.p_value_binomial == stats.binomtest(10, 10, 0.5, alternative="two-sided").pvalue
    assert res.p_value_binomial > stats.binomtest(10, 10, 0.5, alternative="greater").pvalue
    pmf = stats.betabinom.pmf(np.arange(11), 10, 10, 10)
    assert res.p_value == pytest.approx(float(pmf[pmf <= pmf[10] * (1.0 + 1e-7)].sum()), abs=1e-12)


def test_split_conformal_guarantee_holds_under_exchangeability():
    """Coverage of the split conformal set under exchangeability is at least 1 - alpha (06 C.7 l.1024)."""
    rng = np.random.default_rng(2)
    alpha, n_cal, n_test, reps = 0.1, 99, 4000, 200
    cover = []
    for _ in range(reps):
        p = rng.beta(2.0, 5.0, n_cal + n_test)
        y = (rng.random(n_cal + n_test) < p).astype(int)
        scores = np.where(y[:n_cal] == 1, 1.0 - p[:n_cal], p[:n_cal])
        q = split_conformal_quantile(scores, alpha)
        sets = conformal_sets(p[n_cal:], q)
        cover.append(conformal_coverage(sets, y[n_cal:], alpha, n_cal=n_cal).coverage)
    m, se = float(np.mean(cover)), float(np.std(cover, ddof=1) / math.sqrt(reps))
    assert m >= 1.0 - alpha - 4.0 * se


def test_mapie_reconciliation_of_the_split_conformal_quantile():
    """MAPIE 1.5.0, read in `mapie/utils.py` (the optional extra `ml`; skipped when it is absent or another version).

    - `_compute_regression_quantile(scores, [1 - alpha])` takes `np.nanquantile(scores, ceil((1 - alpha)(n + 1)) / n,
      method="lower")`: the k-th order statistic of this function, except where the float product misrounds (finding
      F2).
    - `_compute_classification_quantile(scores, [alpha])`, the one `SplitConformalClassifier` uses, takes
      `np.quantile(scores, (n + 1)(1 - alpha) / n, method="higher")` with a float product: the k-th or the (k + 1)-th
      order statistic, so its sets contain ours (finding F3). The definition of `split_conformal_quantile` is not
      changed."""
    mapie = pytest.importorskip("mapie")
    if not mapie.__version__.startswith("1.5"):
        pytest.skip(f"the finding was read from MAPIE 1.5.0, installed {mapie.__version__}")
    from mapie.utils import _compute_classification_quantile, _compute_regression_quantile

    rng = np.random.default_rng(3)
    regression_off, classification_up = [], []
    for n in (19, 50, 99, 100, 101, 149, 250, 1000):
        for alpha in (0.05, 0.1, 0.18, 0.2):
            scores = rng.random(n)
            srt = np.sort(scores)
            k = math.ceil((n + 1) * (1 - Fraction(str(alpha))))
            mine = split_conformal_quantile(scores, alpha)
            reg = float(_compute_regression_quantile(scores.reshape(-1, 1), np.array([1.0 - alpha]))[0, 0])
            cls = float(_compute_classification_quantile(scores.reshape(-1, 1), np.array([alpha]))[0])
            assert mine == srt[k - 1]
            pos_reg, pos_cls = int(np.searchsorted(srt, reg)) + 1, int(np.searchsorted(srt, cls)) + 1
            if pos_reg != k:
                regression_off.append((n, alpha, k, pos_reg))
            assert pos_cls in (k, k + 1), (n, alpha, k, pos_cls)  # never below the k-th, at most one above
            if pos_cls == k + 1:
                classification_up.append((n, alpha))
    assert regression_off == [(149, 0.18, 123, 124)]  # only the float trap
    assert len(classification_up) >= 5  # the classifier quantile is the (k + 1)-th order statistic in many cases


def test_mapie_split_conformal_classifier_sets_contain_ours():
    """The public `SplitConformalClassifier` (LAC score) on a stub classifier that reads the PD from the first column:
    its prediction sets contain the sets of `conformal_sets` at our quantile, and so cover at least as often."""
    mapie = pytest.importorskip("mapie")
    if not mapie.__version__.startswith("1.5"):
        pytest.skip(f"the finding was read from MAPIE 1.5.0, installed {mapie.__version__}")
    import warnings

    from mapie.classification import SplitConformalClassifier
    from sklearn.base import BaseEstimator, ClassifierMixin

    class Stub(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            p = np.asarray(X)[:, 0]
            return np.column_stack([1.0 - p, p])

        def predict(self, X):
            return (np.asarray(X)[:, 0] >= 0.5).astype(int)

    rng = np.random.default_rng(4)
    alpha, n_cal, n_test = 0.1, 99, 6000
    p = rng.beta(2.0, 5.0, n_cal + n_test)
    y = (rng.random(n_cal + n_test) < p).astype(int)
    stub = Stub().fit(p[:, None], y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SplitConformalClassifier(
            estimator=stub, confidence_level=1 - alpha, prefit=True, conformity_score="lac"
        )
        model.conformalize(p[:n_cal, None], y[:n_cal])
        _, theirs = model.predict_set(p[n_cal:, None])
    theirs = theirs[:, :, 0].astype(bool)
    scores = np.where(y[:n_cal] == 1, 1.0 - p[:n_cal], p[:n_cal])
    ours = conformal_sets(p[n_cal:], split_conformal_quantile(scores, alpha))
    assert np.all(theirs | ~ours)  # ours is inside theirs
    yt = y[n_cal:]
    assert (
        conformal_coverage(ours, yt, alpha, n_cal=n_cal).coverage
        <= conformal_coverage(theirs, yt, alpha, n_cal=n_cal).coverage
    )
