"""RV-1450 to RV-1455: the group metrics of `riskvalidation.fairness.metrics` against hand-computed tables, exact
integer oracles and the engine's own `disc_auc` and `pd_ece`."""

from __future__ import annotations

import dataclasses
import random
from fractions import Fraction

import numpy as np
import pytest

from riskvalidation.fairness.metrics import (
    FourFifths,
    SelectionRate,
    adverse_impact_ratio,
    auc_by_group,
    calibration_within_groups,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    four_fifths_flag,
    group_selection_rates,
)
from riskvalidation.validation.discrimination import disc_auc


def _rates(**counts: tuple[int, int]) -> dict[str, SelectionRate]:
    """Selection rates from (favourable, n) per group, built through the public constructor path."""
    decision, group = [], []
    for g, (x, n) in counts.items():
        decision += [1] * x + [0] * (n - x)
        group += [g] * n
    return group_selection_rates(np.array(decision), np.array(group))


# ---------------------------------------------------------------------------------------------------------------
# RV-1450


def test_group_selection_rates_hand_table():
    # group a: 6 favourable of 8; group b: 2 of 5; group c: 0 of 3 (a hand-computed table)
    decision = np.array([1, 1, 1, 1, 1, 1, 0, 0] + [1, 1, 0, 0, 0] + [0, 0, 0])
    group = np.array(["a"] * 8 + ["b"] * 5 + ["c"] * 3)
    out = group_selection_rates(decision, group)
    assert list(out) == ["a", "b", "c"]
    assert (out["a"].n, out["a"].favourable, out["a"].rate) == (8, 6, 0.75)
    assert (out["b"].n, out["b"].favourable, out["b"].rate) == (5, 2, 0.4)
    assert (out["c"].n, out["c"].favourable, out["c"].rate) == (3, 0, 0.0)
    assert out["a"].group == "a"


def test_selection_rate_is_frozen_and_consistent():
    r = group_selection_rates(np.array([1, 0, 1, 1]), np.array(["x", "x", "y", "y"]))
    with pytest.raises(dataclasses.FrozenInstanceError):
        r["x"].rate = 0.9  # type: ignore[misc]
    for s in r.values():
        assert s.rate == s.favourable / s.n
    with pytest.raises(ValueError, match="rate"):
        SelectionRate("g", 10, 3, 0.5)
    with pytest.raises(ValueError, match="favourable"):
        SelectionRate("g", 10, 11, 1.1)


def test_group_selection_rates_refuses_invalid_inputs():
    with pytest.raises(ValueError, match="decision"):
        group_selection_rates(np.array([0, 2, 1]), np.array(["a", "a", "b"]))
    with pytest.raises(ValueError, match="decision"):
        group_selection_rates(np.array([0.0, np.nan, 1.0]), np.array(["a", "a", "b"]))
    with pytest.raises(ValueError, match="group"):
        group_selection_rates(np.array([0, 1, 1]), np.array(["a", "b"]))
    with pytest.raises(ValueError, match="decision"):
        group_selection_rates(np.array([], dtype=int), np.array([]))
    with pytest.raises(ValueError, match="decision"):
        group_selection_rates(np.array([[0, 1], [1, 0]]), np.array(["a", "b"]))


def test_group_selection_rates_does_not_mutate_and_omits_groups_with_no_rows():
    decision, group = np.array([1, 0, 1, 0]), np.array([1, 1, 3, 3])
    d0, g0 = decision.copy(), group.copy()
    out = group_selection_rates(decision, group)
    assert list(out) == ["1", "3"]  # a group with no rows ("2") is absent; labels are strings
    assert np.array_equal(decision, d0) and np.array_equal(group, g0)


# ---------------------------------------------------------------------------------------------------------------
# RV-1451


def test_parity_difference_and_ratio_golden():
    rates = _rates(white=(750, 1000), asian=(900, 1000), black=(650, 1000))
    diff = demographic_parity_difference(rates, "white")
    assert diff["white"] == 0.0
    assert diff["asian"] == pytest.approx(0.15, abs=1e-15)
    assert diff["black"] == pytest.approx(-0.10, abs=1e-15)
    ratio = demographic_parity_ratio(rates, "white")
    assert ratio["white"] == 1.0
    assert ratio["asian"] == pytest.approx(1.2, abs=1e-15)
    assert ratio["black"] == pytest.approx(650 / 750, abs=1e-15)
    # exact arithmetic: 55 of 100 against 50 of 100 differs by exactly the float 0.05, not by 0.05000000000000004
    exact = demographic_parity_difference(_rates(a=(55, 100), b=(50, 100)), "b")
    assert exact["a"] == 0.05


def test_parity_ratio_is_none_when_reference_rate_is_zero():
    rates = _rates(a=(5, 10), b=(0, 10))
    ratio = demographic_parity_ratio(rates, "b")
    assert ratio == {"a": None, "b": None}
    assert demographic_parity_difference(rates, "b") == {"a": 0.5, "b": 0.0}


def test_parity_refuses_unknown_reference():
    rates = _rates(a=(5, 10), b=(2, 10))
    for fn in (demographic_parity_difference, demographic_parity_ratio, adverse_impact_ratio):
        with pytest.raises(ValueError, match="reference"):
            fn(rates, "zzz")


# ---------------------------------------------------------------------------------------------------------------
# RV-1452


def _outcome_table():
    # group a: y=1 rows 8 (decision 1 on 6), y=0 rows 4 (decision 1 on 1); group b: y=1 rows 5 (decision 1 on 2),
    # y=0 rows 10 (decision 1 on 5). TPR a 0.75, b 0.4; FPR a 0.25, b 0.5.
    y = np.array([1] * 8 + [0] * 4 + [1] * 5 + [0] * 10)
    d = np.array([1] * 6 + [0] * 2 + [1] * 1 + [0] * 3 + [1] * 2 + [0] * 3 + [1] * 5 + [0] * 5)
    g = np.array(["a"] * 12 + ["b"] * 15)
    return y, d, g


def test_equalized_odds_hand_computed():
    y, d, g = _outcome_table()
    # TPR gap 0.75 - 0.4 = 0.35; FPR gap 0.5 - 0.25 = 0.25; the larger is returned
    assert equalized_odds_difference(y, d, g) == pytest.approx(0.35, abs=1e-15)
    # swap the decisions of group b on y = 0 so that the FPR gap exceeds the TPR gap
    d2 = d.copy()
    d2[17:27] = 1  # group b y = 0 rows all favourable: FPR b = 1.0, FPR gap 0.75
    assert equalized_odds_difference(y, d2, g) == pytest.approx(0.75, abs=1e-15)


def test_equal_opportunity_is_the_tpr_part():
    y, d, g = _outcome_table()
    assert equal_opportunity_difference(y, d, g) == pytest.approx(0.35, abs=1e-15)
    d2 = d.copy()
    d2[17:27] = 1  # only the false positive rate of b changes
    assert equal_opportunity_difference(y, d2, g) == pytest.approx(0.35, abs=1e-15)
    assert equalized_odds_difference(y, d2, g) > equal_opportunity_difference(y, d2, g)


def test_outcome_metrics_refuse_missing_classes_and_no_outcome():
    y, d, g = _outcome_table()
    only_pos_in_a = y.copy()
    only_pos_in_a[8:12] = 1  # group a has no negatives
    with pytest.raises(ValueError, match="both classes"):
        equalized_odds_difference(only_pos_in_a, d, g)
    # equal opportunity only needs positives: a group without negatives is fine, a group without positives is not
    assert equal_opportunity_difference(only_pos_in_a, d, g) >= 0.0
    no_pos_in_b = y.copy()
    no_pos_in_b[12:17] = 0
    with pytest.raises(ValueError, match="positives"):
        equal_opportunity_difference(no_pos_in_b, d, g)
    for fn in (equalized_odds_difference, equal_opportunity_difference):
        with pytest.raises(ValueError, match="outcome"):
            fn(None, d, g)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two groups"):
        equalized_odds_difference(y, d, np.array(["a"] * len(y)))
    with pytest.raises(ValueError, match="y"):
        equalized_odds_difference(np.where(y == 1, 2, 0), d, g)


# ---------------------------------------------------------------------------------------------------------------
# RV-1453


def test_adverse_impact_ratio_is_the_parity_ratio():
    for counts, ref in (
        (dict(a=(70, 100), b=(40, 100), c=(55, 100)), "a"),
        (dict(a=(5, 10), b=(0, 10)), "b"),
        (dict(a=(1, 3), b=(2, 3)), "a"),
    ):
        rates = _rates(**counts)
        assert adverse_impact_ratio(rates, ref) == demographic_parity_ratio(rates, ref)


# ---------------------------------------------------------------------------------------------------------------
# RV-1454


def test_four_fifths_boundary_goldens():
    # 60/100 against 75/100 is exactly four-fifths: the float ratio is 0.7999999999999999 and must not flag
    assert 0.6 / 0.75 < 0.8
    exactly = four_fifths_flag(_rates(h=(75, 100), g=(60, 100)))
    assert exactly.flag == {"h": False, "g": False}
    assert exactly.ratio_to_highest["g"] == pytest.approx(0.8, abs=1e-12)
    assert four_fifths_flag(_rates(h=(75, 100), g=(5999, 10000))).flag["g"] is True
    assert four_fifths_flag(_rates(h=(75, 100), g=(6001, 10000))).flag["g"] is False
    assert four_fifths_flag(_rates(h=(75, 100), g=(6000, 10000))).flag["g"] is False


def test_four_fifths_scenario_against_the_highest_rate_group():
    rates = _rates(white=(750, 1000), asian=(900, 1000), black=(650, 1000))
    ff = four_fifths_flag(rates)
    assert ff.highest_rate_group == "asian"
    assert ff.flag == {"white": False, "asian": False, "black": True}  # 650/900 = 0.722 < 0.8
    assert ff.ratio_to_highest["black"] == pytest.approx(0.7222, abs=1e-4)
    assert ff.ratio_to_highest["asian"] == 1.0
    # against the reference group (white) black would not be flagged: 0.867
    assert adverse_impact_ratio(rates, "white")["black"] == pytest.approx(0.8667, abs=1e-4)
    assert adverse_impact_ratio(rates, "white")["black"] >= 0.8


def test_four_fifths_agrees_with_a_fraction_oracle():
    rng = random.Random(20261019)
    for _ in range(300):
        k = rng.randint(2, 4)
        counts = {}
        for i in range(k):
            n = rng.randint(1, 60)
            counts[f"g{i}"] = (rng.randint(0, n), n)
        rates = _rates(**counts)
        ff = four_fifths_flag(rates)
        best = max(counts.values(), key=lambda c: Fraction(c[0], c[1]))
        top_rate = Fraction(*best)
        first = next(g for g, c in counts.items() if Fraction(c[0], c[1]) == top_rate)
        assert ff.highest_rate_group == first
        for g, (x, n) in counts.items():
            expect = top_rate > 0 and g != first and Fraction(x, n) < Fraction(4, 5) * top_rate
            assert ff.flag[g] is bool(expect), (counts, g)


def test_four_fifths_zero_highest_rate_and_labels():
    ff = four_fifths_flag(_rates(a=(0, 10), b=(0, 20)))
    assert ff.flag == {"a": False, "b": False}
    assert ff.ratio_to_highest == {"a": None, "b": None}
    assert ff.highest_rate_group == "a"
    ff = four_fifths_flag(_rates(a=(3, 10), b=(6, 10)))
    assert isinstance(ff, FourFifths)
    assert ff.threshold == 0.8
    assert ff.label == "employment benchmark"
    assert ff.is_credit_law_threshold is False
    assert ff.compared_with == "highest-rate-group"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ff.threshold = 0.5  # type: ignore[misc]
    # the highest-rate group has ratio 1 and no flag; a tie keeps the first group
    tie = four_fifths_flag(_rates(a=(5, 10), b=(10, 20)))
    assert tie.highest_rate_group == "a" and tie.ratio_to_highest["b"] == 1.0 and tie.flag == {"a": False, "b": False}


# ---------------------------------------------------------------------------------------------------------------
# RV-1455


def test_calibration_within_groups_calibrated_and_planted():
    rng = np.random.default_rng(7)
    n = 20000
    p = rng.uniform(0.05, 0.95, n)
    g = np.where(rng.random(n) < 0.5, "a", "b")
    y = (rng.random(n) < p).astype(int)
    ece = calibration_within_groups(p, y, g)
    assert set(ece) == {"a", "b"}
    assert all(0.0 <= v < 0.02 for v in ece.values())  # within Monte Carlo error of zero
    # plant a miscalibration in group b only: its outcomes occur at p + 0.15
    y_bad = np.where(g == "b", (rng.random(n) < np.clip(p + 0.15, 0, 1)).astype(int), y)
    bad = calibration_within_groups(p, y_bad, g)
    assert bad["b"] > 0.10 and bad["a"] < 0.02


def test_calibration_small_groups_are_omitted():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.1, 0.9, 205)
    y = (rng.random(205) < p).astype(int)
    g = np.array(["big"] * 200 + ["tiny"] * 5)
    assert list(calibration_within_groups(p, y, g, n_bins=10)) == ["big"]  # tiny has 5 rows < 10 bins
    assert set(calibration_within_groups(p, y, g, n_bins=5)) == {"big", "tiny"}
    with pytest.raises(ValueError, match="outcome"):
        calibration_within_groups(p, None, g)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_bins"):
        calibration_within_groups(p, y, g, n_bins=0)


def test_auc_by_group_equals_disc_auc():
    rng = np.random.default_rng(11)
    n = 600
    g = np.where(np.arange(n) % 3 == 0, "x", "y")
    y = (rng.random(n) < 0.4).astype(int)
    score = rng.normal(size=n) + 0.8 * y
    out = auc_by_group(score, y, g)
    for grp in ("x", "y"):
        m = g == grp
        assert out[grp] == disc_auc(score[m], y[m]).metric
        assert 0.0 <= out[grp] <= 1.0


def test_auc_by_group_single_class_groups_are_omitted():
    score = np.array([0.1, 0.9, 0.4, 0.6, 0.5, 0.7])
    y = np.array([0, 1, 0, 1, 1, 1])
    g = np.array(["ok", "ok", "ok", "ok", "one", "one"])  # group "one" has only y = 1
    assert list(auc_by_group(score, y, g)) == ["ok"]
    with pytest.raises(ValueError, match="outcome"):
        auc_by_group(score, None, g)  # type: ignore[arg-type]
