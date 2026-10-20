"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the tiering rubric
(contract `engine-governance-api` 1.0.0, module tiering; dossier 06 B.3 l.553-574).
"""

from __future__ import annotations

import itertools

import pytest

from riskvalidation.governance.tiering import (
    EXPOSURE_BAND_CUTS,
    apply_override,
    complexity,
    exposure_band,
    materiality,
    matrix_tier,
    tier_counts,
    tier_sensitivity,
)

# The 19 built cases of the U15 research addendum D2 (3.2 materiality inputs, 3.3 complexity scores):
# case -> (share of the stated balance, regulatory use literal, regulatory use graded, decision criticality,
# six factor scores).
CASES = {
    "C01": (0.0531, 4, 3, 3, [4, 2, 3, 1, 1, 1]),
    "C03": (0.2162, 4, 3, 1, [3, 2, 3, 1, 1, 1]),
    "C04": (0.3243, 4, 3, 1, [2, 2, 1, 1, 1, 1]),
    "C05": (0.3243, 4, 3, 1, [2, 2, 1, 1, 1, 1]),
    "C06": (1.0, 4, 4, 1, [2, 2, 1, 1, 1, 1]),
    "C07": (1.0, 4, 4, 1, [2, 1, 1, 1, 1, 1]),
    "C08": (0.2162, 4, 3, 1, [3, 2, 3, 1, 1, 1]),
    "C09": (0.0531, 4, 3, 1, [1, 1, 1, 1, 1, 1]),
    "C10": (0.7084, 4, 3, 1, [2, 2, 1, 1, 1, 1]),
    "C11": (0.1713, 4, 3, 1, [2, 2, 1, 4, 1, 1]),
    "C12": (None, 4, 4, 1, [2, 2, 1, 1, 1, 1]),
    "C13": (0.1713, 4, 3, 1, [2, 1, 1, 1, 1, 1]),
    "C14": (1.0, 4, 4, 1, [2, 2, 1, 1, 1, 1]),
    "C15": (1.0, 4, 4, 1, [2, 2, 1, 1, 2, 1]),
    "C16": (1.0, 4, 4, 1, [1, 2, 1, 1, 1, 1]),
    "C17": (0.1203, 4, 4, 1, [2, 1, 1, 1, 1, 1]),
    "C18": (1.0, 4, 4, 1, [2, 2, 1, 1, 4, 1]),
    "C21": (0.3456, 2, 2, 3, [2, 2, 1, 1, 1, 1]),
    "C22": (None, 1, 1, 1, [2, 1, 1, 1, 1, 1]),
}
# The contract's golden table of the tiers A, B, C, D per case and its counts per reading.
GOLDEN_TIERS = {
    "C01": (1, 2, 1, 1),
    "C03": (1, 2, 1, 1),
    "C04": (1, 1, 1, 1),
    "C05": (1, 1, 1, 1),
    "C06": (1, 1, 1, 1),
    "C07": (1, 1, 1, 1),
    "C08": (1, 2, 1, 1),
    "C09": (1, 2, 1, 2),
    "C10": (1, 1, 1, 1),
    "C11": (1, 2, 1, 1),
    "C12": (1, 1, 1, 1),
    "C13": (1, 2, 1, 2),
    "C14": (1, 1, 1, 1),
    "C15": (1, 1, 1, 1),
    "C16": (1, 1, 1, 1),
    "C17": (1, 1, 1, 1),
    "C18": (1, 1, 1, 1),
    "C21": (1, 1, 1, 1),
    "C22": (4, 4, 4, 4),
}
GOLDEN_COUNTS = {"A": (18, 0, 0, 1), "B": (12, 6, 0, 1), "C": (18, 0, 0, 1), "D": (16, 2, 0, 1)}


def test_exposure_band_goldens():
    assert EXPOSURE_BAND_CUTS == (0.25, 0.10, 0.02)
    golden = {1.0: 4, 0.25: 4, 0.2499: 3, 0.1: 3, 0.0999: 2, 0.02: 2, 0.0199: 1, 0.0: 1, None: 1}
    for share, band in golden.items():
        assert exposure_band(share) == band, share


def test_exposure_band_is_monotone_and_takes_other_cuts():
    last = 0
    for i in range(0, 1001):
        band = exposure_band(i / 1000)
        assert band >= last
        last = band
    assert exposure_band(0.5, (0.6, 0.4, 0.2)) == 3
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            exposure_band(bad)
    for cuts in ((0.1, 0.25, 0.02), (0.25, 0.25, 0.02), (0.25, 0.1, 0.0), (1.5, 0.1, 0.02), (0.25, 0.1)):
        with pytest.raises(ValueError):
            exposure_band(0.5, cuts)  # type: ignore[arg-type]


def test_materiality_goldens_and_range():
    assert materiality(2, 3, 3) == 3  # C01, graded
    assert materiality(2, 4, 3) == 4  # C01, literal
    assert materiality(1, 1, 1) == 1  # C22
    for triple in itertools.product(range(1, 5), repeat=3):
        assert materiality(*triple) == max(triple)
    for bad in ((0, 1, 1), (1, 5, 1), (1, 1, 0), (1, 1, 5)):
        with pytest.raises(ValueError):
            materiality(*bad)


def test_complexity_goldens():
    assert complexity([4, 2, 3, 1, 1, 1]) == 2  # mean 2.00 (C01)
    assert complexity([2, 2, 1, 1, 2, 1]) == 2  # mean 1.50 rounds up (C15)
    assert complexity([2, 2, 1, 1, 1, 1]) == 1  # mean 1.33
    assert complexity([3, 3, 3, 2, 2, 2]) == 3  # mean 2.50 rounds up
    assert complexity([2, 1, 1, 1, 1, 1]) == 1
    assert complexity([4, 2, 3, 1, 1, 1], aggregation="maximum") == 4


def test_complexity_mean_is_rounded_half_up_in_integer_arithmetic():
    for n in range(1, 8):
        for scores in itertools.product(range(1, 5), repeat=min(n, 4)):
            s, k = sum(scores), len(scores)
            assert complexity(scores) == (2 * s + k) // (2 * k)
    assert complexity([1, 2]) == 2 and complexity([1, 1, 2, 2]) == 2 and complexity([1, 1, 1, 2]) == 1  # 1.5, 1.5, 1.25
    assert complexity([4]) == 4


def test_complexity_refusals():
    for bad in ([], [0, 1], [1, 5]):
        with pytest.raises(ValueError):
            complexity(bad)
    with pytest.raises(ValueError):
        complexity([1, 2], aggregation="median")
    with pytest.raises(ValueError):
        complexity([1, 2], rounding="half-even")


def test_matrix_tier_whole_matrix():
    golden = {4: (1, 1, 1, 1), 3: (2, 2, 1, 1), 2: (3, 3, 2, 2), 1: (4, 4, 3, 3)}
    for m, row in golden.items():
        for c, tier in zip(range(1, 5), row, strict=True):
            assert matrix_tier(m, c) == tier, (m, c)


def test_matrix_tier_never_worsens_and_refuses_out_of_range():
    for m in range(1, 5):
        for c in range(1, 5):
            if m < 4:
                assert matrix_tier(m + 1, c) <= matrix_tier(m, c)
            if c < 4:
                assert matrix_tier(m, c + 1) <= matrix_tier(m, c)
    for bad in ((0, 1), (5, 1), (1, 0), (1, 5)):
        with pytest.raises(ValueError):
            matrix_tier(*bad)


def test_apply_override_goldens_and_refusals():
    assert apply_override(2, None) == 2
    assert apply_override(2) == 2
    assert apply_override(2, (1, "the model feeds the stress test", "Chief Risk Officer")) == 1
    for bad in ((1, "", "Chief Risk Officer"), (1, "a rationale", ""), (1, "  ", "x"), (1, "x", "  ")):
        with pytest.raises(ValueError, match="an override needs a rationale and an approver"):
            apply_override(2, bad)
    with pytest.raises(ValueError):
        apply_override(5, None)
    with pytest.raises(ValueError):
        apply_override(2, (5, "r", "a"))


def _sens(case: str):
    share, lit, grd, dec, scores = CASES[case]
    return tier_sensitivity(share, lit, grd, dec, scores)


def test_tier_sensitivity_golden_table_of_the_nineteen_built_cases():
    for case, golden in GOLDEN_TIERS.items():
        rows = _sens(case)
        assert tuple(r[5] for r in rows) == golden, case


def test_tier_sensitivity_counts_per_reading():
    counts = {r: [0, 0, 0, 0] for r in "ABCD"}
    for case in CASES:
        for row in _sens(case):
            counts[row[0]][row[5] - 1] += 1
    assert {k: tuple(v) for k, v in counts.items()} == GOLDEN_COUNTS
    assert tier_counts([_sens(c)[1][5] for c in CASES]) == GOLDEN_COUNTS["B"]


def test_tier_sensitivity_rows_and_invariants():
    rows = tier_sensitivity(0.0531, 4, 3, 3, [4, 2, 3, 1, 1, 1])
    assert [r[0] for r in rows] == ["A", "B", "C", "D"]
    assert [(r[1], r[2]) for r in rows] == [
        ("literal", "mean"),
        ("graded", "mean"),
        ("literal", "maximum"),
        ("graded", "maximum"),
    ]
    assert [(r[3], r[4], r[5]) for r in rows] == [(4, 2, 1), (3, 2, 2), (4, 4, 1), (3, 4, 1)]
    for case in CASES:
        for _, _, _, m, c, tier in _sens(case):
            assert tier == matrix_tier(m, c)
        a, b = _sens(case)[0], _sens(case)[1]
        assert a[5] <= b[5]  # reading A never gives a higher tier number than reading B
    with pytest.raises(ValueError):
        tier_sensitivity(0.5, 2, 3, 1, [1, 2])  # literal below graded


def test_tier_sensitivity_leaves_unscored_factors_out():
    six = tier_sensitivity(1.0, 4, 4, 1, [2, 2, 1, 1, 1, 1])
    assert six == tier_sensitivity(1.0, 4, 4, 1, [2, 2, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        tier_sensitivity(1.0, 4, 4, 1, [])
    with pytest.raises(ValueError):
        tier_sensitivity(1.0, 4, 4, 1, [2, 0, 1])  # an unscored factor is left out of the list, never counted as 0 or 1


def test_tier_counts():
    assert tier_counts([1, 2, 1, 4]) == (2, 1, 0, 1)
    assert tier_counts([]) == (0, 0, 0, 0)
    for tiers in ([1, 1, 3], [4, 4, 4, 2, 2]):
        assert sum(tier_counts(tiers)) == len(tiers)
    for bad in ([0], [5], [1, 7]):
        with pytest.raises(ValueError):
            tier_counts(bad)
