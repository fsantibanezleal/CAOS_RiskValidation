"""RV-562 to RV-566: Banco de Espana Circular 4/2017 Annex 9 (provisions.local_es), every transcribed cell asserted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from riskvalidation.provisions import local_es as es

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "provisions_local" / "annex9_tables.json")
                     .read_text(encoding="utf-8"))
# Month bounds of point 140 at 30.4375 days per month rounded to the nearest day (the contract's convention).
UPPER_DAYS = [round(m * 30.4375) for m in (6, 9, 12, 15, 18, 21)]
SEGMENTS = list(FIXTURE["point_144"])


def test_month_bounds_are_the_contract_buckets():
    assert UPPER_DAYS == [183, 274, 365, 457, 548, 639]


def test_uncovered_amount():
    assert es.annex9_uncovered_amount(100.0, [30.0, 20.0]) == 50.0
    assert es.annex9_uncovered_amount(100.0, [120.0]) == 0.0
    assert es.annex9_uncovered_amount(100.0, []) == 100.0
    assert es.annex9_uncovered_amount(0.0, []) == 0.0
    prev = 100.0
    for g in range(0, 150, 10):
        u = es.annex9_uncovered_amount(100.0, [float(g)])
        assert 0.0 <= u <= 100.0 and u <= prev
        prev = u
    with pytest.raises(ValueError):
        es.annex9_uncovered_amount(-1.0, [])
    with pytest.raises(ValueError):
        es.annex9_uncovered_amount(100.0, [-1.0])


def test_point_144_every_cell():
    assert len(SEGMENTS) == 12
    n = 0
    for seg, (normal, watch) in FIXTURE["point_144"].items():
        assert es.annex9_normal_coverage(seg, "normal") == pytest.approx(normal / 100, abs=1e-12)
        assert es.annex9_normal_coverage(seg, "normal-special-watch") == pytest.approx(watch / 100, abs=1e-12)
        assert 0 < normal < watch < 100
        n += 2
    assert n == 24
    assert es.annex9_normal_coverage("B.1.1", "normal") == 0.007
    assert es.annex9_normal_coverage("B.1.1", "normal-special-watch") == 0.18
    assert es.annex9_normal_coverage("A.1.1", "normal") == 0.019
    assert es.annex9_normal_coverage("A.1.1", "normal-special-watch") == 0.30
    assert es.annex9_normal_coverage("B.2", "normal-special-watch") == 0.202
    assert es.annex9_normal_coverage("B.2.1", "normal") == 0.010
    with pytest.raises(ValueError, match="point 144"):
        es.annex9_normal_coverage("Z.9", "normal")
    with pytest.raises(ValueError, match="point 144"):
        es.annex9_normal_coverage("B.1.1", "doubtful")


def test_point_140_every_cell_and_bounds():
    assert len(FIXTURE["point_140"]) == 12
    n = 0
    for seg, row in FIXTURE["point_140"].items():
        assert len(row) == 7 and row == sorted(row) and row[-1] == 100
        lowers = [91] + [u + 1 for u in UPPER_DAYS]
        uppers = UPPER_DAYS + [5000]
        for k in range(7):
            expected = row[k] / 100
            for days in (lowers[k], uppers[k], (lowers[k] + uppers[k]) // 2):  # both inclusive ends and a middle
                assert es.annex9_doubtful_coverage(seg, days) == pytest.approx(expected, abs=1e-12), (seg, k, days)
            n += 1
    assert n == 84
    # monotone in age, 1.0 above 639 days for every segment
    for seg in SEGMENTS:
        values = [es.annex9_doubtful_coverage(seg, d) for d in range(91, 800)]
        assert values == sorted(values)
        assert es.annex9_doubtful_coverage(seg, 640) == 1.0
    assert es.annex9_doubtful_coverage("B.1.1", 150) == 0.45
    assert es.annex9_doubtful_coverage("B.1.1", 91) == 0.45
    assert es.annex9_doubtful_coverage("B.1.1", 670) == 1.0
    assert es.annex9_doubtful_coverage("B.2", 243) == 0.65
    assert es.annex9_doubtful_coverage("A.2.3", 122) == 0.35
    # boundary: 183 days is in the first column, 184 in the second
    assert es.annex9_doubtful_coverage("A.1.1", 183) == 0.70 and es.annex9_doubtful_coverage("A.1.1", 184) == 0.75
    assert es.annex9_doubtful_coverage("A.1.1", 639) == 1.0 and es.annex9_doubtful_coverage("A.2.3", 639) == 0.95
    for days in (90, 0, -5):
        with pytest.raises(ValueError, match="point 140"):
            es.annex9_doubtful_coverage("B.1.1", days)
    with pytest.raises(ValueError, match="point 140"):
        es.annex9_doubtful_coverage("Z.9", 120)


def test_category_to_stage():
    for cat, stage in FIXTURE["stage_map"].items():
        assert es.annex9_category_to_stage(cat) == stage
    assert es.annex9_category_to_stage("normal") == "stage1"
    assert es.annex9_category_to_stage("normal-special-watch") == "stage2"
    assert es.annex9_category_to_stage("doubtful") == "stage3"
    assert es.annex9_category_to_stage("written-off") == "written-off"
    assert len({es.annex9_category_to_stage(c) for c in FIXTURE["stage_map"]}) == 4  # injective
    for bad in ("stage1", "Normal", "", "substandard"):
        with pytest.raises(ValueError, match="norma 29"):
            es.annex9_category_to_stage(bad)


def test_annex9_coverage():
    assert es.annex9_coverage("B.1.1", "normal", 100000.0, [40000.0]) == pytest.approx(420.0, abs=1e-9)
    assert es.annex9_coverage("B.1.1", "normal-special-watch", 1000.0, []) == pytest.approx(180.0, abs=1e-9)
    assert es.annex9_coverage("B.1.1", "doubtful", 1000.0, [100.0], days_past_due=150) == pytest.approx(405.0, abs=1e-9)
    assert es.annex9_coverage("B.2", "doubtful", 1000.0, [], days_past_due=700) == pytest.approx(1000.0)
    assert es.annex9_coverage("B.1.1", "written-off", 1000.0, []) == 0.0
    assert es.annex9_coverage("B.1.1", "normal", 1000.0, [1000.0]) == 0.0  # full guarantee cover
    assert es.annex9_coverage("B.1.1", "normal", 0.0, []) == 0.0  # empty
    prev = 0.0
    for ead in range(0, 2000, 100):
        c = es.annex9_coverage("A.2.2", "normal-special-watch", float(ead), [300.0])
        assert 0.0 <= c <= ead and c >= prev
        prev = c
    # a doubtful account without arrears evidence has no transcribed coverage: raises, the caller reports not evaluated
    for days in (None, 90, 0):
        with pytest.raises(ValueError, match="not evaluated|arrears"):
            es.annex9_coverage("B.1.1", "doubtful", 1000.0, [], days_past_due=days)
    with pytest.raises(ValueError):
        es.annex9_coverage("B.1.1", "unknown", 1000.0, [])
    with pytest.raises(ValueError):
        es.annex9_coverage("B.1.1", "normal", -1.0, [])
