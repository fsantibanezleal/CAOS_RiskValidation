"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the B.8 dates and the
opinion (contract `engine-governance-api` 1.0.0, modules schedule and opinion; dossier 06 B.8 l.630-640 and
l.620-623; ECB Instructions of February 2019, section 2.4.2 for the 5 and 10 percent cut points)."""

from __future__ import annotations

import calendar
from datetime import date

import pytest

from riskvalidation.governance.opinion import finding_level, opinion_level, recommendation
from riskvalidation.governance.schedule import FREQUENCY, add_months, next_dates

LABEL = "the rule's output on the evidence; Contraste approves no model"


# ---- schedule


def test_add_months_goldens():
    assert add_months(date(2026, 8, 31), 6) == date(2027, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2026, 10, 12), 12) == date(2027, 10, 12)
    assert add_months(date(2026, 10, 12), 0) == date(2026, 10, 12)


def test_add_months_clamps_rolls_over_the_year_and_refuses_negative():
    assert add_months(date(2026, 11, 30), 3) == date(2027, 2, 28)
    assert add_months(date(2026, 12, 15), 1) == date(2027, 1, 15)
    assert add_months(date(2026, 1, 31), 25) == date(2028, 2, 29)
    assert add_months(date(2026, 3, 31), 1) == date(2026, 4, 30)
    for d in (date(2026, 1, 31), date(2026, 5, 30), date(2024, 2, 29)):
        for n in range(0, 40):
            out = add_months(d, n)
            assert out.day == min(d.day, calendar.monthrange(out.year, out.month)[1])
            assert (out.year * 12 + out.month) - (d.year * 12 + d.month) == n
    with pytest.raises(ValueError):
        add_months(date(2026, 10, 12), -1)


def test_frequency_table_is_the_stated_convention():
    assert FREQUENCY == {1: (12, 36, 3), 2: (12, 36, 6), 3: (24, None, 12), 4: (36, None, None)}


def test_next_dates_goldens():
    last = date(2026, 10, 12)
    assert next_dates(1, last) == (date(2027, 10, 12), date(2029, 10, 12), date(2027, 1, 12))
    assert next_dates(2, last) == (date(2027, 10, 12), date(2029, 10, 12), date(2027, 4, 12))
    assert next_dates(3, last) == (date(2028, 10, 12), None, date(2027, 10, 12))
    assert next_dates(4, last) == (date(2029, 10, 12), None, None)


def test_next_dates_ecb_profile_applies_the_minimum_even_where_the_table_states_none():
    last = date(2026, 10, 12)
    assert next_dates(1, last, profile="ecb") == next_dates(1, last)
    assert next_dates(2, last, profile="ecb") == next_dates(2, last)
    assert next_dates(3, last, profile="ecb") == (date(2027, 10, 12), date(2029, 10, 12), date(2027, 10, 12))
    assert next_dates(4, last, profile="ecb") == (date(2027, 10, 12), date(2029, 10, 12), None)
    for tier in range(1, 5):
        review, reval, _ = next_dates(tier, last, profile="ecb")
        assert review <= add_months(last, 12) and reval is not None and reval <= add_months(last, 36)


def test_next_dates_takes_another_table_and_refuses_unknown_tier_or_profile():
    table = {1: (6, 12, 1), 2: (12, None, None), 3: (12, None, None), 4: (12, None, None)}
    assert next_dates(1, date(2026, 1, 31), table) == (date(2026, 7, 31), date(2027, 1, 31), date(2026, 2, 28))
    for tier in (0, 5):
        with pytest.raises(ValueError):
            next_dates(tier, date(2026, 1, 1))
    with pytest.raises(ValueError):
        next_dates(1, date(2026, 1, 1), profile="pra")


# ---- opinion


def test_finding_level_goldens():
    assert (
        finding_level("S1") == 4 and finding_level("S2") == 3 and finding_level("S3") == 2 and finding_level("S4") == 2
    )
    assert finding_level("S4", rwea_share=0.0499) == 2
    assert finding_level("S4", rwea_share=0.05) == 3
    assert finding_level("S3", rwea_share=0.0999) == 3
    assert finding_level("S3", rwea_share=0.10) == 4
    assert finding_level("S1", rwea_share=0.0499) == 4
    assert finding_level("S2", rwea_share=0.10) == 4


def test_an_anchor_can_only_raise_a_level():
    for sev in ("S1", "S2", "S3", "S4"):
        base = finding_level(sev)
        for share in (None, 0.0, 0.0499, 0.05, 0.0999, 0.10, 0.5, 1.0):
            assert finding_level(sev, share) >= base, (sev, share)
    assert finding_level("S1", 0.0) == 4 and finding_level("S1", 0.01) == 4  # a critical finding is never downgraded


def test_finding_level_refusals():
    with pytest.raises(ValueError):
        finding_level("S1", rwea_share=-0.01)
    with pytest.raises(ValueError):
        finding_level("S5")


def test_opinion_level_goldens():
    assert opinion_level([]) == 1
    assert opinion_level([("S4", None), ("S3", None)]) == 2
    assert opinion_level([("S3", None), ("S2", None)]) == 3
    assert opinion_level([("S2", None), ("S1", None)]) == 4
    assert opinion_level([("S3", 0.0499), ("S4", None)]) == 2
    assert opinion_level([("S3", 0.05)]) == 3
    assert opinion_level([("S4", 0.10)]) == 4
    assert opinion_level([("S1", 0.01)]) == 4


def test_opinion_level_is_the_maximum_of_the_findings_levels():
    pool = [("S1", None), ("S2", 0.06), ("S3", 0.11), ("S4", 0.01), ("S4", None)]
    for k in range(1, len(pool) + 1):
        sub = pool[:k]
        assert opinion_level(sub) == max(finding_level(s, r) for s, r in sub)
    assert opinion_level([("S4", None)] * 100) == 2


def test_recommendation_goldens():
    assert recommendation(1)[0] == "approve"
    assert recommendation(2)[0] == "approve with conditions"
    assert recommendation(3)[0] == "remediate before use" and recommendation(4)[0] == "remediate before use"
    for level in range(1, 5):
        outcome, text, label = recommendation(level)
        assert label == LABEL
        assert text.strip() and "Contraste approves no model" in text
    for bad in (0, 5):
        with pytest.raises(ValueError):
            recommendation(bad)
