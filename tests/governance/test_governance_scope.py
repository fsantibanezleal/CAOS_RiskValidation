"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the B.1 scope test
(contract `engine-governance-api` 1.0.0, module scope).
"""

from __future__ import annotations

import dataclasses

import pytest

from riskvalidation.governance.scope import ScopeAnswers, ScopeDecision, scope_record, scope_test

PROFILES = ("us-2026", "pra", "osfi-2027", "ecb")


def test_scope_goldens():
    assert scope_test(ScopeAnswers(True, True, False, False), "us-2026").in_scope is True
    assert scope_test(ScopeAnswers(True, True, False, True), "us-2026").in_scope is False
    assert scope_test(ScopeAnswers(False, True, True, False), "us-2026").in_scope is False
    assert scope_test(ScopeAnswers(True, False, False, False), "pra").in_scope is True
    assert scope_test(ScopeAnswers(True, False, False, False), "ecb").in_scope is False


def test_scope_rules_over_the_whole_truth_table():
    for a in (True, False):
        for b in (True, False):
            for c in (True, False):
                for d in (True, False):
                    ans = ScopeAnswers(a, b, c, d)
                    assert scope_test(ans, "us-2026").in_scope == (a and b and not d)
                    assert scope_test(ans, "pra").in_scope == a
                    assert scope_test(ans, "osfi-2027").in_scope == (a and b)
                    assert scope_test(ans, "ecb").in_scope == (a and b)


def test_scope_record_is_four_decisions_in_order():
    rec = scope_record(ScopeAnswers(True, True, False, False))
    assert [d.profile for d in rec] == list(PROFILES)
    assert all(d.in_scope for d in rec)
    gen = scope_record(ScopeAnswers(True, True, False, True))
    assert [d.in_scope for d in gen] == [False, True, True, True]
    assert [d.profile for d in gen] == list(PROFILES)
    assert gen[1:] == scope_record(ScopeAnswers(True, True, False, False))[1:]


def test_scope_rationale_names_the_answers_that_decided_it():
    out = scope_test(ScopeAnswers(True, True, False, True), "us-2026")
    assert "generative" in out.rationale and "SR 26-2" in out.rationale
    none = scope_test(ScopeAnswers(False, True, True, False), "us-2026")
    assert "theory" in none.rationale
    est = scope_test(ScopeAnswers(True, False, False, False), "ecb")
    assert "estimate" in est.rationale
    for d in scope_record(ScopeAnswers(True, True, False, False)):
        assert d.rationale.strip()


def test_scope_values_are_immutable_and_unknown_profile_raises():
    ans = ScopeAnswers(True, True, False, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ans.applies_theory = False  # type: ignore[misc]
    dec = scope_test(ans, "pra")
    assert isinstance(dec, ScopeDecision)
    with pytest.raises(dataclasses.FrozenInstanceError):
        dec.in_scope = False  # type: ignore[misc]
    with pytest.raises(ValueError):
        scope_test(ans, "basel")
