"""U16-E1 D5 (RV-1707): every English reason the governance functions can return has its Spanish pair, accented, through
``riskvalidation.governance.reasons.bilingual``; the functions' own return values are unchanged."""

from __future__ import annotations

import itertools
import re

import pytest

from riskvalidation.governance import findings, inventory, policy, reasons, scope
from riskvalidation.governance.findings import NO_CITED_ROW, NO_VALUE_IN_ROW, NOT_RWEA

from .test_governance_spanish import UNACCENTED


def _scope_reasons() -> set[str]:
    out = set()
    for bits in itertools.product((False, True), repeat=4):
        answers = scope.ScopeAnswers(*bits)
        out |= {d.rationale for d in scope.scope_record(answers)}
    return out


def _variant_reasons() -> set[str]:
    fits = {"a": ["s1", "s2"], "b": ["s2", "s3"], "c": ["s2"]}
    cases = [
        ([], {}),
        (["s1"], {"only": ["s1"]}),
        ([], fits),
        (["s1"], fits),
        (["s3", "s1"], fits),
        (["s2"], fits),
        (["zz"], fits),
    ]
    return {inventory.assign_variant(src, f)[1] for src, f in cases}


def _anchor_reasons() -> set[str]:
    rows = [{"test_id": "t.x", "model_id": "m", "statistic": None, "metric": None}]
    out = {findings.anchor_from_test_row(["design:x"], rows)[1], findings.anchor_from_test_row(["t.x@m"], rows)[1],
           findings.anchor_from_test_row(["t.y@m"], rows)[1], findings.rwea_share_of_anchor(None)[1],
           findings.rwea_share_of_anchor({"unit": "count", "value": 1.0})[1]}
    return out


def _policy_reasons() -> set[str]:
    out = {policy.NO_POLICY_FILE, policy.KIND_NOT_REGISTERED, "unresolved identity",
           "a p-value threshold has no direction"}
    default = "riskvalidation-default 2026.10.1"
    for label, test_id in ((default, "pd.binomial"), (default, "zz.none"),
                           ("contraste-c07-battery-1.0.0; alpha=0.05", "zz.none"), ("other-policy 9.9", "pd.binomial"),
                           ("x", "pd.binomial")):
        identity, reason = policy.resolve_policy(label, test_id)
        found = [reason] + [(identity or {}).get(k) for k in ("kind_reason", "direction_reason")]
        out |= {r for r in found if r}
    return out


ALL = (_scope_reasons() | _variant_reasons() | _anchor_reasons() | _policy_reasons())


def test_the_enumeration_reaches_every_kind_of_reason():
    assert len(_scope_reasons()) == 13 and {NO_CITED_ROW, NO_VALUE_IN_ROW, NOT_RWEA} <= _anchor_reasons()
    assert len(_variant_reasons()) >= 6 and len(ALL) >= 28


@pytest.mark.parametrize("english", sorted(ALL))
def test_every_english_reason_has_its_accented_spanish_pair(english):
    pair = reasons.bilingual(english)
    assert set(pair) == {"en", "es"} and pair["en"] == english
    assert pair["es"] and pair["es"] != english
    assert not UNACCENTED.search(re.sub(r"<[^>]*>", "", pair["es"])), pair["es"]  # <version> is a format token


def test_every_fixed_sentence_is_returned_by_some_function():
    """The table holds no sentence that nothing returns (the policy-string reasons with a value are templates)."""
    assert set(reasons.REASONS_ES) <= ALL | {"the single models document of the case"}


def test_values_are_carried_into_the_spanish():
    en = "no fit holds all of the variant's sources (s1, s9)"
    assert reasons.bilingual(en)["es"] == "ningún ajuste contiene todas las fuentes de la variante (s1, s9)"
    assert "pd.binomial" in reasons.bilingual(
        "the packaged policy riskvalidation-default 2026.10.1 holds no packaged threshold for pd.binomial")["es"]


def test_an_unknown_reason_is_refused_never_passed_on():
    with pytest.raises(ValueError, match="reason"):
        reasons.bilingual("some reason the engine never returns")


def test_the_functions_still_return_the_english_strings():
    assert findings.anchor_from_test_row(["design:x"], []) == (None, NO_CITED_ROW)
    assert findings.rwea_share_of_anchor(None) == (None, NOT_RWEA)
    assert inventory.assign_variant([], {}) == (None, "the case has no fit to assign the variant to")
    assert inventory.assign_variant(["x"], {"only": ["y"]}) == ("only", "the single models document of the case")
    d = scope.scope_test(scope.ScopeAnswers(True, True, False, False), "ecb")
    assert d.rationale == "In scope: it applies theory and its output is an estimate (ECB)."
    identity, reason = policy.resolve_policy("x", "pd.binomial")
    assert identity is None and isinstance(reason, str)
