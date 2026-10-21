"""U16-E1 D5 (RV-1707): every reason the governance functions can return has a stable code and its Spanish pair,
accented, through ``riskvalidation.governance.reasons``; the functions' own return values are unchanged."""

from __future__ import annotations

import itertools
import re

import pytest

from riskvalidation.core.policy import packaged_labels, packaged_policy
from riskvalidation.governance import findings, inventory, policy, reasons, scope
from riskvalidation.governance.findings import NO_CITED_ROW, NO_VALUE_IN_ROW, NOT_RWEA

from .test_governance_spanish import UNACCENTED


def _scope_reasons() -> set[str]:
    out = set()
    for bits in itertools.product((False, True), repeat=4):
        out |= {d.rationale for d in scope.scope_record(scope.ScopeAnswers(*bits))}
    return out


def _variant_reasons() -> set[str]:
    fits = {"a": ["s1", "s2"], "b": ["s2", "s3"], "c": ["s2"]}
    cases = [([], {}), (["s1"], {"only": ["s1"]}), ([], fits), (["s1"], fits), (["s3", "s1"], fits), (["s2"], fits),
             (["zz"], fits)]
    return {inventory.assign_variant(src, f)[1] for src, f in cases}


def _anchor_reasons() -> set[str]:
    rows = [{"test_id": "t.x", "model_id": "m", "statistic": None, "metric": None}]
    return {findings.anchor_from_test_row(["design:x"], rows)[1], findings.anchor_from_test_row(["t.x@m"], rows)[1],
            findings.anchor_from_test_row(["t.y@m"], rows)[1], findings.rwea_share_of_anchor(None)[1],
            findings.rwea_share_of_anchor({"unit": "count", "value": 1.0})[1]}


def _policy_reasons() -> set[str]:
    out = {policy.NO_POLICY_FILE, policy.KIND_NOT_REGISTERED, "unresolved identity",
           "a p-value threshold has no direction"}
    default = "riskvalidation-default 2026.10.1"
    cases = [(default, "pd.binomial"), (default, "zz.none"), ("contraste-c07-battery-1.0.0; alpha=0.05", "zz.none"),
             ("other-policy 9.9", "pd.binomial"), ("x", "pd.binomial")]
    for label in packaged_labels():  # every packaged policy, every test id it holds, plus an id it does not hold
        cases += [(label, test_id) for test_id in (*packaged_policy(label).thresholds, "zz.none")]
    cases += [("contraste-market-battery 1.0.0+custom", "market.kupiec_pof")]  # a customised row without its override
    for label, test_id in cases:
        identity, reason = policy.resolve_policy(label, test_id)
        found = [reason] + [(identity or {}).get(k) for k in ("kind_reason", "direction_reason")]
        out |= {r for r in found if r}
    return out


ALL = _scope_reasons() | _variant_reasons() | _anchor_reasons() | _policy_reasons()


def test_the_enumeration_reaches_every_kind_of_reason():
    assert len(_scope_reasons()) == 13 and {NO_CITED_ROW, NO_VALUE_IN_ROW, NOT_RWEA} <= _anchor_reasons()
    assert len(_variant_reasons()) >= 6 and len(ALL) >= 28
    assert any(r.startswith("the light of ") for r in ALL), "the packaged-rule reason of the battery policies"


@pytest.mark.parametrize("english", sorted(ALL))
def test_every_english_reason_has_a_code_and_its_accented_spanish_pair(english):
    code, values = reasons.code_of(english)
    assert code in reasons.REASONS
    pair = reasons.bilingual(english)
    assert pair == reasons.bilingual(code, **values), "the code and the sentence give the same pair"
    assert set(pair) == {"en", "es"} and pair["en"] == english
    assert pair["es"] and pair["es"] != english
    assert not UNACCENTED.search(re.sub(r"<[^>]*>", "", pair["es"])), pair["es"]  # <version> is a format token


def test_every_code_is_reached_by_some_function():
    reached = {reasons.code_of(r)[0] for r in ALL}
    assert reached == set(reasons.codes()), sorted(set(reasons.codes()) - reached)


def test_codes_are_stable_names_and_carry_no_project_wording():
    for code, (en, es) in reasons.REASONS.items():
        assert re.fullmatch(r"[a-z]+(\.[a-z_]+)+", code), code
        for text in (en, es):
            assert not re.search(r"\bBL-\d+|\bbatch \d|\blote \d|\bU1\d\b", text), (code, text)


def test_values_are_carried_into_the_spanish():
    en = "no fit holds all of the variant's sources (s1, s9)"
    assert reasons.bilingual(en)["es"] == "ningún ajuste contiene todas las fuentes de la variante (s1, s9)"
    pair = reasons.bilingual("policy.no_threshold", label="riskvalidation-default 2026.10.1", test_id="pd.binomial")
    assert "pd.binomial" in pair["es"] and "pd.binomial" in pair["en"]


def test_an_unknown_reason_or_a_missing_value_is_refused_never_passed_on():
    with pytest.raises(ValueError, match="reason"):
        reasons.bilingual("some reason the engine never returns")
    with pytest.raises(ValueError, match="reason"):
        reasons.bilingual("policy.no_threshold", label="x")
    with pytest.raises(ValueError, match="reason"):
        reasons.code_of("")


def test_the_functions_still_return_the_english_strings():
    assert findings.anchor_from_test_row(["design:x"], []) == (None, NO_CITED_ROW)
    assert findings.rwea_share_of_anchor(None) == (None, NOT_RWEA)
    assert inventory.assign_variant([], {}) == (None, "the case has no fit to assign the variant to")
    assert inventory.assign_variant(["x"], {"only": ["y"]}) == ("only", "the single models document of the case")
    d = scope.scope_test(scope.ScopeAnswers(True, True, False, False), "ecb")
    assert d.rationale == "In scope: it applies theory and its output is an estimate (ECB)."
    identity, reason = policy.resolve_policy("x", "pd.binomial")
    assert identity is None and isinstance(reason, str)
