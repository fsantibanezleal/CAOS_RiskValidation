"""Requirements RV-1582 to RV-1588 (docs/design/features/u15-governance/requirements.md): the fix round after the
Opus review of U15-E: non-finite shares (F4), the first matching row of an anchor (F5), the kind and the gov. labels of
a policy identity (F6) and the registration of the gov. tests on `import riskvalidation.validation` (deviation 7)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from riskvalidation.governance.findings import Finding, anchor_from_test_row, rwea_share_of_anchor
from riskvalidation.governance.opinion import finding_level, opinion_level
from riskvalidation.governance.policy import resolve_policy
from riskvalidation.governance.scope import ScopeAnswers, scope_record

from .governance_support import AS_OF

NON_FINITE = (float("nan"), float("inf"), float("-inf"))
GOV_IDS = {
    "gov.inventory_completeness",
    "gov.findings_overdue",
    "gov.validation_due",
    "gov.tier_override_documented",
    "gov.edges_acyclic",
}


@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_share_is_refused(bad):
    with pytest.raises(ValueError, match="finite"):
        finding_level("S3", bad)
    with pytest.raises(ValueError, match="finite"):
        opinion_level([("S3", bad)])
    with pytest.raises(ValueError, match="finite"):
        rwea_share_of_anchor({"unit": "share-of-rwea", "value": bad})
    with pytest.raises(ValueError, match="finite"):
        rwea_share_of_anchor({"unit": "rwea-amount", "value": bad}, 100.0)
    with pytest.raises(ValueError, match="finite"):
        rwea_share_of_anchor({"unit": "rwea-amount", "value": 1.0}, bad)
    with pytest.raises(ValueError, match="finite"):
        Finding("C01", "v", "f", "S2", "open", ("pd.jeffreys@P0",), AS_OF, 1, "open", bad)


def test_an_anchor_stops_at_the_first_matching_row():
    rows = [
        {"test_id": "t.a", "model_id": "m", "segment": "s1", "statistic": None, "metric": None},
        {"test_id": "t.a", "model_id": "m", "segment": "s2", "statistic": 7.0, "metric": None},
    ]
    anchor, reason = anchor_from_test_row(["t.a@m"], rows)
    assert anchor is None and reason  # the first matching row holds no value: no second try on another segment
    anchor, reason = anchor_from_test_row(["t.a@m@s2"], rows)
    assert reason is None and anchor["value"] == 7.0 and anchor["basis"].endswith("#/tests/1/statistic")


def test_a_packaged_label_for_an_unregistered_test_has_no_fallback_kind():
    identity, reason = resolve_policy("riskvalidation-default 2026.10.1", "fair.equalized_odds")
    if identity["kind"] is None:  # the fair tests are not registered at this head
        assert identity["kind_reason"]
    identity, _ = resolve_policy("riskvalidation-default 2026.10.1", "zz.not_registered")
    assert identity["kind"] is None and identity["kind_reason"]
    assert "descriptive" not in (identity["direction_reason"] or "")
    assert "no packaged threshold" in identity["direction_reason"]


def test_a_gov_label_resolves_through_the_governance_policy():
    from riskvalidation.validation.governance import gov_default_policy

    label = gov_default_policy().label
    for key, direction in (
        ("gov.inventory_completeness", "lower_bad"),
        ("gov.findings_overdue", "upper_bad"),
        ("gov.edges_acyclic", "upper_bad"),
    ):
        identity, reason = resolve_policy(label, key)
        assert reason is None and identity["source"] == "packaged-policy", key
        assert identity["direction"] == direction and identity["direction_reason"] is None, key
        assert identity["kind"] == "metric", key


def test_importing_the_validation_package_registers_the_gov_tests():
    code = (
        "import riskvalidation.validation; from riskvalidation.core.registry import catalogue;"
        "ids={s.test_id for s in catalogue('governance')}; print(sorted(ids))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert all(i in out.stdout for i in GOV_IDS)


@pytest.mark.xfail(
    strict=True,
    reason="waiting for the corrected Contraste fixture rationale (review F2): the engine copy of "
    "the fixture still holds the osfi-2027 and ecb sentences of the fixture builder; once the main session "
    "sends the corrected fixture this test passes and the xfail mark is removed",
)
def test_scope_rationale_equals_the_fixture_rationale(inventory_doc):
    for e in inventory_doc["entries"]:
        rec = scope_record(ScopeAnswers(**e["scope"]["answers"]))
        assert [d.rationale for d in rec] == [p["rationale"]["en"] for p in e["scope"]["profiles"]]
