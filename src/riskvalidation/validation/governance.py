"""The governance validation tests (contract `engine-governance-api` 1.0.0, module validation.governance; dossier 06
B.2 l.539-551, B.3 l.574-575, B.7 l.614-628, B.8 l.630-640, I.2): each returns one `TestResult` that validates
against `common/test-result`, with a test id prefixed `gov.`, registered in the family `governance`.

The cut points are conventions, never regulatory thresholds (item 30 of the dossier's UNVERIFIED list): they are
`Policy` rows keyed by test id (`GOV_ROWS`), and each result records them in `extras["thresholds"]` with the label
`convention`. A `policy` argument of None means the packaged default policy with the `gov.` rows added where it
holds none (`gov_default_policy`); a policy that lacks a key gives the light `not_evaluated`, never a colour. A
test with nothing to evaluate returns `not_evaluated` with its reason in `notes` and no number (the statistic is
NaN, null once serialised, and the metric and the p-value are None). A malformed input raises `ValueError` naming
the field. The engine approves nothing and certifies nothing: the lights say where the record is incomplete, late
or inconsistent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.policy import Policy, Threshold, default_policy
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.governance.findings import OPEN_STATES, is_overdue
from riskvalidation.governance.inventory import ATTRIBUTE_IDS, find_cycle, nodes_on_cycles

NAN = float("nan")
POLICY_SUFFIX = "gov-1.0.0"
CONVENTION = "convention"

DOSSIER_06 = "dossier 06 (Contraste research)"
REF_COMPLETENESS = f"{DOSSIER_06} B.2 l.539-551; SS1/23 Principle 1.2; OSFI E-23 Appendix 1"
REF_OVERDUE = f"{DOSSIER_06} B.7 l.614-628"
REF_DUE = f"{DOSSIER_06} B.8 l.630-640"
REF_OVERRIDE = f"{DOSSIER_06} B.3 l.574-575; SS1/23 Principle 1.3(d)(e)"
REF_EDGES = f"{DOSSIER_06} I.2; SR 11-7 on model dependencies"


@dataclass(frozen=True)
class GovRow:
    """One `gov.` row of the validation policy. A strict row states a cut that 'below' or 'above' triggers: it is
    stored one float beyond the stated value (`math.nextafter`), so a metric equal to the stated value is not past
    it (`Threshold.classify` is inclusive)."""

    kind: str
    amber: float
    red: float
    strict: bool
    source: str


GOV_ROWS: dict[str, GovRow] = {
    "gov.inventory_completeness": GovRow(
        "lower_bad",
        0.9,
        0.7,
        True,
        "convention: amber below 0.9 and red below 0.7 of the (entry, attribute) pairs with a status other than "
        "missing; no source "
        "prescribes a completeness level (dossier 06 B.2 l.539-551, item 30 of the UNVERIFIED list); strict (a "
        "share equal to the cut is "
        "not below it)",
    ),
    "gov.findings_overdue": GovRow(
        "upper_bad",
        0.0,
        0.25,
        True,
        "convention: amber above 0 and red above 0.25 of the open-state findings past their due date; no source "
        "prescribes days or a "
        "tolerance (dossier 06 B.7, item 30); strict (a share equal to the cut is not above it)",
    ),
    "gov.validation_due": GovRow(
        "upper_bad",
        1,
        2,
        False,
        "convention: amber at 1 and red at 2 or more entries past their next review date (dossier 06 B.8 l.630-640, "
        "item 30)",
    ),
    "gov.tier_override_documented": GovRow(
        "upper_bad",
        1,
        1,
        False,
        "convention: red at 1 or more tier overrides without a rationale or an approver; there is no amber (dossier "
        "06 B.3 l.574-575, "
        "SS1/23 Principle 1.3(d)(e)); amber equals red",
    ),
    "gov.edges_acyclic": GovRow(
        "upper_bad",
        1,
        1,
        False,
        "convention: red at 1 or more entries on a cycle or edges to an unknown entry; there is no amber (dossier "
        "06 I.2); amber equals red",
    ),
}
GOV_KEYS = tuple(GOV_ROWS)


def gov_threshold(key: str) -> Threshold:
    """The `Threshold` of a `gov.` row, a strict cut stored one float beyond its stated value."""
    row = GOV_ROWS[key]
    if not row.strict:
        return Threshold(kind=row.kind, amber=float(row.amber), red=float(row.red), source=row.source)
    toward = -math.inf if row.kind == "lower_bad" else math.inf
    return Threshold(
        kind=row.kind,
        amber=math.nextafter(float(row.amber), toward),
        red=math.nextafter(float(row.red), toward),
        source=row.source,
    )


@lru_cache(maxsize=1)
def gov_default_policy() -> Policy:
    """The packaged default policy with the `gov.` rows added where it holds none; the default policy itself, label
    unchanged, when it already holds every `gov.` key."""
    base = default_policy()
    if all(key in base.thresholds for key in GOV_KEYS):
        return base
    thresholds = dict(base.thresholds)
    for key in GOV_KEYS:
        thresholds.setdefault(key, gov_threshold(key))
    return Policy(
        policy_id=base.policy_id,
        version=f"{base.version}+{POLICY_SUFFIX}",
        description=f"{base.description} Governance rows ({POLICY_SUFFIX}): stated conventions, each labelled as one.",
        thresholds=thresholds,
    )


def _entries(inventory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(inventory, Mapping) or not isinstance(inventory.get("entries"), Sequence):
        raise ValueError("inventory must be a mapping with an 'entries' list")
    return list(inventory["entries"])


def _as_date(value: Any, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be a date or an ISO date string, got {value!r}") from exc


def _extras(key: str, **more: Any) -> dict[str, Any]:
    row = GOV_ROWS[key]
    return {
        "thresholds": {
            "label": CONVENTION,
            "kind": row.kind,
            "amber": row.amber,
            "red": row.red,
            "strict": row.strict,
            "source": row.source,
        },
        **more,
    }


def _result(
    key: str,
    pol: Policy,
    *,
    h0: str,
    alt: str,
    reference: str,
    metric: float | None,
    n: int,
    n_events: int | None,
    notes: str,
    hash_of: Any,
    extras: Mapping[str, Any],
) -> TestResult:
    """Classify the metric under the policy; with no metric, or no row in the policy, the light is `not_evaluated`
    and the notes say why.
    """
    light, th = pol.classify(key, metric=metric)
    if metric is None:
        return TestResult(
            test_id=key,
            h0=h0,
            alternative=alt,
            reference=reference,
            policy_version=pol.label,
            light=Light.NOT_EVALUATED,
            statistic=NAN,
            n=n,
            n_events=n_events,
            notes=f"not evaluated: {notes}",
            inputs_hash=inputs_hash(hash_of),
            extras=dict(extras),
        )
    if th is None:
        notes = f"not evaluated: policy {pol.label} holds no threshold for {key}; {notes}"
    return TestResult(
        test_id=key,
        h0=h0,
        alternative=alt,
        reference=reference,
        policy_version=pol.label,
        light=light,
        metric=metric,
        n=n,
        n_events=n_events,
        notes=notes,
        inputs_hash=inputs_hash(hash_of),
        extras=dict(extras),
    )


@register(
    "gov.inventory_completeness",
    family="governance",
    name="Inventory completeness",
    h0="every one of the 40 attributes of every entry has a status other than missing",
    reference=REF_COMPLETENESS,
    decision="metric",
)
def inventory_completeness(
    inventory: Mapping[str, Any],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """metric = the share of (entry, attribute) pairs whose status is not `missing`; n the pairs; n_events the
    missing pairs. Light (policy key `gov.inventory_completeness`, lower_bad, strict): amber below 0.9, red below
    0.7 (a convention). statistic and p_value None; an inventory with no entry is `not_evaluated` with its reason."""
    key = "gov.inventory_completeness"
    pol = policy if policy is not None else gov_default_policy()
    entries = _entries(inventory)
    h0 = GOV_H0[key]
    alt = "at least one attribute of an entry is missing"
    if not entries:
        return _result(
            key,
            pol,
            h0=h0,
            alt=alt,
            reference=REF_COMPLETENESS,
            metric=None,
            n=0,
            n_events=None,
            notes="the inventory holds no entry, so there is no attribute to count",
            hash_of=[],
            extras=_extras(key),
        )
    missing_by_attribute = dict.fromkeys(ATTRIBUTE_IDS, 0)
    pairs = 0
    for e in entries:
        attrs = e["attributes"]
        for aid in ATTRIBUTE_IDS:
            if aid not in attrs:
                raise ValueError(f"entry {e.get('key')}: attribute {aid} is absent")
            pairs += 1
            missing_by_attribute[aid] += 1 if attrs[aid]["status"] == "missing" else 0
    missing = sum(missing_by_attribute.values())
    return _result(
        key,
        pol,
        h0=h0,
        alt=alt,
        reference=REF_COMPLETENESS,
        metric=(pairs - missing) / pairs,
        n=pairs,
        n_events=missing,
        notes=f"{missing} of {pairs} (entry, attribute) pairs are missing across {len(entries)} entries; a convention, "
        "no source prescribes a completeness level",
        hash_of=[(e.get("key"), [e["attributes"][a]["status"] for a in ATTRIBUTE_IDS]) for e in entries],
        extras=_extras(
            key, entries=len(entries), missing_by_attribute={k: v for k, v in missing_by_attribute.items() if v}
        ),
    )


@register(
    "gov.findings_overdue",
    family="governance",
    name="Overdue findings",
    h0="no open finding is past its due date",
    reference=REF_OVERDUE,
    decision="metric",
)
def findings_overdue(
    register: Sequence[Mapping[str, Any]],
    as_of: date,
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """metric = the share of open-state rows (`open`, `action_plan_agreed`, `in_progress`, `submitted_for_closure`,
    `overdue`) that are past their due date at `as_of`; n the open-state rows; n_events the overdue rows
    (`is_overdue`). Light (policy key `gov.findings_overdue`, upper_bad, strict): amber above 0, red above 0.25 (a
    convention). No open-state row: `not_evaluated`."""
    key = "gov.findings_overdue"
    pol = policy if policy is not None else gov_default_policy()
    as_of_d = _as_date(as_of, "as_of")
    h0 = GOV_H0[key]
    alt = "an open finding is past its due date"
    open_rows = [r for r in register if r["status"] in OPEN_STATES]
    if not open_rows:
        return _result(
            key,
            pol,
            h0=h0,
            alt=alt,
            reference=REF_OVERDUE,
            metric=None,
            n=0,
            n_events=None,
            notes="the register holds no open-state row",
            hash_of=[],
            extras=_extras(key, as_of=as_of_d.isoformat()),
        )
    over = sum(1 for r in open_rows if is_overdue(r["status"], _as_date(r["due"], "due"), as_of_d))
    return _result(
        key,
        pol,
        h0=h0,
        alt=alt,
        reference=REF_OVERDUE,
        metric=over / len(open_rows),
        n=len(open_rows),
        n_events=over,
        notes=f"{over} of {len(open_rows)} open-state register rows are past their due date at "
        f"{as_of_d.isoformat()}; the "
        "deadline table is a convention, no source prescribes days",
        hash_of=[(r["status"], str(r["due"])) for r in open_rows] + [as_of_d.isoformat()],
        extras=_extras(key, as_of=as_of_d.isoformat()),
    )


@register(
    "gov.validation_due",
    family="governance",
    name="Validation due",
    h0="no entry is past its next review date",
    reference=REF_DUE,
    decision="metric",
)
def validation_due(
    inventory: Mapping[str, Any],
    as_of: date,
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """metric = the number of entries whose next review date is before `as_of`; n the entries; n_events the same
    count. Light (policy key `gov.validation_due`, upper_bad): amber at 1, red at 2 or more (a convention)."""
    key = "gov.validation_due"
    pol = policy if policy is not None else gov_default_policy()
    as_of_d = _as_date(as_of, "as_of")
    h0 = GOV_H0[key]
    alt = "an entry is past its next review date"
    entries = _entries(inventory)
    if not entries:
        return _result(
            key,
            pol,
            h0=h0,
            alt=alt,
            reference=REF_DUE,
            metric=None,
            n=0,
            n_events=None,
            notes="the inventory holds no entry",
            hash_of=[],
            extras=_extras(key, as_of=as_of_d.isoformat()),
        )
    late = [
        e["key"] for e in entries if _as_date(e["schedule"]["next_review"], f"{e.get('key')} next_review") < as_of_d
    ]
    return _result(
        key,
        pol,
        h0=h0,
        alt=alt,
        reference=REF_DUE,
        metric=float(len(late)),
        n=len(entries),
        n_events=len(late),
        notes=f"{len(late)} of {len(entries)} entries are past their next review date at {as_of_d.isoformat()}"
        + (f": {', '.join(late)}" if late else "")
        + "; the frequency table is a convention",
        hash_of=[(e["key"], str(e["schedule"]["next_review"])) for e in entries] + [as_of_d.isoformat()],
        extras=_extras(key, as_of=as_of_d.isoformat(), entries_past_review=late),
    )


@register(
    "gov.tier_override_documented",
    family="governance",
    name="Tier overrides documented",
    h0="every tier override has a rationale and an approver",
    reference=REF_OVERRIDE,
    decision="metric",
)
def tier_override_documented(
    inventory: Mapping[str, Any],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """metric = the number of overrides without a rationale or an approver; n the overrides. Light (policy key
    `gov.tier_override_documented`, upper_bad): red at 1 or more (a convention). With no override the row is green
    with n 0."""
    key = "gov.tier_override_documented"
    pol = policy if policy is not None else gov_default_policy()
    entries = _entries(inventory)
    for e in entries:
        if not isinstance(e.get("tier_block"), Mapping):
            raise ValueError(f"entry {e.get('key')}: tier_block is absent")
    overrides = [
        (e["key"], e["tier_block"]["override"]) for e in entries if e["tier_block"].get("override") is not None
    ]
    bad = [k for k, o in overrides if not str(o.get("rationale", "")).strip() or not str(o.get("approver", "")).strip()]
    return _result(
        key,
        pol,
        h0=GOV_H0[key],
        alt="a tier override lacks a rationale or an approver",
        reference=REF_OVERRIDE,
        metric=float(len(bad)),
        n=len(overrides),
        n_events=len(bad),
        notes=f"{len(bad)} of {len(overrides)} tier overrides lack a rationale or an approver"
        + (f": {', '.join(bad)}" if bad else ""),
        hash_of=[
            (k, o.get("to_tier"), bool(str(o.get("rationale", "")).strip()), bool(str(o.get("approver", "")).strip()))
            for k, o in overrides
        ],
        extras=_extras(key, undocumented=bad),
    )


@register(
    "gov.edges_acyclic",
    family="governance",
    name="Dependency graph acyclic",
    h0="the dependency graph has no cycle and every edge joins two entries",
    reference=REF_EDGES,
    decision="metric",
)
def edges_acyclic(
    inventory: Mapping[str, Any],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """metric = the number of entries on a cycle plus the number of edges to an unknown entry; n the edges. Light
    (policy key `gov.edges_acyclic`, upper_bad): red at 1 or more (a convention). The cycle is named in `notes`
    the way `build_inventory` names it."""
    key = "gov.edges_acyclic"
    pol = policy if policy is not None else gov_default_policy()
    keys = [e["key"] for e in _entries(inventory)]
    edges = list(inventory.get("edges", []))
    known = set(keys)
    unknown = [ed for ed in edges if ed["upstream"] not in known or ed["downstream"] not in known]
    pairs = [
        (ed["upstream"], ed["downstream"]) for ed in edges if ed["upstream"] in known and ed["downstream"] in known
    ]
    on_cycle = nodes_on_cycles(keys, pairs)
    cycle = find_cycle(keys, pairs) if on_cycle else None
    notes = []
    if cycle:
        notes.append("dependency cycle: " + ", ".join(cycle))
    for ed in unknown:
        absent = ed["upstream"] if ed["upstream"] not in known else ed["downstream"]
        notes.append(f"edge {ed['upstream']} to {ed['downstream']}: unknown entry {absent}")
    return _result(
        key,
        pol,
        h0=GOV_H0[key],
        alt="the dependency graph holds a cycle or an edge to an unknown entry",
        reference=REF_EDGES,
        metric=float(len(on_cycle) + len(unknown)),
        n=len(edges),
        n_events=len(on_cycle) + len(unknown),
        notes="; ".join(notes) if notes else f"{len(edges)} edges join {len(keys)} entries with no cycle",
        hash_of=[sorted(keys), sorted(pairs)],
        extras=_extras(key, entries_on_cycle=sorted(on_cycle), unknown_edges=len(unknown)),
    )


GOV_H0 = {
    "gov.inventory_completeness": "every one of the 40 attributes of every entry has a status other than missing",
    "gov.findings_overdue": "no open finding is past its due date",
    "gov.validation_due": "no entry is past its next review date",
    "gov.tier_override_documented": "every tier override has a rationale and an approver",
    "gov.edges_acyclic": "the dependency graph has no cycle and every edge joins two entries",
}
