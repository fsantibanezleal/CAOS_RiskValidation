"""The B.7 findings workflow (dossier 06 B.7 l.614-628) over the product's findings (W15-2, W15-5): the severity map,
the status map, the deadline, ageing and overdue arithmetic, the seven-state transitions, the quantitative anchor
and the register counts.

The product keeps its direction S1 critical to S4 minor; the ECB F equivalent is shown beside it (S1 F4 ... S4
F1). The deadline table, the transition graph and the status map are **conventions** (06 l.624-628 names the seven
states and no transition; no source prescribes days). ``overdue`` is never a product status: it is computed from
the due date and the as-of date by :func:`is_overdue`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from riskvalidation.governance.conventions import DEADLINE_DAYS, FINDING_GRAPH

SEVERITIES = ("S1", "S2", "S3", "S4")
STATES = tuple(FINDING_GRAPH)
OPEN_STATES = ("open", "action_plan_agreed", "in_progress", "submitted_for_closure", "overdue")
PRODUCT_STATUSES = ("open", "accepted", "closed")
_ECB_F = {"S1": "F4", "S2": "F3", "S3": "F2", "S4": "F1"}
_STATUS_MAP = {"open": "open", "accepted": "risk_accepted", "closed": "validated_closed"}
_CITED = re.compile(r"[a-z0-9_.]+@[^@]+(@.+)?")
UNIT_STATISTIC = "test statistic (dimensionless)"
UNIT_METRIC = "metric of the test row (the unit its test defines)"
NO_CITED_ROW = (
    "the evidence is not a test_id@model row (it is a design:, contract:, rate: or bare test id), so no statistic "
    "is read"
)
NO_VALUE_IN_ROW = "the cited test row holds no statistic or metric (or no row matches), so the anchor is null"
NOT_RWEA = "the anchor is not stated as a share of RWEA or as an RWEA amount, so no RWEA level applies"


def _severity(severity: str) -> str:
    if severity not in _ECB_F:
        raise ValueError(f"unknown severity {severity!r}; expected one of {SEVERITIES}")
    return severity


def _state(state: str) -> str:
    if state not in FINDING_GRAPH:
        raise ValueError(f"unknown finding state {state!r}; expected one of {STATES}")
    return state


@dataclass(frozen=True)
class Finding:
    """One finding of one variant of a case; the key is the triple, because the same finding repeats in each variant
    (A D3 4.2)."""

    case_id: str
    variant_id: str
    id: str
    severity: str  # "S1" to "S4"
    product_status: str  # "open", "accepted" or "closed"
    evidence: tuple[str, ...]
    raised: date
    tier: int
    status: str  # one of the seven B.7 states
    rwea_share: float | None  # from rwea_share_of_anchor; None where no RWEA level applies

    def __post_init__(self) -> None:
        _severity(self.severity)
        _state(self.status)
        if self.product_status not in PRODUCT_STATUSES:
            raise ValueError(f"{self.key}: unknown product status {self.product_status!r}")
        if not self.evidence:
            raise ValueError(f"{self.key}: a finding needs evidence")
        if isinstance(self.tier, bool) or not isinstance(self.tier, int) or not 1 <= self.tier <= 4:
            raise ValueError(f"{self.key}: tier must be 1 to 4, got {self.tier!r}")
        if self.rwea_share is not None and self.rwea_share < 0:
            raise ValueError(f"{self.key}: rwea_share must not be negative")

    @property
    def key(self) -> str:
        return f"{self.case_id}/{self.variant_id}/{self.id}"


def severity_to_ecb(severity: str) -> str:
    """The ECB TRIM equivalent of a product severity (06 l.616-617)."""
    return _ECB_F[_severity(severity)]


def map_product_status(status: str) -> str:
    """The B.7 state a product status maps to (W15-5); ``overdue`` is never returned here."""
    if status not in _STATUS_MAP:
        raise ValueError(f"unknown product status {status!r}; expected one of {PRODUCT_STATUSES}")
    return _STATUS_MAP[status]


def deadline_days(severity: str, tier: int, table: Mapping[str, Sequence[int]] = DEADLINE_DAYS) -> int:
    """Days from the raised date to the due date, by severity and tier 1 to 4 (a convention)."""
    _severity(severity)
    if isinstance(tier, bool) or not isinstance(tier, int) or not 1 <= tier <= 4:
        raise ValueError(f"tier must be 1 to 4, got {tier!r}")
    if severity not in table:
        raise ValueError(f"the deadline table has no row for {severity}")
    row = table[severity]
    if len(row) != 4:
        raise ValueError(f"the deadline row of {severity} must hold four values, got {len(row)}")
    return int(row[tier - 1])


def due_date(raised: date, days: int) -> date:
    """The due date: the raised date plus a number of calendar days."""
    if days < 0:
        raise ValueError(f"days must not be negative, got {days}")
    return raised + timedelta(days=days)


def ageing_days(raised: date, as_of: date) -> int:
    """The whole days from the raised date to the as-of date."""
    if as_of < raised:
        raise ValueError(f"as_of {as_of.isoformat()} is before the raised date {raised.isoformat()}")
    return (as_of - raised).days


def is_overdue(status: str, due: date, as_of: date) -> bool:
    """True only for an open-state finding and only after the due date (a risk-accepted or closed finding is never
    overdue)."""
    return _state(status) in OPEN_STATES and as_of > due


def transition_finding(state: str, to: str, risk_acceptance: Mapping[str, object] | None = None) -> str:
    """The state after moving from ``state`` to ``to`` along the B.7 graph (a convention); entering ``risk_accepted``
    needs a record.

    ``validated_closed`` is terminal (a reopened finding is a new finding); ``risk_accepted`` leaves only to
    ``open``.
    """
    _state(state)
    _state(to)
    if to not in FINDING_GRAPH[state]:
        raise ValueError(f"cannot move from {state!r} to {to!r}")
    if to == "risk_accepted" and (
        risk_acceptance is None or "approver" not in risk_acceptance or "expiry" not in risk_acceptance
    ):
        raise ValueError("a risk acceptance needs a record with an approver field and an expiry field")
    return to


def anchor_from_test_row(
    evidence: Sequence[str], tests: Sequence[Mapping[str, object]], *, document: str = ""
) -> tuple[dict | None, str | None]:
    """The quantitative anchor of a finding: ``(anchor, reason)``, exactly one of them ``None``.

    Only the first evidence of the form ``test_id@model_id`` or ``test_id@model_id@segment`` is cited. The anchor
    reads the first non-null of the cited row's statistic and metric; its basis is the pointer
    ``<document>#/tests/<i>/<field>``. The unit of a metric is the unit the row states (a ``unit`` key), otherwise
    the unit its test defines; it is never guessed.
    """
    cited = [e for e in evidence if _CITED.fullmatch(e)]
    if not cited:
        return None, NO_CITED_ROW
    parts = cited[0].split("@", 2)
    test_id, model_id = parts[0], parts[1]
    segment = parts[2] if len(parts) > 2 else None
    for i, row in enumerate(tests):
        if row.get("test_id") != test_id or row.get("model_id") != model_id:
            continue
        if segment is not None and row.get("segment") != segment:
            continue
        for name in ("statistic", "metric"):
            value = row.get(name)
            if value is None or isinstance(value, bool) or not math.isfinite(float(value)):  # type: ignore[arg-type]
                continue
            stated = row.get("unit")
            unit = (
                stated if isinstance(stated, str) and stated else UNIT_STATISTIC if name == "statistic" else UNIT_METRIC
            )
            return {
                "metric": name,
                "test_id": test_id,
                "model_id": model_id,
                "segment": row.get("segment"),
                "value": value,
                "unit": unit,
                "basis": f"{document}#/tests/{i}/{name}",
            }, None
    return None, NO_VALUE_IN_ROW


def rwea_share_of_anchor(
    anchor: Mapping[str, object] | None, rwea_total: float | None = None
) -> tuple[float | None, str | None]:
    """The share of RWEA an anchor states, ``(share, reason)``, exactly one of them ``None`` (W15-11).

    Non-null only where the anchor's unit is ``share-of-rwea`` (the share is its value) or ``rwea-amount`` (its
    value over ``rwea_total``, which must be given and positive). Any other unit, or no anchor, gives ``(None,
    reason)``.
    """
    if anchor is None:
        return None, NOT_RWEA
    unit, value = anchor.get("unit"), anchor.get("value")
    if unit == "share-of-rwea":
        if value is None or float(value) < 0:  # type: ignore[arg-type]
            raise ValueError(f"a share of RWEA must not be negative, got {value!r}")
        return float(value), None  # type: ignore[arg-type]
    if unit == "rwea-amount":
        if rwea_total is None or not rwea_total > 0:
            raise ValueError("an rwea-amount anchor needs a positive RWEA total")
        if value is None or float(value) < 0:  # type: ignore[arg-type]
            raise ValueError(f"an RWEA amount must not be negative, got {value!r}")
        return float(value) / float(rwea_total), None  # type: ignore[arg-type]
    return None, NOT_RWEA


def register_counts(rows: Sequence[Mapping[str, Any]]) -> dict:
    """The register counts: ``total``, ``by_severity``, ``by_status``, ``overdue`` and ``by_case`` (sorted by case
    id, the same four)."""

    def tally(sub: Sequence[Mapping[str, Any]]) -> dict:
        by_sev = dict.fromkeys(SEVERITIES, 0)
        by_state = dict.fromkeys(STATES, 0)
        for r in sub:
            by_sev[_severity(r["severity"])] += 1
            by_state[_state(r["status"])] += 1
        return {
            "total": len(sub),
            "by_severity": by_sev,
            "by_status": by_state,
            "overdue": sum(1 for r in sub if r["overdue"]),
        }

    out = tally(rows)
    out["by_case"] = [
        {"case_id": c, **tally([r for r in rows if r["case_id"] == c])} for c in sorted({r["case_id"] for r in rows})
    ]
    return out
