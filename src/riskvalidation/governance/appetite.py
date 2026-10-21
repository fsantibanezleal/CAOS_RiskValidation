"""The model risk appetite metrics (dossier 06 B.6 l.606-612): each with its value, unit, numerator, denominator and
a light under a stated threshold, which is a **convention** (a direction, a green limit and an amber limit per
metric, inclusive on the good side).

Entries are read in their document form (a mapping with ``tier_block``, ``schedule``, ``lifecycle`` and
``attributes``) and register rows as mappings (``entry_key``, ``id``, ``status``, ``severity``, ``overdue``). The
two finding counts count distinct (entry, id) findings, which is what B.6 counts (W15-14); the number of register
rows behind the numerator is the secondary number beside them, and a finding repeated in n variants of an entry
counts once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from riskvalidation.governance.conventions import APPETITE_THRESHOLDS, CONVENTION
from riskvalidation.governance.findings import OPEN_STATES

DIRECTIONS = ("lower-is-better", "higher-is-better")
METRIC_IDS = tuple(APPETITE_THRESHOLDS)
_COMPLETE_OWNER = ("present", "derivable", "convention")
_PERCENT = ("share", "share-of-model-output")


def light(value: float | None, direction: str, green_limit: float, amber_limit: float) -> str:
    """``green``, ``amber``, ``red`` or ``not_evaluated`` (no value); the limits are inclusive on the good side."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if value is None:
        return "not_evaluated"
    if direction == "lower-is-better":
        return "green" if value <= green_limit else "amber" if value <= amber_limit else "red"
    return "green" if value >= green_limit else "amber" if value >= amber_limit else "red"


def _next_review(entry: Mapping[str, Any]) -> date:
    value = entry["schedule"]["next_review"]
    return value if isinstance(value, date) else date.fromisoformat(value)


def _tier1(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [e for e in entries if e["tier_block"]["tier"] == 1]


def tier1_validated_on_time(
    entries: Sequence[Mapping[str, Any]], as_of: date
) -> tuple[float | None, float | None, float | None]:
    """(Tier 1 entries whose next review is on or after the as-of date, Tier 1 entries, their share); the share is
    ``None`` with none."""
    t1 = _tier1(entries)
    on_time = sum(1 for e in t1 if _next_review(e) >= as_of)
    return on_time, len(t1), (on_time / len(t1) if t1 else None)


def _distinct(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({(r["entry_key"], r["id"]) for r in rows})


def open_high_severity(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, int]:
    """(distinct open S1 or S2 findings, distinct findings of the register, the count, register rows behind the
    numerator)."""
    sel = [r for r in rows if r["status"] in OPEN_STATES and r["severity"] in ("S1", "S2")]
    n = _distinct(sel)
    return n, _distinct(rows), n, len(sel)


def overdue_remediation(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, int]:
    """(distinct findings with an overdue row, distinct findings of the register, the count, the overdue rows)."""
    sel = [r for r in rows if r["overdue"]]
    n = _distinct(sel)
    return n, _distinct(rows), n, len(sel)


def entries_under_exception(entries: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    """(entries whose lifecycle holds an exception record, the entries, the count)."""
    n = sum(1 for e in entries if e["lifecycle"]["exception"])
    return n, len(entries), n


def _red_rows(entry: Mapping[str, Any]) -> int:
    a36 = entry["attributes"]["A36"]["value"]
    if not isinstance(a36, Mapping) or "red" not in a36:
        who = entry.get("key", "an entry")
        raise ValueError(f"{who}: A36 holds no light counts, so the entry cannot be counted in breach or not")
    return a36["red"]


def entries_in_breach(entries: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    """(entries with at least one red test row on any variant, the entries, the count)."""
    n = sum(1 for e in entries if _red_rows(e) > 0)
    return n, len(entries), n


def pma_magnitude(entries: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None, float | None]:
    """(sum of the PMA overlays, sum of the model outputs they adjust, their ratio); ``(None, None, None)`` with no
    PMA amount.

    A PMA is attribute A26 with status ``present`` holding ``overlays_total``, ``currency`` and ``model_output``.
    Amounts are never converted: two currencies raise ``ValueError``.
    """
    pmas = [e["attributes"]["A26"]["value"] for e in entries if e["attributes"]["A26"]["status"] == "present"]
    pmas = [
        p
        for p in pmas
        if isinstance(p, Mapping) and p.get("overlays_total") is not None and p.get("model_output") is not None
    ]
    if not pmas:
        return None, None, None
    currencies = sorted({p.get("currency") for p in pmas}, key=str)
    if len(currencies) > 1:
        raise ValueError(
            f"PMA amounts are in different currencies ({', '.join(map(str, currencies))}); amounts are never converted"
        )
    overlays = sum(p["overlays_total"] for p in pmas)
    output = sum(p["model_output"] for p in pmas)
    return overlays, output, (overlays / output if output else None)


def owner_and_complete_record(entries: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    """(entries with an owner (A28 present, derivable or convention) and no missing attribute, the entries, their
    share).
    """
    n = sum(
        1
        for e in entries
        if e["attributes"]["A28"]["status"] in _COMPLETE_OWNER
        and all(a["status"] != "missing" for a in e["attributes"].values())
    )
    return n, len(entries), (n / len(entries) if entries else None)  # type: ignore[return-value]


def appetite_population(entries: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The ``population`` of the appetite document: entries, register rows, distinct findings and Tier 1 entries."""
    return {
        "entries": len(entries),
        "findings": len(rows),
        "distinct_findings": _distinct(rows),
        "tier1_entries": len(_tier1(entries)),
    }


def _bilingual(en: str, es: str) -> dict[str, str]:
    return {"en": en, "es": es}


# id -> (name, kind, unit, reason when the metric has no value)
_META: dict[str, tuple[dict[str, str], str, str, dict[str, str] | None]] = {
    "tier1-validated-on-time": (
        _bilingual("Share of Tier 1 entries validated on time", "Proporción de entradas de Nivel 1 validadas a tiempo"),
        "share",
        "share",
        _bilingual("no Tier 1 entry exists", "no existe entrada de Nivel 1"),
    ),
    "open-high-severity": (
        _bilingual("Open high-severity findings (S1 and S2)", "Hallazgos abiertos de severidad alta (S1 y S2)"),
        "count",
        "count",
        None,
    ),
    "overdue-remediation": (_bilingual("Overdue remediation", "Remediación vencida"), "count", "count", None),
    "entries-under-exception": (
        _bilingual("Entries used under exception", "Entradas usadas bajo excepción"),
        "count",
        "count",
        None,
    ),
    "entries-in-breach": (
        _bilingual(
            "Entries in breach of their thresholds (a red test row)",
            "Entradas que incumplen sus umbrales (una fila de prueba roja)",
        ),
        "count",
        "count",
        None,
    ),
    "pma-magnitude": (
        _bilingual("PMA magnitude relative to the model output", "Magnitud de PMA relativa al resultado del modelo"),
        "magnitude",
        "share-of-model-output",
        _bilingual("no entry holds a PMA with an amount", "ninguna entrada tiene una PMA con monto"),
    ),
    "owner-and-complete-record": (
        _bilingual(
            "Share of entries with an owner role and a complete record",
            "Proporción de entradas con rol de propietario y registro completo",
        ),
        "share",
        "share",
        _bilingual("no entry exists", "no existe ninguna entrada"),
    ),
}


def _limit(x: float, unit: str) -> str:
    return f"{x * 100:g} percent" if unit in _PERCENT else f"{x:g}"


def _basis(unit: str, direction: str, green: float, amber: float) -> str:
    if direction == "lower-is-better":
        text = f"green up to {_limit(green, unit)}, amber up to {_limit(amber, unit)}, red above"
    else:
        text = f"green from {_limit(green, unit)}, amber from {_limit(amber, unit)}, red below"
    return f"a convention (item 30 of the dossier UNVERIFIED list): {text}"


def appetite_metrics(
    entries: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    as_of: date,
    thresholds: Mapping[str, tuple[str, float, float]] = APPETITE_THRESHOLDS,
) -> tuple[dict, ...]:
    """The seven metric records in the order of B.6, each with its value, unit, numerator, denominator,
    ``register_rows`` (a number for the two distinct-finding counts, ``None`` for the others), threshold (labelled
    ``convention``) and light; a metric with no value has the light ``not_evaluated`` and a reason."""
    missing = [m for m in METRIC_IDS if m not in thresholds]
    if missing:
        raise ValueError(f"the thresholds hold no row for {', '.join(missing)}")
    t1 = tier1_validated_on_time(entries, as_of)
    high = open_high_severity(rows)
    over = overdue_remediation(rows)
    exc = entries_under_exception(entries)
    breach = entries_in_breach(entries)
    pma = pma_magnitude(entries)
    owner = owner_and_complete_record(entries)
    parts: dict[str, tuple[Any, Any, int | None]] = {
        "tier1-validated-on-time": (t1[0], t1[1], None),
        "open-high-severity": (high[0], high[1], high[3]),
        "overdue-remediation": (over[0], over[1], over[3]),
        "entries-under-exception": (exc[0], exc[1], None),
        "entries-in-breach": (breach[0], breach[1], None),
        "pma-magnitude": (pma[0], pma[1], None),
        "owner-and-complete-record": (owner[0], owner[1], None),
    }
    out = []
    for mid in METRIC_IDS:
        name, kind, unit, reason = _META[mid]
        num, den, reg_rows = parts[mid]
        direction, green, amber = thresholds[mid]
        if num is None:
            value = None
        elif unit == "count":
            value = num
        else:
            value = num / den if den else None
        out.append(
            {
                "id": mid,
                "name": dict(name),
                "kind": kind,
                "value": value,
                "unit": unit,
                "numerator": num,
                "denominator": den,
                "register_rows": reg_rows,
                "reason": dict(reason) if (value is None and reason is not None) else None,
                "threshold": {
                    "label": CONVENTION,
                    "basis": _basis(unit, direction, green, amber),
                    "direction": direction,
                    "green_limit": green,
                    "amber_limit": amber,
                },
                "light": light(value, direction, green, amber),
            }
        )
    return tuple(out)
