"""The IFRS 7 35H (allowance) and 35I (gross carrying amount) roll-forwards and their identities (dossier 02 I.1 items 9
and 10, I.2).

Tolerance basis, one definition for every identity: a money identity closes when its absolute difference is at most
`tolerance * max(1, G)`, with `G` the sum of the absolute closing amounts of the gross carrying amount rows. A
`RollForwardRow.residual` is always `closing - (opening + sum(movements))` itself, never clipped or adjusted; the
identity is judged by `check_identities`, never by the row. Contraste internal gate, not a regulatory threshold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from riskvalidation.provisions.overlays import Overlay, check_unique

STAGES = ("stage1", "stage2", "stage3", "poci")
GCA_LINE_IDS = (
    "originations", "transfer-1-2", "transfer-2-3", "transfer-2-1", "transfer-3-2", "repayments", "write-off",
    "derecognition", "other",
)
_OPENING, _CLOSING = "opening", "closing"


@dataclass(frozen=True)
class RollForwardRow:
    """One stage row of a roll-forward: opening, signed movements (order kept), closing and the residual."""

    stage: str
    opening: float
    movements: tuple[tuple[str, float], ...]
    closing: float
    residual: float


def _row(stage: str, opening: float, movements: Mapping[str, float], closing: float) -> RollForwardRow:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    values = [float(opening), float(closing), *(float(v) for v in movements.values())]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("opening, closing and every movement must be finite")
    moves = tuple((str(k), float(v)) for k, v in movements.items())
    return RollForwardRow(stage=stage, opening=float(opening), movements=moves, closing=float(closing),
                          residual=_residual(float(opening), moves, float(closing)))


def _residual(opening: float, movements: Sequence[tuple[str, float]], closing: float) -> float:
    return closing - (opening + math.fsum(v for _, v in movements))


def allowance_rollforward(stage: str, opening: float, movements: Mapping[str, float], closing: float) -> RollForwardRow:
    """A row of the IFRS 7 35H table. The allowance of stages 1 to 3 is not negative; only POCI may be (5.5.14)."""
    row = _row(stage, opening, movements, closing)
    if stage != "poci" and (row.opening < 0.0 or row.closing < 0.0):
        raise ValueError(f"the allowance of {stage} cannot be negative (only POCI may be, 5.5.14)")
    return row


def gca_rollforward(stage: str, opening: float, movements: Mapping[str, float], closing: float) -> RollForwardRow:
    """A row of the IFRS 7 35I table; `line_id` is one of `GCA_LINE_IDS`."""
    for line_id in movements:
        if line_id not in GCA_LINE_IDS:
            raise ValueError(f"line_id {line_id!r} is not one of {GCA_LINE_IDS}")
    row = _row(stage, opening, movements, closing)
    if row.opening < 0.0 or row.closing < 0.0:
        raise ValueError("the gross carrying amount cannot be negative")
    return row


def _split_waterfall(lines: Sequence[Mapping]) -> tuple[float | None, float | None, dict[str, float]]:
    """(opening amount, closing amount, movement amount by line id) of the waterfall lines."""
    opening = closing = None
    moves: dict[str, float] = {}
    for line in lines:
        kind = line.get("kind", line.get("id"))
        if kind == _OPENING:
            opening = float(line["amount"])
        elif kind == _CLOSING:
            closing = float(line["amount"])
        else:
            moves[str(line["id"])] = moves.get(str(line["id"]), 0.0) + float(line["amount"])
    return opening, closing, moves


def carried_by_stage(overlays: Sequence[Overlay]) -> dict[str, float]:
    """The overlay money carried in each stage's allowance closing (portfolio level goes to stage1)."""
    check_unique(overlays)
    return {s: math.fsum(float(o.amount) for o in overlays if o.carried_stage == s) for s in STAGES}


def carried_from_lines(waterfall_lines: Sequence[Mapping]) -> dict[str, float]:
    """The same amounts read from the overlay lines of a waterfall (`kind = overlay`, a null `stage` is stage1)."""
    out = {s: 0.0 for s in STAGES}
    for line in waterfall_lines:
        if line.get("kind") == "overlay":
            out[line.get("stage") or "stage1"] += float(line["amount"])
    return out


def identity_residuals(allowance: Sequence[RollForwardRow], gca: Sequence[RollForwardRow],
                       waterfall_lines: Sequence[Mapping], stage_ecl: Mapping[str, float],
                       overlays: Sequence[Overlay] = (),
                       carried: Mapping[str, float] | None = None) -> dict[str, float]:
    """The absolute residual of every identity, by name; an identity that cannot be evaluated (no opening or closing
    waterfall line, a stage without a stage ECL) is `inf`; a movement with no waterfall line counts as a line of 0.
    The names are those of `check_identities`. `carried` (the overlay money by stage,
    for a caller that holds the waterfall lines but not the overlays) replaces `overlays` when given."""
    carry = carried_by_stage(overlays) if carried is None else dict(carried)
    out: dict[str, float] = {}
    for r in allowance:
        out[f"allowance-rollforward:{r.stage}"] = abs(_residual(r.opening, r.movements, r.closing))
    for r in gca:
        out[f"gca-rollforward:{r.stage}"] = abs(_residual(r.opening, r.movements, r.closing))
    opening, closing, line_moves = _split_waterfall(waterfall_lines)
    stage_closing = math.fsum(r.closing for r in allowance)
    if opening is None or closing is None:
        out["waterfall-closes"] = math.inf
        out["waterfall-equals-stage-closings"] = math.inf
    else:
        out["waterfall-closes"] = abs(opening + math.fsum(line_moves.values()) - closing)
        out["waterfall-equals-stage-closings"] = abs(stage_closing - closing)
    stage_moves: dict[str, list[float]] = {}
    for r in allowance:
        for line_id, amount in r.movements:
            stage_moves.setdefault(line_id, []).append(amount)
    worst = 0.0
    for line_id in set(stage_moves) | set(line_moves):
        worst = max(worst, abs(line_moves.get(line_id, 0.0) - math.fsum(stage_moves.get(line_id, []))))
    out["waterfall-lines-equal-movements"] = worst
    for r in allowance:
        key = f"stage-closing-equals-ecl:{r.stage}"
        if r.stage not in stage_ecl:
            out[key] = math.inf
            continue
        out[key] = abs(r.closing - (float(stage_ecl[r.stage]) + carry.get(r.stage, 0.0)))
    return out


def gross_total(gca: Sequence[RollForwardRow]) -> float:
    """G: the total gross carrying amount, the sum of the absolute closing amounts of the 35I rows."""
    return math.fsum(abs(r.closing) for r in gca)


def check_identities(allowance: Sequence[RollForwardRow], gca: Sequence[RollForwardRow],
                     waterfall_lines: Sequence[dict], stage_ecl: Mapping[str, float], overlays: Sequence[Overlay] = (),
                     tolerance: float = 1e-6) -> list[str]:
    """The names of the identities that fail, empty when all close, within `tolerance * max(1, G)`. Never raises on a
    mismatch.

    Names: `allowance-rollforward:<stage>`, `gca-rollforward:<stage>`, `waterfall-closes`,
    `waterfall-equals-stage-closings`, `waterfall-lines-equal-movements`, `stage-closing-equals-ecl:<stage>` (the
    allowance closing of a stage equals its stage ECL plus the overlays carried in it).
    """
    limit = float(tolerance) * max(1.0, gross_total(gca))
    residuals = identity_residuals(allowance, gca, waterfall_lines, stage_ecl, overlays)
    return [name for name, value in residuals.items() if not value <= limit]
