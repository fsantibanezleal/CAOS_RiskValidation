"""Overlays as explicit line items (dossier 02 I.1 item 9; ECB July 2024; BCBS d350 paras 50 and 51).

Each overlay is reported separately in every output and never merged into the model ECL. The stage rule is a
convention of the contract `engine-provisions-api`, not an IFRS rule: an overlay with a stage is carried in that
stage's allowance closing; an overlay with `stage = None` (portfolio level) is carried in the stage1 closing, because
the roll-forward has no portfolio row. This module states no compliance claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

METHODS = ("simulation", "scenario", "expert")
TARGETS = ("pd", "lgd", "ead", "staging-share")
STAGING_EFFECTS = ("none", "transfers-to-stage2")
STAGES = ("stage1", "stage2", "stage3", "poci")


@dataclass(frozen=True)
class Overlay:
    """One overlay: money in `currency`, signed (a positive amount raises the allowance)."""

    id: str
    rationale_en: str
    rationale_es: str
    method: str
    target: str
    segment_id: str | None
    stage: str | None
    amount: float
    staging_effect: str
    review_date: str
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Overlay.id must be a non-empty string")
        for field, value in (("rationale_en", self.rationale_en), ("rationale_es", self.rationale_es)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Overlay.{field} must be a non-empty string")
        for field, value, allowed in (
            ("method", self.method, METHODS),
            ("target", self.target, TARGETS),
            ("staging_effect", self.staging_effect, STAGING_EFFECTS),
        ):
            if value not in allowed:
                raise ValueError(f"Overlay.{field} must be one of {allowed}, got {value!r}")
        if self.stage is not None and self.stage not in STAGES:
            raise ValueError(f"Overlay.stage must be None or one of {STAGES}, got {self.stage!r}")
        if not math.isfinite(float(self.amount)):
            raise ValueError(f"Overlay.amount must be finite, got {self.amount!r}")
        try:
            date.fromisoformat(self.review_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Overlay.review_date must be an ISO date, got {self.review_date!r}") from exc
        if not (isinstance(self.currency, str) and len(self.currency) == 3 and self.currency.isalpha()
                and self.currency.isupper()):
            raise ValueError(f"Overlay.currency must be an ISO 4217 code, got {self.currency!r}")

    @property
    def carried_stage(self) -> str:
        """The stage whose allowance closing carries this overlay (portfolio level goes to stage1)."""
        return self.stage if self.stage is not None else "stage1"


def check_unique(overlays: Sequence[Overlay]) -> None:
    """Overlay ids are unique within a run; raises `ValueError` otherwise."""
    seen: set[str] = set()
    for o in overlays:
        if o.id in seen:
            raise ValueError(f"overlay ids must be unique within a run, {o.id!r} repeats")
        seen.add(o.id)


def sum_amounts(overlays: Sequence[Overlay]) -> float:
    """The sum of the signed overlay amounts, exactly rounded."""
    check_unique(overlays)
    return math.fsum(float(o.amount) for o in overlays)


def overlay_lines(overlays: Sequence[Overlay]) -> list[dict]:
    """The waterfall lines of the overlays, one each, in id order; never merged into another line."""
    check_unique(overlays)
    return [
        {"id": f"overlay-{o.id}", "kind": "overlay", "overlay_id": o.id, "amount": float(o.amount), "stage": o.stage}
        for o in sorted(overlays, key=lambda x: x.id)
    ]


def overlay_share(overlay_total: float, total_allowance: float) -> float:
    """overlay_total / total_allowance; raises when the allowance is not positive (no placeholder)."""
    if not float(total_allowance) > 0.0:
        raise ValueError(f"total_allowance must be positive to state an overlay share, got {total_allowance!r}")
    return float(overlay_total) / float(total_allowance)


def total_allowance(model_ecl: float, overlays: Sequence[Overlay]) -> float:
    """model_ecl + sum(overlay.amount): the allowance with the overlays as separate lines."""
    return float(model_ecl) + sum_amounts(overlays)
