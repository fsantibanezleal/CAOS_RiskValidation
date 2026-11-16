"""The residual risk add-on, RRAO (MAR23.1 to MAR23.8, in force 2023-01-01, published 2020-03-27).

The RRAO is calculated in addition to the other components of the standardised approach (MAR23.1, MAR23.8). It is the
simple sum of the gross notional amounts of the instruments bearing residual risks, multiplied by a risk weight
(MAR23.8):

    RRAO = 1.0% x sum of the gross notionals with an exotic underlying (MAR23.3)
         + 0.1% x sum of the gross notionals bearing other residual risks (MAR23.4)

There is no offsetting of a long against a short: each notional is a gross, non-negative amount and the sums are plain
(a negative notional is refused, because a net notional is not a gross one). `classify_instrument` encodes only the
exclusions of MAR23.7: a back-to-back transaction (the instruments of both transactions are excluded), and an instrument
that is listed or eligible for central clearing, which is excluded from the other residual risks and included when its
underlying is exotic. Not implemented (the caller's): the instrument-level judgement of MAR23.3 to MAR23.6, that is
whether an underlying is exotic, whether a pay-off bears gap, correlation or behavioural risk (MAR23.5) and whether a
risk type alone leaves the instrument outside the add-on (MAR23.6), and the treatment of a fund assumed to be exposed
to the maximum extent of its mandate (MAR23.6 (5)); the scope of the RRAO does not change the risk factors of the
delta, vega, curvature or default risk treatments (MAR23.8 (1)). The engine computes and states no compliance claim.

An instrument that is flagged both as having an exotic underlying and as bearing other residual risks takes the exotic
weight only (the paragraphs state one weight per category; the higher is taken); the caller decides whether the two
flags apply to it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from riskvalidation.regulatory.standardised import RegulatoryParameter, _nonnegative

MAR23_IN_FORCE = "2023-01-01"
#: MAR23.8: the risk weight of instruments with an exotic underlying (MAR23.3)
RW_EXOTIC = RegulatoryParameter(0.01, "MAR23.8", "MAR23.8 (2) first bullet", MAR23_IN_FORCE)
#: MAR23.8: the risk weight of instruments bearing other residual risks (MAR23.4)
RW_OTHER = RegulatoryParameter(0.001, "MAR23.8", "MAR23.8 (2) second bullet", MAR23_IN_FORCE)

_PARAGRAPHS = {
    "charge": "MAR23.8 (simple sum of gross notional amounts times the risk weight)",
    "exotic_underlying": "MAR23.3",
    "other_residual_risks": "MAR23.4",
    "exclusions": "MAR23.7",
}


@dataclass(frozen=True)
class RraoResult:
    """The RRAO with its two sums, its two charges and the paragraphs."""

    exotic_notional: float
    other_notional: float
    exotic_charge: float
    other_charge: float
    charge: float
    paragraphs: dict[str, str]


def _gross_sum(name: str, notionals: Iterable[float]) -> float:
    return sum(_nonnegative(name, x) for x in notionals)


def rrao(exotic_gross_notionals: Iterable[float], other_gross_notionals: Iterable[float]) -> RraoResult:
    """MAR23.8: 1.0% of the simple sum of the gross notionals with an exotic underlying plus 0.1% of those bearing other
    residual risks; no offsetting of long and short."""
    exotic = _gross_sum("exotic_gross_notionals", exotic_gross_notionals)
    other = _gross_sum("other_gross_notionals", other_gross_notionals)
    exotic_charge = RW_EXOTIC.value * exotic
    other_charge = RW_OTHER.value * other
    return RraoResult(exotic, other, exotic_charge, other_charge, exotic_charge + other_charge, dict(_PARAGRAPHS))


def _flag(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def classify_instrument(
    *, exotic_underlying: bool, other_residual_risk: bool, back_to_back: bool, listed_or_cleared: bool
) -> str:
    """The category of an instrument for `rrao`: "exotic", "other" or "excluded", by the exclusions of MAR23.7 only.

    A back-to-back transaction is excluded; an instrument that is listed and/or eligible for central clearing is
    excluded from the other residual risks of MAR23.4 and included when its underlying is exotic. Whether an underlying
    is exotic (MAR23.3), whether an instrument bears other residual risks (MAR23.4, MAR23.5) and whether a risk type
    alone leaves it outside the add-on (MAR23.6) is the instrument-level judgement and stays the caller's: the flags are
    its answers."""
    exotic = _flag("exotic_underlying", exotic_underlying)
    other = _flag("other_residual_risk", other_residual_risk)
    paired = _flag("back_to_back", back_to_back)
    listed = _flag("listed_or_cleared", listed_or_cleared)
    if paired:
        return "excluded"
    if exotic:
        return "exotic"
    if other and not listed:
        return "other"
    return "excluded"


__all__ = ["RW_EXOTIC", "RW_OTHER", "RraoResult", "classify_instrument", "rrao"]
