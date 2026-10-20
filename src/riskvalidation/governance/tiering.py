"""The tiering rubric (dossier 06 B.3 l.553-574): materiality, complexity, the 4 by 4 matrix, the override and the
sensitivity of the tier to the rubric's readings (U15 research addendum D2).

Every cut point, grade and aggregation is a **convention** (:mod:`riskvalidation.governance.conventions`), an
argument with its default. The mean of the complexity scores is rounded half up in integer arithmetic, ``(2 * sum
+ n) // (2 * n)``, so no floating point decides a boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

from riskvalidation.governance.conventions import EXPOSURE_BAND_CUTS

AGGREGATIONS = ("mean", "maximum")
ROUNDINGS = ("half-up",)
READINGS = (("A", "literal", "mean"), ("B", "graded", "mean"), ("C", "literal", "maximum"), ("D", "graded", "maximum"))


def _score(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
        raise ValueError(f"{name} must be an integer from 1 to 4, got {value!r}")
    return value


def exposure_band(share: float | None, cuts: tuple[float, float, float] = EXPOSURE_BAND_CUTS) -> int:
    """The exposure band 1 to 4 of a share of the stated balance (``None``: no balance exists, the floor)."""
    if len(cuts) != 3:
        raise ValueError(f"cuts must hold three cut points, got {len(cuts)}")
    hi, mid, lo = cuts
    if not (1.0 >= hi > mid > lo > 0.0):
        raise ValueError(f"cuts must be decreasing in (0, 1], got {tuple(cuts)}")
    if share is None:
        return 1
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must lie in [0, 1], got {share}")
    return 4 if share >= hi else 3 if share >= mid else 2 if share >= lo else 1


def materiality(band: int, regulatory_use: int, decision_criticality: int) -> int:
    """M, the maximum of the exposure band, the regulatory use and the decision criticality (06 l.563-566)."""
    return max(
        _score(band, "band"),
        _score(regulatory_use, "regulatory_use"),
        _score(decision_criticality, "decision_criticality"),
    )


def complexity(scores: Sequence[int], aggregation: str = "mean", rounding: str = "half-up") -> int:
    """C from the scored factors (an unscored factor is left out of ``scores``, never counted as 1)."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding must be one of {ROUNDINGS}, got {rounding!r}")
    values = [_score(s, "a complexity score") for s in scores]
    if not values:
        raise ValueError("complexity needs at least one scored factor")
    if aggregation == "maximum":
        return max(values)
    return (2 * sum(values) + len(values)) // (2 * len(values))


def matrix_tier(m: int, c: int) -> int:
    """The tier 1 to 4 (1 is the highest) of the 4 by 4 matrix (06 l.571-574)."""
    _score(m, "m")
    _score(c, "c")
    if m == 4 or (m == 3 and c >= 3):
        return 1
    if m == 3 or (m == 2 and c >= 3):
        return 2
    if m == 2 or (m == 1 and c >= 3):
        return 3
    return 4


def apply_override(computed_tier: int, override: tuple[int, str, str] | None = None) -> int:
    """The final tier: the computed one, or the override's ``to_tier``; an override never goes without a rationale
    and an approver.
    """
    _score(computed_tier, "computed_tier")
    if override is None:
        return computed_tier
    to_tier, rationale, approver = override
    if not str(rationale).strip() or not str(approver).strip():
        raise ValueError("an override needs a rationale and an approver")
    return _score(to_tier, "the override's tier")


def tier_sensitivity(
    share: float | None,
    regulatory_use_literal: int,
    regulatory_use_graded: int,
    decision_criticality: int,
    scores: Sequence[int],
    cuts: tuple[float, float, float] = EXPOSURE_BAND_CUTS,
) -> tuple[tuple[str, str, str, int, int, int], ...]:
    """The tier under the four readings A (literal, mean), B (graded, mean), C (literal, maximum) and D (graded,
    maximum).

    Returns four rows (reading, regulatory use, complexity aggregation, M, C, tier). B is the adopted reading
    (W15-3). The literal regulatory use is never below the graded one.
    """
    if _score(regulatory_use_literal, "regulatory_use_literal") < _score(
        regulatory_use_graded, "regulatory_use_graded"
    ):
        raise ValueError(
            f"the literal regulatory use {regulatory_use_literal} is below the graded {regulatory_use_graded}"
        )
    band = exposure_band(share, cuts)
    use = {"literal": regulatory_use_literal, "graded": regulatory_use_graded}
    rows = []
    for name, reg, agg in READINGS:
        m = materiality(band, use[reg], decision_criticality)
        c = complexity(scores, aggregation=agg)
        rows.append((name, reg, agg, m, c, matrix_tier(m, c)))
    return tuple(rows)


def tier_counts(tiers: Sequence[int]) -> tuple[int, int, int, int]:
    """The number of entries in tiers 1, 2, 3 and 4."""
    counts = [0, 0, 0, 0]
    for t in tiers:
        counts[_score(t, "a tier") - 1] += 1
    return counts[0], counts[1], counts[2], counts[3]
