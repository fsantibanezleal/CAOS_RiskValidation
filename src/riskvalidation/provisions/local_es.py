r"""Banco de Espana Circular 4/2017 Annex 9 (Anejo 9, "Analisis y cobertura del riesgo de credito"): the coverage
tables of the alternative solutions and the category-to-stage mapping.

Source: Contraste dossier 03 sections K.2 and K.3, read 2026-10-04 from Circular 4/2017 (BOE-A-2017-14334) as amended by
Circular 6/2021 of 22 December (BOE-A-2021-21666), whose tables of points 140, 144 and 172 apply from 2022-06-30.
Coverage is applied to the amount not covered by effective guarantees:
$Cobertura = c_{s,k} \max(0, EAD - \sum_g ValorAjustado_g)$.

Point numbers (144, 140) are as of Circular 6/2021. Whether they shifted after Circular 1/2025 is UNVERIFIED in the
dossier and stays a documented limitation. Not transcribed (dossier K.3): the collateral haircut table for effective
guarantees, the treatment of a doubtful risk for reasons other than arrears, the country-risk tables revised by
Circular 1/2025. The functions compute; they state no compliance claim. Percentages are decimals here: 1.9 per cent is
0.019.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Point 144 (riesgo normal; riesgo normal en vigilancia especial), decimals of the uncovered amount. Dossier 03 K.3.
_POINT_144: dict[str, tuple[float, float]] = {
    "A.1.1": (0.019, 0.30),   # Construccion y promocion inmobiliaria, incl. suelo
    "A.1.2": (0.02, 0.188),   # Construccion de obra civil
    "A.1.3": (0.006, 0.096),  # Resto de financiacion especializada
    "A.2.1": (0.006, 0.096),  # Grandes empresas
    "A.2.2": (0.011, 0.178),  # Pymes
    "A.2.3": (0.014, 0.139),  # Empresarios individuales
    "B.1.1": (0.007, 0.18),   # Vivienda habitual, LTV <= 80%
    "B.1.2": (0.007, 0.18),   # Vivienda habitual, LTV > 80%
    "B.1.3": (0.007, 0.18),   # Vivienda distinta de la habitual
    "B.2": (0.018, 0.202),    # Credito al consumo
    "B.2.1": (0.01, 0.116),   # of which credit cards
    "B.3": (0.018, 0.202),    # Otros fines
}

# Point 140 (riesgo dudoso por razon de la morosidad), decimals of the uncovered amount by age of the past-due
# amounts; columns >90d-6m, 6-9m, 9-12m, 12-15m, 15-18m, 18-21m, >21m. Dossier 03 K.3.
_ROW_A11 = (0.70, 0.75, 0.85, 0.90, 0.95, 1.0, 1.0)
_ROW_A12 = (0.55, 0.65, 0.70, 0.80, 0.95, 1.0, 1.0)
_ROW_A13 = (0.55, 0.65, 0.75, 0.90, 0.95, 1.0, 1.0)
_ROW_A22 = (0.65, 0.70, 0.75, 0.85, 0.90, 0.95, 1.0)
_ROW_A23 = (0.35, 0.45, 0.60, 0.65, 0.80, 0.95, 1.0)
_ROW_B1 = (0.45, 0.50, 0.65, 0.70, 0.85, 0.95, 1.0)  # B.1.1 / B.1.2 / B.1.3 Vivienda
_ROW_B2 = (0.55, 0.65, 0.80, 0.85, 0.95, 1.0, 1.0)   # B.2 Consumo (and B.2.1 cards), B.3 Otros fines
_POINT_140: dict[str, tuple[float, ...]] = {
    "A.1.1": _ROW_A11,
    "A.1.2": _ROW_A12,
    "A.1.3": _ROW_A13,
    "A.2.1": _ROW_A13,  # Grandes empresas: the same row as A.1.3 in the dossier
    "A.2.2": _ROW_A22,
    "A.2.3": _ROW_A23,
    "B.1.1": _ROW_B1,
    "B.1.2": _ROW_B1,
    "B.1.3": _ROW_B1,
    "B.2": _ROW_B2,
    "B.2.1": _ROW_B2,
    "B.3": _ROW_B2,
}

# Point 140 counts months; converted once at 30.4375 days per month rounded to the nearest day (Contraste's
# convention; a bank's own day count overrides it in the caller): 6, 9, 12, 15, 18, 21 months. Upper bounds are
# inclusive and the first bucket opens above 90 days.
_DAYS_PER_MONTH = 30.4375
_UPPER_DAYS = tuple(round(m * _DAYS_PER_MONTH) for m in (6, 9, 12, 15, 18, 21))  # 183, 274, 365, 457, 548, 639
_MIN_DOUBTFUL_DAYS = 90  # the doubtful buckets exist for arrears above 90 days

# Category to stage (norma 29.11; the mapping Circular 6/2021 states for reporting). Dossier 03 K.2.
_STAGE = {
    "normal": "stage1",
    "normal-special-watch": "stage2",
    "doubtful": "stage3",
    "written-off": "written-off",
}


def _check_segment(segment: str, point: str) -> None:
    if segment not in _POINT_144:
        raise ValueError(f"segment {segment!r} is outside {list(_POINT_144)} (Annex 9 {point})")


def annex9_uncovered_amount(ead: float, adjusted_guarantee_values: Sequence[float]) -> float:
    """``max(0, EAD - sum(adjusted guarantees))``: the amount not covered by effective guarantees (Annex 9)."""
    if not math.isfinite(ead) or ead < 0:
        raise ValueError(f"ead must be a non-negative amount, got {ead!r}")
    total = 0.0
    for g in adjusted_guarantee_values:
        if not math.isfinite(g) or g < 0:
            raise ValueError(f"an adjusted guarantee value must be a non-negative amount, got {g!r}")
        total += g
    return max(0.0, float(ead) - total)


def annex9_normal_coverage(segment: str, category: str) -> float:
    """Coverage decimal of point 144 for ``normal`` or ``normal-special-watch`` in a segment (from 2022-06-30).

    Segments: A.1.1, A.1.2, A.1.3, A.2.1, A.2.2, A.2.3, B.1.1, B.1.2, B.1.3, B.2, B.2.1, B.3.
    """
    _check_segment(segment, "point 144")
    normal, watch = _POINT_144[segment]
    if category == "normal":
        return normal
    if category == "normal-special-watch":
        return watch
    raise ValueError(f"category {category!r} is outside ['normal', 'normal-special-watch'] (Annex 9 point 144)")


def annex9_doubtful_coverage(segment: str, days_past_due: int) -> float:
    """Coverage decimal of point 140 for a doubtful account by the age of its oldest past-due amount, in days.

    Buckets (90, 183], (183, 274], (274, 365], (365, 457], (457, 548], (548, 639] and above 639 days (upper bounds
    inclusive; the seven columns >90d-6m to >21m). Raises ``ValueError`` for 90 days or fewer.
    """
    _check_segment(segment, "point 140")
    if not math.isfinite(days_past_due) or days_past_due <= _MIN_DOUBTFUL_DAYS:
        raise ValueError(
            f"days past due {days_past_due!r} must exceed {_MIN_DOUBTFUL_DAYS}: Annex 9 point 140 covers doubtful "
            "risk by arrears above 90 days"
        )
    row = _POINT_140[segment]
    for k, upper in enumerate(_UPPER_DAYS):
        if days_past_due <= upper:
            return row[k]
    return row[-1]


def annex9_category_to_stage(category: str) -> str:
    """``stage1``, ``stage2``, ``stage3`` or ``written-off`` for an Annex 9 category (norma 29.11, Circular 6/2021)."""
    try:
        return _STAGE[category]
    except (KeyError, TypeError):
        raise ValueError(f"category {category!r} is outside {list(_STAGE)} (Circular 4/2017 norma 29.11)") from None


def annex9_coverage(segment: str, category: str, ead: float, adjusted_guarantee_values: Sequence[float],
                    days_past_due: int | None = None) -> float:
    """Coverage amount ``c x annex9_uncovered_amount(ead, adjusted_guarantee_values)`` (Annex 9).

    For ``doubtful`` the coefficient is point 140 by ``days_past_due`` (required and above 90); otherwise point 144.
    ``written-off`` returns 0. A doubtful account for reasons other than arrears has no transcribed coverage (dossier
    03 K.3): with ``days_past_due`` missing or at most 90 this raises ``ValueError`` and the caller reports the account
    as not evaluated.
    """
    if category == "written-off":
        _check_segment(segment, "coverage")
        annex9_uncovered_amount(ead, adjusted_guarantee_values)
        return 0.0
    if category == "doubtful":
        if days_past_due is None:
            raise ValueError(
                "a doubtful account needs days_past_due above 90; a doubtful risk for reasons other than arrears has "
                "no transcribed coverage (dossier 03 K.3): report the account as not evaluated"
            )
        c = annex9_doubtful_coverage(segment, days_past_due)
    else:
        c = annex9_normal_coverage(segment, category)
    return c * annex9_uncovered_amount(ead, adjusted_guarantee_values)
