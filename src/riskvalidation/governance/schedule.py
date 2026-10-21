"""The B.8 dates (dossier 06 B.8 l.630-640): the next review, the next full revalidation and the next monitoring
date.

The frequency table is a **convention** (tier to (review, full revalidation, monitoring) in calendar months; tier
4 is an attestation). With the profile ``ecb`` the IRB minimums apply: a review interval of at most 12 months and
a full revalidation of at most 36 months, the minimum applying even where the table states none (06 l.633-636,
Delegated Regulation 2022/439 Article 11).
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping
from datetime import date

from riskvalidation.governance.conventions import FREQUENCY

PROFILES = ("default", "ecb")
ECB_REVIEW_MONTHS = 12
ECB_REVALIDATION_MONTHS = 36


def add_months(d: date, n: int) -> date:
    """``d`` plus ``n`` calendar months; the day is clamped to the end of the target month."""
    if n < 0:
        raise ValueError(f"months must not be negative, got {n}")
    year, month0 = divmod(d.year * 12 + d.month - 1 + n, 12)
    month = month0 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def next_dates(
    tier: int,
    last_validation: date,
    table: Mapping[int, tuple[int, int | None, int | None]] = FREQUENCY,
    profile: str = "default",
) -> tuple[date, date | None, date | None]:
    """(next review, next full revalidation or ``None``, next monitoring date or ``None``) from the date of the last
    validation.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown schedule profile {profile!r}; expected one of {PROFILES}")
    if tier not in table:
        raise ValueError(f"the frequency table has no row for tier {tier!r}")
    review, revalidation, monitoring = table[tier]
    if profile == "ecb":
        review = min(review, ECB_REVIEW_MONTHS)
        revalidation = min(
            revalidation if revalidation is not None else ECB_REVALIDATION_MONTHS, ECB_REVALIDATION_MONTHS
        )
    return (
        add_months(last_validation, review),
        add_months(last_validation, revalidation) if revalidation is not None else None,
        add_months(last_validation, monitoring) if monitoring is not None else None,
    )
