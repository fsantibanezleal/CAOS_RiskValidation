"""The Basel output floor (RBC20.4, RBC20.11 to RBC20.12, RBC90.1 to RBC90.2; dossier 01 section C.1).

    RWA = max(RWA_nominated, f_t RWA_SA-only),   f_t from the RBC90.1 Table 1 phase-in, 72.5 percent fully phased in.

The floor base takes standardised approaches only (RBC20.11 to RBC20.12: the IRB approach, SEC-IRBA, the internal models
approach for market risk and the VaR models approach to CCR are excluded), which is the caller's `rwa_sa_only`. The
optional RBC90.2 national cap limits the increase from the floor to a share (25 percent) of the pre-floor RWA during
the phase-in. The phase-in dates are the BCBS ones (RBC90 in force 2023-01-01, published 2020-03-27); jurisdictional
dates differ (dossier 01 L) and are the caller's factor. RBC20 is in force 2023-01-01 (published 2020-11-26). The
engine states no compliance claim.
"""

from __future__ import annotations

import numbers

from riskvalidation.regulatory.standardised import RegulatoryParameter, _nonnegative

RBC_IN_FORCE = "2023-01-01"
FULLY_PHASED_IN_YEAR = 2028
#: RBC90.1 Table 1: the factor applied from 1 January of the year
PHASE_IN = {2023: 0.5, 2024: 0.55, 2025: 0.6, 2026: 0.65, 2027: 0.7, 2028: 0.725}


def phase_in_factor(year: int) -> RegulatoryParameter:
    """RBC90.1 Table 1: 0.50 in 2023, 0.55, 0.60, 0.65, 0.70, and 0.725 from 2028; a year before 2023 is refused."""
    if isinstance(year, bool) or not isinstance(year, numbers.Integral):
        raise ValueError(f"year must be an integer calendar year, got {year!r} (RBC90.1 Table 1, RBC90.1)")
    y = int(year)
    if y < min(PHASE_IN):
        raise ValueError(f"year {y} is before the first phase-in year {min(PHASE_IN)} of RBC90.1 Table 1 (RBC90.1)")
    return RegulatoryParameter(PHASE_IN[min(y, FULLY_PHASED_IN_YEAR)], "RBC90.1", "RBC90.1 Table 1", RBC_IN_FORCE)


def floored_rwa(rwa_nominated: float, rwa_sa_only: float, factor: float, cap_share: float | None = None) -> float:
    """max(rwa_nominated, factor x rwa_sa_only), at most rwa_nominated x (1 + cap_share) when a cap is given
    (RBC90.2)."""
    nominated = _nonnegative("rwa_nominated", rwa_nominated)
    standardised = _nonnegative("rwa_sa_only", rwa_sa_only)
    f = _nonnegative("factor", factor)
    floored = max(nominated, f * standardised)
    if cap_share is None:
        return floored
    cap = _nonnegative("cap_share", cap_share)
    return min(floored, nominated * (1.0 + cap))
