"""Run-off: constant daily compounding and its inverse, the contractual roll-off rules and the transcribed parameter
sets as module data (`LIST_PARAMETERS`, `TABLE_A1`, `SEGMENTS`).

Sources: dossier 05 E.1 line 237 (the open-maturity deposits of the ECB LiST 2019 run off at a constant daily
compounding
rate on relevant days, cumulative outflow `1 - (1 - q)^n`) and E.2 (BCBS d555 Table A.1); the U11 research addendum L1
(the ECB methodological note of February 2019, Tables 3.2 and 3.3, paragraphs 94 to 126) and L2.1. Contract
`engine-liquidity-api` 2.0.0, module runoff.

The arithmetic is that of `provisions.term_structure.from_hazard`, considered and not reused: that function refuses a
rate of exactly 0 or 1, both legitimate for a segment, and indexes months. Money is a float in USD thousands (the
engine never converts), rates are decimals per relevant day, days are integers counted from 1. The relevant days of
the note are the 2019 TARGET2 calendar (125 in the six months); a ladder on another calendar (a US business-day grid)
is a stated convention and every LiST cumulative figure says which calendar it is on (UNVERIFIED as a calendar
choice, the contract's open item A2).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from riskvalidation.liquidity._checks import check_nonnegative_stock, check_unit_share

LIST_SOURCE = (
    "ECB Banking Supervision, Sensitivity analysis of liquidity risk, stress test 2019, methodological note "
    "(2019-02-06), Table 3.2 (contractual roll-off, para 94) and Table 3.3 (open-maturity daily compounding rates, "
    "para 107), committed-facility drawdown (para 121); relevant days are the 2019 TARGET2 calendar "
    "(para 16, 125 relevant days in six months, para 107), "
    "https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.ECB_sensitivity_analysis_of_liquidity_risk-"
    "stress_test_2019-methodological_note_20190206~6771e88926.en.pdf"
)
LIST_RELEVANT_DAYS = 125  # relevant days in the six months after the 2019 reference date (note para 107, Table 3.1)
LIST_FACILITY_DAYS = 63   # relevant days in the first three months, the drawdown window (note para 121, Table 2.4)

_FULL = 1.0

# Table 3.3, open-maturity items, the daily compounding rate on the stock of the day before, relevant days only.
# Column order of the note: Baseline, Adverse, Extreme.
_OPEN_MATURITY: dict[str, tuple[float, float, float]] = {
    "open-repos": (_FULL, _FULL, _FULL),
    "stable-retail": (0.0, 0.00103258, 0.00161494),
    "other-retail": (0.0, 0.00373218, 0.00440873),
    "operational": (0.0, 0.00514464, 0.00750509),
    "non-operational-credit-institutions": (_FULL, _FULL, _FULL),
    "non-operational-other-financial": (_FULL, _FULL, _FULL),
    "non-operational-non-financial-corporates": (0.0, 0.00688541, 0.01065229),
    "non-operational-others": (0.0, 0.00688541, 0.01065229),
    "other-transactions": (_FULL, _FULL, _FULL),
}
# Table 3.2, contractual items, the outflow rate applied to each flow when it matures (constant across maturities).
_CONTRACTUAL_OUTFLOW: dict[str, tuple[float, float, float]] = {
    "securities-issued": (_FULL, _FULL, _FULL),
    "secured-funding": (_FULL, _FULL, _FULL),
    "stable-retail": (0.0, 0.18, 0.27),
    "other-retail": (0.0, 0.39, 0.48),
    "operational": (0.0, 0.37, 0.50),
    "non-operational-credit-institutions": (_FULL, _FULL, _FULL),
    "non-operational-other-financial": (_FULL, _FULL, _FULL),
    "non-operational-non-financial-corporates": (0.0, 0.52, 0.76),
    "non-operational-others": (0.0, 0.52, 0.76),
    "fx-swaps": (_FULL, _FULL, _FULL),
    "derivatives": (_FULL, _FULL, _FULL),
    "other-cash-outflows": (_FULL, _FULL, _FULL),
}
# Table 3.2, contractual inflows: 1.0 for secured lending, loans to credit institutions and other financial customers,
# FX swaps, derivatives, own paper and other inflows; 0 for loans to retail, non-financial corporates and others.
_CONTRACTUAL_INFLOW: dict[str, tuple[float, float, float]] = {
    "secured-lending": (_FULL, _FULL, _FULL),
    "loans-retail": (0.0, 0.0, 0.0),
    "loans-non-financial-corporates": (0.0, 0.0, 0.0),
    "loans-credit-institutions": (_FULL, _FULL, _FULL),
    "loans-other-financial": (_FULL, _FULL, _FULL),
    "loans-others": (0.0, 0.0, 0.0),
    "fx-swaps": (_FULL, _FULL, _FULL),
    "derivatives": (_FULL, _FULL, _FULL),
    "own-paper": (_FULL, _FULL, _FULL),
    "other-inflows": (_FULL, _FULL, _FULL),
}
# Para 121: committed facilities are drawn only in the first three months, at a daily compounding rate on the undrawn
# stock; the Baseline has no outflow from committed facilities (para 33).
_DRAWDOWN: dict[str, tuple[float, float, float]] = {
    "committed-credit-facilities": (0.0, 0.00202704, 0.00257634),
    "liquidity-facilities": (0.0, 0.01443904, 0.02176434),
}


def _by_severity(block: dict[str, tuple[float, float, float]], column: int) -> dict[str, float]:
    return {key: values[column] for key, values in block.items()}


LIST_PARAMETERS: Mapping[str, Mapping[str, Mapping[str, float]]] = MappingProxyType({
    severity: MappingProxyType({
        "open_maturity": MappingProxyType(_by_severity(_OPEN_MATURITY, column)),
        "contractual_outflow": MappingProxyType(_by_severity(_CONTRACTUAL_OUTFLOW, column)),
        "contractual_inflow": MappingProxyType(_by_severity(_CONTRACTUAL_INFLOW, column)),
        "drawdown": MappingProxyType(_by_severity(_DRAWDOWN, column)),
    })
    for column, severity in enumerate(("baseline", "adverse", "extreme"))
})

SEGMENTS = ("retail-insured", "retail-uninsured", "operational", "non-operational-corporate", "financial")

TABLE_A1_SOURCE = (
    "BCBS, Report on the 2023 banking turmoil (d555, 2023-10), Box A, Table A.1, "
    "https://www.bis.org/bcbs/publ/d555.htm (dossier 05 E.2 lines 248 to 259; addendum L2.1). Footnote (a): "
    "First Republic's outflow excludes 30 bn USD of deposits placed by banks in Q1 2023; SVB's includes the "
    "expected outflow of management for 10 March."
)
# The tenth row of Table A.1: LCR deposit outflow rates, 3 percent to 40 percent over 30 days (low, high, days). The
# LCR rate table itself belongs to `regulatory.lcr_nsfr`.
TABLE_A1_LCR_RANGE = (0.03, 0.40, 30)


def cumulative_outflow_constant(q: np.ndarray | float, n: np.ndarray | int) -> np.ndarray | float:
    """The cumulative outflow share after `n` relevant days at the constant daily rate `q`: `1 - (1 - q)^n`.

    `q` is a decimal per relevant day in `[0, 1]`, `n` a non-negative integer number of days. Zero at `n = 0` and at
    `q = 0`, one at `q = 1` for `n >= 1`, non-decreasing in `n` and in `q`; NumPy broadcasting. A scalar call returns
    a float. Golden: `q = 0.00103258`, `n = 125` gives `0.121148` (dossier 05 line 237)."""
    scalar = np.ndim(q) == 0 and np.ndim(n) == 0
    qa = check_unit_share(q, "q")
    na = _days(n, "n", minimum=0)
    out = 1.0 - np.power(1.0 - qa, na)
    return float(out[0]) if scalar else out


def cumulative_outflow_path(daily_rates: np.ndarray) -> np.ndarray:
    """The cumulative outflow shares of a path of daily rates (a rate that changes with the stress state): `1 -
    cumprod(1 - q_d)`, shape `(H,)`. Non-decreasing; a rate of 1 on any day makes every later value 1."""
    q = check_unit_share(daily_rates, "daily_rates")
    if q.ndim != 1:
        raise ValueError(f"daily_rates: must be one-dimensional, got shape {q.shape}")
    return 1.0 - np.cumprod(1.0 - q)


def daily_rate_from_cumulative(share: np.ndarray | float, days: np.ndarray | int) -> np.ndarray | float:
    """The daily rate that compounds to the cumulative outflow `share` in `days` days: `1 - (1 - F)^(1 / n)`.

    The inverse of `cumulative_outflow_constant`: the round trip returns `share` to 1e-12; a share of 1 gives 1.
    Golden (recomputed from Table A.1): SVB `0.85` in `2` days gives `0.612702`."""
    scalar = np.ndim(share) == 0 and np.ndim(days) == 0
    f = check_unit_share(share, "share")
    d = _days(days, "days", minimum=1)
    out = 1.0 - np.power(1.0 - f, 1.0 / d)
    return float(out[0]) if scalar else out


def contractual_rolloff(maturing: np.ndarray, outflow_rate: float) -> np.ndarray:
    """The outflow of a contractual class: `rate x maturing`, day by day (LiST Table 3.2, constant across maturities,
    para 94). The rolled-over remainder `(1 - rate) x maturing` stays past the horizon and is not returned.
    Golden: `[100, 50]` at `0.18` gives `[18, 9]`."""
    m = _flows(maturing, "maturing")
    rate = float(check_unit_share(outflow_rate, "outflow_rate")[0])
    return rate * m


def financial_no_rollover(maturing: np.ndarray) -> np.ndarray:
    """Maturing interbank and other financial deposits do not roll over: the outflow equals the maturing flow (LiST
    para 98, rate 1.0). The input is returned as a new array."""
    return _flows(maturing, "maturing")


def _flows(x: Any, name: str) -> np.ndarray:
    a = check_nonnegative_stock(x, name)
    if a.ndim != 1:
        raise ValueError(f"{name}: must be one-dimensional, got shape {a.shape}")
    return a


def _days(x: Any, name: str, *, minimum: int) -> np.ndarray:
    a = np.atleast_1d(np.array(x, dtype=float))
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be finite integers, got {x!r}")
    if np.any(a != np.floor(a)):
        raise ValueError(f"{name}: must be integers, got {x!r}")
    if np.any(a < minimum):
        raise ValueError(f"{name}: must be at least {minimum}, got {x!r}")
    return a


@dataclass(frozen=True)
class ObservedRun:
    """One row of BCBS d555 Table A.1: a deposit outflow share observed over a number of days. Pass `ObservedRun` rows
    to `scenarios.fast_run_scenario`: it recognises a 2023 row of a scored calibration by `year`, whereas a plain
    `(share, days)` pair is matched to `TABLE_A1` by exact float equality, so a pair that differs by rounding passes
    as a pre-2023 row."""

    row_id: str
    bank: str
    year: int
    outflow_share: float
    days: int
    source: str = TABLE_A1_SOURCE

    @property
    def implied_daily_rate(self) -> float:
        """`1 - (1 - share)^(1 / days)`, recomputed here, not a dossier value."""
        return float(daily_rate_from_cumulative(self.outflow_share, self.days))


TABLE_A1: tuple[ObservedRun, ...] = (
    ObservedRun("svb-2023", "SVB", 2023, 0.85, 2),
    ObservedRun("first-republic-2023", "First Republic", 2023, 0.57, 90),
    ObservedRun("svb-uk-2023", "SVB UK", 2023, 0.30, 1),
    ObservedRun("credit-suisse-2023", "Credit Suisse", 2023, 0.21, 90),
    ObservedRun("signature-2023", "Signature", 2023, 0.20, 1),
    ObservedRun("icesave-2008", "Icesave", 2008, 0.20, 75),
    ObservedRun("northern-rock-2007", "Northern Rock", 2007, 0.20, 4),
    ObservedRun("wamu-2008", "WaMu", 2008, 0.10, 10),
    ObservedRun("indymac-2008", "IndyMac", 2008, 0.08, 14),
)

__all__ = ["LIST_FACILITY_DAYS", "LIST_PARAMETERS", "LIST_RELEVANT_DAYS", "LIST_SOURCE", "ObservedRun", "SEGMENTS",
           "TABLE_A1", "TABLE_A1_LCR_RANGE", "TABLE_A1_SOURCE", "contractual_rolloff", "cumulative_outflow_constant",
           "cumulative_outflow_path", "daily_rate_from_cumulative", "financial_no_rollover"]
