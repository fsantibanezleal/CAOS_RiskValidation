"""The Basel standardised credit risk weights (CRE20, in force 2023-01-01, published 2025-06-10) as data with their
paragraph.

Every function returns a `RegulatoryParameter`: the value (a risk weight, a credit conversion factor, a phase-in
factor or a multiplier, as a decimal) with its paragraph, its table and the in-force date of the chapter version, so a
view can show where each number comes from (dossier 01 M.3 item 8). A key outside a table raises `ValueError` naming
the table and the paragraph; the engine never returns a placeholder weight. The tables are transcribed from dossier 01
section C (CRE20.7 Table 1, CRE20.9 Table 2, CRE20.18 Table 6, CRE20.21 Table 7, CRE20.43 Table 10, CRE20.44 to
CRE20.47, CRE20.57 to CRE20.60, CRE20.68, CRE20.82 to CRE20.87 Tables 11 to 14, CRE20.92, CRE20.94 to CRE20.101,
CRE20.104 to CRE20.107). The engine computes and states no compliance claim. Rating buckets are the labels of the
tables; mapping an external rating to a bucket is the caller's (CRE21). A further CRE20 version in force 2028-01-01 is
not compared.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

#: the in-force date of the CRE20 version transcribed here (published 2025-06-10)
CRE20_IN_FORCE = "2023-01-01"

SOVEREIGN_BUCKETS = ("AAA to AA-", "A+ to A-", "BBB+ to BBB-", "BB+ to B-", "below B-", "unrated")
BANK_BUCKETS = SOVEREIGN_BUCKETS[:5]
CORPORATE_BUCKETS = ("AAA to AA-", "A+ to A-", "BBB+ to BBB-", "BB+ to BB-", "below BB-", "unrated")


@dataclass(frozen=True)
class RegulatoryParameter:
    """A regulatory number with where it comes from."""

    value: float  # a risk weight, a credit conversion factor, a phase-in factor or a multiplier, as a decimal
    paragraph: str  # for example "CRE20.7"
    table: str  # for example "CRE20 Table 1"
    in_force: str  # ISO date of the chapter version


RiskWeight = RegulatoryParameter  # alias: the class also carries the CCF (CRE20.94) and the phase-in factor (RBC90.1)


def _p(value: float, paragraph: str, table: str, in_force: str = CRE20_IN_FORCE) -> RegulatoryParameter:
    return RegulatoryParameter(float(value), paragraph, table, in_force)


def _refuse(name: str, got: object, table: str, paragraph: str) -> ValueError:
    return ValueError(f"{name} {got!r} is not a key of {table} ({paragraph})")


def _finite(name: str, x: object) -> float:
    """A finite real number, or `ValueError` naming the field (booleans, strings and None are refused)."""
    if isinstance(x, (bool, str)) or x is None:
        raise ValueError(f"{name} must be a finite number, got {x!r}")
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {x!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def _nonnegative(name: str, x: object) -> float:
    v = _finite(name, x)
    if v < 0.0:
        raise ValueError(f"{name} must be at least 0, got {x!r}")
    return v


# CRE20.7 Table 1
_SOVEREIGN = dict(zip(SOVEREIGN_BUCKETS, (0.0, 0.2, 0.5, 1.0, 1.5, 1.0), strict=True))
# CRE20.9 Table 2 (export credit agency scores 0 to 7)
_ECA = dict(zip(range(8), (0.0, 0.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.5), strict=True))
# CRE20.18 Table 6: ECRA base and short-term claims (the unrated column is the SCRA, so it is not here)
_ECRA = dict(zip(BANK_BUCKETS, (0.2, 0.3, 0.5, 1.0, 1.5), strict=True))
_ECRA_SHORT = dict(zip(BANK_BUCKETS, (0.2, 0.2, 0.2, 0.5, 1.5), strict=True))
# CRE20.21 Table 7: SCRA grades A, B and C, base and short-term
_SCRA = {"A": 0.4, "B": 0.75, "C": 1.5}
_SCRA_SHORT = {"A": 0.2, "B": 0.5, "C": 1.5}
# CRE20.43 Table 10
_CORPORATE = dict(zip(CORPORATE_BUCKETS, (0.2, 0.5, 0.75, 1.0, 1.5, 1.0), strict=True))
# CRE20.68
_RETAIL = {"regulatory": 0.75, "transactor": 0.45, "other": 1.0}
# CRE20.57 to CRE20.60
_EQUITY = {"equity": 2.5, "speculative-unlisted": 4.0, "legislated-programme": 1.0, "subordinated-debt": 1.5}
# CRE20.82 Table 11 and CRE20.84 Table 12: upper edge of the LTV band (inclusive) and the weight; the last band is open
_LTV_EDGES = (0.5, 0.6, 0.8, 0.9, 1.0)
_RESIDENTIAL = (0.2, 0.25, 0.3, 0.4, 0.5, 0.7)
_RESIDENTIAL_CFD = (0.3, 0.35, 0.45, 0.6, 0.75, 1.05)
# CRE20.87 Table 14 (cash-flow dependent commercial real estate): edges 0.6 and 0.8
_COMMERCIAL_CFD = ((0.6, 0.7), (0.8, 0.9), (math.inf, 1.1))
# CRE20.94 to CRE20.101
_CCF = {
    "direct-credit-substitute": 1.0,
    "repo": 1.0,
    "securities-lending": 1.0,
    "forward-purchase": 1.0,
    "nif-ruf": 0.5,
    "transaction-related-contingent": 0.5,
    "commitment": 0.4,
    "trade-letter-of-credit": 0.2,
    "unconditionally-cancellable": 0.1,
}
CURRENCY_MISMATCH_MULTIPLIER = 1.5  # CRE20.92
CURRENCY_MISMATCH_CAP = 1.5  # CRE20.92: the adjusted weight is at most 150 percent
LOAN_SPLITTING_WEIGHT = 0.2  # CRE20.83: up to 55 percent of the property value
LOAN_SPLITTING_SHARE = 0.55


def sovereign_risk_weight(rating: str) -> RegulatoryParameter:
    """CRE20.7 Table 1: sovereigns and central banks by rating bucket."""
    if rating not in _SOVEREIGN:
        raise _refuse("rating", rating, "CRE20 Table 1", "CRE20.7")
    return _p(_SOVEREIGN[rating], "CRE20.7", "CRE20 Table 1")


def eca_risk_weight(score: int) -> RegulatoryParameter:
    """CRE20.9 Table 2: sovereigns scored by an export credit agency, scores 0 to 7."""
    if isinstance(score, bool) or not isinstance(score, numbers.Integral) or int(score) not in _ECA:
        raise ValueError(f"score {score!r} must be an integer 0 to 7 (CRE20 Table 2, CRE20.9)")
    return _p(_ECA[int(score)], "CRE20.9", "CRE20 Table 2")


def bank_risk_weight_ecra(rating: str, short_term: bool = False) -> RegulatoryParameter:
    """CRE20.18 Table 6: banks under the external credit risk assessment approach; `unrated` belongs to the SCRA."""
    if rating == "unrated":
        raise ValueError(
            "rating 'unrated' is not in CRE20 Table 6 (CRE20.18): the SCRA applies (bank_risk_weight_scra)"
        )
    table = _ECRA_SHORT if short_term else _ECRA
    if rating not in table:
        raise _refuse("rating", rating, "CRE20 Table 6", "CRE20.18")
    return _p(table[rating], "CRE20.18", "CRE20 Table 6")


def bank_risk_weight_scra(grade: str, short_term: bool = False) -> RegulatoryParameter:
    """CRE20.21 Table 7: banks under the standardised credit risk assessment approach, grades A, B and C."""
    table = _SCRA_SHORT if short_term else _SCRA
    if grade not in table:
        raise _refuse("grade", grade, "CRE20 Table 7", "CRE20.21")
    return _p(table[grade], "CRE20.21", "CRE20 Table 7")


def corporate_risk_weight(
    rating: str,
    sme: bool = False,
    investment_grade_unrated: bool = False,
    regulatory_retail_sme: bool = False,
) -> RegulatoryParameter:
    """CRE20.43 Table 10 by rating bucket; for an unrated corporate, the CRE20.44 to CRE20.47 alternatives by flag.

    The three flags describe an unrated corporate (investment grade where ratings are not allowed, an SME, a
    regulatory retail SME): a flag on a rated bucket and two flags at once are refused.
    """
    flags = {
        "sme": sme,
        "investment_grade_unrated": investment_grade_unrated,
        "regulatory_retail_sme": regulatory_retail_sme,
    }
    raised = [name for name, on in flags.items() if on]
    if raised:
        if rating != "unrated":
            raise ValueError(
                f"{raised[0]} applies to an unrated corporate only (CRE20.44 to CRE20.47), rating is {rating!r}"
            )
        if len(raised) > 1:
            raise ValueError(f"{raised[0]} and {raised[1]} cannot both be set (CRE20.44 to CRE20.47)")
        if sme:
            return _p(0.85, "CRE20.47", "CRE20.47")
        if regulatory_retail_sme:
            return _p(0.75, "CRE20.47", "CRE20.47")
        return _p(0.65, "CRE20.44", "CRE20.44 and CRE20.46")
    if rating not in _CORPORATE:
        raise _refuse("rating", rating, "CRE20 Table 10", "CRE20.43")
    return _p(_CORPORATE[rating], "CRE20.43", "CRE20 Table 10")


def retail_risk_weight(kind: str) -> RegulatoryParameter:
    """CRE20.68: regulatory retail 75 percent, transactors 45 percent, other retail 100 percent."""
    if kind not in _RETAIL:
        raise _refuse("kind", kind, "CRE20.68", "CRE20.68")
    return _p(_RETAIL[kind], "CRE20.68", "CRE20.68")


def equity_risk_weight(kind: str) -> RegulatoryParameter:
    """CRE20.57 to CRE20.60: equity, speculative unlisted equity, legislated programmes and subordinated debt."""
    if kind not in _EQUITY:
        raise _refuse("kind", kind, "CRE20.57 to CRE20.60", "CRE20.57")
    return _p(_EQUITY[kind], "CRE20.57", "CRE20.57 to CRE20.60")


def _band(ltv: float) -> int:
    for i, edge in enumerate(_LTV_EDGES):
        if ltv <= edge:
            return i
    return len(_LTV_EDGES)


def residential_re_risk_weight(
    ltv: float, cash_flow_dependent: bool = False, loan_splitting: bool = False
) -> RegulatoryParameter:
    """CRE20.82 Table 11 (and CRE20.84 Table 12 when cash-flow dependent) by LTV band; CRE20.83 loan splitting.

    The upper edge of a band belongs to it (LTV exactly 0.8 gives 0.3). With loan splitting the weight is 0.2 for the
    part of the loan up to 55 percent of the property value; splitting the loan is the caller's.
    """
    value = _nonnegative("ltv", ltv)
    if loan_splitting:
        if cash_flow_dependent:
            raise ValueError("loan_splitting applies to exposures that are not cash-flow dependent (CRE20.83)")
        return _p(LOAN_SPLITTING_WEIGHT, "CRE20.83", "CRE20.83")
    band = _band(value)
    if cash_flow_dependent:
        return _p(_RESIDENTIAL_CFD[band], "CRE20.84", "CRE20 Table 12")
    return _p(_RESIDENTIAL[band], "CRE20.82", "CRE20 Table 11")


def commercial_re_risk_weight(
    ltv: float, cash_flow_dependent: bool, counterparty_weight: float | None = None
) -> RegulatoryParameter:
    """CRE20.85 Table 13 (not cash-flow dependent) and CRE20.87 Table 14 (cash-flow dependent) by LTV."""
    value = _nonnegative("ltv", ltv)
    if cash_flow_dependent:
        for edge, weight in _COMMERCIAL_CFD:
            if value <= edge:
                return _p(weight, "CRE20.87", "CRE20 Table 14")
    if counterparty_weight is None:
        raise ValueError(
            "counterparty_weight is required when the exposure is not cash-flow dependent (CRE20 Table 13, CRE20.85)"
        )
    cp = _nonnegative("counterparty_weight", counterparty_weight)
    return _p(min(0.6, cp) if value <= 0.6 else cp, "CRE20.85", "CRE20 Table 13")


def apply_currency_mismatch(weight: float) -> RegulatoryParameter:
    """CRE20.92: 1.5 times the unadjusted weight, at most 1.5 (150 percent); the result is the adjusted weight."""
    w = _nonnegative("weight", weight)
    return _p(min(CURRENCY_MISMATCH_MULTIPLIER * w, CURRENCY_MISMATCH_CAP), "CRE20.92", "CRE20.92")


def credit_conversion_factor(kind: str) -> RegulatoryParameter:
    """CRE20.94 to CRE20.101: the credit conversion factor of an off-balance-sheet item (the `value` is the factor)."""
    if kind not in _CCF:
        raise _refuse("kind", kind, "CRE20.94 to CRE20.101", "CRE20.94")
    return _p(_CCF[kind], "CRE20.94", "CRE20.94 to CRE20.101")


def defaulted_risk_weight(
    specific_provision_share: float, residential_non_cash_flow_dependent: bool = False
) -> RegulatoryParameter:
    """CRE20.104 to CRE20.107: 150 percent below a specific provision share of 20 percent, 100 percent from it."""
    share = _nonnegative("specific_provision_share", specific_provision_share)
    if residential_non_cash_flow_dependent:
        return _p(1.0, "CRE20.104", "CRE20.104 to CRE20.107")
    return _p(1.5 if share < 0.2 else 1.0, "CRE20.104", "CRE20.104 to CRE20.107")


_CLASSES = (
    "sovereign",
    "eca",
    "bank-ecra",
    "bank-scra",
    "corporate",
    "retail",
    "equity",
    "residential-re",
    "commercial-re",
    "defaulted",
    "ccf",
    "currency-mismatch",
)
_OPTION_USERS = {
    "short_term": ("bank-ecra", "bank-scra"),
    "cash_flow_dependent": ("residential-re", "commercial-re"),
    "counterparty_weight": ("commercial-re",),
}
_OPTION_DEFAULTS = {"short_term": False, "cash_flow_dependent": False, "counterparty_weight": None}
_CORPORATE_FLAGS = {
    "investment-grade-unrated": "investment_grade_unrated",
    "sme-unrated": "sme",
    "regulatory-retail-sme": "regulatory_retail_sme",
}
_DEFAULTED_KEYS = ("provision-below-20pct", "provision-at-least-20pct", "residential-non-cash-flow-dependent")


def _ltv_key(key: str) -> float:
    try:
        return float(key)
    except (TypeError, ValueError):
        raise ValueError(f"key {key!r} must be the LTV as a string of the number") from None


def lookup(exposure_class: str, key: str, **options: object) -> RegulatoryParameter:
    """The `RegulatoryParameter` of the function the class dispatches to.

    `exposure_class` is one of sovereign, eca, bank-ecra, bank-scra, corporate, retail, equity, residential-re,
    commercial-re, defaulted, ccf, currency-mismatch; `key` the rating bucket, the grade, the kind, an ECA score or an
    LTV as a string, or for currency-mismatch `multiplier` or `cap`. The options are `short_term`,
    `cash_flow_dependent` and `counterparty_weight`; an option the class does not use must be left at its default, and
    an unknown option is refused.
    """
    if exposure_class not in _CLASSES:
        raise ValueError(f"exposure_class {exposure_class!r} is not one of {', '.join(_CLASSES)}")
    for name, got in options.items():
        if name not in _OPTION_DEFAULTS:
            raise ValueError(f"option {name!r} is not one of short_term, cash_flow_dependent, counterparty_weight")
        if exposure_class not in _OPTION_USERS[name] and got != _OPTION_DEFAULTS[name]:
            raise ValueError(f"{name} does not apply to exposure_class {exposure_class!r}")
    short_term = bool(options.get("short_term", False))
    cash_flow_dependent = bool(options.get("cash_flow_dependent", False))
    counterparty_weight = options.get("counterparty_weight")
    if exposure_class == "sovereign":
        return sovereign_risk_weight(key)
    if exposure_class == "eca":
        try:
            score = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"key {key!r} must be an ECA score as a string of the integer 0 to 7") from None
        return eca_risk_weight(score)
    if exposure_class == "bank-ecra":
        return bank_risk_weight_ecra(key, short_term)
    if exposure_class == "bank-scra":
        return bank_risk_weight_scra(key, short_term)
    if exposure_class == "corporate":
        if key in _CORPORATE_FLAGS:
            return corporate_risk_weight("unrated", **{_CORPORATE_FLAGS[key]: True})
        return corporate_risk_weight(key)
    if exposure_class == "retail":
        return retail_risk_weight(key)
    if exposure_class == "equity":
        return equity_risk_weight(key)
    if exposure_class == "residential-re":
        return residential_re_risk_weight(_ltv_key(key), cash_flow_dependent)
    if exposure_class == "commercial-re":
        return commercial_re_risk_weight(_ltv_key(key), cash_flow_dependent, counterparty_weight)  # type: ignore[arg-type]
    if exposure_class == "defaulted":
        if key not in _DEFAULTED_KEYS:
            raise _refuse("key", key, "CRE20.104 to CRE20.107", "CRE20.104")
        if key == "residential-non-cash-flow-dependent":
            return defaulted_risk_weight(0.0, residential_non_cash_flow_dependent=True)
        return defaulted_risk_weight(0.0 if key == "provision-below-20pct" else 0.2)
    if exposure_class == "ccf":
        return credit_conversion_factor(key)
    if key == "multiplier":
        return _p(CURRENCY_MISMATCH_MULTIPLIER, "CRE20.92", "CRE20.92")
    if key == "cap":
        return _p(CURRENCY_MISMATCH_CAP, "CRE20.92", "CRE20.92")
    raise _refuse("key", key, "CRE20.92", "CRE20.92")

__all__ = [
    "RegulatoryParameter",
    "apply_currency_mismatch",
    "bank_risk_weight_ecra",
    "bank_risk_weight_scra",
    "commercial_re_risk_weight",
    "corporate_risk_weight",
    "credit_conversion_factor",
    "defaulted_risk_weight",
    "eca_risk_weight",
    "equity_risk_weight",
    "lookup",
    "residential_re_risk_weight",
    "retail_risk_weight",
    "sovereign_risk_weight",
]
