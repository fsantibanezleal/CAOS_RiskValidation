"""The standardised approach to CVA risk, SA-CVA (MAR50.27 to MAR50.77, in force 2023-01-01, published 2020-07-08).

The capital is the sum of the delta and vega requirements of the whole CVA portfolio including the eligible hedges
(MAR50.42): K_delta is the simple sum over six risk classes (interest rate, FX, counterparty credit spread, reference
credit spread, equity, commodity; MAR50.43) and K_vega the simple sum over five (the same without counterparty credit
spread: there is no vega for counterparty credit spread risk, MAR50.45 and MAR50.63). There is no curvature charge and
no default risk charge (MAR50.27). The risk-weighted assets are 12.5 times the capital (MAR50.1). Per risk class
(MAR50.51 to MAR50.53):

    WS_k^CVA = RW_k s_k^CVA,   WS_k^Hdg = RW_k s_k^Hdg,   WS_k = WS_k^CVA - WS_k^Hdg
    K_b = sqrt(max(0, sum_k WS_k^2 + sum_{k != l} rho_kl WS_k WS_l + R sum_k (WS_k^Hdg)^2))
    S_b = max(-K_b, min(sum_{k in b} WS_k, K_b))
    K   = m_CVA sqrt(max(0, sum_b K_b^2 + sum_{b != c} gamma_bc S_b S_c))

with R = 0.01 the hedging disallowance parameter, which sits INSIDE the square root, and m_CVA = 1 (MAR50.41). S_b is
always capped at K_b and floored at -K_b (not only when the radicand is negative, as in MAR21.4). The interest rate
delta risk factors are by currency class: the reporting currency and USD, EUR, GBP, AUD, CAD, SEK and JPY have five
tenors and inflation (Tables 3 and 4, MAR50.56), any other currency one parallel factor and inflation at 1.58 percent
(MAR50.57); the vega risk weights are flat, 100 percent and 78 percent for the large market capitalisation equity
buckets (MAR50.73). Tables 3 to 14 are module data as read-only parameters with their paragraphs.

Readings taken where the text is silent, stated here and on the page of the documentation:

* A bucket with a single risk factor (reference credit spread, equity, commodity, FX, and an interest rate bucket with
  one factor) takes no correlation term: MAR50 states no within-bucket correlation for it, so K_b is the root of WS^2
  plus the hedging disallowance.
* The large market capitalisation buckets of the equity vega weight are those of Table 11 marked large (1 to 8) and the
  qualified index bucket 12 ("Large cap, advanced economies").
* The correlation tables are symmetric (gamma_bc = gamma_cb, rho_kl = rho_lk); the printed upper triangle is the source.

Stated limits: the CVA sensitivities are the caller's: their computation from exposure models, regulatory CVA and the
prudent valuation standards (MAR50.31 to MAR50.36, CAP50) is the bank's, and the engine takes the sensitivities as
inputs; index look-through (MAR50.49, MAR50.50) is the caller's too, so an index hedge is passed as its sensitivities to
the constituents' risk factors, or to the qualified index buckets under the option of MAR50.50. The eligibility of a
hedge (MAR50.37 to MAR50.39), the supervisor's multiplier above 1 (MAR50.41; it is an argument) and the securitisation
and CTP rules are not computed. The engine computes and states no compliance claim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np

from riskvalidation.regulatory.standardised import RegulatoryParameter, _finite, _nonnegative

MAR50_IN_FORCE = "2023-01-01"


def _rp(value: float, paragraph: str, table: str) -> RegulatoryParameter:
    return RegulatoryParameter(float(value), paragraph, table, MAR50_IN_FORCE)


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


def _symmetric(upper_percent: Sequence[Sequence[float | None]]) -> np.ndarray:
    """A symmetric matrix of decimals from the printed upper triangle (the lower triangle is printed blank)."""
    n = len(upper_percent)
    m = np.zeros((n, n))
    for i, row in enumerate(upper_percent):
        for j, v in enumerate(row):
            if v is not None:
                m[i, j] = m[j, i] = v / 100.0
    return _readonly(m)


#: MAR50.53 (1): the hedging disallowance parameter
R = _rp(0.01, "MAR50.53", "MAR50.53 (1)")
#: MAR50.41: the multiplier, which a supervisor may raise
M_CVA = _rp(1.0, "MAR50.41", "MAR50.41")
#: MAR50.1: the risk-weighted assets are the capital requirement times 12.5
RWA_FACTOR = _rp(12.5, "MAR50.1", "MAR50.1")

RISK_CLASSES = ("ir", "fx", "ccs", "rcs", "eq", "com")
KINDS = ("delta", "vega")
_CLASS_NAMES = {
    "ir": "interest rate",
    "fx": "FX",
    "ccs": "counterparty credit spread",
    "rcs": "reference credit spread",
    "eq": "equity",
    "com": "commodity",
}

# --- interest rate (MAR50.54 to MAR50.58) -----------------------------------------------------------------------------
#: MAR50.56: the reporting currency and these have five tenors and inflation
SPECIFIED_CURRENCIES = ("USD", "EUR", "GBP", "AUD", "CAD", "SEK", "JPY")
IR_FACTORS_SPECIFIED = ("1y", "2y", "5y", "10y", "30y", "inflation")
IR_FACTORS_OTHER = ("parallel", "inflation")
IR_FACTORS_VEGA = ("ir_vol", "inflation_vol")
#: MAR50.55: the cross-bucket correlation of all currency pairs
IR_GAMMA = _rp(0.5, "MAR50.55", "MAR50.55")
_TABLE_3_PERCENT = (1.11, 0.93, 0.74, 0.74, 0.74, 1.11)
IR_DELTA_RISK_WEIGHTS_SPECIFIED: dict[str, RegulatoryParameter] = {
    f: _rp(w / 100.0, "MAR50.56", "MAR50.56 Table 3")
    for f, w in zip(IR_FACTORS_SPECIFIED, _TABLE_3_PERCENT, strict=True)
}
_TABLE_4_PERCENT = (
    (100.0, 91.0, 72.0, 55.0, 31.0, 40.0),
    (None, 100.0, 87.0, 72.0, 45.0, 40.0),
    (None, None, 100.0, 91.0, 68.0, 40.0),
    (None, None, None, 100.0, 83.0, 40.0),
    (None, None, None, None, 100.0, 40.0),
    (None, None, None, None, None, 100.0),
)
IR_DELTA_CORRELATION_SPECIFIED = _symmetric(_TABLE_4_PERCENT)
IR_DELTA_RISK_WEIGHT_OTHER = _rp(0.0158, "MAR50.57", "MAR50.57")
IR_DELTA_RHO_OTHER = _rp(0.4, "MAR50.57", "MAR50.57")
IR_VEGA_RISK_WEIGHT = _rp(1.0, "MAR50.58", "MAR50.58")
IR_VEGA_RHO = _rp(0.4, "MAR50.58", "MAR50.58")

# --- FX (MAR50.59 to MAR50.62) ----------------------------------------------------------------------------------------
FX_GAMMA = _rp(0.6, "MAR50.60", "MAR50.60")
FX_DELTA_RISK_WEIGHT = _rp(0.11, "MAR50.61", "MAR50.61")
FX_VEGA_RISK_WEIGHT = _rp(1.0, "MAR50.62", "MAR50.62")

# --- counterparty credit spread (MAR50.63 to MAR50.65), delta only  ---
_S_SOV = "Sovereigns including central banks, multilateral development banks"
_S_LOC = "Local government, government-backed non-financials, education and public administration"
_S_FIN = "Financials including government-backed financials"
_S_BASIC = "Basic materials, energy, industrials, agriculture, manufacturing, mining and quarrying"
_S_CONS = "Consumer goods and services, transportation and storage, administrative and support service activities"
_S_TECH = "Technology, telecommunications"
_S_HEALTH = "Health care, utilities, professional and technical activities"
#: Table 5: the buckets by label; bucket 1 has the two sub-buckets 1a and 1b
CCS_BUCKET_SECTORS: dict[str, str] = {
    "1a": _S_SOV,
    "1b": _S_LOC,
    "2": _S_FIN,
    "3": _S_BASIC,
    "4": _S_CONS,
    "5": _S_TECH,
    "6": _S_HEALTH,
    "7": "Other sector",
    "8": "Qualified Indices",
}
#: MAR50.65: the tenors in years of the counterparty credit spread delta risk factors
CCS_TENORS = (0.5, 1.0, 3.0, 5.0, 10.0)
_TABLE_6_PERCENT = (
    (100.0, 10.0, 20.0, 25.0, 20.0, 15.0, 0.0, 45.0),
    (None, 100.0, 5.0, 15.0, 20.0, 5.0, 0.0, 45.0),
    (None, None, 100.0, 20.0, 25.0, 5.0, 0.0, 45.0),
    (None, None, None, 100.0, 25.0, 5.0, 0.0, 45.0),
    (None, None, None, None, 100.0, 5.0, 0.0, 45.0),
    (None, None, None, None, None, 100.0, 0.0, 45.0),
    (None, None, None, None, None, None, 100.0, 0.0),
    (None, None, None, None, None, None, None, 100.0),
)
CCS_GAMMA = _symmetric(_TABLE_6_PERCENT)
#: Table 7 in percent: investment grade names, then high yield and not rated names, by bucket label
_TABLE_7_PERCENT = {
    "IG": (0.5, 1.0, 5.0, 3.0, 3.0, 2.0, 1.5, 5.0, 1.5),
    "HY_NR": (2.0, 4.0, 12.0, 7.0, 8.5, 5.5, 5.0, 12.0, 5.0),
}
CCS_RISK_WEIGHTS: dict[tuple[str, str], RegulatoryParameter] = {
    (label, quality): _rp(w / 100.0, "MAR50.65", "MAR50.65 Table 7")
    for quality, weights in _TABLE_7_PERCENT.items()
    for label, w in zip(CCS_BUCKET_SECTORS, weights, strict=True)
}
_CCS_NAME_RELATIONS = ("same", "related", "other")
CCS_RHO_TENOR_OTHER = _rp(0.9, "MAR50.65", "MAR50.65 rho_tenor")
CCS_RHO_NAME_RELATED = _rp(0.9, "MAR50.65", "MAR50.65 rho_name")
CCS_RHO_NAME_OTHER = _rp(0.5, "MAR50.65", "MAR50.65 rho_name, buckets 1 to 7")
CCS_RHO_NAME_OTHER_INDEX = _rp(0.8, "MAR50.65", "MAR50.65 rho_name, bucket 8")
CCS_RHO_QUALITY_OTHER = _rp(0.8, "MAR50.65", "MAR50.65 rho_quality")

# --- reference credit spread (MAR50.66 to MAR50.69) -------------------------------------------------------------------
#: Table 8: (number, credit quality, sector); the quality of bucket 15 is not applicable
RCS_BUCKETS: tuple[tuple[int, str, str], ...] = (
    (1, "IG", _S_SOV),
    (2, "IG", _S_LOC),
    (3, "IG", _S_FIN),
    (4, "IG", _S_BASIC),
    (5, "IG", _S_CONS),
    (6, "IG", _S_TECH),
    (7, "IG", _S_HEALTH),
    (8, "HY_NR", _S_SOV),
    (9, "HY_NR", _S_LOC),
    (10, "HY_NR", _S_FIN),
    (11, "HY_NR", _S_BASIC),
    (12, "HY_NR", _S_CONS),
    (13, "HY_NR", _S_TECH),
    (14, "HY_NR", _S_HEALTH),
    (15, "n/a", "Other sector"),
    (16, "IG", "Qualified Indices"),
    (17, "HY", "Qualified Indices"),
)
#: Table 9: the correlation between buckets of the same credit quality, the pairs 1/8 to 7/14 and the buckets 15, 16, 17
_TABLE_9_PERCENT = (
    (100.0, 75.0, 10.0, 20.0, 25.0, 20.0, 15.0, 0.0, 45.0, 45.0),
    (None, 100.0, 5.0, 15.0, 20.0, 15.0, 10.0, 0.0, 45.0, 45.0),
    (None, None, 100.0, 5.0, 15.0, 20.0, 5.0, 0.0, 45.0, 45.0),
    (None, None, None, 100.0, 20.0, 25.0, 5.0, 0.0, 45.0, 45.0),
    (None, None, None, None, 100.0, 25.0, 5.0, 0.0, 45.0, 45.0),
    (None, None, None, None, None, 100.0, 5.0, 0.0, 45.0, 45.0),
    (None, None, None, None, None, None, 100.0, 0.0, 45.0, 45.0),
    (None, None, None, None, None, None, None, 100.0, 0.0, 0.0),
    (None, None, None, None, None, None, None, None, 100.0, 75.0),
    (None, None, None, None, None, None, None, None, None, 100.0),
)
RCS_GAMMA_TABLE_9 = _symmetric(_TABLE_9_PERCENT)


def _rcs_gamma() -> np.ndarray:
    """The 17 x 17 matrix: Table 9 by bucket group, halved between buckets of 1 to 7 and 8 to 14 (MAR50.67 (2))."""
    group = {b: (b - 1) % 7 for b in range(1, 15)} | {15: 7, 16: 8, 17: 9}
    g = np.eye(17)
    for b in range(1, 18):
        for c in range(1, 18):
            if b != c:
                base = RCS_GAMMA_TABLE_9[group[b], group[c]]
                halved = b <= 14 and c <= 14 and (b <= 7) != (c <= 7)
                g[b - 1, c - 1] = base / 2.0 if halved else base
    return _readonly(g)


RCS_GAMMA = _rcs_gamma()
_TABLE_10_PERCENT = (0.5, 1.0, 5.0, 3.0, 3.0, 2.0, 1.5, 2.0, 4.0, 12.0, 7.0, 8.5, 5.5, 5.0, 12.0, 1.5, 5.0)
RCS_DELTA_RISK_WEIGHTS: dict[int, RegulatoryParameter] = {
    b: _rp(w / 100.0, "MAR50.68", "MAR50.68 Table 10") for b, w in enumerate(_TABLE_10_PERCENT, start=1)
}
RCS_VEGA_RISK_WEIGHT = _rp(1.0, "MAR50.69", "MAR50.69")

# --- equity (MAR50.70 to MAR50.73)  ---
_E_CONS = (
    "Consumer goods and services, transportation and storage, administrative and support service activities, "
    "healthcare, utilities"
)
_E_TEL = "Telecommunications, industrials"
_E_BASIC = "Basic materials, energy, agriculture, manufacturing, mining and quarrying"
_E_FIN = "Financials including government-backed financials, real estate activities, technology"
#: Table 11: (number, size and region as printed, sector)
EQ_BUCKETS: tuple[tuple[int, str, str], ...] = (
    (1, "Large | Emerging market economies", _E_CONS),
    (2, "Large | Emerging market economies", _E_TEL),
    (3, "Large | Emerging market economies", _E_BASIC),
    (4, "Large | Emerging market economies", _E_FIN),
    (5, "Large | Advanced economies", _E_CONS),
    (6, "Large | Advanced economies", _E_TEL),
    (7, "Large | Advanced economies", _E_BASIC),
    (8, "Large | Advanced economies", _E_FIN),
    (9, "Small | Emerging market economies", "All sectors described under bucket numbers 1, 2, 3, and 4"),
    (10, "Small | Advanced economies", "All sectors described under bucket numbers 5, 6, 7, and 8"),
    (11, "(Not applicable)", "Other sector"),
    (12, "Large cap, advanced economies", "Qualified Indices"),
    (13, "Other", "Qualified Indices"),
)
_TABLE_12_PERCENT = (55.0, 60.0, 45.0, 55.0, 30.0, 35.0, 40.0, 50.0, 70.0, 50.0, 70.0, 15.0, 25.0)
EQ_DELTA_RISK_WEIGHTS: dict[int, RegulatoryParameter] = {
    b: _rp(w / 100.0, "MAR50.72", "MAR50.72 Table 12") for b, w in enumerate(_TABLE_12_PERCENT, start=1)
}
#: MAR50.73: 78 percent for the large market capitalisation buckets (Table 11: 1 to 8, and the qualified index bucket 12
#: marked "Large cap"), 100 percent for the others
EQ_LARGE_CAP_BUCKETS = (1, 2, 3, 4, 5, 6, 7, 8, 12)
EQ_VEGA_RISK_WEIGHTS: dict[int, RegulatoryParameter] = {
    b: _rp(0.78 if b in EQ_LARGE_CAP_BUCKETS else 1.0, "MAR50.73", "MAR50.73") for b in range(1, 14)
}


def _eq_gamma() -> np.ndarray:
    """MAR50.71: 15 percent inside buckets 1 to 10; 75 percent between 12 and 13; 45 percent between 12 or 13 and
    1 to 10; 0 with bucket 11."""
    g = np.eye(13)
    for b in range(1, 14):
        for c in range(1, 14):
            if b == c:
                continue
            if 11 in (b, c):
                v = 0.0
            elif {b, c} == {12, 13}:
                v = 0.75
            elif max(b, c) >= 12:
                v = 0.45
            else:
                v = 0.15
            g[b - 1, c - 1] = v
    return _readonly(g)


EQ_GAMMA = _eq_gamma()

# --- commodity (MAR50.74 to MAR50.77)  ---
#: Table 13: (number, commodity group)
COM_BUCKETS: tuple[tuple[int, str], ...] = (
    (1, "Energy - Solid combustibles"),
    (2, "Energy - Liquid combustibles"),
    (3, "Energy - Electricity and carbon trading"),
    (4, "Freight"),
    (5, "Metals - non-precious"),
    (6, "Gaseous combustibles"),
    (7, "Precious metals (including gold)"),
    (8, "Grains & oilseed"),
    (9, "Livestock & dairy"),
    (10, "Softs and other agriculturals"),
    (11, "Other commodity"),
)
_TABLE_14_PERCENT = (30.0, 35.0, 60.0, 80.0, 40.0, 45.0, 20.0, 35.0, 25.0, 35.0, 50.0)
COM_DELTA_RISK_WEIGHTS: dict[int, RegulatoryParameter] = {
    b: _rp(w / 100.0, "MAR50.76", "MAR50.76 Table 14") for b, w in enumerate(_TABLE_14_PERCENT, start=1)
}
COM_VEGA_RISK_WEIGHT = _rp(1.0, "MAR50.77", "MAR50.77")
COM_GAMMA_WITHIN = _rp(0.2, "MAR50.75", "MAR50.75")


def _com_gamma() -> np.ndarray:
    g = np.eye(11)
    for b in range(1, 12):
        for c in range(1, 12):
            if b != c:
                g[b - 1, c - 1] = 0.0 if 11 in (b, c) else COM_GAMMA_WITHIN.value
    return _readonly(g)


COM_GAMMA = _com_gamma()


# --- the parameters of one risk class  ---
def _label_index(risk_class: str, label: str) -> int:
    """The zero-based row of a bucket label in the gamma matrix: the bucket number minus one ('1a' is bucket 1)."""
    digits = label.rstrip("ab") if risk_class == "ccs" else label
    try:
        return int(digits) - 1
    except ValueError:
        raise ValueError(f"bucket {label!r} is not a bucket of {_CLASS_NAMES[risk_class]} risk") from None


@dataclass(frozen=True, eq=False)
class SaCvaParameters:
    """The parameters of one risk class and kind with the paragraphs they come from.

    `risk_weights` maps (bucket label, column) to the weight: ("*", factor) for interest rate, ("*", "all") for FX,
    (bucket label, credit quality) for counterparty credit spread and (bucket label, "all") for the single-factor
    classes. `correlation` is the within-bucket matrix over `factors` (None where MAR50 states none: a single risk
    factor, and the counterparty credit spread, whose correlation is the product of `ccs_rho`); `gamma` is the
    cross-bucket matrix over `buckets`, or `gamma_constant` where one value holds for every pair of buckets."""

    risk_class: str
    kind: str
    factors: tuple[str, ...]
    buckets: tuple[str, ...]
    risk_weights: dict[tuple[str, str], RegulatoryParameter]
    correlation: np.ndarray | None
    gamma: np.ndarray | None
    gamma_constant: RegulatoryParameter | None
    in_force: str
    paragraphs: dict[str, str]

    def risk_weight(self, bucket: str, factor: str | None = None, quality: str | None = None) -> float:
        """The weight as a decimal: `factor` for interest rate, `quality` ("IG" or "HY_NR") for counterparty credit
        spread, neither for the single-factor classes."""
        if self.risk_class == "ir":
            key = ("*", factor if factor is not None else "")
        elif self.risk_class == "fx":
            key = ("*", "all")
        elif self.risk_class == "ccs":
            key = (str(bucket), quality if quality is not None else "")
        else:
            key = (str(bucket), "all")
        if key not in self.risk_weights:
            raise ValueError(
                f"no risk weight for bucket {bucket!r}, factor {factor!r}, quality {quality!r} "
                f"({self.paragraphs['risk_weights']})"
            )
        return self.risk_weights[key].value

    def gamma_between(self, bucket: str, other: str) -> float:
        """The cross-bucket correlation between two buckets (1 for the same bucket)."""
        if bucket == other:
            return 1.0
        if self.gamma_constant is not None:
            return self.gamma_constant.value
        assert self.gamma is not None
        i, j = _label_index(self.risk_class, str(bucket)), _label_index(self.risk_class, str(other))
        n = self.gamma.shape[0]
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(
                f"buckets {bucket!r} and {other!r} are not both buckets of {_CLASS_NAMES[self.risk_class]} risk"
            )
        return float(self.gamma[i, j])


def _two_by_two(rho: float) -> np.ndarray:
    return _readonly(np.array([[1.0, rho], [rho, 1.0]]))


@cache
def sa_cva_parameters(risk_class: str, kind: str, specified_currency: bool = True) -> SaCvaParameters:
    """The parameters of a risk class and kind (MAR50.54 to MAR50.77). `specified_currency` matters for the interest
    rate delta only: True for the reporting currency and the currencies of MAR50.56, False for any other (MAR50.57)."""
    _check_class_kind(risk_class, kind)
    if risk_class == "ir":
        if kind == "vega":
            return SaCvaParameters(
                "ir", "vega", IR_FACTORS_VEGA, (), {("*", f): IR_VEGA_RISK_WEIGHT for f in IR_FACTORS_VEGA},
                _two_by_two(IR_VEGA_RHO.value), None, IR_GAMMA, MAR50_IN_FORCE,
                {"risk_weights": "MAR50.58", "correlation": "MAR50.58", "gamma": "MAR50.55"},
            )  # fmt: skip
        if specified_currency:
            return SaCvaParameters(
                "ir", "delta", IR_FACTORS_SPECIFIED, (),
                {("*", f): rp for f, rp in IR_DELTA_RISK_WEIGHTS_SPECIFIED.items()},
                IR_DELTA_CORRELATION_SPECIFIED, None, IR_GAMMA, MAR50_IN_FORCE,
                {"risk_weights": "MAR50.56 Table 3", "correlation": "MAR50.56 Table 4", "gamma": "MAR50.55"},
            )  # fmt: skip
        return SaCvaParameters(
            "ir", "delta", IR_FACTORS_OTHER, (), {("*", f): IR_DELTA_RISK_WEIGHT_OTHER for f in IR_FACTORS_OTHER},
            _two_by_two(IR_DELTA_RHO_OTHER.value), None, IR_GAMMA, MAR50_IN_FORCE,
            {"risk_weights": "MAR50.57", "correlation": "MAR50.57", "gamma": "MAR50.55"},
        )  # fmt: skip
    if risk_class == "fx":
        weight = FX_DELTA_RISK_WEIGHT if kind == "delta" else FX_VEGA_RISK_WEIGHT
        return SaCvaParameters(
            "fx", kind, ("all",), (), {("*", "all"): weight}, None, None, FX_GAMMA, MAR50_IN_FORCE,
            {"risk_weights": weight.paragraph, "gamma": "MAR50.60"},
        )  # fmt: skip
    if risk_class == "ccs":
        return SaCvaParameters(
            "ccs", "delta", tuple(str(t) for t in CCS_TENORS), tuple(CCS_BUCKET_SECTORS), dict(CCS_RISK_WEIGHTS), None,
            CCS_GAMMA, None, MAR50_IN_FORCE,
            {"risk_weights": "MAR50.65 Table 7", "correlation": "MAR50.65 (ccs_rho)", "gamma": "MAR50.64 Table 6"},
        )  # fmt: skip
    if risk_class == "rcs":
        weights = {
            (str(b), "all"): (rp if kind == "delta" else RCS_VEGA_RISK_WEIGHT)
            for b, rp in RCS_DELTA_RISK_WEIGHTS.items()
        }
        return SaCvaParameters(
            "rcs", kind, ("all",), tuple(str(b[0]) for b in RCS_BUCKETS), weights, None, RCS_GAMMA, None,
            MAR50_IN_FORCE,
            {"risk_weights": "MAR50.68 Table 10" if kind == "delta" else "MAR50.69", "gamma": "MAR50.67 Table 9"},
        )  # fmt: skip
    if risk_class == "eq":
        table = EQ_DELTA_RISK_WEIGHTS if kind == "delta" else EQ_VEGA_RISK_WEIGHTS
        return SaCvaParameters(
            "eq", kind, ("all",), tuple(str(b[0]) for b in EQ_BUCKETS),
            {(str(b), "all"): rp for b, rp in table.items()}, None, EQ_GAMMA, None, MAR50_IN_FORCE,
            {"risk_weights": "MAR50.72 Table 12" if kind == "delta" else "MAR50.73", "gamma": "MAR50.71"},
        )  # fmt: skip
    table = COM_DELTA_RISK_WEIGHTS
    weights = {(str(b), "all"): (rp if kind == "delta" else COM_VEGA_RISK_WEIGHT) for b, rp in table.items()}
    return SaCvaParameters(
        "com", kind, ("all",), tuple(str(b[0]) for b in COM_BUCKETS), weights, None, COM_GAMMA, None, MAR50_IN_FORCE,
        {"risk_weights": "MAR50.76 Table 14" if kind == "delta" else "MAR50.77", "gamma": "MAR50.75"},
    )  # fmt: skip


def _check_class_kind(risk_class: object, kind: object) -> None:
    if risk_class not in RISK_CLASSES:
        raise ValueError(f"risk_class {risk_class!r} is not one of {list(RISK_CLASSES)} (MAR50.43)")
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {list(KINDS)} (MAR50.42)")
    if risk_class == "ccs" and kind == "vega":
        raise ValueError("counterparty credit spread risk has no vega capital requirement (MAR50.45, MAR50.63)")


# --- the formulas of MAR50.51 to MAR50.53  ---
def _vector(name: str, x: object) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a sequence of finite numbers, got {x!r}") from None
    if a.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return a


def _matrix(name: str, x: object, n: int) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a square matrix of finite numbers, got {x!r}") from None
    if a.shape != (n, n):
        raise ValueError(f"{name} must be a {n} x {n} matrix, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    if np.any(np.abs(a) > 1.0 + 1e-12):
        raise ValueError(f"{name} entries must lie in [-1, 1]")
    if not np.allclose(a, a.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return a


def weighted_sensitivity(rw: object, s_cva: object, s_hdg: object) -> tuple:
    """MAR50.51 and MAR50.52: (WS^CVA, WS^Hdg, WS) with WS^CVA = RW s^CVA, WS^Hdg = RW s^Hdg and WS = WS^CVA - WS^Hdg.
    Scalars give floats, arrays give arrays."""
    w, c, h = (np.asarray(x, dtype=float) for x in (rw, s_cva, s_hdg))
    for name, a in (("rw", w), ("s_cva", c), ("s_hdg", h)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite, got {a!r}")
    if np.any(w < 0.0):
        raise ValueError(f"rw must be at least 0, got {rw!r}")
    ws_cva, ws_hdg = w * c, w * h
    ws = ws_cva - ws_hdg
    if ws.ndim == 0:
        return float(ws_cva), float(ws_hdg), float(ws)
    return ws_cva, ws_hdg, ws


def bucket_k(ws_cva: object, ws_hdg: object, rho: object, r: float = R.value) -> float:
    """MAR50.53 (1): K_b = sqrt(max(0, sum WS_k^2 + sum_{k != l} rho_kl WS_k WS_l + R sum (WS_k^Hdg)^2)) with
    WS = WS^CVA - WS^Hdg and the hedging disallowance inside the root. `rho` is the within-bucket matrix (its diagonal
    is not used); a bucket with one risk factor takes no correlation term and may pass `rho=None`. An empty bucket has
    K_b = 0."""
    cva = _vector("ws_cva", ws_cva)
    hdg = _vector("ws_hdg", ws_hdg)
    if cva.shape != hdg.shape:
        raise ValueError(f"ws_hdg must have the same length as ws_cva ({cva.size}), got {hdg.size}")
    disallowance = _nonnegative("r", r)
    n = cva.size
    if n == 0:
        return 0.0
    if rho is None:
        if n > 1:
            raise ValueError(f"rho is required for a bucket of {n} risk factors")
        correlation = np.zeros((1, 1))
    else:
        correlation = _matrix("rho", rho, n)
    ws = cva - hdg
    off_diagonal = correlation - np.diag(np.diag(correlation))
    radicand = float(ws @ ws + ws @ off_diagonal @ ws + disallowance * (hdg @ hdg))
    return math.sqrt(max(0.0, radicand))


def bucket_s(ws: object, k_b: float) -> float:
    """MAR50.53 (3): S_b = max(-K_b, min(sum_k WS_k, K_b)), always capped and floored by K_b."""
    v = _vector("ws", ws)
    k = _nonnegative("k_b", k_b)
    return max(-k, min(float(v.sum()), k))


def risk_class_k(k_b: object, s_b: object, gamma: object, m_cva: float = M_CVA.value) -> float:
    """MAR50.53 (2): K = m_CVA sqrt(max(0, sum_b K_b^2 + sum_{b != c} gamma_bc S_b S_c)); `gamma` is the cross-bucket
    matrix over the buckets given (its diagonal is not used)."""
    k = _vector("k_b", k_b)
    s = _vector("s_b", s_b)
    if np.any(k < 0.0):
        raise ValueError(f"k_b must be at least 0, got {k_b!r}")
    if k.shape != s.shape:
        raise ValueError(f"s_b must have the same length as k_b ({k.size}), got {s.size}")
    m = _finite("m_cva", m_cva)
    if m < M_CVA.value:
        raise ValueError(f"m_cva must be at least {M_CVA.value} (MAR50.41), got {m_cva!r}")
    n = k.size
    if n == 0:
        return 0.0
    g = _matrix("gamma", gamma, n)
    off_diagonal = g - np.diag(np.diag(g))
    radicand = float(k @ k + s @ off_diagonal @ s)
    return m * math.sqrt(max(0.0, radicand))


def ccs_rho(same_tenor: bool, name_relation: str, same_quality: bool, index: bool = False) -> float:
    """MAR50.65: rho_kl = rho_tenor x rho_name x rho_quality for counterparty credit spread factors. rho_tenor is 1 for
    the same tenor and 0.9 otherwise; `name_relation` is "same" (1), "related" (0.9: distinct names that are legally
    related; for indices, the same index of distinct series) or "other" (0.5; 0.8 for indices); rho_quality is 1 for the
    same credit quality (IG with IG, HY or NR with HY or NR) and 0.8 otherwise."""
    if name_relation not in _CCS_NAME_RELATIONS:
        raise ValueError(f"name_relation {name_relation!r} is not one of {list(_CCS_NAME_RELATIONS)} (MAR50.65)")
    tenor = 1.0 if same_tenor else CCS_RHO_TENOR_OTHER.value
    if name_relation == "same":
        name = 1.0
    elif name_relation == "related":
        name = CCS_RHO_NAME_RELATED.value
    else:
        name = (CCS_RHO_NAME_OTHER_INDEX if index else CCS_RHO_NAME_OTHER).value
    quality = 1.0 if same_quality else CCS_RHO_QUALITY_OTHER.value
    return tenor * name * quality


# --- the capital from the sensitivities  ---
@dataclass(frozen=True)
class Sensitivity:
    """One net sensitivity of the CVA portfolio and of its eligible hedges to a risk factor (MAR50.47).

    `risk_class` is one of "ir", "fx", "ccs", "rcs", "eq", "com" and `kind` "delta" or "vega". `bucket` is the currency
    code for interest rate and FX, the bucket number for the others (counterparty credit spread: 1a, 1b, 2 to 8).
    `factor` is the tenor label for interest rate ("1y", "2y", "5y", "10y", "30y", "inflation"; "parallel" and
    "inflation" for a currency outside the specified ones; "ir_vol" and "inflation_vol" for vega), the tenor in years
    for counterparty credit spread (0.5, 1, 3, 5, 10) and None for the single-factor classes. `s_cva` and `s_hdg` are
    the sensitivities as MAR50.47 defines them (the change of the value for the prescribed shift divided by the
    shift). The counterparty credit spread factors also carry the entity (`name`), its credit quality
    (`investment_grade`), the legal group that makes two names legally related (`legal_group`) and, for an index, its
    `series`."""

    risk_class: str
    kind: str
    bucket: str | int
    factor: str | float | None
    s_cva: float
    s_hdg: float = 0.0
    name: str | None = None
    investment_grade: bool | None = None
    legal_group: str | None = None
    series: str | None = None


@dataclass(frozen=True)
class RiskClassResult:
    """K of one risk class and kind with the K_b and S_b of its buckets."""

    risk_class: str
    kind: str
    k: float
    bucket_k: dict[str, float]
    bucket_s: dict[str, float]


@dataclass(frozen=True)
class SaCvaCapital:
    """The SA-CVA capital: K_delta, K_vega, their sum, the RWA (12.5 times it), the risk classes' results."""

    k_delta: float
    k_vega: float
    capital: float
    rwa: float
    m_cva: float
    classes: tuple[RiskClassResult, ...]
    paragraphs: dict[str, str]


@dataclass
class _Net:
    bucket: str  # the bucket of the cross-bucket aggregation ('1a' and '1b' are bucket 1)
    label: str  # the bucket label of the risk weight
    rw: float
    idx: int  # the position of the factor in the interest rate matrix
    s_cva: float = 0.0
    s_hdg: float = 0.0
    name: str | None = None
    series: str | None = None
    tenor: float | None = None
    investment_grade: bool | None = None
    legal_group: str | None = None


def _bucket_label(risk_class: str, bucket: object) -> str:
    sizes = {"rcs": 17, "eq": 13, "com": 11, "ccs": 8}
    text = str(bucket).strip()
    if risk_class == "ccs" and text in ("1", "1a", "1b"):
        if text == "1":
            raise ValueError("bucket 1 of counterparty credit spread risk is '1a' or '1b' (Table 5, MAR50.63)")
        return text
    try:
        number = int(text)
    except ValueError:
        raise ValueError(f"bucket {bucket!r} is not a bucket of {_CLASS_NAMES[risk_class]} risk") from None
    if isinstance(bucket, bool) or not 1 <= number <= sizes[risk_class]:
        raise ValueError(
            f"bucket {bucket!r} is not a bucket of {_CLASS_NAMES[risk_class]} risk "
            f"(it has buckets 1 to {sizes[risk_class]})"
        )
    return str(number)


def _normalise(s: Sensitivity, reporting_currency: str) -> tuple[tuple, _Net]:
    if not isinstance(s, Sensitivity):
        raise ValueError(f"sensitivities must hold Sensitivity records, got {s!r}")
    _check_class_kind(s.risk_class, s.kind)
    c, h = _finite("s_cva", s.s_cva), _finite("s_hdg", s.s_hdg)
    cls, kind = s.risk_class, s.kind
    if cls in ("ir", "fx"):
        if not isinstance(s.bucket, str) or not s.bucket.strip():
            raise ValueError(f"bucket must be a currency code for {_CLASS_NAMES[cls]} risk, got {s.bucket!r}")
        currency = s.bucket.strip()
        if cls == "fx":
            if currency == reporting_currency:
                raise ValueError(f"bucket {currency!r} is the reporting currency, which has no FX bucket (MAR50.59)")
            if s.factor is not None:
                raise ValueError(
                    f"factor {s.factor!r} is not a risk factor of FX risk (one factor per currency: pass None)"
                )
            return ("fx", kind, currency, None), _Net(
                currency, currency, sa_cva_parameters("fx", kind).risk_weight("*"), 0, c, h
            )
        specified = kind == "vega" or currency == reporting_currency or currency in SPECIFIED_CURRENCIES
        params = sa_cva_parameters("ir", kind, specified)
        if s.factor not in params.factors:
            raise ValueError(
                f"factor {s.factor!r} is not an interest rate {kind} risk factor of {currency!r}: "
                f"{list(params.factors)}" + ("" if kind == "vega" else f" ({'MAR50.56' if specified else 'MAR50.57'})")
            )
        rw = params.risk_weight("*", s.factor)
        return ("ir", kind, currency, s.factor), _Net(currency, currency, rw, params.factors.index(s.factor), c, h)
    label = _bucket_label(cls, s.bucket)
    if cls != "ccs":
        if s.factor is not None:
            raise ValueError(
                f"factor {s.factor!r} is not a risk factor of {_CLASS_NAMES[cls]} risk "
                "(one factor per bucket: pass None)"
            )
        rw = sa_cva_parameters(cls, kind).risk_weight(label)
        return (cls, kind, label, None), _Net(label, label, rw, 0, c, h)
    # counterparty credit spread
    if not isinstance(s.name, str) or not s.name:
        raise ValueError(
            "name (the entity or the index) is required for a counterparty credit spread factor (MAR50.65)"
        )
    if not isinstance(s.investment_grade, bool):
        raise ValueError(
            "investment_grade must be True or False for a counterparty credit spread factor, "
            f"got {s.investment_grade!r}"
        )
    try:
        tenor = float(s.factor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        tenor = math.nan
    if tenor not in CCS_TENORS:
        raise ValueError(f"tenor {s.factor!r} is not one of {list(CCS_TENORS)} years (MAR50.65)")
    quality = "IG" if s.investment_grade else "HY_NR"
    rw = sa_cva_parameters("ccs", "delta").risk_weight(label, quality=quality)
    net = _Net(label.rstrip("ab"), label, rw, 0, c, h, s.name, s.series, tenor, s.investment_grade, s.legal_group)
    return ("ccs", kind, label, s.name, s.series, tenor), net


def _name_relation(a: _Net, b: _Net, index: bool) -> str:
    if index:
        if a.name == b.name and a.series == b.series:
            return "same"
        return "related" if a.name == b.name else "other"
    if a.name == b.name:
        return "same"
    return "related" if a.legal_group is not None and a.legal_group == b.legal_group else "other"


def _within_correlation(risk_class: str, kind: str, factors: list[_Net], params: SaCvaParameters | None):
    n = len(factors)
    rho = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = factors[i], factors[j]
            if risk_class == "ir":
                assert params is not None and params.correlation is not None
                rho[i, j] = params.correlation[a.idx, b.idx]
            elif risk_class == "ccs":
                rho[i, j] = ccs_rho(
                    a.tenor == b.tenor,
                    _name_relation(a, b, a.bucket == "8"),
                    a.investment_grade == b.investment_grade,
                    index=a.bucket == "8",
                )
    return rho


def _class_charge(
    risk_class: str, kind: str, nets: list[_Net], m_cva: float, reporting_currency: str
) -> RiskClassResult:
    by_bucket: dict[str, list[_Net]] = {}
    for n in nets:
        by_bucket.setdefault(n.bucket, []).append(n)
    names = sorted(by_bucket, key=lambda b: (len(b), b))
    k_list, s_list = [], []
    for name in names:
        factors = by_bucket[name]
        params = None
        if risk_class == "ir":
            specified = kind == "vega" or name == reporting_currency or name in SPECIFIED_CURRENCIES
            params = sa_cva_parameters("ir", kind, specified)
        ws_cva = [f.rw * f.s_cva for f in factors]
        ws_hdg = [f.rw * f.s_hdg for f in factors]
        k_b = bucket_k(ws_cva, ws_hdg, _within_correlation(risk_class, kind, factors, params))
        k_list.append(k_b)
        s_list.append(bucket_s([a - b for a, b in zip(ws_cva, ws_hdg, strict=True)], k_b))
    reference = sa_cva_parameters(risk_class, kind)
    n_buckets = len(names)
    gamma = np.eye(n_buckets)
    for i in range(n_buckets):
        for j in range(n_buckets):
            if i != j:
                gamma[i, j] = reference.gamma_between(names[i], names[j])
    k = risk_class_k(k_list, s_list, gamma, m_cva)
    return RiskClassResult(
        risk_class, kind, k, dict(zip(names, k_list, strict=True)), dict(zip(names, s_list, strict=True))
    )


def sa_cva_capital(
    sensitivities: Iterable[Sensitivity], reporting_currency: str, m_cva: float = M_CVA.value
) -> SaCvaCapital:
    """MAR50.42 to MAR50.53: the SA-CVA capital from the net sensitivities of the CVA portfolio and its eligible hedges.

    Sensitivities of the same risk factor are summed before they are weighted. The interest rate class of a currency is
    the specified one for the reporting currency and for USD, EUR, GBP, AUD, CAD, SEK and JPY, the other one for any
    other (MAR50.56, MAR50.57); an FX sensitivity has no bucket for the reporting currency (MAR50.59). It refuses a
    counterparty credit spread vega (MAR50.45), an unknown risk class or kind and a non-finite sensitivity, naming
    them. The CVA sensitivities are the caller's (see the module docstring)."""
    if not isinstance(reporting_currency, str) or not reporting_currency.strip():
        raise ValueError(f"reporting_currency must be a currency code, got {reporting_currency!r}")
    reporting = reporting_currency.strip()
    m = _finite("m_cva", m_cva)
    if m < M_CVA.value:
        raise ValueError(f"m_cva must be at least {M_CVA.value} (MAR50.41), got {m_cva!r}")
    merged: dict[tuple, _Net] = {}
    for s in sensitivities:
        key, net = _normalise(s, reporting)
        if key in merged:
            have = merged[key]
            if have.investment_grade != net.investment_grade:
                raise ValueError(f"investment_grade differs between two sensitivities of the factor {key[3:]!r}")
            if have.legal_group != net.legal_group:
                raise ValueError(f"legal_group differs between two sensitivities of the factor {key[3:]!r}")
            have.s_cva += net.s_cva
            have.s_hdg += net.s_hdg
        else:
            merged[key] = net
    grouped: dict[tuple[str, str], list[_Net]] = {}
    for key, net in merged.items():
        grouped.setdefault((key[0], key[1]), []).append(net)
    classes = []
    for kind in KINDS:
        for risk_class in RISK_CLASSES:
            if (risk_class, kind) in grouped:
                classes.append(_class_charge(risk_class, kind, grouped[(risk_class, kind)], m, reporting))
    k_delta = sum(c.k for c in classes if c.kind == "delta")
    k_vega = sum(c.k for c in classes if c.kind == "vega")
    capital = k_delta + k_vega
    paragraphs = {
        "capital": "MAR50.42 (delta plus vega), MAR50.43 and MAR50.45 (the simple sums over the risk classes)",
        "rwa": "MAR50.1 (12.5 times the capital)",
        "aggregation": "MAR50.53",
        "weighted_sensitivities": "MAR50.51, MAR50.52",
    }
    return SaCvaCapital(k_delta, k_vega, capital, RWA_FACTOR.value * capital, m, tuple(classes), paragraphs)


__all__ = [
    "COM_BUCKETS",
    "COM_DELTA_RISK_WEIGHTS",
    "COM_GAMMA",
    "COM_GAMMA_WITHIN",
    "COM_VEGA_RISK_WEIGHT",
    "CCS_BUCKET_SECTORS",
    "CCS_GAMMA",
    "CCS_RISK_WEIGHTS",
    "CCS_TENORS",
    "EQ_BUCKETS",
    "EQ_DELTA_RISK_WEIGHTS",
    "EQ_GAMMA",
    "EQ_LARGE_CAP_BUCKETS",
    "EQ_VEGA_RISK_WEIGHTS",
    "FX_DELTA_RISK_WEIGHT",
    "FX_GAMMA",
    "FX_VEGA_RISK_WEIGHT",
    "IR_DELTA_CORRELATION_SPECIFIED",
    "IR_DELTA_RHO_OTHER",
    "IR_DELTA_RISK_WEIGHTS_SPECIFIED",
    "IR_DELTA_RISK_WEIGHT_OTHER",
    "IR_FACTORS_SPECIFIED",
    "IR_GAMMA",
    "IR_VEGA_RHO",
    "IR_VEGA_RISK_WEIGHT",
    "M_CVA",
    "R",
    "RCS_BUCKETS",
    "RCS_DELTA_RISK_WEIGHTS",
    "RCS_GAMMA",
    "RCS_GAMMA_TABLE_9",
    "RCS_VEGA_RISK_WEIGHT",
    "RWA_FACTOR",
    "SPECIFIED_CURRENCIES",
    "RiskClassResult",
    "SaCvaCapital",
    "SaCvaParameters",
    "Sensitivity",
    "bucket_k",
    "bucket_s",
    "ccs_rho",
    "risk_class_k",
    "sa_cva_capital",
    "sa_cva_parameters",
    "weighted_sensitivity",
]
