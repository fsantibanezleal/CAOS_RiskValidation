"""SA-CCR, the standardised approach to counterparty credit risk (CRE52, in force 2023-01-01, published 2020-06-05;
dossier 01 E.2).

    EAD = alpha (RC + PFE),  alpha = 1.4 (CRE52.1);   PFE = multiplier x AddOn^aggregate (CRE52.20)
    RC  = max(V - C, 0) unmargined (CRE52.10),  max(V - C, TH + MTA - NICA, 0) margined (CRE52.18)
    multiplier = min(1, Floor + (1 - Floor) exp((V - C) / (2 (1 - Floor) AddOn^aggregate)))  (CRE52.23, Floor 5 percent)
    AddOn^aggregate = the sum of the five asset-class add-ons (CRE52.24 to CRE52.25)

with the supervisory duration (CRE52.34), delta (CRE52.38 to CRE52.41), maturity factor (CRE52.48, CRE52.52), the
interest-rate maturity buckets and offset formula (CRE52.57), FX per currency pair (CRE52.59), credit and equity
entity aggregation (CRE52.61, CRE52.66), commodity hedging sets (CRE52.70) and the supervisory parameters of CRE52.72
Table 2 as module data. Every intermediate value is a field of a result, so the CRE99 worked examples (CRE99.20 to
CRE99.97) are checked step by step from their trades. Money is in one currency the caller states; times are years;
MPoR is in business days, 250 to the year (CRE52.52). The engine computes and states no compliance claim. Not
implemented (stated limits): the CRE52.40 shift of the option strike and price for negative rates, FX and commodity
options vega, basis and volatility transactions, and the CRE52.58 to CRE52.60 sign rules of FX trades beyond the
signed effective notional the caller passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import ndtr

from riskvalidation.regulatory.standardised import RegulatoryParameter, _finite, _nonnegative

CRE52_IN_FORCE = "2023-01-01"
BUSINESS_DAYS_PER_YEAR = 250  # CRE52.52
ALPHA = 1.4  # CRE52.1
MULTIPLIER_FLOOR = 0.05  # CRE52.23
IR_OFFSET_CROSS_B1_B2 = 1.4  # CRE52.57
IR_OFFSET_CROSS_B2_B3 = 1.4
IR_OFFSET_CROSS_B1_B3 = 0.6


def _table2(key: str, **values: float) -> dict[tuple[str, str], RegulatoryParameter]:
    return {
        (key, name): RegulatoryParameter(float(v), "CRE52.72", "CRE52.72 Table 2", CRE52_IN_FORCE)
        for name, v in values.items()
    }


#: CRE52.72 Table 2: the supervisory factor, the correlation and the option volatility by asset class and sub-class
# (decimals)
SUPERVISORY_PARAMETERS: dict[tuple[str, str], RegulatoryParameter] = {}
SUPERVISORY_PARAMETERS.update(_table2("interest-rate", supervisory_factor=0.005, option_volatility=0.5))
SUPERVISORY_PARAMETERS.update(_table2("fx", supervisory_factor=0.04, option_volatility=0.15))
for _name, _sf in (
    ("AAA", 0.0038),
    ("AA", 0.0038),
    ("A", 0.0042),
    ("BBB", 0.0054),
    ("BB", 0.0106),
    ("B", 0.016),
    ("CCC", 0.06),
):
    SUPERVISORY_PARAMETERS.update(
        _table2(f"credit-single-name-{_name}", supervisory_factor=_sf, correlation=0.5, option_volatility=1.0)
    )
SUPERVISORY_PARAMETERS.update(
    _table2("credit-index-IG", supervisory_factor=0.0038, correlation=0.8, option_volatility=0.8)
)
SUPERVISORY_PARAMETERS.update(
    _table2("credit-index-SG", supervisory_factor=0.0106, correlation=0.8, option_volatility=0.8)
)
SUPERVISORY_PARAMETERS.update(
    _table2("equity-single-name", supervisory_factor=0.32, correlation=0.5, option_volatility=1.2)
)
SUPERVISORY_PARAMETERS.update(_table2("equity-index", supervisory_factor=0.2, correlation=0.8, option_volatility=0.75))
SUPERVISORY_PARAMETERS.update(
    _table2("commodity-electricity", supervisory_factor=0.4, correlation=0.4, option_volatility=1.5)
)
SUPERVISORY_PARAMETERS.update(
    _table2(
        "commodity-oil-gas-metals-agricultural-other", supervisory_factor=0.18, correlation=0.4, option_volatility=0.7
    )
)
del _name, _sf


def supervisory_parameter(key: str, parameter: str) -> RegulatoryParameter:
    """One cell of CRE52.72 Table 2; `parameter` is supervisory_factor, correlation or option_volatility."""
    if not any(k == key for k, _ in SUPERVISORY_PARAMETERS):
        raise ValueError(f"key {key!r} is not a row of CRE52.72 Table 2 (CRE52.72)")
    if (key, parameter) not in SUPERVISORY_PARAMETERS:
        raise ValueError(f"parameter {parameter!r} is not given for {key!r} in CRE52.72 Table 2 (CRE52.72)")
    return SUPERVISORY_PARAMETERS[(key, parameter)]


def supervisory_duration(start: float, end: float) -> float:
    """CRE52.34: (exp(-0.05 S) - exp(-0.05 E)) / 0.05, with 0 <= S < E in years (a started trade has S = 0, the
    caller's)."""
    s = _nonnegative("start", start)
    e = _finite("end", end)
    if e <= s:
        raise ValueError(f"end {end!r} must be greater than start {start!r}")
    return (math.exp(-0.05 * s) - math.exp(-0.05 * e)) / 0.05


def supervisory_delta_option(
    price: float, strike: float, vol: float, expiry: float, is_call: bool, is_long: bool
) -> float:
    """CRE52.38 to CRE52.41: +-Phi(+-(ln(P / K) + 0.5 vol^2 T) / (vol sqrt(T))) by bought or sold call or put.

    Bought call +Phi(d), sold call -Phi(d), bought put -Phi(-d), sold put +Phi(-d), with d the argument above.
    """
    p = _finite("price", price)
    k = _finite("strike", strike)
    v = _finite("vol", vol)
    t = _finite("expiry", expiry)
    for name, x in (("price", p), ("strike", k), ("vol", v), ("expiry", t)):
        if x <= 0.0:
            raise ValueError(
                f"{name} must be positive, got {x!r} (the CRE52.40 shift for negative rates is not implemented)"
            )
    d = (math.log(p / k) + 0.5 * v * v * t) / (v * math.sqrt(t))
    if is_call:
        delta = float(ndtr(d))
        return delta if is_long else -delta
    delta = float(ndtr(-d))
    return -delta if is_long else delta


def supervisory_delta_tranche(attachment: float, detachment: float, is_long: bool) -> float:
    """CRE52.41: +-15 / ((1 + 14 A) (1 + 14 D)) for a CDO tranche with 0 <= A < D <= 1; short is the negative."""
    a = _finite("attachment", attachment)
    d = _finite("detachment", detachment)
    if not 0.0 <= a < 1.0:
        raise ValueError(f"attachment must satisfy 0 <= A < 1, got {attachment!r}")
    if not a < d <= 1.0:
        raise ValueError(f"detachment must satisfy A < D <= 1, got {detachment!r} with attachment {attachment!r}")
    value = 15.0 / ((1.0 + 14.0 * a) * (1.0 + 14.0 * d))
    return value if is_long else -value


def maturity_factor(maturity_years: float, mpor_business_days: int | None = None) -> float:
    """CRE52.48 unmargined sqrt(min(M, 1)); CRE52.52 margined 1.5 sqrt(MPoR / 250). The 10 day MPoR floor is the
    caller's."""
    m = _nonnegative("maturity_years", maturity_years)
    if mpor_business_days is None:
        return math.sqrt(min(m, 1.0))
    mpor = _finite("mpor_business_days", mpor_business_days)
    if mpor <= 0.0:
        raise ValueError(f"mpor_business_days must be positive, got {mpor_business_days!r}")
    return 1.5 * math.sqrt(mpor / BUSINESS_DAYS_PER_YEAR)


def maturity_bucket(end_years: float) -> str:
    """CRE52.57: B1 below 1 year, B2 from 1 to 5 years inclusive, B3 above 5 years."""
    e = _nonnegative("end_years", end_years)
    if e < 1.0:
        return "B1"
    return "B2" if e <= 5.0 else "B3"


def ir_effective_notional(d_b1: float, d_b2: float, d_b3: float, offset: bool = True) -> float:
    """CRE52.57: sqrt(D1^2 + D2^2 + D3^2 + 1.4 D1 D2 + 1.4 D2 D3 + 0.6 D1 D3), or |D1| + |D2| + |D3| without offset."""
    a = _finite("d_b1", d_b1)
    b = _finite("d_b2", d_b2)
    c = _finite("d_b3", d_b3)
    if not offset:
        return abs(a) + abs(b) + abs(c)
    radicand = (
        a * a
        + b * b
        + c * c
        + IR_OFFSET_CROSS_B1_B2 * a * b
        + IR_OFFSET_CROSS_B2_B3 * b * c
        + IR_OFFSET_CROSS_B1_B3 * a * c
    )
    return math.sqrt(max(0.0, radicand))


@dataclass(frozen=True)
class IrTrade:
    """An interest-rate trade: the hedging set (its currency), start and end in years, the notional and the
    supervisory delta."""

    hedging_set: str
    start: float
    end: float
    notional: float
    delta: float


@dataclass(frozen=True)
class IrTradeResult:
    """The intermediates of one interest-rate trade (the `trades` of the `sa_ccr` block of `c12-outputs`)."""

    hedging_set: str
    start: float
    end: float
    notional: float
    delta: float
    supervisory_duration: float
    adjusted_notional: float  # notional x supervisory duration
    maturity_factor: float
    bucket: str
    effective_notional: float  # delta x adjusted notional x maturity factor


@dataclass(frozen=True)
class IrHedgingSet:
    """The intermediates of one hedging set (the `hedging_sets` of the `sa_ccr` block)."""

    hedging_set: str
    d_b1: float
    d_b2: float
    d_b3: float
    offset: bool
    effective_notional: float
    supervisory_factor: float
    addon: float


@dataclass(frozen=True)
class IrAddOn:
    """Per trade, per hedging set and in total: the interest-rate add-on of a netting set."""

    trades: tuple[IrTradeResult, ...]
    hedging_sets: tuple[IrHedgingSet, ...]
    addon: float


def ir_addon(trades: list[IrTrade], mpor_business_days: int | None = None, offset: bool = True) -> IrAddOn:
    """CRE52.34 to CRE52.57: the interest-rate add-on, 0.005 x the effective notional of each hedging set, summed.

    The maturity factor of every trade is the unmargined one from the trade's end date, or the margined one from the
    netting set's MPoR. A payer swap has delta +1 and a receiver -1; hedging sets keep the order of their first trade.
    """
    factor = SUPERVISORY_PARAMETERS[("interest-rate", "supervisory_factor")].value
    rows: list[IrTradeResult] = []
    for trade in trades:
        notional = _nonnegative("notional", trade.notional)
        delta = _finite("delta", trade.delta)
        duration = supervisory_duration(trade.start, trade.end)
        adjusted = notional * duration
        mf = maturity_factor(trade.end, mpor_business_days)
        rows.append(
            IrTradeResult(
                str(trade.hedging_set),
                float(trade.start),
                float(trade.end),
                notional,
                delta,
                duration,
                adjusted,
                mf,
                maturity_bucket(trade.end),
                delta * adjusted * mf,
            )
        )
    sets: list[IrHedgingSet] = []
    for name in dict.fromkeys(r.hedging_set for r in rows):
        d = {"B1": 0.0, "B2": 0.0, "B3": 0.0}
        for r in rows:
            if r.hedging_set == name:
                d[r.bucket] += r.effective_notional
        en = ir_effective_notional(d["B1"], d["B2"], d["B3"], offset)
        sets.append(IrHedgingSet(name, d["B1"], d["B2"], d["B3"], bool(offset), en, factor, factor * en))
    return IrAddOn(tuple(rows), tuple(sets), sum(h.addon for h in sets))


def fx_addon(hedging_sets: dict[str, float]) -> float:
    """CRE52.59: 0.04 x |EN| summed over the currency pairs (EN the signed effective notional in the reporting
    currency)."""
    factor = SUPERVISORY_PARAMETERS[("fx", "supervisory_factor")].value
    return factor * sum(abs(_finite("hedging_sets", en)) for en in hedging_sets.values())


def entity_aggregate(addons: list[float], rhos: list[float]) -> float:
    """CRE52.61, CRE52.66, CRE52.70: sqrt((sum rho_e A_e)^2 + sum (1 - rho_e^2) A_e^2), one rho per entity."""
    if len(addons) != len(rhos):
        raise ValueError(
            f"rhos has {len(rhos)} entries for {len(addons)} addons: one correlation per entity is required"
        )
    a = [_finite("addons", x) for x in addons]
    r = []
    for x in rhos:
        v = _finite("rhos", x)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"rhos must lie in [0, 1], got {x!r}")
        r.append(v)
    systematic = sum(rho * x for rho, x in zip(r, a, strict=True)) ** 2
    idiosyncratic = sum((1.0 - rho * rho) * x * x for rho, x in zip(r, a, strict=True))
    return math.sqrt(systematic + idiosyncratic)


@dataclass(frozen=True)
class CreditEntity:
    """A credit entity: the rating (AAA to CCC for a single name, IG or SG for an index) and its effective notional."""

    rating: str
    is_index: bool
    effective_notional: float


@dataclass(frozen=True)
class EquityEntity:
    """An equity entity (a single name or an index) and its effective notional."""

    is_index: bool
    effective_notional: float


@dataclass(frozen=True)
class CommodityEntity:
    """A commodity entity: its supervisory factor, its correlation and its effective notional."""

    sf: float
    rho: float
    effective_notional: float


def _credit_key(entity: CreditEntity) -> str:
    key = ("credit-index-" if entity.is_index else "credit-single-name-") + str(entity.rating)
    if (key, "supervisory_factor") not in SUPERVISORY_PARAMETERS:
        kind = "index (IG or SG)" if entity.is_index else "single-name (AAA to CCC)"
        raise ValueError(f"rating {entity.rating!r} is not a {kind} row of CRE52.72 Table 2 (CRE52.72)")
    return key


def credit_addon(entities: list[CreditEntity]) -> float:
    """CRE52.61 to CRE52.65: entity add-ons SF x EN with correlation 0.5 for a single name and 0.8 for an index."""
    addons, rhos = [], []
    for e in entities:
        key = _credit_key(e)
        addons.append(
            SUPERVISORY_PARAMETERS[(key, "supervisory_factor")].value
            * _finite("effective_notional", e.effective_notional)
        )
        rhos.append(SUPERVISORY_PARAMETERS[(key, "correlation")].value)
    return entity_aggregate(addons, rhos)


def equity_addon(entities: list[EquityEntity]) -> float:
    """CRE52.66 to CRE52.69: factors 32 and 20 percent, correlations 50 and 80 percent for single names and indices."""
    addons, rhos = [], []
    for e in entities:
        key = "equity-index" if e.is_index else "equity-single-name"
        addons.append(
            SUPERVISORY_PARAMETERS[(key, "supervisory_factor")].value
            * _finite("effective_notional", e.effective_notional)
        )
        rhos.append(SUPERVISORY_PARAMETERS[(key, "correlation")].value)
    return entity_aggregate(addons, rhos)


def commodity_addon(hedging_sets: dict[str, list[CommodityEntity]]) -> float:
    """CRE52.70: the sum over hedging sets of the entity aggregate of SF x EN."""
    total = 0.0
    for entities in hedging_sets.values():
        addons, rhos = [], []
        for e in entities:
            addons.append(_nonnegative("sf", e.sf) * _finite("effective_notional", e.effective_notional))
            rhos.append(e.rho)
        total += entity_aggregate(addons, rhos)
    return total


def replacement_cost(
    v: float, c: float, th: float = 0.0, mta: float = 0.0, nica: float = 0.0, margined: bool = False
) -> float:
    """CRE52.10 max(V - C, 0) unmargined; CRE52.18 max(V - C, TH + MTA - NICA, 0) margined."""
    v_c = _finite("v", v) - _finite("c", c)
    margin_terms = _finite("th", th) + _finite("mta", mta) - _finite("nica", nica)
    if not margined:
        return max(v_c, 0.0)
    return max(v_c, margin_terms, 0.0)


def multiplier(v_minus_c: float, addon_aggregate: float, floor: float = MULTIPLIER_FLOOR) -> float:
    """CRE52.23: min(1, floor + (1 - floor) exp((V - C) / (2 (1 - floor) AddOn))); 1 for V - C >= 0 and for a zero
    add-on."""
    x = _finite("v_minus_c", v_minus_c)
    addon = _nonnegative("addon_aggregate", addon_aggregate)
    fl = _finite("floor", floor)
    if not 0.0 <= fl < 1.0:
        raise ValueError(f"floor must lie in [0, 1), got {floor!r}")
    if x >= 0.0 or addon == 0.0:
        return 1.0
    return min(1.0, fl + (1.0 - fl) * math.exp(x / (2.0 * (1.0 - fl) * addon)))


@dataclass(frozen=True)
class AddOnByClass:
    """The add-ons of the five asset classes of a netting set, in one currency (CRE52.24 to CRE52.25)."""

    interest_rate: float = 0.0
    fx: float = 0.0
    credit: float = 0.0
    equity: float = 0.0
    commodity: float = 0.0


@dataclass(frozen=True)
class SaCcrResult:
    """RC, the aggregate add-on, the multiplier, PFE and EAD of a netting set."""

    rc: float
    addon_aggregate: float
    multiplier: float
    pfe: float
    ead: float
    alpha: float


def netting_set_ead(
    v: float,
    c: float,
    addon_by_class: AddOnByClass,
    margined: bool = False,
    th: float = 0.0,
    mta: float = 0.0,
    nica: float = 0.0,
    alpha: float = ALPHA,
) -> SaCcrResult:
    """CRE52.1: EAD = alpha (RC + multiplier x AddOn^aggregate), the aggregate being the sum of the five classes."""
    a = _nonnegative("alpha", alpha)
    if a == 0.0:
        raise ValueError("alpha must be positive, got 0")
    parts = [
        _nonnegative(name, getattr(addon_by_class, name))
        for name in ("interest_rate", "fx", "credit", "equity", "commodity")
    ]
    aggregate = sum(parts)
    rc = replacement_cost(v, c, th, mta, nica, margined)
    mult = multiplier(float(v) - float(c), aggregate)
    pfe = mult * aggregate
    return SaCcrResult(rc, aggregate, mult, pfe, a * (rc + pfe), a)
