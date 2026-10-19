"""The Liquidity Coverage Ratio and the Net Stable Funding Ratio as Basel formulas on aggregates, with the parameters as
data: each with its paragraph and the in-force date 2019-12-15 (dossier 01 G.1 and G.2, M.3 item 8).

Sources: BCBS Basel Framework LCR20 (published 2022-12-08), LCR30 (2020-06-05), LCR40 (2023-03-30), LCR99 (2019-12-15),
NSF20, NSF30 and NSF99 Tables 1 to 3, all in force 2019-12-15, https://www.bis.org/basel_framework/ . The text is never
reproduced, only the parameters. Contract `engine-liquidity-api` 1.0.0, module regulatory.lcr_nsfr.

The categories are inputs the caller maps to (the C16 mapping table); the engine does not map customers to categories.
The LCR computed here is a Basel formula applied to aggregates with stated assumptions: it is never the bank's reported
or regulatory LCR, and the label is the caller's, required in the document. No official LCR or NSFR worked example
exists (dossier 01 lines 1045 to 1046), so none is claimed: the golden values are constructed (01 M.1). The engine
computes and states no compliance claim. Money is USD thousands as the call report gives it; ratios are decimals (1.0
is 100 percent).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

IN_FORCE = "2019-12-15"
LCR_INFLOW_CAP = 0.75               # inflows are capped at 75 percent of gross outflows (LCR40.1, LCR99)
TRADE_FINANCE_KEY = "trade_finance"
TRADE_FINANCE_MAX_RATE = 0.05       # trade finance contingent outflows: 0 to 5 percent at national discretion (LCR99)
NSFR_DERIVATIVE_ADDON_MIN = 0.05    # NSF30.8: 5 percent to 20 percent of derivative liabilities
NSFR_DERIVATIVE_ADDON_MAX = 0.20

_LCR99 = "LCR99 "
_LCR40 = "LCR40 "

# key: (rate, reference), dossier 01 G.1 "Key outflow rates (LCR99 summary table)"
_OUTFLOWS: dict[str, tuple[float, str]] = {
    "retail_stable_insurance_criteria": (0.03,
        _LCR99 + "retail stable deposits, deposit insurance scheme meets additional criteria"),
    "retail_stable": (0.05, _LCR99 + "retail stable deposits"),
    "retail_less_stable": (0.10, _LCR99 + "retail less stable deposits"),
    "retail_term_gt_30d": (0.00, _LCR99 + "retail term deposits with residual maturity over 30 days"),
    "small_business_stable": (0.05, _LCR99 + "small business stable"),
    "small_business_less_stable": (0.10, _LCR99 + "small business less stable"),
    "operational": (0.25, _LCR99 + "operational deposits (insured portion 0.05)"),
    "operational_insured_portion": (0.05, _LCR99 + "operational deposits, insured portion"),
    "cooperative_network": (0.25, _LCR99 + "cooperative network qualifying deposits"),
    "non_financial_corporate": (0.40, _LCR99 + "non-financial corporates, sovereigns, central banks, MDBs, PSEs"),
    "non_financial_corporate_fully_insured": (0.20, _LCR99 + "the same when fully insured"),
    "other_legal_entity": (1.00, _LCR99 + "other legal entity customers (for example financial institutions)"),
    "secured_central_bank": (0.00, _LCR40 + "secured funding with a central bank"),
    "secured_level1": (0.00, _LCR40 + "secured funding backed by Level 1"),
    "secured_level2a": (0.15, _LCR40 + "secured funding backed by Level 2A"),
    "secured_sovereign_mdb_pse": (0.25,
        _LCR40 + "backed by non-L1, non-L2A assets, domestic sovereign, MDB or PSE counterparty"),
    "secured_level2b_rmbs": (0.25, _LCR40 + "backed by eligible Level 2B RMBS"),
    "secured_other_level2b": (0.50, _LCR40 + "backed by other Level 2B"),
    "secured_other": (1.00, _LCR40 + "all other secured funding"),
    "non_level1_collateral_valuation": (0.20, _LCR99 + "valuation changes on non-Level-1 posted collateral"),
    "undrawn_retail_small_business": (0.05, _LCR99 + "undrawn committed facilities, retail and small business"),
    "undrawn_credit_non_financial_corporate": (0.10,
        _LCR99 + "undrawn credit facilities, non-financial corporates, sovereigns, MDBs, PSEs"),
    "undrawn_liquidity_non_financial_corporate": (0.30,
        _LCR99 + "undrawn liquidity facilities, the same counterparties"),
    "undrawn_credit_supervised_bank": (0.40, _LCR99 + "undrawn facilities, supervised banks"),
    "undrawn_liquidity_supervised_bank": (0.40, _LCR99 + "undrawn facilities, supervised banks"),
    "undrawn_credit_other_fi": (0.40, _LCR99 + "undrawn credit facilities, other financial institutions"),
    "undrawn_liquidity_other_fi": (1.00, _LCR99 + "undrawn liquidity facilities, other financial institutions"),
    "undrawn_other_legal_entity": (1.00, _LCR99 + "undrawn facilities, other legal entities"),
    "net_derivative_outflow": (1.00, _LCR99 + "net derivative outflows"),
    "other_contractual_outflow": (1.00, _LCR99 + "other contractual outflows"),
}
_INFLOWS: dict[str, tuple[float, str]] = {
    "secured_lending_level1": (0.00, _LCR99 + "maturing secured lending backed by Level 1"),
    "secured_lending_level2a": (0.15, _LCR99 + "backed by Level 2A"),
    "secured_lending_level2b_rmbs": (0.25, _LCR99 + "backed by eligible RMBS"),
    "secured_lending_other_level2b": (0.50, _LCR99 + "backed by other Level 2B"),
    "secured_lending_other": (1.00, _LCR99 + "other collateral"),
    "margin_lending_other_collateral": (0.50, _LCR99 + "margin lending on other collateral"),
    "facilities_granted_to_bank": (0.00, _LCR99 + "facilities granted to the bank"),
    "operational_deposits_at_other_institutions": (0.00, _LCR99 + "operational deposits held at other institutions"),
    "retail_and_nonfinancial_wholesale": (0.50, _LCR99 + "retail and non-financial wholesale inflows"),
    "financial_institutions_and_central_banks": (1.00, _LCR99 + "financial institutions and central banks"),
    "net_derivative_inflow": (1.00, _LCR99 + "net derivative inflows"),
}
_HAIRCUTS: dict[str, tuple[float, str]] = {
    "level_1": (0.00, "LCR30 Level 1, no haircut (factor 100 percent)"),
    "level_2a": (0.15, "LCR30.43 Level 2A"),
    "level_2b_rmbs": (0.25, "LCR30.45 qualifying RMBS"),
    "level_2b_corporate_debt": (0.50, "LCR30.45 qualifying corporate debt A+ to BBB-"),
    "level_2b_common_equity": (0.50, "LCR30.45 qualifying common equity"),
    "level_2b_sovereign_bbb": (0.50, "LCR30.45 sovereign BBB+ to BBB- within the 15 percent cap (FAQ)"),
}
_NSF99_ASF = "NSF99 ASF: "
_NSF99_RSF = "NSF99 RSF: "
_ASF: dict[str, tuple[float, str]] = {
    "regulatory_capital": (1.00,
        _NSF99_ASF + "regulatory capital excluding Tier 2 with residual maturity under one year"),
    "long_term_liabilities_ge_1y": (1.00,
        _NSF99_ASF + "other capital and liabilities with effective residual maturity of one year or more"),
    "stable_retail_deposits": (0.95,
        _NSF99_ASF + "stable retail and small business deposits (non-maturity or under one year)"),
    "less_stable_retail_deposits": (0.90, _NSF99_ASF + "less stable retail and small business deposits"),
    "funding_lt_1y_non_financial_corporates": (0.50,
        _NSF99_ASF + "funding under one year from non-financial corporates"),
    "operational_deposits": (0.50, _NSF99_ASF + "operational deposits"),
    "funding_lt_1y_sovereigns_pse_mdb_ndb": (0.50,
        _NSF99_ASF + "funding under one year from sovereigns, PSEs, MDBs, NDBs"),
    "other_funding_6m_to_1y": (0.50,
        _NSF99_ASF + "other funding of 6 months to under one year including central banks and financial institutions"),
    "other_liabilities": (0.00,
        _NSF99_ASF + "all other liabilities and equity, net derivative liabilities, trade-date payables"),
}
_RSF: dict[str, tuple[float, str]] = {
    "coins_banknotes_reserves": (0.00,
        _NSF99_RSF + "coins, banknotes, central bank reserves, central bank claims under 6 months, trade-date "
                     "receivables"),
    "unencumbered_level_1": (0.05, _NSF99_RSF + "unencumbered Level 1 excluding cash and reserves"),
    "loans_fi_lt_6m_level1_secured": (0.10,
        _NSF99_RSF + "loans to financial institutions under 6 months secured by Level 1 with rehypothecation ability"),
    "loans_fi_lt_6m_other": (0.15, _NSF99_RSF + "other unencumbered loans to financial institutions under 6 months"),
    "unencumbered_level_2a": (0.15, _NSF99_RSF + "unencumbered Level 2A"),
    "unencumbered_level_2b": (0.50, _NSF99_RSF + "unencumbered Level 2B"),
    "encumbered_hqla_6m_to_1y": (0.50, _NSF99_RSF + "HQLA encumbered 6 months to under one year"),
    "loans_fi_6m_to_1y": (0.50,
        _NSF99_RSF + "loans to financial institutions and central banks 6 months to under one year"),
    "other_assets_lt_1y": (0.50, _NSF99_RSF + "all other assets under one year (corporate, retail, sovereign loans)"),
    "residential_mortgages_ge_1y_rw35": (0.65,
        _NSF99_RSF + "unencumbered residential mortgages of one year or more, SA risk weight at most 35 percent"),
    "other_loans_ge_1y_rw35": (0.65, _NSF99_RSF + "other loans of one year or more, risk weight at most 35 percent"),
    "initial_margin_ccp_default_fund": (0.85, _NSF99_RSF + "initial margin and CCP default fund"),
    "other_loans_ge_1y_rw_gt_35": (0.85,
        _NSF99_RSF + "other performing loans of one year or more, risk weight above 35 percent"),
    "non_hqla_securities_ge_1y": (0.85,
        _NSF99_RSF + "non-HQLA securities of one year or more and exchange-traded equities"),
    "physical_commodities": (0.85, _NSF99_RSF + "physical commodities including gold"),
    "encumbered_ge_1y": (1.00, _NSF99_RSF + "assets encumbered one year or more"),
    "net_derivative_assets": (1.00, _NSF99_RSF + "net derivative assets"),
    "other_assets": (1.00, _NSF99_RSF + "all other assets including NPLs and fixed assets"),
    "undrawn_facilities_obs": (0.05,
        _NSF99_RSF + "irrevocable and conditionally revocable credit and liquidity facilities, 5 percent of the "
                     "undrawn amount"),
}
OBS_KEYS = frozenset({"undrawn_facilities_obs"})  # the off-balance-sheet keys of the RSF table

LCR_OUTFLOW_RATES: dict[str, float] = {k: v[0] for k, v in _OUTFLOWS.items()}
LCR_OUTFLOW_REFERENCES: dict[str, str] = {k: v[1] for k, v in _OUTFLOWS.items()}
LCR_INFLOW_RATES: dict[str, float] = {k: v[0] for k, v in _INFLOWS.items()}
LCR_INFLOW_REFERENCES: dict[str, str] = {k: v[1] for k, v in _INFLOWS.items()}
HQLA_HAIRCUTS: dict[str, float] = {k: v[0] for k, v in _HAIRCUTS.items()}
HQLA_REFERENCES: dict[str, str] = {k: v[1] for k, v in _HAIRCUTS.items()}
NSFR_ASF_FACTORS: dict[str, float] = {k: v[0] for k, v in _ASF.items()}
NSFR_ASF_REFERENCES: dict[str, str] = {k: v[1] for k, v in _ASF.items()}
NSFR_RSF_FACTORS: dict[str, float] = {k: v[0] for k, v in _RSF.items()}
NSFR_RSF_REFERENCES: dict[str, str] = {k: v[1] for k, v in _RSF.items()}
TRADE_FINANCE_REFERENCE = _LCR99 + "trade finance contingent outflows, 0 to 5 percent at national discretion"


@dataclass(frozen=True)
class HqlaResult:
    """The stock of HQLA after haircuts and the LCR30.39 adjustments, with the Level 2 shares of the stock."""

    l1: float
    l2a: float
    l2b: float
    adj_15: float
    adj_40: float
    hqla: float
    l2_share: float
    l2b_share: float


@dataclass(frozen=True)
class LcrRow:
    """One category of an LCR outflow or inflow table: the balance, the rate applied and the amount."""

    key: str
    balance: float
    rate: float
    amount: float
    reference: str


@dataclass(frozen=True)
class OutflowResult:
    total: float
    rows: tuple[LcrRow, ...]


@dataclass(frozen=True)
class NetOutflowResult:
    inflows_counted: float
    net: float
    cap_binds: bool


def _amount(value: float, name: str, *, positive: bool = False) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: must be numeric, got {value!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"{name}: must be finite, got {value!r}")
    if x < 0.0:
        raise ValueError(f"{name}: must be non-negative, got {x}")
    if positive and x == 0.0:
        raise ValueError(f"{name}: must be positive, got {x}")
    return x


def hqla_stock(l1: float, l2a: float, l2b: float, adjusted_l1: float | None = None, adjusted_l2a: float | None = None,
               adjusted_l2b: float | None = None) -> HqlaResult:
    """`HQLA = L1 + L2A + L2B - Adj15 - Adj40` (LCR30.38) with the amounts after haircuts.

    `Adj15 = max(aL2B - 15/85 (aL1 + aL2A), aL2B - 15/60 aL1, 0)` and `Adj40 = max(aL2A + aL2B - Adj15 - 2/3 aL1, 0)`
    (LCR30.39), from the adjusted amounts (after unwinding short-term SFTs and collateral swaps maturing within 30
    days, LCR30.37), which default to the unadjusted ones. Level 1 is unlimited; Level 2 at most 0.40 of HQLA
    (LCR30.31) and Level 2B at most 0.15 (LCR30.33), both after haircuts (LCR30.34); adjusted Level 2 at most
    two-thirds of adjusted Level 1 (LCR30.35). `l2b_share = (L2B - Adj15) / HQLA` and `l2_share = (L2A + L2B - Adj15 -
    Adj40) / HQLA`, both 0 for an empty stock. Money in USD thousands."""
    a1, a2a, a2b = _amount(l1, "l1"), _amount(l2a, "l2a"), _amount(l2b, "l2b")
    j1 = a1 if adjusted_l1 is None else _amount(adjusted_l1, "adjusted_l1")
    j2a = a2a if adjusted_l2a is None else _amount(adjusted_l2a, "adjusted_l2a")
    j2b = a2b if adjusted_l2b is None else _amount(adjusted_l2b, "adjusted_l2b")
    adj15 = max(j2b - 15.0 / 85.0 * (j1 + j2a), j2b - 15.0 / 60.0 * j1, 0.0)
    adj40 = max(j2a + j2b - adj15 - 2.0 / 3.0 * j1, 0.0)
    hqla = a1 + a2a + a2b - adj15 - adj40
    l2b_share = (a2b - adj15) / hqla if hqla > 0.0 else 0.0
    l2_share = (a2a + a2b - adj15 - adj40) / hqla if hqla > 0.0 else 0.0
    return HqlaResult(l1=a1, l2a=a2a, l2b=a2b, adj_15=adj15, adj_40=adj40, hqla=hqla, l2_share=l2_share,
                      l2b_share=l2b_share)


def _rate_table(balances: Mapping[str, float], rates: Mapping[str, float] | None, table: Mapping[str, float],
                references: Mapping[str, str], *, trade_finance: bool) -> OutflowResult:
    overrides = dict(rates or {})
    for key in overrides:
        if key not in table and not (trade_finance and key == TRADE_FINANCE_KEY):
            raise KeyError(f"{key}: not a key of the rate table")
    rows = []
    for key, balance in balances.items():
        if key == TRADE_FINANCE_KEY and trade_finance:
            if key not in overrides:
                raise ValueError(f"{key}: the rate is national discretion within 0 to {TRADE_FINANCE_MAX_RATE} and "
                                 "must be supplied in rates, never a default")
            rate = float(overrides[key])
            if not 0.0 <= rate <= TRADE_FINANCE_MAX_RATE:
                raise ValueError(f"{key}: rate must lie in [0, {TRADE_FINANCE_MAX_RATE}], got {rate}")
            reference = TRADE_FINANCE_REFERENCE
        elif key in table:
            rate = float(overrides.get(key, table[key]))
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{key}: rate must lie in [0, 1], got {rate}")
            reference = references[key]
        else:
            raise KeyError(f"{key}: not a key of the rate table")
        amount_in = _amount(balance, key)
        rows.append(LcrRow(key=key, balance=amount_in, rate=rate, amount=amount_in * rate, reference=reference))
    return OutflowResult(total=math.fsum(row.amount for row in rows), rows=tuple(rows))


def lcr_outflows(balances: Mapping[str, float], rates: Mapping[str, float] | None = None) -> OutflowResult:
    """`amount = balance x rate` per category of `LCR_OUTFLOW_RATES` (LCR40 and LCR99), the total their sum.

    `rates` overrides a rate (a national discretion) and supplies the trade finance rate within 0 to 0.05, which has
    no default: a balance under the key `trade_finance` without a rate raises. A key outside the table raises
    `KeyError` naming it. The engine does not map customers to categories."""
    return _rate_table(balances, rates, LCR_OUTFLOW_RATES, LCR_OUTFLOW_REFERENCES, trade_finance=True)


def lcr_inflows(balances: Mapping[str, float], rates: Mapping[str, float] | None = None) -> OutflowResult:
    """As `lcr_outflows` with the keys of `LCR_INFLOW_RATES`: the gross inflow before the cap (LCR99)."""
    return _rate_table(balances, rates, LCR_INFLOW_RATES, LCR_INFLOW_REFERENCES, trade_finance=False)


def net_cash_outflows(outflows: float, inflows: float, cap: float = LCR_INFLOW_CAP) -> NetOutflowResult:
    """`net = outflows - min(inflows, 0.75 x outflows)` (LCR40.1): inflows never offset more than 0.75 of outflows, so
    `net >= 0.25 x outflows`. `cap_binds` is `inflows > cap x outflows`."""
    out, inn = _amount(outflows, "outflows"), _amount(inflows, "inflows")
    c = float(cap)
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"cap: must lie in [0, 1], got {cap}")
    counted = min(inn, c * out)
    return NetOutflowResult(inflows_counted=counted, net=out - counted, cap_binds=inn > c * out)


def lcr_ratio(hqla: float, net_outflows: float) -> float:
    """`LCR = HQLA / net outflows` (LCR20.4, at least 1.0 required in normal times; a ratio below 1.0 under stress is
    not a breach, dossier 01 J.3). A net outflow at or below zero raises: the ratio is undefined, never infinite. The
    label "a Basel formula on call-report aggregates, never the bank's reported or regulatory LCR" is the caller's."""
    h = _amount(hqla, "hqla")
    n = float(net_outflows)
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"net_outflows: must be positive for the ratio to be defined, got {net_outflows}")
    return h / n


def _weighted(amounts: Mapping[str, float], table: Mapping[str, float], factors: Mapping[str, float] | None,
              *, allowed: frozenset[str] | None = None, excluded: frozenset[str] = frozenset()) -> float:
    effective = dict(table)
    for key, factor in (factors or {}).items():
        if key not in table:
            raise KeyError(f"{key}: not a key of the factor table")
        f = float(factor)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"{key}: factor must lie in [0, 1], got {factor}")
        effective[key] = f
    total = []
    for key, amount in amounts.items():
        if key not in effective or key in excluded or (allowed is not None and key not in allowed):
            raise KeyError(f"{key}: not a key of the factor table")
        total.append(effective[key] * _amount(amount, key))
    return math.fsum(total)


def nsfr_asf(liabilities: Mapping[str, float], factors: Mapping[str, float] | None = None) -> float:
    """Available stable funding `ASF = sum of factor x amount` (NSF20.2, NSF99 Table 1) by key of
    `NSFR_ASF_FACTORS`; `factors` overrides a factor of the table. Unknown keys raise `KeyError`, negative amounts
    raise `ValueError`."""
    return _weighted(liabilities, NSFR_ASF_FACTORS, factors)


def nsfr_rsf(assets: Mapping[str, float], obs: Mapping[str, float] | None = None,
             factors: Mapping[str, float] | None = None, derivative_liabilities: float = 0.0,
             derivative_liabilities_addon: float | None = None) -> float:
    """Required stable funding `RSF = sum of factor x asset + sum of factor x OBS` (NSF20.2, NSF99 Tables 2 and 3) plus
    the derivative liabilities add-on `derivative_liabilities x derivative_liabilities_addon` (NSF30.8).

    `obs` holds the off-balance-sheet undrawn amounts (key `undrawn_facilities_obs`). The add-on is a caller-supplied
    rate within `[0.05, 0.20]`, never a default (the in-force value is the supervisor's): a rate that is missing or
    outside the interval raises `ValueError` naming `derivative_liabilities_addon` when derivative liabilities are
    positive, and a rate that is given is always checked. Unknown keys raise `KeyError`."""
    body = _weighted(assets, NSFR_RSF_FACTORS, factors, excluded=OBS_KEYS)
    off = _weighted(obs or {}, NSFR_RSF_FACTORS, factors, allowed=OBS_KEYS)
    deriv = _amount(derivative_liabilities, "derivative_liabilities")
    addon = 0.0
    if derivative_liabilities_addon is not None:
        rate = float(derivative_liabilities_addon)
        if not NSFR_DERIVATIVE_ADDON_MIN <= rate <= NSFR_DERIVATIVE_ADDON_MAX:
            raise ValueError(f"derivative_liabilities_addon: must lie in [{NSFR_DERIVATIVE_ADDON_MIN}, "
                             f"{NSFR_DERIVATIVE_ADDON_MAX}] (NSF30.8), got {derivative_liabilities_addon}")
        addon = deriv * rate
    elif deriv > 0.0:
        raise ValueError(f"derivative_liabilities_addon: a rate within [{NSFR_DERIVATIVE_ADDON_MIN}, "
                         f"{NSFR_DERIVATIVE_ADDON_MAX}] is required for positive derivative liabilities, never a "
                         "default (NSF30.8), got None")
    return body + off + addon


def nsfr_ratio(asf: float, rsf: float) -> float:
    """`NSFR = ASF / RSF` (NSF20.2, at least 1.0). `rsf <= 0` raises; monotone increasing in ASF and decreasing in
    RSF."""
    a = _amount(asf, "asf")
    r = float(rsf)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError(f"rsf: must be positive for the ratio to be defined, got {rsf}")
    return a / r


__all__ = ["HQLA_HAIRCUTS", "HQLA_REFERENCES", "HqlaResult", "IN_FORCE", "LCR_INFLOW_CAP", "LCR_INFLOW_RATES",
           "LCR_INFLOW_REFERENCES", "LCR_OUTFLOW_RATES", "LCR_OUTFLOW_REFERENCES", "LcrRow", "NSFR_ASF_FACTORS",
           "NSFR_ASF_REFERENCES", "NSFR_DERIVATIVE_ADDON_MAX", "NSFR_DERIVATIVE_ADDON_MIN", "NSFR_RSF_FACTORS",
           "NSFR_RSF_REFERENCES", "NetOutflowResult", "OBS_KEYS", "OutflowResult", "TRADE_FINANCE_KEY",
           "TRADE_FINANCE_MAX_RATE", "TRADE_FINANCE_REFERENCE", "hqla_stock", "lcr_inflows", "lcr_outflows",
           "lcr_ratio", "net_cash_outflows", "nsfr_asf", "nsfr_ratio", "nsfr_rsf"]
