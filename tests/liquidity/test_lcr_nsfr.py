"""RV-1119 to RV-1122: the LCR and NSFR calculators (contract `engine-liquidity-api`, module regulatory.lcr_nsfr;
dossier 01 G.1 and G.2). No official worked example exists (dossier 01 lines 1045 to 1046): these are constructed tests
and invariants (dossier 01 M.1), recomputed here through a second route. The directory is `tests/liquidity/` because
U9-E1 owns `tests/regulatory/`."""

from __future__ import annotations

import dataclasses
import math
from types import MappingProxyType

import pytest

from riskvalidation.regulatory import lcr_nsfr as m


def hqla_by_definition(l1, l2a, l2b, al1=None, al2a=None, al2b=None):
    """The LCR30.38 and LCR30.39 formulas written out with the caps as tests of the result: the second route."""
    al1, al2a, al2b = (l1 if al1 is None else al1), (l2a if al2a is None else al2a), (l2b if al2b is None else al2b)
    adj15 = max(al2b - 15.0 / 85.0 * (al1 + al2a), al2b - 15.0 / 60.0 * al1, 0.0)
    adj40 = max(al2a + al2b - adj15 - 2.0 / 3.0 * al1, 0.0)
    return adj15, adj40, l1 + l2a + l2b - adj15 - adj40


def test_hqla_stock_goldens_and_caps():
    r = m.hqla_stock(100.0, 20.0, 30.0)
    assert isinstance(r, m.HqlaResult)
    assert r.adj_15 == pytest.approx(8.823529, abs=5e-7) and r.adj_40 == 0.0
    assert r.hqla == pytest.approx(141.176471, abs=5e-7) and r.l2b_share == pytest.approx(0.15, abs=1e-12)
    r = m.hqla_stock(100.0, 80.0, 0.0)
    assert r.adj_15 == 0.0 and r.adj_40 == pytest.approx(13.333333, abs=5e-7)
    assert r.hqla == pytest.approx(166.666667, abs=5e-7) and r.l2_share == pytest.approx(0.40, abs=1e-12)
    r = m.hqla_stock(100.0, 0.0, 0.0)
    assert r.hqla == 100.0 and r.l2_share == 0.0 and r.l2b_share == 0.0
    assert (r.l1, r.l2a, r.l2b) == (100.0, 0.0, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.hqla = 0.0  # type: ignore[misc]
    # a grid of balance sheets: the formulas, and the three caps hold after the adjustments
    for l1 in (0.0, 10.0, 100.0, 1000.0):
        for l2a in (0.0, 5.0, 60.0, 400.0):
            for l2b in (0.0, 5.0, 30.0, 500.0):
                if l1 + l2a + l2b == 0.0:
                    continue
                r = m.hqla_stock(l1, l2a, l2b)
                a15, a40, h = hqla_by_definition(l1, l2a, l2b)
                assert (r.adj_15, r.adj_40) == pytest.approx((a15, a40), abs=1e-9)
                assert r.hqla == pytest.approx(h, abs=1e-9)
                assert r.hqla == pytest.approx(l1 + l2a + l2b - r.adj_15 - r.adj_40, abs=1e-12)
                l2 = l2a + l2b - r.adj_15 - r.adj_40
                assert l2 <= 0.40 * r.hqla + 1e-9  # Level 2 at most 40 percent of HQLA (LCR30.31)
                assert (l2b - r.adj_15) <= 0.15 * r.hqla + 1e-9  # Level 2B at most 15 percent (LCR30.33)
                assert (l2b - r.adj_15) <= 15.0 / 85.0 * (l1 + l2a) + 1e-9  # 15/85 of L1 plus L2A
                assert l2 <= 2.0 / 3.0 * l1 + 1e-9  # two-thirds of Level 1 (LCR30.35)
                if r.hqla > 0.0:
                    assert r.l2b_share == pytest.approx((l2b - r.adj_15) / r.hqla, abs=1e-12)
                    assert r.l2_share == pytest.approx(l2 / r.hqla, abs=1e-12)
                else:  # an empty stock has no shares to report: NaN, never 0 (F11)
                    assert math.isnan(r.l2_share) and math.isnan(r.l2b_share)
    for bad in ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, -1.0)):
        with pytest.raises(ValueError, match="^l"):
            m.hqla_stock(*bad)


def test_hqla_stock_adjusted_amounts():
    # the adjusted amounts (after unwinding short-term SFTs, LCR30.37) drive the adjustments; the stock is unadjusted
    r = m.hqla_stock(100.0, 20.0, 30.0, adjusted_l1=80.0, adjusted_l2a=20.0, adjusted_l2b=30.0)
    a15, a40, _ = hqla_by_definition(100.0, 20.0, 30.0, 80.0, 20.0, 30.0)
    assert r.adj_15 == pytest.approx(a15, abs=1e-12) and r.adj_40 == pytest.approx(a40, abs=1e-12)
    assert r.hqla == pytest.approx(150.0 - a15 - a40, abs=1e-12)
    assert a15 > 8.823529  # a smaller Level 1 binds harder
    # a default is the unadjusted amount, one at a time
    assert m.hqla_stock(100.0, 20.0, 30.0, adjusted_l1=100.0) == m.hqla_stock(100.0, 20.0, 30.0)
    with pytest.raises(ValueError, match="^adjusted_l2b: "):
        m.hqla_stock(100.0, 20.0, 30.0, adjusted_l2b=-5.0)


def test_lcr_outflows_goldens_and_refusals():
    r = m.lcr_outflows({"retail_stable": 1000.0})
    assert isinstance(r, m.OutflowResult) and r.total == pytest.approx(50.0, abs=1e-12)
    r = m.lcr_outflows({"retail_less_stable": 100.0, "non_financial_corporate": 100.0, "other_legal_entity": 10.0})
    assert r.total == pytest.approx(10.0 + 40.0 + 10.0, abs=1e-12) and r.total == pytest.approx(60.0, abs=1e-12)
    assert [row.key for row in r.rows] == ["retail_less_stable", "non_financial_corporate", "other_legal_entity"]
    first = r.rows[0]
    assert (first.balance, first.rate, first.amount) == (100.0, 0.10, pytest.approx(10.0))
    assert "LCR99" in first.reference
    assert r.total == pytest.approx(sum(row.amount for row in r.rows), abs=1e-12)
    # a rate override (a national discretion), and the trade finance rate supplied within 0 to 0.05
    assert m.lcr_outflows({"retail_stable": 100.0}, {"retail_stable": 0.08}).total == pytest.approx(8.0)
    assert m.lcr_outflows({"trade_finance": 200.0}, {"trade_finance": 0.03}).total == pytest.approx(6.0)
    with pytest.raises(ValueError, match="trade_finance"):  # never a default
        m.lcr_outflows({"trade_finance": 200.0})
    with pytest.raises(ValueError, match="trade_finance"):
        m.lcr_outflows({"trade_finance": 200.0}, {"trade_finance": 0.2})
    with pytest.raises(KeyError, match="no_such_category"):
        m.lcr_outflows({"no_such_category": 1.0})
    with pytest.raises(KeyError, match="bad_key"):
        m.lcr_outflows({"retail_stable": 1.0}, {"bad_key": 0.1})
    with pytest.raises(ValueError, match="retail_stable"):
        m.lcr_outflows({"retail_stable": -1.0})
    with pytest.raises(ValueError, match="retail_stable"):
        m.lcr_outflows({"retail_stable": 1.0}, {"retail_stable": 1.5})
    assert m.lcr_outflows({}).total == 0.0
    # every key of the table gives amount = balance x rate
    for key, rate in m.LCR_OUTFLOW_RATES.items():
        assert m.lcr_outflows({key: 7.0}).total == pytest.approx(7.0 * rate, abs=1e-12), key


def test_lcr_inflows_goldens():
    assert m.lcr_inflows({"retail_and_nonfinancial_wholesale": 100.0}).total == pytest.approx(50.0)
    assert m.lcr_inflows({"secured_lending_level2a": 100.0}).total == pytest.approx(15.0)
    both = m.lcr_inflows({"retail_and_nonfinancial_wholesale": 100.0, "secured_lending_level2a": 100.0})
    assert both.total == pytest.approx(65.0) and len(both.rows) == 2
    with pytest.raises(KeyError, match="retail_stable"):  # an outflow key is not an inflow key
        m.lcr_inflows({"retail_stable": 1.0})
    with pytest.raises(ValueError, match="secured_lending_other"):
        m.lcr_inflows({"secured_lending_other": -1.0})
    for key, rate in m.LCR_INFLOW_RATES.items():
        assert m.lcr_inflows({key: 3.0}).total == pytest.approx(3.0 * rate, abs=1e-12), key


def test_net_cash_outflows_golden_and_cap():
    r = m.net_cash_outflows(100.0, 90.0)
    assert isinstance(r, m.NetOutflowResult)
    assert (r.inflows_counted, r.net, r.cap_binds) == (75.0, 25.0, True)
    r = m.net_cash_outflows(100.0, 50.0)
    assert (r.inflows_counted, r.net, r.cap_binds) == (50.0, 50.0, False)
    # inflows never offset more than 75 percent of outflows: net is at least 25 percent of outflows
    for outflows in (10.0, 100.0, 1e6):
        for inflows in (0.0, 1.0, 74.0, 75.0, 76.0, 1e7):
            r = m.net_cash_outflows(outflows, inflows)
            assert r.net == pytest.approx(outflows - min(inflows, 0.75 * outflows), abs=1e-9)
            assert r.net >= 0.25 * outflows - 1e-9
            assert r.cap_binds == (inflows > 0.75 * outflows)
    assert not m.net_cash_outflows(100.0, 75.0).cap_binds  # exactly at the cap does not bind
    assert m.net_cash_outflows(100.0, 90.0, cap=0.5).net == 50.0
    with pytest.raises(ValueError, match="^outflows: "):
        m.net_cash_outflows(-1.0, 1.0)
    with pytest.raises(ValueError, match="^inflows: "):
        m.net_cash_outflows(1.0, -1.0)
    with pytest.raises(ValueError, match="^cap: "):
        m.net_cash_outflows(1.0, 1.0, cap=1.5)


def test_lcr_ratio_golden_and_refusals():
    # a balance sheet of Level 1 and stable retail deposits only: HQLA 100 against 5 percent of 1000
    nco = m.lcr_outflows({"retail_stable": 1000.0}).total
    assert m.lcr_ratio(100.0, nco) == pytest.approx(2.0, abs=1e-12)
    assert m.lcr_ratio(0.0, 10.0) == 0.0
    # the whole chain on a constructed balance sheet, by the formula: LCR = HQLA / (O - min(I, 0.75 O))
    hq = m.hqla_stock(100.0, 20.0, 30.0).hqla
    out = m.lcr_outflows({"retail_less_stable": 400.0, "non_financial_corporate": 200.0}).total  # 40 + 80
    inn = m.lcr_inflows({"retail_and_nonfinancial_wholesale": 300.0}).total  # 150, capped at 90
    net = m.net_cash_outflows(out, inn)
    assert (out, inn, net.net) == (pytest.approx(120.0), pytest.approx(150.0), pytest.approx(30.0))
    assert m.lcr_ratio(hq, net.net) == pytest.approx(hq / (120.0 - 0.75 * 120.0), abs=1e-12)
    with pytest.raises(ValueError, match="net_outflows"):
        m.lcr_ratio(100.0, 0.0)
    with pytest.raises(ValueError, match="net_outflows"):
        m.lcr_ratio(100.0, -5.0)
    with pytest.raises(ValueError, match="^hqla: "):
        m.lcr_ratio(-1.0, 5.0)


def test_nsfr_asf_golden():
    assert m.nsfr_asf({"regulatory_capital": 10.0, "stable_retail_deposits": 100.0}) == pytest.approx(105.0)
    assert m.nsfr_asf({}) == 0.0
    assert m.nsfr_asf({"operational_deposits": 80.0}) == pytest.approx(40.0)
    assert m.nsfr_asf({"other_liabilities": 1e6}) == 0.0
    # a factor override replaces the factor of a key of the table
    assert m.nsfr_asf({"stable_retail_deposits": 100.0}, {"stable_retail_deposits": 0.9}) == pytest.approx(90.0)
    with pytest.raises(KeyError, match="nope"):
        m.nsfr_asf({"nope": 1.0})
    with pytest.raises(KeyError, match="nope"):
        m.nsfr_asf({"regulatory_capital": 1.0}, {"nope": 0.5})
    with pytest.raises(ValueError, match="regulatory_capital"):
        m.nsfr_asf({"regulatory_capital": -1.0})
    for key, factor in m.NSFR_ASF_FACTORS.items():
        assert m.nsfr_asf({key: 9.0}) == pytest.approx(9.0 * factor, abs=1e-12), key


def test_nsfr_rsf_goldens_and_addon():
    assert m.nsfr_rsf({"unencumbered_level_2b": 100.0}) == pytest.approx(50.0)
    obs = {"undrawn_facilities_obs": 100.0}
    assert m.nsfr_rsf({"unencumbered_level_2b": 100.0}, obs) == pytest.approx(55.0)
    r = m.nsfr_rsf({"unencumbered_level_2b": 100.0}, obs, derivative_liabilities=200.0,
                   derivative_liabilities_addon=0.10)
    assert r == pytest.approx(75.0, abs=1e-12)  # 55 + 20
    with pytest.raises(ValueError, match="derivative_liabilities_addon"):
        m.nsfr_rsf({"unencumbered_level_2b": 100.0}, obs, derivative_liabilities=200.0,
                   derivative_liabilities_addon=0.25)
    with pytest.raises(ValueError, match="derivative_liabilities_addon"):  # no default rate
        m.nsfr_rsf({"unencumbered_level_2b": 100.0}, obs, derivative_liabilities=200.0)
    with pytest.raises(ValueError, match="derivative_liabilities_addon"):
        m.nsfr_rsf({"unencumbered_level_2b": 100.0}, derivative_liabilities=1.0, derivative_liabilities_addon=0.04)
    # the interval is closed at both edges
    for rate in (0.05, 0.20):
        got = m.nsfr_rsf({}, derivative_liabilities=100.0, derivative_liabilities_addon=rate)
        assert got == pytest.approx(100.0 * rate, abs=1e-12)
    assert m.nsfr_rsf({}) == 0.0  # nothing to fund
    with pytest.raises(KeyError, match="nope"):
        m.nsfr_rsf({"nope": 1.0})
    with pytest.raises(KeyError, match="nope"):
        m.nsfr_rsf({}, {"nope": 1.0})
    with pytest.raises(KeyError, match="undrawn_facilities_obs"):  # an OBS key is not an asset key
        m.nsfr_rsf({"undrawn_facilities_obs": 1.0})
    with pytest.raises(ValueError, match="unencumbered_level_2b"):
        m.nsfr_rsf({"unencumbered_level_2b": -1.0})
    with pytest.raises(ValueError, match="derivative_liabilities"):
        m.nsfr_rsf({}, derivative_liabilities=-1.0, derivative_liabilities_addon=0.1)
    for key, factor in m.NSFR_RSF_FACTORS.items():
        if key == "undrawn_facilities_obs":
            assert m.nsfr_rsf({}, {key: 9.0}) == pytest.approx(9.0 * factor, abs=1e-12)
        else:
            assert m.nsfr_rsf({key: 9.0}) == pytest.approx(9.0 * factor, abs=1e-12), key


def test_nsfr_ratio_golden():
    assert m.nsfr_ratio(105.0, 50.0) == pytest.approx(2.1, abs=1e-12)
    assert m.nsfr_ratio(0.0, 50.0) == 0.0
    # monotone increasing in ASF and decreasing in RSF
    assert m.nsfr_ratio(110.0, 50.0) > m.nsfr_ratio(105.0, 50.0) > m.nsfr_ratio(105.0, 51.0)
    with pytest.raises(ValueError, match="^rsf: "):
        m.nsfr_ratio(10.0, 0.0)
    with pytest.raises(ValueError, match="^asf: "):
        m.nsfr_ratio(-1.0, 5.0)


# the transcription typed from the contract tables (dossier 01 G.1 and G.2)
OUTFLOW = {
    "retail_stable_insurance_criteria": 0.03, "retail_stable": 0.05, "retail_less_stable": 0.10,
    "retail_term_gt_30d": 0.00, "small_business_stable": 0.05, "small_business_less_stable": 0.10,
    "operational": 0.25, "operational_insured_portion": 0.05, "cooperative_network": 0.25,
    "non_financial_corporate": 0.40, "non_financial_corporate_fully_insured": 0.20, "other_legal_entity": 1.00,
    "secured_central_bank": 0.00, "secured_level1": 0.00, "secured_level2a": 0.15, "secured_sovereign_mdb_pse": 0.25,
    "secured_level2b_rmbs": 0.25, "secured_other_level2b": 0.50, "secured_other": 1.00,
    "non_level1_collateral_valuation": 0.20, "undrawn_retail_small_business": 0.05,
    "undrawn_credit_non_financial_corporate": 0.10, "undrawn_liquidity_non_financial_corporate": 0.30,
    "undrawn_credit_supervised_bank": 0.40, "undrawn_liquidity_supervised_bank": 0.40,
    "undrawn_credit_other_fi": 0.40, "undrawn_liquidity_other_fi": 1.00, "undrawn_other_legal_entity": 1.00,
    "net_derivative_outflow": 1.00, "other_contractual_outflow": 1.00,
}
INFLOW = {
    "secured_lending_level1": 0.00, "secured_lending_level2a": 0.15, "secured_lending_level2b_rmbs": 0.25,
    "secured_lending_other_level2b": 0.50, "secured_lending_other": 1.00, "margin_lending_other_collateral": 0.50,
    "facilities_granted_to_bank": 0.00, "operational_deposits_at_other_institutions": 0.00,
    "retail_and_nonfinancial_wholesale": 0.50, "financial_institutions_and_central_banks": 1.00,
    "net_derivative_inflow": 1.00,
}
HAIRCUT = {"level_1": 0.00, "level_2a": 0.15, "level_2b_rmbs": 0.25, "level_2b_corporate_debt": 0.50,
           "level_2b_common_equity": 0.50, "level_2b_sovereign_bbb": 0.50}
ASF = {
    "regulatory_capital": 1.00, "long_term_liabilities_ge_1y": 1.00, "stable_retail_deposits": 0.95,
    "less_stable_retail_deposits": 0.90, "funding_lt_1y_non_financial_corporates": 0.50, "operational_deposits": 0.50,
    "funding_lt_1y_sovereigns_pse_mdb_ndb": 0.50, "other_funding_6m_to_1y": 0.50, "other_liabilities": 0.00,
}
RSF = {
    "coins_banknotes_reserves": 0.00, "unencumbered_level_1": 0.05, "loans_fi_lt_6m_level1_secured": 0.10,
    "loans_fi_lt_6m_other": 0.15, "unencumbered_level_2a": 0.15, "unencumbered_level_2b": 0.50,
    "encumbered_hqla_6m_to_1y": 0.50, "loans_fi_6m_to_1y": 0.50, "other_assets_lt_1y": 0.50,
    "operational_deposits_held": 0.50,
    "residential_mortgages_ge_1y_rw35": 0.65, "other_loans_ge_1y_rw35": 0.65, "initial_margin_ccp_default_fund": 0.85,
    "other_loans_ge_1y_rw_gt_35": 0.85, "non_hqla_securities_ge_1y": 0.85, "physical_commodities": 0.85,
    "encumbered_ge_1y": 1.00, "net_derivative_assets": 1.00, "other_assets": 1.00, "undrawn_facilities_obs": 0.05,
}


def test_parameter_tables_transcription():
    assert dict(m.LCR_OUTFLOW_RATES) == OUTFLOW
    assert dict(m.LCR_INFLOW_RATES) == INFLOW
    assert dict(m.HQLA_HAIRCUTS) == HAIRCUT
    assert dict(m.NSFR_ASF_FACTORS) == ASF
    assert dict(m.NSFR_RSF_FACTORS) == RSF
    assert m.IN_FORCE == "2019-12-15"
    # every parameter carries its paragraph reference; no parameter without one, no reference without a parameter
    for table, refs, prefix in ((m.LCR_OUTFLOW_RATES, m.LCR_OUTFLOW_REFERENCES, "LCR"),
                                (m.LCR_INFLOW_RATES, m.LCR_INFLOW_REFERENCES, "LCR"),
                                (m.HQLA_HAIRCUTS, m.HQLA_REFERENCES, "LCR"),
                                (m.NSFR_ASF_FACTORS, m.NSFR_ASF_REFERENCES, "NSF"),
                                (m.NSFR_RSF_FACTORS, m.NSFR_RSF_REFERENCES, "NSF")):
        assert set(refs) == set(table)
        assert all(ref.startswith(prefix) or "NSF" in ref or "LCR" in ref for ref in refs.values())
    assert m.LCR_INFLOW_CAP == 0.75 and m.TRADE_FINANCE_MAX_RATE == 0.05
    assert (m.NSFR_DERIVATIVE_ADDON_MIN, m.NSFR_DERIVATIVE_ADDON_MAX) == (0.05, 0.20)
    # the factors are shares
    for table in (OUTFLOW, INFLOW, HAIRCUT, ASF, RSF):
        assert all(0.0 <= v <= 1.0 for v in table.values())


def test_net_cash_outflows_refuses_a_cap_above_the_basel_ceiling():
    """F1: inflows never offset more than 0.75 of outflows; a stricter cap is allowed."""
    for cap in (0.9, 1.0, 0.75 + 1e-9):
        with pytest.raises(ValueError, match="^cap: "):
            m.net_cash_outflows(100.0, 90.0, cap=cap)
    assert m.net_cash_outflows(100.0, 90.0, cap=0.75).net == 25.0
    assert m.net_cash_outflows(100.0, 90.0, cap=0.0).net == 100.0


def test_hqla_stock_empty_stock_shares_are_nan():
    """F11: a share of an empty stock has no value, never 0 and never an error."""
    r = m.hqla_stock(0.0, 0.0, 0.0)
    assert r.hqla == 0.0 and math.isnan(r.l2_share) and math.isnan(r.l2b_share)
    full = m.hqla_stock(100.0, 20.0, 30.0)
    assert not math.isnan(full.l2_share) and not math.isnan(full.l2b_share)


def test_nsfr_rsf_operational_deposits_held_key():
    """C2: dossier 01 l.634 lists "operational deposits held" at a 50 percent RSF factor."""
    assert m.NSFR_RSF_FACTORS["operational_deposits_held"] == 0.50
    assert m.nsfr_rsf({"operational_deposits_held": 10.0}) == 5.0
    assert "operational deposits held" in m.NSFR_RSF_REFERENCES["operational_deposits_held"]


def test_module_tables_are_read_only():
    """F6: the module tables are mappings that cannot be written."""
    for table in (m.LCR_OUTFLOW_RATES, m.LCR_INFLOW_RATES, m.HQLA_HAIRCUTS, m.NSFR_ASF_FACTORS, m.NSFR_RSF_FACTORS,
                  m.LCR_OUTFLOW_REFERENCES, m.LCR_INFLOW_REFERENCES, m.HQLA_REFERENCES, m.NSFR_ASF_REFERENCES,
                  m.NSFR_RSF_REFERENCES):
        assert isinstance(table, MappingProxyType)
        assert not hasattr(table, "__setitem__")
