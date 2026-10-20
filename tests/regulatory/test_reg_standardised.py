"""RV-901 to RV-907: the CRE20 standardised risk weights as data with their paragraph, the lookup (contract
`engine-regulatory-api`, module standardised). The goldens are the tables of the contract and of dossier 01 section C;
the lookup
is compared with every cell of the `cre20_grid` of the regulatory-calculators fixture."""

from __future__ import annotations

import dataclasses
import re

import pytest
from reg_helpers import load_calculators

from riskvalidation.regulatory import standardised as st

DOC = load_calculators()
GRID = DOC["cre20_grid"]
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SOV = ("AAA to AA-", "A+ to A-", "BBB+ to BBB-", "BB+ to B-", "below B-", "unrated")
CORP = ("AAA to AA-", "A+ to A-", "BBB+ to BBB-", "BB+ to BB-", "below BB-", "unrated")
ECRA = SOV[:5]


def test_regulatory_parameter_is_frozen_and_aliased():
    p = st.RegulatoryParameter(0.5, "CRE20.7", "CRE20 Table 1", "2023-01-01")
    assert (p.value, p.paragraph, p.table, p.in_force) == (0.5, "CRE20.7", "CRE20 Table 1", "2023-01-01")
    assert st.RiskWeight is st.RegulatoryParameter
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.value = 1.0  # type: ignore[misc]
    assert dataclasses.is_dataclass(p)


def test_every_lookup_carries_paragraph_table_and_in_force():
    from riskvalidation.regulatory import output_floor as of

    returned = [
        st.sovereign_risk_weight("unrated"),
        st.eca_risk_weight(3),
        st.bank_risk_weight_ecra("A+ to A-", True),
        st.bank_risk_weight_scra("B"),
        st.corporate_risk_weight("unrated"),
        st.retail_risk_weight("other"),
        st.equity_risk_weight("equity"),
        st.residential_re_risk_weight(0.7),
        st.commercial_re_risk_weight(0.5, True),
        st.apply_currency_mismatch(0.75),
        st.credit_conversion_factor("commitment"),
        st.defaulted_risk_weight(0.1),
        st.lookup("ccf", "repo"),
        of.phase_in_factor(2025),
    ]
    for p in returned:
        assert isinstance(p, st.RegulatoryParameter)
        assert isinstance(p.value, float)
        assert p.paragraph and p.table and ISO.match(p.in_force), p
    # every cell of the fixture grid also carries all three
    for cell in GRID:
        assert cell["paragraph"] and cell["table"] and ISO.match(cell["in_force_date"])


def test_sovereign_and_eca_weights():
    for rating, want in zip(SOV, (0.0, 0.2, 0.5, 1.0, 1.5, 1.0), strict=True):
        p = st.sovereign_risk_weight(rating)
        assert (
            p.value == want and p.paragraph == "CRE20.7" and p.table == "CRE20 Table 1" and p.in_force == "2023-01-01"
        )
    for score, want in zip(range(8), (0.0, 0.0, 0.2, 0.5, 1.0, 1.0, 1.0, 1.5), strict=True):
        p = st.eca_risk_weight(score)
        assert p.value == want and p.paragraph == "CRE20.9" and p.table == "CRE20 Table 2"


def test_sovereign_and_eca_refusals():
    with pytest.raises(ValueError, match=r"CRE20 Table 1.*CRE20\.7|CRE20\.7.*CRE20 Table 1"):
        st.sovereign_risk_weight("AAA")
    with pytest.raises(ValueError, match="rating"):
        st.sovereign_risk_weight("")
    for bad in (-1, 8, 100):
        with pytest.raises(ValueError, match=r"score.*CRE20\.9|CRE20\.9.*score"):
            st.eca_risk_weight(bad)
    for bad in (2.5, "3", None, True):
        with pytest.raises(ValueError, match="score"):
            st.eca_risk_weight(bad)  # type: ignore[arg-type]


def test_bank_ecra_and_scra_weights():
    for rating, base, short in zip(ECRA, (0.2, 0.3, 0.5, 1.0, 1.5), (0.2, 0.2, 0.2, 0.5, 1.5), strict=True):
        assert st.bank_risk_weight_ecra(rating).value == base
        p = st.bank_risk_weight_ecra(rating, short_term=True)
        assert p.value == short and p.paragraph == "CRE20.18" and p.table == "CRE20 Table 6"
    for grade, base, short in (("A", 0.4, 0.2), ("B", 0.75, 0.5), ("C", 1.5, 1.5)):
        assert st.bank_risk_weight_scra(grade).value == base
        p = st.bank_risk_weight_scra(grade, short_term=True)
        assert p.value == short and p.paragraph == "CRE20.21" and p.table == "CRE20 Table 7"


def test_bank_refusals():
    with pytest.raises(ValueError, match="SCRA"):
        st.bank_risk_weight_ecra("unrated")
    with pytest.raises(ValueError, match="rating"):
        st.bank_risk_weight_ecra("BB+ to BB-")
    with pytest.raises(ValueError, match=r"grade.*CRE20\.21|CRE20\.21.*grade"):
        st.bank_risk_weight_scra("D")


def test_corporate_weights_and_flags():
    for rating, want in zip(CORP, (0.2, 0.5, 0.75, 1.0, 1.5, 1.0), strict=True):
        p = st.corporate_risk_weight(rating)
        assert p.value == want and p.paragraph == "CRE20.43" and p.table == "CRE20 Table 10"
    ig = st.corporate_risk_weight("unrated", investment_grade_unrated=True)
    assert (ig.value, ig.paragraph) == (0.65, "CRE20.44") and "CRE20.46" in ig.table
    sme = st.corporate_risk_weight("unrated", sme=True)
    assert (sme.value, sme.paragraph) == (0.85, "CRE20.47")
    retail_sme = st.corporate_risk_weight("unrated", regulatory_retail_sme=True)
    assert (retail_sme.value, retail_sme.paragraph) == (0.75, "CRE20.47")


def test_corporate_refusals():
    with pytest.raises(ValueError, match="rating"):
        st.corporate_risk_weight("BB+ to B-")
    for flag in ("sme", "investment_grade_unrated", "regulatory_retail_sme"):
        with pytest.raises(ValueError, match=flag):
            st.corporate_risk_weight("A+ to A-", **{flag: True})
    with pytest.raises(ValueError, match="sme"):
        st.corporate_risk_weight("unrated", sme=True, investment_grade_unrated=True)


def test_retail_and_equity_weights():
    for kind, want in (("regulatory", 0.75), ("transactor", 0.45), ("other", 1.0)):
        p = st.retail_risk_weight(kind)
        assert p.value == want and p.paragraph == "CRE20.68"
    for kind, want in (
        ("equity", 2.5),
        ("speculative-unlisted", 4.0),
        ("legislated-programme", 1.0),
        ("subordinated-debt", 1.5),
    ):
        p = st.equity_risk_weight(kind)
        assert p.value == want and p.table == "CRE20.57 to CRE20.60"
    with pytest.raises(ValueError, match="kind"):
        st.retail_risk_weight("mortgage")
    with pytest.raises(ValueError, match="kind"):
        st.equity_risk_weight("preferred")


T11 = (0.2, 0.25, 0.3, 0.4, 0.5, 0.7)
T12 = (0.3, 0.35, 0.45, 0.6, 0.75, 1.05)
EDGES = (0.5, 0.6, 0.8, 0.9, 1.0)


def test_residential_re_bands_and_edges():
    # inside every band, at the upper edge (which belongs to the band) and just above it
    probes = [
        (0.0, 0),
        (0.3, 0),
        (0.5, 0),
        (0.5000001, 1),
        (0.55, 1),
        (0.6, 1),
        (0.61, 2),
        (0.8, 2),
        (0.85, 3),
        (0.9, 3),
        (0.95, 4),
        (1.0, 4),
        (1.0000001, 5),
        (2.5, 5),
    ]
    for ltv, band in probes:
        p = st.residential_re_risk_weight(ltv)
        assert p.value == T11[band] and p.paragraph == "CRE20.82" and p.table == "CRE20 Table 11", (ltv, band)
        q = st.residential_re_risk_weight(ltv, cash_flow_dependent=True)
        assert q.value == T12[band] and q.paragraph == "CRE20.84" and q.table == "CRE20 Table 12", (ltv, band)
    for ltv, want in zip((*EDGES, 1.5), T11, strict=True):
        assert st.residential_re_risk_weight(ltv).value == want
    split = st.residential_re_risk_weight(0.8, loan_splitting=True)
    assert split.value == 0.2 and split.paragraph == "CRE20.83"
    assert st.residential_re_risk_weight(0.3, loan_splitting=True).value == 0.2


def test_commercial_re_tables():
    # Table 13: up to 0.6 the weight is min(0.6, counterparty weight), above it the counterparty weight
    for cp in (0.2, 0.5, 0.6, 1.0, 1.5):
        low = st.commercial_re_risk_weight(0.6, False, cp)
        assert low.value == min(0.6, cp) and low.paragraph == "CRE20.85" and low.table == "CRE20 Table 13"
        assert st.commercial_re_risk_weight(0.3, False, counterparty_weight=cp).value == min(0.6, cp)
        assert st.commercial_re_risk_weight(0.61, False, cp).value == cp
        assert st.commercial_re_risk_weight(1.4, False, cp).value == cp
    # Table 14: cash-flow dependent, the counterparty weight is not used
    for ltv, want in ((0.0, 0.7), (0.6, 0.7), (0.61, 0.9), (0.8, 0.9), (0.81, 1.1), (3.0, 1.1)):
        p = st.commercial_re_risk_weight(ltv, True)
        assert p.value == want and p.paragraph == "CRE20.87" and p.table == "CRE20 Table 14"
        assert st.commercial_re_risk_weight(ltv, True, counterparty_weight=0.2).value == want


def test_real_estate_refusals():
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ltv"):
            st.residential_re_risk_weight(bad)
        with pytest.raises(ValueError, match="ltv"):
            st.commercial_re_risk_weight(bad, True)
    with pytest.raises(ValueError, match="loan_splitting"):
        st.residential_re_risk_weight(0.5, cash_flow_dependent=True, loan_splitting=True)
    with pytest.raises(ValueError, match="counterparty_weight"):
        st.commercial_re_risk_weight(0.5, False)
    with pytest.raises(ValueError, match="counterparty_weight"):
        st.commercial_re_risk_weight(0.5, False, -0.1)


def test_currency_mismatch():
    assert st.apply_currency_mismatch(0.75).value == pytest.approx(1.125, abs=1e-15)
    assert st.apply_currency_mismatch(1.2).value == 1.5  # 1.8 capped at 150 percent
    assert st.apply_currency_mismatch(0.0).value == 0.0
    p = st.apply_currency_mismatch(0.2)
    assert p.value == pytest.approx(0.3) and p.paragraph == "CRE20.92"
    # monotone and never above the cap
    values = [st.apply_currency_mismatch(w / 10).value for w in range(0, 25)]
    assert values == sorted(values) and max(values) == 1.5
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="weight"):
            st.apply_currency_mismatch(bad)


def test_credit_conversion_factors():
    kinds = (
        "direct-credit-substitute",
        "repo",
        "securities-lending",
        "forward-purchase",
        "nif-ruf",
        "transaction-related-contingent",
        "commitment",
        "trade-letter-of-credit",
        "unconditionally-cancellable",
    )
    for kind, want in zip(kinds, (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.4, 0.2, 0.1), strict=True):
        p = st.credit_conversion_factor(kind)
        assert p.value == want and p.paragraph == "CRE20.94" and p.table == "CRE20.94 to CRE20.101"
    with pytest.raises(ValueError, match="kind"):
        st.credit_conversion_factor("guarantee")


def test_defaulted_risk_weights():
    for share, want in ((0.0, 1.5), (0.1, 1.5), (0.19999, 1.5), (0.2, 1.0), (0.5, 1.0), (1.0, 1.0)):
        p = st.defaulted_risk_weight(share)
        assert p.value == want and p.paragraph == "CRE20.104", share
    res = st.defaulted_risk_weight(0.0, residential_non_cash_flow_dependent=True)
    assert res.value == 1.0 and res.paragraph == "CRE20.104"
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="specific_provision_share"):
            st.defaulted_risk_weight(bad)


def _call(cell: dict) -> st.RegulatoryParameter:
    return st.lookup(
        cell["exposure_class"],
        cell["key"],
        short_term=cell["short_term"],
        cash_flow_dependent=cell["cash_flow_dependent"],
        counterparty_weight=cell["counterparty_weight"],
    )


def test_lookup_equals_every_cell_of_the_cre20_grid():
    assert len(GRID) == 79
    seen = set()
    for cell in GRID:
        p = _call(cell)
        assert p == st.RegulatoryParameter(
            cell["risk_weight"], cell["paragraph"], cell["table"], cell["in_force_date"]
        ), cell
        seen.add(cell["exposure_class"])
    assert seen == {
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
    }
    # the dispatch returns exactly the function's own parameter
    assert st.lookup("sovereign", "A+ to A-") == st.sovereign_risk_weight("A+ to A-")
    assert st.lookup("eca", "7") == st.eca_risk_weight(7)
    assert st.lookup("bank-ecra", "BBB+ to BBB-", short_term=True) == st.bank_risk_weight_ecra("BBB+ to BBB-", True)
    assert st.lookup("residential-re", "0.8", cash_flow_dependent=True) == st.residential_re_risk_weight(0.8, True)
    assert st.lookup("commercial-re", "0.5", counterparty_weight=0.3) == st.commercial_re_risk_weight(0.5, False, 0.3)
    assert st.lookup("defaulted", "provision-below-20pct") == st.defaulted_risk_weight(0.0)
    assert st.lookup("currency-mismatch", "multiplier").value == 1.5
    assert st.lookup("currency-mismatch", "cap").value == 1.5


def test_lookup_refusals():
    with pytest.raises(ValueError, match="exposure_class"):
        st.lookup("project-finance", "AAA to AA-")
    with pytest.raises(ValueError, match="key"):
        st.lookup("sovereign", "AAAA")
    with pytest.raises(ValueError, match="key"):
        st.lookup("eca", "seven")
    with pytest.raises(ValueError, match="key"):
        st.lookup("residential-re", "high")
    with pytest.raises(ValueError, match="key"):
        st.lookup("currency-mismatch", "floor")
    with pytest.raises(ValueError, match="key"):
        st.lookup("defaulted", "provision")
    with pytest.raises(ValueError, match="short_term"):
        st.lookup("sovereign", "unrated", short_term=True)
    with pytest.raises(ValueError, match="cash_flow_dependent"):
        st.lookup("corporate", "unrated", cash_flow_dependent=True)
    with pytest.raises(ValueError, match="counterparty_weight"):
        st.lookup("residential-re", "0.5", counterparty_weight=0.5)
    with pytest.raises(ValueError, match="verbose"):
        st.lookup("sovereign", "unrated", verbose=True)
