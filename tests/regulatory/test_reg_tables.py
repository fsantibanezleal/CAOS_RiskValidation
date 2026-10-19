"""RV-916: the parameter tables of the engine (CRE52.72 Table 2, MAR50.16 Table 1, MAR33.4 Table 1 and MAR33.12 Table
2) equal, cell by cell, the parameter tables of the regulatory-calculators fixture, each entry carrying its paragraph,
table and the
in-force date of the chapter version."""

from __future__ import annotations

import pytest
from reg_helpers import load_calculators

from riskvalidation.regulatory import ba_cva, frtb, sa_ccr
from riskvalidation.regulatory.standardised import RegulatoryParameter

DOC = load_calculators()
TABLES = {t["id"]: t for t in DOC["parameter_tables"]}


def _check(data: dict, table: dict) -> None:
    assert len(data) == len(table["rows"])
    for row in table["rows"]:
        entry = data[(row["key"], row["parameter"])]
        assert isinstance(entry, RegulatoryParameter)
        assert entry.value == row["value"], row
        assert entry.paragraph == table["paragraph"] and entry.table == table["table"]
        assert entry.in_force == table["in_force_date"]


def test_sa_ccr_parameters_equal_the_fixture():
    _check(sa_ccr.SUPERVISORY_PARAMETERS, TABLES["cre52-72-table-2"])
    assert sa_ccr.supervisory_parameter("interest-rate", "supervisory_factor").value == 0.005
    assert sa_ccr.supervisory_parameter("fx", "supervisory_factor").paragraph == "CRE52.72"
    with pytest.raises(ValueError, match=r"CRE52\.72"):
        sa_ccr.supervisory_parameter("credit-single-name-D", "supervisory_factor")
    with pytest.raises(ValueError, match="parameter"):
        sa_ccr.supervisory_parameter("fx", "correlation")


def test_ba_cva_sector_weights_equal_the_fixture():
    _check(ba_cva.SECTOR_RISK_WEIGHTS, TABLES["mar50-16-table-1"])
    assert len({k for k, _ in ba_cva.SECTOR_RISK_WEIGHTS}) == 8
    # the lookup function returns the table cell
    for row in TABLES["mar50-16-table-1"]["rows"]:
        grade = row["parameter"] == "risk_weight_investment_grade"
        assert ba_cva.sector_risk_weight(row["key"], grade).value == row["value"]


def test_liquidity_horizon_tables_equal_the_fixture():
    _check(frtb.LIQUIDITY_HORIZONS_TABLE_2, TABLES["mar33-12-table-2"])
    _check(frtb.LIQUIDITY_HORIZONS_TABLE_1, TABLES["mar33-4-table-1"])
    assert frtb.LIQUIDITY_HORIZONS == (10, 20, 40, 60, 120)
    assert [frtb.LIQUIDITY_HORIZONS_TABLE_1[(f"j{j}", "liquidity_horizon_days")].value for j in range(1, 6)] == [
        10,
        20,
        40,
        60,
        120,
    ]
