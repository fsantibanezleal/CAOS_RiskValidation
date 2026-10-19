"""RV-1106 to RV-1108: the counterbalancing capacity (contract `engine-liquidity-api`, module cbc; dossier 05 E.1
line 238, LiST note paras 73 to 75)."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np
import pytest

from riskvalidation.liquidity import cbc
from riskvalidation.regulatory import lcr_nsfr


def test_haircut_rule_goldens():
    """The nine cases of dossier 05 line 238."""
    h = cbc.haircut_rule
    assert h("equity", equity_rating="AAA-A-") == (0.20, "ecb-equity-table")
    assert h("equity", equity_rating="BBB+-BBB-") == (0.35, "ecb-equity-table")
    assert h("equity", equity_rating="lower-or-unrated") == (0.50, "ecb-equity-table")
    assert h("non-equity", hqla_level="level_1") == (0.30, "ecb-non-equity-hqla")
    assert h("non-equity") == (0.50, "fifty-percent-fallback")
    assert h("non-equity", central_bank_eligible=True, central_bank_haircut=0.04) == (0.04, "central-bank-framework")
    assert h("non-equity", central_bank_eligible=True, hqla_level="level_2a") == (0.15, "lcr-haircut-hqla")
    assert h("non-equity", central_bank_eligible=True) == (0.50, "fifty-percent-fallback")
    assert h("cash") == (0.0, "nominal-value")
    assert set(cbc.RULE_LABELS) == {"nominal-value", "central-bank-framework", "lcr-haircut-hqla",
                                    "fifty-percent-fallback", "ecb-equity-table", "ecb-non-equity-hqla"}


def test_haircut_rule_order_and_refusals():
    h = cbc.haircut_rule
    # cash is at nominal value whatever else is said
    assert h("cash", central_bank_eligible=True, central_bank_haircut=0.2) == (0.0, "nominal-value")
    # the central bank's haircut wins over the LCR haircut when it is given, and a zero haircut is a given haircut
    assert h("non-equity", central_bank_eligible=True, central_bank_haircut=0.0, hqla_level="level_2a") == (
        0.0, "central-bank-framework")
    # the LCR haircut is read from the one table of lcr_nsfr, for every level
    for level, value in lcr_nsfr.HQLA_HAIRCUTS.items():
        assert h("non-equity", central_bank_eligible=True, hqla_level=level) == (value, "lcr-haircut-hqla")
    # a non-eligible equity ignores the HQLA level; the central bank haircut is ignored when not eligible
    assert h("equity", equity_rating="BBB+-BBB-", hqla_level="level_1", central_bank_haircut=0.01) == (
        0.35, "ecb-equity-table")
    # the equity table is the module data the validation test compares against
    assert cbc.ECB_EQUITY_HAIRCUTS == {"AAA-A-": 0.20, "BBB+-BBB-": 0.35, "lower-or-unrated": 0.50}
    with pytest.raises(ValueError, match="^kind: "):
        h("bond")
    with pytest.raises(ValueError, match="^equity_rating: "):
        h("equity")
    with pytest.raises(ValueError, match="^equity_rating: "):
        h("equity", equity_rating="AA")
    with pytest.raises(ValueError, match="^hqla_level: "):
        h("non-equity", hqla_level="level_9")
    with pytest.raises(ValueError, match="^central_bank_haircut: "):
        h("non-equity", central_bank_eligible=True, central_bank_haircut=1.5)


def test_liquidity_value_golden_and_invariants():
    f = cbc.liquidity_value
    assert f(100.0, 0.2) == pytest.approx(80.0, abs=1e-12)
    assert f(100.0, 0.1, encumbered=20.0) == pytest.approx(72.0, abs=1e-12)
    assert isinstance(f(100.0, 0.2), float)
    # the second route: market value less a haircut on the unencumbered amount
    assert f(100.0, 0.1, encumbered=20.0) == pytest.approx(100.0 - 20.0 - 0.1 * (100.0 - 20.0), abs=1e-12)
    # monotone decreasing in the haircut
    hc = np.linspace(0.0, 1.0, 21)
    assert np.all(np.diff(f(100.0, hc)) <= 0.0)
    assert f(100.0, 1.0) == 0.0 and f(100.0, 0.0, encumbered=100.0) == 0.0
    assert f(np.array([100.0, 50.0]), np.array([0.2, 0.5])).tolist() == pytest.approx([80.0, 25.0])
    with pytest.raises(ValueError, match="^encumbered: "):
        f(100.0, 0.1, encumbered=120.0)
    with pytest.raises(ValueError, match="^haircut: "):
        f(100.0, 1.1)
    with pytest.raises(ValueError, match="^market_value: "):
        f(-1.0, 0.1)


def test_monetisation_capacity_golden_and_invariants():
    m = cbc.monetisation_capacity(100.0, [(1, 0.5), (2, 0.8), (3, 1.0)], 4)
    assert m.tolist() == pytest.approx([50.0, 80.0, 100.0, 100.0], abs=1e-12)
    # zero before the first point; the share of day d is that of the last point with day <= d
    m = cbc.monetisation_capacity(100.0, [(3, 0.4), (5, 1.0)], 6)
    assert m.tolist() == pytest.approx([0.0, 0.0, 40.0, 40.0, 100.0, 100.0], abs=1e-12)
    # points beyond the horizon are legitimate and ignored
    assert cbc.monetisation_capacity(10.0, [(1, 0.5), (9, 1.0)], 3).tolist() == [5.0, 5.0, 5.0]
    with pytest.raises(ValueError, match="^points: "):  # shares must not decrease
        cbc.monetisation_capacity(100.0, [(1, 0.8), (2, 0.5)], 3)
    with pytest.raises(ValueError, match="^points: "):  # at most 1
        cbc.monetisation_capacity(100.0, [(1, 1.2)], 3)
    with pytest.raises(ValueError, match="^points: "):  # days strictly increasing and from 1
        cbc.monetisation_capacity(100.0, [(2, 0.5), (2, 0.6)], 3)
    with pytest.raises(ValueError, match="^points: "):
        cbc.monetisation_capacity(100.0, [(0, 0.5)], 3)
    with pytest.raises(ValueError, match="^horizon: "):
        cbc.monetisation_capacity(100.0, [(1, 0.5)], 0)
    with pytest.raises(ValueError, match="^value: "):
        cbc.monetisation_capacity(-1.0, [(1, 0.5)], 3)


def test_cbc_stack_golden_and_invariants():
    classes = {"A": (100.0, 20.0, 0.1), "B": (50.0, 0.0, 0.5)}
    stack = cbc.cbc_stack(classes, [(1, 1.0)], 2)
    assert list(stack) == ["A", "B"]
    assert stack["A"].tolist() == pytest.approx([72.0, 72.0], abs=1e-12)
    assert stack["B"].tolist() == pytest.approx([25.0, 25.0], abs=1e-12)
    # each class is its liquidity value times the monetisation share; the sum over classes is CBC(t)
    pts = [(1, 0.5), (2, 1.0)]
    stack = cbc.cbc_stack(classes, pts, 3)
    share = cbc.monetisation_capacity(1.0, pts, 3)
    for key, (mv, enc, hc) in classes.items():
        assert stack[key] == pytest.approx(cbc.liquidity_value(mv, hc, enc) * share, abs=1e-12)
    total = sum(stack.values())
    assert total.tolist() == pytest.approx([48.5, 97.0, 97.0], abs=1e-12)
    with pytest.raises(ValueError, match="^classes: "):
        cbc.cbc_stack({}, pts, 3)
    with pytest.raises(ValueError, match="encumbered"):
        cbc.cbc_stack({"A": (10.0, 20.0, 0.1)}, pts, 3)


def test_cbc_with_htm_loss_golden_and_invariants():
    r = cbc.cbc_with_htm_loss(100.0, 85.0, 20.0, 0.1)
    assert isinstance(r, cbc.HtmResult)
    assert r.unrealised_loss == pytest.approx(-15.0, abs=1e-12)
    assert r.cbc_without_loss == pytest.approx(72.0, abs=1e-12)
    assert r.cbc_with_loss == pytest.approx(58.5, abs=1e-12)
    assert r.cbc_removed == pytest.approx(13.5, abs=1e-12)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.cbc_removed = 0.0  # type: ignore[misc]
    # the second route: the removed CBC is the loss net of the haircut when the encumbrance is the same on both bases
    assert r.cbc_removed == pytest.approx(-r.unrealised_loss * (1 - 0.1), abs=1e-12)
    # non-negative whenever fair value is at most carrying value; no loss removes nothing
    for fair in (100.0, 99.0, 60.0, 20.0):
        assert cbc.cbc_with_htm_loss(100.0, fair, 20.0, 0.3).cbc_removed >= 0.0
    assert cbc.cbc_with_htm_loss(100.0, 100.0, 0.0, 0.2).cbc_removed == 0.0
    # a gain raises the CBC: the removed amount is negative, honestly
    assert cbc.cbc_with_htm_loss(100.0, 110.0, 0.0, 0.2).cbc_removed == pytest.approx(-8.0, abs=1e-12)
    # an encumbered amount above the fair value is refused: the same amount is encumbered on both bases
    with pytest.raises(ValueError, match="encumbered"):
        cbc.cbc_with_htm_loss(100.0, 15.0, 20.0, 0.1)


def test_equity_haircut_table_is_read_only():
    """F6."""
    assert isinstance(cbc.ECB_EQUITY_HAIRCUTS, MappingProxyType)
    assert not hasattr(cbc.ECB_EQUITY_HAIRCUTS, "__setitem__")
