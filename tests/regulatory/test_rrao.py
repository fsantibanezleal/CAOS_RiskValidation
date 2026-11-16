"""RV-2010: the residual risk add-on (MAR23.1 to MAR23.8). The Basel text holds no numeric example; the fixtures
RRAO-A to RRAO-C are computed here from the weights of MAR23.8 and by an independent loop."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from riskvalidation.regulatory import rrao as rr

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "regulatory" / "u9" / "08_rrao.md"


def _classify_ref(exotic, other, back_to_back, listed):
    """MAR23.7 by cases, written apart from the engine."""
    if back_to_back:
        return "excluded"
    if exotic:
        return "exotic"  # listed or cleared with an exotic underlying: included
    if other and not listed:
        return "other"
    return "excluded"


def test_rrao_formula_and_exclusions():
    # the weights, naming the paragraph
    assert (rr.RW_EXOTIC.value, rr.RW_EXOTIC.paragraph) == (0.01, "MAR23.8")
    assert (rr.RW_OTHER.value, rr.RW_OTHER.paragraph) == (0.001, "MAR23.8")
    assert rr.RW_EXOTIC.in_force == rr.RW_OTHER.in_force == "2023-01-01"
    # RRAO-A: 1.0% of 200 plus 0.1% of 1,000 is 2.0 + 1.0 = 3.0
    res = rr.rrao([200.0], [1000.0])
    assert res.exotic_charge == pytest.approx(0.01 * 200.0, rel=1e-15) == pytest.approx(2.0)
    assert res.other_charge == pytest.approx(0.001 * 1000.0, rel=1e-15) == pytest.approx(1.0)
    assert res.charge == pytest.approx(3.0, rel=1e-15)
    assert (res.exotic_notional, res.other_notional) == (200.0, 1000.0)
    assert "MAR23.8" in res.paragraphs["charge"]
    # RRAO-C: a simple sum of gross notionals, no offsetting of a long and a short of the same size
    res = rr.rrao([100.0, 100.0], [50.0, 50.0, 400.0])
    assert res.exotic_notional == 200.0 and res.other_notional == 500.0
    assert res.charge == pytest.approx(0.01 * 200.0 + 0.001 * 500.0, rel=1e-15) == pytest.approx(2.5)
    # one side or none
    assert rr.rrao([], []).charge == 0.0
    assert rr.rrao([1e6], []).charge == pytest.approx(1e4) and rr.rrao([], [1e6]).charge == pytest.approx(1e3)
    # a net notional is not a gross one: a negative notional is refused, naming the field
    with pytest.raises(ValueError, match="exotic_gross_notionals"):
        rr.rrao([-100.0], [])
    with pytest.raises(ValueError, match="other_gross_notionals"):
        rr.rrao([], [float("nan")])
    with pytest.raises(ValueError, match="exotic_gross_notionals"):
        rr.rrao([float("inf")], [])

    # RRAO-B and MAR23.7: a back-to-back pair is excluded; a listed or centrally cleared instrument is excluded from the
    # other residual risks and included when its underlying is exotic
    c = rr.classify_instrument
    assert (
        c(exotic_underlying=True, other_residual_risk=False, back_to_back=True, listed_or_cleared=False) == "excluded"
    )
    assert (
        c(exotic_underlying=False, other_residual_risk=True, back_to_back=True, listed_or_cleared=False) == "excluded"
    )
    assert (
        c(exotic_underlying=False, other_residual_risk=True, back_to_back=False, listed_or_cleared=True) == "excluded"
    )
    assert c(exotic_underlying=True, other_residual_risk=False, back_to_back=False, listed_or_cleared=True) == "exotic"
    assert c(exotic_underlying=False, other_residual_risk=True, back_to_back=False, listed_or_cleared=False) == "other"
    assert c(exotic_underlying=True, other_residual_risk=False, back_to_back=False, listed_or_cleared=False) == "exotic"
    # an instrument with neither risk is outside the add-on
    assert (
        c(exotic_underlying=False, other_residual_risk=False, back_to_back=False, listed_or_cleared=False) == "excluded"
    )
    # the full truth table against the cases written apart
    for exotic, other, b2b, listed in itertools.product((False, True), repeat=4):
        assert c(
            exotic_underlying=exotic, other_residual_risk=other, back_to_back=b2b, listed_or_cleared=listed
        ) == _classify_ref(exotic, other, b2b, listed)
    with pytest.raises(ValueError, match="back_to_back"):
        c(exotic_underlying=False, other_residual_risk=True, back_to_back=1, listed_or_cleared=False)
    with pytest.raises(ValueError, match="listed_or_cleared"):
        c(exotic_underlying=False, other_residual_risk=True, back_to_back=False, listed_or_cleared=None)

    # the stated limits, in the docstring and on the page
    doc = (rr.classify_instrument.__doc__ or "").lower()
    assert "mar23.3" in doc and "mar23.6" in doc and "the caller's" in doc
    mod = (rr.__doc__ or "").lower()
    for phrase in ("simple sum", "gross notional", "no compliance claim", "mar23.7", "mar23.8", "the caller's"):
        assert phrase in mod, phrase
    page = PAGE.read_text(encoding="utf-8").lower()
    for phrase in (
        "https://www.bis.org/basel_framework/chapter/mar/23.htm?inforce=20230101&published=20200327",
        "mar23.8",
        "mar23.7",
        "back-to-back",
        "no offsetting",
        "the caller's",
    ):
        assert phrase in page, phrase
    assert "—" not in page and "—" not in rr.__doc__
    assert (ROOT / "docs" / "regulatory" / "u9" / "README.md").read_text(encoding="utf-8").count("08_rrao.md") == 1
