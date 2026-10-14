"""RV-1219 to RV-1221: the Basel III standardised approach of OPE25 (dossier 01 section F).

References: the BIC marginal coefficients of OPE25.7 Table 1 and its footnote 2 (BI of EUR 35bn gives BIC of EUR
5.37bn), the ILM formula of OPE25.8, ILM = 1 when LC = BIC (OPE25.9), and the BIC-only rule under 5 years of loss data
(OPE25.10, OPE25.15).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from riskvalidation.oprisk import sma

E = math.e


def arr(*x):
    return np.array(x, dtype=float)


def test_ildc_goldens_and_year_by_year_absolute_values():
    ii, ie, di = arr(10, 10, 10), arr(2, 2, 2), arr(1, 1, 1)
    assert sma.ildc(ii, ie, arr(100, 100, 100), di) == pytest.approx(3.25, abs=1e-12)
    assert sma.ildc(ii, ie, arr(1000, 1000, 1000), di) == pytest.approx(9.0, abs=1e-12)
    # |ii - ie| is taken year by year before averaging (OPE25.5 footnote 1)
    year_by_year = sma.ildc(arr(10, 2, 10), arr(2, 10, 2), arr(1000, 1000, 1000), arr(0, 0, 0))
    assert year_by_year == pytest.approx(8.0, abs=1e-12)
    netted = abs(arr(10, 2, 10).mean() - arr(2, 10, 2).mean())
    assert netted == pytest.approx(2.666667, abs=5e-7) and netted != year_by_year
    with pytest.raises(ValueError):
        sma.ildc(arr(1, 2), arr(1, 2), arr(1, 2), arr(1, 2))


def test_service_and_financial_components():
    assert sma.service_component(arr(5, 5, 5), arr(3, 3, 3), arr(2, 2, 2), arr(4, 4, 4)) == pytest.approx(9.0)
    # the maximum is taken of the three-year averages, not year by year
    assert sma.service_component(arr(6, 0, 0), arr(0, 3, 0), arr(0, 0, 0), arr(0, 0, 0)) == pytest.approx(2.0)
    assert sma.financial_component(arr(10, -10, 10), arr(-3, 3, -3)) == pytest.approx(13.0, abs=1e-12)
    netted = abs(arr(10, -10, 10).mean()) + abs(arr(-3, 3, -3).mean())
    assert netted == pytest.approx(4.333333, abs=5e-7)
    with pytest.raises(ValueError):
        sma.financial_component(arr(1, 2), arr(1, 2))


def test_business_indicator_golden_and_refusals():
    assert sma.business_indicator(3.25, 9.0, 13.0) == pytest.approx(25.25, abs=1e-12)
    for bad in ((-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -0.01)):
        with pytest.raises(ValueError):
            sma.business_indicator(*bad)


def test_bic_bucket_boundaries():
    assert sma.bic_bucket(1e9) == 1 and sma.bic_bucket(0.0) == 1
    assert sma.bic_bucket(1.0000001e9) == 2 and sma.bic_bucket(30e9) == 2
    assert sma.bic_bucket(35e9) == 3
    with pytest.raises(ValueError):
        sma.bic_bucket(-1.0)


def test_bic_goldens_continuity_and_marginal_coefficients():
    assert sma.bic(35e9) == pytest.approx(5.37e9, rel=1e-13)               # OPE25.7 footnote 2
    assert sma.bic(35e9) == pytest.approx(1e9 * 0.12 + 29e9 * 0.15 + 5e9 * 0.18, rel=1e-13)
    assert sma.bic(0.5e9) == pytest.approx(0.06e9, rel=1e-13)
    assert sma.bic(1e9) == pytest.approx(0.12e9, rel=1e-13)
    assert sma.bic(30e9) == pytest.approx(4.47e9, rel=1e-13)
    assert sma.bic(0.0) == 0.0
    x = np.linspace(0.0, 60e9, 4001)
    y = np.array([sma.bic(v) for v in x])
    assert np.all(np.diff(y) > 0.0)
    for lo, hi, slope in ((0.1e9, 0.9e9, 0.12), (2e9, 29e9, 0.15), (31e9, 60e9, 0.18)):   # the marginal coefficients
        assert (sma.bic(hi) - sma.bic(lo)) / (hi - lo) == pytest.approx(slope, rel=1e-12)
    eps = 1.0
    for knot in (1e9, 30e9):                                               # continuous at the knots
        assert abs(sma.bic(knot + eps) - sma.bic(knot - eps)) < 0.2 * 2 * eps
    with pytest.raises(ValueError):
        sma.bic(-1.0)


def test_loss_component_goldens_and_window():
    assert sma.loss_component(np.full(10, 10.0)) == pytest.approx(150.0)
    assert sma.loss_component(arr(20, 10, 30, 40, 0)) == pytest.approx(300.0)
    # only the most recent window_years count (the array is oldest first)
    series = np.concatenate([np.full(5, 1000.0), np.full(10, 10.0)])
    assert sma.loss_component(series) == pytest.approx(150.0)
    assert sma.loss_component(series, window_years=15) == pytest.approx(15 * (5000 + 100) / 15)
    for bad in (arr(), arr(1.0, -1.0), arr(1.0, np.nan)):
        with pytest.raises(ValueError):
            sma.loss_component(bad)
    with pytest.raises(ValueError):
        sma.loss_component(arr(1.0), window_years=0)


def test_ilm_goldens_and_properties():
    assert sma.ilm(150.0, 150.0) == 1.0                                    # OPE25.9, exactly
    assert sma.ilm(300.0, 150.0) == pytest.approx(1.241090, abs=5e-7)
    assert sma.ilm(75.0, 150.0) == pytest.approx(0.829700, abs=5e-7)
    assert sma.ilm(0.0, 150.0) == pytest.approx(0.541325, abs=5e-7)
    assert sma.ilm(0.0, 150.0) == pytest.approx(math.log(E - 1.0), rel=1e-14)
    assert sma.ilm(285.0, 150.0) == pytest.approx(1.220648, abs=5e-7)
    r = np.linspace(0.0, 10.0, 200)
    v = np.array([sma.ilm(x * 100.0, 100.0) for x in r])
    assert np.all(np.diff(v) > 0.0)
    assert np.all(v[r > 1.0] > 1.0) and np.all(v[r < 1.0] < 1.0)
    for lc, b in ((1.0, 0.0), (1.0, -1.0), (-1.0, 1.0)):
        with pytest.raises(ValueError):
            sma.ilm(lc, b)


def test_sma_capital_golden_bucket_three():
    losses = np.full(10, 5.37e9 / 15)
    r = sma.sma_capital(35e9, losses)
    assert r.bic == pytest.approx(5.37e9, rel=1e-13) and r.bucket == 3
    assert r.loss_component == pytest.approx(5.37e9, rel=1e-12) and r.ilm == 1.0
    assert r.orc == pytest.approx(5.37e9, rel=1e-12) and r.rwa == pytest.approx(67.125e9, rel=1e-12)
    assert r.basis == "bic-times-ilm" and r.discretion_applied == "none" and r.reason is None and r.years_used == 10
    assert r.bi == 35e9
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.orc = 1.0  # type: ignore[misc]


def test_sma_capital_under_five_years_is_bic_only_and_never_an_ilm():
    r = sma.sma_capital(1.4e9, arr(1e7, 2e7, 3e7))
    assert r.basis == "bic-only" and r.bic == pytest.approx(0.18e9, rel=1e-13)
    assert r.orc == pytest.approx(0.18e9, rel=1e-13)
    assert r.loss_component is None and r.ilm is None and isinstance(r.reason, str) and r.reason
    assert r.discretion_applied == "transitional-bic-only" and r.years_used == 3 and r.bucket == 2
    assert r.rwa == pytest.approx(12.5 * 0.18e9, rel=1e-13)
    # four years is under the minimum, five is not
    assert sma.sma_capital(35e9, np.full(4, 1e9)).basis == "bic-only"
    assert sma.sma_capital(35e9, np.full(5, 1e9)).basis == "bic-times-ilm"
    # a stricter minimum is the caller's policy
    assert sma.sma_capital(35e9, np.full(8, 1e9), minimum_years=8).basis == "bic-times-ilm"
    assert sma.sma_capital(35e9, np.full(7, 1e9), minimum_years=8).basis == "bic-only"


def test_sma_capital_refuses_a_minimum_below_the_transitional_five_years():
    """An ILM on fewer than 5 years is unreachable: OPE25.15 is the transitional minimum, not a caller's choice."""
    for minimum in (4, 3, 1):
        with pytest.raises(ValueError, match="OPE25.15"):
            sma.sma_capital(35e9, np.full(4, 1e9), minimum_years=minimum)
    with pytest.raises(ValueError):
        sma.sma_capital(35e9, np.full(10, 1e9), minimum_years=0)
    with pytest.raises(ValueError):
        sma.sma_capital(35e9, np.full(10, 1e9), minimum_years=5.5)
    assert sma.sma_capital(35e9, np.full(5, 1e9), minimum_years=5).basis == "bic-times-ilm"


def test_sma_capital_discretions_and_rule_order():
    losses = np.full(10, 1e9)
    b1 = sma.sma_capital(0.5e9, losses)                                    # bucket 1: ILM = 1 (OPE25.11)
    assert b1.bucket == 1 and b1.ilm == 1.0 and b1.discretion_applied == "bucket-1-ilm-one"
    assert b1.orc == pytest.approx(0.06e9, rel=1e-13) and b1.loss_component == pytest.approx(15e9)
    used = sma.sma_capital(0.5e9, losses, bucket1_uses_losses=True)
    assert used.discretion_applied == "none" and used.ilm == pytest.approx(sma.ilm(15e9, 0.06e9), rel=1e-13)
    assert used.orc == pytest.approx(0.06e9 * used.ilm, rel=1e-13) and used.ilm > 1.0
    nat = sma.sma_capital(35e9, losses, national_ilm_one=True)             # a supervisor sets ILM = 1
    assert nat.ilm == 1.0 and nat.discretion_applied == "national-ilm-one" and nat.orc == pytest.approx(nat.bic)
    assert nat.loss_component == pytest.approx(15e9)
    both = sma.sma_capital(0.5e9, losses, national_ilm_one=True)           # rule (2) before rule (3)
    assert both.discretion_applied == "national-ilm-one"
    short = sma.sma_capital(0.5e9, arr(1e6), national_ilm_one=True)        # rule (1) before everything
    assert short.discretion_applied == "transitional-bic-only" and short.ilm is None
    with pytest.raises(ValueError):
        sma.sma_capital(-1.0, losses)
    # the invariants of the record
    for r in (b1, used, nat, both, short, sma.sma_capital(35e9, losses)):
        assert r.rwa == pytest.approx(12.5 * r.orc, rel=1e-13)
        assert (r.basis == "bic-only") == (r.loss_component is None and r.ilm is None and r.reason is not None
                                           and r.discretion_applied == "transitional-bic-only")
        if r.ilm == 1.0 or r.basis == "bic-only":
            assert r.orc == r.bic


def test_sma_capital_window_uses_only_the_recent_years():
    old_big = np.concatenate([np.full(5, 1e12), np.full(10, 1e8)])
    r = sma.sma_capital(35e9, old_big)
    assert r.years_used == 10 and r.loss_component == pytest.approx(15e8, rel=1e-12)
