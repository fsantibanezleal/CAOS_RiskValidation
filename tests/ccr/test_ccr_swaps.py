"""RV-957 to RV-960: the fixed-float swaps of `riskvalidation.ccr` (contract `engine-ccr-api`, module swaps). The
valuation between two resets is checked through a route of its own (a deterministic curve, flows discounted by hand) and
by the martingale property of the discounted swap value along simulated paths."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from riskvalidation.ccr import hw1f, swaps
from riskvalidation.ccr.swaps import Swap

TENORS = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
ZEROS = np.array([0.030, 0.032, 0.035, 0.040, 0.042, 0.043])


def curve():
    return hw1f.curve_functions(ZEROS, TENORS)


def flat(rate=0.03):
    def p0(t):
        return np.exp(-rate * np.asarray(t, dtype=float))

    def fwd(t):
        return np.full_like(np.asarray(t, dtype=float), rate)

    return p0, fwd


def arr(*x):
    return np.array(x, dtype=float)


def swap(rate=0.03, payer=False, start=0.0, maturity=2.0, freq=2, notional=1.0, trade_id="s"):
    return Swap(trade_id, notional, rate, payer, start, maturity, freq)


# ---------------------------------------------------------------------------------------------------- RV-957


def test_swap_and_cash_flow_times():
    s = swap()
    assert (s.trade_id, s.notional, s.fixed_rate, s.is_payer, s.start, s.maturity, s.fixed_frequency) == (
        "s", 1.0, 0.03, False, 0.0, 2.0, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.notional = 2.0                                              # type: ignore[misc]
    for kw, field in (({"maturity": 0.0}, "maturity"), ({"maturity": -1.0}, "maturity"), ({"start": -0.5}, "start"),
                      ({"notional": 0.0}, "notional"), ({"notional": -1.0}, "notional"),
                      ({"freq": 0}, "fixed_frequency")):
        with pytest.raises(ValueError, match=field):
            swap(**kw)
    with pytest.raises(ValueError, match="maturity"):
        swap(start=2.0, maturity=2.0)
    assert np.array_equal(swaps.cash_flow_times(s), arr(0.5, 1.0, 1.5, 2.0))
    assert np.allclose(swaps.cash_flow_times(swap(start=0.5, maturity=2.25, freq=2)), arr(1.0, 1.5, 2.0, 2.25))
    assert swaps.cash_flow_times(swap(start=0.0, maturity=7.0, freq=4))[-1] == 7.0
    assert np.allclose(swaps.cash_flow_times(swap(start=1.0, maturity=3.0, freq=1)), arr(2.0, 3.0))
    assert np.array_equal(swaps.reset_times(s), arr(0.0, 0.5, 1.0, 1.5))
    assert np.array_equal(swaps.reset_times(swap(start=1.0, maturity=3.0, freq=1)), arr(1.0, 2.0))


# ---------------------------------------------------------------------------------------------------- RV-958


def _fixed_leg(s, t, r, a, sigma, p0, fwd):
    times = swaps.cash_flow_times(s)
    prev = np.concatenate([[s.start], times[:-1]])
    tot = np.zeros_like(r)
    for tt, tp in zip(times, prev, strict=True):
        if tt > t + 1e-9:
            tot += s.notional * s.fixed_rate * (tt - tp) * hw1f.bond_price(r, t, float(tt), a, sigma, p0, fwd)
    return tot


def test_swap_value_par_and_between_resets():
    for p0, fwd in (flat(0.03), curve()):
        for a, sigma in ((0.1, 0.01), (0.5, 0.03), (0.02, 0.0)):
            par = swaps.par_fixed_rate(3.0, 2, p0)
            r0 = np.repeat(hw1f.phi(arr(0.0), a, sigma, fwd), 4)
            recv = swaps.swap_value(swap(par, False, 0.0, 3.0), 0.0, r0, a, sigma, p0, fwd)
            pay = swaps.swap_value(swap(par, True, 0.0, 3.0), 0.0, r0, a, sigma, p0, fwd)
            assert np.max(np.abs(recv)) < 1e-12 and np.max(np.abs(pay)) < 1e-12
            off = swaps.swap_value(swap(par + 0.01, False, 0.0, 3.0), 0.0, r0, a, sigma, p0, fwd)
            assert np.all(off > 0.0)                                  # a receiver 1 percent above par is in the money
    # a payer is minus the receiver of the same terms, for random short rates
    p0, fwd = curve()
    r = np.linspace(0.0, 0.08, 9)
    for t in (0.0, 0.5, 1.0, 1.5):
        lf = np.full(9, 0.03)
        a_ = swaps.swap_value(swap(0.04, True, 0.0, 3.0), t, r, 0.1, 0.01, p0, fwd, lf)
        b_ = swaps.swap_value(swap(0.04, False, 0.0, 3.0), t, r, 0.1, 0.01, p0, fwd, lf)
        assert np.allclose(a_, -b_, atol=1e-15, rtol=0)
    # on a reset date: floating leg = N (1 - P(t, T_n)); only the flows after t
    s = swap(0.04, False, 0.0, 3.0, notional=2.0)
    t = 1.0
    float_leg = 2.0 * (1.0 - hw1f.bond_price(r, t, 3.0, 0.1, 0.01, p0, fwd))
    want = _fixed_leg(s, t, r, 0.1, 0.01, p0, fwd) - float_leg
    assert np.allclose(swaps.swap_value(s, t, r, 0.1, 0.01, p0, fwd), want, atol=1e-13, rtol=0)
    # a last_fixing given on a reset date is not used
    assert np.array_equal(swaps.swap_value(s, t, r, 0.1, 0.01, p0, fwd, np.full(9, 0.5)),
                          swaps.swap_value(s, t, r, 0.1, 0.01, p0, fwd))
    # before the first reset: N (P(t, start) - P(t, T_n))
    s2 = swap(0.04, False, 1.0, 3.0)
    t = 0.5
    float_leg = hw1f.bond_price(r, t, 1.0, 0.1, 0.01, p0, fwd) - hw1f.bond_price(r, t, 3.0, 0.1, 0.01, p0, fwd)
    assert np.allclose(swaps.swap_value(s2, t, r, 0.1, 0.01, p0, fwd),
                       _fixed_leg(s2, t, r, 0.1, 0.01, p0, fwd) - float_leg, atol=1e-13, rtol=0)
    # between resets: the golden 2 sinh(0.0075) = 0.01500014, sigma = 0, a one-period swap [0, 0.5], t = 0.25
    p1, f1 = flat(0.03)
    one = swap(0.0, True, 0.0, 0.5, freq=2)                          # payer of a zero fixed rate: value = floating leg
    fixing = (math.exp(0.015) - 1.0) / 0.5                           # the simple rate fixed at t = 0
    r25 = hw1f.phi(arr(0.25), 0.1, 0.0, f1)
    got = swaps.swap_value(one, 0.25, r25, 0.1, 0.0, p1, f1, np.array([fixing]))
    assert got[0] == pytest.approx(2.0 * math.sinh(0.0075), abs=5e-9)
    assert got[0] == pytest.approx(0.01500014, abs=5e-9)
    # the same through a route of its own: a two-period swap, deterministic curve, flows discounted by hand
    two = swap(0.0, True, 0.0, 1.0, freq=2)
    coupon1 = math.exp(0.015) - 1.0                                   # fixed at 0, paid at 0.5
    coupon2 = math.exp(0.015) - 1.0                                   # forward coupon fixed at 0.5, paid at 1.0
    by_hand = coupon1 * math.exp(-0.03 * 0.25) + coupon2 * math.exp(-0.03 * 0.75)
    got2 = swaps.swap_value(two, 0.25, r25, 0.1, 0.0, p1, f1, np.array([fixing]))
    assert got2[0] == pytest.approx(by_hand, abs=1e-12)
    # linear in the fixing for the payer; the fixing is required strictly between resets
    v_lo = swaps.swap_value(two, 0.25, r25, 0.1, 0.0, p1, f1, np.array([0.0]))
    v_hi = swaps.swap_value(two, 0.25, r25, 0.1, 0.0, p1, f1, np.array([0.05]))
    assert v_hi[0] - v_lo[0] == pytest.approx(0.5 * 0.05 * math.exp(-0.03 * 0.25), abs=1e-12)
    with pytest.raises(ValueError, match="last_fixing"):
        swaps.swap_value(two, 0.25, r25, 0.1, 0.0, p1, f1)
    with pytest.raises(ValueError, match="last_fixing"):
        swaps.swap_value(two, 0.25, r25, 0.1, 0.0, p1, f1, np.array([0.01, 0.02]))
    # no flow after t: value 0
    for t in (3.0, 3.5, 10.0):
        assert np.array_equal(swaps.swap_value(s, t, r, 0.1, 0.01, p0, fwd), np.zeros(9))


# ---------------------------------------------------------------------------------------------------- RV-959


def test_par_fixed_rate_golden():
    p0, _ = flat(0.03)
    got = swaps.par_fixed_rate(1.0, 1, p0)
    assert got == pytest.approx((1.0 - math.exp(-0.03)) / math.exp(-0.03), abs=1e-15)
    assert got == pytest.approx(0.030455, abs=5e-7)
    # forward starting, semi-annual: (P(0, 1) - P(0, 3)) / sum 0.5 P(0, T_i)
    want = (math.exp(-0.03) - math.exp(-0.09)) / sum(0.5 * math.exp(-0.03 * tt) for tt in (1.5, 2.0, 2.5, 3.0))
    assert swaps.par_fixed_rate(3.0, 2, p0, start=1.0) == pytest.approx(want, abs=1e-14)
    # the value of a swap at that rate is 0 (any start, frequency and curve)
    p1, f1 = curve()
    for start, mat, freq in ((0.0, 5.0, 2), (1.0, 4.0, 4), (0.5, 2.25, 2)):
        rate = swaps.par_fixed_rate(mat, freq, p1, start=start)
        r0 = hw1f.phi(arr(0.0), 0.1, 0.01, f1)
        v = swaps.swap_value(swap(rate, False, start, mat, freq), 0.0, r0, 0.1, 0.01, p1, f1)
        assert abs(v[0]) < 1e-12
    with pytest.raises(ValueError, match="maturity"):
        swaps.par_fixed_rate(0.5, 2, p1, start=1.0)
    with pytest.raises(ValueError, match="frequency"):
        swaps.par_fixed_rate(1.0, 0, p1)


# ---------------------------------------------------------------------------------------------------- RV-960


def _simulate(swap_list, report, n, seed=17, a=0.1, sigma=0.01):
    p0, fwd = curve()
    grid = swaps.simulation_grid(report, swap_list)
    r = hw1f.simulate_short_rate(grid, a, sigma, fwd, n, seed)
    return grid, r, p0, fwd


def test_swap_values_on_grid():
    p0, fwd = curve()
    a, sigma = 0.1, 0.01
    par = swaps.par_fixed_rate(2.0, 2, p0)
    book = [swap(par, False, 0.0, 2.0, trade_id="a"), swap(par + 0.005, True, 0.0, 1.5, trade_id="b")]
    report = arr(0.0, 0.25, 0.5, 0.9, 1.0, 1.75, 2.0)
    grid = swaps.simulation_grid(report, book)
    assert np.allclose(grid, arr(0.0, 0.25, 0.5, 0.9, 1.0, 1.5, 1.75, 2.0)) and np.all(np.diff(grid) > 0.0)
    assert np.array_equal(swaps.simulation_grid(grid, book), grid)    # already a union
    n = 20_000
    r = hw1f.simulate_short_rate(grid, a, sigma, fwd, n, 17)
    v = swaps.swap_values_on_grid(book, grid, r, a, sigma, p0, fwd)
    assert v.shape == (2, n, grid.size)
    # equal to swap_value at each grid point, the fixing read from the same path
    first = book[0]
    resets = swaps.reset_times(first)
    for k, t in enumerate(grid):
        j = int(np.searchsorted(resets, t + 1e-9, side="right")) - 1                # last reset on or before t
        if abs(resets[j] - t) <= 1e-9 or t < resets[0]:
            fix = None
        else:
            nxt = swaps.cash_flow_times(first)[j]
            idx = int(np.argmin(np.abs(grid - resets[j])))
            p_fix = hw1f.bond_price(r[:, idx], float(resets[j]), float(nxt), a, sigma, p0, fwd)
            fix = (1.0 / p_fix - 1.0) / (nxt - resets[j])
        want = swaps.swap_value(first, float(t), r[:, k], a, sigma, p0, fwd, fix)
        assert np.allclose(v[0, :, k], want, atol=1e-13, rtol=0), k
    # a receiver at par: mean value 0 at the first grid point after 0 within the Monte Carlo error, and the discounted
    # mean is 0 at every grid point (the martingale property, which tests the between-reset values on the paths)
    numeraire = hw1f.bank_account(grid, r, a, sigma, p0, fwd)
    par_book = [swap(par, False, 0.0, 2.0)]
    vp = swaps.swap_values_on_grid(par_book, grid, r, a, sigma, p0, fwd)[0]
    assert np.max(np.abs(vp[:, 0])) < 1e-12                           # t = 0: every path at par
    k = 1
    assert abs(float(vp[:, k].mean())) <= 4.0 * float(vp[:, k].std(ddof=1)) / math.sqrt(n)
    disc = vp / numeraire
    se = disc.std(axis=0, ddof=1) / math.sqrt(n)
    # the value after the flows of the dates already passed is not a martingale on its own: the flow paid at a date
    # leaves
    # the portfolio, so the discounted mean is minus the present value, today, of the flows paid at or before the date
    s0 = par_book[0]
    flow_pv = np.array([s0.notional * s0.fixed_rate * (tt - tp) * float(p0(tt)) - (float(p0(tp)) - float(p0(tt)))
                        for tt, tp in zip(swaps.cash_flow_times(s0), swaps.reset_times(s0), strict=True)])
    paid = np.array([flow_pv[swaps.cash_flow_times(s0) <= t + 1e-9].sum() for t in grid])
    assert np.all(np.abs(disc.mean(axis=0) + paid) <= 4.0 * np.maximum(se, 1e-12))
    assert abs(paid[-1]) < 1e-12 and np.max(np.abs(paid)) > 1e-3              # par: all flows together are worth 0
    payer = swaps.swap_values_on_grid([swap(par, True, 0.0, 2.0)], grid, r, a, sigma, p0, fwd)[0]
    assert np.allclose(payer, -vp, atol=1e-15, rtol=0)
    # the grid must hold every reset date the valuation reads
    bad_grid = arr(0.0, 0.25, 0.75)
    r_bad = hw1f.simulate_short_rate(bad_grid, a, sigma, fwd, 10, 1)
    with pytest.raises(ValueError, match="grid"):
        swaps.swap_values_on_grid(par_book, bad_grid, r_bad, a, sigma, p0, fwd)
    with pytest.raises(ValueError, match="short_rate"):
        swaps.swap_values_on_grid(par_book, grid, r[:, :-1], a, sigma, p0, fwd)
    with pytest.raises(ValueError, match="swaps"):
        swaps.swap_values_on_grid([], grid, r, a, sigma, p0, fwd)
