"""RV-1021 to RV-1023: the dynamic NII simulation, the Hull-White one-factor and dynamic Nelson-Siegel scenarios, EaR
and
EVaR (contract `engine-irrbb-api`, module simulation)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from riskvalidation.irrbb import curves as c
from riskvalidation.irrbb import simulation as sim


def arr(*x):
    return np.array(x, dtype=float)


def book(path, reprice=(0, 0), beta_l=0.5, g=0.0):
    """The golden book: one asset (100, 5%, spread 2%, beta 1) and one liability (100, 1.5%, beta 0.5)."""
    return sim.simulate_nii(arr(100, 100), arr(1, -1), arr(0.05, 0.015), arr(0.02, 0.0), arr(1.0, beta_l),
                            np.array(reprice), np.asarray(path, dtype=float), g)


def nii_loop(balances, sides, rate0, spread, beta, reprice, path, g):
    """The reset rule written as a loop over the 0-based month m: k = 0 resets every month (path[m]); k >= 1 keeps rate0
    for m < k and from m = k uses the rate set at the latest reset month (m // k) * k."""
    out = []
    for m in range(len(path)):
        total = 0.0
        for i in range(len(balances)):
            k = int(reprice[i])
            if k == 0:
                rate = spread[i] + beta[i] * path[m]
            elif m < k:
                rate = rate0[i]
            else:
                rate = spread[i] + beta[i] * path[(m // k) * k]
            total += sides[i] * balances[i] * (1 + g) ** (m / 12) * rate / 12
        out.append(total)
    return np.array(out)


# ---------------------------------------------------------------------------------------------------- RV-1021


def test_simulate_nii_goldens():
    flat = book(np.full(12, 0.03))
    assert flat.total == pytest.approx(100 * 0.05 - 100 * 0.015, abs=1e-12)
    assert flat.total == pytest.approx(3.5, abs=1e-12)
    assert flat.monthly_nii.shape == (12,) and np.allclose(flat.monthly_nii, 3.5 / 12, atol=1e-15)
    # the path at 0.04 in every month: asset 0.02 + 0.04, liability 0.5 * 0.04
    up = book(np.full(12, 0.04))
    assert up.total == pytest.approx(100 * 0.06 - 100 * 0.02, abs=1e-12)
    assert up.total == pytest.approx(4.0, abs=1e-12)
    # a step to 0.04 from the second month: the first month still at 0.03
    step = book(np.r_[0.03, np.full(11, 0.04)])
    assert step.total == pytest.approx((3.5 + 11 * 4.0) / 12, abs=1e-12)
    assert dataclasses.is_dataclass(flat) and flat.total == pytest.approx(float(flat.monthly_nii.sum()), abs=1e-15)


def test_simulate_nii_invariants():
    rng = np.random.default_rng(9)
    n, months = 6, 24
    bal = rng.uniform(10, 100, n)
    side = np.array([1, 1, 1, -1, -1, -1])
    spread = rng.uniform(0.0, 0.02, n)
    beta = rng.uniform(0.0, 1.0, n)
    reprice = np.array([0, 1, 3, 0, 6, 40])
    path = 0.03 + 0.001 * rng.normal(size=months).cumsum()
    rate0 = spread + beta * 0.03
    path = np.abs(path)
    got = sim.simulate_nii(bal, side, rate0, spread, beta, reprice, path, 0.0)
    assert np.allclose(got.monthly_nii, nii_loop(bal, side, rate0, spread, beta, reprice, path, 0.0), atol=1e-12)
    grown = sim.simulate_nii(bal, side, rate0, spread, beta, reprice, path, 0.05)
    assert np.allclose(grown.monthly_nii, nii_loop(bal, side, rate0, spread, beta, reprice, path, 0.05), atol=1e-12)
    # a flat path with rate0 = spread + beta r: the monthly NII is constant
    flatp = np.full(months, 0.03)
    const = sim.simulate_nii(bal, side, rate0, spread, beta, reprice, flatp, 0.0)
    assert np.allclose(const.monthly_nii, const.monthly_nii[0], atol=1e-12)
    # an item that does not reset within the horizon is unaffected by the path
    def fixed(p):
        return sim.simulate_nii(bal[5:], side[5:], rate0[5:], spread[5:], beta[5:], reprice[5:], p, 0.0)

    assert np.array_equal(fixed(path).monthly_nii, fixed(path + 0.05).monthly_nii)
    # an item that resets only at month 6 (0-based index 6) is unaffected by the path before it
    def r6(p):
        return sim.simulate_nii(bal[4:5], side[4:5], rate0[4:5], spread[4:5], beta[4:5], reprice[4:5], p, 0.0)

    assert np.array_equal(r6(path).monthly_nii[:6], r6(path + 0.05).monthly_nii[:6])
    assert not np.allclose(r6(path).monthly_nii[6:], r6(path + 0.05).monthly_nii[6:])
    # the rate series of an item with reprice_months = 3 (0-based months): rate0 for m = 0, 1, 2, then the rate set at
    # m = 3 (path[3]) for m = 3, 4, 5, then path[6] for m = 6, 7, 8
    one = lambda p: sim.simulate_nii(arr(1200.0), arr(1.0), arr(0.07), arr(0.01), arr(2.0), np.array([3]), p, 0.0)  # noqa: E731
    pth = np.arange(12) * 0.001 + 0.01
    rates = one(pth).monthly_nii * 12 / 1200.0
    assert rates[:3].tolist() == pytest.approx([0.07] * 3, abs=1e-15)
    assert rates[3:6].tolist() == pytest.approx([0.01 + 2.0 * pth[3]] * 3, abs=1e-15)
    assert rates[6:9].tolist() == pytest.approx([0.01 + 2.0 * pth[6]] * 3, abs=1e-15)
    # linear in the balances, and a zero balance contributes nothing
    twice = sim.simulate_nii(2 * bal, side, rate0, spread, beta, reprice, path, 0.0)
    assert np.allclose(twice.monthly_nii, 2 * got.monthly_nii, atol=1e-12)
    z = bal.copy()
    z[2] = 0.0
    keep = [0, 1, 3, 4, 5]
    assert np.allclose(sim.simulate_nii(z, side, rate0, spread, beta, reprice, path, 0.0).monthly_nii,
                       sim.simulate_nii(bal[keep], side[keep], rate0[keep], spread[keep], beta[keep], reprice[keep],
                                        path, 0.0).monthly_nii, atol=1e-12)
    # raising a liability beta never raises NII (positive rates)
    lo = book(np.full(12, 0.03), beta_l=0.3).total
    hi = book(np.full(12, 0.03), beta_l=0.9).total
    assert hi <= lo
    # NII rises with a rate path when the asset is indexed and the liability is not
    assert book(np.full(12, 0.05), beta_l=0.0).total > book(np.full(12, 0.03), beta_l=0.0).total
    # inputs are not mutated
    p0 = path.copy()
    sim.simulate_nii(bal, side, rate0, spread, beta, reprice, path, 0.0)
    assert np.array_equal(path, p0)
    # refusals
    base = (bal, side, rate0, spread, beta, reprice)
    with pytest.raises(ValueError):
        sim.simulate_nii(*base, np.full(11, 0.03), 0.0)
    with pytest.raises(ValueError):
        sim.simulate_nii(*base, np.full(37, 0.03), 0.0)
    with pytest.raises(ValueError):
        sim.simulate_nii(-bal, side, rate0, spread, beta, reprice, path, 0.0)
    with pytest.raises(ValueError):
        sim.simulate_nii(bal, np.array([1, 0, 1, -1, -1, -1]), rate0, spread, beta, reprice, path, 0.0)
    with pytest.raises(ValueError):
        sim.simulate_nii(bal[:3], side, rate0, spread, beta, reprice, path, 0.0)
    with pytest.raises(ValueError):
        sim.simulate_nii(bal, side, rate0, spread, beta, np.array([0, 1, 3, 0, -6, 40]), path, 0.0)


# ---------------------------------------------------------------------------------------------------- RV-1022


def test_hw1f_scenarios_golden_and_repricing():
    flat = sim.hw1f_scenarios(arr(0.03, 0.03), arr(1, 10), a=0.1, sigma=0.0, horizon_months=24, n_paths=5, seed=1)
    assert flat.shape == (5, 25) and np.all(flat == 0.03)
    # at zero volatility every path is the instantaneous forward curve f(0, t) = R(t) + t R'(t)
    ten, zr = arr(1, 5, 10), arr(0.02, 0.03, 0.035)
    det = sim.hw1f_scenarios(zr, ten, a=0.1, sigma=0.0, horizon_months=60, n_paths=2, seed=0)
    t = np.arange(61) / 12
    h = 1e-6
    def capital(x):  # the capitalisation factor R(x) x of the zero curve
        return float(c.zero_rates_at(ten, zr, np.array([x]))[0]) * x

    fwd = np.array([(capital(x + h) - capital(max(x - h, 0.0))) / (x + h - max(x - h, 0.0)) for x in t])
    inner = (t > 1e-3) & (np.abs(t - 1) > 1e-3) & (np.abs(t - 5) > 1e-3)
    assert np.allclose(det[0][inner], fwd[inner], atol=2e-6)
    assert np.array_equal(det[0], det[1])
    # seeded, antithetic pairs
    p = sim.hw1f_scenarios(zr, ten, a=0.1, sigma=0.01, horizon_months=36, n_paths=1000, seed=3)
    same = sim.hw1f_scenarios(zr, ten, a=0.1, sigma=0.01, horizon_months=36, n_paths=1000, seed=3)
    other = sim.hw1f_scenarios(zr, ten, a=0.1, sigma=0.01, horizon_months=36, n_paths=1000, seed=4)
    assert np.array_equal(p, same) and not np.array_equal(p, other)
    pair_sum = p[:500] + p[500:]  # path k and k + P/2 are mirror images around the same deterministic centre alpha(t)
    assert np.allclose(pair_sum, pair_sum[0], atol=1e-12)
    # the convexity term sigma^2 / (2 a^2) (1 - e^(-a t))^2
    convexity = 0.01**2 / (2 * 0.1**2) * (1 - np.exp(-0.1 * np.arange(37) / 12)) ** 2
    assert np.allclose(pair_sum[0] / 2, det[0][:37] + convexity, atol=1e-12)
    assert np.all(p[:, 0] == p[0, 0])
    # alpha(t) satisfies the dossier's theta(t) = f'(0, t) + a f(0, t) + sigma^2 / (2 a) (1 - e^(-2 a t)) through
    # alpha' + a alpha (central differences inside the segment from 1 to 5 years, where f is linear in t)
    a_, s_ = 0.1, 0.01
    alpha = pair_sum[0] / 2
    k = 24  # month 24 = 2 years; f(0, t) = R(t) + t R'(t) with R' = 0.0025 in the segment, so f' = 2 * 0.0025
    t0 = k / 12
    d_alpha = (alpha[k + 1] - alpha[k - 1]) / (2 / 12)
    f0 = float(c.zero_rates_at(ten, zr, arr(t0))[0]) + t0 * 0.0025
    theta = 2 * 0.0025 + a_ * f0 + s_**2 / (2 * a_) * (1 - math.exp(-2 * a_ * t0))
    assert d_alpha + a_ * alpha[k] == pytest.approx(theta, abs=2e-6)
    # the average discount factor of the paths reprices the initial curve within Monte Carlo error
    paths = sim.hw1f_scenarios(zr, ten, a=0.1, sigma=0.01, horizon_months=36, n_paths=20000, seed=11)
    integral = (paths[:, :-1] + paths[:, 1:]).sum(axis=1) / 2 / 12  # trapezoid on the monthly grid, 3 years
    df = np.exp(-integral)
    target = math.exp(-float(c.zero_rates_at(ten, zr, arr(3.0))[0]) * 3.0)
    assert abs(df.mean() - target) <= 4 * df.std(ddof=1) / math.sqrt(len(df)) + 3e-4
    # a positive volatility spreads the paths, a negative one is refused
    assert paths[:, -1].std() > 0.005
    for bad in (dict(a=0.0, sigma=0.01), dict(a=0.1, sigma=-0.01)):
        with pytest.raises(ValueError):
            sim.hw1f_scenarios(zr, ten, horizon_months=12, n_paths=10, seed=0, **bad)
    with pytest.raises(ValueError):
        sim.hw1f_scenarios(zr, ten, a=0.1, sigma=0.01, horizon_months=0, n_paths=10, seed=0)
    with pytest.raises(ValueError):
        sim.hw1f_scenarios(zr, ten, a=0.1, sigma=0.01, horizon_months=12, n_paths=0, seed=0)


def test_dns_scenarios_golden_and_no_lookahead():
    m = arr(3, 12, 36, 120)
    const = np.tile(arr(0.05, -0.02, 0.01), (12, 1))
    out = sim.dns_scenarios(const, m, 6, 4, 2)
    assert out.shape == (4, 6, 4)
    assert np.allclose(out, c.dns_yield(m, 0.05, -0.02, 0.01), atol=1e-12)
    rng = np.random.default_rng(5)
    hist = np.zeros((100, 3))
    for t in range(1, 100):
        hist[t] = 0.01 + 0.8 * hist[t - 1] + rng.normal(0, 0.01, 3)
    h0 = hist.copy()
    a = sim.dns_scenarios(hist, m, 12, 500, 7)
    assert np.array_equal(a, sim.dns_scenarios(hist, m, 12, 500, 7)) and np.array_equal(hist, h0)
    # the scenarios are the simulated factors through the loadings
    fac = c.forecast_dns_factors(hist, 12, 500, 7)
    assert np.allclose(a, fac @ c.dns_loadings(m).T, atol=1e-14)
    # only the history given is used: truncating the history changes the scenarios (the bake passes the factors up to
    # the valuation date), while the same truncated history always gives the same scenarios
    b = sim.dns_scenarios(hist[:60], m, 12, 500, 7)
    assert not np.allclose(a, b) and np.array_equal(b, sim.dns_scenarios(hist[:60], m, 12, 500, 7))
    with pytest.raises(ValueError):
        sim.dns_scenarios(hist, arr(0.0, 12.0), 12, 10, 1)


# ---------------------------------------------------------------------------------------------------- RV-1023


def test_ear_evar_golden_and_invariants():
    paths = np.arange(1.0, 101.0)
    r = sim.ear_evar(paths, paths, 60.0, 60.0, level=0.99)
    assert r.ear == pytest.approx(58.01, abs=1e-12) and r.evar == pytest.approx(58.01, abs=1e-12)
    assert np.quantile(paths, 0.01) == pytest.approx(1.99, abs=1e-12)
    # EaR and EVaR have their own bases and paths
    r2 = sim.ear_evar(paths, 2 * paths, 60.0, 130.0, level=0.99)
    assert r2.ear == pytest.approx(58.01, abs=1e-12) and r2.evar == pytest.approx(130.0 - 3.98, abs=1e-12)
    # all-equal paths give base - path
    same = np.full(50, 7.0)
    assert sim.ear_evar(same, same, 10.0, 12.0).ear == 3.0 and sim.ear_evar(same, same, 10.0, 12.0).evar == 5.0
    # monotone in the level
    rng = np.random.default_rng(1)
    x = rng.normal(0, 10, 1000)
    levels = [0.5, 0.9, 0.95, 0.99, 0.999]
    ears = [sim.ear_evar(x, x, 0.0, 0.0, level=lv).ear for lv in levels]
    assert all(p <= q + 1e-12 for p, q in zip(ears, ears[1:], strict=False))
    assert dataclasses.is_dataclass(r)
    for bad in (np.array([1.0]), np.array([])):
        with pytest.raises(ValueError):
            sim.ear_evar(bad, bad, 0.0, 0.0)
    with pytest.raises(ValueError):
        sim.ear_evar(paths, paths[:50], 0.0, 0.0)
    for lv in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            sim.ear_evar(paths, paths, 0.0, 0.0, level=lv)
