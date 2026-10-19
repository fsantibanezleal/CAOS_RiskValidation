"""RV-968 to RV-971: collateral, the margin period of risk and the dynamic initial margin of `riskvalidation.ccr`
(contract `engine-ccr-api`, module collateral; ORE `xva.tex` at tag 1.8.17.0, CRE53.24 to CRE53.26)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from riskvalidation.ccr import collateral as co


def arr(*x):
    return np.array(x, dtype=float)


# ---------------------------------------------------------------------------------------------------- RV-968


def test_credit_support_and_delivery():
    csa = co.credit_support_amount
    assert csa(arr(100.0), 0.0, 0.0, 0.0)[0] == 100.0
    assert csa(arr(100.0), 0.0, 30.0, 0.0)[0] == 70.0
    assert csa(arr(-50.0), 0.0, 0.0, 20.0)[0] == -30.0
    assert csa(arr(0.0), 5.0, 0.0, 0.0)[0] == 5.0                     # the independent amount offsets the threshold
    # below the threshold nothing is called; x = NPV + IA decides the branch
    assert csa(arr(20.0), 0.0, 30.0, 0.0)[0] == 0.0
    assert csa(arr(-50.0), 60.0, 0.0, 0.0)[0] == 10.0
    assert csa(arr(-50.0), 0.0, 0.0, 80.0)[0] == 0.0                  # the whole shortfall is inside the threshold
    assert csa(arr(0.0), 0.0, 10.0, 10.0)[0] == 0.0
    npv = np.linspace(-200.0, 200.0, 41)
    out = csa(npv, 10.0, 25.0, 15.0)
    assert out.shape == npv.shape
    x = npv + 10.0
    want = np.where(x >= 0.0, np.maximum(0.0, x - 25.0), np.minimum(0.0, x + 15.0))
    assert np.array_equal(out, want)
    with pytest.raises(ValueError, match="threshold_receive"):
        csa(npv, 0.0, -1.0, 0.0)
    with pytest.raises(ValueError, match="threshold_pay"):
        csa(npv, 0.0, 0.0, -1.0)
    # delivery amount
    da = co.delivery_amount
    assert da(arr(100.0), arr(90.0), arr(0.0), 10.0)[0] == 10.0
    assert da(arr(100.0), arr(92.0), arr(0.0), 10.0)[0] == 0.0         # the shortfall 8 is below the MTA
    assert da(arr(100.0), arr(112.0), arr(0.0), 10.0)[0] == -12.0
    assert da(arr(100.0), arr(111.0), arr(0.0), 10.0)[0] == -11.0
    assert da(arr(100.0), arr(92.0), arr(0.0), 0.0)[0] == 8.0          # MTA = 0 gives the shortfall
    assert da(arr(100.0), arr(90.0), arr(6.0), 5.0)[0] == 0.0          # a call of 6 outstanding leaves 4, below the MTA
    assert da(arr(100.0), arr(90.0), arr(2.0), 5.0)[0] == 8.0
    with pytest.raises(ValueError, match="mta"):
        da(arr(1.0), arr(0.0), arr(0.0), -1.0)
    with pytest.raises(ValueError, match="collateral"):
        da(arr(1.0, 2.0), arr(0.0), arr(0.0), 0.0)


# ---------------------------------------------------------------------------------------------------- RV-969


def test_mpor():
    f = co.mpor_floor
    assert f() == 10 and f(repo_only=True) == 5 and f(n_trades=5001) == 20 and f(disputes_over_two=True) == 20
    assert f(n_trades=5000) == 10                                     # "exceeds 5000"
    assert f(illiquid=True) == 20 and f(repo_only=True, n_trades=6000) == 20
    assert f(repo_only=True, disputes_over_two=True) == 10            # double of the floor of 5
    assert f(illiquid=True, disputes_over_two=True) == 40
    with pytest.raises(ValueError, match="n_trades"):
        f(n_trades=-1)
    g = co.mpor_with_remargining
    assert g(10, 1) == 10 and g(10, 5) == 14 and g(5, 3) == 7 and g(20, 2) == 21
    for floor, n, field in ((0, 1, "floor"), (10, 0, "remargining_days"), (10, -3, "remargining_days")):
        with pytest.raises(ValueError, match=field):
            g(floor, n)


# ---------------------------------------------------------------------------------------------------- RV-970


def test_collateralised_exposure_and_vm():
    lagged = co.collateralised_exposure(arr(10.0).reshape(1, 1), arr(8.0).reshape(1, 1))
    assert lagged[0, 0] == 2.0                                        # the ORE lagged toy example
    assert co.collateralised_exposure(arr(8.0).reshape(1, 1), arr(10.0).reshape(1, 1))[0, 0] == 0.0
    with pytest.raises(ValueError, match="collateral"):
        co.collateralised_exposure(np.zeros((2, 3)), np.zeros((2, 2)))
    # a constant value with zero threshold and MTA: exposure 0 (and the independent amount is posted at the first date)
    const = np.full((4, 6), 7.0)
    exposure, balance = co.variation_margin_exposure(const, const, 0.0, 0.0, 0.0, 0.0)
    assert np.array_equal(exposure, np.zeros((4, 6))) and np.array_equal(balance, const)
    negative = np.full((4, 6), -7.0)
    exposure, balance = co.variation_margin_exposure(negative, negative, 0.0, 0.0, 0.0, 0.0)
    assert np.array_equal(exposure, np.zeros((4, 6))) and np.array_equal(balance, negative)
    zero = np.zeros((1, 3))
    _, bal = co.variation_margin_exposure(zero, zero, 5.0, 0.0, 0.0, 0.0)
    assert np.array_equal(bal, np.full((1, 3), 5.0))                  # CSA = IA at NPV = 0
    # the balance follows credit_support_amount and delivery_amount with the MTA, date by date, by hand
    path = np.array([[0.0, 5.0, 12.0, 11.0, 3.0]])
    _, bal = co.variation_margin_exposure(path, path, 0.0, 0.0, 0.0, 4.0)
    assert np.array_equal(bal, np.array([[0.0, 5.0, 12.0, 12.0, 3.0]]))
    exposure, _ = co.variation_margin_exposure(path, path + 1.0, 0.0, 0.0, 0.0, 4.0)
    assert np.array_equal(exposure, np.maximum(path + 1.0 - bal, 0.0))
    # a threshold keeps the first part uncollateralised
    exposure, bal = co.variation_margin_exposure(path, path, 0.0, 6.0, 0.0, 0.0)
    assert np.array_equal(bal, np.array([[0.0, 0.0, 6.0, 5.0, 0.0]]))
    assert np.array_equal(exposure, np.maximum(path - bal, 0.0))
    # zero threshold and an MPoR tending to zero: the close-out value tends to the collateral, the exposure to zero
    rng = np.random.default_rng(3)
    main = np.cumsum(rng.normal(0.0, 1.0, (500, 8)), axis=1)
    jump = rng.normal(0.0, 1.0, (500, 8))
    means = []
    for lag in (1.0, 0.1, 0.01, 0.0):
        e, _ = co.variation_margin_exposure(main, main + lag * jump, 0.0, 0.0, 0.0, 0.0)
        means.append(float(e.mean()))
    assert means[0] > means[1] > means[2] > means[3] == 0.0
    assert means[1] == pytest.approx(0.1 * means[0], rel=1e-9)        # linear in the lag by construction
    # an uncollateralised netting set (no CSA: infinite thresholds) takes the main grid value
    inf = math.inf
    e, bal = co.variation_margin_exposure(main, main + jump, 0.0, inf, inf, 0.0)
    assert np.array_equal(e, np.maximum(main, 0.0)) and np.array_equal(bal, np.zeros_like(main))
    # never above the uncollateralised exposure plus the collateral posted by the bank
    csas = ((0.0, 0.0, 0.0, 0.0), (5.0, 3.0, 1.0, 0.0), (2.0, 2.0, 0.5, 3.0), (20.0, 20.0, 5.0, -1.0))
    for th_r, th_p, mta, ia in csas:
        e, bal = co.variation_margin_exposure(main, main + jump, abs(ia), th_r, th_p, mta)
        assert np.all(e >= 0.0)
        assert np.all(e <= np.maximum(main + jump, 0.0) + np.maximum(-bal, 0.0) + 1e-12)
    with pytest.raises(ValueError, match="values_closeout"):
        co.variation_margin_exposure(main, main[:, :3], 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="mta"):
        co.variation_margin_exposure(main, main, 0.0, 0.0, 0.0, -1.0)


# ---------------------------------------------------------------------------------------------------- RV-971


def test_dynamic_initial_margin():
    x = arr(-2.0, -1.0, 1.0, 2.0)
    zeros = np.zeros(4)
    dim = co.dynamic_initial_margin(zeros, x, zeros, 0.99, 14, 14, order=0)
    s = math.sqrt(2.5)
    assert s == pytest.approx(1.581139, abs=5e-7)
    assert np.allclose(dim, norm.ppf(0.99) * s, atol=1e-12, rtol=0)
    assert dim[0] == pytest.approx(2.326348 * 1.581139, abs=5e-6)
    assert dim[0] == pytest.approx(3.678279, abs=5e-7)
    assert abs(2.33 * s - 3.684053) < 1e-6 and abs(dim[0] - 3.684053) > 1e-3          # the rounded factor is not used
    # the horizon scales by sqrt(horizon / MPoR) in calendar days (10 business days are 14 calendar days)
    twice = co.dynamic_initial_margin(zeros, x, zeros, 0.99, 28, 14, order=0)
    assert np.allclose(twice, math.sqrt(2.0) * dim, atol=1e-12, rtol=0)
    # the flows of the window are part of X: X = V(t + MPoR) + F - V(t)
    flows = arr(1.0, -2.0, 0.5, 3.0)
    shifted = co.dynamic_initial_margin(zeros, x - flows, flows, 0.99, 14, 14, order=0)
    assert np.allclose(shifted, dim, atol=1e-12, rtol=0)
    # E[X] is the sample mean: a constant change has no variance, DIM 0
    const = co.dynamic_initial_margin(zeros, np.full(4, 3.0), zeros, 0.99, 14, 14, order=0)
    assert np.array_equal(const, zeros)
    # the regression of X^2 on a polynomial of the regressor recovers a quadratic conditional variance
    rng = np.random.default_rng(8)
    n = 40_000
    s_reg = rng.normal(0.0, 1.0, n)
    z = rng.normal(0.0, 1.0, n)
    now = s_reg
    mpor = now + (0.02 + 0.01 * s_reg) * z
    d2 = co.dynamic_initial_margin(now, mpor, np.zeros(n), 0.99, 14, 14, order=2)
    truth = norm.ppf(0.99) * np.abs(0.02 + 0.01 * s_reg)
    assert np.all(d2 >= 0.0)
    assert float(np.mean(np.abs(d2 / truth - 1.0))) < 0.05
    d0 = co.dynamic_initial_margin(now, mpor, np.zeros(n), 0.99, 14, 14, order=0)
    assert np.ptp(d0) == 0.0                                           # order 0: one DIM for all paths
    # an explicit regressor replaces the default (the value at t)
    d2b = co.dynamic_initial_margin(now, mpor, np.zeros(n), 0.99, 14, 14, order=2, regressor=s_reg)
    assert np.allclose(d2b, d2, atol=1e-12, rtol=0)
    # where the variance is zero the DIM is zero; a constant regressor does not break the fit
    flat = co.dynamic_initial_margin(np.zeros(6), np.zeros(6), np.zeros(6), 0.99, 14, 14, order=2, regressor=np.ones(6))
    assert np.array_equal(flat, np.zeros(6))
    assert np.all(np.isfinite(co.dynamic_initial_margin(zeros, x, zeros, 0.99, 14, 14, order=2, regressor=np.ones(4))))
    for kw, field in (({"quantile": 0.0}, "quantile"), ({"quantile": 1.0}, "quantile"),
                      ({"horizon_calendar_days": 0}, "horizon"),
                      ({"mpor_calendar_days": 0}, "mpor"), ({"order": -1}, "order")):
        args = dict(quantile=0.99, horizon_calendar_days=14, mpor_calendar_days=14, order=0)
        args.update(kw)
        with pytest.raises(ValueError, match=field):
            co.dynamic_initial_margin(zeros, x, zeros, **args)
    with pytest.raises(ValueError, match="values_mpor"):
        co.dynamic_initial_margin(zeros, x[:3], zeros, 0.99, 14, 14)
    with pytest.raises(ValueError, match="regressor"):
        co.dynamic_initial_margin(zeros, x, zeros, 0.99, 14, 14, regressor=np.ones(3))


def test_apply_initial_margin():
    assert co.apply_initial_margin(arr(5.0), arr(2.0))[0] == 3.0
    assert co.apply_initial_margin(arr(1.0), arr(2.0))[0] == 0.0
    e = np.random.default_rng(1).uniform(0.0, 5.0, (30, 4))
    assert np.array_equal(co.apply_initial_margin(e, np.zeros_like(e)), e)          # DIM = 0: unchanged
    d = np.random.default_rng(2).uniform(0.0, 3.0, (30, 4))
    out = co.apply_initial_margin(e, d)
    assert np.all(out >= 0.0) and np.all(out <= e) and np.array_equal(out, np.maximum(e - d, 0.0))
    with pytest.raises(ValueError, match="dim"):
        co.apply_initial_margin(arr(5.0), arr(-1.0))
    with pytest.raises(ValueError, match="dim"):
        co.apply_initial_margin(np.ones((2, 3)), np.ones((2, 2)))
