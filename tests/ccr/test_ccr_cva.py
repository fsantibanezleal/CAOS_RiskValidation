"""RV-972 to RV-975: CVA, DVA and wrong-way risk of `riskvalidation.ccr` (contract `engine-ccr-api`, module cva): the
credit triangle, the discretised CVA of dossier 05 C.2 and the Hull and White (2012) hazard model with its Appendix
example."""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskvalidation.ccr import cva as cv


def arr(*x):
    return np.array(x, dtype=float)


# ---------------------------------------------------------------------------------------------------- RV-972


def test_survival_pd_cva_dva():
    s = cv.survival_from_spread(arr(0.012), arr(5.0), 0.6)
    assert s[0] == pytest.approx(math.exp(-0.012 * 5.0 / 0.6), abs=1e-15)
    assert s[0] == pytest.approx(0.904837, abs=5e-7)
    t = np.linspace(0.0, 10.0, 21)
    curve = cv.survival_from_spread(np.full(21, 0.02), t, 0.6)
    assert curve[0] == 1.0 and np.all(np.diff(curve) < 0.0) and np.all((curve > 0.0) & (curve <= 1.0))
    assert np.array_equal(cv.survival_from_spread(np.zeros(21), t, 0.4), np.ones(21))
    assert cv.survival_from_spread(arr(0.01), arr(2.0), 1.0)[0] == pytest.approx(math.exp(-0.02), abs=1e-15)
    for kw, field in (({"lgd": 0.0}, "lgd"), ({"lgd": 1.5}, "lgd")):
        with pytest.raises(ValueError, match=field):
            cv.survival_from_spread(arr(0.01), arr(1.0), **kw)
    with pytest.raises(ValueError, match="spread"):
        cv.survival_from_spread(arr(-0.01), arr(1.0), 0.6)
    with pytest.raises(ValueError, match="t"):
        cv.survival_from_spread(arr(0.01), arr(-1.0), 0.6)
    # default probabilities
    pd = cv.default_probabilities(arr(1.0, 0.9, 0.8))
    assert np.allclose(pd, arr(0.1, 0.1), atol=1e-15, rtol=0)
    sv = cv.survival_from_spread(np.full(11, 0.015), np.linspace(0.0, 5.0, 11), 0.6)
    p = cv.default_probabilities(sv)
    assert p.shape == (10,) and np.all(p >= 0.0) and p.sum() == pytest.approx(sv[0] - sv[-1], abs=1e-15)
    with pytest.raises(ValueError, match="survival"):
        cv.default_probabilities(arr(1.0, 0.8, 0.9))
    with pytest.raises(ValueError, match="survival"):
        cv.default_probabilities(arr(1.0, 1.2))
    # CVA: epe [10, 20], pd [0.1, 0.1], lgd 0.6 gives 0.6 + 1.2 = 1.8
    value, terms = cv.cva(arr(10.0, 20.0), arr(0.1, 0.1), 0.6)
    assert value == pytest.approx(1.8, abs=1e-15) and np.allclose(terms, arr(0.6, 1.2), atol=1e-15, rtol=0)
    assert value == pytest.approx(float(np.sum(terms)), abs=1e-15)
    zero, zt = cv.cva(arr(10.0, 20.0), np.zeros(2), 0.6)
    assert zero == 0.0 and np.array_equal(zt, np.zeros(2))            # a zero spread: exactly 0
    assert cv.cva(arr(10.0, 20.0), arr(0.1, 0.1), 0.3)[0] == pytest.approx(value / 2.0, abs=1e-15)    # linear in lgd
    assert cv.cva(arr(20.0, 40.0), arr(0.1, 0.1), 0.6)[0] == pytest.approx(2.0 * value, abs=1e-15)    # linear in the EE
    assert value > 0.0
    # DVA: the same sum on the bank's own terms
    dvalue, dterms = cv.dva(arr(10.0, 20.0), arr(0.1, 0.1), 0.6)
    assert dvalue == pytest.approx(1.8, abs=1e-15) and np.allclose(dterms, terms, atol=1e-15, rtol=0)
    assert cv.dva(arr(10.0, 20.0), np.zeros(2), 0.6)[0] == 0.0 and dvalue > 0.0
    # a whole chain by hand: spread 125 bp, lgd 0.6, a profile on a quarterly grid
    t = np.arange(0, 9) * 0.25
    surv = np.exp(-0.0125 * t / 0.6)
    epe = np.array([5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0])
    by_hand = sum((surv[i - 1] - surv[i]) * 0.6 * epe[i - 1] for i in range(1, 9))
    assert cv.cva(epe, cv.default_probabilities(cv.survival_from_spread(np.full(9, 0.0125), t, 0.6)), 0.6)[0] == \
        pytest.approx(by_hand, abs=1e-13)
    with pytest.raises(ValueError, match="pd"):
        cv.cva(arr(1.0, 2.0), arr(0.1), 0.6)
    with pytest.raises(ValueError, match="pd"):
        cv.cva(arr(1.0, 2.0), arr(0.1, 1.5), 0.6)
    with pytest.raises(ValueError, match="epe_discounted"):
        cv.cva(arr(-1.0, 2.0), arr(0.1, 0.1), 0.6)
    with pytest.raises(ValueError, match="lgd"):
        cv.cva(arr(1.0, 2.0), arr(0.1, 0.1), 0.0)
    with pytest.raises(ValueError, match="lgd_bank"):
        cv.dva(arr(1.0, 2.0), arr(0.1, 0.1), 1.2)


# ---------------------------------------------------------------------------------------------------- RV-973


def test_b_from_two_points():
    b = cv.b_from_two_points(3.0, 0.05, 20.0, 0.10)
    assert b == pytest.approx((math.log(0.10) - math.log(0.05)) / 17.0, abs=1e-15)
    assert b == pytest.approx(0.040773, abs=5e-7) and abs(b - 0.0408) < 5e-5
    assert cv.b_from_two_points(3.0, 0.05, 20.0, 0.05) == 0.0
    assert cv.b_from_two_points(3.0, 0.10, 20.0, 0.05) < 0.0
    for args, field in (((3.0, 0.0, 20.0, 0.1), "hazard1"), ((3.0, 0.05, 20.0, -0.1), "hazard2"),
                        ((3.0, 0.05, 3.0, 0.1), "w2")):
        with pytest.raises(ValueError, match=field):
            cv.b_from_two_points(*args)


# ---------------------------------------------------------------------------------------------------- RV-974


def _bisect_a(w_col, cum, dt, b, target):
    """The sequential equation of the Appendix solved by a plain bisection in Python (a second route)."""
    lo, hi = -40.0, 10.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        avg = sum(math.exp(-(c + math.exp(mid + b * w) * dt)) for w, c in zip(w_col, cum, strict=True)) / len(w_col)
        if avg > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_wwr_calibrate_a_appendix():
    w = np.array([[100.0, 100.0], [200.0, 300.0], [300.0, 400.0]])    # paths in rows, steps in columns
    target = np.exp(-0.01 * arr(0.5, 1.0) / (1.0 - 0.0))              # spread 1 percent, recovery 0
    dt = arr(0.5, 0.5)
    a, h = cv.wwr_calibrate_a(w, target, dt, 0.01)
    assert a.shape == (2,) and h.shape == (3, 2)
    # the values recomputed on the printed equations
    assert np.allclose(a, arr(-6.91283, -7.85077), atol=5e-6, rtol=0)
    assert np.allclose(h[:, 0], arr(0.002705, 0.007352, 0.019984), atol=5e-7, rtol=0)
    assert np.allclose(h[:, 1], arr(0.001059, 0.007822, 0.021263), atol=5e-7, rtol=0)
    # the paper's printed digits: a within 2e-4, h within one unit of the last printed digit
    assert np.allclose(a, arr(-6.9128, -7.8509), atol=2e-4, rtol=0)
    assert np.allclose(h[:, 0], arr(0.00270, 0.00735, 0.01998), atol=1e-5, rtol=0)
    assert np.allclose(h[:, 1], arr(0.00106, 0.00782, 0.02126), atol=1e-5, rtol=0)
    # an independent bisection gives the same a, step by step, and the averages meet the targets
    a1 = _bisect_a(w[:, 0], [0.0, 0.0, 0.0], 0.5, 0.01, target[0])
    cum1 = [float(x) * 0.5 for x in h[:, 0]]
    a2 = _bisect_a(w[:, 1], cum1, 0.5, 0.01, target[1])
    assert a[0] == pytest.approx(a1, abs=1e-9) and a[1] == pytest.approx(a2, abs=1e-9)
    surv = np.exp(-np.cumsum(h * dt[None, :], axis=1)).mean(axis=0)
    assert np.max(np.abs(surv - target)) <= 1e-10
    assert np.allclose(h, np.exp(a[None, :] + 0.01 * w), atol=1e-15, rtol=0)
    # b = 0: every path has the same hazard, the one of the spread
    a0, h0 = cv.wwr_calibrate_a(w, target, dt, 0.0)
    assert np.all(h0 == h0[0:1, :])
    assert h0[0, 0] == pytest.approx(-math.log(target[0]) / 0.5, abs=1e-9)
    assert h0[0, 1] == pytest.approx(-math.log(target[1] / target[0]) / 0.5, abs=1e-9)
    # a realistic case: the calibration error stays below 1e-10 at every step, with and without the noise term
    rng = np.random.default_rng(6)
    steps = 8
    t = np.arange(1, steps + 1) * 0.5
    big_w = np.cumsum(rng.normal(0.0, 1.0, (2000, steps)), axis=1)
    tgt = np.exp(-0.0125 * t / 0.6)
    d = np.full(steps, 0.5)
    for b in (0.0, 0.03, 0.2, -0.1):
        a_, h_ = cv.wwr_calibrate_a(big_w, tgt, d, b)
        assert np.max(np.abs(np.exp(-np.cumsum(h_ * d, axis=1)).mean(axis=0) - tgt)) <= 1e-10
        assert np.all(h_ > 0.0)
    eps = rng.standard_normal((2000, steps))
    a_, h_ = cv.wwr_calibrate_a(big_w, tgt, d, 0.03, sigma_noise=0.2, eps=eps)
    assert np.max(np.abs(np.exp(-np.cumsum(h_ * d, axis=1)).mean(axis=0) - tgt)) <= 1e-10
    assert np.allclose(h_, np.exp(a_[None, :] + 0.03 * big_w + 0.2 * eps), atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="eps"):
        cv.wwr_calibrate_a(big_w, tgt, d, 0.03, sigma_noise=0.2)
    with pytest.raises(ValueError, match="eps"):
        cv.wwr_calibrate_a(big_w, tgt, d, 0.03, sigma_noise=0.2, eps=eps[:, :3])
    # a target that nonnegative hazards cannot reach (survival rising) is refused
    with pytest.raises(ValueError, match="survival_target"):
        cv.wwr_calibrate_a(w, arr(0.99, 0.995), dt, 0.01)
    with pytest.raises(ValueError, match="dt"):
        cv.wwr_calibrate_a(w, target, arr(0.5, 0.0), 0.01)
    with pytest.raises(ValueError, match="survival_target"):
        cv.wwr_calibrate_a(w, target[:1], dt, 0.01)


# ---------------------------------------------------------------------------------------------------- RV-975


def test_wwr_cva():
    rng = np.random.default_rng(12)
    p_paths, steps = 4000, 6
    t = np.arange(1, steps + 1) * 0.5
    dt = np.full(steps, 0.5)
    w = np.cumsum(rng.normal(0.0, 1.0, (p_paths, steps)), axis=1)
    exposure = np.maximum(w, 0.0) * 1.5
    target = np.exp(-0.0125 * t / 0.6)
    lgd = 0.6
    pd = cv.default_probabilities(np.concatenate([[1.0], target]))
    independent = cv.cva(exposure.mean(axis=0), pd, lgd)[0]
    got, se = cv.wwr_cva(w, exposure, target, dt, lgd, 0.0)
    assert got == independent                                          # bit for bit
    assert se > 0.0
    # the general path at b = 0 (no shortcut) agrees with the independent value to the calibration accuracy
    _, h0 = cv.wwr_calibrate_a(w, target, dt, 0.0)
    surv = np.exp(-np.cumsum(h0 * dt[None, :], axis=1))
    pdij = np.diff(np.concatenate([np.ones((p_paths, 1)), surv], axis=1), axis=1) * -1.0
    assert lgd * float(np.sum(np.mean(pdij * exposure, axis=0))) == pytest.approx(independent, rel=1e-9)
    # wrong-way risk raises CVA for a positively correlated exposure; right-way risk lowers it
    values = [cv.wwr_cva(w, exposure, target, dt, lgd, b)[0] for b in (-0.2, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4)]
    assert all(x < y for x, y in zip(values, values[1:], strict=False))
    # the result is the path-wise sum of the contract, recomputed with the calibrated hazards
    b = 0.1
    a_b, h_b = cv.wwr_calibrate_a(w, target, dt, b)
    s_b = np.exp(-np.cumsum(h_b * dt[None, :], axis=1))
    pd_ij = np.concatenate([np.ones((p_paths, 1)), s_b], axis=1)
    pd_ij = pd_ij[:, :-1] - pd_ij[:, 1:]
    per_path = lgd * np.sum(pd_ij * exposure, axis=1)
    val, err = cv.wwr_cva(w, exposure, target, dt, lgd, b)
    assert val == pytest.approx(float(per_path.mean()), abs=1e-12)
    assert err == pytest.approx(float(per_path.std(ddof=1)) / math.sqrt(p_paths), abs=1e-12)
    # an exposure independent of w: CVA stays at the independent value within the Monte Carlo error
    noise = np.maximum(rng.normal(0.0, 1.5, (p_paths, steps)), 0.0)
    base, base_se = cv.wwr_cva(w, noise, target, dt, lgd, 0.0)
    shifted, shifted_se = cv.wwr_cva(w, noise, target, dt, lgd, 0.1)
    assert abs(shifted - base) <= 4.0 * max(base_se, shifted_se)
    with pytest.raises(ValueError, match="exposure_discounted"):
        cv.wwr_cva(w, exposure[:, :3], target, dt, lgd, 0.1)
    with pytest.raises(ValueError, match="lgd"):
        cv.wwr_cva(w, exposure, target, dt, 0.0, 0.1)
    with pytest.raises(ValueError, match="exposure_discounted"):
        cv.wwr_cva(w, -exposure, target, dt, lgd, 0.1)
