"""RV-1315 to RV-1317: the scenario models, rungs 2 and 3 of the macro stress ladder (contract `engine-stress-api`,
module
stress.scenario_models; the optional extra `stress` brings statsmodels)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, optimize, stats

from riskvalidation.stress import scenario_models as sm

A = np.array([[0.5, 0.0], [0.2, 0.8]])


def _diag_fit(a=A, sigma=None):
    return sm.VarFit(intercept=np.zeros(2), coefficients=(a,), sigma=np.eye(2) * 0.01 if sigma is None else sigma)


# ---------------------------------------------------------------------------------------------------- RV-1315


def test_fit_var_recovers_the_generating_matrix():
    y = np.zeros((9, 2))
    y[0] = [1.0, 1.0]
    for t in range(1, 9):
        y[t] = A @ y[t - 1]
    fit = sm.fit_var(y, 1)
    assert np.max(np.abs(fit.coefficients[0] - A)) <= 1e-10 and np.max(np.abs(fit.intercept)) <= 1e-10
    assert np.max(np.abs(fit.sigma)) <= 1e-12 and np.allclose(fit.sigma, fit.sigma.T)
    # with an intercept, two lags and noise: the equation by equation OLS of an explicit loop
    rng = np.random.default_rng(8)
    n, t_obs = 3, 200
    a1 = np.array([[0.4, 0.1, 0.0], [0.0, 0.3, 0.1], [0.1, 0.0, 0.2]])
    a2 = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    c = np.array([0.5, -0.2, 0.1])
    yy = np.zeros((t_obs, n))
    for t in range(2, t_obs):
        yy[t] = c + a1 @ yy[t - 1] + a2 @ yy[t - 2] + rng.normal(0, 0.1, n)
    f2 = sm.fit_var(yy, 2)
    x = np.array([np.concatenate([[1.0], yy[t - 1], yy[t - 2]]) for t in range(2, t_obs)])
    for j in range(n):
        b = np.linalg.lstsq(x, yy[2:, j], rcond=None)[0]
        assert f2.intercept[j] == pytest.approx(b[0], abs=1e-10)
        assert np.max(np.abs(f2.coefficients[0][j] - b[1 : 1 + n])) <= 1e-10
        assert np.max(np.abs(f2.coefficients[1][j] - b[1 + n :])) <= 1e-10
    assert np.all(np.linalg.eigvalsh(f2.sigma) >= -1e-12) and np.allclose(f2.sigma, f2.sigma.T)
    assert np.max(np.abs(np.diag(f2.sigma) - 0.01)) <= 0.004
    # the independent route when statsmodels is installed
    tsa = pytest.importorskip("statsmodels.tsa.api")
    res = tsa.VAR(yy).fit(2, trend="c")
    assert np.max(np.abs(res.coefs[0] - f2.coefficients[0])) <= 1e-8
    assert np.max(np.abs(res.coefs[1] - f2.coefficients[1])) <= 1e-8
    assert np.max(np.abs(res.params[0] - f2.intercept)) <= 1e-8
    assert np.max(np.abs(res.sigma_u - f2.sigma)) <= 1e-8
    # refusals
    with pytest.raises(ValueError, match="T"):
        sm.fit_var(np.zeros((3, 2)), 1)
    with pytest.raises(ValueError, match="lags"):
        sm.fit_var(yy, 0)
    with pytest.raises(ValueError, match="y"):
        sm.fit_var(np.zeros(10), 1)
    before = yy.copy()
    sm.fit_var(yy, 2)
    assert np.array_equal(before, yy)


def test_var_conditional_forecast_golden_and_invariants():
    fit = _diag_fit()
    y0 = np.array([[1.0, 1.0]])
    free = sm.var_conditional_forecast(fit, y0, np.array([[2.0], [3.0]]), [0])
    assert free[:, 0] == pytest.approx([2.0, 3.0], abs=1e-15) and free[:, 1] == pytest.approx([1.0, 1.2], abs=1e-12)
    # no conditioning: A^h y0 = ((0.5, 1.0), (0.25, 0.9))
    none = sm.var_conditional_forecast(fit, y0, np.zeros((2, 0)), [])
    assert none == pytest.approx(np.array([[0.5, 1.0], [0.25, 0.9]]), abs=1e-12)
    assert none[1] == pytest.approx(A @ A @ np.array([1.0, 1.0]), abs=1e-12)
    # conditioning on the second variable: the first has no feedback from it (A[0, 1] = 0)
    cond2 = sm.var_conditional_forecast(fit, y0, np.array([[0.0], [0.0]]), [1])
    assert cond2[:, 1] == pytest.approx([0.0, 0.0]) and cond2[:, 0] == pytest.approx([0.5, 0.25])
    # a VAR(2) with an intercept: the free variable follows its own equation with the conditioned path substituted
    a1, a2 = np.array([[0.3, 0.1], [0.2, 0.4]]), np.array([[0.1, 0.0], [0.05, 0.1]])
    f2 = sm.VarFit(intercept=np.array([0.1, -0.2]), coefficients=(a1, a2), sigma=np.diag([0.02, 0.03]))
    hist = np.array([[0.5, 1.0], [1.0, 0.0]])
    path = np.array([[2.0], [1.0], [0.0]])
    out = sm.var_conditional_forecast(f2, hist, path, [0])
    prev2, prev1 = hist[0], hist[1]
    for h in range(3):
        step = f2.intercept + a1 @ prev1 + a2 @ prev2
        expect = np.array([path[h, 0], step[1] + 0.0])
        # the free variable's own equation, with the conditioned variable's realised value in the lags and at lag zero
        # only
        assert out[h, 0] == path[h, 0]
        assert out[h, 1] == pytest.approx(expect[1], abs=1e-12)
        prev2, prev1 = prev1, out[h]
    # correlated shocks: the free shock is its regression on the conditioned one, E[u_f | u_c] = s_fc / s_cc u_c
    sigma = np.array([[0.04, 0.02], [0.02, 0.09]])
    fc = sm.var_conditional_forecast(_diag_fit(sigma=sigma), y0, np.array([[2.0], [3.0]]), [0])
    u1 = 2.0 - A[0] @ np.array([1.0, 1.0])
    second1 = A[1] @ np.array([1.0, 1.0]) + (0.02 / 0.04) * u1
    u2 = 3.0 - A[0] @ np.array([2.0, second1])
    second2 = A[1] @ np.array([2.0, second1]) + (0.02 / 0.04) * u2
    assert fc[:, 0] == pytest.approx([2.0, 3.0]) and fc[:, 1] == pytest.approx([second1, second2], abs=1e-12)
    with pytest.raises(ValueError, match="conditioned"):
        sm.var_conditional_forecast(fit, y0, np.array([[2.0]]), [2])
    with pytest.raises(ValueError, match="path"):
        sm.var_conditional_forecast(fit, y0, np.array([[2.0, 1.0]]), [0])
    with pytest.raises(ValueError, match="y_history"):
        sm.var_conditional_forecast(fit, np.zeros((2, 2)), np.array([[2.0]]), [0])
    with pytest.raises(ValueError, match="conditioned"):
        sm.var_conditional_forecast(fit, y0, np.array([[2.0, 2.0]]), [0, 0])
    snapshot = (y0.copy(), path.copy())
    sm.var_conditional_forecast(fit, y0, np.array([[2.0], [3.0]]), [0])
    assert np.array_equal(snapshot[0], y0)


# ---------------------------------------------------------------------------------------------------- RV-1316


def test_skewed_t_pdf_golden_and_invariants():
    assert float(sm.skewed_t_pdf(np.array([0.0]), 0.0, 1.0, 0.0, 5.0)[0]) == pytest.approx(0.379607, abs=5e-7)
    assert float(sm.skewed_t_pdf(np.array([0.0]), 0.0, 1.0, 0.0, 5.0)[0]) == pytest.approx(
        stats.t.pdf(0.0, 5), abs=1e-12
    )
    # alpha = 0 is the scaled Student t everywhere
    y = np.linspace(-6, 6, 25)
    assert sm.skewed_t_pdf(y, 0.5, 2.0, 0.0, 7.0) == pytest.approx(stats.t.pdf((y - 0.5) / 2.0, 7) / 2.0, abs=1e-12)
    # the unit integral for (0, 1, 2, 5), by adaptive quadrature of the formula's own density
    val, _ = integrate.quad(
        lambda v: float(sm.skewed_t_pdf(np.array([v]), 0.0, 1.0, 2.0, 5.0)[0]),
        -np.inf,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
    )
    assert val == pytest.approx(1.0, abs=1e-6)
    val, _ = integrate.quad(
        lambda v: float(sm.skewed_t_pdf(np.array([v]), 1.0, 2.5, -3.0, 4.0)[0]),
        -np.inf,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
    )
    assert val == pytest.approx(1.0, abs=1e-6)
    grid = np.linspace(-8, 8, 4001)
    for a in (-4.0, -1.0, 1.0, 4.0):
        d = sm.skewed_t_pdf(grid, 0.0, 1.0, a, 5.0)
        assert np.all(d >= 0.0)
        mode = grid[int(np.argmax(d))]
        assert np.sign(mode) == np.sign(a)
    # the formula by an explicit loop with scipy's t density and cdf
    w = 0.7
    expect = 2.0 * stats.t.pdf(w, 5) * stats.t.cdf(1.5 * w * math.sqrt(6.0 / (5.0 + w * w)), 6)
    assert float(sm.skewed_t_pdf(np.array([w]), 0.0, 1.0, 1.5, 5.0)[0]) == pytest.approx(expect, abs=1e-12)
    with pytest.raises(ValueError, match="sigma"):
        sm.skewed_t_pdf(np.array([0.0]), 0.0, -1.0, 0.0, 5.0)
    with pytest.raises(ValueError, match="nu"):
        sm.skewed_t_pdf(np.array([0.0]), 0.0, 1.0, 0.0, -2.0)
    with pytest.raises(ValueError, match="sigma"):
        sm.skewed_t_pdf(np.array([0.0]), 0.0, 0.0, 0.0, 5.0)
    with pytest.raises(ValueError, match="nu"):
        sm.skewed_t_pdf(np.array([0.0]), 0.0, 1.0, 0.0, 0.0)


def test_fit_skewed_t_to_quantiles_recovers_student_t():
    taus = (0.05, 0.25, 0.75, 0.95)
    q = stats.t.ppf(taus, 5)
    fit = sm.fit_skewed_t_to_quantiles(q, taus)
    assert abs(fit.alpha) <= 1e-3 and fit.distance <= 1e-8 and fit.distance >= 0.0
    # the fitted density reproduces the quantiles it was asked for: F(Q_tau) = tau by quadrature
    for tau, target in zip(taus, q, strict=False):
        cdf, _ = integrate.quad(
            lambda v: float(sm.skewed_t_pdf(np.array([v]), fit.mu, fit.sigma, fit.alpha, fit.nu)[0]),
            -np.inf,
            target,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        assert cdf == pytest.approx(tau, abs=5e-4)
    # a skewed target: quantiles of the formula's own density (found by root finding) are recovered with a tiny distance
    mu, sigma, alpha, nu = 0.5, 1.2, -2.0, 6.0

    def cdf(x):
        return integrate.quad(
            lambda v: float(sm.skewed_t_pdf(np.array([v]), mu, sigma, alpha, nu)[0]),
            -np.inf,
            x,
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]

    qs = np.array([optimize.brentq(lambda x, t=t: cdf(x) - t, -40, 40, xtol=1e-12) for t in taus])
    skew = sm.fit_skewed_t_to_quantiles(qs, taus)
    assert skew.distance <= 1e-8
    assert skew.alpha < -0.5
    with pytest.raises(ValueError, match="quantiles"):
        sm.fit_skewed_t_to_quantiles(q[::-1], taus)
    with pytest.raises(ValueError, match="taus"):
        sm.fit_skewed_t_to_quantiles(q, (0.05, 0.25, 0.75))
    with pytest.raises(ValueError, match="taus"):
        sm.fit_skewed_t_to_quantiles(q, (0.05, 0.25, 0.75, 1.0))


# ---------------------------------------------------------------------------------------------------- RV-1317


def _lp_quantile_regression(y, x, tau):
    """The quantile regression as the linear program min tau 1'u + (1 - tau) 1'v, x b + u - v = y, by HiGHS."""
    t, k = x.shape
    cost = np.concatenate([np.zeros(k), tau * np.ones(t), (1 - tau) * np.ones(t)])
    a_eq = np.hstack([x, np.eye(t), -np.eye(t)])
    res = optimize.linprog(cost, A_eq=a_eq, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * t), method="highs")
    assert res.success
    return res.x[:k]


def test_gar_quantile_regressions_golden_and_lp_cross_check():
    pytest.importorskip("statsmodels")
    xs = np.linspace(-1.0, 3.0, 25)
    x = np.column_stack([np.ones_like(xs), xs])
    exact = sm.gar_quantile_regressions(1.0 + 2.0 * xs, x)
    assert exact.shape == (4, 2) and np.max(np.abs(exact - np.array([1.0, 2.0]))) <= 1e-6
    rng = np.random.default_rng(21)
    n = 400
    nfci = rng.normal(size=n)
    growth = rng.normal(2.0, 1.0, n)
    y = (
        2.0
        + 0.3 * growth
        - (1.0 + 0.5 * np.maximum(0, 1 + nfci)) * np.abs(rng.normal(size=n)) * nfci
        + rng.normal(0, 0.5, n)
    )
    xm = np.column_stack([np.ones(n), growth, nfci])
    taus = (0.05, 0.25, 0.75, 0.95)
    coefs = sm.gar_quantile_regressions(y, xm, taus)
    assert coefs.shape == (4, 3)
    for tau, row in zip(taus, coefs, strict=False):
        lp = _lp_quantile_regression(y, xm, tau)

        # the two solve the same convex problem: compare the attained check-loss, which is flat near the optimum
        def loss(b, tau=tau):
            e = y - xm @ b
            return float(np.sum(np.where(e >= 0, tau * e, (tau - 1) * e)))

        assert loss(row) == pytest.approx(loss(lp), rel=1e-6)
        assert np.max(np.abs(row - lp)) <= 5e-3
    fitted_at_mean = coefs @ xm.mean(axis=0)
    assert np.all(np.diff(fitted_at_mean) >= -1e-9)
    default = sm.gar_quantile_regressions(y, xm)
    assert default == pytest.approx(coefs, abs=1e-12)


def test_gar_quantile_regressions_refusals(monkeypatch):
    xs = np.linspace(0, 1, 12)
    x = np.column_stack([np.ones(12), xs])
    y = 1.0 + xs
    for bad in ((0.0, 0.5), (0.5, 1.0), (-0.1, 0.5)):
        with pytest.raises(ValueError, match="tau"):
            sm.gar_quantile_regressions(y, x, bad)
    with pytest.raises(ValueError, match="x"):
        sm.gar_quantile_regressions(y, x[:5])
    # without statsmodels the function names the optional extra
    import builtins

    real_import = builtins.__import__

    def no_statsmodels(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("no statsmodels")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_statsmodels)
    with pytest.raises(ImportError, match="stress"):
        sm.gar_quantile_regressions(y, x)
