"""RV-1911 to RV-1914: the Minnesota BVAR by dummy observations (Banbura, Giannone and Reichlin 2008; Karlsson 2012),
in `stress.scenario_models` (contract `engine-stress-api`)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from riskvalidation.stress import scenario_models as sm


def _walk(t=80, n=2, seed=20261115, drift=0.05):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(t, n)) + drift, axis=0)


def _own_dummies(sigma, delta, lam, p, eps, tau=None, mu=None):
    """Banbura et al. (5) and (9) written out in the test, entry by entry (a second route to `bvar_dummies`)."""
    n = len(sigma)
    k = n * p + 1
    rows_y, rows_x = [], []
    for lag in range(1, p + 1):                          # first block: the prior on the autoregressive coefficients
        for i in range(n):
            ry = np.zeros(n)
            if lag == 1:
                ry[i] = delta[i] * sigma[i] / lam
            rx = np.zeros(k)
            rx[(lag - 1) * n + i] = lag * sigma[i] / lam
            rows_y.append(ry)
            rows_x.append(rx)
    for i in range(n):                                   # second block: the prior on the covariance
        ry = np.zeros(n)
        ry[i] = sigma[i]
        rows_y.append(ry)
        rows_x.append(np.zeros(k))
    rows_y.append(np.zeros(n))                           # third block: the intercept
    rx = np.zeros(k)
    rx[-1] = eps
    rows_x.append(rx)
    if tau is not None:                                  # sum of coefficients (9): a row of ones over the lags
        for i in range(n):
            ry = np.zeros(n)
            ry[i] = delta[i] * mu[i] / tau
            rx = np.zeros(k)
            for lag in range(1, p + 1):
                rx[(lag - 1) * n + i] = delta[i] * mu[i] / tau
            rows_y.append(ry)
            rows_x.append(rx)
    return np.array(rows_y), np.array(rows_x)


def _regressors(y, p):
    t = y.shape[0]
    x = np.column_stack([y[p - k:t - k] for k in range(1, p + 1)] + [np.ones(t - p)])
    return x, y[p:]


def test_fit_returns_a_varfit_the_forecast_consumes():
    y = _walk()
    fit = sm.fit_bvar_minnesota(y, lags=2, lam=0.2)
    assert isinstance(fit, sm.VarFit)
    assert fit.intercept.shape == (2,) and len(fit.coefficients) == 2
    assert all(a.shape == (2, 2) for a in fit.coefficients)
    assert fit.sigma.shape == (2, 2) and np.allclose(fit.sigma, fit.sigma.T)
    assert np.all(np.linalg.eigvalsh(fit.sigma) > 0.0)
    # a random-walk prior at lam = 0.2 keeps the first own lag near 1 and the others near 0
    assert abs(fit.coefficients[0][0, 0] - 1.0) < 0.3 and abs(fit.coefficients[1][0, 0]) < 0.3
    # the conditional forecast consumes it unchanged
    path = np.array([[y[-1, 0] + 1.0], [y[-1, 0] + 2.0], [y[-1, 0] + 3.0]])
    out = sm.var_conditional_forecast(fit, y[-2:], path, [0])
    assert out.shape == (3, 2) and np.all(np.isfinite(out))
    assert np.allclose(out[:, 0], path[:, 0])
    # delta: ones by default, a per-variable override, the intercept epsilon default 1e-6
    p0 = sm.bvar_posterior(y, 2, 1e-3)
    p1 = sm.bvar_posterior(y, 2, 1e-3, delta=[1.0, 0.0])
    assert p0.b_mean[0, 0] == pytest.approx(1.0, abs=0.02) and p0.b_mean[1, 1] == pytest.approx(1.0, abs=0.02)
    assert p1.b_mean[1, 1] == pytest.approx(0.0, abs=0.02)
    assert sm.bvar_posterior(y, 2, 0.2).eps == 1e-6
    # the scales sigma_i are the residual standard deviations of a univariate AR(p) per variable unless given
    for i in range(2):
        x, tgt = _regressors(y[:, [i]], 2)
        beta = np.linalg.lstsq(x, tgt, rcond=None)[0]
        resid = tgt - x @ beta
        expected = np.sqrt(resid.T @ resid / (tgt.shape[0] - 3)).item()
        assert sm.bvar_posterior(y, 2, 0.2).sigma_scale[i] == pytest.approx(expected, rel=1e-10)
    given = sm.bvar_posterior(y, 2, 0.2, sigma_scale=[2.0, 3.0])
    assert tuple(given.sigma_scale) == (2.0, 3.0)
    # the sum-of-coefficients dummies add n rows
    assert sm.bvar_posterior(y, 2, 0.2, sum_coef_tau=2.0).n_dummy == sm.bvar_posterior(y, 2, 0.2).n_dummy + 2
    # the input is not mutated
    before = y.copy()
    sm.fit_bvar_minnesota(y, 2, 0.2, sum_coef_tau=2.0)
    assert np.array_equal(before, y)


def test_dummies_are_banbura_five_and_nine():
    n, p, lam, eps = 3, 2, 0.3, 1e-6
    sigma = np.array([1.5, 0.7, 2.2])
    delta = np.array([1.0, 0.5, 0.0])
    mu = np.array([4.0, -2.0, 1.0])
    yd, xd = sm.bvar_dummies(sigma, delta, lam, p, eps)
    ry, rx = _own_dummies(sigma, delta, lam, p, eps)
    assert yd.shape == (n * p + n + 1, n) and xd.shape == (n * p + n + 1, n * p + 1)
    assert np.allclose(yd, ry, atol=0, rtol=1e-15) and np.allclose(xd, rx, atol=0, rtol=1e-15)
    yd2, xd2 = sm.bvar_dummies(sigma, delta, lam, p, eps, sum_coef_tau=3.0, mu=mu)
    ry2, rx2 = _own_dummies(sigma, delta, lam, p, eps, tau=3.0, mu=mu)
    assert np.allclose(yd2, ry2, rtol=1e-15) and np.allclose(xd2, rx2, rtol=1e-15)
    assert yd2.shape[0] == n * p + 2 * n + 1


def test_inverse_wishart_convention_and_draws():
    y = _walk(t=60, n=3, seed=3)
    post = sm.bvar_posterior(y, 2, 0.2, sum_coef_tau=2.0)
    n, k = 3, 3 * 2 + 1
    t_rows, t_dummy = y.shape[0] - 2, post.n_dummy
    assert t_dummy == n * 2 + n + 1 + n
    # iW(Sigma_tilde, T + T_d + 2 - k): Banbura et al. (7); Karlsson's proper-prior form has 2 fewer
    assert post.dof == t_rows + t_dummy + 2 - k
    fit = post.var_fit()
    # sigma is its mean under the convention |Psi|^-(v + n + 1)/2 exp(-tr(S Psi^-1) / 2) of SciPy's invwishart
    assert fit.sigma == pytest.approx(post.scale / (post.dof - n - 1), rel=1e-13)
    assert fit.sigma == pytest.approx(stats.invwishart(df=post.dof, scale=post.scale).mean(), rel=1e-12)
    # the scale is the residual sum of squares of the augmented regression
    yd, xd = sm.bvar_dummies(post.sigma_scale, post.delta, post.lam, 2, post.eps, post.sum_coef_tau, post.mu)
    x, tgt = _regressors(y, 2)
    ys, xs = np.vstack([tgt, yd]), np.vstack([x, xd])
    b = np.linalg.solve(xs.T @ xs, xs.T @ ys)
    assert post.b_mean == pytest.approx(b, rel=1e-10, abs=1e-12)
    assert post.scale == pytest.approx((ys - xs @ b).T @ (ys - xs @ b), rel=1e-10)
    # the posterior draws: the mean of Psi is within Monte Carlo error of that mean, the mean of B of the posterior mean
    draws = sm.bvar_posterior_draws(post, 20_000, seed=9)
    assert draws.sigma.shape == (20_000, n, n) and draws.coefficients.shape == (20_000, k, n)
    mean_psi = draws.sigma.mean(axis=0)
    se = draws.sigma.std(axis=0) / np.sqrt(draws.sigma.shape[0])
    assert np.all(np.abs(mean_psi - fit.sigma) < 5.0 * se)
    se_b = draws.coefficients.std(axis=0) / np.sqrt(draws.coefficients.shape[0])
    assert np.all(np.abs(draws.coefficients.mean(axis=0) - post.b_mean) < 5.0 * se_b)
    # reproducible for a seed, and a draw converts to a VarFit the forecast takes
    again = sm.bvar_posterior_draws(post, 20_000, seed=9)
    assert np.array_equal(draws.sigma, again.sigma) and np.array_equal(draws.coefficients, again.coefficients)
    one = draws.var_fit(0)
    assert isinstance(one, sm.VarFit) and len(one.coefficients) == 2
    with pytest.raises(ValueError, match="n_draws"):
        sm.bvar_posterior_draws(post, 0, seed=0)


def test_identities_and_limits():
    y = _walk(t=70, n=2, seed=5)
    n, p, lam, eps = 2, 2, 0.25, 1e-6
    sigma = np.array([0.9, 1.3])
    delta = np.array([1.0, 1.0])
    post = sm.bvar_posterior(y, p, lam, sigma_scale=sigma, eps=eps)
    x, tgt = _regressors(y, p)
    yd, xd = _own_dummies(sigma, delta, lam, p, eps)
    # Karlsson (18) and (19): the normal-Wishart posterior of the prior B0 = (Xd'Xd)^-1 Xd'Yd, Omega0 = (Xd'Xd)^-1
    omega0_inv = xd.T @ xd
    b0 = np.linalg.solve(omega0_inv, xd.T @ yd)
    omega_bar_inv = omega0_inv + x.T @ x
    b_bar = np.linalg.solve(omega_bar_inv, omega0_inv @ b0 + x.T @ tgt)
    assert np.max(np.abs(post.b_mean - b_bar)) < 1e-12
    s0 = (yd - xd @ b0).T @ (yd - xd @ b0)
    b_hat = np.linalg.solve(x.T @ x, x.T @ tgt)
    s_res = (tgt - x @ b_hat).T @ (tgt - x @ b_hat)
    gap = b0 - b_hat
    s_bar = s0 + s_res + gap.T @ np.linalg.inv(np.linalg.inv(omega0_inv) + np.linalg.inv(x.T @ x)) @ gap
    assert np.max(np.abs(post.scale - s_bar)) < 1e-10
    # a very large lambda reproduces the OLS coefficients of fit_var
    ols = sm.fit_var(y, p)
    big = sm.fit_bvar_minnesota(y, p, 1e8)
    for a, b in zip(big.coefficients, ols.coefficients, strict=True):
        assert np.max(np.abs(a - b)) < 1e-6
    assert np.max(np.abs(big.intercept - ols.intercept)) < 1e-6
    # lambda tending to 0 returns the prior mean: A1 = diag(delta), the other lags 0
    for dl in ([1.0, 1.0], [1.0, 0.0]):
        tiny = sm.fit_bvar_minnesota(y, p, 1e-8, delta=dl)
        assert np.max(np.abs(tiny.coefficients[0] - np.diag(dl))) < 1e-6
        assert np.max(np.abs(tiny.coefficients[1])) < 1e-6
    # a univariate AR(1): the scalar ridge-to-random-walk closed form (the intercept is nearly free, so demeaned)
    z = _walk(t=60, n=1, seed=8)[:, 0]
    s1 = 0.8
    lam1 = 0.2
    one = sm.bvar_posterior(z[:, None], 1, lam1, sigma_scale=[s1])
    xs, ys = z[:-1], z[1:]
    xc, yc = xs - xs.mean(), ys - ys.mean()
    ridge = s1**2 / lam1**2
    closed = (xc @ yc + ridge) / (xc @ xc + ridge)
    assert one.b_mean[0, 0] == pytest.approx(closed, abs=1e-8)
    # the sum-of-coefficients dummies at tau tending to 0: the lag matrices sum to the identity
    soc = sm.fit_bvar_minnesota(y, p, 1.0, sum_coef_tau=1e-7)
    assert np.max(np.abs(soc.coefficients[0] + soc.coefficients[1] - np.eye(n))) < 1e-5
    loose = sm.fit_bvar_minnesota(y, p, 1.0)
    assert np.max(np.abs(loose.coefficients[0] + loose.coefficients[1] - np.eye(n))) > 1e-3
    # shrinkage moves the first own lag monotonically from the prior mean to OLS on a stationary series
    rng = np.random.default_rng(1)
    st = np.zeros((200, 1))
    for t in range(1, 200):
        st[t] = 0.5 * st[t - 1] + rng.normal()
    seq = [sm.fit_bvar_minnesota(st, 1, lm).coefficients[0][0, 0] for lm in (0.01, 0.05, 0.2, 1.0, 10.0, 1e4)]
    assert all(a > b for a, b in zip(seq[:-1], seq[1:], strict=True)) and seq[0] > 0.9
    assert abs(seq[-1] - sm.fit_var(st, 1).coefficients[0][0, 0]) < 1e-3


def test_refusals():
    y = _walk(t=30, n=2)
    with pytest.raises(ValueError, match="y"):
        sm.fit_bvar_minnesota(y[:3], lags=2, lam=0.2)          # fewer than lags + 2 observations
    sm.fit_bvar_minnesota(y[:4], lags=2, lam=0.2, sigma_scale=[1.0, 1.0])  # lags + 2 observations are enough
    with pytest.raises(ValueError, match="sigma_scale"):                  # but no AR(p) scale can be estimated
        sm.fit_bvar_minnesota(y[:4], lags=2, lam=0.2)
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="lam"):
            sm.fit_bvar_minnesota(y, 2, bad)
    bad_y = y.copy()
    bad_y[3, 1] = np.nan
    with pytest.raises(ValueError, match="y"):
        sm.fit_bvar_minnesota(bad_y, 2, 0.2)
    with pytest.raises(ValueError, match="y"):
        sm.fit_bvar_minnesota(y[:, 0], 2, 0.2)
    with pytest.raises(ValueError, match="delta"):
        sm.fit_bvar_minnesota(y, 2, 0.2, delta=[1.0])
    with pytest.raises(ValueError, match="delta"):
        sm.fit_bvar_minnesota(y, 2, 0.2, delta=[1.0, np.nan])
    with pytest.raises(ValueError, match="lags"):
        sm.fit_bvar_minnesota(y, 0, 0.2)
    with pytest.raises(ValueError, match="sum_coef_tau"):
        sm.fit_bvar_minnesota(y, 2, 0.2, sum_coef_tau=0.0)
    with pytest.raises(ValueError, match="eps"):
        sm.fit_bvar_minnesota(y, 2, 0.2, eps=0.0)
    with pytest.raises(ValueError, match="sigma_scale"):
        sm.fit_bvar_minnesota(y, 2, 0.2, sigma_scale=[1.0, -1.0])
    with pytest.raises(ValueError, match="sigma_scale"):
        sm.fit_bvar_minnesota(y, 2, 0.2, sigma_scale=[1.0])
