"""RV-1018 to RV-1020: the static replicating portfolio, the weighted average maturity, the deposit beta, the prepayment
logit and its maximum-likelihood fit (contract `engine-irrbb-api`, module behavioural)."""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest

from riskvalidation.irrbb import behavioural as b


def arr(*x):
    return np.array(x, dtype=float)


def margin_var(w, y, i):
    return float(np.var(y @ w - i, ddof=1))


def brute_force_min(y, i):
    """The global minimum of var(Y w - i) over the simplex, by exhaustive search over the supports (equality-constrained
    QP on each support by its KKT system, kept when nonnegative)."""
    t, j = y.shape
    cov = np.cov(y, rowvar=False).reshape(j, j)
    c = np.array([np.cov(y[:, k], i)[0, 1] for k in range(j)])
    best = (np.inf, None)
    for r in range(1, j + 1):
        for sup in itertools.combinations(range(j), r):
            s = list(sup)
            kkt = np.zeros((r + 1, r + 1))
            kkt[:r, :r] = 2 * cov[np.ix_(s, s)]
            kkt[:r, r] = 1.0
            kkt[r, :r] = 1.0
            rhs = np.r_[2 * c[s], 1.0]
            sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
            w = np.zeros(j)
            w[s] = sol[:r]
            if np.all(w >= -1e-12) and abs(w.sum() - 1) < 1e-9:
                v = margin_var(np.clip(w, 0, None), y, i)
                if v < best[0]:
                    best = (v, np.clip(w, 0, None))
    return best


# ---------------------------------------------------------------------------------------------------- RV-1018


def test_replicating_portfolio_golden_and_global_optimum():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 0.002, (60, 3)), axis=0) + 0.03
    y0 = y.copy()
    rp = b.replicating_portfolio(y[:, 1], y, arr(1, 3, 5))
    assert rp.weights.tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert rp.margin_volatility == pytest.approx(0.0, abs=1e-9)
    assert rp.maturities.tolist() == [1, 3, 5] and np.array_equal(y, y0)
    # a client rate that is a mix of two tranches plus noise: the weights solve the constrained least-variance problem
    for seed in range(8):
        r = np.random.default_rng(seed)
        j = 4
        yy = np.cumsum(r.normal(0, 0.003, (80, j)), axis=0) + 0.04
        client = 0.5 * yy[:, 0] + 0.3 * yy[:, 2] + 0.2 * 0.04 + r.normal(0, 0.0008, 80)
        out = b.replicating_portfolio(client, yy, arr(1, 2, 3, 5))
        assert np.all(out.weights >= 0) and abs(out.weights.sum() - 1) < 1e-9
        v_star, w_star = brute_force_min(yy, client)
        assert margin_var(out.weights, yy, client) == pytest.approx(v_star, rel=1e-6, abs=1e-14)
        assert out.margin_volatility == pytest.approx(math.sqrt(v_star), rel=1e-5, abs=1e-12)
        # no feasible random point does better
        for w in np.random.default_rng(seed + 100).dirichlet(np.ones(j), 200):
            assert margin_var(out.weights, yy, client) <= margin_var(w, yy, client) + 1e-12
    again = b.replicating_portfolio(client, yy, arr(1, 2, 3, 5))
    assert np.array_equal(again.weights, out.weights) and again.margin_volatility == out.margin_volatility
    assert dataclasses.is_dataclass(rp)
    with pytest.raises(ValueError):
        b.replicating_portfolio(client[:10], yy, arr(1, 2, 3, 5))
    with pytest.raises(ValueError):
        b.replicating_portfolio(client, yy, arr(1, 2, 3))
    with pytest.raises(ValueError):
        b.replicating_portfolio(client, yy, arr(1, 2, 3, -5))


def test_replicating_portfolio_few_observations():
    rng = np.random.default_rng(3)
    j = 3
    y = rng.normal(0.03, 0.002, (j + 1, j))  # fewer than J + 2 observations
    rp = b.replicating_portfolio(rng.normal(0.03, 0.002, j + 1), y, arr(1, 3, 5))
    assert rp.margin_volatility is None
    assert abs(rp.weights.sum() - 1) < 1e-9 and np.all(rp.weights >= 0)
    y2 = rng.normal(0.03, 0.002, (j + 2, j))  # exactly J + 2: a number is returned
    exact = b.replicating_portfolio(rng.normal(0.03, 0.002, j + 2), y2, arr(1, 3, 5))
    assert isinstance(exact.margin_volatility, float)


# ---------------------------------------------------------------------------------------------------- RV-1019


def test_weighted_average_maturity():
    assert b.weighted_average_maturity(arr(5.0), arr(1.0)) == 2.5
    w, m = arr(0.2, 0.3, 0.3, 0.2), arr(1, 3, 5, 10)
    assert b.weighted_average_maturity(m, w) == pytest.approx((0.2 * 1 + 0.3 * 3 + 0.3 * 5 + 0.2 * 10) / 2, abs=1e-12)
    # linear in the weights
    w1, w2 = arr(1, 0, 0, 0), arr(0, 0, 0, 1)
    mix = 0.3 * w1 + 0.7 * w2
    assert b.weighted_average_maturity(m, mix) == pytest.approx(0.3 * b.weighted_average_maturity(m, w1)
                                                              + 0.7 * b.weighted_average_maturity(m, w2), abs=1e-12)
    with pytest.raises(ValueError):
        b.weighted_average_maturity(m, arr(0.5, 0.3, 0.3, 0.2))
    with pytest.raises(ValueError):
        b.weighted_average_maturity(m, arr(0.5, 0.5))


def test_deposit_beta_passthrough():
    assert b.deposit_beta_passthrough(0.01, 0.02, 0.4, 0.3) == pytest.approx(0.018, abs=1e-15)
    assert b.deposit_beta_passthrough(0.01, -0.02, 0.4, 0.3) == pytest.approx(0.004, abs=1e-15)
    assert b.deposit_beta_passthrough(0.01, 0.0, 0.4, 0.3) == 0.01
    # the result is not floored: a large fall leaves a negative rate for the caller to floor
    assert b.deposit_beta_passthrough(0.01, -0.10, 0.4, 0.9) < 0
    with pytest.raises(ValueError):
        b.deposit_beta_passthrough(0.01, 0.02, -0.1, 0.3)
    with pytest.raises(ValueError):
        b.deposit_beta_passthrough(0.01, 0.02, 0.4, -0.3)


def test_prepayment_logit():
    x, a = arr(-0.02, 0.0, 0.03), arr(1, 12, 360)
    assert np.all(b.prepayment_logit(x, a, arr(0, 0, 0)) == 0.5)
    coef = arr(-2.0, 20.0, 0.01)
    got = b.prepayment_logit(x, a, coef)
    ref = [1 / (1 + math.exp(-(-2.0 + 20.0 * xi + 0.01 * ai))) for xi, ai in zip(x, a, strict=True)]
    assert got.tolist() == pytest.approx(ref, abs=1e-15)
    grid = np.linspace(-0.1, 0.1, 81)
    p = b.prepayment_logit(grid, np.full(81, 24.0), coef)
    assert np.all(np.diff(p) > 0) and np.all((p > 0) & (p < 1))
    # an extreme index saturates without overflow
    ext = b.prepayment_logit(arr(10.0, -10.0), arr(1, 1), arr(0.0, 1000.0, 0.0))
    assert ext.tolist() == [1.0, 0.0] or (ext[0] > 0.999 and ext[1] < 0.001)
    with pytest.raises(ValueError):
        b.prepayment_logit(x, a[:2], coef)
    with pytest.raises(ValueError):
        b.prepayment_logit(x, a, arr(0, 0))


# ---------------------------------------------------------------------------------------------------- RV-1020


def simulate_design(n, seed, coef, inc_halfwidth):
    """The stated design: incentive ~ U(-h, h), seasoning ~ U(0, 360) months, prepaid ~ Bernoulli(logit)."""
    r = np.random.default_rng(seed)
    inc = r.uniform(-inc_halfwidth, inc_halfwidth, n)
    sea = r.uniform(0, 360, n)
    p = b.prepayment_logit(inc, sea, np.asarray(coef, dtype=float))
    return inc, sea, r.random(n) < p


def standard_errors(inc, sea, coef):
    x = np.column_stack([np.ones_like(inc), inc, sea])
    p = 1 / (1 + np.exp(-(x @ coef)))
    info = x.T @ (x * (p * (1 - p))[:, None])
    return np.sqrt(np.diag(np.linalg.inv(info)))


def test_fit_prepayment_logit_recovers_stated_design():
    # the design of the contract golden: N = 20000, incentive ~ U(-0.10, 0.10), seasoning ~ U(0, 360) months,
    # independent, prepaid ~ Bernoulli(logit) with coef = [-2, 20, 0.01], seed 0
    true = np.array([-2.0, 20.0, 0.01])
    inc, sea, y = simulate_design(20000, 0, true, 0.10)
    i0, s0, y0 = inc.copy(), sea.copy(), y.copy()
    fit = b.fit_prepayment_logit(inc, sea, y)
    assert fit.converged and fit.n == 20000
    se = standard_errors(inc, sea, fit.coef)
    assert np.allclose(fit.std_err, se, rtol=1e-6)  # the standard errors are those of the inverse observed information
    assert np.all(np.abs(fit.coef - true) <= 3 * fit.std_err), (fit.coef, true, fit.std_err)
    assert fit.std_err[1] == pytest.approx(0.35, abs=0.02)  # about 0.35 for the incentive coefficient under this design
    assert 3 * fit.std_err[1] < 1.2  # the tolerance bites: about 1.0 for the incentive coefficient
    # the score equations X'(y - p) = 0 hold at the solution
    x = np.column_stack([np.ones_like(inc), inc, sea])
    score = x.T @ (y - 1 / (1 + np.exp(-(x @ fit.coef))))
    assert np.max(np.abs(score / np.sqrt(len(y)))) < 1e-5
    # deterministic, inputs not mutated
    again = b.fit_prepayment_logit(inc, sea, y)
    assert np.array_equal(again.coef, fit.coef)
    assert np.array_equal(inc, i0) and np.array_equal(sea, s0) and np.array_equal(y, y0)
    # a different seed gives a different but also consistent fit
    inc2, sea2, y2 = simulate_design(20000, 5, true, 0.10)
    fit2 = b.fit_prepayment_logit(inc2, sea2, y2)
    assert fit2.converged and not np.array_equal(fit2.coef, fit.coef)
    assert np.all(np.abs(fit2.coef - true) <= 4 * fit2.std_err)
    # boolean and 0/1 inputs agree
    assert np.array_equal(b.fit_prepayment_logit(inc, sea, y.astype(int)).coef, fit.coef)
    # maximum likelihood: no nearby coefficient has a higher log-likelihood
    def ll(c):
        z = x @ c
        return float(np.sum(y * z - np.logaddexp(0, z)))
    for d in np.eye(3) * np.array([1e-3, 1e-2, 1e-5]):
        assert ll(fit.coef) >= ll(fit.coef + d) and ll(fit.coef) >= ll(fit.coef - d)


def test_fit_prepayment_logit_separation_is_not_converged(monkeypatch):
    inc = np.linspace(-0.1, 0.1, 200)
    sea = np.random.default_rng(3).uniform(0, 360, 200)  # a regressor with spread: the separation guard is reached
    sep = inc > 0  # complete separation by the incentive
    fit = b.fit_prepayment_logit(inc, sea, sep)
    assert fit.converged is False and fit.n == 200 and np.all(np.isnan(fit.coef)) and np.all(np.isnan(fit.std_err))
    # quasi-separation: a single overlapping pair does not make the likelihood finite enough to be called converged
    assert b.fit_prepayment_logit(inc, sea, inc > 0.0005).converged is False
    # a constant regressor with a mixed outcome (the design is singular) is its own guard
    const_sea = np.full(200, 24.0)
    assert b.fit_prepayment_logit(inc, const_sea, np.random.default_rng(1).random(200) < 0.4).converged is False
    assert b.fit_prepayment_logit(inc, const_sea, sep).converged is False
    # the separation guard stops the iteration early (a handful of Newton steps), not after the whole budget of 100
    calls = []
    real_solve = np.linalg.solve
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **k: calls.append(1) or real_solve(*a, **k))
    for outcome in (sep, inc > 0.0005):
        calls.clear()
        assert b.fit_prepayment_logit(inc, sea, outcome).converged is False
        assert 0 < len(calls) < 30, len(calls)
    monkeypatch.undo()
    # all one class: no information, not converged, no exception
    assert b.fit_prepayment_logit(inc, np.linspace(1, 100, 200), np.zeros(200, dtype=bool)).converged is False
    assert b.fit_prepayment_logit(inc, np.linspace(1, 100, 200), np.ones(200, dtype=bool)).converged is False
    # a constant regressor makes the design singular: not converged rather than an exception
    r = np.random.default_rng(1)
    noisy = r.random(200) < 0.3
    assert b.fit_prepayment_logit(np.full(200, 0.01), sea, noisy).converged is False
    with pytest.raises(ValueError):
        b.fit_prepayment_logit(inc, sea[:10], sep)
    with pytest.raises(ValueError):
        b.fit_prepayment_logit(inc, sea, np.full(200, 2))
