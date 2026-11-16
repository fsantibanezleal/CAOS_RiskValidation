"""RV-1901 to RV-1905: Clayton Kendall tau and its inverse, the Marshall-Olkin and conditional-inverse samplers, the
h-function and its inverse and `clayton_aggregate` (contract `engine-aggregation-api`, module aggregation.copulas)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from riskvalidation.aggregation import bounds
from riskvalidation.aggregation import copulas as c

N_TAU = 400_000


def _clayton_cdf(u, v, theta):
    """The textbook closed form, written here for the test (a second route to `clayton_copula`)."""
    return (u ** -theta + v ** -theta - 1.0) ** (-1.0 / theta)


def test_kendall_tau_and_inverse():
    assert c.clayton_kendall_tau(1.0) == pytest.approx(1.0 / 3.0, abs=1e-15)
    assert c.clayton_kendall_tau(2.0) == pytest.approx(0.5, abs=1e-15)
    assert c.clayton_kendall_tau(4.0) == pytest.approx(2.0 / 3.0, abs=1e-15)
    for tau in (0.05, 0.3, 0.5, 0.9):
        assert c.clayton_kendall_tau(c.clayton_theta_from_tau(tau)) == pytest.approx(tau, abs=1e-14)
    assert c.clayton_theta_from_tau(0.5) == pytest.approx(2.0, abs=1e-14)
    # a numerical second route: tau = 1 + 4 int_0^1 phi / phi' with phi(t) = (t^-theta - 1) / theta (Genest and MacKay)
    for theta in (0.7, 1.0, 2.0, 4.0):
        val = 1.0 + 4.0 * integrate.quad(lambda t, th=theta: (t ** -th - 1.0) / th / (-t ** (-th - 1.0)), 0.0, 1.0)[0]
        assert c.clayton_kendall_tau(theta) == pytest.approx(val, abs=1e-9)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="theta"):
            c.clayton_kendall_tau(bad)
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="tau"):
            c.clayton_theta_from_tau(bad)


def test_simulate_shape_seed_survival_and_refusals():
    u = c.simulate_clayton_uniforms(2.0, 3, 1000, seed=7)
    assert u.shape == (1000, 3)
    assert np.all(u > c.UNIFORM_FLOOR) and np.all(u < 1.0 - c.UNIFORM_FLOOR)
    assert np.array_equal(u, c.simulate_clayton_uniforms(2.0, 3, 1000, seed=7))
    assert not np.array_equal(u, c.simulate_clayton_uniforms(2.0, 3, 1000, seed=8))
    # survival is the default and equals 1 - U of the lower-tail copula
    lower = c.simulate_clayton_uniforms(2.0, 3, 1000, seed=7, survival=False)
    assert np.allclose(u, 1.0 - lower, atol=1e-15)
    # the dependence of the default sits in the upper tail
    big = c.simulate_clayton_uniforms(2.0, 2, 200_000, seed=3)
    both_up = np.mean((big[:, 0] > 0.99) & (big[:, 1] > 0.99))
    both_down = np.mean((big[:, 0] < 0.01) & (big[:, 1] < 0.01))
    assert both_up > 20 * 1e-4
    assert both_down < both_up / 10
    low = c.simulate_clayton_uniforms(2.0, 2, 200_000, seed=3, survival=False)
    assert np.mean((low[:, 0] < 0.01) & (low[:, 1] < 0.01)) > 20 * 1e-4
    # the conditional method at d = 2 is reproducible and inside the open interval too
    cond = c.simulate_clayton_uniforms(1.0, 2, 500, seed=1, method="conditional")
    assert cond.shape == (500, 2) and np.all(cond > 0.0) and np.all(cond < 1.0)
    assert np.array_equal(cond, c.simulate_clayton_uniforms(1.0, 2, 500, seed=1, method="conditional"))
    # a huge theta and a tiny theta stay inside (0, 1) with no NaN
    for theta in (200.0, 0.01):
        z = c.simulate_clayton_uniforms(theta, 4, 5000, seed=2)
        assert np.all(np.isfinite(z)) and np.all(z > 0.0) and np.all(z < 1.0)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="theta"):
            c.simulate_clayton_uniforms(bad, 2, 10, seed=0)
    with pytest.raises(ValueError, match="d "):
        c.simulate_clayton_uniforms(1.0, 1, 10, seed=0)
    with pytest.raises(ValueError, match="n "):
        c.simulate_clayton_uniforms(1.0, 2, 0, seed=0)
    with pytest.raises(ValueError, match="conditional"):
        c.simulate_clayton_uniforms(1.0, 3, 10, seed=0, method="conditional")
    with pytest.raises(ValueError, match="method"):
        c.simulate_clayton_uniforms(1.0, 2, 10, seed=0, method="nope")


def test_chunking_does_not_change_the_losses():
    ppfs = [lambda x: np.asarray(x, dtype=float)] * 3
    a = c.simulate_clayton_losses(ppfs, 2.0, 50_000, seed=5, chunk_size=1_000)
    b = c.simulate_clayton_losses(ppfs, 2.0, 50_000, seed=5, chunk_size=33_333)
    assert np.array_equal(a, b)
    assert a.shape == (50_000, 3)
    assert np.array_equal(a, c.simulate_clayton_uniforms(2.0, 3, 50_000, seed=5))


@pytest.mark.parametrize("theta", [1.0, 2.0, 4.0])
def test_draws_match_tau_and_the_copula(theta):
    u = c.simulate_clayton_uniforms(theta, 2, N_TAU, seed=20261115, survival=False)
    tau = stats.kendalltau(u[:, 0], u[:, 1]).statistic
    assert tau == pytest.approx(theta / (theta + 2.0), abs=0.003)
    emp = np.mean((u[:, 0] <= 0.5) & (u[:, 1] <= 0.5))
    assert emp == pytest.approx((2.0 * 0.5 ** -theta - 1.0) ** (-1.0 / theta), abs=0.003)
    assert emp == pytest.approx(c.clayton_copula(0.5, 0.5, theta), abs=0.003)
    # the survival copula: the same statistics for the joint upper orthant
    s = c.simulate_clayton_uniforms(theta, 2, N_TAU, seed=20261115)
    assert stats.kendalltau(s[:, 0], s[:, 1]).statistic == pytest.approx(theta / (theta + 2.0), abs=0.003)
    assert np.mean((s[:, 0] > 0.5) & (s[:, 1] > 0.5)) == pytest.approx(_clayton_cdf(0.5, 0.5, theta), abs=0.003)
    for j in range(2):
        assert stats.kstest(u[:, j], "uniform").pvalue > 1e-4


def test_trivariate_value():
    u = c.simulate_clayton_uniforms(2.0, 3, N_TAU, seed=11, survival=False)
    emp = np.mean(np.all(u <= 0.5, axis=1))
    assert emp == pytest.approx((3 * 0.5 ** -2.0 - 2.0) ** -0.5, abs=0.003)
    assert emp == pytest.approx(0.31623, abs=0.003)


@pytest.mark.parametrize("theta", [1.0, 4.0])
def test_two_algorithms_agree_at_d2(theta):
    a = c.simulate_clayton_uniforms(theta, 2, 100_000, seed=1, survival=False)
    b = c.simulate_clayton_uniforms(theta, 2, 100_000, seed=2, survival=False, method="conditional")
    ca = _clayton_cdf(a[:, 0], a[:, 1], theta)
    cb = _clayton_cdf(b[:, 0], b[:, 1], theta)
    assert stats.ks_2samp(ca, cb).pvalue > 0.01
    assert stats.kendalltau(b[:, 0], b[:, 1]).statistic == pytest.approx(theta / (theta + 2.0), abs=0.01)


def test_h_function_inverse():
    grid = np.linspace(0.01, 0.99, 25)
    u1, u2 = np.meshgrid(grid, grid)
    eps = np.finfo(float).eps
    for theta in (0.5, 1.0, 2.0, 4.0):
        w = c.clayton_h_function(u1, u2, theta)
        assert np.all((w > 0.0) & (w < 1.0))
        # the forward direction: h(h^-1(w | u2) | u2) = w to 1e-12 on the whole grid, for every theta
        back = c.clayton_h_function(c.clayton_h_inverse(w, u2, theta), u2, theta)
        assert np.max(np.abs(back - w)) < 1e-12
        # the other direction: h^-1(h(u1 | u2) | u2) = u1. dh/du1 is the copula density c(u1, u2), so the rounding of w
        # (a few ulp in h and in h^-1) moves u1 by about 64 eps w / c: that propagation is added to the 1e-12
        # (the density is as small as 5e-8 at theta 4 in the corner (0.99, 0.01), where no double-precision inverse
        # reaches 1e-12; the unit's report records it against RV-1904)
        dens = (1.0 + theta) * (u1 * u2) ** (-1.0 - theta) * (u1 ** -theta + u2 ** -theta - 1.0) ** (-1.0 / theta - 2.0)
        err = np.abs(c.clayton_h_inverse(w, u2, theta) - u1)
        assert np.all(err <= 1e-12 + 64.0 * eps * w / dens)
        if theta <= 1.0:
            assert err.max() < 1e-12
        # h(u1 | u2) is dC/du2: a central finite difference of the closed form is the second route
        step = 1e-6
        fd = (_clayton_cdf(u1, u2 + step, theta) - _clayton_cdf(u1, u2 - step, theta)) / (2 * step)
        assert np.max(np.abs(fd - w)) < 1e-6
    with pytest.raises(ValueError, match="theta"):
        c.clayton_h_function(0.5, 0.5, 0.0)


def _ident(u):
    return np.asarray(u, dtype=float)


def _pareto(u):
    return (1.0 - np.asarray(u, dtype=float)) ** -0.5 - 1.0


def test_clayton_aggregate_limits_and_seed():
    ppfs = [_pareto, _pareto, _pareto]
    kw = dict(el=0.0, level=0.99, n_draws=400_000, seed=4)
    hi = c.clayton_aggregate(ppfs, 200.0, **kw)
    como = bounds.comonotonic_var(ppfs, 0.99)
    assert hi.var == pytest.approx(como, rel=0.02)
    lo = c.clayton_aggregate(ppfs, 0.01, **kw)
    ref = c.copula_aggregate(ppfs, np.eye(3), "gaussian", None, 0.0, 0.99, 400_000, 9)
    assert lo.var == pytest.approx(ref.var, rel=0.04)
    assert lo.es == pytest.approx(ref.es, rel=0.06)
    a = c.clayton_aggregate(ppfs, 2.0, **kw)
    b = c.clayton_aggregate(ppfs, 2.0, **kw)
    assert a == b
    assert a.theta == 2.0 and a.survival is True and a.n_draws == 400_000
    assert a.es >= a.var and a.ec == pytest.approx(a.var - a.el)
    assert a.var_se is not None and a.es_se is not None
    assert lo.var < a.var < hi.var
    with pytest.raises(ValueError, match="theta"):
        c.clayton_aggregate(ppfs, 0.0, el=0.0)
    with pytest.raises(ValueError, match="el"):
        c.clayton_aggregate(ppfs, 1.0, el=float("nan"))
    with pytest.raises(ValueError, match="ppfs"):
        c.clayton_aggregate([_ident], 1.0, el=0.0)


def test_reconciles_with_statsmodels():
    cop = pytest.importorskip("statsmodels.distributions.copula.api")
    for theta in (1.0, 2.0, 4.0):
        sm = cop.ClaytonCopula(theta=theta)
        assert float(sm.tau()) == pytest.approx(c.clayton_kendall_tau(theta), abs=1e-12)
        assert float(sm.theta_from_tau(c.clayton_kendall_tau(theta))) == pytest.approx(theta, abs=1e-10)
        assert float(np.squeeze(sm.cdf([0.3, 0.6]))) == pytest.approx(c.clayton_copula(0.3, 0.6, theta), abs=1e-12)
        ours = c.simulate_clayton_uniforms(theta, 2, 100_000, seed=0, survival=False)
        theirs = sm.rvs(100_000, rng=np.random.default_rng(0))
        t_o = stats.kendalltau(ours[:, 0], ours[:, 1]).statistic
        t_t = stats.kendalltau(theirs[:, 0], theirs[:, 1]).statistic
        assert math.isfinite(t_t)
        assert t_o == pytest.approx(t_t, abs=0.01)
