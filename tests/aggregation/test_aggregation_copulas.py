"""RV-1307 to RV-1312: the marginal quantile functions, the Gaussian and Student t copula draws, the tail dependence
formulas, Clayton and the copula aggregate (contract `engine-aggregation-api`, module aggregation.copulas)."""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest
from scipy import integrate, stats

from riskvalidation.aggregation import copulas as c

R5 = np.array([[1.0, 0.5], [0.5, 1.0]])


def arr(*x):
    return np.array(x, dtype=float)


def test_marginal_ppf_sample_golden_and_invariants():
    ppf = c.marginal_ppf_sample(arr(0, 10))
    assert ppf(arr(0.5)) == pytest.approx([5.0], abs=1e-12)
    assert ppf(arr(0.1)) == pytest.approx([0.0], abs=1e-12)
    assert ppf(arr(0.9)) == pytest.approx([10.0], abs=1e-12)
    assert ppf(arr(0.0, 1.0)) == pytest.approx([0.0, 10.0])  # clamped at the ends
    rng = np.random.default_rng(0)
    x = np.sort(rng.lognormal(size=57))
    f = c.marginal_ppf_sample(x)
    n = x.size
    nodes = (np.arange(1, n + 1) - 0.5) / n
    assert f(nodes) == pytest.approx(x, rel=1e-12)  # exact at the Hazen positions
    mid = 0.5 * (nodes[:-1] + nodes[1:])
    assert f(mid) == pytest.approx(0.5 * (x[:-1] + x[1:]), rel=1e-12)  # linear between them
    u = np.sort(rng.uniform(size=500))
    out = f(u)
    assert np.all(np.diff(out) >= 0.0)
    assert out.min() >= x[0] and out.max() <= x[-1]
    with pytest.raises(ValueError, match="sorted_losses"):
        c.marginal_ppf_sample(arr(2, 1, 3))
    with pytest.raises(ValueError, match="sorted_losses"):
        c.marginal_ppf_sample(arr(1.0))
    with pytest.raises(ValueError, match="sorted_losses"):
        c.marginal_ppf_sample(arr(1.0, float("nan")))


def test_marginal_ppf_pmf_golden_and_invariants():
    f = c.marginal_ppf_pmf(arr(0, 1, 2), arr(0.5, 0.3, 0.2))
    assert f(arr(0.5, 0.51, 0.8, 0.81)) == pytest.approx([0.0, 1.0, 1.0, 2.0])
    rng = np.random.default_rng(2)
    grid = np.sort(rng.uniform(0, 50, 40))
    pmf = rng.uniform(size=40)
    pmf /= pmf.sum()
    g = c.marginal_ppf_pmf(grid, pmf)
    u = rng.uniform(1e-6, 1 - 1e-6, 300)
    cum = np.cumsum(pmf)
    brute = np.array([grid[np.argmax(cum >= ui - 1e-12)] for ui in u])  # a plain search, the rule of the contract
    assert g(u) == pytest.approx(brute)
    assert np.all(np.diff(g(np.sort(u))) >= 0.0)
    assert g(arr(1.0))[0] == grid[-1]
    with pytest.raises(ValueError, match="pmf"):
        c.marginal_ppf_pmf(arr(0, 1), arr(1.2, -0.2))
    with pytest.raises(ValueError, match="pmf"):
        c.marginal_ppf_pmf(arr(0, 1), arr(0.5, 0.4))
    with pytest.raises(ValueError, match="grid"):
        c.marginal_ppf_pmf(arr(1, 0), arr(0.5, 0.5))


@pytest.mark.parametrize("family,nu", [("gaussian", None), ("t", 5.0)])
def test_simulate_copula_uniforms_marginals_and_kendall(family, nu):
    n = 100_000
    u = c.simulate_copula_uniforms(R5, family, nu, n, seed=11)
    assert u.shape == (n, 2)
    assert u.min() > 0.0 and u.max() < 1.0
    for k in range(2):
        assert stats.kstest(u[:, k], "uniform").statistic < 3.0 / math.sqrt(n)
    tau = stats.kendalltau(u[:20_000, 0], u[:20_000, 1]).statistic
    assert tau == pytest.approx(2.0 / math.pi * math.asin(0.5), abs=0.015)
    assert 2.0 / math.pi * math.asin(0.5) == pytest.approx(0.3333, abs=5e-5)
    # the all-ones correlation is singular positive semi-definite and is accepted
    u1 = c.simulate_copula_uniforms(np.ones((2, 2)), family, nu, 1000, seed=1)
    assert np.allclose(u1[:, 0], u1[:, 1], atol=1e-9)


def test_simulate_copula_uniforms_seed_and_refusals():
    a = c.simulate_copula_uniforms(R5, "gaussian", None, 500, seed=3)
    b = c.simulate_copula_uniforms(R5, "gaussian", None, 500, seed=3)
    d = c.simulate_copula_uniforms(R5, "gaussian", None, 500, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, d)
    # the same standard normal draw feeds the t copula: with a huge nu it is the Gaussian copula
    g = c.simulate_copula_uniforms(R5, "gaussian", None, 2000, seed=5)
    t = c.simulate_copula_uniforms(R5, "t", 1e7, 2000, seed=5)
    assert stats.spearmanr(g[:, 0], t[:, 0]).statistic > 0.999
    bad = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    with pytest.raises(ValueError, match="positive semi-definite"):
        c.simulate_copula_uniforms(bad, "gaussian", None, 10, seed=0)
    with pytest.raises(ValueError, match="n"):
        c.simulate_copula_uniforms(R5, "gaussian", None, 0, seed=0)
    with pytest.raises(ValueError, match="nu"):
        c.simulate_copula_uniforms(R5, "t", None, 10, seed=0)
    with pytest.raises(ValueError, match="nu"):
        c.simulate_copula_uniforms(R5, "t", 2.0, 10, seed=0)
    with pytest.raises(ValueError, match="family"):
        c.simulate_copula_uniforms(R5, "clayton", None, 10, seed=0)
    with pytest.raises(ValueError, match="nu"):
        c.simulate_copula_uniforms(R5, "gaussian", 5.0, 10, seed=0)


def _ident(u):
    return np.asarray(u, dtype=float)


def test_simulate_component_losses_dependence_and_marginals():
    s = 100_000
    ppfs = [_ident, lambda u: 3.0 + 2.0 * stats.norm.ppf(u), lambda u: -np.log1p(-u)]
    como = c.simulate_component_losses(ppfs, np.ones((3, 3)), "gaussian", None, s, seed=1)
    assert como.shape == (s, 3)
    # one common U: the columns are increasing functions of the first one
    assert np.allclose(como[:, 1], 3.0 + 2.0 * stats.norm.ppf(como[:, 0]), atol=1e-7)
    assert np.allclose(como[:, 2], -np.log1p(-como[:, 0]), atol=1e-7)
    ind = c.simulate_component_losses(ppfs, np.eye(3), "gaussian", None, s, seed=2)
    cc = np.corrcoef(ind.T)
    for i in range(3):
        for j in range(i):
            assert abs(cc[i, j]) < 4.0 / math.sqrt(s)
    assert stats.kstest(ind[:, 1], "norm", args=(3.0, 2.0)).statistic < 3.0 / math.sqrt(s)
    assert stats.kstest(ind[:, 2], "expon").statistic < 3.0 / math.sqrt(s)
    assert stats.kstest(ind[:, 0], "uniform").statistic < 3.0 / math.sqrt(s)
    dep = c.simulate_component_losses(ppfs[:2], R5, "t", 4.0, s, seed=3)
    assert stats.kstest(dep[:, 1], "norm", args=(3.0, 2.0)).statistic < 3.0 / math.sqrt(s)
    with pytest.raises(ValueError, match="ppfs"):
        c.simulate_component_losses(ppfs, R5, "gaussian", None, 10, seed=0)
    with pytest.raises(ValueError, match="n_draws"):
        c.simulate_component_losses(ppfs[:2], R5, "gaussian", None, 0, seed=0)
    with pytest.raises(ValueError, match="chunk_size"):
        c.simulate_component_losses(ppfs[:2], R5, "gaussian", None, 10, seed=0, chunk_size=0)


def test_simulate_component_losses_chunks_and_seed():
    ppfs = [_ident, _ident]
    a = c.simulate_component_losses(ppfs, R5, "gaussian", None, 10_500, seed=9, chunk_size=1000)
    b = c.simulate_component_losses(ppfs, R5, "gaussian", None, 10_500, seed=9, chunk_size=1000)
    d = c.simulate_component_losses(ppfs, R5, "gaussian", None, 10_500, seed=9, chunk_size=2000)
    assert a.shape == (10_500, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, d)  # another chunk size: a different, equally valid sample
    assert stats.kstest(d[:, 0], "uniform").statistic < 3.0 / math.sqrt(10_500)
    n, k, chunk = 60_000, 3, 500
    ppfs3 = [_ident] * 3
    tracemalloc.start()
    out = c.simulate_component_losses(ppfs3, np.eye(3), "t", 5.0, n, seed=1, chunk_size=chunk)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert out.nbytes == n * k * 8
    assert peak <= out.nbytes + 60 * chunk * k * 8  # the working memory of the draw scales with the chunk, not with n


def test_t_tail_dependence_goldens_and_monotone():
    def by_simpson(rho, nu):
        x = -math.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
        t = np.linspace(-2000.0, x, 2_000_001)
        return 2.0 * integrate.simpson(stats.t.pdf(t, nu + 1.0), x=t)

    for (rho, nu), printed in {(0.5, 4): 0.2532, (0.0, 4): 0.0756, (0.5, 8): 0.1173, (0.5, 30): 0.0030}.items():
        got = c.t_tail_dependence(rho, nu)
        assert got == pytest.approx(printed, abs=5e-5)
        assert got == pytest.approx(by_simpson(rho, nu), rel=1e-6)
    nus = [3.0, 4.0, 8.0, 30.0, 300.0]
    vals = [c.t_tail_dependence(0.5, v) for v in nus]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))
    assert vals[-1] < 1e-6
    rhos = [-0.9, -0.3, 0.0, 0.4, 0.9]
    vr = [c.t_tail_dependence(r, 4.0) for r in rhos]
    assert all(a < b for a, b in zip(vr, vr[1:], strict=False))
    with pytest.raises(ValueError, match="rho"):
        c.t_tail_dependence(1.0, 4.0)
    with pytest.raises(ValueError, match="rho"):
        c.t_tail_dependence(-1.0, 4.0)
    with pytest.raises(ValueError, match="nu"):
        c.t_tail_dependence(0.5, 0.0)


def test_clayton_copula_golden_and_bounds():
    assert c.clayton_copula(0.5, 0.5, 1.0) == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert c.clayton_copula(0.5, 0.5, 1.0) == pytest.approx(0.3333, abs=5e-5)
    for u in (0.1, 0.5, 1.0):
        assert c.clayton_copula(u, 1.0, 2.0) == pytest.approx(u, abs=1e-12)
    rng = np.random.default_rng(5)
    for _ in range(200):
        u, v = rng.uniform(0.01, 1.0, 2)
        th = rng.uniform(0.05, 10.0)
        val = c.clayton_copula(u, v, th)
        assert max(u + v - 1.0, 0.0) - 1e-12 <= val <= min(u, v) + 1e-12
        assert val == pytest.approx(c.clayton_copula(v, u, th), rel=1e-12)
    with pytest.raises(ValueError, match="theta"):
        c.clayton_copula(0.5, 0.5, 0.0)
    with pytest.raises(ValueError, match="u"):
        c.clayton_copula(0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="v"):
        c.clayton_copula(0.5, 1.5, 1.0)


def test_clayton_lower_tail_dependence_golden():
    assert c.clayton_lower_tail_dependence(1.0) == pytest.approx(0.5, abs=1e-12)
    assert c.clayton_lower_tail_dependence(2.0) == pytest.approx(0.7071, abs=5e-5)
    # the limit definition: C(u, u) / u as u tends to 0
    th = 1.7
    assert c.clayton_copula(1e-9, 1e-9, th) / 1e-9 == pytest.approx(c.clayton_lower_tail_dependence(th), rel=1e-6)
    vals = [c.clayton_lower_tail_dependence(t) for t in (0.1, 0.5, 1.0, 3.0, 20.0)]
    assert all(0.0 < v < 1.0 for v in vals)
    assert all(a < b for a, b in zip(vals, vals[1:], strict=False))
    with pytest.raises(ValueError, match="theta"):
        c.clayton_lower_tail_dependence(-1.0)


def test_copula_aggregate_comonotonic_triangular_and_elliptical():
    n = 300_000
    # comonotonic: the VaR of the total is the sum of the marginal quantiles
    ppfs = [lambda u: -np.log1p(-u), lambda u: 3.0 * (1.0 - u) ** (-0.25) - 3.0]
    level = 0.99
    agg = c.copula_aggregate(ppfs, np.ones((2, 2)), "gaussian", None, 0.0, level, n, seed=1)
    target = float(-np.log1p(-level) + 3.0 * (1.0 - level) ** (-0.25) - 3.0)
    assert abs(agg.var - target) <= 5.0 * agg.var_se
    # two independent uniforms: the triangular law, VaR_0.5 = 1
    tri = c.copula_aggregate([_ident, _ident], np.eye(2), "gaussian", None, 0.0, 0.5, n, seed=2)
    assert abs(tri.var - 1.0) <= 5.0 * tri.var_se
    # normal marginals under the Gaussian copula: the variance-covariance quantile
    mu = arr(10.0, 20.0, 5.0)
    sig = arr(2.0, 3.0, 1.0)
    corr = np.array([[1, 0.5, 0.2], [0.5, 1, -0.3], [0.2, -0.3, 1]])
    nppfs = [(lambda u, m=m, s=s: m + s * stats.norm.ppf(u)) for m, s in zip(mu, sig, strict=True)]
    ell = c.copula_aggregate(nppfs, corr, "gaussian", None, float(mu.sum()), 0.99, n, seed=3)
    vc = float(mu.sum() + stats.norm.ppf(0.99) * math.sqrt(sig @ corr @ sig))
    assert abs(ell.var - vc) <= 5.0 * ell.var_se
    assert ell.ec == pytest.approx(ell.var - ell.el, abs=1e-9)


def test_copula_aggregate_invariants_and_refusals():
    ppfs = [lambda u: -np.log1p(-u), lambda u: -np.log1p(-u)]
    a = c.copula_aggregate(ppfs, R5, "t", 4.0, 1.0, 0.99, 20_000, seed=4, chunk_size=5000)
    b = c.copula_aggregate(ppfs, R5, "t", 4.0, 1.0, 0.99, 20_000, seed=4, chunk_size=5000)
    assert a == b
    assert (a.family, a.nu, a.n_draws, a.seed, a.chunk_size, a.level, a.el) == ("t", 4.0, 20_000, 4, 5000, 0.99, 1.0)
    assert a.ec == pytest.approx(a.var - a.el, abs=1e-9 * max(1.0, a.var))
    assert a.es >= a.var - 1e-9
    assert a.var_se is not None and a.es_se is not None
    short = c.copula_aggregate(ppfs, R5, "gaussian", None, 0.0, 0.9, 100, seed=4)
    assert short.var_se is None and short.es_se is None
    edge = c.copula_aggregate(ppfs, R5, "gaussian", None, 0.0, 0.9, 200, seed=4)
    assert edge.var_se is not None and edge.es_se is not None
    gauss = c.copula_aggregate(ppfs, R5, "gaussian", None, 0.0, 0.99, 20_000, seed=4)
    # the t copula's tail dependence puts more mass on joint extremes at a high level
    heavy = c.copula_aggregate(ppfs, R5, "t", 3.0, 0.0, 0.999, 300_000, seed=5)
    light = c.copula_aggregate(ppfs, R5, "gaussian", None, 0.0, 0.999, 300_000, seed=5)
    assert heavy.es > light.es
    assert gauss.var > 0.0
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="level"):
            c.copula_aggregate(ppfs, R5, "gaussian", None, 0.0, bad, 1000, seed=0)
    with pytest.raises(ValueError, match="nu"):
        c.copula_aggregate(ppfs, R5, "t", None, 0.0, 0.99, 1000, seed=0)
    with pytest.raises(ValueError, match="positive semi-definite"):
        c.copula_aggregate(ppfs * 1, np.array([[1, 2], [2, 1]]), "gaussian", None, 0.0, 0.99, 1000, seed=0)
    with pytest.raises(ValueError, match="el"):
        c.copula_aggregate(ppfs, R5, "gaussian", None, float("nan"), 0.99, 1000, seed=0)
