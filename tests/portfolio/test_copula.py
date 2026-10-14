"""RV-706 to RV-709: the default-mode Gaussian and Student-t copula Monte Carlo (dossier 04 H.3, ladder rungs C2
and C3)."""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest
from _portfolio_helpers import identical
from scipy import integrate, stats

from riskvalidation.engines.vasicek import conditional_pd, lhp_quantile
from riskvalidation.portfolio.copula import (
    LossSample,
    mc_standard_errors,
    simulate_gaussian,
    simulate_student_t,
)
from riskvalidation.portfolio.portfolio import Portfolio, es_from_sample, expected_loss, var_from_sample


def _pair(pd=0.01, rho=0.3) -> Portfolio:
    return identical(2, pd=pd, rho=rho)


def _joint_gaussian(pd: float, rho: float) -> float:
    val, _ = integrate.quad(lambda z: conditional_pd(pd, rho, z) ** 2 * stats.norm.pdf(z), -12, 12, epsabs=1e-14)
    return float(val)


def _joint_student_t(pd: float, rho: float, nu: float) -> float:
    """P(both default) = E[p(Z, W)^2] with W = nu / chi2_nu, by quadrature over Z and the chi-square variable."""
    q = stats.t.ppf(pd, nu)
    nodes, wts = np.polynomial.hermite_e.hermegauss(120)  # probabilists' Gauss-Hermite: integral of f(z) phi(z) dz

    def inner(w: float) -> float:
        scale = math.sqrt(nu / w)
        val = float(np.sum(wts * stats.norm.cdf((q / scale - math.sqrt(rho) * nodes) / math.sqrt(1 - rho)) ** 2))
        return val / math.sqrt(2.0 * math.pi) * stats.chi2.pdf(w, nu)

    val, _ = integrate.quad(inner, 0, 300, epsabs=1e-13, limit=200)
    return float(val)


def _both(sample: LossSample) -> tuple[float, float]:
    """The fraction of scenarios in which both unit names default and its Monte Carlo standard error."""
    p = float((sample.losses == 2.0).mean())
    return p, float(np.sqrt(p * (1.0 - p) / sample.n_scenarios))


def test_gaussian_joint_default_golden():
    assert _joint_gaussian(0.01, 0.3) == pytest.approx(0.000556, abs=5e-7)  # the independent anchor of the golden
    s = simulate_gaussian(_pair(), n_scenarios=20_000_000, seed=11)
    assert isinstance(s, LossSample) and s.n_scenarios == 20_000_000 and s.seed == 11 and s.chunk_scenarios == 100_000
    p, se = _both(s)
    assert abs(p - _joint_gaussian(0.01, 0.3)) <= 4 * se
    assert s.group_losses is None and (s.losses >= 0).all() and s.losses.shape == (20_000_000,)


def test_gaussian_marginals_mean_and_groups():
    # three names, two factors with a correlation between them, one group per name: the marginal PD of each name
    pds = np.array([0.02, 0.05, 0.10])
    p = Portfolio(
        ead=np.array([1.0, 1.0, 1.0]), pd=pds, lgd=np.array([1.0, 1.0, 1.0]),
        loadings=np.array([[0.5, 0.0], [0.4, 0.3], [0.0, 0.6]]), factor_corr=np.array([[1.0, 0.5], [0.5, 1.0]]),
        group=np.array([0, 1, 2]), group_ids=("a", "b", "c"), lgd_gamma=0.25, currency="USD", label="illustrative",
    )
    s = simulate_gaussian(p, 400_000, seed=5, chunk_scenarios=50_000, track_groups=True)
    assert s.group_losses is not None and s.group_losses.shape == (400_000, 3)
    assert np.allclose(s.group_losses.sum(axis=1), s.losses, rtol=0, atol=1e-9)
    for g in range(3):
        freq = float(s.group_losses[:, g].mean())
        se = float(np.sqrt(pds[g] * (1 - pds[g]) / 400_000))
        assert abs(freq - pds[g]) <= 4 * se
    se_mean = s.losses.std(ddof=1) / np.sqrt(s.losses.size)
    assert abs(s.losses.mean() - expected_loss(p)) <= 4 * se_mean
    # the names sharing a factor default together more often than independent names would
    both01 = float(((s.group_losses[:, 0] > 0) & (s.group_losses[:, 1] > 0)).mean())
    assert both01 > pds[0] * pds[1] * 1.5
    # loss given default and exposure: a single name loses ead * lgd
    one = identical(1, pd=0.1, rho=0.3, lgd=0.4, ead=5.0)
    s1 = simulate_gaussian(one, 20_000, seed=2)
    assert set(np.unique(s1.losses)) <= {0.0, 2.0}


def test_gaussian_reproducible_and_memory_bounded():
    p = identical(50)
    a = simulate_gaussian(p, 30_000, seed=3, chunk_scenarios=10_000)
    b = simulate_gaussian(p, 30_000, seed=3, chunk_scenarios=10_000)
    c = simulate_gaussian(p, 30_000, seed=4, chunk_scenarios=10_000)
    d = simulate_gaussian(p, 30_000, seed=3, chunk_scenarios=7_000)
    assert np.array_equal(a.losses, b.losses) and not np.array_equal(a.losses, c.losses)
    assert not np.array_equal(a.losses, d.losses) and d.chunk_scenarios == 7_000 and d.n_scenarios == 30_000
    assert abs(a.losses.mean() - d.losses.mean()) < 6 * a.losses.std() / np.sqrt(30_000)
    # one chunk holds at most chunk_scenarios * N * 8 bytes of float64
    big = identical(2000)
    chunk = 4000
    bound = chunk * 2000 * 8
    tracemalloc.start()
    simulate_gaussian(big, 12_000, seed=1, chunk_scenarios=chunk)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak <= bound * 1.02 + 2_000_000, (peak, bound)
    for bad in (0, -5):
        with pytest.raises(ValueError):
            simulate_gaussian(p, bad, seed=1)
        with pytest.raises(ValueError):
            simulate_gaussian(p, 100, seed=1, chunk_scenarios=bad)


def test_gaussian_converges_to_lhp():
    rho, pd = 0.15, 0.02
    anchor = {q: float(lhp_quantile(q, pd, rho)) for q in (0.99, 0.999)}
    gaps = {}
    for n in (25, 200, 1000):
        s = simulate_gaussian(identical(n, pd=pd, rho=rho), 200_000, seed=21, chunk_scenarios=50_000)
        for q in anchor:
            var = var_from_sample(s.losses, q) / n
            se = mc_standard_errors(s.losses, q)[0] / n
            gaps[n, q] = (var - anchor[q], se)
    # the granularity adds to the loss quantile of a small portfolio and the gap closes as the names grow
    assert gaps[25, 0.99][0] > 4 * gaps[25, 0.99][1] and gaps[25, 0.999][0] > 4 * gaps[25, 0.999][1]
    assert abs(gaps[200, 0.99][0]) < abs(gaps[25, 0.99][0]) and abs(gaps[1000, 0.99][0]) < abs(gaps[200, 0.99][0])
    # for many names the Monte Carlo quantile sits on the LHP anchor within its Monte Carlo interval (plus one
    # default step)
    for q in anchor:
        assert abs(gaps[1000, q][0]) <= 4 * gaps[1000, q][1] + 2.0 / 1000, (q, gaps[1000, q])


def test_student_t_joint_default_golden():
    truth = _joint_student_t(0.01, 0.3, 5.0)  # the mixture representation by two-dimensional quadrature
    assert truth == pytest.approx(0.001647, abs=5e-6)
    s = simulate_student_t(_pair(), nu=5.0, n_scenarios=20_000_000, seed=17)
    p, se = _both(s)
    assert abs(p - truth) <= 4 * se
    assert p > _joint_gaussian(0.01, 0.3) + 5 * se  # tail dependence: above the Gaussian 0.000556 for the same rho
    assert abs(p - 0.001647) <= 1e-5 + 4 * se


def test_student_t_marginals_limit_and_refusal():
    pds = np.array([0.02, 0.05])
    p = Portfolio(
        ead=np.array([1.0, 1.0]), pd=pds, lgd=np.array([1.0, 1.0]), loadings=np.sqrt(np.array([[0.3], [0.3]])),
        factor_corr=np.eye(1), group=np.array([0, 1]), group_ids=("a", "b"), lgd_gamma=0.0, currency="EUR",
        label="illustrative",
    )
    s = simulate_student_t(p, nu=6.0, n_scenarios=1_000_000, seed=8, chunk_scenarios=100_000, track_groups=True)
    assert s.group_losses is not None and np.allclose(s.group_losses.sum(axis=1), s.losses)
    for g in range(2):
        se = float(np.sqrt(pds[g] * (1 - pds[g]) / 1_000_000))
        assert abs(float(s.group_losses[:, g].mean()) - pds[g]) <= 4 * se
    # nu -> infinity: the Gaussian joint default probability
    two = _pair(pd=0.05, rho=0.3)
    s_inf = simulate_student_t(two, nu=5000.0, n_scenarios=2_000_000, seed=9)
    pj, se = _both(s_inf)
    assert abs(pj - _joint_gaussian(0.05, 0.3)) <= 4 * se
    again = simulate_student_t(two, nu=5000.0, n_scenarios=2_000_000, seed=9)
    assert np.array_equal(again.losses, s_inf.losses)
    for nu in (2.0, 1.5, 0.0, float("nan")):
        with pytest.raises(ValueError):
            simulate_student_t(two, nu=nu, n_scenarios=100, seed=1)
    with pytest.raises(ValueError):
        simulate_student_t(two, nu=5.0, n_scenarios=0, seed=1)


def test_mc_standard_errors_golden_scaling_and_refusals():
    losses = np.tile(np.arange(10.0), 20)
    assert mc_standard_errors(losses, 0.9) == (0.0, 0.0)
    rng = np.random.default_rng(0)
    ratios_var, ratios_es = [], []
    for seed in range(6):
        r = np.random.default_rng(seed)
        small = r.gamma(2.0, 1.0, size=100_000)
        large = r.gamma(2.0, 1.0, size=400_000)
        sv, se_ = mc_standard_errors(small, 0.99)
        lv, le = mc_standard_errors(large, 0.99)
        assert sv > 0 and se_ > 0 and lv > 0 and le > 0
        ratios_var.append(sv / lv)
        ratios_es.append(se_ / le)
    assert 1.5 < float(np.mean(ratios_var)) < 2.6 and 1.5 < float(np.mean(ratios_es)) < 2.6
    x = rng.gamma(2.0, 1.0, size=5000)
    assert mc_standard_errors(x, 0.95) == mc_standard_errors(x.copy(), 0.95)
    # the remainder of an unequal split is dropped
    assert mc_standard_errors(x[:4999], 0.95, n_batches=20) == mc_standard_errors(x[:4980], 0.95, n_batches=20)
    with pytest.raises(ValueError):
        mc_standard_errors(x, 0.95, n_batches=1)
    with pytest.raises(ValueError):
        mc_standard_errors(x[:150], 0.95, n_batches=20)  # 7 losses per batch
    with pytest.raises(ValueError):
        mc_standard_errors(x, 1.0)
    # the batch estimators are the sample estimators of the contract
    batch = x[:250]
    assert var_from_sample(batch, 0.95) <= es_from_sample(batch, 0.95)
