"""BL-058: the default correlation from a panel of yearly default counts by rating grade (`portfolio.panel_fit`).

Gordy and Heitfield's MLE3 (one factor, one common loading) and Frey and McNeil's moment estimator (equation 20, k = 2).
The recovery tolerances come from a Monte Carlo of 40 seeds per design measured before the tests were written (the
standard deviation of the asset correlation estimate: design A 0.030, design B 0.0123, design C 0.044, design D
0.023); a mean over ``SEEDS`` seeds is held to 3 sd / sqrt(SEEDS) and each seed to 4 sd, so that a planted correlation
recovered at one lucky seed does not pass."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from scipy import integrate
from scipy.optimize import approx_fprime
from scipy.special import gammaln, log_ndtr
from scipy.stats import multivariate_normal, norm

from riskvalidation.engines.vasicek import conditional_pd, lhp_quantile
from riskvalidation.portfolio import panel_fit as pf
from riskvalidation.portfolio.asrf_fit import fit_moments
from riskvalidation.transitions.intervals import effective_n


def simulate(t: int, pds, cohorts, rho: float, seed: int):
    """T years of defaults of G grades from the one-factor model: X_t ~ N(0, 1), D ~ Binomial(N, p_g(X_t))."""
    rng = np.random.default_rng(seed)
    gam = norm.ppf(np.asarray(pds, dtype=float))
    x = rng.standard_normal(t)
    p = norm.cdf((gam[None, :] - math.sqrt(rho) * x[:, None]) / math.sqrt(1.0 - rho))
    n = np.tile(np.asarray(cohorts), (t, 1))
    return rng.binomial(n, p), n


# design: (T, PDs, cohorts, asset correlation, sd of the estimate, relative sd of each PD estimate), measured
DESIGNS = {
    "A": (100, [0.0015, 0.01, 0.05], [400, 250, 100], 0.2, 0.030, [0.273, 0.166, 0.125]),
    "B": (100, [0.002, 0.01, 0.03, 0.08], [1000, 800, 500, 300], 0.1, 0.0123, [0.148, 0.107, 0.080, 0.067]),
    "C": (50, [0.02, 0.1], [2000, 1000], 0.3, 0.044, [0.252, 0.155]),
}
SEEDS = 8


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_a_planted_correlation_and_the_pds_are_recovered_over_seeds(name):
    t, pds, cohorts, rho, sd, pd_rel_sd = DESIGNS[name]
    errs, rel, covered = [], [], 0
    for seed in range(SEEDS):
        d, n = simulate(t, pds, cohorts, rho, 5000 + seed)
        fit = pf.fit_panel_asrf(d, n)
        assert fit.converged and fit.flags == () and fit.grades == tuple(range(len(pds)))
        assert abs(fit.asset_correlation - rho) <= 4.0 * sd, (seed, fit.asset_correlation)
        errs.append(fit.asset_correlation - rho)
        rel.append(fit.pd / np.asarray(pds) - 1.0)
        lo, hi = fit.rho_interval
        assert BOUND_LOW_TOL <= lo <= fit.asset_correlation <= hi <= 1.0
        covered += lo <= rho <= hi
    assert abs(np.mean(errs)) <= 3.0 * sd / math.sqrt(SEEDS)
    # the mean relative error of each PD within 3 standard errors (relative sd measured over 40 seeds)
    assert np.all(np.abs(np.mean(rel, axis=0)) <= 3.0 * np.asarray(pd_rel_sd) / math.sqrt(SEEDS))
    assert covered >= 6  # a 95% interval over 8 seeds: P(fewer than 6 covering) is below 0.6%


BOUND_LOW_TOL = 1e-4


def test_the_estimates_do_not_depend_on_the_seed_of_one_panel_only():
    """Two designs and different T: the sign of the error alternates and the spread shrinks with T."""
    err_short = [pf.fit_panel_asrf(*simulate(40, [0.01, 0.05], [300, 200], 0.2, s), interval=False).asset_correlation
                 for s in range(8)]
    err_long = [pf.fit_panel_asrf(*simulate(400, [0.01, 0.05], [300, 200], 0.2, s), interval=False).asset_correlation
                for s in range(8)]
    assert np.std(err_long) < np.std(err_short)
    assert any(e > 0.2 for e in err_short) and any(e < 0.2 for e in err_short)
    assert abs(np.mean(err_long) - 0.2) <= 0.02


def test_mle_and_moments_agree_on_a_large_panel():
    """T = 1000: the moment estimator has sd about 0.015 and a small downward bias; the tolerance is 0.04 (asset
    correlation) and 0.008 (default correlation of each grade)."""
    d, n = simulate(1000, [0.0015, 0.01, 0.05], [400, 250, 100], 0.2, 11)
    fit, mom = pf.fit_panel_asrf(d, n, interval=False), pf.fit_panel_moments(d, n)
    assert fit.converged and mom.status == ("ok", "ok", "ok")
    assert abs(fit.asset_correlation - np.mean(mom.asset_correlation)) <= 0.04
    assert np.all(np.abs(fit.default_correlation - mom.default_correlation) <= 0.008)
    assert np.all(np.abs(fit.pd - mom.pd) <= 0.2 * mom.pd)
    # the default correlation grows with the PD at a common asset correlation (equation 5)
    assert np.all(np.diff(fit.default_correlation) > 0)


def test_moment_estimator_hand_case_and_the_effective_n_it_feeds():
    """One grade, four years: D = (0, 5, 0, 5), N = 10. pi = 0.25, pi2 = (20 / 90) / 2 = 1/9, the default correlation
    (1/9 - 1/16) / (1/4 - 1/16) = 7/27, and the Schuermann and Hanson adjusted number for 100 obligors
    100 / (1 + 99 * 7/27) = 3.75."""
    d = np.array([[0], [5], [0], [5]])
    n = np.full((4, 1), 10)
    m = pf.fit_panel_moments(d, n)
    assert m.status == ("ok",) and m.n_years == (4,)
    assert m.pd[0] == pytest.approx(0.25, abs=1e-15)
    assert m.joint_default[0] == pytest.approx(1.0 / 9.0, abs=1e-15)
    assert m.default_correlation[0] == pytest.approx(7.0 / 27.0, abs=1e-15)
    assert effective_n(100, float(m.default_correlation[0])) == pytest.approx(3.75, abs=1e-12)
    # the asset correlation reproduces the joint default probability through the bivariate normal
    g = norm.ppf(0.25)
    joint = multivariate_normal.cdf([g, g], mean=[0, 0], cov=[[1, m.asset_correlation[0]], [m.asset_correlation[0], 1]])
    assert joint == pytest.approx(1.0 / 9.0, abs=1e-6)


def test_the_default_correlation_of_a_fit_feeds_effective_n():
    d, n = simulate(60, [0.01, 0.05], [300, 200], 0.2, 3)
    fit = pf.fit_panel_asrf(d, n, interval=False)
    for rho in fit.default_correlation:
        assert effective_n(200, float(rho)) == pytest.approx(200.0 / (1.0 + 199.0 * rho), rel=1e-12)
        assert 1.0 < effective_n(200, float(rho)) < 200.0


def test_default_correlation_equation_5():
    # Phi_2(G(0.05), G(0.05); 0.2) by scipy's bivariate normal, then (joint - pd^2) / (pd (1 - pd))
    pd_, rho = 0.05, 0.2
    g = norm.ppf(pd_)
    joint = multivariate_normal.cdf([g, g], mean=[0, 0], cov=[[1, rho], [rho, 1]])
    want = (joint - pd_**2) / (pd_ * (1.0 - pd_))
    assert pf.default_correlation(pd_, rho) == pytest.approx(want, abs=1e-7)
    vec = pf.default_correlation(np.array([0.01, 0.05]), 0.2)
    assert vec.shape == (2,) and vec[0] < vec[1] and vec[1] == pytest.approx(want, abs=1e-7)
    assert pf.default_correlation(0.05, 0.001) < pf.default_correlation(0.05, 0.5) < 1.0
    for bad_pd in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="pd"):
            pf.default_correlation(bad_pd, 0.2)
    for bad_rho in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="asset_correlation"):
            pf.default_correlation(0.05, bad_rho)


def test_log_likelihood_against_adaptive_quadrature_by_year():
    d, n = simulate(5, [0.02, 0.1], [30, 20], 0.2, 2)
    gam, rho = np.array([-2.0, -1.3]), 0.2
    w, s = math.sqrt(rho), math.sqrt(1.0 - rho)
    total = 0.0
    for t in range(5):
        def f(x, t=t):
            a = (gam - w * x) / s
            lp, lq = log_ndtr(a), log_ndtr(-a)
            c = gammaln(n[t] + 1) - gammaln(d[t] + 1) - gammaln(n[t] - d[t] + 1)
            return math.exp(float(np.sum(d[t] * lp + (n[t] - d[t]) * lq + c)) - 0.5 * x * x) / math.sqrt(2 * math.pi)

        v, _ = integrate.quad(f, -12, 12, epsabs=0, epsrel=1e-13, limit=500)
        total += math.log(v)
    assert pf.panel_log_likelihood(d, n, gam, rho) == pytest.approx(total, abs=1e-10)


def test_quadrature_node_count_changes_nothing_at_the_stated_tolerance():
    """The default rule (trapezoid on [-10, 10], 401 nodes) against 801 nodes on the reference panel (40 years,
    cohorts 400, 250, 100): the log-likelihood moves by less than 1e-10 at the truth and at the fit; the fit does not
    move with the nodes."""
    d, n = simulate(40, [0.0015, 0.01, 0.05], [400, 250, 100], 0.2, 1)
    for gam, rho in (([-2.97, -2.33, -1.64], 0.2), ([-2.9, -2.3, -1.6], 0.35)):
        a = pf.panel_log_likelihood(d, n, gam, rho, n_nodes=pf.N_NODES)
        b = pf.panel_log_likelihood(d, n, gam, rho, n_nodes=2 * pf.N_NODES - 1)
        assert abs(a - b) < 1e-10
    f1 = pf.fit_panel_asrf(d, n, interval=False)
    f2 = pf.fit_panel_asrf(d, n, interval=False, n_nodes=2 * pf.N_NODES - 1)
    assert f1.n_nodes == pf.N_NODES and f1.quadrature_diff < 1e-10
    assert f1.asset_correlation == pytest.approx(f2.asset_correlation, abs=1e-6)
    assert f1.log_likelihood == pytest.approx(f2.log_likelihood, abs=1e-9)
    np.testing.assert_allclose(f1.pd, f2.pd, rtol=1e-6)


def test_gauss_hermite_is_available_and_converges_on_small_cohorts_only():
    """Gauss-Hermite agrees with the trapezoid and is stable under doubling (128 against 256 nodes, below 1e-10) on a
    panel of small cohorts; on the 400-obligor panel it is off by 1e-5 even at 256 nodes, which is why it is not the
    default (method page)."""
    d, n = simulate(8, [0.02, 0.1], [30, 20], 0.2, 5)
    gam = [-2.0, -1.3]
    g128 = pf.panel_log_likelihood(d, n, gam, 0.2, n_nodes=128, quadrature="gauss-hermite")
    g256 = pf.panel_log_likelihood(d, n, gam, 0.2, n_nodes=256, quadrature="gauss-hermite")
    assert abs(g128 - g256) < 1e-10
    assert g256 == pytest.approx(pf.panel_log_likelihood(d, n, gam, 0.2), abs=1e-10)
    big_d, big_n = simulate(40, [0.0015, 0.01, 0.05], [400, 250, 100], 0.2, 1)
    big = [pf.panel_log_likelihood(big_d, big_n, [-2.9, -2.3, -1.6], 0.2, n_nodes=k, quadrature="gauss-hermite")
           for k in (128, 256)]
    ref = pf.panel_log_likelihood(big_d, big_n, [-2.9, -2.3, -1.6], 0.2)
    assert abs(big[1] - ref) > 1e-8  # the rule is not accurate here
    fit = pf.fit_panel_asrf(d, n, interval=False, n_nodes=64, quadrature="gauss-hermite")
    assert fit.quadrature == "gauss-hermite" and fit.quadrature_diff < 1e-6


def test_a_too_coarse_rule_is_flagged_never_silent():
    d, n = simulate(40, [0.0015, 0.01, 0.05], [400, 250, 100], 0.2, 1)
    coarse = pf.fit_panel_asrf(d, n, interval=False, n_nodes=21)
    assert "quadrature_not_converged" in coarse.flags and coarse.converged is False
    assert coarse.quadrature_diff > pf.QUADRATURE_TOL and coarse.n_nodes == 21
    # the default refines itself until the check passes, on cohorts of thousands
    d2, n2 = simulate(50, [0.02, 0.1], [2000, 1000], 0.3, 1002)
    f = pf.fit_panel_asrf(d2, n2, interval=False)
    assert f.flags == () and f.quadrature_diff <= pf.QUADRATURE_TOL and f.n_nodes >= pf.N_NODES
    with pytest.raises(ValueError, match="gauss-hermite"):
        pf.fit_panel_asrf(d, n, n_nodes=200, quadrature="gauss-hermite")
    with pytest.raises(ValueError, match="quadrature"):
        pf.fit_panel_asrf(d, n, quadrature="simpson")
    with pytest.raises(ValueError, match="n_nodes"):
        pf.fit_panel_asrf(d, n, n_nodes=0)


def test_the_gradient_equals_finite_differences():
    d, n = simulate(30, [0.01, 0.05, 0.1], [300, 200, 100], 0.25, 4)
    dd, nn, u, _, _ = pf._counts(d, n)
    panel = pf._Panel(dd, nn, u, 401)
    x = np.array([-2.3, -1.6, -1.2, 0.25])

    def fun(v):
        return panel.nll(v[:3], v[3], want_grad=False)[0]

    analytic = panel.nll(x[:3], x[3])[1]
    np.testing.assert_allclose(analytic, approx_fprime(x, fun, 1e-7), rtol=2e-5, atol=1e-5)


def test_one_grade_agrees_with_the_moment_logic_of_fit_moments():
    """Cohorts of 10^9 make D (D - 1) / (N (N - 1)) equal the rate squared, so the one-grade moment estimator is
    `asrf_fit.fit_moments` on the rates."""
    rates = np.asarray(lhp_quantile((np.arange(1, 201) - 0.5) / 200, 0.02, 0.15))
    n = np.full((200, 1), 10**9)
    d = np.rint(rates * 1e9).reshape(-1, 1)
    m = pf.fit_panel_moments(d, n)
    ref = fit_moments(d[:, 0] / 1e9)
    assert m.status == ("ok",)
    assert m.pd[0] == pytest.approx(ref.pd, rel=1e-8)
    assert m.asset_correlation[0] == pytest.approx(ref.rho, rel=1e-5)
    assert m.default_correlation[0] == pytest.approx(
        (np.mean(rates**2) - np.mean(rates) ** 2) / (np.mean(rates) - np.mean(rates) ** 2), rel=1e-5)


def test_one_grade_mle_and_moments_agree_within_their_error():
    d, n = simulate(500, [0.03], [1000], 0.2, 8)
    fit, mom = pf.fit_panel_asrf(d, n, interval=False), pf.fit_panel_moments(d, n)
    assert fit.grades == (0,) and abs(fit.asset_correlation - mom.asset_correlation[0]) <= 0.06
    assert abs(fit.asset_correlation - 0.2) <= 0.05 and abs(mom.asset_correlation[0] - 0.2) <= 0.06


def test_no_excess_dispersion_is_a_flag_at_the_boundary_never_a_silent_zero():
    d = np.full((30, 1), 2)
    n = np.full((30, 1), 100)  # the same count every year: pi2 = 2 / (100 * 99) < pi^2 = 4e-4
    mom = pf.fit_panel_moments(d, n)
    assert mom.status == ("no_excess_dispersion",)
    assert mom.default_correlation[0] == 0.0 and mom.asset_correlation[0] == 0.0
    fit = pf.fit_panel_asrf(d, n)
    assert "asset_correlation_at_lower_bound" in fit.flags and fit.converged is False
    assert fit.asset_correlation == pytest.approx(1e-4, rel=1e-6)
    assert fit.rho_interval[0] == pytest.approx(1e-4, rel=1e-6)
    # independent defaults simulated at a near-zero correlation
    d2, n2 = simulate(60, [0.02, 0.05], [200, 200], 1e-4, 3)
    f2 = pf.fit_panel_asrf(d2, n2, interval=False)
    assert f2.asset_correlation < 0.06


def test_a_grade_without_defaults_is_dropped_and_reported():
    d, n = simulate(40, [0.01, 0.05], [300, 200], 0.2, 6)
    d3 = np.column_stack([np.zeros(40, dtype=int), d])
    n3 = np.column_stack([np.full(40, 500), n])
    fit = pf.fit_panel_asrf(d3, n3, interval=False)
    assert fit.grades == (1, 2) and fit.dropped == ((0, "no_defaults"),) and len(fit.pd) == 2
    ref = pf.fit_panel_asrf(d, n, interval=False)
    assert fit.asset_correlation == pytest.approx(ref.asset_correlation, abs=1e-9)
    mom = pf.fit_panel_moments(d3, n3)
    assert mom.status[0] == "fewer_than_two_defaults" and np.isnan(mom.asset_correlation[0])
    assert mom.pd[0] == 0.0 and mom.status[1:] == ("ok", "ok")
    with pytest.raises(pf.PanelFitError, match="no grade has a default"):
        pf.fit_panel_asrf(np.zeros((10, 2), dtype=int), np.full((10, 2), 100))
    allmiss = np.full((10, 2), np.nan)
    with pytest.raises(pf.PanelFitError):
        pf.fit_panel_asrf(allmiss, allmiss)


def test_fewer_than_two_defaults_is_flagged_and_not_identified_by_moments():
    d, n = simulate(60, [0.01, 0.05], [300, 200], 0.2, 7)
    d = d.copy()
    d[:, 0] = 0
    d[10, 0] = 1  # a single default in the grade over the whole panel
    fit = pf.fit_panel_asrf(d, n, interval=False)
    assert fit.grades == (0, 1) and "grade_few_defaults" in fit.flags
    mom = pf.fit_panel_moments(d, n)
    assert mom.status[0] == "fewer_than_two_defaults"
    assert np.isnan(mom.default_correlation[0]) and np.isnan(mom.asset_correlation[0])


def test_missing_and_small_cells_are_excluded_counted_never_filled():
    d, n = simulate(40, [0.01, 0.05], [300, 200], 0.2, 9)
    d = d.astype(float)
    n = n.astype(float)
    d[3, 0] = np.nan                  # a missing grade-year
    n[5, 1] = np.nan
    n[7, 0], d[7, 0] = 0.0, 0.0       # an empty cohort
    n[9, 1], d[9, 1] = 1.0, 1.0       # a single obligor: D (D - 1) / (N (N - 1)) is undefined
    fit = pf.fit_panel_asrf(d, n, interval=False)
    assert (fit.n_missing, fit.n_small_cohort, fit.n_cells) == (2, 2, 80 - 4)
    mom = pf.fit_panel_moments(d, n)
    assert (mom.n_missing, mom.n_small_cohort, mom.n_cells) == (2, 2, 76) and mom.n_years == (38, 38)
    # excluding a cell is the same as the cell not being there: the NaN panel equals the panel with those cells emptied
    d2, n2 = d.copy(), n.copy()
    d2[3, 0], n2[3, 0] = 0.0, 0.0
    d2[5, 1], n2[5, 1] = 0.0, 0.0
    fit2 = pf.fit_panel_asrf(d2, n2, interval=False)
    assert fit.asset_correlation == pytest.approx(fit2.asset_correlation, abs=1e-12)
    np.testing.assert_allclose(fit.gamma, fit2.gamma, atol=1e-12)
    # the moment estimate of a grade is the mean over its usable years
    u = np.ones(40, bool)
    u[[3, 7]] = False
    want = np.mean(d[u, 0] * (d[u, 0] - 1) / (n[u, 0] * (n[u, 0] - 1)))
    assert mom.joint_default[0] == pytest.approx(want, abs=1e-15)


def test_malformed_panels_raise_typed_errors():
    ok_d, ok_n = np.ones((5, 2), dtype=int), np.full((5, 2), 10)
    bad = [
        (ok_d[0], ok_n[0]),                              # one-dimensional
        (ok_d, ok_n[:4]),                                # shapes differ
        (np.empty((0, 2)), np.empty((0, 2))),            # empty
        (ok_d * 11, ok_n),                               # more defaults than obligors
        (ok_d - 2, ok_n),                                # negative defaults
        (ok_d + 0.5, ok_n),                              # not an integer
        (ok_d, ok_n + 0.25),
        (np.where(ok_d > 0, np.inf, 0), ok_n),           # infinite
    ]
    for dd, nn in bad:
        with pytest.raises(pf.PanelFitError):
            pf.fit_panel_asrf(dd, nn)
        with pytest.raises(pf.PanelFitError):
            pf.fit_panel_moments(dd, nn)
    assert issubclass(pf.PanelFitError, ValueError)
    with pytest.raises(pf.PanelFitError, match="usable years"):
        pf.fit_panel_asrf(ok_d[:2], ok_n[:2])
    with pytest.raises(ValueError, match="level"):
        pf.fit_panel_asrf(ok_d, ok_n, level=1.0)
    with pytest.raises(ValueError, match="gamma"):
        pf.panel_log_likelihood(ok_d, ok_n, [-2.0], 0.2)
    with pytest.raises(ValueError, match="asset_correlation"):
        pf.panel_log_likelihood(ok_d, ok_n, [-2.0, -2.0], 1.2)


def test_a_short_panel_is_flagged_and_not_bias_corrected():
    d, n = simulate(12, [0.01, 0.05], [300, 200], 0.2, 5)
    fit = pf.fit_panel_asrf(d, n, interval=False)
    assert "short_panel" in fit.flags and fit.n_years == 12
    d2, n2 = simulate(20, [0.01, 0.05], [300, 200], 0.2, 5)
    assert "short_panel" not in pf.fit_panel_asrf(d2, n2, interval=False).flags
    mom = pf.fit_panel_moments(d[:2], n[:2])
    assert mom.status == ("too_few_years", "too_few_years") and np.all(np.isnan(mom.asset_correlation))


def test_moments_second_moment_above_the_bound_is_a_status_not_a_number():
    # almost every default in one year: pi2 close to pi, no correlation below 1 - 1e-9 reproduces it
    d = np.zeros((6, 1), dtype=int)
    d[0, 0] = 1000
    d[1, 0] = 1000
    n = np.full((6, 1), 1000)
    m = pf.fit_panel_moments(d, n)
    assert m.status == ("second_moment_above_bound",) and np.isnan(m.asset_correlation[0])
    assert m.default_correlation[0] == pytest.approx(1.0)


def test_inputs_are_not_mutated_and_the_fit_is_deterministic():
    d, n = simulate(30, [0.01, 0.05], [300, 200], 0.2, 10)
    before = copy.deepcopy((d, n))
    a, b = pf.fit_panel_asrf(d, n), pf.fit_panel_asrf(d, n)
    assert np.array_equal(d, before[0]) and np.array_equal(n, before[1])
    assert a.asset_correlation == b.asset_correlation and a.rho_interval == b.rho_interval
    np.testing.assert_array_equal(a.gamma, b.gamma)
    # permuting the years changes nothing; permuting the grades permutes the grade outputs
    perm = np.random.default_rng(0).permutation(30)
    c = pf.fit_panel_asrf(d[perm], n[perm], interval=False)
    assert c.asset_correlation == pytest.approx(a.asset_correlation, abs=1e-7)
    e = pf.fit_panel_asrf(d[:, ::-1], n[:, ::-1], interval=False)
    assert e.asset_correlation == pytest.approx(a.asset_correlation, abs=1e-7)
    np.testing.assert_allclose(e.pd, a.pd[::-1], rtol=1e-6)


def test_the_profile_interval_widens_with_the_level_and_shrinks_with_the_panel_length():
    d, n = simulate(80, [0.01, 0.05], [300, 200], 0.2, 12)
    f90, f99 = pf.fit_panel_asrf(d, n, level=0.90), pf.fit_panel_asrf(d, n, level=0.99)
    assert f99.rho_interval[0] < f90.rho_interval[0] < f90.asset_correlation < f90.rho_interval[1] < f99.rho_interval[1]
    assert f90.level == 0.90 and f99.level == 0.99
    d2, n2 = simulate(320, [0.01, 0.05], [300, 200], 0.2, 12)
    long90 = pf.fit_panel_asrf(d2, n2, level=0.90)
    assert long90.rho_interval[1] - long90.rho_interval[0] < f90.rho_interval[1] - f90.rho_interval[0]
    assert pf.fit_panel_asrf(d, n, interval=False).rho_interval is None


def test_the_panel_fit_agrees_with_the_year_by_year_likelihood_at_the_estimate():
    d, n = simulate(30, [0.01, 0.05], [300, 200], 0.2, 14)
    fit = pf.fit_panel_asrf(d, n, interval=False)
    full = np.zeros(2)
    full[list(fit.grades)] = fit.gamma
    assert pf.panel_log_likelihood(d, n, full, fit.asset_correlation) == pytest.approx(fit.log_likelihood, abs=1e-9)
    # the estimate is a maximum: nearby parameters have a lower log-likelihood
    for dg, dr in ((0.05, 0.0), (-0.05, 0.0), (0.0, 0.02), (0.0, -0.02)):
        assert pf.panel_log_likelihood(d, n, full + dg, fit.asset_correlation + dr) < fit.log_likelihood
    assert conditional_pd(0.05, 0.2, 0.0) > 0.0  # the engine's conditional PD is the one the likelihood integrates


def test_goldens_of_the_method_page():
    """The values printed on docs/portfolio/08_panel-fit.md."""
    assert pf.default_correlation(0.05, 0.2) == pytest.approx(0.057799, abs=5e-7)
    assert pf.default_correlation(0.01, 0.2) == pytest.approx(0.024133, abs=5e-7)
    d, n = simulate(40, [0.0015, 0.01, 0.05], [400, 250, 100], 0.2, 1)
    assert pf.panel_log_likelihood(d, n, [-2.9, -2.3, -1.6], 0.2) == pytest.approx(-188.809685, abs=5e-7)
    fit = pf.fit_panel_asrf(d, n)
    assert fit.asset_correlation == pytest.approx(0.203868, abs=5e-6)
    assert fit.log_likelihood == pytest.approx(-188.014542, abs=5e-6)
    np.testing.assert_allclose(fit.pd, [0.001441, 0.009478, 0.045882], atol=5e-7)
    np.testing.assert_allclose(fit.default_correlation, [0.007699, 0.024079, 0.056719], atol=5e-7)
    assert fit.rho_interval == pytest.approx((0.131120, 0.315634), abs=5e-6)
    mom = pf.fit_panel_moments(d, n)
    np.testing.assert_allclose(mom.default_correlation, [0.012414, 0.036157, 0.073581], atol=5e-7)
    np.testing.assert_allclose(mom.asset_correlation, [0.245774, 0.257601, 0.248043], atol=5e-7)
