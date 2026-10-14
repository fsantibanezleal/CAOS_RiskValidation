"""RV-1226: the findings of the Opus review of the U12 engine (B1, M1, M2, m1 to m4, the surviving bootstrap mutant)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.core.result import Light
from riskvalidation.oprisk import aggregate as a
from riskvalidation.oprisk import distributions as d
from riskvalidation.oprisk import scenarios as sc
from riskvalidation.oprisk.scenarios import ScenarioPoint as P
from riskvalidation.validation import oprisk as v


def gpd_points(xi, beta, xs, frequency):
    tail = stats.genpareto.sf(np.asarray(xs), xi, scale=beta)
    pairs = enumerate(zip(xs, tail, strict=True))
    return [P(f"p{i}", "percentile", float(x), float(frequency * s)) for i, (x, s) in pairs]


# ---- B1: the expected shortfall of a severity without a finite mean --------------------------------------------


def coarse_infinite_mean_grid():
    f = a.discretise_severity("gpd", (1.1, 100.0, 0.0), 1e6, 32768)
    return a.fft_aggregate("poisson", (10.0,), f, padding_factor=4), 1e6


def test_a_coarse_grid_of_an_infinite_mean_severity_passes_the_pmf_guard_so_the_severity_gate_is_required():
    h, step = coarse_infinite_mean_grid()
    assert 1.0 - h.sum() < 1e-3 * (1.0 - 0.999)                      # the grid guard is satisfied ...
    assert a.es_from_pmf(h, step, 0.999) > 1e7                        # ... and a finite, meaningless ES comes out
    with pytest.raises(ValueError, match="infinite mean"):
        a.es_from_pmf(h, step, 0.999, infinite_mean=True)             # the caller's gate
    for xi in (1.0, 1.1, 2.0):
        with pytest.raises(ValueError, match="infinite mean"):
            a.es_for_severity(h, step, 0.999, "gpd", (xi, 100.0, 0.0))


def test_es_for_severity_refuses_any_grid_of_an_infinite_mean_severity_and_matches_es_from_pmf_otherwise():
    for step, n in ((1.0, 1000), (1e5, 4096), (1e6, 32768)):
        f = a.discretise_severity("gpd", (1.1, 100.0, 0.0), step, n)
        h = a.fft_aggregate("poisson", (10.0,), f, padding_factor=4)
        with pytest.raises(ValueError, match="infinite mean"):
            a.es_for_severity(h, step, 0.999, "gpd", (1.1, 100.0, 0.0))
    f = a.discretise_severity("lognormal", (0.0, 1.0), 0.25, 8192)
    h = a.fft_aggregate("poisson", (10.0,), f)
    assert a.es_for_severity(h, 0.25, 0.99, "lognormal", (0.0, 1.0)) == a.es_from_pmf(h, 0.25, 0.99)
    # the splice with a tail index above 1 has no mean either
    with pytest.raises(ValueError, match="infinite mean"):
        a.es_for_severity(h, 0.25, 0.99, "lognormal-gpd-splice", (0.0, 1.0, 5.0, 1.2, 3.0, 0.9))
    # a finite-mean GPD is allowed through to the grid guard
    g = a.discretise_severity("gpd", (0.5, 100.0, 0.0), 10.0, 4096)
    hg = a.fft_aggregate("poisson", (3.0,), g, padding_factor=4)
    assert a.es_for_severity(hg, 10.0, 0.9, "gpd", (0.5, 100.0, 0.0)) >= a.quantile_from_pmf(hg, 10.0, 0.9)


# ---- M1: no optimiser bound is a statistic ---------------------------------------------------------------------


def test_the_scenario_prior_refuses_points_worth_less_than_one_expected_event_in_the_basis_years():
    # 1 in 10, 30 and 100 years above EUR 20,000 with 5 years of basis: expected counts 0.5, 0.17 and 0.05
    pts = [P("a", "percentile", 100_000.0, 0.1), P("b", "percentile", 400_000.0, 1 / 30),
           P("c", "percentile", 2e6, 0.01)]
    for seed in range(3):
        with pytest.raises(ValueError, match="basis_years"):
            sc.gamma_prior_from_scenarios(pts, 20_000.0, 41.0, basis_years=5.0, n_bootstrap=50, seed=seed)


def test_the_scenario_prior_interval_never_ends_at_the_optimiser_bound():
    pts = gpd_points(0.96, 22_350.0, [30_000.0, 200_000.0, 2e6], 41.0)        # 1 in 1, 10 and 100 per year ...
    pts = [P(p.scenario_id, p.kind, p.loss, p.exceedance_frequency * 25.0) for p in pts]   # ... 25x the events
    prior = sc.gamma_prior_from_scenarios(pts, 0.0, 41.0 * 25.0, basis_years=5.0, n_bootstrap=200, seed=3)
    assert prior.interval[1] < 100.0 and prior.interval[0] > 0.0
    assert prior.interval[0] < prior.mean < prior.interval[1]


def test_a_scenario_fit_on_the_search_bound_is_not_converged_and_a_bootstrap_on_it_raises():
    # a larger loss that is more frequent: no tail
    flat = [P("a", "percentile", 1000.0, 1.0), P("b", "percentile", 5000.0, 2.0)]
    fit = sc.fit_tail_to_scenarios(flat, 0.0, 10.0)
    assert fit.converged is False
    good = sc.fit_tail_to_scenarios([P("a", "percentile", 200.0, 2.5), P("b", "percentile", 1800.0, 0.1)], 0.0, 10.0)
    assert good.converged is True
    with pytest.raises(ValueError, match="search bound"):
        sc.gamma_prior_from_scenarios(flat, 0.0, 10.0, basis_years=10.0, n_bootstrap=20, seed=1)


def test_the_bootstrap_interval_is_the_5th_and_95th_percentile_of_the_refitted_alphas():
    """Kills the mutant that moved the lower percentile to the 10th."""
    pts = gpd_points(0.5, 100.0, [150.0, 400.0, 1_500.0], 200.0)
    prior = sc.gamma_prior_from_scenarios(pts, 0.0, 200.0, basis_years=10.0, n_bootstrap=300, seed=7)
    base = sc.fit_tail_to_scenarios(pts, 0.0, 200.0)
    u = np.array([p.loss for p in pts])
    freq = np.array([p.exceedance_frequency for p in pts])
    rng = np.random.Generator(np.random.PCG64(7))
    alphas = []
    for _ in range(300):
        counts = rng.poisson(freq * 10.0)
        redrawn = np.where(counts == 0, 0.5, counts) / 10.0
        xi_b, _, _ = sc._fit_scenario_gpd(sc._conditional(redrawn, 200.0), u, starts=(), warm=tuple(base.params[:2]))
        alphas.append(1.0 / xi_b)
    lo, hi = np.percentile(alphas, [5.0, 95.0])
    assert prior.interval == (pytest.approx(lo, rel=1e-12), pytest.approx(hi, rel=1e-12))
    assert prior.interval[0] != pytest.approx(float(np.percentile(alphas, 10.0)), rel=1e-6)


# ---- M2: the benchmark tolerance against the four-decimal references -------------------------------------------


def test_the_benchmark_is_green_on_the_engines_exact_moments_at_a_tolerance_above_the_reference_rounding():
    cm = a.compound_poisson_moments(100.0, "lognormal", (0.0, 2.0))
    quantiles = {"panjer": 5849.0, "fft": 5849.0, "monte-carlo": 5853.1}
    exact = v.benchmark_reproduction((cm.mean, cm.variance, cm.skewness), quantiles, 1e-5, 0.01)
    assert exact.light is Light.GREEN and exact.metric <= 1.0
    # the references carry four decimals (e^6 / 10 = 40.342879 is quoted as 40.3428): 1e-6 is below that rounding
    assert abs(cm.skewness - 40.3428) / 40.3428 > 1e-6
    assert "1e-5" in (v.benchmark_reproduction.__doc__ or "")
    # the contract default (moment_rel_tol 1e-5, quantile_rel_tol 0.01) is green on the exact moments, metric 0.196688
    default = v.benchmark_reproduction((cm.mean, cm.variance, cm.skewness), quantiles)
    assert default.light is Light.GREEN and default.metric == pytest.approx(0.196688, abs=5e-7)
    assert default.extras["thresholds"] == {"moment_rel_tol": 1e-5, "quantile_rel_tol": 0.01}
    amber = v.benchmark_reproduction((cm.mean, cm.variance, cm.skewness), quantiles, 1e-6, 0.01)
    assert amber.light is Light.AMBER and amber.metric == pytest.approx(1.966876, abs=5e-7)


# ---- minors ----------------------------------------------------------------------------------------------------


def test_m1_threshold_stability_counts_the_infinite_mean_fit_when_not_evaluated():
    r = v.threshold_stability(np.array([1.0, 2.0]), np.array([1.2, math.nan]))
    assert r.light is Light.NOT_EVALUATED and r.extras["n_infinite_mean"] == 1
    assert v.threshold_stability(np.array([1.0, 2.0]), np.array([0.4, math.nan])).extras["n_infinite_mean"] == 0


def test_m2_the_monte_carlo_quantile_of_the_benchmark_is_within_four_standard_errors_of_5853_1():
    s = a.simulate_annual_losses("poisson", (100.0,), "lognormal", (0.0, 2.0), n_years=1_000_000, seed=2026,
                                 chunk_years=20_000)
    q, se = a.mc_quantile(s.losses, 0.999)
    assert abs(q - 5853.1) <= 4.0 * se


def test_m3_a_nan_loss_is_refused_not_read_as_a_loss_at_the_origin():
    nan = np.array([1.0, math.nan])
    with pytest.raises(ValueError, match="NaN"):
        d.severity_cdf("lognormal", (0.0, 1.0), nan)
    with pytest.raises(ValueError, match="NaN"):
        d.truncated_severity_cdf("gpd", (0.5, 100.0, 0.0), nan, 100.0)
    assert d.severity_cdf("lognormal", (0.0, 1.0), np.array([math.inf]))[0] == 1.0


def test_a_bootstrap_refit_on_the_search_bound_raises_naming_the_share():
    """Points worth 1 to 4 events a year (so the precondition holds) with one year of basis: some redraws collapse
    and the refit runs onto the box; that must raise, not enter the percentiles as alpha = 1000."""
    pts = [P("a", "percentile", 30_000.0, 4.0), P("b", "percentile", 200_000.0, 1.5),
           P("c", "percentile", 2e6, 1.0)]
    for seed in (1, 3):
        with pytest.raises(ValueError, match=r"\d+ of 100 bootstrap refits sit on the search bound"):
            sc.gamma_prior_from_scenarios(pts, 20_000.0, 10.0, basis_years=1.0, n_bootstrap=100, seed=seed)
