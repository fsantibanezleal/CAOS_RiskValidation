"""RV-1217 and RV-1218: the tail fitted to scenario quantiles by the dossier's objective, and the Bayesian credibility
of Shevchenko and Wuthrich (2006): a Gamma prior on the Pareto shape (section 3.3, equations 21 to 25), the prior built
from an
expert's mean and interval (section 4.3) and the mapping of scenario points to it (decision D4)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.oprisk import scenarios as sc
from riskvalidation.oprisk.scenarios import ScenarioPoint as P


def gpd_points(xi, beta, location, xs, frequency):
    tail = stats.genpareto.sf(np.asarray(xs) - location, xi, scale=beta)
    pairs = enumerate(zip(xs, tail, strict=True))
    return [P(f"p{i}", "percentile", float(x), float(frequency * s)) for i, (x, s) in pairs]


def losses_of_e(n):
    return np.full(n, math.e)


def test_fit_tail_to_scenarios_minimises_the_log_excess_quantile_objective():
    """Dossier 05 F.6's fit: least squares on ln(q_model - threshold) - ln(loss - threshold), reproduced on the four
    LDCE
    median percentiles (xi 0.96378, beta 22349.4, fitted quantiles 27409, 42039, 85025 and 412906)."""
    h = 20_000.0
    levels = (0.25, 0.50, 0.75, 0.95)
    quantiles = (27_632.0, 41_916.0, 82_608.0, 418_400.0)
    # a bank with 41 losses per year above the threshold: the point at percentile p has exceedance frequency 41 (1 - p)
    pts = [P(f"ldce{int(100 * p)}", "percentile", x, 41.0 * (1.0 - p)) for p, x in zip(levels, quantiles, strict=True)]
    fit = sc.fit_tail_to_scenarios(pts, h, 41.0)
    assert fit.params[0] == pytest.approx(0.96378, abs=2e-5) and fit.params[1] == pytest.approx(22_349.4, rel=2e-4)
    xi, beta = fit.params[0], fit.params[1]
    fitted = h + beta / xi * (np.power(1.0 - np.array(levels), -xi) - 1.0)
    assert fitted == pytest.approx([27_409.0, 42_039.0, 85_025.0, 412_906.0], abs=1.5)

    def sse(x, b):
        model = b / x * (np.power(1.0 - np.array(levels), -x) - 1.0)
        return float(np.sum((np.log(model) - np.log(np.array(quantiles) - h)) ** 2))

    # it is the optimum of that objective: a perturbation raises the sum of squares
    assert sse(xi, beta) < min(sse(xi + 1e-3, beta), sse(xi - 1e-3, beta), sse(xi, beta * 1.001), sse(xi, beta * 0.999))
    assert fit.std_errors[0] > 0.0 and math.isfinite(fit.std_errors[1])


def test_gamma_prior_from_mean_interval_golden_and_refusals():
    prior = sc.gamma_prior_from_mean_interval(1.0, -math.log(0.95), -math.log(0.05), 0.9)
    assert prior.shape == pytest.approx(1.0, abs=1e-6) and prior.rate == pytest.approx(1.0, abs=1e-6)
    assert prior.mean == pytest.approx(1.0, abs=1e-12)
    assert prior.mean == pytest.approx(prior.shape / prior.rate, abs=1e-12)
    assert prior.interval[0] == pytest.approx(0.051293, abs=5e-7)
    assert prior.interval[1] == pytest.approx(2.995732, abs=5e-7)
    assert prior.interval_level == 0.9 and prior.kind == "unrestricted-gamma-on-pareto-shape"
    g = stats.gamma(prior.shape, scale=1.0 / prior.rate)
    assert g.cdf(prior.interval[1]) - g.cdf(prior.interval[0]) == pytest.approx(0.9, abs=1e-9)
    narrow = sc.gamma_prior_from_mean_interval(2.0, 1.5, 2.6, 0.9)
    assert narrow.shape > 1.0 and narrow.mean == 2.0
    for args in ((2.0, 2.5, 3.0), (2.0, 1.0, 1.5), (-1.0, 0.5, 1.5), (1.0, 0.9999, 1.0001)):
        with pytest.raises(ValueError):
            sc.gamma_prior_from_mean_interval(*args)
    with pytest.raises(ValueError):
        sc.gamma_prior_from_mean_interval(1.0, 0.5, 1.5, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        prior.shape = 3.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        sc.GammaPrior(shape=2.0, rate=1.0, mean=5.0, interval=(1.0, 3.0), interval_level=0.9)
    with pytest.raises(ValueError):
        sc.GammaPrior(shape=2.0, rate=1.0, mean=2.0, interval=(1.0, 3.0), interval_level=0.9, kind="truncated")


def test_gamma_prior_from_scenarios_is_deterministic_and_narrows_with_the_basis_years():
    pts = gpd_points(0.5, 100.0, 0.0, [150.0, 400.0, 1_500.0], 200.0)
    a = sc.gamma_prior_from_scenarios(pts, 0.0, 200.0, basis_years=10.0, n_bootstrap=300, seed=7)
    b = sc.gamma_prior_from_scenarios(pts, 0.0, 200.0, basis_years=10.0, n_bootstrap=300, seed=7)
    c = sc.gamma_prior_from_scenarios(pts, 0.0, 200.0, basis_years=10.0, n_bootstrap=300, seed=8)
    assert a == b and a != c
    fit = sc.fit_tail_to_scenarios(pts, 0.0, 200.0)
    assert a.mean == pytest.approx(1.0 / fit.params[0], rel=1e-12)       # alpha_s = 1 / xi_s is the prior mean
    assert a.interval[0] < a.mean < a.interval[1] and a.interval_level == 0.9
    assert a.kind == "unrestricted-gamma-on-pareto-shape"
    wide = sc.gamma_prior_from_scenarios(pts, 0.0, 200.0, basis_years=2.0, n_bootstrap=300, seed=7)
    tight = sc.gamma_prior_from_scenarios(pts, 0.0, 200.0, basis_years=100.0, n_bootstrap=300, seed=7)
    width = [(p.interval[1] - p.interval[0]) for p in (wide, a, tight)]
    assert width[0] > width[1] > width[2]
    assert wide.shape < a.shape < tight.shape
    g = stats.gamma(a.shape, scale=1.0 / a.rate)
    assert g.cdf(a.interval[1]) - g.cdf(a.interval[0]) == pytest.approx(0.9, abs=1e-9)
    # the unrestricted prior keeps a tail index of 1.1 (alpha 0.909) inside its support
    heavy = gpd_points(1.1, 100.0, 0.0, [150.0, 1_000.0, 20_000.0], 200.0)
    hp = sc.gamma_prior_from_scenarios(heavy, 0.0, 200.0, basis_years=40.0, n_bootstrap=300, seed=1)
    assert hp.mean == pytest.approx(1.0 / 1.1, rel=1e-6) and hp.mean < 1.0
    for kwargs in ({"basis_years": 0.0}, {"n_bootstrap": 1}, {"seed": -1}, {"interval_level": 1.0}):
        with pytest.raises(ValueError):
            sc.gamma_prior_from_scenarios(pts, 0.0, 200.0, **{"basis_years": 10.0, "n_bootstrap": 50, **kwargs})
    with pytest.raises(ValueError):
        sc.gamma_prior_from_scenarios(pts[:1], 0.0, 200.0, basis_years=10.0, n_bootstrap=50)


def test_credibility_pareto_golden():
    prior = sc.GammaPrior(shape=2.0, rate=1.0, mean=2.0, interval=(0.5, 5.0), interval_level=0.9)
    fit = sc.credibility_pareto(prior, losses_of_e(3), 1.0)
    assert fit.posterior_shape == 5.0 and fit.posterior_rate == pytest.approx(4.0, abs=1e-12)
    assert fit.alpha_mean == pytest.approx(1.25, abs=1e-12) and fit.alpha_mle == pytest.approx(1.0, abs=1e-12)
    assert fit.credibility_weight == pytest.approx(0.75, abs=1e-12)
    assert fit.alpha_mean == pytest.approx(0.75 * 1.0 + 0.25 * 2.0, abs=1e-12)    # equation 25
    assert math.sqrt(fit.posterior_shape) / fit.posterior_rate == pytest.approx(0.559017, abs=5e-7)
    assert fit.xi == pytest.approx(0.8, abs=1e-12) and fit.beta == pytest.approx(0.8, abs=1e-12) and fit.location == 1.0
    assert fit.infinite_mean is False and fit.n == 3 and fit.prior is prior
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.xi = 1.0  # type: ignore[misc]


def test_credibility_pareto_limits_and_the_infinite_mean_flag():
    prior = sc.GammaPrior(shape=2.0, rate=1.0, mean=2.0, interval=(0.5, 5.0), interval_level=0.9)
    none = sc.credibility_pareto(prior, np.array([]), 5.0)
    assert none.alpha_mean == 2.0 and none.credibility_weight == 0.0 and none.alpha_mle is None and none.n == 0
    assert none.xi == 0.5 and none.beta == 2.5 and none.location == 5.0
    # as N grows the weight tends to 1 and the posterior mean to the maximum likelihood estimate
    rng = np.random.default_rng(2)
    big_l, alpha_true = 100.0, 0.8
    x = big_l * (1.0 - rng.random(20_000)) ** (-1.0 / alpha_true)
    small = sc.credibility_pareto(prior, x[:20], big_l)
    large = sc.credibility_pareto(prior, x, big_l)
    assert 0.0 < small.credibility_weight < large.credibility_weight < 1.0 and large.credibility_weight > 0.99
    assert abs(large.alpha_mean - large.alpha_mle) < abs(small.alpha_mean - small.alpha_mle)
    assert large.alpha_mean == pytest.approx(alpha_true, rel=0.03)
    # the prior is unrestricted: a posterior mean of 1 or below is kept and flagged, and no mean follows
    assert large.infinite_mean is True and large.xi > 1.0
    with pytest.raises(ValueError):
        sc.credibility_pareto(prior, losses_of_e(3), 10.0)                 # a loss below the threshold
    with pytest.raises(ValueError):
        sc.credibility_pareto(prior, np.full(3, 5.0), 5.0)                 # every loss equals L: no information
    with pytest.raises(ValueError):
        sc.credibility_pareto(prior, losses_of_e(3), 0.0)
    with pytest.raises(ValueError):
        sc.credibility_pareto("prior", losses_of_e(3), 1.0)                # type: ignore[arg-type]
