"""RV-306, RV-307, RV-314: the harness's streams, rates and intervals; exact rejection probabilities held to the
simulation; estimator performance."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from riskvalidation.generators import (
    BinormalScores,
    DefaultCounts,
    TwoSample,
    feed_cell,
    feed_grades,
    feed_series,
)
from riskvalidation.harness import (
    Rate,
    agrees,
    agrees_with_published,
    count_pmf,
    critical_count,
    estimator_performance,
    metric_above,
    rejection_probability,
    simulate,
    size_bound,
    traffic_lights_rejection_probability,
    wilson_interval,
    within_size,
    wp14_true_confidence,
)
from riskvalidation.validation.calibration import critical_default_count, default_count_sf
from riskvalidation.validation.discrimination import auc_with_variance


def test_streams_rates_and_intervals():
    # Wilson interval and Morris SE
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(0, 10)[1] == pytest.approx(0.27753, abs=1e-5)
    lo, hi = wilson_interval(50, 1000)
    assert lo < 0.05 < hi and lo == pytest.approx(0.03813, abs=1e-5)
    # the interval contains its rate for every k, with the bounds exact at 0 and n (at n = 4,000 the closed form's
    # subtraction left 5.4e-20 above a zero rate, issue #18)
    for n in (10, 999, 1000, 2000, 4000, 25_000, 200_000):
        assert wilson_interval(0, n)[0] == 0.0 and wilson_interval(n, n)[1] == 1.0
        for k in sorted({0, 1, 2, n // 7, n // 2, n - 2, n - 1, n}):
            lo_k, hi_k = wilson_interval(k, n)
            assert 0.0 <= lo_k <= k / n <= hi_k <= 1.0, (k, n)
    r = Rate(50, 1000)
    assert r.rate == 0.05 and r.se == pytest.approx(math.sqrt(0.05 * 0.95 / 1000))
    json.dumps(r.to_dict())
    assert size_bound(0.05, 4000) == pytest.approx(0.05 + 3.090232306167813 * math.sqrt(0.0475 / 4000))
    assert within_size(Rate(200, 4000), 0.05) and not within_size(Rate(300, 4000), 0.05)
    ok, z = agrees_with_published(Rate(500, 10_000), 0.05, 10_000)
    assert ok and z == 0.0
    # undefined results are counted and never rejections: the normal test has no statistic when every year's
    # difference is the same (three years of one obligor at PD 50%: probability 1/4)
    sim = simulate("pd.normal_multiperiod", DefaultCounts.over_years([1] * 3, [0.5] * 3), n_rep=4000, seed=5,
                   feed=feed_series())
    r5 = sim.rate(0.05)
    assert abs(r5.undefined / 4000 - 0.25) < 4 * math.sqrt(0.25 * 0.75 / 4000)
    assert np.isnan(sim.p_values).sum() == r5.undefined
    # each repetition has its own stream: repetition i re-runs alone to the same p-value
    gen = DefaultCounts(1000, 0.01)
    sim = simulate("pd.binomial", gen, n_rep=50, seed=99, feed=feed_cell())
    child = np.random.SeedSequence(99).spawn(50)[17]
    d = gen.draw(np.random.default_rng(child))
    from riskvalidation.validation.calibration import pd_binomial

    assert pd_binomial(**feed_cell()(d)).p_value == sim.p_values[17]
    # a test with its own randomness gets a fresh seed per repetition
    grades = DefaultCounts((300, 800, 1500), (0.002, 0.01, 0.05))
    sim = simulate("pd.default_profile", grades, n_rep=6, seed=1, feed=feed_grades(), test_kwargs={"n_sim": 1000})
    assert sim.seed_injected and sim.extras["cached_inputs"] == 0
    fixed = simulate("pd.default_profile", grades, n_rep=6, seed=1, feed=feed_grades(),
                     test_kwargs={"n_sim": 1000, "seed": 0})
    assert not fixed.seed_injected
    # rules beyond p-values, and the serialised record
    two = simulate("stability.psi", TwoSample(100, 100), n_rep=200, seed=3, rules=(metric_above(0.10),))
    assert "metric>0.1" in two.rates and two.rate("metric>0.1").rate > two.rate(0.05).rate
    rec = two.to_dict()
    json.dumps(rec)
    assert sum(rec["p_histogram"]) == 200 and rec["generator"]["generator"] == "two_sample"


@pytest.mark.parametrize(
    ("test", "kw"), [("pd.binomial", {}), ("pd.jeffreys", {}), ("pd.binomial_vasicek", {"rho": 0.05})]
)
def test_exact_rejection_matches_simulation(test, kw):
    """RV-307: the exact probability (critical count from the test, tail of the one-factor mixture) against the
    harness, cell by cell, each cell on its own seed."""
    seed = 100 + len(test)
    for rho_t, ratio in ((0.0, 1.0), (0.05, 1.0), (0.10, 1.0), (0.05, 1.5), (0.0, 1.3)):
        seed += 1
        gen = DefaultCounts(obligors=1000, pd_true=0.01 * ratio, pd_forecast=0.01, rho=rho_t)
        sim = simulate(test, gen, n_rep=4000, seed=seed, feed=feed_cell(), test_kwargs=kw)
        for a in (0.05, 0.01):
            ex = rejection_probability(test, 1000, 0.01, a, pd_true=0.01 * ratio, rho_true=rho_t, **kw)
            assert agrees(sim.rate(a), ex["probability"]), (test, rho_t, ratio, a, sim.rate(a).rate, ex)


def test_exact_count_distribution_and_wp14_binomial_statement():
    """The count distribution sums to one with the right mean; the independence-based binomial test of WP14 Example
    3 (PD 1%, N 1,000, q 99%: reject from 19 defaults) has a true size of 11% to 15% under asset correlation 5% to
    20% (WP14: 'can be much larger than the nominal level'), exactly."""
    pmf = count_pmf(1000, 0.01, 0.12)
    assert pmf.sum() == pytest.approx(1.0) and np.arange(1001) @ pmf == pytest.approx(10.0)
    assert critical_count("pd.binomial", 1000, 0.01, 0.01) == critical_default_count(1000, 0.01, 0.99) == 19
    sizes = [rejection_probability("pd.binomial", 1000, 0.01, 0.01, rho_true=r)["probability"]
             for r in (0.0, 0.05, 0.10, 0.15, 0.20)]
    assert sizes[0] == pytest.approx(default_count_sf(19, 1000, 0.01)) and sizes[0] < 0.01
    assert [round(s, 3) for s in sizes[1:]] == [0.111, 0.144, 0.154, 0.155]
    # WP14 Figure 9: the large-portfolio true confidence of a 99.9% binomial test under the Basel corporate
    # correlation lies in the figure's bands (70% to 100%) over its grid of PD 0.5% to 9.5% and pools 100 to 1,900
    for pd in (0.005, 0.015, 0.035, 0.055, 0.075, 0.095):
        w = (1.0 - math.exp(-50.0 * pd)) / (1.0 - math.exp(-50.0))
        rho = 0.12 * w + 0.24 * (1.0 - w)
        for n in (100, 400, 700, 1000, 1300, 1600, 1900):
            assert 0.70 <= wp14_true_confidence(pd, n, rho, 0.999) < 1.0


def test_exact_traffic_lights_matches_simulation():
    """RV-307 extended: the traffic-lights test with independent years, exact by convolution, against the harness."""
    cases = (("lower", (0.03,) * 5, (0.03,) * 5), ("upper", (0.003,) * 5, (0.003,) * 5),
             ("upper", (0.01, 0.02, 0.03, 0.04, 0.06), (0.015, 0.025, 0.035, 0.045, 0.065)))
    for seed, (ties, fore, true) in enumerate(cases, start=300):
        gen = DefaultCounts.over_years([1000] * 5, true, pd_forecast=fore)
        sim = simulate("pd.traffic_lights", gen, n_rep=4000, seed=seed, feed=feed_series(), alphas=(0.1, 0.05),
                       test_kwargs={"ties": ties})
        for a in (0.1, 0.05):
            ex = traffic_lights_rejection_probability([1000] * 5, fore, a, pd_true=true, ties=ties)["probability"]
            assert agrees(sim.rate(a), ex), (ties, fore, a, sim.rate(a).rate, ex)


def test_estimator_performance():
    """RV-314: the measures of Morris et al. (2019, Table 6) on the AUC of binormal scores: unbiased, the ECB standard
    error equal to the exact one, the empirical SE consistent with it, coverage of the 95% interval near 95%."""
    gen = BinormalScores(500, 4500, separation=1.2)
    tr = gen.truth()
    est, se = [], []
    for child in np.random.SeedSequence(41).spawn(2000):
        d = gen.draw(np.random.default_rng(child))
        a, s2, _, _ = auc_with_variance(d["risk"], d["defaults"])
        est.append(a)
        se.append(math.sqrt(s2))
    est_a, se_a = np.array(est), np.array(se)
    perf = estimator_performance(est_a, tr["auc_current"], model_se=se_a, lower=est_a - 1.96 * se_a,
                                 upper=est_a + 1.96 * se_a)
    assert abs(perf["bias"]) < 4 * perf["bias_mcse"]
    exact_se = math.sqrt(tr["auc_variance_current"])
    assert perf["model_se"] == pytest.approx(exact_se, rel=0.01)
    assert abs(perf["empirical_se"] - exact_se) < 4 * perf["empirical_se_mcse"]
    assert abs(perf["coverage"] - 0.95) < 4 * perf["coverage_mcse"]
    # the MSE (Table 6) and its root: MSE = bias^2 + (n - 1)/n EmpSE^2 exactly; the MSE's Monte Carlo SE is the
    # standard error of a mean of squared errors; the root's, the delta method's MCSE(MSE) / (2 RMSE)
    err2 = (est_a - tr["auc_current"]) ** 2
    n = est_a.size
    assert perf["mse"] == pytest.approx(err2.mean(), rel=1e-12)
    assert perf["mse"] == pytest.approx(perf["bias"] ** 2 + (n - 1) / n * perf["empirical_se"] ** 2, rel=1e-9)
    assert perf["mse_mcse"] == pytest.approx(err2.std(ddof=1) / math.sqrt(n), rel=1e-9)
    assert perf["rmse"] == pytest.approx(math.sqrt(perf["mse"]), rel=1e-12)
    assert perf["rmse_mcse"] == pytest.approx(perf["mse_mcse"] / (2 * perf["rmse"]), rel=1e-12)
    exact = estimator_performance([0.3, 0.3, 0.3], 0.3)
    assert exact["mse"] == exact["rmse"] == exact["mse_mcse"] == exact["rmse_mcse"] == 0.0
    json.dumps(perf)


def test_harness_accepts_a_plain_callable():
    """A product's own wrapper of a registered test (for example the portfolio row of the Jeffreys grades) runs too."""
    from riskvalidation.validation.calibration import pd_jeffreys_grades

    def portfolio_jeffreys(n, d, pd):
        return pd_jeffreys_grades(n, d, pd)[-1]

    sim = simulate(portfolio_jeffreys, DefaultCounts((300, 800, 1500), (0.002, 0.01, 0.05)), n_rep=500, seed=2,
                   feed=feed_grades())
    assert sim.test_id == "pd.jeffreys" and 0.0 <= sim.rate(0.05).rate < 0.15
