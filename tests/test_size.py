"""RV-311 and RV-312: the size of every registered test with a p-value, measured on a generator where its null holds
at the boundary with its assumptions met; and the Hosmer-Lemeshow degrees of freedom decided by measurement."""

from __future__ import annotations

from pathlib import Path

from riskvalidation.core.registry import catalogue
from riskvalidation.generators import (
    BinormalScores,
    DefaultCounts,
    GradeFrequencies,
    LogisticPortfolio,
    Migrations,
    TwoSample,
    break_monotonicity,
    decaying_matrix,
    feed_cell,
    feed_grades,
    feed_series,
)
from riskvalidation.generators.paths import RatingPaths
from riskvalidation.harness import simulate, size_bound, within_size

ROOT = Path(__file__).resolve().parents[1]
N_REP = 4000
# seven grades with 5 to 32 expected defaults each (the chi-square approximation's assumption met)
GRADES_N = (2500, 2000, 1500, 1500, 800, 400, 150)
GRADES_PD = (0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
# the same scale with a thin best grade: 0.6 expected defaults (WP14: slow convergence for small PDs)
THIN_N = (300, 800, 1500, 1500, 800, 400, 150)

# a Markov chain with five grades and default (U4): five annual snapshots in CEREP's stock model, the paths kept
Q6 = (
    (-0.12, 0.10, 0.015, 0.004, 0.00099, 0.00001),
    (0.02, -0.15, 0.11, 0.015, 0.004, 0.001),
    (0.002, 0.04, -0.17, 0.11, 0.015, 0.003),
    (0.0, 0.004, 0.06, -0.22, 0.14, 0.016),
    (0.0, 0.0, 0.005, 0.08, -0.30, 0.215),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
CHAIN = RatingPaths(Q6, (1000,) * 5, horizon=5.0, snapshots=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))


def _expm_q6():
    import numpy as np
    from scipy.linalg import expm

    return expm(np.asarray(Q6))


# test id -> (generator under the null, feed, keyword arguments)
NULL_SCENARIOS = {
    "pd.binomial": (DefaultCounts(1000, 0.01), feed_cell(), {}),
    "pd.jeffreys": (DefaultCounts(1000, 0.01), feed_cell(), {}),
    "pd.binomial_vasicek": (DefaultCounts(1000, 0.01, rho=0.12), feed_cell(), {"rho": 0.12}),
    "pd.chi2_grades": (DefaultCounts(GRADES_N, GRADES_PD), feed_grades(), {}),
    "pd.default_profile": (DefaultCounts(GRADES_N, GRADES_PD), feed_grades(), {"n_sim": 2000}),
    "pd.hosmer_lemeshow": (LogisticPortfolio(n=5000), None, {}),
    "pd.spiegelhalter": (LogisticPortfolio(n=5000), None, {}),
    "pd.normal_multiperiod": (DefaultCounts.over_years([1000] * 5, [0.03] * 5), feed_series(), {}),
    "pd.traffic_lights": (DefaultCounts.over_years([1000] * 5, [0.03] * 5), feed_series(), {}),
    "disc.auc_vs_initial": (BinormalScores(150, 4850, separation=1.2), None, {}),
    "disc.delong": (BinormalScores(150, 4850, separation=1.2, challenger_separation=1.2), None, {}),
    "stability.psi": (TwoSample(2000, 2000), None, {}),
    "stability.csi": (TwoSample(2000, 2000), None, {}),
    "stability.chi2": (TwoSample(2000, 2000), None, {}),
    "stability.ks": (TwoSample(2000, 2000), None, {}),
    "rating.migration_ztests": (Migrations([500] * 7, break_monotonicity(decaying_matrix(7), 3, 1, 1.0)), None, {}),
    "rating.hhi": (GradeFrequencies(5000, (0.02, 0.05, 0.10, 0.15, 0.20, 0.18, 0.13, 0.09, 0.05, 0.03)), None, {}),
    "rating.time_homogeneity": (CHAIN, lambda d: {"counts_by_period": [s["counts"] for s in d["steps"]]}, {}),
    "rating.markov_order": (CHAIN, lambda d: {"triplets": d["triplets"]}, {}),
    "rating.matrix_reference": (CHAIN, lambda d: {"counts": d["steps"][0]["counts"], "reference": _expm_q6()}, {}),
    "rating.momentum": (CHAIN, lambda d: {"paths": d["segments"]}, {}),
}
# each test's seed, fixed when it joins the study, so that adding a test never moves another test's measurement (the
# pages quote them); the U3 tests keep the seeds their sorted order gave them in 0.03.000
SEEDS = {
    "disc.auc_vs_initial": 5001, "disc.delong": 5002, "pd.binomial": 5003, "pd.binomial_vasicek": 5004,
    "pd.chi2_grades": 5005, "pd.default_profile": 5006, "pd.hosmer_lemeshow": 5007, "pd.jeffreys": 5008,
    "pd.normal_multiperiod": 5009, "pd.spiegelhalter": 5010, "pd.traffic_lights": 5011, "rating.hhi": 5012,
    "rating.migration_ztests": 5013, "stability.chi2": 5014, "stability.csi": 5015, "stability.ks": 5016,
    "stability.psi": 5017,
    # U4 (0.04.000)
    "rating.markov_order": 5018, "rating.matrix_reference": 5019, "rating.momentum": 5020,
    "rating.time_homogeneity": 5021,
}
# tests whose measured size exceeds the bound in their null scenario: the page states the measured size
DOCUMENTED_EXCESS = {"pd.normal_multiperiod": ROOT / "docs" / "tests" / "01_pd-calibration.md"}
# tests whose page states the measured size of their default form at 5% and at 1% (the form was chosen by it)
MARKOV_PAGE = ROOT / "docs" / "transitions" / "03_markov-tests.md"
DOCUMENTED_SIZE = {tid: MARKOV_PAGE for tid in ("rating.time_homogeneity", "rating.markov_order",
                                                "rating.matrix_reference", "rating.momentum")}


# registered p-value tests whose size is measured in a named suite instead of NULL_SCENARIOS (a `sizepower` study, or a
# size test of the CI suite such as the market backtests'): test id -> "tests/<path>.py::<test function>"; the suite's
# test must exist and name the test (by its id or by its function name, which is how the fairness suite labels its rows)
_FAIRNESS_SIZE = "tests/fairness/test_fairness_size_power.py::test_size_and_power_two_proportion_z_and_fisher"
_MARKET_SIZE = "tests/validation_market/test_market_size.py::"
SIZE_SUITES = {
    "fair.two_proportion_z": _FAIRNESS_SIZE,
    "fair.fisher_exact": _FAIRNESS_SIZE,
    "market.kupiec_pof": _MARKET_SIZE + "test_size_of_the_var_backtests",
    "market.christoffersen_independence": _MARKET_SIZE + "test_size_of_the_var_backtests",
    "market.christoffersen_conditional_coverage": _MARKET_SIZE + "test_size_of_the_var_backtests",
    "market.duration_test": _MARKET_SIZE + "test_size_of_the_duration_test",
    "market.acerbi_szekely": _MARKET_SIZE + "test_size_of_the_es_backtests",
    "market.du_escanciano_unconditional": _MARKET_SIZE + "test_size_of_the_es_backtests",
    "market.du_escanciano_conditional": _MARKET_SIZE + "test_size_of_the_es_backtests",
    "market.dynamic_quantile_test": _MARKET_SIZE + "test_size_of_the_dq_test_at_production_configurations",
}
# registered p-value tests whose size has no measurement in either place yet. Each is a finding, not a pass: the list
# is pinned (a test below fails when an entry is measured, so it leaves the list, and when a new p-value test is
# registered without a measurement), and the reason says where a measurement is or what is missing
SIZE_GAPS = {
    "ccr.pfe_exceedance_backtest": "no size study: the Kupiec likelihood ratio of the PFE exceedances is not simulated",
    "ifrs9.stage_migration_backtest": "no size study of the Anderson-Goodman chi-square on stage transitions",
    "lgd_ead.lgd_calibration_by_segment": "no size study of the paired t-test of mean LGD",
    "lgd_ead.lgd_calibration_over_time": "no size study of the Bonferroni-adjusted per-period t-tests",
    "lgd_ead.downturn_coverage": "no size study of the one-sided downturn t-test",
    "market.mcneil_frey_residual_test": "no size study of the bootstrap residual test",
    "portfolio.tail_backtest": "no size study of the binomial exceedance count",
    "portfolio.mc_convergence": "no size study of the Monte Carlo z-test against the anchor",
    "portfolio.crp_recursion_vs_mc": "no size study of the Kolmogorov distance test",
}


def _p_value_tests() -> set[str]:
    import riskvalidation.validation.aggregation  # noqa: F401  (registers the families of batch 3 and U5 to U12)
    import riskvalidation.validation.ccr  # noqa: F401
    import riskvalidation.validation.fairness  # noqa: F401
    import riskvalidation.validation.ifrs9  # noqa: F401
    import riskvalidation.validation.irrbb  # noqa: F401
    import riskvalidation.validation.lgd_ead  # noqa: F401
    import riskvalidation.validation.liquidity  # noqa: F401
    import riskvalidation.validation.market  # noqa: F401
    import riskvalidation.validation.oprisk  # noqa: F401
    import riskvalidation.validation.portfolio  # noqa: F401
    import riskvalidation.validation.stress  # noqa: F401

    return {s.test_id for s in catalogue() if s.decision == "p_value" and s.family != "selftest"}


def test_every_registered_p_value_test_has_its_size_measured_or_named_as_a_gap():
    """A registered p-value test is measured in NULL_SCENARIOS, in a named sizepower suite, or listed in SIZE_GAPS
    with its reason; a test in none of the three fails here with its id, never skipped."""
    p_value_tests = _p_value_tests()
    covered = set(NULL_SCENARIOS) | set(SIZE_SUITES) | set(SIZE_GAPS)
    assert not p_value_tests - covered, f"p-value tests with no size measurement: {sorted(p_value_tests - covered)}"
    assert covered <= p_value_tests, f"entries that are not registered p-value tests: {sorted(covered - p_value_tests)}"
    assert not set(NULL_SCENARIOS) & (set(SIZE_SUITES) | set(SIZE_GAPS))
    assert not set(SIZE_SUITES) & set(SIZE_GAPS), "a measured test is no gap"


def test_every_size_suite_exists_and_names_its_test():
    """Read from the source, not imported: the suite is a size study (a 4,000-repetition `sizepower` study or a CI size
    test) that is not executed here."""
    import ast

    registered = {s.test_id: s for s in catalogue()}
    for tid, target in sorted(SIZE_SUITES.items()):
        path, _, name = target.partition("::")
        file = ROOT / path
        assert file.is_file(), (tid, path)
        text = file.read_text(encoding="utf-8")
        tree = ast.parse(text)
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert name in funcs, (tid, target, "no such test")
        source = ast.get_source_segment(text, funcs[name])
        assert tid in source or registered[tid].func.__name__ in source, (tid, target, "does not name the test")


def test_size_of_every_p_value_test():
    p_value_tests = {s.test_id for s in catalogue() if s.decision == "p_value" and s.family != "selftest"}
    # the null scenarios of the older families; the p-value tests measured elsewhere are SIZE_SUITES and SIZE_GAPS
    assert set(NULL_SCENARIOS) <= p_value_tests, set(NULL_SCENARIOS) - p_value_tests
    assert set(NULL_SCENARIOS) == set(SEEDS), set(NULL_SCENARIOS) ^ set(SEEDS)
    assert len(set(SEEDS.values())) == len(SEEDS)
    measured = {}
    for tid, (gen, feed, kw) in sorted(NULL_SCENARIOS.items()):
        sim = simulate(tid, gen, n_rep=N_REP, seed=SEEDS[tid], feed=feed, test_kwargs=kw)
        measured[tid] = sim.rate(0.05)
        holds = within_size(sim.rate(0.05), 0.05) and within_size(sim.rate(0.01), 0.01)
        if tid in DOCUMENTED_EXCESS:
            assert not holds, f"{tid} now holds its size: update its page"
            text = DOCUMENTED_EXCESS[tid].read_text(encoding="utf-8")
            assert "measured size" in text and f"{100 * sim.rate(0.05).rate:.1f}%" in text, tid
        else:
            assert holds, (tid, sim.rate(0.05).to_dict(), sim.rate(0.01).to_dict())
        if tid in DOCUMENTED_SIZE:
            text = DOCUMENTED_SIZE[tid].read_text(encoding="utf-8")
            for level in (0.05, 0.01):
                assert f"{100 * sim.rate(level).rate:.2f}%" in text, (tid, level, "the page must state it")
    # the ECB concentration test never rejects at the boundary of its null (its statistic ignores the obligor count)
    assert measured["rating.hhi"].rate == 0.0


def test_hosmer_lemeshow_degrees_of_freedom_by_measurement():
    """RV-312: with PDs given before the outcomes, G degrees of freedom hold the size and G - 2 do not."""
    gen = LogisticPortfolio(n=5000)
    given = simulate("pd.hosmer_lemeshow", gen, n_rep=N_REP, seed=6001)
    fitted = simulate("pd.hosmer_lemeshow", gen, n_rep=N_REP, seed=6001, test_kwargs={"fitted": True})
    assert within_size(given.rate(0.05), 0.05) and within_size(given.rate(0.01), 0.01)
    assert fitted.rate(0.05).rate > size_bound(0.05, N_REP) and fitted.rate(0.01).rate > size_bound(0.01, N_REP)
    assert 0.09 < fitted.rate(0.05).rate < 0.13  # P(chi2_10 > chi2_8 at 95%) = 0.115


def test_chi2_grades_over_rejects_with_few_expected_defaults():
    """The chi-square test over grades exceeds its level when a grade expects under one default (WP14: "if the p_i
    are very small the rate of convergence to the chi-square distribution may be low"); the page states it."""
    sim = simulate("pd.chi2_grades", DefaultCounts(THIN_N, GRADES_PD), n_rep=N_REP, seed=6101, feed=feed_grades())
    assert sim.rate(0.01).rate > size_bound(0.01, N_REP)
    text = (ROOT / "docs" / "tests" / "01_pd-calibration.md").read_text(encoding="utf-8")
    assert f"{100 * sim.rate(0.01).rate:.2f}%" in text


def test_auc_vs_initial_with_an_estimated_development_auc():
    """The ECB AUC test treats the development AUC as known; estimated on a sample as large as the current one it
    over-rejects, and adding the estimate's variance (`auc_initial_variance`) holds the size."""

    def extended(d):
        return {"auc_initial": d["auc_initial"], "risk": d["risk"], "defaults": d["defaults"],
                "auc_initial_variance": d["development_auc_variance"]}

    gen = BinormalScores(150, 4850, separation=1.2, development_size=1.0)
    ecb = simulate("disc.auc_vs_initial", gen, n_rep=N_REP, seed=7410)
    ext = simulate("disc.auc_vs_initial", gen, n_rep=N_REP, seed=7410, feed=extended)
    assert ecb.rate(0.05).rate > size_bound(0.05, N_REP) and ecb.rate(0.01).rate > size_bound(0.01, N_REP)
    assert within_size(ext.rate(0.05), 0.05) and within_size(ext.rate(0.01), 0.01)
