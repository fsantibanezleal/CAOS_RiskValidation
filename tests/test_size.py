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
# tests whose measured size exceeds the bound in their null scenario: the page states the measured size
DOCUMENTED_EXCESS = {"pd.normal_multiperiod": ROOT / "docs" / "tests" / "01_pd-calibration.md"}


def test_size_of_every_p_value_test():
    p_value_tests = {s.test_id for s in catalogue() if s.decision == "p_value" and s.family != "selftest"}
    assert p_value_tests == set(NULL_SCENARIOS), p_value_tests ^ set(NULL_SCENARIOS)
    seed = 5000
    measured = {}
    for tid, (gen, feed, kw) in sorted(NULL_SCENARIOS.items()):
        seed += 1
        sim = simulate(tid, gen, n_rep=N_REP, seed=seed, feed=feed, test_kwargs=kw)
        measured[tid] = sim.rate(0.05)
        holds = within_size(sim.rate(0.05), 0.05) and within_size(sim.rate(0.01), 0.01)
        if tid in DOCUMENTED_EXCESS:
            assert not holds, f"{tid} now holds its size: update its page"
            text = DOCUMENTED_EXCESS[tid].read_text(encoding="utf-8")
            assert "measured size" in text and f"{100 * sim.rate(0.05).rate:.1f}%" in text, tid
        else:
            assert holds, (tid, sim.rate(0.05).to_dict(), sim.rate(0.01).to_dict())
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
