"""RV-301 to RV-305: the generators' contract and their exact truths, each held to a large draw on its own stream."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import special, stats

from riskvalidation.generators import (
    BinormalScores,
    DefaultCounts,
    GradeFrequencies,
    LogisticPortfolio,
    Migrations,
    NestedLda,
    TwoSample,
    binormal_auc,
    binormal_auc_variance,
    break_monotonicity,
    concentrate,
    decaying_matrix,
    default_correlation,
    monotone_cells,
)
from riskvalidation.validation.discrimination import auc_with_variance
from riskvalidation.validation.rating_system import concentration

GENERATORS = [
    DefaultCounts(obligors=(300, 800, 1500), pd_true=(0.002, 0.01, 0.05)),
    DefaultCounts.over_years([1000] * 5, [0.03] * 5, rho=0.05, theta=0.2),
    BinormalScores(50, 950, separation=1.0, challenger_separation=0.8, development_size=1.0, leak=0.5),
    LogisticPortfolio(n=500, shift=0.2, calibration_intercept=-0.3, rho=0.1),
    LogisticPortfolio(n=500, quadratic=0.4),
    TwoSample(200, 300, shift=0.25),
    TwoSample(200, 300, binning="sample"),
    Migrations((100, 200, 300), decaying_matrix(3)),
    GradeFrequencies(1000, (0.2, 0.5, 0.3), (0.3, 0.4, 0.3)),
    NestedLda(n_cases=60, n_noncases=600),
]


def _equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


@pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: type(g).__name__)
def test_generator_contract(gen):
    cfg = gen.config()
    assert cfg["generator"] == gen.name and "riskvalidation" in cfg
    json.dumps(cfg)  # JSON-safe
    json.dumps(gen.truth(), default=str)
    a = gen.draw(np.random.default_rng(np.random.SeedSequence(7)))
    b = gen.draw(np.random.default_rng(np.random.SeedSequence(7)))
    c = gen.draw(np.random.default_rng(np.random.SeedSequence(8)))
    assert a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    assert any(not _equal(a[k], c[k]) for k in a if isinstance(a[k], np.ndarray) and a[k].size > 1)
    with pytest.raises(AttributeError):
        gen.name = "other"  # type: ignore[misc]


def test_generators_refuse_bad_parameters():
    for bad in (
        lambda: DefaultCounts(1000, 1.2),
        lambda: DefaultCounts(1000, 0.01, rho=1.0),
        lambda: DefaultCounts.over_years([10, 10], [0.01, 0.02], theta=1.0),
        lambda: BinormalScores(1, 10),
        lambda: LogisticPortfolio(correlation=-0.9),
        lambda: TwoSample(10, 10, binning="other"),
        lambda: Migrations((10, 10), ((0.5, 0.6), (0.5, 0.5))),
        lambda: GradeFrequencies(10, (0.5, 0.6)),
        lambda: GradeFrequencies(10, (0.5, 0.5), development_obligors=0),
        lambda: break_monotonicity(decaying_matrix(4), 2, 1, 2.0),
    ):
        with pytest.raises(ValueError):
            bad()


def test_default_counts_follow_wp14_model():
    """RV-302: the WP14 model's counts, variances and factor correlations on the generator's own stream."""
    for gen in (DefaultCounts.over_years([1000] * 5, [0.03] * 5, rho=0.05, theta=0.2),
                DefaultCounts.over_years([1000] * 5, (0.01, 0.02, 0.03, 0.04, 0.06),
                                         rho=(0.05, 0.06, 0.07, 0.08, 0.09), theta=0.2),
                DefaultCounts(obligors=(300, 800, 1500), pd_true=(0.002, 0.01, 0.05), rho=0.12)):
        checks = gen.verify(n_draws=20_000)
        assert all(c.ok() for c in checks), [c.to_dict() for c in checks if not c.ok()]
    # without correlation the count variance is binomial; with it the one-factor variance exceeds it
    tr = DefaultCounts(1000, 0.01, rho=0.05).truth()
    assert tr["variance_defaults"][0][0] > 1000 * 0.01 * 0.99
    assert DefaultCounts(1000, 0.01).truth()["variance_defaults"][0][0] == pytest.approx(1000 * 0.01 * 0.99)


def test_default_counts_draw_is_the_one_factor_panel():
    """The draw consumes the stream as factors first, then binomials (C05's generator, kept identical)."""
    n, pd, rho = np.array([5, 10, 20]), np.array([0.0005, 0.002, 0.0055]), 0.12
    gen = DefaultCounts(obligors=np.tile(n, (50, 1)), pd_true=np.tile(pd, (50, 1)), rho=rho)
    draw = gen.draw(np.random.default_rng(3))
    rng = np.random.default_rng(3)
    z = rng.standard_normal(50)
    cond = special.ndtr((special.ndtri(pd)[None, :] - math.sqrt(rho) * z[:, None]) / math.sqrt(1.0 - rho))
    assert np.array_equal(draw["d"], rng.binomial(n[None, :], cond))
    assert np.array_equal(draw["factor"], z)


def test_default_correlation_wp14_examples():
    """RV-302: the default correlations next to the asset correlations in WP14's binomial-test examples."""
    printed = {0.01: (0.41, 0.94, 1.60, 2.41), 0.005: (0.25, 0.58, 1.03, 1.60), 0.05: (1.20, 2.55, 4.08, 5.78)}
    for pd, col in printed.items():  # Examples 1, 3 and 5 share PD 1%; Example 2 PD 0.5%; Example 4 PD 5%
        ours = [round(100 * default_correlation(pd, r), 2) for r in (0.05, 0.10, 0.15, 0.20)]
        assert ours == list(col), (pd, ours)
    assert default_correlation(0.01, 0.0) == 0.0


def test_binormal_truth():
    """RV-303: the AUC Phi(delta / sqrt 2) and the exact variance of the Mann-Whitney AUC."""
    assert binormal_auc(0.0) == 0.5
    assert binormal_auc(1.2) == pytest.approx(stats.norm.cdf(1.2 / math.sqrt(2.0)))
    # the variance formula against a direct computation of Var(U) for a tiny case by its definition
    assert binormal_auc_variance(1.0, 1, 1) == pytest.approx(binormal_auc(1.0) * (1 - binormal_auc(1.0)))
    for gen in (BinormalScores(150, 4850, separation=1.2),
                BinormalScores(80, 1920, separation=1.0, challenger_separation=0.7, development_size=2.0),
                BinormalScores(80, 1920, separation=1.0, leak=0.8, development_size=1.0)):
        checks = gen.verify(n_draws=2_000)
        assert all(c.ok() for c in checks), [c.to_dict() for c in checks if not c.ok()]
    leak = BinormalScores(100, 900, separation=1.0, leak=0.8).truth()
    assert leak["auc_development"] == pytest.approx(stats.norm.cdf(1.8 / 2.0))
    assert leak["auc_current"] == pytest.approx(stats.norm.cdf(1.0 / 2.0))
    assert leak["auc_current"] < leak["auc_genuine_current"]  # the noise feature hurts the model now
    assert sum(leak["feature_bins_development"]) == pytest.approx(1.0)
    assert not leak["discrimination_null_holds"]
    assert BinormalScores(100, 900, separation=1.0, leak=0.0).truth()["discrimination_null_holds"]


def test_logistic_portfolio_truth():
    """RV-303: default rate, mean PDs and the model's Brier score; the AUC of the linear predictor."""
    for gen in (LogisticPortfolio(n=5000), LogisticPortfolio(n=5000, calibration_intercept=-0.4, shift=0.3),
                LogisticPortfolio(n=5000, quadratic=0.5, rho=0.08)):
        checks = gen.verify(n_draws=1_000)
        assert all(c.ok() for c in checks), [c.to_dict() for c in checks if not c.ok()]
    gen = LogisticPortfolio(n=200_000)
    d = gen.draw(np.random.default_rng(np.random.SeedSequence(21)))
    auc, s2, _, _ = auc_with_variance(d["risk"], d["defaults"])
    assert abs(auc - gen.truth()["auc_model"]) < 4.0 * math.sqrt(s2)
    tr = gen.truth()
    assert tr["calibration_null_holds"] and tr["mean_pd_model"] == pytest.approx(tr["mean_pd_true"])
    under = LogisticPortfolio(n=10, calibration_intercept=-math.log(1.5)).truth()
    assert under["mean_pd_model"] < under["mean_pd_true"] and not under["calibration_null_holds"]


def test_two_sample_truth():
    """RV-304: Yurdakul and Naranjo's design, population deciles; the bin probabilities of both laws."""
    gen = TwoSample(400, 400, shift=0.5)
    tr = gen.truth()
    assert tr["bins_base"] == pytest.approx([0.1] * 10)
    assert sum(tr["bins_target"]) == pytest.approx(1.0) and tr["bins_target"][0] < 0.1 < tr["bins_target"][-1]
    assert tr["population_psi"] > 0 and tr["wasserstein_1"] == pytest.approx(4.0)
    checks = gen.verify(n_draws=4_000)
    assert all(c.ok() for c in checks), [c.to_dict() for c in checks if not c.ok()]
    assert TwoSample(400, 400).truth()["population_psi"] == pytest.approx(0.0, abs=1e-15)
    d = TwoSample(400, 400, binning="sample").draw(np.random.default_rng(1))
    assert np.all(np.abs(d["expected_counts"] - 40) <= 1)  # the base sample's own deciles


def test_rating_generators_truth():
    """RV-305: the decaying matrix, the planted cell, the multinomial draws and the concentration truths."""
    m = decaying_matrix(7)
    assert np.allclose(m.sum(axis=1), 1.0)
    assert monotone_cells(m) == {"inside": 42, "boundary": 0, "outside": 0}
    b = break_monotonicity(m, 3, 1, 1.0)
    assert monotone_cells(b)["boundary"] == 1 and np.allclose(b.sum(axis=1), 1.0)
    assert monotone_cells(break_monotonicity(m, 3, 1, 2.0))["outside"] == 1
    gen = Migrations((500,) * 7, b)
    assert gen.truth()["null_holds"]
    checks = gen.verify(n_draws=2_000)
    assert all(c.ok() for c in checks), [c.to_dict() for c in checks if not c.ok()]
    p = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.18, 0.13, 0.09, 0.05, 0.03])
    c = concentrate(p, 0.2)
    assert c.sum() == pytest.approx(1.0) and concentration(c)[1] > concentration(p)[1]
    gf = GradeFrequencies(5000, c, p)
    assert not gf.truth()["null_holds"] and GradeFrequencies(5000, p).truth()["null_holds"]
    checks = gf.verify(n_draws=2_000)
    assert all(ch.ok() for ch in checks), [ch.to_dict() for ch in checks if not ch.ok()]
    # a development sample from the initial distribution, for two-sample tests of the counts
    two = GradeFrequencies(5000, c, p, development_obligors=4000).draw(np.random.default_rng(5))
    assert two["expected_counts"].sum() == 4000 and np.array_equal(two["actual_counts"], two["freq_current"])
    one = GradeFrequencies(5000, c, p).draw(np.random.default_rng(5))
    assert np.array_equal(one["freq_current"], two["freq_current"]) and "expected_counts" not in one


def test_nested_lda_truth():
    """RV-310 support: the Framingham-like design; equal population AUCs under the null."""
    tr = NestedLda().truth()
    assert tr["null_holds"] and tr["auc_full"] == pytest.approx(tr["auc_reduced"])
    assert 0.75 < tr["auc_full"] < 0.85
    assert NestedLda(added_effect=0.25).truth()["auc_full"] > tr["auc_full"]


def test_rating_paths():
    """RV-412: exact Markov paths (the verifier), CEREP's stock-model snapshots, and each planted defect visible in the
    paths: momentum raises the downgrade hazard after a downgrade, a regime changes the rates after its start, and
    informative withdrawals take ratings out before they default."""
    from scipy.linalg import expm

    from riskvalidation.generators.paths import RatingPaths

    q = np.array([
        [-0.12, 0.10, 0.015, 0.004, 0.001, 0.0],
        [0.02, -0.15, 0.11, 0.015, 0.004, 0.001],
        [0.002, 0.04, -0.17, 0.11, 0.015, 0.003],
        [0.0, 0.004, 0.06, -0.22, 0.14, 0.016],
        [0.0, 0.0, 0.005, 0.08, -0.30, 0.215],
        [0.0] * 6,
    ])
    ob = (2000, 3000, 4000, 3000, 2000)
    base = RatingPaths(q, ob, horizon=5.0, snapshots=(0.0, 1.0, 2.0))
    assert all(c.ok() for c in base.verify())
    truth = base.truth()
    np.testing.assert_allclose(truth["one_year_matrix"], expm(q))
    np.testing.assert_allclose(truth["pd_by_grade"]["1"], expm(q)[:5, 5])
    d = base.draw(np.random.default_rng(3))
    assert d["steps"][0]["counts"].sum() + d["steps"][0]["withdrawn"].sum() == sum(ob)
    # nobody withdrawn: every rating is seen at all three dates
    assert d["triplets"].sum() == d["steps"][1]["counts"].sum() == sum(ob)
    assert json.dumps(base.config())

    def down_rate(seg, entered_by_downgrade):
        """Downgrades per unit of time spent in grades 1 to 4, by how the stay began."""
        st, to, start, stop, ob_ = seg["state"], seg["to"], seg["start"], seg["stop"], seg["obligor"]
        prev = np.full(st.size, -1)
        same = np.r_[False, ob_[1:] == ob_[:-1]]
        prev[same] = st[np.flatnonzero(same) - 1]
        by_down = same & (prev < st)
        sel = (st >= 1) & (st <= 4) & (by_down == entered_by_downgrade)
        return np.sum(to[sel] > st[sel]) / np.sum(stop[sel] - start[sel])

    mom = RatingPaths(q, ob, horizon=5.0, momentum=((1.0, 1.0), (1.0, 1.0)), investment_grades=3)
    dm = mom.draw(np.random.default_rng(4))
    assert down_rate(dm["segments"], True) > 1.5 * down_rate(dm["segments"], False)
    assert down_rate(d["segments"], True) < 1.3 * down_rate(d["segments"], False)  # no momentum in the Markov case

    q2 = q.copy()
    up = np.triu_indices(6, k=1)
    q2[up] *= 3.0
    q2[-1] = 0.0
    np.fill_diagonal(q2, 0.0)
    np.fill_diagonal(q2, -q2.sum(axis=1))
    reg = RatingPaths(q, ob, horizon=5.0, regimes=((2.5, q2),))
    sg = reg.draw(np.random.default_rng(5))["segments"]
    moved = sg["to"] >= 0
    early = np.sum(moved & (sg["to"] > sg["state"]) & (sg["stop"] < 2.5))
    late = np.sum(moved & (sg["to"] > sg["state"]) & (sg["stop"] >= 2.5))
    assert late > 1.8 * early

    plain = RatingPaths(q, ob, horizon=5.0, withdrawal_rate=0.05)
    informative = RatingPaths(q, ob, horizon=5.0, withdrawal_rate=0.05, informative=20.0, window=1.0)
    n_def = [np.sum(g.draw(np.random.default_rng(6))["segments"]["to"] == 5) for g in (plain, informative)]
    assert n_def[1] < 0.8 * n_def[0]
    with pytest.raises(ValueError):
        RatingPaths(q, (1, 2, 3), horizon=5.0)
    with pytest.raises(ValueError):
        RatingPaths(q, ob, momentum=((1.0, 1.0), (1.0, 1.0)))
