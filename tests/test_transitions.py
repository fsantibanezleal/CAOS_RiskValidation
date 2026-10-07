"""U4: the transition estimators, the embedding problem, the EM generator, the registered tests, intervals for
PDs by grade, mobility and the TTC portfolio (RV-401 to RV-411)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from riskvalidation.generators.paths import RatingPaths
from riskvalidation.transitions import embedding as emb
from riskvalidation.transitions.estimators import aalen_johansen, cohort, duration_generator, pooled_cohort

# a six-state generator (five grades and default) with a rare direct jump from the top grade to default
Q6 = np.array([
    [-0.12, 0.10, 0.015, 0.004, 0.00099, 0.00001],
    [0.02, -0.15, 0.11, 0.015, 0.004, 0.001],
    [0.002, 0.04, -0.17, 0.11, 0.015, 0.003],
    [0.0, 0.004, 0.06, -0.22, 0.14, 0.016],
    [0.0, 0.0, 0.005, 0.08, -0.30, 0.215],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])


def test_cohort_estimator():
    """RV-401: rows normalised, withdrawals removed (industry) or kept as a state, default absorbing, pooling is
    Anderson and Goodman's (2.8)."""
    counts = [[90, 8, 2], [5, 80, 15], [3, 1, 0]]  # the default row as a source may report it (reinstatements)
    withdrawn = [10, 20, 0]
    ex = cohort(counts, withdrawn)
    np.testing.assert_allclose(ex["matrix"][0], [0.90, 0.08, 0.02])
    np.testing.assert_allclose(ex["matrix"][1], [0.05, 0.80, 0.15])
    np.testing.assert_allclose(ex["matrix"][2], [0.0, 0.0, 1.0])  # default absorbing whatever was reported
    np.testing.assert_allclose(ex["withdrawn_share"][:2], [10 / 110, 20 / 120])
    st = cohort(counts, withdrawn, treatment="state")
    assert st["matrix"].shape == (4, 4)
    np.testing.assert_allclose(st["matrix"][0], [90 / 110, 8 / 110, 2 / 110, 10 / 110])
    np.testing.assert_allclose(st["matrix"][3], [0, 0, 0, 1])
    # removing withdrawals spreads their mass in proportion to the row, as the industry treatment intends
    np.testing.assert_allclose(st["matrix"][0, :3] / st["matrix"][0, :3].sum(), ex["matrix"][0])
    empty = cohort([[0, 0, 0], [5, 80, 15], [0, 0, 1]])
    assert empty["empty_rows"] == [0] and np.all(np.isnan(empty["matrix"][0]))
    pooled = pooled_cohort([[[50, 5, 1], [2, 40, 8], [0, 0, 1]], [[40, 3, 1], [3, 40, 7], [0, 0, 1]]])
    np.testing.assert_allclose(pooled["matrix"][0], np.array([90, 8, 2]) / 100)
    assert pooled["periods"] == 2
    with pytest.raises(ValueError):
        cohort([[1, -1], [0, 1]])
    with pytest.raises(ValueError):
        cohort([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        pooled_cohort([[[1, 0], [0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])


def test_duration_and_aalen_johansen_recover_truth():
    """RV-402: the duration generator and the Aalen-Johansen matrix recover a known chain; the cohort estimator puts
    zero on a cell the truth makes positive when no obligor made that move within the year."""
    gen = RatingPaths(Q6, (3000, 3000, 3000, 3000, 3000), horizon=5.0, snapshots=(0.0, 1.0))
    d = gen.draw(np.random.default_rng(7))
    dur = duration_generator(d["segments"], 6)
    s = dur["holding"]
    for i in range(5):
        for j in range(6):
            if i == j or Q6[i, j] * s[i] < 100:
                continue
            se = math.sqrt(Q6[i, j] * s[i]) / s[i]
            assert abs(dur["generator"][i, j] - Q6[i, j]) <= 4 * se, (i, j)
    assert np.allclose(dur["generator"].sum(axis=1), 0.0)
    truth = expm(Q6)
    aj = aalen_johansen(d["segments"], 6, 0.0, 1.0)["matrix"]
    n0 = np.bincount(d["snapshot_states"][:, 0], minlength=6)
    for i in range(5):
        se = np.sqrt(truth[i] * (1 - truth[i]) / n0[i])
        assert np.all(np.abs(aj[i] - truth[i]) <= 4 * se + 1e-4), i
    co = cohort(d["steps"][0]["counts"], d["steps"][0]["withdrawn"])["matrix"]
    # top grade to default: about 3000 x 3e-5 = 0.09 expected in a year, almost surely none observed
    assert truth[0, 5] > 0 and co[0, 5] == 0.0
    assert expm(dur["generator"])[0, 5] > 0.0  # the duration estimate reaches it through intermediate moves


def test_embedding_conditions_and_adjustments():
    """RV-403: the series, Theorem 3's conditions, the adjustments, the JLT approximation and Lemma 1."""
    p = expm(Q6)
    ser = emb.log_series(p)
    assert ser["converged"] and ser["S"] < 1
    np.testing.assert_allclose(ser["generator"], Q6, atol=1e-10)
    diag = emb.embedding_diagnostics(p)
    assert not diag["exact_generator_excluded"] and not diag["series_negative_off_diagonal"]
    # a zero cell reachable through other states: Theorem 3(c)
    p0 = np.array([[0.90, 0.10, 0.00], [0.05, 0.90, 0.05], [0.0, 0.0, 1.0]])
    d0 = emb.embedding_diagnostics(p0)
    assert (0, 2) in d0["theorem3_c"] and d0["exact_generator_excluded"]
    # det <= 0: Theorem 3(a)
    assert emb.embedding_diagnostics([[0.4, 0.6], [0.6, 0.4]])["theorem3_a"]
    # the adjustments keep rows at 0 and off-diagonals non-negative; exp(Q) close to P
    for method in ("diagonal", "weighted"):
        g = emb.generator(p0, method=method)
        assert g["valid"] and g["l1_distance"] < 0.01
    # JLT, equation (3), by hand on the first row
    j = emb.jlt_generator(p0)
    assert j[0, 0] == pytest.approx(math.log(0.9))
    assert j[0, 1] == pytest.approx(0.10 * math.log(0.9) / (0.9 - 1.0))
    assert np.allclose(j.sum(axis=1), 0.0)
    # Lemma 1: a tail sum that falls from one row to the next
    pm = np.array([[0.8, 0.15, 0.05], [0.10, 0.85, 0.05], [0, 0, 1]])
    assert emb.embedding_diagnostics(pm)["stochastically_monotone"]
    bad = np.array([[0.8, 0.10, 0.10], [0.10, 0.85, 0.05], [0, 0, 1]])
    v = emb.monotonicity_violations(bad)
    assert any(x["row"] == 0 and x["column"] == 2 for x in v)
    with pytest.raises(ValueError):
        emb.generator([[0.5, 0.6], [0.5, 0.5]])
    with pytest.raises(ValueError):
        emb.generator(p0, method="cubic")


def test_em_generator():
    """RV-404: the EM generator of snapshot counts: a non-decreasing log-likelihood, closed-form expectations that
    agree with the simulated paths, and a known generator recovered from ten annual snapshots."""
    from riskvalidation.transitions.em import _blocks, em_generator

    gen = RatingPaths(Q6, (1000,) * 5, horizon=10.0, snapshots=tuple(float(x) for x in range(11)))
    # the E-step under the true generator against the jumps the paths made in the first year, over ten draws
    p, jb, _ = _blocks(Q6, 1.0)
    expected = np.zeros((6, 6))
    observed = np.zeros((6, 6))
    rng = np.random.default_rng(12)
    for _ in range(10):
        d = gen.draw(rng)
        c = d["steps"][0]["counts"]
        expected += Q6 * np.einsum("ijab,ab->ij", jb, np.divide(c, p, out=np.zeros_like(c), where=c > 0))
        seg = d["segments"]
        first = (seg["to"] >= 0) & (seg["stop"] <= 1.0)
        np.add.at(observed, (seg["state"][first], seg["to"][first]), 1.0)
    for i in range(5):
        for j in range(6):
            if i != j and expected[i, j] >= 50:
                assert abs(observed[i, j] - expected[i, j]) <= 4 * math.sqrt(expected[i, j]), (i, j)
    d = gen.draw(np.random.default_rng(11))
    fit = em_generator([{"counts": s["counts"], "dt": 1.0} for s in d["steps"]])
    assert fit["converged"] and np.all(np.diff(fit["loglik"]) >= -1e-8)
    assert emb.is_generator(fit["generator"])
    pd1, true1 = expm(fit["generator"])[:5, 5], expm(Q6)[:5, 5]
    assert np.all(np.abs(pd1[2:] - true1[2:]) / true1[2:] < 0.15)
    with pytest.raises(ValueError):
        em_generator([])


def _snapshots(**kw):
    return RatingPaths(Q6, (1000,) * 5, horizon=5.0, snapshots=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0), **kw)


def test_time_homogeneity():
    """RV-405: Anderson and Goodman's chi-square (3.8) and likelihood ratio (3.5) across periods: both reject a
    recession regime that triples every downgrade rate from the third year; the degrees of freedom leave out the empty
    cells."""
    from riskvalidation.validation.transitions import rating_time_homogeneity

    q2 = Q6.copy()
    q2[np.triu_indices(6, k=1)] *= 3.0
    q2[-1] = 0.0
    np.fill_diagonal(q2, 0.0)
    np.fill_diagonal(q2, -q2.sum(axis=1))
    regime = _snapshots(regimes=((2.0, q2),)).draw(np.random.default_rng(31))
    r = rating_time_homogeneity([s["counts"] for s in regime["steps"]])
    assert r.p_value < 1e-6 and r.extras["form"] == "chi2" and r.p_value == r.extras["chi2_p_value"]
    lr = rating_time_homogeneity([s["counts"] for s in regime["steps"]], form="lr")
    assert lr.statistic == pytest.approx(r.extras["lr"]) and lr.p_value == pytest.approx(r.extras["lr_p_value"])
    assert lr.p_value < 1e-6
    # the default row carries no information and empty cells leave the degrees of freedom
    same = [[[50, 10, 0], [5, 40, 5], [0, 0, 7]], [[45, 15, 0], [6, 38, 6], [0, 0, 9]]]
    r2 = rating_time_homogeneity(same)
    assert r2.extras["dof"] == (2 - 1) * (2 - 1) + (2 - 1) * (3 - 1)
    with pytest.raises(ValueError):
        rating_time_homogeneity([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        rating_time_homogeneity(same, form="g2")


def test_markov_order():
    """RV-406: the order test on the states grouped by direction sees downgrade momentum; the full table is in
    extras."""
    from riskvalidation.validation.transitions import rating_markov_order

    d = _snapshots(momentum=((0.5, 2.0), (0.5, 2.0)), investment_grades=3).draw(np.random.default_rng(32))
    r = rating_markov_order(d["triplets"])
    assert r.p_value < 1e-6 and r.extras["grouping"] == "direction"
    assert {"lr", "chi2", "dof"} <= set(r.extras["full"])
    lr = rating_markov_order(d["triplets"], form="lr", grouping="none")
    assert lr.extras["form"] == "lr" and lr.extras["dof"] == lr.extras["full"]["dof"]
    with pytest.raises(ValueError):
        rating_markov_order(d["triplets"], grouping="pairs")


def test_matrix_against_reference():
    """RV-407: a count matrix against a reference: the true matrix is not rejected in this draw, a matrix of
    half the mobility is, and a move the reference forbids rejects outright."""
    from riskvalidation.validation.transitions import rating_matrix_reference

    d = _snapshots().draw(np.random.default_rng(33))
    counts = d["steps"][0]["counts"]
    p = expm(Q6)
    assert rating_matrix_reference(counts, p).p_value > 0.01
    assert rating_matrix_reference(counts, expm(0.5 * Q6)).p_value < 1e-6
    forbid = p.copy()
    forbid[0, 1] += forbid[0, 3]
    forbid[0, 3] = 0.0
    r = rating_matrix_reference(counts, forbid)
    if counts[0, 3] > 0:
        assert r.p_value == 0.0 and (0, 3) in r.extras["impossible_moves"]
    chi2 = rating_matrix_reference(counts, p, form="chi2")
    assert chi2.extras["form"] == "chi2" and chi2.p_value == pytest.approx(chi2.extras["chi2_p_value"])


def test_momentum_hazard():
    """RV-408: the hazard test finds the planted downward momentum (a positive coefficient), and no upward effect."""
    from riskvalidation.validation.transitions import rating_momentum

    d = _snapshots(momentum=((1.0, 1.0), (1.0, 1.0)), investment_grades=3).draw(np.random.default_rng(34))
    down = rating_momentum(d["segments"])
    assert down.p_value < 1e-6 and down.extras["coefficient"] > 0.2
    up = rating_momentum(d["segments"], direction="up")
    assert up.extras["coefficient"] < down.extras["coefficient"]
    with pytest.raises(ValueError):
        rating_momentum(d["segments"], direction="sideways")



def test_pd_intervals():
    """RV-409: Wald (2.2), Agresti-Coull (3.3) with the dependence correction (3.4), Jeffreys, and the two bootstraps;
    each refuses an empty grade. Exact coverage by enumeration: with one expected default the Wald interval covers
    63%, the other two 98%; with twenty all three are near 95%."""
    from scipy import stats

    from riskvalidation.transitions import intervals as iv

    w = iv.pd_wald(10, 100)
    assert w["estimate"] == 0.1 and w["upper"] - 0.1 == pytest.approx(0.1 - w["lower"])
    assert w["rule_of_thumb"] is True and iv.pd_wald(3, 100)["rule_of_thumb"] is False
    assert iv.pd_wald(3, 100)["lower"] == 0.0  # 0.03 - 1.96 sqrt(0.03 x 0.97 / 100) < 0: bounded at 0, as printed
    zero = iv.pd_wald(0, 100)
    assert zero["lower"] == zero["upper"] == 0.0  # the Wald interval collapses at no default
    j = iv.pd_jeffreys([0, 3, 10], [100, 100, 10])
    assert j["lower"][0] == 0.0 and j["upper"][2] == 1.0
    assert j["upper"][0] == pytest.approx(stats.beta.ppf(0.975, 0.5, 100.5))
    ac = iv.pd_agresti_coull(0, 100)
    assert ac["lower"] == 0.0 and ac["upper"] > 0.0 and ac["estimate"] > 0.0
    # (3.4): N-dagger with unequal trials, and N / (1 + (N - 1) rho) with one trial each
    trials = [1.0, 2.0, 3.0]
    pairs = math.sqrt(2) + math.sqrt(3) + math.sqrt(6)
    assert iv.effective_n(6, 0.1, trials) == pytest.approx(1.0 / (1.0 / 6 + 2.0 * pairs * 0.1 / 36))
    assert iv.effective_n(531, 0.01) == pytest.approx(531 / (1 + 530 * 0.01))
    assert iv.pd_wald(3, 6, rho=0.1, trials=trials)["n_effective"] == pytest.approx(iv.effective_n(6, 0.1, trials))
    for f in (iv.pd_wald, iv.pd_agresti_coull, iv.pd_jeffreys):
        with pytest.raises(ValueError):
            f(0, 0)
        with pytest.raises(ValueError):
            f(5, 3)
    with pytest.raises(ValueError):
        iv.pd_wald([1, 2], [10, 10], rho=0.1, trials=[5, 5])
    expected = {(200, 0.005): (0.6325, 0.9813, 0.9813), (1000, 0.02): (0.9497, 0.9467, 0.9589)}
    for (n, p), covs in expected.items():
        d = np.arange(n + 1)
        weight = stats.binom.pmf(d, n, p)
        for f, c in zip((iv.pd_wald, iv.pd_agresti_coull, iv.pd_jeffreys), covs, strict=True):
            r = f(d, np.full(n + 1, n))
            assert float(np.sum(weight * ((r["lower"] <= p) & (p <= r["upper"])))) == pytest.approx(c, abs=5e-5)
    # the bootstraps of the duration PD: both cover the truth in every grade of this draw, with similar errors
    d = RatingPaths(Q6, (2000,) * 5, horizon=3.0).draw(np.random.default_rng(41))
    truth = expm(Q6)[:5, -1]
    res = iv.pd_bootstrap(d["segments"], 6, method="resample", n_boot=200, seed=7)
    par = iv.pd_bootstrap(d["segments"], 6, method="parametric", n_boot=200, seed=7, keep=True)
    for r in (res, par):
        assert np.all(r["lower"] < truth) and np.all(truth < r["upper"])
    np.testing.assert_allclose(res["estimate"], par["estimate"])
    assert np.all(np.abs(res["se"] / par["se"] - 1.0) < 0.3) and par["replicates"].shape == (200, 5)
    again = iv.pd_bootstrap(d["segments"], 6, method="resample", n_boot=200, seed=7)
    np.testing.assert_array_equal(again["lower"], res["lower"])  # the seed makes it reproducible
    empty = {k: v[d["segments"]["state"] != 0] for k, v in d["segments"].items()}
    with pytest.raises(ValueError):
        iv.pd_bootstrap(empty, 6)


def test_mobility():
    """RV-410: M_SVD and the trace index are zero on the identity; on two states both have a closed form; the trace
    index cannot tell mass moved one notch from mass spread over every grade, M_SVD can."""
    from riskvalidation.transitions.mobility import svd_mobility, trace_mobility

    assert svd_mobility(np.eye(8)) == 0.0 and trace_mobility(np.eye(8)) == 0.0
    a, b = 0.1, 0.2
    two = [[1 - a, a], [b, 1 - b]]  # P - I has rank one, singular value sqrt(2 (a^2 + b^2))
    assert svd_mobility(two) == pytest.approx(math.sqrt(2.0 * (a * a + b * b)) / 2.0)
    assert trace_mobility(two) == pytest.approx(a + b)
    n = 7
    notch = np.eye(n) * 0.9
    for i in range(n - 1):
        notch[i, i + 1] = 0.1
    notch[n - 1, n - 2] = 0.1
    spread = np.full((n, n), 0.1 / (n - 1))
    np.fill_diagonal(spread, 0.9)
    assert trace_mobility(notch) == pytest.approx(trace_mobility(spread)) == pytest.approx(0.7 / 6)
    assert svd_mobility(notch) == pytest.approx(0.1206046, abs=1e-7)
    assert svd_mobility(spread) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        svd_mobility([[1.0, 0.0]])


def test_ttc_portfolio():
    """RV-411: Engelmann's TTC portfolio is the fixed point of (10), reached from any start, in closed form on two
    grades; a non-primitive performing block or an origination into default is refused (Theorem 1)."""
    from riskvalidation.transitions import ttc

    a, b, p1, p2, o1 = 0.1, 0.2, 0.01, 0.05, 0.7
    t = [[1 - a - p1, a, p1], [b, 1 - b - p2, p2], [0.0, 0.0, 1.0]]
    r = ttc.ttc_portfolio(t, [o1, 1 - o1, 0.0])
    w1 = (b + p2 * o1) / (a + p1 * (1 - o1) + b + p2 * o1)  # the balance of flows between the two grades
    np.testing.assert_allclose(r["portfolio"], [w1, 1 - w1, 0.0], atol=1e-12)
    assert r["converged"] and r["eigenvector_distance"] < 1e-10
    assert r["default_rate"] == pytest.approx(w1 * p1 + (1 - w1) * p2)
    m = ttc.propagation_matrix(t, [o1, 1 - o1])
    np.testing.assert_allclose(r["portfolio"] @ m, r["portfolio"], atol=1e-13)
    np.testing.assert_allclose(m.sum(axis=1), 1.0)
    # a migration matrix built from Q6, any start: the projection reaches the TTC portfolio
    p = expm(Q6)
    o = [0.0, 0.3, 0.4, 0.3, 0.0]
    for start in ([1.0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0]):
        pr = ttc.project(p, start, o, 300)
        assert pr["portfolio"].shape == (301, 6) and np.allclose(pr["portfolio"].sum(axis=1), 1.0)
        assert pr["distance_to_ttc"][-1] < 1e-9 and pr["distance_to_ttc"][0] > 0.5
        assert pr["default_rate"][0] == pytest.approx(float(np.asarray(start) @ p[:5, -1]))
    stressed = ttc.project([p, expm(2 * Q6), p], [0.2] * 5, o)
    assert stressed["portfolio"].shape == (4, 6) and "ttc" not in stressed
    assert stressed["default_rate"][1] > stressed["default_rate"][0]
    # Theorem 1's conditions and Engelmann's counterexample
    with pytest.raises(ValueError):
        ttc.ttc_portfolio([[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0.5, 0.5, 0.0])
    assert not ttc.is_primitive([[0, 1], [1, 0]]) and ttc.is_primitive([[0.5, 0.5], [1, 0]])
    with pytest.raises(ValueError):
        ttc.ttc_portfolio(t, [0.5, 0.4, 0.1])
    with pytest.raises(ValueError):
        ttc.ttc_portfolio([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.5, 0.0, 0.5]], [0.5, 0.5])
    with pytest.raises(ValueError):
        ttc.ttc_portfolio([[0.9, 0.2, 0.0], [0.1, 0.8, 0.1], [0, 0, 1]], [0.5, 0.5])
