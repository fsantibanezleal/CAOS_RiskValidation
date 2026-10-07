"""U4: the transition estimators and the embedding problem (RV-401 to RV-403)."""

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
