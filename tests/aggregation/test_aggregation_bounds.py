"""RV-1313 to RV-1315: the comonotonic VaR, the rearrangement algorithm and the worst and best VaR bounds (contract
`engine-aggregation-api`, module aggregation.bounds; Embrechts, Puccetti and Ruschendorf 2013, Tables 1 and 4)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.aggregation import bounds as b
from riskvalidation.aggregation import copulas as c


def pareto2_ppf(u):
    """F(x) = 1 - (1 + x)^-2, so F^-1(u) = (1 - u)^(-1/2) - 1 (equation 11 of the paper with theta = 2)."""
    with np.errstate(divide="ignore"):  # F^-1(1) is +infinity, the last cell of the upper matrix
        return (1.0 - np.asarray(u, dtype=float)) ** -0.5 - 1.0


def test_comonotonic_var_pareto_goldens():
    assert pareto2_ppf(0.99) == pytest.approx(9.0, abs=1e-9)
    printed = {
        8: {0.99: 72.00, 0.995: 105.14, 0.999: 244.98},
        56: {0.99: 504.00, 0.995: 735.96, 0.999: 1714.88},
        648: {0.99: 5832.00, 0.995: 8516.10, 0.999: 19843.56},
    }
    for d, row in printed.items():
        for q, want in row.items():
            got = b.comonotonic_var([pareto2_ppf] * d, q)
            assert got == pytest.approx(want, abs=5e-3)
            assert got == pytest.approx(d * ((1.0 - q) ** -0.5 - 1.0), rel=1e-12)
    # equals the copula aggregate with the all-ones correlation within Monte Carlo error
    ppfs = [pareto2_ppf] * 3
    agg = c.copula_aggregate(ppfs, np.ones((3, 3)), "gaussian", None, 0.0, 0.99, 400_000, seed=1)
    assert abs(agg.var - b.comonotonic_var(ppfs, 0.99)) <= 5.0 * agg.var_se
    for bad in (0.0, 1.0, 2.0):
        with pytest.raises(ValueError, match="q"):
            b.comonotonic_var(ppfs, bad)
    with pytest.raises(ValueError, match="ppfs"):
        b.comonotonic_var([], 0.99)


def _table1_matrix():
    n, d, alpha = 50, 3, 0.99
    u = alpha + (1.0 - alpha) * np.arange(n) / n  # F5 (18): alpha + (1 - alpha)(i - 1) / N
    return np.tile(pareto2_ppf(u)[:, None], (1, d))


def test_rearrangement_table1_golden():
    x = _table1_matrix()
    assert x[0] == pytest.approx([9.0, 9.0, 9.0], abs=1e-9)
    assert x[0].sum() == pytest.approx(27.0, abs=1e-9)
    assert x[-1] == pytest.approx([69.71068] * 3, abs=5e-6)
    assert x[-1].sum() == pytest.approx(209.1320, abs=5e-4)
    assert x.sum(axis=0) == pytest.approx([851.72901] * 3, abs=5e-5)
    ys = []
    for seed in (0, 1, 2):
        y, value, sweeps, converged = b.rearrangement_algorithm(x, "max-min-row-sum", 1e-9, 1000, seed)
        ys.append(y)
        assert converged and sweeps >= 1
        assert y.sum(axis=0) == pytest.approx([851.72901] * 3, abs=5e-5)
        for j in range(3):
            assert np.array_equal(np.sort(y[:, j]), np.sort(x[:, j]))  # a column-wise permutation of the input
        rows = y.sum(axis=1)
        assert value == pytest.approx(rows.min(), abs=1e-12)
        assert rows.min() == pytest.approx(44.7671, abs=0.5)
        assert rows.max() == pytest.approx(87.8122, abs=0.5)
    assert not np.array_equal(ys[0], ys[1])  # the random start of F5 step 3 makes seeds differ (review F3)


def test_rearrangement_invariants_and_refusals():
    col = np.arange(1.0, 5.0)
    m = np.column_stack([col, col])
    for objective in ("max-min-row-sum", "min-max-row-sum"):
        y, value, sweeps, converged = b.rearrangement_algorithm(m, objective, 1e-9, 100, 0)
        assert value == pytest.approx(5.0, abs=1e-12)  # oppositely ordered columns: every row sums to 5
        assert np.allclose(y.sum(axis=1), 5.0)
        assert converged
    assert np.array_equal(m, np.column_stack([col, col]))  # the input is not mutated
    # the same seed gives the same matrix, another seed may permute differently but keeps the columns
    rng = np.random.default_rng(4)
    big = np.sort(rng.lognormal(size=(200, 4)), axis=0)
    r1 = b.rearrangement_algorithm(big, "max-min-row-sum", 1e-6, 500, 7)
    r2 = b.rearrangement_algorithm(big, "max-min-row-sum", 1e-6, 500, 7)
    assert np.array_equal(r1[0], r2[0]) and r1[1:] == r2[1:]
    # the best (lowest) maximum row sum is below the comonotonic maximum and the worst minimum is above the
    # comonotonic minimum
    lo = b.rearrangement_algorithm(big, "min-max-row-sum", 1e-6, 500, 7)[1]
    hi = r1[1]
    assert lo <= big.sum(axis=1).max() + 1e-12
    assert hi >= big.sum(axis=1).min() - 1e-12
    # max_sweeps stops the loop and says so
    _, _, sweeps, converged = b.rearrangement_algorithm(big, "max-min-row-sum", 1e-15, 1, 7)
    assert sweeps == 1 and not converged
    with pytest.raises(ValueError, match="epsilon"):
        b.rearrangement_algorithm(m, "max-min-row-sum", 0.0, 10, 0)
    with pytest.raises(ValueError, match="matrix"):
        b.rearrangement_algorithm(m[:1], "max-min-row-sum", 1e-3, 10, 0)
    with pytest.raises(ValueError, match="matrix"):
        b.rearrangement_algorithm(col[:, None], "max-min-row-sum", 1e-3, 10, 0)
    with pytest.raises(ValueError, match="objective"):
        b.rearrangement_algorithm(m, "median", 1e-3, 10, 0)
    with pytest.raises(ValueError, match="max_sweeps"):
        b.rearrangement_algorithm(m, "max-min-row-sum", 1e-3, 0, 0)
    with pytest.raises(ValueError, match="matrix"):
        b.rearrangement_algorithm(np.array([[1.0, float("nan")], [2.0, 3.0]]), "max-min-row-sum", 1e-3, 10, 0)


TABLE4_D8 = {
    # alpha: (lower range, upper range, tolerance)
    0.99: ((9.00, 9.00), (141.66, 141.67), 0.05),
    0.995: ((13.13, 13.14), (203.65, 203.66), 0.05),
    0.999: ((30.47, 30.62), (465.28, 465.30), 0.5),
}


@pytest.mark.parametrize("alpha", sorted(TABLE4_D8))
def test_worst_var_bounds_table4_d8(alpha):
    lower, upper, tol = TABLE4_D8[alpha]
    res = b.worst_var_bounds([pareto2_ppf] * 8, alpha, n=100_000, epsilon=1e-3, max_sweeps=1000, seed=0)
    assert res.alpha == alpha and res.n == 100_000 and res.epsilon == 1e-3
    assert res.lower_range[0] == pytest.approx(lower[0], abs=tol)
    assert res.lower_range[1] == pytest.approx(lower[1], abs=tol)
    assert res.upper_range[0] == pytest.approx(upper[0], abs=tol)
    assert res.upper_range[1] == pytest.approx(upper[1], abs=tol)
    assert res.converged
    assert res.upper_range[1] - res.upper_range[0] > 1e-4  # the s-bar and s matrices differ (review F2)
    assert res.lower_range[1] >= res.lower_range[0]
    assert len(res.sweeps) == 4 and all(s >= 1 for s in res.sweeps)
    assert res.lower_range[0] <= res.lower_range[1] <= b.comonotonic_var([pareto2_ppf] * 8, alpha)
    assert b.comonotonic_var([pareto2_ppf] * 8, alpha) <= res.upper_range[0] <= res.upper_range[1]


def test_worst_var_bounds_bracket_and_refusals():
    ppfs = [pareto2_ppf] * 3
    res = b.worst_var_bounds(ppfs, 0.99, n=5000, epsilon=1e-3, max_sweeps=500, seed=3)
    como = b.comonotonic_var(ppfs, 0.99)
    assert res.lower_range[1] < como < res.upper_range[0]
    again = b.worst_var_bounds(ppfs, 0.99, n=5000, epsilon=1e-3, max_sweeps=500, seed=3)
    assert again == res
    # bounded marginals: the worst VaR cannot exceed the sum of the supports' tops
    unif = [lambda u: 10.0 * np.asarray(u, dtype=float)] * 3
    r2 = b.worst_var_bounds(unif, 0.9, n=2000, epsilon=1e-6, max_sweeps=500, seed=1)
    assert r2.upper_range[1] <= 30.0 + 1e-9
    assert r2.lower_range[0] >= 0.0
    # three U(0, 10) at 0.9: the upper tail [9, 10] rearranged to a constant row sum of 3 x 9.5 is the worst VaR
    assert r2.upper_range[1] == pytest.approx(28.5, abs=0.05)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="alpha"):
            b.worst_var_bounds(ppfs, bad)
    with pytest.raises(ValueError, match="n"):
        b.worst_var_bounds(ppfs, 0.99, n=1)
    with pytest.raises(ValueError, match="ppfs"):
        b.worst_var_bounds(ppfs[:1], 0.99, n=100)


def test_worst_var_bounds_refuses_a_non_finite_bound():
    # review F4: with n = d the last cell of the upper matrix is F^-1(1) = +inf in every row
    with pytest.raises(ValueError, match="n is too small for the unbounded upper tail"):
        b.worst_var_bounds([pareto2_ppf] * 8, 0.99, n=8)
