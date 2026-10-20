"""Rung 6 of the aggregation ladder: worst-case (and best-case) VaR bounds for given marginals and an unknown
dependence, by
the rearrangement algorithm of Embrechts, Puccetti and Ruschendorf, "Model uncertainty and VaR aggregation", Journal of
Banking and Finance 37(8), 2750-2764, 2013, doi:10.1016/j.jbankfin.2013.03.014 (accepted manuscript
https://people.math.ethz.ch/~embrecht/ftp/MUVaR.pdf, section 2.2, pp.8-11; transcribed in F5 of the U13 research
addendum).

VaR is not subadditive, so the comonotonic sum is not the worst case (dossier 05 l.366). For N x d matrices X the
operators
are s(X) = the smallest row sum and t(X) = the largest row sum. Upper VaR bound (the worst VaR, steps 1 to 7 of the
paper,
p.10): the matrices of (18), x_ij = F_j^-1(alpha + (1 - alpha)(i - 1) / N) and the same with i in place of i - 1, have
their
columns permuted at random, then each column in turn is rearranged oppositely ordered to the sum of the others, sweep
after
sweep, until s changes by less than epsilon; then s_N <= s-bar_N are the rearrangement range of the worst VaR (19).
The lower
bound (the best VaR, (20) and (21)) uses the matrices over [0, alpha], z_ij = F_j^-1(alpha (i - 1) / N), and the
largest row
sum t, which the loop minimises. No distribution function is inverted: the quantile functions are evaluated on the grid.

Where a marginal has unbounded support the last cell of the upper matrix x-bar is F_j^-1(1) = +infinity. The engine
keeps it
as +infinity (the paper's text read for F5 does not say how it treats that cell: UNVERIFIED): a row that holds it has an
infinite row sum, so it is never the smallest row and the smallest row sum stays finite. The lower matrices over [0,
alpha]
never reach probability 1 and are finite.

Remark 6 of the paper (p.11): there is no analytic proof that (19) and (21) hold for all starting configurations
(UNVERIFIED
as a theorem); the random permutation of step 3 is the paper's mitigation, and `converged = False` reports a run
stopped by
`max_sweeps`.

Randomness: `numpy.random.Generator(numpy.random.PCG64(seed))`; the result is reproducible for a seed. Inputs are never
mutated. Contract: `engine-aggregation-api` 1.0.0, module aggregation.bounds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from riskvalidation.portfolio.portfolio import check_level, positive_int

Ppf = Callable[[np.ndarray], np.ndarray]
OBJECTIVES = ("max-min-row-sum", "min-max-row-sum")


def comonotonic_var(ppfs: Sequence[Ppf], q: float) -> float:
    """The VaR of the comonotonic total, sum_k F_k^-1(q) (money). Golden: Pareto(2) marginals F(x) = 1 - (1 + x)^-2, so
    F^-1(q) = (1 - q)^(-1/2) - 1: d = 8 at q = 0.99 gives 72.00, at 0.995 105.14, at 0.999 244.98 (Table 4, "comonotonic
    (exact)", PDF p.14, recomputed from the formula)."""
    level = check_level(q)
    if len(ppfs) == 0:
        raise ValueError("ppfs is empty")
    grid = np.array([level])
    return float(sum(float(np.asarray(f(grid), dtype=float)[0]) for f in ppfs))


def rearrangement_algorithm(matrix: np.ndarray, objective: str, epsilon: float, max_sweeps: int,
                            seed: int) -> tuple[np.ndarray, float, int, bool]:
    """The rearrangement algorithm (F5, p.10) on an (N, D) matrix of money: permute each column randomly under the seed,
    then in a sweep rearrange column j to be oppositely ordered to the sum of the other columns for j = 1..D, and repeat
    sweeps until the value changes by less than `epsilon` between two successive sweeps or `max_sweeps` is reached.

    `objective` "max-min-row-sum" maximises the smallest row sum (the upper VaR bound), "min-max-row-sum" minimises the
    largest (the lower VaR bound); both use the same oppositely-ordered step, only the value that stops the loop
    differs.
    Returns (the rearranged matrix, the value, the number of sweeps, converged). The output is a column-wise
    permutation of
    the input. A +infinity entry (the last cell of an unbounded upper matrix) is kept; NaN and -infinity are refused.
    """
    m = np.array(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got shape {m.shape}")
    n_rows, n_cols = m.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError(f"matrix needs at least 2 rows and 2 columns, got shape {m.shape}")
    if np.any(np.isnan(m)) or np.any(np.isneginf(m)):
        raise ValueError("matrix must not contain NaN or -infinity")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    eps = float(epsilon)
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    sweeps_max = positive_int(max_sweeps, "max_sweeps")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    for j in range(n_cols):
        m[:, j] = rng.permutation(m[:, j])
    finite = np.isfinite(m)
    zeroed = np.where(finite, m, 0.0)
    inf_count = (~finite).sum(axis=1)
    sorted_cols = np.sort(m, axis=0)

    def value() -> float:
        rows = np.where(inf_count > 0, np.inf, zeroed.sum(axis=1))
        return float(rows.min() if objective == "max-min-row-sum" else rows.max())

    previous = value()
    converged = False
    sweeps = 0
    for sweep in range(1, sweeps_max + 1):
        sweeps = sweep
        total = zeroed.sum(axis=1)
        for j in range(n_cols):
            others = np.where(inf_count - (~finite[:, j]) > 0, np.inf, total - zeroed[:, j])
            order = np.argsort(others, kind="stable")
            new = np.empty(n_rows)
            new[order] = sorted_cols[::-1, j]  # the largest value goes to the row with the smallest rest
            m[:, j] = new
            new_finite = np.isfinite(new)
            new_zeroed = np.where(new_finite, new, 0.0)
            total += new_zeroed - zeroed[:, j]
            inf_count += (~new_finite).astype(int) - (~finite[:, j]).astype(int)
            zeroed[:, j], finite[:, j] = new_zeroed, new_finite
        current = value()
        if current == previous or abs(current - previous) < eps:
            converged = True
            break
        previous = current
    return m, value(), sweeps, converged


@dataclass(frozen=True)
class VarBounds:
    """The rearrangement ranges of the worst and the best VaR, money. `lower_range` is (t_N, t-bar_N), the best VaR (F5
    (21)); `upper_range` is (s_N, s-bar_N), the worst VaR (F5 (19)); `sweeps` are the sweep counts of the four runs in
    the
    order t_N, t-bar_N, s_N, s-bar_N; `converged` is True when all four stopped on `epsilon`."""

    alpha: float
    n: int
    epsilon: float
    lower_range: tuple[float, float]
    upper_range: tuple[float, float]
    sweeps: tuple[int, int, int, int]
    converged: bool


def _grid(ppfs: Sequence[Ppf], probs: np.ndarray, name: str) -> np.ndarray:
    cols = []
    for j, f in enumerate(ppfs):
        col = np.asarray(f(probs), dtype=float)
        if col.shape != probs.shape or np.any(np.isnan(col)) or np.any(np.isneginf(col)):
            raise ValueError(f"ppfs[{j}] must return {probs.size} values, none NaN or -infinity, on the {name} grid")
        cols.append(col)
    return np.column_stack(cols)


def worst_var_bounds(ppfs: Sequence[Ppf], alpha: float, n: int = 100_000, epsilon: float = 1e-3, max_sweeps: int =
1_000,
                     seed: int = 0) -> VarBounds:
    """The best and the worst VaR of the total of the marginals `ppfs` at level `alpha` for any dependence, money.

    The upper bound uses the matrices of F5 (18) over [alpha, 1] (cell edges F_j^-1(alpha + (1 - alpha)(i - 1) / N) and
    F_j^-1(alpha + (1 - alpha) i / N)), the lower bound those of F5 (20) over [0, alpha]; the four runs use the seeds
    `seed`, `seed + 1`, `seed + 2`, `seed + 3`. Golden (Table 4, Pareto(2), N = 1e5, epsilon = 1e-3, d = 8): alpha 0.99
    lower 9.00 - 9.00, upper 141.66 - 141.67; alpha 0.995 lower 13.13 - 13.14, upper 203.65 - 203.66; alpha 0.999 lower
    30.47 - 30.62, upper 465.28 - 465.30.
    """
    level = check_level(alpha, "alpha")
    cells = positive_int(n, "n")
    if cells < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if len(ppfs) < 2:
        raise ValueError(f"ppfs must hold at least 2 marginals, got {len(ppfs)}")
    i = np.arange(cells, dtype=float)
    upper = (level + (1.0 - level) * i / cells, level + (1.0 - level) * (i + 1.0) / cells)
    lower = (level * i / cells, level * (i + 1.0) / cells)
    mats = (
        (_grid(ppfs, lower[0], "lower"), "min-max-row-sum"),
        (_grid(ppfs, lower[1], "lower"), "min-max-row-sum"),
        (_grid(ppfs, upper[0], "upper"), "max-min-row-sum"),
        (_grid(ppfs, upper[1], "upper"), "max-min-row-sum"),
    )
    values, sweeps, flags = [], [], []
    for k, (matrix, objective) in enumerate(mats):
        _, v, sw, ok = rearrangement_algorithm(matrix, objective, epsilon, max_sweeps, int(seed) + k)
        values.append(v)
        sweeps.append(sw)
        flags.append(ok)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("n is too small for the unbounded upper tail: s-bar is infinite; use n > the number of "
                         "marginals")
    return VarBounds(alpha=level, n=cells, epsilon=float(epsilon), lower_range=(values[0], values[1]),
                     upper_range=(values[2], values[3]), sweeps=(sweeps[0], sweeps[1], sweeps[2], sweeps[3]),
                     converged=all(flags))
