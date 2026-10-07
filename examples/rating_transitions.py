"""Rating transitions end to end on a known generator (riskvalidation 0.04.000).

Estimate a migration matrix from snapshot counts and from paths, turn it into a generator, test whether the chain is
Markov and time-homogeneous, bound the PD by grade, measure mobility, and compute the through-the-cycle portfolio the
matrix implies; then plant rating momentum and watch which tests see it.

Run: python examples/rating_transitions.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from riskvalidation.generators.paths import RatingPaths
from riskvalidation.transitions import embedding, intervals, mobility, ttc
from riskvalidation.transitions.em import em_generator
from riskvalidation.transitions.estimators import cohort, duration_generator, pooled_cohort
from riskvalidation.validation.transitions import (
    rating_markov_order,
    rating_matrix_reference,
    rating_momentum,
    rating_time_homogeneity,
)

# five grades and default; a rare direct jump from the top grade to default
Q = np.array([
    [-0.12, 0.10, 0.015, 0.004, 0.00099, 0.00001],
    [0.02, -0.15, 0.11, 0.015, 0.004, 0.001],
    [0.002, 0.04, -0.17, 0.11, 0.015, 0.003],
    [0.0, 0.004, 0.06, -0.22, 0.14, 0.016],
    [0.0, 0.0, 0.005, 0.08, -0.30, 0.215],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])
SNAPSHOTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
TRUE_PD = expm(Q)[:5, -1]


def bp(x) -> str:
    return " ".join(f"{1e4 * v:8.2f}" for v in np.atleast_1d(x))


rng = np.random.default_rng(2026)
data = RatingPaths(Q, (2000,) * 5, horizon=5.0, snapshots=SNAPSHOTS, withdrawal_rate=0.02).draw(rng)
steps = data["steps"]

print("one-year PD by grade, basis points")
print("  truth                 ", bp(TRUE_PD))
year1 = cohort(steps[0]["counts"], steps[0]["withdrawn"])
print("  cohort, first year    ", bp(year1["matrix"][:5, -1]))
pooled = pooled_cohort([s["counts"] for s in steps], [s["withdrawn"] for s in steps])
print("  cohort, pooled (2.8)  ", bp(pooled["matrix"][:5, -1]))
dur = duration_generator(data["segments"], 6)
print("  duration              ", bp(expm(dur["generator"])[:5, -1]))
em = em_generator([{"counts": s["counts"], "dt": 1.0} for s in steps])
print("  EM from snapshots     ", bp(expm(em["generator"])[:5, -1]), f"({len(em['loglik'])} iterations)")

p = pooled["matrix"]
diag = embedding.embedding_diagnostics(p)
unseen = ", ".join(f"{i + 1} to {j + 1}" for i, j in diag["theorem3_c"]) or "none"
print(f"\nembedding of the pooled matrix: S = {diag['S']:.3f} (the log series converges when S < 1)")
print(f"  Theorem 3: (a) det P <= 0 {diag['theorem3_a']}, (b) det P > prod p_ii {diag['theorem3_b']}, "
      f"(c) reachable moves never observed: {unseen}")
print(f"  an exact generator is excluded: {diag['exact_generator_excluded']}")
for method in ("diagonal", "weighted", "jlt"):
    g = embedding.generator(p, method=method)
    print(f"  {method:9s} valid {g['valid']!s:5s}  L1 to P = {g['l1_distance']:.6f}")

print("\nis the chain Markov and time-homogeneous? (no defect planted)")
tests = [
    rating_time_homogeneity([s["counts"] for s in steps]),
    rating_markov_order(data["triplets"]),
    rating_matrix_reference(steps[0]["counts"], expm(Q)),
    rating_momentum(data["segments"]),
]
for r in tests:
    print(f"  {r.test_id:24s} statistic {r.statistic:8.2f}  p = {r.p_value:.4f}  {r.light.value}")

print("\n95% intervals for the first year's cohort PD, basis points (grade: Wald | Agresti-Coull | Jeffreys)")
d = steps[0]["counts"][:5, -1]
n = steps[0]["counts"][:5].sum(axis=1)
w, ac, jf = intervals.pd_wald(d, n), intervals.pd_agresti_coull(d, n), intervals.pd_jeffreys(d, n)
for g in range(5):
    print(f"  {g + 1}: {int(d[g]):3d}/{int(n[g])}  ({1e4 * w['lower'][g]:7.2f}, {1e4 * w['upper'][g]:7.2f}) | "
          f"({1e4 * ac['lower'][g]:7.2f}, {1e4 * ac['upper'][g]:7.2f}) | "
          f"({1e4 * jf['lower'][g]:7.2f}, {1e4 * jf['upper'][g]:7.2f})")
boot = intervals.pd_bootstrap(data["segments"], 6, n_boot=300, seed=1)
print("  duration PD, bootstrap: ", "  ".join(f"({1e4 * lo:.2f}, {1e4 * hi:.2f})"
                                               for lo, hi in zip(boot["lower"], boot["upper"], strict=True)))
print("  with a default correlation of 1%, 2,000 obligors count as", f"{intervals.effective_n(2000, 0.01):.1f}")

print(f"\nmobility of the pooled matrix: M_SVD {mobility.svd_mobility(p):.4f}, "
      f"trace index {mobility.trace_mobility(p):.4f}")

origination = [0.0, 0.3, 0.4, 0.3, 0.0]
t = ttc.ttc_portfolio(p, origination)
print("TTC portfolio of the pooled matrix:", " ".join(f"{v:.3f}" for v in t["portfolio"][:5]),
      f"(PD {100 * t['default_rate']:.2f}%)")
proj = ttc.project(p, [0.0, 0.0, 0.2, 0.4, 0.4], origination, 10)
print("unstressed projection of a speculative-heavy book, PD by year:",
      " ".join(f"{100 * v:.2f}%" for v in proj["default_rate"]))

print("\nthe same chain with downward momentum planted (alpha = 0.5, beta = 1)")
mom = RatingPaths(Q, (2000,) * 5, horizon=5.0, snapshots=SNAPSHOTS, momentum=((0.5, 1.0), (0.5, 1.0)),
                  investment_grades=3).draw(rng)
for r in (rating_time_homogeneity([s["counts"] for s in mom["steps"]]), rating_markov_order(mom["triplets"]),
          rating_momentum(mom["segments"])):
    extra = f"  coefficient {r.extras['coefficient']:.3f}" if r.test_id == "rating.momentum" else ""
    print(f"  {r.test_id:24s} p = {r.p_value:.2e}  {r.light.value}{extra}")
