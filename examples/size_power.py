"""Size and power of validation tests on known-truth generators (riskvalidation 0.03.000).

Run: python examples/size_power.py
"""

from __future__ import annotations

import numpy as np

from riskvalidation.generators import (
    BinormalScores,
    DefaultCounts,
    GradeFrequencies,
    LogisticPortfolio,
    Migrations,
    TwoSample,
    break_monotonicity,
    concentrate,
    decaying_matrix,
    feed_cell,
    feed_series,
)
from riskvalidation.harness import (
    estimator_performance,
    metric_above,
    rejection_probability,
    simulate,
    traffic_lights_rejection_probability,
)
from riskvalidation.validation.discrimination import auc_with_variance

# 1. Exact: the binomial test is a count test, so its size under correlated defaults needs no simulation
for rho in (0.0, 0.05, 0.10):
    ex = rejection_probability("pd.binomial", 1000, 0.01, 0.01, rho_true=rho)
    print(f"binomial test, N 1,000, PD 1%, nominal 1%, asset correlation {rho:.0%}: true size {ex['probability']:.2%}"
          f" (reject from {ex['critical_count']} defaults)")

# 2. The same by simulation: the harness agrees with the exact value within its Monte Carlo error
sim = simulate("pd.binomial", DefaultCounts(1000, 0.01, rho=0.05), n_rep=2000, seed=1, feed=feed_cell())
r = sim.rate(0.01)
print(f"simulated: {r.rate:.2%} (SE {r.se:.2%}, Wilson {r.interval[0]:.2%} to {r.interval[1]:.2%})")

# 3. Power against an underestimated PD, exact, for the independence and the Vasicek-corrected test
for test, kw in (("pd.binomial", {}), ("pd.binomial_vasicek", {"rho": 0.05})):
    p = rejection_probability(test, 1000, 0.01, 0.05, pd_true=0.015, rho_true=0.05, **kw)["probability"]
    print(f"{test}: power at 5% when the true PD is 1.5% and the correlation 5%: {p:.1%}")

# 4. The traffic-lights test under its two tie rules (WP14 scenario I_LC), exact
for ties in ("lower", "upper"):
    p = traffic_lights_rejection_probability([1000] * 5, 0.03, 0.05, ties=ties)["probability"]
    print(f"traffic lights, ties={ties}: size at 5% {p:.2%}")

# 5. The normal test over five years, measured
sim = simulate("pd.normal_multiperiod", DefaultCounts.over_years([1000] * 5, [0.03] * 5), n_rep=2000, seed=2,
               feed=feed_series())
print(f"normal test, five years: size at 5% {sim.rate(0.05).rate:.2%}")

# 6. Hosmer-Lemeshow on a validation sample: G degrees of freedom, and G - 2
gen = LogisticPortfolio(n=3000)
for kw in ({}, {"fitted": True}):
    s = simulate("pd.hosmer_lemeshow", gen, n_rep=500, seed=3, test_kwargs=kw)
    print(f"Hosmer-Lemeshow {kw or '(given PDs)'}: size at 5% {s.rate(0.05).rate:.1%}")

# 7. PSI: the rule of thumb against the benchmark (Yurdakul and Naranjo), no shift, samples of 100
s = simulate("stability.psi", TwoSample(100, 100), n_rep=2000, seed=4, rules=(metric_above(0.10),))
print(f"PSI, n = m = 100, no shift: PSI > 0.10 in {s.rate('metric>0.1').rate:.0%}, benchmark size "
      f"{s.rate(0.05).rate:.1%}")

# 8. A migration matrix with a cell twice its closer neighbour, and a concentration the ECB test cannot see
mig = Migrations([500] * 7, break_monotonicity(decaying_matrix(7), 3, 1, 2.0))
power = simulate("rating.migration_ztests", mig, n_rep=300, seed=5).rate(0.05).rate
print(f"migration z-tests, planted cell twice its neighbour: power {power:.0%}")
p0 = [0.02, 0.05, 0.10, 0.15, 0.20, 0.18, 0.13, 0.09, 0.05, 0.03]
hhi = simulate("rating.hhi", GradeFrequencies(5000, concentrate(p0, 0.5), p0), n_rep=300, seed=6)
print(f"ECB concentration test, half the mass moved into one grade: power {hhi.rate(0.05).rate:.0%}")

# 9. Estimator performance: the AUC and its ECB standard error on binormal scores
bs = BinormalScores(150, 4850, separation=1.2)
est, se = [], []
for i in range(400):
    d = bs.draw(np.random.default_rng(np.random.SeedSequence([7, i])))
    a, s2, _, _ = auc_with_variance(d["risk"], d["defaults"])
    est.append(a)
    se.append(s2**0.5)
perf = estimator_performance(est, bs.truth()["auc_current"], model_se=se)
print(f"AUC {bs.truth()['auc_current']:.4f}: bias {perf['bias']:+.4f} (MCSE {perf['bias_mcse']:.4f}), "
      f"empirical SE {perf['empirical_se']:.4f}, ECB SE {perf['model_se']:.4f}")
