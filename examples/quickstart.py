"""Quickstart: validate a PD model's calibration, discrimination and stability under one policy.

Run: python examples/quickstart.py
"""

import numpy as np

from riskvalidation import default_policy, run_suite
from riskvalidation.validation import (
    bin_counts,
    disc_auc_vs_initial,
    pd_jeffreys_grades,
    quantile_edges,
    stability_psi,
)

rng = np.random.default_rng(2026)

# Jeffreys test per grade and for the portfolio (ECB 2019, section 2.5.3.1)
for r in pd_jeffreys_grades(n=[400, 900, 300], d=[2, 14, 19], pd=[0.004, 0.012, 0.05], grades=["A", "B", "C"]):
    print(f"{r.segment:>9}  p = {r.p_value:.4f}  {r.light.value}")

# a scored sample: PDs and observed defaults, plus a development-period score sample
pd_scores = rng.beta(1.2, 40.0, 5000)
defaults = (rng.uniform(size=5000) < pd_scores).astype(int)
development = rng.beta(1.2, 45.0, 8000)
edges = quantile_edges(development, 10)

# a battery under one policy; a failing test is recorded and the rest still run
run = run_suite(
    [
        (disc_auc_vs_initial, {"auc_initial": 0.78, "risk": pd_scores, "defaults": defaults}),
        (stability_psi, {"expected_counts": bin_counts(development, edges),
                         "actual_counts": bin_counts(pd_scores, edges)}),
    ],
    policy=default_policy(),
    model_id="retail-pd-v3",
)
for r in run.results:
    print(f"{r.test_id:>22}  statistic = {r.statistic:.4f}  p = {r.p_value:.4f}  {r.light.value}")
print("lights:", run.counts(), "worst:", run.worst().value)
