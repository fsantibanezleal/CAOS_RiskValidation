"""Fairness: group metrics, inference on selection rates, conditional disparity, replication, situation testing and
planted-bias generators.

Modules are imported by path (for example ``from riskvalidation.fairness.metrics import four_fifths_flag``); nothing is
re-exported here, because `fairness.metrics` and `validation.fairness` each define a `calibration_within_groups` and an
`auc_by_group` (the pure metrics and the `TestResult` functions). The engine states no legal claim: the four-fifths line
is an employment benchmark (29 CFR 1607.4(D)) and never a credit-law threshold, and the outcome-conditioned metrics
never run on a dataset with no outcome. The `TestResult` functions are in ``riskvalidation.validation.fairness``.
Contract: ``engine-fairness-api`` 1.0.0.
"""

__all__ = [
    "conditional",
    "inference",
    "metrics",
    "plants",
    "policy",
    "replication",
    "situation",
]
