"""Rating transitions: the transition matrix estimators (cohort, duration, Aalen-Johansen), the embedding problem and
the EM generator, the confidence intervals of a PD by grade, the mobility indices, the through-the-cycle portfolio and
the PD by grade under informative rating withdrawals.

Modules are imported by path (for example ``from riskvalidation.transitions import estimators``); nothing is
re-exported here. The validation tests are in ``riskvalidation.validation.transitions``.
"""

__all__ = [
    "em",
    "embedding",
    "estimators",
    "intervals",
    "mobility",
    "ttc",
    "withdrawals",
]
