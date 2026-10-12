"""riskvalidation: validation tests, regulatory calculators and reference engines for financial risk models.

Every validation test returns one :class:`TestResult`; a versioned :class:`Policy` turns its statistic into a
traffic light; :func:`run_suite` runs a battery without stopping at the first failure. The core depends
only on NumPy and SciPy, so it also runs in the browser under Pyodide.
"""

from riskvalidation import validation  # noqa: E402,F401  (registers the test catalogue)
from riskvalidation.core.policy import Policy, Threshold, default_policy, load_policy
from riskvalidation.core.registry import TestSpec, catalogue
from riskvalidation.core.result import Light, TestResult
from riskvalidation.core.suite import SuiteRun, run_suite

__version__ = "0.5.0"
"""PEP 440 form of the display version in ``VERSION``."""

__all__ = [
    "Light",
    "Policy",
    "SuiteRun",
    "TestResult",
    "TestSpec",
    "Threshold",
    "__version__",
    "catalogue",
    "default_policy",
    "load_policy",
    "run_suite",
    "validation",
]
