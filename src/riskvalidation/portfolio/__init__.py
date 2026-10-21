"""Portfolio credit capital: the loss distribution of a credit portfolio by the rungs of dossier 04 and its Euler
contributions.

Rungs: C1 the large homogeneous portfolio (`portfolio.lhp_measures`, from `riskvalidation.engines.vasicek`), C2 and C3
the Gaussian and Student-t copula Monte Carlo (`copula`), C4 CreditRisk+ with the stable recursion
(`creditrisk_plus`), C5 the granularity adjustment of Gordy and Lutkebohmert (`granularity`), C6 importance sampling
(`importance`); the Euler contributions and the concentration measures (`euler`); the fit of the Vasicek distribution
to a series of yearly loss rates (`asrf_fit`). The validation tests of an economic capital model are in
``riskvalidation.validation.portfolio``.

Modules are imported by path (for example ``from riskvalidation.portfolio import copula``); nothing is re-exported
here. The engine computes and states no compliance claim: confidence levels, tolerances, light parameters and seeds
are parameters the caller supplies or Contraste internal gates. Money is in the portfolio currency and never
converted. A value that cannot be computed is a ValueError or a not_evaluated result, never a placeholder number.
Contract: ``engine-portfolio-api`` 1.0.0.
"""

__all__ = [
    "asrf_fit",
    "copula",
    "creditrisk_plus",
    "euler",
    "granularity",
    "importance",
    "portfolio",
]
