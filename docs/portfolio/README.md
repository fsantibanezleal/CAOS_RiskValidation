# Portfolio credit capital

`riskvalidation.portfolio` computes the loss distribution of a credit portfolio by the rungs of dossier 04 (credit risk models) and
its Euler contributions; `riskvalidation.validation.portfolio` validates the economic capital model. The engine computes and makes
no compliance claim: confidence levels, tolerances, light parameters and seeds are parameters the caller supplies or Contraste
internal gates, not regulatory requirements. Money is in the portfolio currency and never converted. A value that cannot be
computed is a `ValueError` or a `not_evaluated` result, never a placeholder number. The existing `engines.vasicek` (the large
homogeneous portfolio) and `regulatory.irb` (the IRB capital) are the analytic anchors and are reused, never duplicated.

| Page | Contents |
|---|---|
| [01. The portfolio and its measures](01_portfolio-and-measures.md) | the portfolio record, EL, VaR, ES and EC, the sample estimators, the large homogeneous portfolio (rung C1) |
| [02. The copulas](02_copulas.md) | the Gaussian and Student-t copula Monte Carlo (rungs C2 and C3), chunking, seeds, Monte Carlo standard errors |
| [03. CreditRisk+](03_creditrisk-plus.md) | the Poisson-Gamma model, the stable recursion, the Panjer comparison (rung C4) |
| [04. Granularity and importance sampling](04_granularity-and-importance.md) | the Gordy and Lutkebohmert adjustment (rung C5), the exponential twist and its unbiasedness test (rung C6) |
| [05. Euler contributions and concentration](05_euler-and-concentration.md) | ES and VaR contributions, the asymptotic anchor, HHI and the effective number of names |
| [06. The Vasicek fit](06_vasicek-fit.md) | maximum likelihood, moments, the profile interval, the parametric bootstrap, the implied factors |
| [07. The validation tests](07_validation-tests.md) | the five tests with their light rules, and what must not be claimed |

Modules are imported by path, for example `from riskvalidation.portfolio import copula` and
`from riskvalidation.validation import portfolio`; nothing is re-exported from the package root. Requirements `RV-701` to `RV-729`
are in `docs/design/features/u7-portfolio/requirements.md`, each naming its test. Sources: dossier 04 of the Contraste initiative
(sections C.4 and H.1 to H.8 and the ladder "Portfolio credit risk", rungs C1 to C6) and dossier 05 (G.4 and G.5), cited with their
DOIs on each page; the granularity adjustment and the Emmer and Tasche anchor were read on the primary papers by the review of the
contract (U7-S).

## What the dossiers mark UNVERIFIED

An item the dossiers mark UNVERIFIED is marked UNVERIFIED here and is not silently decided:

- the exponential twist and the factor mean shift of Glasserman and Li (2005) (dossier 04 H.7, the paper is paywalled): the module
  accepts the estimator only by its unbiasedness against plain Monte Carlo (page 04);
- the equation numbers of Tasche (2008) (dossier 04 H.2) and the kernel bandwidth of Tasche (2009) (the bandwidth is the contract's
  choice, Silverman's rule; page 05);
- the original CSFB CreditRisk+ document, not accessed (page 03); the page-level formulas of the CreditMetrics technical document
  (migration mode is not implemented); the sector concentration adjustment of Pykhtin (2004), not implemented.

Not in this release: the exact granularity adjustment (eq. 14), random LGD in the copulas, CreditMetrics migration mode, a
multi-factor importance sampling twist.
