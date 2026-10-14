- engine U7-E: new package `riskvalidation.portfolio` (the portfolio record and its EL, VaR, ES and EC with the Acerbi-Tasche sample
  estimators and the large homogeneous portfolio measures; the Gaussian and Student-t copula Monte Carlo, seeded and chunked to a stated
  memory bound, with batch-means standard errors; CreditRisk+ with Gamma sectors by the stable recursion of Haaf, Reiss and Schoenmakers
  (2004) and the Panjer comparison; the Gordy and Lutkebohmert (2013) granularity adjustment as a share of EAD from the caller's UL capital
  K_i, with `k_creditrisk_plus` for eq. 13; importance sampling for the Gaussian copula accepted by its unbiasedness against plain
  Monte Carlo (the twist is UNVERIFIED in dossier 04 H.7); Euler ES contributions that add up exactly, kernel VaR contributions, the
  asymptotic one-factor anchor, HHI and the effective number of names; the fit of the Vasicek distribution to yearly loss rates by
  maximum likelihood and moments with the profile interval and a parametric bootstrap, and the implied factors) and
  `riskvalidation.validation.portfolio` (the tail back-test, Monte Carlo convergence, Euler additivity, CreditRisk+ recursion against
  Monte Carlo, and the size and power of the correlation estimate), contract `engine-portfolio-api` 1.0.0, requirements RV-701 to RV-729,
  documentation in `docs/portfolio/`. Existing engine results are unchanged.
- engine U7-E review fixes: `importance.weighted_var_es` builds the quantile and the ES on the upper tail probability
  sum(w 1{L > x}) / S normalised by the scenario count, not by the realised sum of the weights (which gave a VaR of 35
  against 21.0 at the plain 99.9 percent VaR; blocker B1) and refuses a level whose tail mass the weights cannot
  account for; tests for the production threshold, for the size of the batch-means standard errors against the spread over
  replications and for the moments estimator on an asymmetric series; documentation of the lattice case.
