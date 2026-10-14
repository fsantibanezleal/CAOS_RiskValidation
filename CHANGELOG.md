# Changelog

All notable changes are recorded here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use the display form `X.XX.XXX`; the package
metadata carries the PEP 440 form (`0.01.000` is `0.1.0`).

## [0.07.000] - 2026-10-14

Portfolio credit capital, market risk, interest rate risk in the banking book and operational risk (Contraste units U7, U8, U10 and U12): `riskvalidation.portfolio` with `riskvalidation.validation.portfolio`, `riskvalidation.market` with `riskvalidation.validation.market` (the GARCH family through the optional extra `market`), `riskvalidation.irrbb` with `riskvalidation.validation.irrbb`, and `riskvalidation.oprisk` with `riskvalidation.validation.oprisk`, each reproducing the golden values of its Contraste contract (`engine-portfolio-api` 1.0.0, `engine-market-api` 1.0.0, `engine-irrbb-api` 1.1.0, `engine-oprisk-api` 1.0.0) and each reviewed by Opus with its fixes applied. No existing engine result changes.

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
- U8 market risk (RV-801 to RV-830): `riskvalidation.market` with the sign convention (`to_loss_numbers`, `var_es_from_sample`,
  `var_es_from_distribution`), historical, age-weighted, filtered and volatility-updated simulation, delta-normal, Cornish-Fisher
  and EWMA, GARCH/GJR/EGARCH through `arch` (optional extra `market`), conditional EVT, CAViaR with the Taylor ES extension
  and the dynamic quantile statistic, the GAS one-factor, two-factor and hybrid FZ0 models, the FZ0 and tick losses,
  Diebold-Mariano and the Model Confidence Set, and the FRTB statistics (MAR32.9 table, zones, multipliers, desk limits, PLA
  Spearman, exact KS step count and zones).
- U8 back-tests (`riskvalidation.validation.market`): Kupiec, Christoffersen independence and conditional coverage, the
  duration test, the Basel traffic light, Acerbi-Szekely, Du-Escanciano unconditional and conditional, McNeil-Frey, the
  dynamic quantile test, the PLA test and the desk back-test, each a `TestResult` with its size checked by simulation.
- Documentation under `docs/market/` (equations, goldens, sources, decisions D-1 to D-9, limits) and the feature SDD
  `docs/design/features/u8-market/requirements.md`.
- U8-E review fixes: the age-weighted HS now has the two knots per return of Boudoukh, Richardson and Whitelaw (1998) and
  reproduces their worked example (VaR 2.63% and 2.34%); the dynamic quantile test takes a Monte Carlo p-value (`n_mc`, `seed`;
  size between 0.03 and 0.07 at (0.01, 250), (0.025, 250) and (0.01, 750)); the CI installs `.[dev,market]`; the ES-CAViaR is
  declared two-step (decision D-10, `caviar_es_fit`); the duration test censors the first spell only when the sample does not
  start with a hit and reports a capped shape; the PLA zone is decided in exact arithmetic (`frtb.compare_spearman`,
  `frtb.pla_zone_exact`).
- U10 IRRBB (U10-E): the new subpackage `riskvalidation.irrbb` with `curves` (zero rates, discount factors, the par-yield bootstrap, Nelson-Siegel, Svensson, the Diebold-Li dynamic Nelson-Siegel loadings, fit and AR(1) factor forecast), `shocks` (the six SRP31 shapes, the 2024 and 2016 size tables and the RAN 21-13 Annex 1 Hoja 9 table as data with their source, `shock_size` with its origin and the Contraste-estimate label for a currency without an entry, the post-shock floors, the EU floor), `slotting` (the 19 buckets of SRP31.96 Table 3, notional cash-flow slotting, the non-maturity deposit split with the SRP31 Table 4 caps applied and reported, the prepayment and term-deposit multipliers, SMM, CPR and PSA), `eve_nii` (the repricing and duration gaps, the standardised EVE and its risk measure, delta NII, the automatic-option add-on, the CLP and UF pair with the 50 percent offset on both sign branches, the Annex 1 delta NII and delta EVE, the 2.5 percent indexation charge, the outlier and prioritised-bank tests), `behavioural` (the static replicating portfolio as the global optimum, the weighted average maturity, the deposit beta, the prepayment logit and its maximum-likelihood fit) and `simulation` (the dynamic NII projection, Hull-White one-factor and dynamic Nelson-Siegel scenarios, EaR and EVaR); a positive delta EVE or delta NII is a loss and discounting is continuously compounded.
- U10 IRRBB (U10-E): the new module `riskvalidation.validation.irrbb` with seven validation tests that each return one `TestResult`: the SRP31.92 golden values, the RAN 21-13 Annex 1 formulas, the EVE sum-of-parts reconciliation, the NMD parameter stability across estimation windows, the EVE with and without behavioural assumptions (red whatever the size when the behavioural measure hides an outlier breach), the behavioural change at a limit breach and the out-of-sample NII check; a test that cannot be evaluated is `not_evaluated` with a null statistic, metric and p-value. The acceptance gates are Contraste internal gates, not regulatory thresholds.
- U10 IRRBB (U10-E): documentation pages under `docs/irrbb/` (curves, shocks, slotting, EVE and NII, Annex 1, behavioural models, simulation, the validation tests) with the formulas, the golden values, the sources and the UNVERIFIED items, and requirements `RV-1001` to `RV-1032` in `docs/design/features/u10-irrbb/requirements.md`, each naming its test. The engine core stays NumPy and SciPy; QuantLib is only an optional test cross-check of the discount factors, skipped when its wheel is not installed. No existing engine result changes.
- U10 IRRBB (U10-E review fixes): `simulation.ear_evar_mc_error` returns the Monte Carlo standard error of the EaR and of the EVaR at a stated level (batch means over contiguous batches, with an exact golden and a closed-form check), so the C14 documents carry an engine number; `eve_risk_measure` refuses a matrix without exactly six shock rows (a transposed matrix is no longer accepted silently); `forecast_dns_factors` keeps the innovation spread of a factor whose lagged series is flat; `cmf_annex1_goldens` counts a raising Annex 1 EVE function as a failed case and states the printed-precision tolerances of the two Hoja 10 values in its thresholds and policy version; four test gaps found by mutation are closed (the `cmf_pair` share on the first branch, the logit separation guard, the Hoja 10 tolerances).
- Operational risk (U12-E): `riskvalidation.oprisk` with the frequency and severity laws and their truncated maximum likelihood fits (lognormal, Weibull, GPD, the lognormal body and GPD tail splice, the tail-index profile), the compound annual loss by Monte Carlo (seeded, chunked), Panjer recursion and FFT with Shevchenko's benchmark reproduced (Table 2: H_5847 0.998999329, H_5849 0.999000217, quantile 5849), the quantile, expected shortfall and exceedance read from the pmf, the single loss approximation with its UNVERIFIED mean-corrected form (refused for an infinite mean), the scenario combination with the Shevchenko and Wuthrich Gamma-Pareto credibility, and the OPE25 standardised approach (business indicator, BIC 5.37bn at BI 35bn, loss component, ILM, the five-year BIC-only rule); contract `engine-oprisk-api` 1.0.0.
- Operational risk validation tests (U12-E): `riskvalidation.validation.oprisk` with the benchmark reproduction, the threshold stability of the tail index, the capital stability across independent histories, the SMA ILM response to one large loss and the scenario influence, each returning one `TestResult` (descriptive: no p-value, tolerances in `extras["thresholds"]`).
- Documentation: `docs/oprisk/` (five pages and an index) and the feature requirements `RV-1201` to `RV-1225` under `docs/design/features/u12-oprisk/`.
- Review fixes (U12-E): `es_for_severity` and the `infinite_mean` keyword of `es_from_pmf` refuse an expected shortfall when the severity has no finite mean on any grid (a coarse grid passes the missing-mass guard); `gamma_prior_from_scenarios` refuses points worth less than one expected event in `basis_years` and any refit on the search bound, and `fit_tail_to_scenarios` reports `converged = False` on a bound; `benchmark_reproduction` documents `moment_rel_tol` of at least 1e-5 for engine moments; the not_evaluated threshold-stability result counts infinite-mean fits; `severity_cdf` and `truncated_severity_cdf` refuse a NaN loss.

## [0.06.000] - 2026-10-13

Loss given default and exposure at default (Contraste unit U6): `riskvalidation.lgd_ead` (workout LGD, the LGD model rungs, downturn LGD, CCF and EAD by reference-date design, the IRB floors) and `riskvalidation.validation.lgd_ead` (LGD calibration and discrimination, downturn coverage, EAD back-testing by utilisation band, CCF estimator bias), each function reproducing the golden values of Contraste's `engine-lgd-ead-api` contract (reviewed: 74 of 74 recomputed independently).

- engine U6-E: new package `riskvalidation.lgd_ead` (workout LGD with the EU discount rate convention, range flags,
  incomplete workout treatment and the LGD distribution; the LGD rungs OLS, fractional logit, beta, hurdle and Tobit with
  `predict_lgd`; the cohort downturn add-on; the EU-convention CCF with three reference-date designs and four treatments,
  the direct EAD and utilisation change models; the IRB LGD and EAD floors, the partly secured floor formula documented for the caller and the
  supervisory CCFs) and the six tests of `riskvalidation.validation.lgd_ead`, reproducing every golden value of the contract
  `engine-lgd-ead-api` 1.0.0 (requirements RV-601 to RV-625, pages under `docs/lgd_ead/`).

## [0.05.000] - 2026-10-12

Provisions (Contraste unit U5): the IFRS 9 expected credit loss engine and its validation battery, and the Chilean
(CMF CNCB B-1) and Spanish (BdE Circular 4/2017 Annex 9) statutory provisions, implementing the provisions API
contract of CAOS_Contraste. Reviewed by Opus: every golden value and transcribed cell reproduces.

### Added

- Added `riskvalidation.provisions.term_structure`: `TermStructure` and the constructors `from_hazard`, `from_marginal` and
  `from_cumulative` (survival, marginal and cumulative PD identities to 1e-12), `competing_risks` (default incidence next to
  prepayment, never 1 minus the survival of both exits), `forward_pd`, `annualise_pd`, the one-factor `ttc_to_pit` and
  `implied_z`, `weighted_pit_pd` and the Gauss-Hermite cross check `gauss_hermite_pit_pd`.
- Added `riskvalidation.provisions.staging`: `StagingPolicy` (the threefold ratio, the 30 and 90 days past due thresholds and
  the probation are parameters, not constants), `sicr_quantitative`, `allocate_stages` with the low credit risk exemption
  suppressing only the two quantitative triggers, `apply_cure_probation` and the report-only `collective_transfer`.
- Added `riskvalidation.provisions.ecl`: `discount_factors`, `ecl_scenario`, `probability_weighted_ecl`,
  `non_linearity_index`, `weight_sensitivity`, `stage3_ecl`, `poci_allowance`, `simplified_matrix_ecl`, `commitment_ead` and
  `portfolio_allowance` (model ECL by the four stage keys; overlays added to the total only).
- Added `riskvalidation.provisions.overlays` (overlays as separate lines with a stage) and
  `riskvalidation.provisions.reconciliation` (the IFRS 7 35H and 35I roll-forwards, `check_identities` and
  `identity_residuals` on one tolerance basis, 1e-6 times max(1, G)).
- Added `riskvalidation.validation.ifrs9`: nine tests of dossier 02 I.2 (lifetime PD backtest with the Vasicek band, SICR
  effectiveness and threshold stability, stage migration and ECL backtests, scenario sensitivity, non-linearity, overlay
  diagnostics, reconciliation identities), each a `TestResult` with its reference, null hypothesis and policy parameters.
- Added documentation pages `docs/provisions/ecl/` and requirements RV-501 to RV-519 with golden values from the provisions API
  contract; the new modules are imported by path and are not re-exported from the package root.
- Engine review fixes: `portfolio_allowance` and `probability_weighted_ecl` refuse a negative model ECL for stages 1 to 3 (POCI may
  be negative); `lifetime_pd_backtest` refuses `n_events` above `n_at_risk` at an observed horizon; `reconciliation_identities`
  returns `not_evaluated` with its reason (never an infinite metric) when the waterfall has no opening or closing line or a stage
  lacks its stage ECL, and a movement without a waterfall line counts as a line of zero; the Vasicek band has a value assertion.
- Contract change (decision C2): `apply_cure_probation(stage_now, stage_previous, months_since_cure, policy, now)` takes the
  current `StageResult` as a required argument `now`; the trigger `unspecified` no longer exists, so every unchanged Stage 2 or 3
  account keeps a trigger of `TRIGGERS`. `riskvalidation.provisions.__init__` carries one docstring naming both halves, identical
  on both U5 branches.
- Added `riskvalidation.provisions.local_cl`: CMF CNCB Chapter B-1 (the individual categories A1 to B4, the default portfolio C1 to C6,
  the Circular 2.346 standard-method tables for mortgages, commercial leasing, student loans, generic commercial and consumer loans
  with every cell transcribed, the group provision with guarantor substitution, the 0.50% minimum on the individually assessed
  normal portfolio, the max rule and additional provisions as their own line). B-1 is kept apart from IFRS 9.
- Added `riskvalidation.provisions.local_es`: Banco de Espana Circular 4/2017 Annex 9 coverage tables (point 144 and point 140 as
  replaced by Circular 6/2021), the uncovered amount and the category-to-stage map.
- Added documentation pages `docs/provisions/local/` and requirements RV-550 to RV-567 with the transcribed tables as fixtures.
- Fix (engine review F1): `local_cl` and `local_es` refuse non-finite money, rates, buckets, days and amounts (`inf` as well as
  NaN) with `ValueError`; `b1_group_provision(inf, 0.02, ...)` no longer returns NaN.
- `riskvalidation.provisions.__init__` carries one docstring naming both halves, identical on both U5 branches (F6).

## [0.04.003] - 2026-10-07

### Fixed

- `generators.paths.RatingPaths` refuses a generator with a non-finite rate. Its checks compared entries, and every
  comparison is False on NaN: a NaN rate passed and made a draw run without end; an infinite rate balancing an
  infinite diagonal passed the row sums. Found by the adversarial review of Contraste's case C04.

### Documented

- `docs/transitions/03_markov-tests.md`: which form of the order test holds its size depends on the design, measured on
  two seven-grade chains (Contraste C04's S&P-like chain: the chi-square over-rejects, the likelihood ratio holds; a
  hand-made chain: the reverse at 5%). The default stays the chi-square, chosen on the size study's chain.

## [0.04.002] - 2026-10-07

### Added

- `harness.estimator_performance` reports the mean squared error with its Monte Carlo SE (Morris, White and Crowther
  2019, Table 6) and its root, whose Monte Carlo SE is the MSE's carried by the delta method (the package's step:
  Table 6 gives none for the root). Contraste's case C04 computed both itself.
- `transitions.intervals.exact_coverage(method, n, pd)`: the exact coverage and expected length of the Wald,
  Agresti-Coull or Jeffreys interval for a binomial count, by enumeration, for one or many true PDs; and
  `overlap_probability(method, n, pd_a, pd_b)`: the probability that two grades' intervals overlap, summed over every
  pair of counts. The engine's own tests enumerated the coverage by hand, and C04 re-implemented both.

## [0.04.001] - 2026-10-07

### Fixed

- `regulatory.capital_requirement` refused a PD of exactly 0 although it floors the PD at the regime's floor: the
  long-run average of a grade that never defaulted (an agency's AAA over CEREP's cohorts, in Contraste's case C04) is
  the case the floor exists for (CRE32.4), and every caller had to apply the floor itself, a second copy of the regime's
  value. A PD of 0 now takes the floor wherever one applies (the QRRE revolver floor for a revolver), with the usual
  note, and is still refused for a class the regime exempts from the floor (sovereigns under Basel III final, CRR3 and
  Basel II), where the formula has no value at 0. A negative PD, a PD of 1 and NaN are refused as before.

## [0.04.000] - 2026-10-06

### Added

- `riskvalidation.transitions`: rating transitions.
  - `estimators`: the cohort matrix (Anderson and Goodman's (2.9), pooled (2.8)) with withdrawals removed, the industry
    treatment, or kept as an absorbing state, as EBA/GL/2017/16 paragraph 76 keeps them in the default rate; the
    duration generator $K_{ij}/S_i$; the Aalen-Johansen product integral.
  - `embedding`: Israel, Rosenthal and Wei's log series with its convergence condition, the three non-existence
    conditions of their Theorem 3, the diagonal, weighted and JLT generators with the L1 distance of $\exp(Q)$ to $P$,
    and stochastic monotonicity.
  - `em`: the maximum likelihood generator of snapshot counts by EM, with Smith and dos Reis's closed-form E-step
    (Van Loan block exponentials).
  - `intervals`: Wald (2.2), Agresti-Coull (3.3) with the dependence correction (3.4) of Schuermann and Hanson, the
    Jeffreys interval, and the duration PD's bootstrap by resampling histories or parametric (Christensen, Hansen and
    Lando).
  - `mobility`: Jafry and Schuermann's $M_{SVD}$ and the trace index.
  - `ttc`: Engelmann's through-the-cycle portfolio (the fixed point of his (10) under Theorem 1, primitivity checked)
    and the projection (9) for one matrix or a stressed path.
- Four registered tests (family `transitions`, policy 2026.10.1): `rating.time_homogeneity`, `rating.markov_order`,
  `rating.matrix_reference` (Anderson and Goodman 1957) and `rating.momentum` (dos Reis, Pfeuffer and Smith 2020), each
  in its size study.
- `generators.paths.RatingPaths`: exact continuous-time rating paths from a generator, with dos Reis et al.'s
  self-exciting downgrade momentum (Ogata thinning), regimes, informative withdrawals and snapshots in ESMA CEREP's
  stock model; truth and verifier.
- Published results reproduced: Israel, Rosenthal and Wei (2001) section 4 and Engelmann (2024) section 4, read from
  the papers in the device data root (`RISKVALIDATION_DATA`, the tests skip without them: the matrices are agency data
  reprinted there); Schuermann and Hanson (2004) Table 5.
- Docs: the rating-transitions section (five pages), the rating-paths generator page, seven figures (six computed,
  theme-aware), `examples/rating_transitions.py` (run by the suite), and `tests/test_docs.py`, which checks every
  relative link, every figure and that every registered test is documented on a user-facing page.

### Changed

- The size study gives every test a fixed seed (`tests/test_size.py`, `SEEDS`): the seed used to depend on the
  number of tests sorted before it, so adding a test moved other tests' measured sizes, which their pages quote. The
  existing tests keep their seeds and measurements; `scripts/make_doc_figures.py` reads the same seeds and can draw
  named figures only.
- Each default form of the transition tests was chosen by its measured size on sparse rating tables (the
  time-homogeneity chi-square (3.8), the order chi-square on states grouped by direction, the reference likelihood
  ratio); the size study requires the test's page to state its measured size at 5% and at 1%.

## [0.03.002] - 2026-10-06

### Fixed

- `harness.wilson_interval` returns its bounds exactly at 0 and at n successes: the closed form's subtraction left a
  residue of about 1e-20 above a zero rate for some n (5.4e-20 at n = 4,000), so an interval did not contain its own
  rate (#18). The suite checks the containment for every k over seven values of n.

## [0.03.001] - 2026-10-06

### Fixed

- Docs: the reproduction of Yurdakul and Naranjo's Table 4 now states its small systematic offset (the harness sits
  about 0.7 standard errors above the paper on average, so how many of the 54 cells pass depends on the seed: 53 with
  the suite's, 50 with another), which "53 of 54" alone hid; the suite holds the mean z below 1.2. No code changed.

## [0.03.000] - 2026-10-05

### Added

- `riskvalidation.generators`: known-truth data-generating mechanisms, each a frozen configuration that serialises
  itself, draws one dataset per seed, returns its exact truth and verifies itself on its own stream:
  `DefaultCounts` (BCBS WP14's one-factor model with factor correlation across years; the default correlation
  reproduces WP14's examples), `BinormalScores` (exact AUC and AUC variance; decay, an estimated development AUC, a
  challenger, a leaked feature), `LogisticPortfolio` (true PDs, a planted calibration intercept and slope, an omitted
  quadratic term, covariate shift, one-factor dependence), `TwoSample` (Yurdakul and Naranjo's design, population or
  sample binning), `Migrations` and `decaying_matrix` with `break_monotonicity`, `GradeFrequencies` with `concentrate`
  (and a development sample for two-sample tests of the counts), and `NestedLda` (Demler et al.'s Framingham-like
  nested discriminants).
- `riskvalidation.harness`: `simulate` runs a registered test over repetitions with one independent stream per
  repetition (a test with its own randomness gets a fresh seed), counts undefined results, and reports rates per level
  and rule with the Monte Carlo SE of Morris, White and Crowther (2019) and the Wilson interval; `size_bound`,
  `within_size`, `agrees`, `agrees_with_published`; exact rejection probabilities for the count tests
  (`rejection_probability`, `critical_count`, `count_pmf`) and for the traffic-lights test with independent years;
  WP14's large-portfolio true confidence; `estimator_performance` (bias, empirical SE, model SE, coverage).
- Published simulations reproduced by the suite: BCBS WP14 Tables 7 and 8 (143 of 144 cells, 25,000 runs), Yurdakul
  and Naranjo (2020) Table 4 (53 of 54, 10,000 runs), Demler, Pencina and D'Agostino (2012) (1 rejection in 1,000);
  the two cells that do not reproduce are listed with their evidence.
- The size of every registered test with a p-value, measured under its null (`tests/test_size.py`).
- `disc.auc_vs_initial(auc_initial_variance=...)`: when the development AUC is itself an estimate, its variance is
  added to the statistic (an extension of the ECB formula, stated in the result); the ECB form rejects 11.6% of the
  time at 5% with equal development and current samples, the extended form 5.4%.
- `pd.traffic_lights(ties=...)`: the colour of a year on a boundary; `"lower"` (default) is WP14's printed mapping,
  `"upper"` the rule its own simulation must have used. `traffic_light_colours` and `traffic_profile_cdf` expose the
  colour rule and the profile table.
- Docs: the generators section (four pages), the measured sizes and the findings on every test family's page, five
  computed theme-aware figures, `examples/size_power.py` (run by the suite).

### Changed

- `pd.hosmer_lemeshow` uses as many degrees of freedom as groups by default (PDs given before the outcomes, the
  validation case) and G - 2 only with `fitted=True`. Measured on independent samples with the true PDs, G - 2
  rejects 11.7% of the time at the 5% level and G 5.1%. Results computed with the old default on validation samples
  had p-values that were too small.
- The traffic-lights p-value is a lookup in a cached profile table, and the normal test, the traffic lights and the
  PSI call SciPy's special functions directly: identical numbers, several times faster.
- A standardised default rate within 1e-12 of a traffic-lights boundary is on the boundary (it was coloured by
  rounding noise).

### Documented (found by the harness)

- The normal test over five years rejects 8.5% of the time at the 5% level, as WP14's own Table 7 shows, while WP14's
  summary calls it conservative.
- The chi-square test over grades exceeds its level when a grade expects under one default.
- The ECB AUC test treats the development AUC as known; estimated on a sample as large as the current one, it rejects
  10.9% of the time at the 5% level (the extension above holds the size).
- The ECB concentration statistic is bounded in the current CV: with ten grades and an initial CV of 0.6 it cannot
  reject at 5%, whatever the current distribution.
- The Jeffreys test's exact size for 1,000 obligors at PD 1% is 1.38% at the 1% level.

## [0.02.000] - 2026-10-05

### Added

- `riskvalidation.regulatory.irb`: the IRB risk-weight functions (corporate with the firm-size and
  financial-institution adjustments, sovereign, bank, HVCRE, SME, residential mortgage, QRRE, other retail), defaulted
  exposures, expected loss against eligible provisions, and four regimes as data with paragraph references and in-force
  dates: Basel III final, EU CRR3, Basel II (the 1.06 factor, 0.03% floors) and CMF RAN 21-6 (its own correlations, the
  61.3% consumer LGD). Reproduces all 144 cells of BCBS CRE99 Table 1 within 0.01 percentage points (141 exact).
- `riskvalidation.engines.vasicek`: the one-factor conditional PD, the large homogeneous portfolio's distribution,
  quantile and density, and the systematic factor implied by an observed default rate.
- `riskvalidation.engines.low_default`: most prudent estimation (Pluto and Tasche), independent and with asset
  correlation by Gauss-Hermite quadrature, scaled to a central tendency or to the portfolio's upper bound (section 5),
  and over several years with intertemporally correlated factors by seeded Monte Carlo (section 6). Reproduces the
  paper's Tables 1 to 14 and documents where its own values deviate (Table 4 at 75%, Table 8, Table 12, Tables 13
  and 14).
- `riskvalidation.engines.pd_curve`: the rating model's identities (Tasche 2013, section 3, theorem 3.3), quasi moment
  matching (appendix A), and every calibration approach of the paper: invariant default profile, invariant accuracy
  ratio, scaled PDs, scaled likelihood ratio (case 1), invariant likelihood ratio (cases 2 and 3), invariant PD curve
  and invariant conditional profiles by least squares or least chi-square (case 3), with improper models refused.
  Reproduces Tables 2 to 9 of the paper from the raw S&P counts.
- `pd.default_profile`: the chi-square test of the implied default profile with a seeded Monte Carlo p-value; it
  reproduces the p-values of Tasche (2013) Table 7. Policy threshold 5% and 1%, with its source.
- Docs: the regulatory and engines sections (four pages with the formulas from the primary sources, the
  reproductions and what not to claim), four computed and two hand-authored theme-aware figures, and
  `examples/credit_math.py`, run by the test suite.

### Fixed

- The QRRE PD floor of 0.10% applies to revolvers only; transactors take the retail 0.05% (CRE32.58), which is what
  CRE99 Table 1 tabulates.

## [0.01.000] - 2026-10-04

### Added

- Core contracts: the `TestResult` record with `Light` (green, amber, red, not evaluated, error), versioned
  `Policy` files with three threshold kinds, the test registry, `make_result`, a canonical inputs hash, and
  `run_suite`, which records a failing test and finishes the battery.
- Packaged default policy file, with no thresholds until the test catalogue lands.
- Guards: content standards (no em-dash, no emoji), CI budget (ADR-0074 shape), SDD gate (every requirement names
  a gate that exists).
- CI and the PyPI trusted-publishing workflow; the `docs/` wiki with the SDD, the U0 feature SDD and three
  architecture pages.
- 22 validation tests, each reproducing a published worked example or an independent reference:
  - PD calibration: Jeffreys per grade and portfolio (ECB 2019), binomial and Vasicek-corrected binomial with
    the BCBS WP14 critical counts (19, 35, 49, 63, 77), chi-square over grades, Hosmer-Lemeshow, Spiegelhalter,
    multi-period normal test, traffic lights by exact multinomial enumeration, Brier score with the Murphy
    decomposition, expected calibration error.
  - Discrimination: AUC with the ECB standard error and Gini, AUC against the development AUC (ECB 2019), the
    DeLong paired test, KS separation, a CAP accuracy ratio that equals 2 AUC - 1, a stratified bootstrap.
  - Rating system: matrix weighted bandwidth, migration z-tests with a Holm view, Herfindahl concentration test
    (ECB 2019).
  - Stability: PSI and CSI with the Yurdakul-Naranjo chi-square benchmark beside the conventional bands,
    chi-square homogeneity, two-sample KS, Wasserstein distance.
- Default policy thresholds (5% and 1%) for every p-value test, each with its stated source; descriptive
  measures stay unthresholded.
- Docs: the test catalogue and four family pages with formulas, null hypotheses, decision rules and primary
  sources; five figures, two computed by `scripts/make_doc_figures.py`.

### Changed

- Line length 120 for this repository.
