# Changelog

All notable changes are recorded here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use the display form `X.XX.XXX`; the package
metadata carries the PEP 440 form (`0.01.000` is `0.1.0`).

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
