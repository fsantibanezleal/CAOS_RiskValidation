# Changelog

All notable changes are recorded here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use the display form `X.XX.XXX`; the package
metadata carries the PEP 440 form (`0.01.000` is `0.1.0`).

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
  correlation by Gauss-Hermite quadrature; reproduces the paper's Tables 1 to 7 and documents a misprint in Table 4.
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
