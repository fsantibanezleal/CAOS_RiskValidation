# Changelog

All notable changes are recorded here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use the display form `X.XX.XXX`; the package
metadata carries the PEP 440 form (`0.01.000` is `0.1.0`).

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
