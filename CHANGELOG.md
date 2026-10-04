# Changelog

All notable changes are recorded here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use the display form `X.XX.XXX`; the package
metadata carries the PEP 440 form (`0.01.000` is `0.1.0`).

## [Unreleased]

### Added

- Core contracts: the `TestResult` record with `Light` (green, amber, red, not evaluated, error), versioned
  `Policy` files with three threshold kinds, the test registry, `make_result`, a canonical inputs hash, and
  `run_suite`, which records a failing test and finishes the battery.
- Packaged default policy file, with no thresholds until the test catalogue lands.
- Guards: content standards (no em-dash, no emoji), CI budget (ADR-0074 shape), SDD gate (every requirement names
  a gate that exists).
- CI and the PyPI trusted-publishing workflow; the `docs/` wiki with the SDD, the U0 feature SDD and three
  architecture pages.
