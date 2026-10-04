# riskvalidation

[![CI](https://github.com/fsantibanezleal/CAOS_RiskValidation/actions/workflows/ci.yml/badge.svg)](https://github.com/fsantibanezleal/CAOS_RiskValidation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Validation tests, regulatory calculators and reference engines for financial risk models: the toolkit of an
independent model validator.

Every test returns one `TestResult` that carries the statistic, the null hypothesis, the primary source that
defines the test, the policy that classified it and a fingerprint of the inputs. Thresholds live in versioned
policy files and are presented as policy, never as regulation. The core depends only on NumPy and SciPy, so the
same wheel runs in a Python pipeline and in the browser under Pyodide.

## Status

Version 0.01.000 is in development: the core contracts (result record, policy, registry, suite runner) are in
place. The test catalogue, regulatory calculators and reference engines arrive unit by unit; the
[CHANGELOG](CHANGELOG.md) lists exactly what each release contains. Not yet published on PyPI.

## Install (from source, until the first PyPI release)

```bash
git clone https://github.com/fsantibanezleal/CAOS_RiskValidation.git
cd CAOS_RiskValidation
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Use

```python
from riskvalidation import run_suite, load_policy, catalogue

policy = load_policy("my_policy.json")       # an institution's thresholds, versioned
for spec in catalogue():                     # every registered test with its reference
    print(spec.test_id, spec.reference)
```

## Documentation

The [`docs/`](docs/README.md) wiki: the software design document, the architecture (result contract, policies
and lights, suites and reproducibility) and, as each unit ships, one page per test family with its formula,
null hypothesis, decision rule and primary source.

## What this is not

A compliance verdict on any model, a training framework for heavy learners, or a data service. A green light
means a statistic fell inside a policy band.

## License

MIT, see [LICENSE](LICENSE). Developed by Felipe Santibáñez-Leal.
