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

Version 0.09.000: model governance (the scope test, the inventory and its dependency graph, tiering, the lifecycle, the findings register, the review schedule, the overall opinion, the risk appetite metrics, the policy identity of a test row and the validation report), with its five `gov.` validation tests; 0.08.001: the fairness metric rows carry null alphas, their cut points in extras; 0.08.000: liquidity (run-off, deposits, the counterbalancing capacity, the maturity ladder and reverse liquidity stress), aggregation (simple sum and variance-covariance, copulas, bounds and the dependence band, aggregation model risk), stress testing (satellites, scenario models, path transposition, the capital path, reverse stress), the regulatory calculators (the standardised approach, IRB, the output floor, SA-CCR, BA-CVA, the FRTB sensitivities-based method and liquidity-horizon cascade, LCR and NSFR), counterparty credit risk (Hull-White one-factor exposure, swaps, collateral, CVA), machine-learning model validation and fair lending, each with its validation tests; 0.07.000: portfolio credit capital (copula Monte Carlo, CreditRisk+, the granularity adjustment, importance sampling, Euler contributions, the Vasicek fit), market risk (VaR and ES estimators, scoring, the model confidence set, the FRTB back-tests and PLA), interest rate risk in the banking book (SRP31 and RAN 21-13 Annex 1 EVE and NII, behavioural models, EaR and EVaR) and operational risk (LDA by Monte Carlo, Panjer and FFT, the SMA), each with its validation tests; 0.06.000: loss given default and exposure at default (workout LGD, the LGD rungs, downturn LGD, CCF and EAD, the IRB floors, and their validation tests); 0.05.000: provisions (the IFRS 9 expected credit loss engine with its validation battery, and the CMF B-1 and BdE Annex 9 statutory provisions); before it, the core contracts (result record, policy, registry, suite runner); 27 validation tests for
PD calibration, discrimination, rating-system stability, population stability and rating transitions; the IRB
risk-weight functions under four regimes; three reference engines (the one-factor model with its large homogeneous
portfolio, most prudent estimation for low-default portfolios, PD curve calibration after Tasche 2013); rating
transitions (cohort, duration, Aalen-Johansen and EM estimators, the embedding problem, intervals for PDs by grade,
mobility, the through-the-cycle portfolio a migration matrix implies); and known-truth generators, among them rating
paths with planted momentum, regimes and withdrawals, with a size and power harness that measures how often each test
rejects a right model and catches a wrong one, with the Monte Carlo error of every rate. Each reproduces its primary
source's own tables, worked examples or published simulations. Further calculators and engines arrive unit by unit; the
[CHANGELOG](CHANGELOG.md) lists exactly what each release contains.

## Install

```bash
pip install riskvalidation
```

The public API, its semantic-versioning policy and its deprecation rule are in [docs/api-policy.md](docs/api-policy.md). From source:

```bash
git clone https://github.com/fsantibanezleal/CAOS_RiskValidation.git
cd CAOS_RiskValidation
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Use

[`examples/quickstart.py`](examples/quickstart.py), run by the test suite:

```python
import numpy as np

from riskvalidation import default_policy, run_suite
from riskvalidation.validation import (
    bin_counts,
    disc_auc_vs_initial,
    pd_jeffreys_grades,
    quantile_edges,
    stability_psi,
)

rng = np.random.default_rng(2026)

# Jeffreys test per grade and for the portfolio (ECB 2019, section 2.5.3.1)
for r in pd_jeffreys_grades(n=[400, 900, 300], d=[2, 14, 19], pd=[0.004, 0.012, 0.05], grades=["A", "B", "C"]):
    print(f"{r.segment:>9}  p = {r.p_value:.4f}  {r.light.value}")

# a scored sample: PDs and observed defaults, plus a development-period score sample
pd_scores = rng.beta(1.2, 40.0, 5000)
defaults = (rng.uniform(size=5000) < pd_scores).astype(int)
development = rng.beta(1.2, 45.0, 8000)
edges = quantile_edges(development, 10)

# a battery under one policy; a failing test is recorded and the rest still run
run = run_suite(
    [
        (disc_auc_vs_initial, {"auc_initial": 0.78, "risk": pd_scores, "defaults": defaults}),
        (stability_psi, {"expected_counts": bin_counts(development, edges),
                         "actual_counts": bin_counts(pd_scores, edges)}),
    ],
    policy=default_policy(),
    model_id="retail-pd-v3",
)
for r in run.results:
    print(f"{r.test_id:>22}  statistic = {r.statistic:.4f}  p = {r.p_value:.4f}  {r.light.value}")
print("lights:", run.counts(), "worst:", run.worst().value)
```

## Documentation

The [`docs/`](docs/README.md) wiki: the software design document, the architecture (result contract, policies
and lights, suites and reproducibility), one page per test family with its formula, null hypothesis, decision rule
and primary source, the [regulatory calculators](docs/regulatory.md), the [reference engines](docs/engines.md),
[rating transitions](docs/transitions.md) and the [generators and the size and power harness](docs/generators.md). [`examples/size_power.py`](examples/size_power.py)
measures the size and power of a few tests in seconds.
[`examples/credit_math.py`](examples/credit_math.py) runs the credit mathematics end to end: IRB capital under two
regimes, the large homogeneous portfolio behind it, most prudent PDs, and a PD curve calibrated to a new year and
tested for fit. [`examples/rating_transitions.py`](examples/rating_transitions.py) estimates, embeds, tests, bounds
and projects a migration matrix on a known generator, then plants rating momentum and shows which tests see it.

## What this is not

A compliance verdict on any model, a training framework for heavy learners, or a data service. A green light
means a statistic fell inside a policy band.

## License

MIT, see [LICENSE](LICENSE). Developed by Felipe Santibáñez-Leal.
