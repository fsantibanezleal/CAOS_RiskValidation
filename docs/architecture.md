# Architecture

How a validation test becomes evidence: a statistic computed from the caller's inputs, classified under a named
policy, recorded with its source and a fingerprint of what was tested.

![From inputs to a classified result](assets/result-flow.svg)

| Page | Topic |
|---|---|
| [01_result-contract.md](architecture/01_result-contract.md) | the `TestResult` record and why each field exists |
| [02_policy-and-lights.md](architecture/02_policy-and-lights.md) | thresholds as policy, the three threshold kinds, what the literature says about conventional bands |
| [03_suite-and-reproducibility.md](architecture/03_suite-and-reproducibility.md) | running a battery, error handling, inputs hashing, the Pyodide lane |

## Core modules

`riskvalidation.core` holds the contracts every family uses. Its modules are public by path and each lists its names
in its own `__all__` (the full list is in [api-policy.md](api-policy.md)):

| Module | Holds |
|---|---|
| `result` | `TestResult` and the `Light` enumeration |
| `policy` | `Policy`, `Threshold`, `default_policy`, `load_policy`, `policy_from_mapping` (a policy from the mapping of a policy file), `packaged_policy` and `packaged_labels` (the policies packaged with the engine) |
| `registry` | the test catalogue: `register`, `spec`, `catalogue` |
| `suite` | `run_suite` and `SuiteRun` |
| `hashing` | the deterministic fingerprint of the inputs of a test |
| `build` | `make_result`, which assembles a `TestResult` from a statistic and a policy |
| `deprecation` | the helpers of the deprecation rule: `deprecated`, `deprecated_parameter` and `module_getattr` |
