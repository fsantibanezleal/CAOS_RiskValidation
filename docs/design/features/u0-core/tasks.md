# U0 core contracts, tasks

| # | Task | Requirement |
|---|---|---|
| 1 | `TestResult` and `Light` with field validation and JSON export | RV-001 |
| 2 | `Threshold` kinds and boundaries; `Policy`, JSON loader, packaged default | RV-002, RV-004, RV-007 |
| 3 | registry and `make_result` | RV-001, RV-002 |
| 4 | canonical inputs hash | RV-005 |
| 5 | suite runner that records errors and continues | RV-003 |
| 6 | import isolation of the core | RV-006 |
| 7 | guards (content standards, CI budget, SDD), CI and publish workflows, docs | all |

## Convergence verdict (2026-10-04)

| Requirement | Gate | Result |
|---|---|---|
| RV-001 | `tests/test_result_schema.py::test_required_fields` | pass |
| RV-002 | `tests/test_policy.py::test_missing_policy_no_light` | pass |
| RV-003 | `tests/test_suite.py::test_runner_never_aborts` | pass |
| RV-004 | `tests/test_policy.py::test_policy_version_carried` | pass |
| RV-005 | `tests/test_hashing.py::test_hash_deterministic_and_sensitive` | pass |
| RV-006 | `tests/test_imports.py::test_core_imports_without_optional_deps` | pass |
| RV-007 | `tests/test_policy.py::test_threshold_boundaries` | pass |

Also run: ruff (clean), `scripts/check_ci_budget.py`, `scripts/check_content_standards.py`, `scripts/check_sdd.py`
(all OK), and `python -m build` (the wheel contains `policies/default.json`). Converged.
