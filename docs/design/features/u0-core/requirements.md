# U0 core contracts, requirements

EARS statements, each with the gate that fails when it is violated.

| ID | Requirement | Gate |
|---|---|---|
| RV-001 | THE `TestResult` record SHALL carry a non-empty test id, null hypothesis, reference and policy version, and a non-error result SHALL carry a statistic, a p-value or a metric. | `tests/test_result_schema.py::test_required_fields` |
| RV-002 | IF a test has no thresholds in the policy, THEN THE result SHALL be recorded as not evaluated with no thresholds and no colour. | `tests/test_policy.py::test_missing_policy_no_light` |
| RV-003 | WHILE running a suite, WHEN a test raises, THE runner SHALL record an error result with the exception text and continue with the next test. | `tests/test_suite.py::test_runner_never_aborts` |
| RV-004 | THE thresholds SHALL be read from a versioned policy file, and every result SHALL carry the policy id and version that classified it. | `tests/test_policy.py::test_policy_version_carried` |
| RV-005 | THE inputs hash SHALL be identical for equal inputs and SHALL change when any value, dtype, shape or order changes. | `tests/test_hashing.py::test_hash_deterministic_and_sensitive` |
| RV-006 | THE core package SHALL import without loading any module beyond NumPy and SciPy (no pandas, scikit-learn, statsmodels, torch or plotting). | `tests/test_imports.py::test_core_imports_without_optional_deps` |
| RV-007 | THE classifier SHALL assign red strictly below the red p-value cut, amber strictly below the amber cut, and the inclusive bounds documented for metric thresholds. | `tests/test_policy.py::test_threshold_boundaries` |
