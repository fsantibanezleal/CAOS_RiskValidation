# Registry complete: every validation test of the engine is in the catalogue

EARS statements, each with the gate that fails when it is violated. Scope: the eleven modules aggregation, ccr, fairness,
ifrs9, irrbb, lgd_ead, liquidity, market, oprisk, portfolio and stress (89 test ids); `validation/ml.py` registers in its own unit.

| ID | Requirement | Gate |
|---|---|---|
| RV-1500 | THE catalogue SHALL list every test id the eleven modules write, with the family, name, hypothesis, reference and decision of the function; WHEN any engine function builds a `TestResult`, THE row SHALL name a registered test and SHALL carry the `h0` and the `reference` of its spec; AND every registered p-value test SHALL have its size measured in `NULL_SCENARIOS` or in a named `sizepower` suite of `SIZE_SUITES`, or be pinned in `SIZE_GAPS` with its reason. | `tests/test_registry_complete.py::test_every_result_names_a_registered_test_session_wide`, `tests/test_registry_complete.py::test_every_inventory_id_is_registered_with_its_family_function_and_decision`, `tests/test_size.py::test_every_registered_p_value_test_has_its_size_measured_or_named_as_a_gap` |
