- The test catalogue (`riskvalidation.core.registry`) now lists every validation test of the aggregation, ccr, fairness, ifrs9,
  irrbb, lgd_ead, liquidity, market, oprisk, portfolio and stress modules (89 test ids), each with its family, name, hypothesis,
  reference and decision (`p_value` where the light is decided by a p-value against alphas, `metric` otherwise), so that
  `spec(test_id)` answers for every row the engine writes. No function body, signature or result changes. `import
  riskvalidation.validation` registers them all (the package imports the eleven modules; no optional dependency is loaded).
  `ccr.pfe_exceedance_backtest` holds its hypothesis with `{q}` and `{1 - q}` where the row writes the quantile and its
  complement; `liq.reverse_stress_round_trip` writes a second hypothesis on its threshold branch.
- The engine's own tests check, over the whole session, that every `TestResult` an engine function builds names a registered test
  with the `h0` and `reference` of its spec; the size rule now accepts a measurement in `NULL_SCENARIOS` or in a named `sizepower`
  suite (`SIZE_SUITES`) and pins the registered p-value tests that have none (`SIZE_GAPS`).
