- The thirteen `ml.` validation tests (`ml.robustness`, `ml.shift_resilience`, `ml.seed_stability`,
  `ml.reason_code_stability`, `ml.effect_agreement`, `ml.additivity`, `ml.overfit`, `ml.weak_spot`,
  `ml.conformal_coverage`, `ml.leakage_screen`, `ml.data_quality`, `ml.cbpe_estimate`, `ml.challenger_brier`) are
  registered in the test catalogue (family `ml`) with the `h0` and the `reference` their row carries; the two p-value
  tests (`ml.shift_resilience`, `ml.conformal_coverage`) are named to their size suite. No result changes.
- `riskvalidation.validation.ml` registers on `import riskvalidation.validation.ml`; `import riskvalidation.validation`
  does not import it, as before.
