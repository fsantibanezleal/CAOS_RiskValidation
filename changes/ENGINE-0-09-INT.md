- Engine 0.09.000 integration of the governance unit under the 0.08.000 catalogue rules: the five `gov.` tests (`gov.inventory_completeness`,
  `gov.findings_overdue`, `gov.validation_due`, `gov.tier_override_documented`, `gov.edges_acyclic`; all decided by a metric gate, so none
  needs a size suite) are in the registry inventory (family `governance`, prefix `gov`), and the five `gov.` rows are in the packaged
  `policies/default.json`, written from `GOV_ROWS` and `gov_threshold` (30 lines added, no existing row changed, label 2026.10.1 unchanged).
  `gov_default_policy()` now returns the default policy itself; the `+gov-` label branch of `resolve_policy` stays and is harmless.
  The governance report keeps reproducing the Contraste fixtures except the `kind` and `kind_reason` of 11 (C07) and 17 (C18) report rows
  whose test ids the fixture wrote as unregistered at the batch 3 head; the engine registers them since 0.08.000 and states the decision.
