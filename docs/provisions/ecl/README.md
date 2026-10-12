# IFRS 9 expected credit loss

`riskvalidation.provisions` computes a loss allowance from the parameters a caller supplies and
`riskvalidation.validation.ifrs9` validates it. The engine computes and makes no compliance claim: it is not an IFRS 9
compliant or audited implementation, its outputs are not IFRS 7 disclosures, and the multipliers, floors, probation lengths,
horizons and acceptance gates it uses are policy parameters or Contraste internal gates, not requirements. Only two
presumptions are in IFRS 9: more than 30 days past due is a rebuttable presumption of a significant increase in credit risk
(5.5.11), and 90 days past due of default (B5.5.37). The threefold increase of the annualised PD is a supervisory expectation
(EBA/Rep/2023/36 para 32), not IFRS 9 (dossier 02 of the Contraste initiative, section I.5).

| Page | Contents |
|---|---|
| [01. PD term structures](01_term-structures.md) | the identities of hazard, survival, marginal and cumulative PD; competing risks; the forward PD and the annualisation; the one-factor TTC to PIT conversion with the 2.68% against 1.80% example |
| [02. Stage allocation](02_staging.md) | the relative and absolute PD test, the threefold backstop as a configurable parameter, the 30 and 90 days past due presumptions, forbearance, watch list, the low credit risk exemption, cure probation, the collective transfer |
| [03. Measurement](03_measurement.md) | ECL as a sum of discounted marginal losses; discounting; probability weighting and the non-linearity index; Stage 3, POCI, the simplified matrix, commitments; the portfolio allowance |
| [04. Overlays and reconciliation](04_overlays-and-reconciliation.md) | overlays as separate lines with a stage; the IFRS 7 35H and 35I roll-forwards and the identity check on one tolerance basis |
| [05. The validation battery](05_validation-battery.md) | the nine tests of dossier 02 I.2 with their light rules, and what must not be claimed |

Modules are imported by path, for example `from riskvalidation.provisions import ecl` and
`from riskvalidation.validation import ifrs9`; nothing is re-exported from the package root. Requirements `RV-501` to
`RV-519` are in `docs/design/features/u5-provisions/requirements.md`, each naming its test.
