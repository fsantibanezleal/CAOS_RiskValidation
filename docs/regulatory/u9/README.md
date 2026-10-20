# U9 regulatory calculators

Pages of the engine modules `riskvalidation.regulatory.standardised`, `output_floor`, `sa_ccr`, `ba_cva` and `frtb` (unit U9-E1). They implement the
engine API contract `engine-regulatory-api` 1.0.0 (Contraste `docs/contracts/engine/regulatory-api.md`, with the regulatory-calculators fixture),
transcribed from dossier 01 of the Contraste initiative (sections C, C.1, D.2, D.3, E.1, E.2, M.1 and M.3) and the U9 research addendum (R7, R8).
Every parameter a function returns carries its paragraph, its table and the in-force date of the chapter version. The engine computes and states no
compliance claim: it reproduces published formulas and tables.

| Page | Module | Content |
|---|---|---|
| [01 Standardised credit risk weights](01_standardised-credit.md) | `standardised` | CRE20 tables as data, the lookup |
| [02 The output floor](02_output-floor.md) | `output_floor` | RBC20.4, the RBC90.1 phase-in and the RBC90.2 cap |
| [03 SA-CCR](03_sa-ccr.md) | `sa_ccr` | CRE52, the five CRE99 netting sets from their trades |
| [04 BA-CVA](04_ba-cva.md) | `ba_cva` | MAR50.14 to MAR50.25 |
| [05 FRTB SBM and the liquidity-horizon cascade](05_frtb-sbm-and-cascade.md) | `frtb` | MAR21 with GIRR, MAR33.4 |

The requirements, each naming its test, are in `docs/design/features/u9-regulatory/requirements.md`. LCR and NSFR are not here (U11), the internal
ratings-based formulas are in [the IRB page](../01_irb.md), and the standardised measurement approach to operational risk is `riskvalidation.oprisk.sma`.
