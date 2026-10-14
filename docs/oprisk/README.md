# Operational risk

`riskvalidation.oprisk` computes the loss distribution approach (LDA) of dossier 05 F and the Basel III standardised approach
(SMA) of OPE25: the frequency and severity laws and their truncated fits, the compound annual loss by Monte Carlo, Panjer
recursion and FFT, the single loss approximation, the scenario combination, the business indicator, the BIC and the internal loss
multiplier; `riskvalidation.validation.oprisk` validates them. The engine computes and makes no compliance claim: the LDA is the
economic-capital and validation tool of dossier 05 F.1, never a Pillar 1 model, and collection thresholds, quantile levels and
light parameters are policy parameters the caller supplies or Contraste internal gates, not regulatory requirements. Money is a
float in the caller's currency (Contraste uses EUR throughout); the engine never converts. A tail index of 1 or more has no
mean: no function returns a finite mean-based number for it (`severity_moment` returns `inf`, `sla_var_mean_corrected` and the
expected shortfall refuse), and a value that cannot be computed is an error or a `not_evaluated` result, never a placeholder.

| Page | Contents |
|---|---|
| [01. Distributions and truncation](01_distributions-and-truncation.md) | the frequency and severity laws, the lognormal and GPD splice, the truncated likelihood and the rescaled frequency, the fits, the tail-index profile |
| [02. The aggregate loss](02_aggregate-loss.md) | Monte Carlo, Panjer recursion, FFT, the quantile and expected shortfall read from the pmf, Shevchenko's benchmark and its table values |
| [03. SLA and scenarios](03_sla-and-scenarios.md) | the single loss approximation, its UNVERIFIED mean-corrected form, the scenario combination and the Bayesian credibility of Shevchenko and Wuthrich |
| [04. The standardised approach](04_sma-ope25.md) | OPE25: the business indicator, the BIC, the loss component, the ILM, the discretions and the five-year rule |
| [05. The validation tests](05_validation-tests.md) | the five tests with their light rules, and what must not be claimed |

Modules are imported by path, for example `from riskvalidation.oprisk import aggregate` and
`from riskvalidation.validation import oprisk`; nothing is re-exported from the package root. Requirements `RV-1201` to
`RV-1225` are in `docs/design/features/u12-oprisk/requirements.md`, each naming its test. Sources: dossier 05 of the Contraste
initiative (section F and the operational risk ladder and case O1), dossier 01 (section F, OPE25) and Shevchenko (2010),
cited with their DOIs on each page. An item the dossiers mark UNVERIFIED is marked UNVERIFIED here and is not silently decided.
Contract: `engine-oprisk-api` 1.0.0.
