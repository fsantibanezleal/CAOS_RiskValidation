# IRRBB: interest-rate risk in the banking book

`riskvalidation.irrbb` computes the zero curves and curve fits, the six SRP31 shocks and the size tables, the 19-bucket slotting,
the standardised EVE and NII, the RAN 21-13 Annex 1 variant (CLP and UF), the behavioural models of non-maturity deposits and
prepayments, and the dynamic NII and stochastic EaR and EVaR simulations; `riskvalidation.validation.irrbb` validates them. The
engine computes and makes no compliance claim: thresholds, floors a supervisor may choose and light parameters are supplied by
the caller or are Contraste internal gates, not regulatory requirements.

Conventions of every page:

- A positive delta EVE or delta NII is a **loss** (SRP31.129; dossier 01 H.4). Only losses are aggregated across currencies.
- Rates are decimals and discounting is continuously compounded, $DF(t)=e^{-R(t)\,t}$ (SRP31.129). Shocks are basis points.
  Bucket midpoints are years. Money is a float in the currency of the call; the engine never converts between currencies (the
  UF amount is a CLF amount that the caller converts with the UF value of the date).
- The shock order is `SHOCKS = (parallel-up, parallel-down, steepener, flattener, short-up, short-down)`, the order of SRP31.119
  Table 5 and SRP31.125 Table 6.
- Every function is pure and deterministic (a function with randomness takes a `seed` and uses `numpy.random.default_rng`),
  never mutates its inputs, and raises `ValueError` with the reason when data is missing; a validation test returns the light
  `not_evaluated` with a null statistic, metric and p-value instead. A shock size outside the transcribed tables is refused
  unless the caller supplies it with its label.
- Every constant and table is transcribed from the dossier named on its page. An item the dossiers mark UNVERIFIED, or that the
  contract reads in a way the dossier does not settle, is marked UNVERIFIED here and is not silently decided.

| Page | Contents |
|---|---|
| [01. Curves](01_curves.md) | zero rates and discount factors, the bootstrap, Nelson-Siegel, Svensson and the dynamic Nelson-Siegel with the Diebold-Li decay |
| [02. Shocks](02_shocks.md) | the six SRP31 shapes, the 2024 and 2016 size tables, the CMF Hoja 9 table, the post-shock floors, shock sizes with their origin |
| [03. Slotting](03_slotting.md) | the 19 buckets, notional repricing cash flows, the non-maturity deposit split with the SRP31 caps, prepayment, SMM, CPR and PSA, term-deposit redemption |
| [04. EVE and NII](04_eve-and-nii.md) | the repricing gap, the duration gap, the standardised EVE and its risk measure, delta NII, the automatic-option add-on, the outlier test |
| [05. Annex 1](05_annex1.md) | the CLP and UF pair with the 50 percent offset, the Annex 1 delta NII and delta EVE, the indexation charge, the prioritised-bank tests |
| [06. Behavioural models](06_behavioural.md) | the replicating portfolio, the weighted average maturity, the deposit beta, the prepayment logit and its maximum-likelihood fit |
| [07. Simulation](07_simulation.md) | the dynamic NII projection, Hull-White one-factor and dynamic Nelson-Siegel scenarios, EaR and EVaR |
| [08. The validation tests](08_validation-tests.md) | the seven tests with their light rules, and what must not be claimed |

Modules are imported by path, for example `from riskvalidation.irrbb import eve_nii` and
`from riskvalidation.validation import irrbb`; nothing is re-exported from the package root. Requirements `RV-1001` to
`RV-1032` are in `docs/design/features/u10-irrbb/requirements.md`, each naming its test. Sources: dossier 01 of the Contraste
initiative (section H), dossier 03 (sections F.1 and F.2), dossier 05 (sections C.4 and D), and dossier 08 (C.2); each page cites
the primary sources with their DOIs or URLs. Contract: `engine-irrbb-api` 1.1.0.
