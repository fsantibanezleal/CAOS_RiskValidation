# CCR: counterparty credit risk, exposure, collateral, CVA and wrong-way risk

`riskvalidation.ccr` computes the Hull-White one-factor exposure of an interest rate swap netting set: the affine bond prices and the
closed forms (the zero-coupon bond option and the Jamshidian swaption), the paths of the short rate, the swaps along the paths, the
exposure measures (EE, ENE, the effective EE and EPE, PFE, netting, the effective maturity), the variation margin with its margin
period of risk, the dynamic initial margin, and CVA, DVA and wrong-way risk. `riskvalidation.validation.ccr` validates them. The engine
computes and makes no compliance claim: thresholds and light parameters are supplied by the caller or are Contraste internal gates,
not regulatory requirements. It is pure NumPy and SciPy: ORE and QuantLib never enter it, and the reconciliation against ORE 1.8.17.0
and the QuantLib 1.44 check of the closed forms belong to the case bake that calls the engine.

Conventions of every page:

- Arrays are `float64`. `P` paths are rows and `K` grid points are columns. Money is a float in the currency of the netting set (the
  currency code travels in the document, never in a function). Rates are decimals compounded continuously, times are years from the
  valuation date, and the margin period of risk is in business days with 250 to the year.
- **Sign convention.** EE is the positive exposure to the counterparty, ENE is the bank's own, reported positive; CVA is a positive cost
  and DVA a positive benefit. A swap value is to the bank (positive is an asset).
- Every function is pure and deterministic (a function with randomness takes a `seed` and uses `numpy.random.default_rng`), never
  mutates its inputs, and raises `ValueError` naming the field when an input is missing or invalid; a validation test returns the light
  `not_evaluated` with a null statistic, metric and p-value instead of a placeholder number.
- The counterparty ladder of dossier 05: rung 1 current exposure plus add-on is SA-CCR's RC plus PFE (decision D3, not re-implemented;
  `exposure.rung1_ead` is a thin wrapper of `riskvalidation.regulatory.sa_ccr`); rung 2 the analytic EE of a single swap
  (`hw1f` and `exposure.analytic_ee_single_swap`); rung 3 the Monte Carlo netting set; rung 4 collateral with the MPoR and initial
  margin; rung 5 wrong-way risk.
- Every constant and formula is transcribed from the record named on its page. An item the record marks UNVERIFIED, or that the
  contract reads in a way the record does not settle, is marked UNVERIFIED here and is not silently decided.

| Page | Contents |
|---|---|
| [01. Hull-White one-factor](01_hw1f.md) | the fit to today's curve, the affine bond price, the simulation of the state and of the short rate, the bond option, the Jamshidian swaption, the AR(1) calibration |
| [02. Swaps](02_swaps.md) | the fixed-float swap, its cash-flow dates, the value at any date including between two resets, the par rate, the values along the paths |
| [03. Exposure measures](03_exposure.md) | EE and ENE, the effective EE and EPE, PFE, netting, the effective maturity, the analytic EE of a single swap, rung 1 |
| [04. Collateral and initial margin](04_collateral.md) | the credit support amount, the delivery amount, the MPoR floors, the close-out exposure, the dynamic initial margin as a regression proxy |
| [05. CVA and wrong-way risk](05_cva-and-wrong-way-risk.md) | the credit triangle, CVA and DVA, the Hull and White hazard model and the calibration of its deterministic part |
| [06. The validation tests](06_validation-tests.md) | the eight tests with their light rules, the PIT backtest with simulated critical values, and what must not be claimed |

Modules are imported by path, for example `from riskvalidation.ccr import hw1f` and `from riskvalidation.validation import ccr`;
nothing is re-exported from the package root. Requirements `RV-950` to `RV-986` are in `docs/design/features/u9-ccr/requirements.md`,
each naming its test. Sources: dossier 05 of the Contraste initiative (section C and the counterparty ladder), the U9 research addendum
(R1 wrong-way risk, R2 initial margin, R3 the CSA and the MPoR, R4 the Hull-White closed forms, R5 the exposure backtest, R6 the AR(1)
calibration, R9 the CRE53 definitions); each page cites the primary sources with their URLs. Contract: `engine-ccr-api` 1.0.0.
