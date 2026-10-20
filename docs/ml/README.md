# ML model validation

`riskvalidation.ml` implements the ML validation tests of the dossiers on a model that enters as a predict callable on arrays:
perturbation robustness and shift resilience, seed stability, the explanation checks, partial dependence, ICE and accumulated
local effects, the overfit gap and the weak spots, split conformal prediction, the leakage screen, the data-quality counts, the
label-free performance estimate, the champion and challenger comparison and the planted-defect generators;
`riskvalidation.validation.ml` turns them into `TestResult` rows. The engine imports no learning library (LightGBM, XGBoost, EBM
and `shap` stay in the pipeline, which passes predictions and attributions as arrays). It computes and makes **no compliance
claim**: tolerances, grids, counts and cut points are conventions, never regulatory thresholds (dossier 06 I.6 l.1268-1270).

The contract is `engine-ml-api` 1.0.0 (Contraste `docs/contracts/engine/ml-api.md`, branch `task/u14-s`, reviewed by Opus and
fixed on 2026-10-19). Sources: dossier 06 (model risk management and validation) C.7 l.979-1058 and I.1 l.1184-1200, dossier 04
(credit risk models) J.2 l.881-883 and ladder P4 l.968, dossier 07 (SantanderAI and bank open source) D patterns 3 and 6.

Conventions of every page:

- Arrays are `float64` (integers `int64`, flags `bool`); `n` rows, `p` features. Every function is pure and deterministic (a
  function with randomness takes a `seed` and uses `numpy.random.default_rng`), never mutates its inputs, and raises
  `ValueError` naming the field when data is missing or invalid; a validation test returns the light `not_evaluated` with a
  null statistic, metric and p-value instead.
- A rule stated as "exceeds" (a tolerance) keeps a value equal to the tolerance in the better colour: the policy row stores
  `math.nextafter(cut, inf)`, so a light and the strict boolean (`robustness_finding`, `overfit_gap`) agree at the boundary.
- The lights come from `riskvalidation.core.policy.Policy`; the rows of the contract's Policy thresholds table are in
  `riskvalidation.ml.policy` and added to a base policy only where it holds no row for the key (page 09).
- Constants the contract and the dossiers do not name are fixed in one place and labelled conventions: the ridge strength 1 and
  the 5 cross-fitting folds of the default logistic fit (page 01), the interval level 0.95 of the paired bootstrap (page 07), the
  design constants of the plants (page 08).

| Page | Contents |
|---|---|
| [01. Robustness and shift](01_robustness.md) | Gaussian input perturbation, the AUC and Brier curve, the finding, the shift score, the worst subsample, out-of-time windows |
| [02. Seeds and explanations](02_seeds-and-explanations.md) | seed dispersion, grade and rank stability, the reason-code overlap, the explainer agreement, additivity, sign and monotonicity |
| [03. Effects](03_effects.md) | partial dependence, ICE and accumulated local effects |
| [04. Generalisation](04_generalisation.md) | the overfit gap and the weak spots |
| [05. Conformal prediction](05_conformal.md) | the split conformal quantile and sets, the coverage test, the MAPIE reconciliation |
| [06. Leakage and data quality](06_leakage-and-data-quality.md) | temporal order, single-feature AUC, entity overlap, the resemblance AUC, the six counted data-quality checks |
| [07. CBPE and challenger](07_cbpe-and-challenger.md) | the label-free performance estimate, the paired Brier bootstrap, the grade disagreement |
| [08. Size and power](08_size-and-power.md) | the planted-defect generators and the measured size and power of every test |
| [09. The validation tests](09_validation-tests.md) | the `ml.` test ids, their light rules, the policy rows and the open points |

Findings of unit U14-E1 against the contract (each is stated again on its page): F1 the 5th percentile of the reason-code overlap is
not bounded by the mean (page 02); F2 the float product of the split-conformal rank misrounds for some `(n, alpha)` and the
contract's two examples are not among them (page 05); F3 MAPIE's classification quantile is the k-th or the (k + 1)-th order
statistic (page 05); F4 the share of the worst-subsample test has a method-inherent false-alarm rate under no shift (page 08), closed by the development-only
reference (page 01). The fix round of the review (engine-ml-api 1.1.0) adds the Beta-Binomial null of the conformal coverage test
(page 05), the signed reason codes (page 02), the always-`not_evaluated` CBPE row (page 07) and the 12-decimal strict boundaries
(page 09).
