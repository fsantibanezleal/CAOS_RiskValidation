# Fairness: group metrics, inference, conditional disparity, replication, situation testing and planted bias

`riskvalidation.fairness` computes the group fairness metrics of a credit decision (selection rates, demographic parity, equalized
odds, equal opportunity, the four-fifths flag, calibration and AUC within groups), the tests on two selection rates, the disparity
that persists after the permitted controls and its replication on the next year, situation testing with nearest-neighbour
comparators, and the planted-bias generators that measure the size and power of each test. `riskvalidation.validation.fairness`
returns each as a `TestResult` with a traffic light. The engine states **no legal claim**: the four-fifths line is an
employment benchmark (29 CFR 1607.4(D)) and never a credit-law threshold, and a controlled disparity is an approximation of
comparative evidence with its missing controls named, never a finding of a violation. The contract is `engine-fairness-api`
1.0.0 (Contraste `docs/contracts/engine/fairness-api.md`), amended to 1.1.0 after the Opus review of the engine: red capped at
amber on the two 'never red' lights, `protected_effect` refusing the intercept, the `categorical` keyword of `situation_test`, the
situation-testing amber cut 0.20.

Conventions of every page:

- A decision is 1 for the favourable outcome (approval) and 0 otherwise; the group label of a row is read as `str(label)` and groups
  are ordered by label (numeric labels numerically). Situation testing reads the group as 1 for the protected group and 0 for the
  others and the decision 0 as the negative one.
- **Exact arithmetic on counts.** A comparison of two rates that sits exactly on a convention is decided on the integer counts and
  rounded once: 60 of 100 over 75 of 100 is `0.7999999999999999` in IEEE doubles and exactly 4/5.
- **No outcome, no outcome-conditioned test.** Equalized odds, equal opportunity, calibration within groups and AUC by group need an
  outcome. A dataset that declares no outcome (public HMDA has none) never runs them: the metrics raise `ValueError` naming `y`, the
  validation tests return `not_evaluated`.
- Every function is pure and deterministic (randomness takes a seed), never mutates its inputs and imports only NumPy and SciPy.
  Malformed input raises `ValueError` naming the field; thin data (one group, a group without both classes) is `InsufficientData`,
  a `ValueError` that a validation test turns into `not_evaluated` with a null statistic, metric and p-value.
- Every tolerance is a convention, never a regulatory threshold (dossier 06 I.6 l.1268-1270). The lights come from `Policy` rows
  keyed by test id (`riskvalidation.fairness.policy`); the values are named parameters for approval.
- Optional references: fairlearn 0.15.0 (the parity and equalized-odds reconciliation) and statsmodels (the z-test, Logit and OLS
  references). Neither is imported by the engine; their tests are skipped with the reason when absent.

| Page | Contents |
|---|---|
| [01. Metrics](01_metrics.md) | selection rates, parity difference and ratio, equalized odds and equal opportunity, the adverse impact ratio, the exact four-fifths flag, calibration and AUC by group |
| [02. Inference](02_inference.md) | the pooled two-proportion z-test and Fisher's exact test |
| [03. Conditional disparity and replication](03_conditional-and-replication.md) | the logistic decision model, the rate-spread model, the protected effect with its Wald interval and the year-to-year replication |
| [04. Situation testing](04_situation-testing.md) | the Gower distance with two normalizations, k nearest neighbours, the statistic, the one-sided Wald interval, ceteris paribus and mutatis mutandis, the golden values |
| [05. Planted bias](05_planted-bias.md) | the threshold-bias, label-noise and proxy-feature generators with their exact truths |
| [06. The validation tests](06_validation-tests.md) | the twelve `fair.` tests, their lights and policy rows, and the measured size and power |
