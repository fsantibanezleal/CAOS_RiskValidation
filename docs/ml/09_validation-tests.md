# 09. The validation tests

Module `riskvalidation.validation.ml`; cut points in `riskvalidation.ml.policy`. Sources: dossier 06 C.7 and I.1 (the ML test
catalogue, l.1184-1200), I.6 (what not to claim, l.1261-1277), the contract `engine-ml-api` 1.0.0 (Policy thresholds). Each
function returns one `TestResult` (`riskvalidation.core.result`) that serialises into a row of `common/test-result`, with the id
prefixed `ml.`, a reference, a null hypothesis, an alternative, an inputs hash, and the cut points and named parameters in
`extras["thresholds"]` and in `policy_version`.

The lights come from `riskvalidation.core.policy.Policy` (`Threshold(kind, amber, red)` keyed by test id; `Threshold.classify` is
inclusive at its cut points). A `policy` argument of `None` is the packaged default policy completed, for the `ml.` keys it does not
hold, by the rows below (`ml_policy()`); a policy that lacks a key returns the light `not_evaluated`, never a colour. A rule stated
as "exceeds" (a tolerance) stores `math.nextafter(cut, inf)`, so a value equal to the tolerance is green and the light agrees with
the strict boolean of the pure function; the tolerance an argument names must be the one the policy's cut points state (the
function raises otherwise: build the policy with `ml_policy(perturbation_tolerance=...)` or put the row in the policy file). A test
that cannot be evaluated returns `not_evaluated` with its reason in `notes` and a null statistic, metric and p-value once
serialised, never a number.

| Test id | Metric read | Kind | Amber | Red | Boundary |
|---|---|---|---|---|---|
| `ml.robustness` | AUC degradation at the preset scale, $-\Delta_{\mathrm{AUC}}$ | upper_bad | 0.02 (`perturbation_tolerance`) | 0.04 (twice it) | strict |
| `ml.shift_resilience` | development-only Monte Carlo p-value of the AUC drop on the worst subsample (statistic: the drop) | p_value | 0.05 | 0.01 | inclusive |
| `ml.seed_stability` | standard deviation of the AUC across seeds | upper_bad | 0.01 | 0.02 | inclusive |
| `ml.reason_code_stability` | mean top-four reason overlap across refits | lower_bad | 0.80 | 0.60 | inclusive |
| `ml.effect_agreement` | Spearman rho of mean absolute SHAP against the ALE range | lower_bad | 0.70 | 0.40 | inclusive |
| `ml.additivity` | maximum Shapley additivity gap | upper_bad | 1e-9 | 1e-9 | strict |
| `ml.overfit` | train minus holdout AUC gap | upper_bad | 0.05 (`overfit_tolerance`) | 0.05 | strict |
| `ml.weak_spot` | error ratio of the worst slice (0 when none) | upper_bad | 2.0 (`weak_spot_multiple`) | 1e12 (sentinel: never red) | strict |
| `ml.conformal_coverage` | Beta-Binomial p-value of the coverage against $1-\alpha$ given `n_cal` | p_value | 0.05 | 0.01 | inclusive |
| `ml.leakage_screen.single_feature_auc` | largest single-feature $\max(\mathrm{AUC},1-\mathrm{AUC})$ | upper_bad | 0.95 | 0.95 | strict |
| `ml.leakage_screen.resemblance_auc` | cross-fitted development against out-of-time AUC | upper_bad | 0.70 | 0.70 | strict |
| `ml.leakage_screen.temporal_order` | feature values dated after the observation date | upper_bad | 1 | 1 | inclusive (a rule) |
| `ml.leakage_screen.entity_overlap` | indicator of any test entity present in training | upper_bad | 1 | 1 | inclusive (a rule) |
| `ml.data_quality` | largest share of failed checks | upper_bad | 0.01 | 0.05 | inclusive |
| `ml.cbpe_estimate` | no policy threshold: an estimate, not a test (always `not_evaluated`) | none | | | |
| `ml.challenger_brier` | no policy threshold: the 95 percent paired interval | none | | | |

The four named parameters are the approved values of decision V6 of unit U14; every other cut point is the contract's proposal with
no dossier value, a convention awaiting the main session's approval, never a regulatory threshold (dossier 06 I.6). The rows are in
`riskvalidation.ml.policy` because `policies/default.json` is the integrator's file; `threshold_rows()` gives them as plain mappings
(`kind`, `amber`, `red`, `source`; the strict rows as their `nextafter` floats) to write there as they stand.

**Conventions added by the review.** The strict rows are decided on 12-decimal values: `overfit_gap` rounds the gap, and
`robustness_finding` and `ml.robustness` round the AUC loss, to 12 decimals before comparing, so `0.80 - 0.75` (0.050000000000000044 in
floats) and `0.85 - 0.80` (0.04999999999999993) are both a gap of exactly 0.05, green at the tolerance 0.05. Two constants of the
resemblance fit are conventions of the engine, not regulatory values: the ridge strength `RIDGE = 1` and the number of cross-fitting
folds `FOLDS = 5` (`ml/_logistic.py`); they change the numbers `resemblance_auc` and `shift_score_rank` produce.

Light rules (each in the function's docstring):

- `robustness`: green when the AUC degradation at the preset scale is at most the tolerance, amber above it up to twice it, red
  above; statistic and metric are the degradation.
- `shift_resilience`: the p-value of `shift_development_reference` (page 01) against the p-value cut points; statistic and
  metric are the observed drop, `extras` holds `null_mean`, `null_p95` and `shift_permutations`; `not_evaluated` without development
  labels or with fewer than twice as many development rows as evaluation rows; `n` is the subsample rows.
- `seed_stability`, `reason_code_stability`, `effect_agreement`: the metric against its row; `n` is the seeds, the refits and the
  features.
- `additivity` (red when the gap exceeds the numeric tolerance 1e-9, never amber), `overfit` (red when the gap exceeds the
  tolerance, never amber), `weak_spot` (amber when the worst ratio exceeds $k$, never red).
- `conformal_coverage`: red below `alpha_red`, amber below `alpha_amber`, on the Beta-Binomial p-value; the metric is the coverage;
  `extras` holds `n_cal`, `k` and `p_value_binomial`; `not_evaluated` when `k` exceeds `n_cal`.
- `leakage_screen`: the worst of four screens read on their own keys; red when any trips; statistic is the number of screens that
  tripped, `extras["screens"]` holds each value and light.
- `data_quality`: the largest failed share over the counts; `n` and `n_events` are the cells or rows checked and failed, summed.
- `cbpe_estimate`: always `not_evaluated` (an estimate, not a test); statistic, metric and p-value null; the estimated AUC in
  `extras["estimate"]`, the four assumptions in the notes; with a label supplied the reason is that the labelled AUC is read.
- `challenger_brier`: amber when the paired interval lies above zero (the challenger's Brier is higher), green otherwise.

## Open points for the main session (decisions the contract leaves open)

1. **`cbpe_estimate` without a label.** The contract says "an estimate row, never green or red evidence" and defines a
   `not_evaluated` row as one with no number. The engine `Light` has no estimate value, so the estimate row carries the light
   `amber` (never green or red) with the number, the label `estimate` and the four assumptions, and the note "estimate, not
   evidence". Recommendation: add a `Light.ESTIMATE` to the engine core in a later unit and move this row to it.
2. **`leakage_screen` entity overlap.** The function takes the overlap as a share, the policy row reads "the number of test
   entities present in training" with cut points 1 and 1. The engine reads the share as the indicator of any overlap (1 when above
   0), which makes "any entity overlap" red; the count is not recoverable from a share. Recommendation: pass the count, or state the
   row as an indicator in the contract.
3. **`reason_code_overlap` sign.** Ranking by absolute attribution is an UNVERIFIED reading (page 02).
4. **Finding F1** (the 5th percentile is not bounded by the mean), **F2** and **F3** (conformal rank and MAPIE, page 05), **F4**
   (the false-alarm rate of the worst-subsample test, page 08).

## What these tests do not claim

- A light is a convention applied to a statistic: green means the statistic fell inside a convention, nothing more. No light is a
  finding of non-compliance with any rule, and none is a regulatory threshold.
- The size and power of each test are measured on planted defects (page 08), not on a bank's portfolio.
- The engine computes and makes no compliance claim.

## Sources

- Dossier 06 C.7 l.979-1058, I.1 l.1184-1200 and I.6 l.1261-1277; dossier 04 J.2 l.881-883; the contract `engine-ml-api` 1.0.0.
- NannyML CBPE, https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html; ING probatus,
  https://github.com/ing-bank/probatus; Angelopoulos and Bates, arXiv 2107.07511 (the references of the individual tests).

## Limits

The policy rows are proposals for approval. A policy file that holds an `ml.` key keeps its own row in `ml_policy()`.
