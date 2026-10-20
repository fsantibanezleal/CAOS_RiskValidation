# 08. The validation tests

Module `riskvalidation.validation.liquidity`. Sources: dossier 05 section E.1 to E.5 and its acceptance rows (lines 234 to 240, 263, 276 to
279, 480), dossier 08 (lines 566 to 568, 698, G.4 lines 739 to 742), dossier 06 lines 971 to 973. Each function returns one `TestResult`
(`riskvalidation.core.result`) that serialises into a row of `common/test-result`, with the ids `liq.<function name>`.

Acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds: the lights are conventions (decision Q14, the named
parameters of the case documents). Each function records its parameters in `policy_version` (`contraste-liquidity-battery-1.0.0; ...`) and
in `extras["thresholds"]`, so that a view can explain a light. A test that cannot be evaluated returns the light `not_evaluated` with its
reason in `notes` and a null statistic, metric and p-value once serialised, never a fabricated number. The tests are plain functions that do
not read the registry or the default policy. They make no compliance claim: a green light means a statistic fell inside a gate, nothing more.
The tests of `lcr_nsfr.py` are constructed tests and invariants (`tests/liquidity/test_lcr_nsfr.py`), not `TestResult` functions.

| Test | Statistic | Light rule (Contraste internal gate) | Golden |
|---|---|---|---|
| `list_compounding_golden` (`liq.list_compounding_golden`) | `abs(1 - (1 - q)^n - expected)` (metric: the computed outflow) | green at most `tol` (5e-7), red above | defaults: metric `0.1211484`, statistic `3.7e-7`, green; `expected = 0.12`: red |
| `survival_monotone_in_severity` (`liq.survival_monotone_in_severity`) | the largest increase of the survival period along the order, in days, at least 0 (`None` = survives the horizon, longer than any day) | green at most `tol_days` (0, never negative), red above; fewer than two: not_evaluated | `[None, 176, 122]` green, 0; `[100, 120]` red, 20 |
| `cbc_haircut_rules` (`liq.cbc_haircut_rules`) | the largest absolute haircut deviation over the nine cases of 05 line 238 (`n = 9`) | green when every haircut and label is as expected (`tol` 1e-12), red otherwise | green, 0; a table with 0.25 for `BBB+-BBB-`: red |
| `nlp_identity` (`liq.nlp_identity`) | the largest absolute residual of `nlp - (cbc - cumsum(outflows - inflows))` | green at most `tol` (1e-6), red above | the hand ladder: 0, green; `nlp[0] = 71`: 1, red |
| `runoff_adequacy_back_test` (`liq.runoff_adequacy_back_test`) | the ratio `modelled / realised` (metric: the shortfall) | green at least 0.8, amber at least 0.5, red below; not_evaluated for a realised share of 0 | `0.013723` against `0.85`: `0.016145`, shortfall `0.836277`, red; `(0.85, 0.85)` green; `(0.5, 0.85)`: `0.588235`, amber |
| `runoff_vs_generator_truth` (`liq.runoff_vs_generator_truth`) | the largest `abs(estimated - truth) / mc_se`, in standard errors | green at most 2, amber at most 3, red above; a non-positive `mc_se` raises | equal: 0, green; `2.5` amber; `3.5` red |
| `cliff_effect_diagnostic` (`liq.cliff_effect_diagnostic`) | the cliff effect `(NLP(far) - NLP(near)) / TA` (metric: `NLP(far) - NLP(near)`) | green at least `-0.01`, amber at least `-0.03`, red below; not_evaluated for a horizon shorter than `far` | `100` to `40` on 1000: `-0.06`, red; to `95`: `-0.005` green; to `80`: `-0.02` amber |
| `htm_loss_effect` (`liq.htm_loss_effect`) | the CBC removed `without - with` (metric: its share of `without`) | green when the loss does not raise the CBC and does not lengthen the survival period, red otherwise | `(72, 58.5, None, 48)`: 13.5, 0.1875, green; `(72, 80, 40, 60)`: red |
| `uninsured_share_sensitivity` (`liq.uninsured_share_sensitivity`) | the largest increase of the survival period along an increasing uninsured share, in days | green at most `tol_days` (0, never negative), red above | `[None, 67]` green, 0; `[40, 67]` red, 27 |
| `reverse_stress_round_trip` (`liq.reverse_stress_round_trip`) | `abs(survival_at_multiplier - target_days)` (metric: the multiplier); with the keyword `survival_just_below` the threshold property, the larger of `max(SP(m*) - target, 0)` and `max(target + 1 - SP(m* - 1e-6), 0)` | green at most `tol_days` (0, never negative), red above; not_evaluated for a refused solve; when the curve stepped the notes give the gap `target - SP(m*)` | `(2.085164, 5, 5)` green; `(2.0, 7, 5)` red, 2; `(None, None, 30)` not_evaluated; `(1.5, 1, 5, survival_just_below=inf)` green, gap 4 |
| `htm_revaluation_vs_reported` (`liq.htm_revaluation_vs_reported`) | `abs(model - reported) / abs(reported - amortised)` | green at most 0.05, amber at most 0.15, red above; not_evaluated for a missing model value or a zero reported loss | reported 80, amortised 100: model 81 `0.05` green, 82 `0.10` amber, 85 `0.25` red |

## Equations

$$\Delta_{LiST}=\Big|1-(1-q)^n-F_{printed}\Big|,\qquad
\rho=\frac{F_{model}}{F_{real}},\qquad
z=\max_k\frac{|\hat F_k-F_k|}{se_k},\qquad
\text{residual}=\max_t\Big|NLP_t-\big(CBC_t-\textstyle\sum_{s\le t}(O_s-I_s)\big)\Big| .$$

## Conventions

A survival period compared across scenarios counts `None` ("survives the horizon") as longer than any day: a finite period followed by
`None` is an unbounded lengthening (an infinite statistic, null once serialised) and red. The thresholds are the named parameters of the
case documents and are conventions; a caller's own values replace them and travel in `extras["thresholds"]`. The tolerances and the lights
are conventions (decisions Q13 and Q14), never regulatory thresholds.

## Limits

- `runoff_adequacy_back_test` is a descriptive score on one real event: one observation, no p-value. The 2023 rows of Table A.1 are
  scored against and never calibrated on when the variant is scored.
- `cliff_effect_diagnostic`: dossier 05 line 236 defines the diagnostic and no limit, so its lights are conventions; the day basis of
  `near` and `far` is UNVERIFIED (page 03).
- `htm_revaluation_vs_reported` compares a revaluation computed by the caller through `riskvalidation.irrbb` curves (decision Q11) with the
  reported fair value; the engine does not revalue inside this function. The held-to-maturity loss is a whole-book or pro rata assumption.
- `runoff_vs_generator_truth` is a test of an estimator against a generator whose mechanism is a convention (page 06).

## Sources

- Dossier 05 section E (ECB LiST 2019 results annex and methodological note; BCBS d555): https://www.bis.org/bcbs/publ/d555.htm
- ECB LiST 2019 methodological note and annex: https://www.bankingsupervision.europa.eu/
- Jiang, Matvos, Piskorski and Seru, NBER WP 31048, published as Journal of Financial Economics 159:103899 (2024), the mark-to-market
  fall in US bank assets (dossier 05 E.2): https://doi.org/10.1016/j.jfineco.2024.103899
