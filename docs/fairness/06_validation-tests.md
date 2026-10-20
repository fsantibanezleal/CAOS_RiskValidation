# 06. The validation tests

Module `riskvalidation.validation.fairness`. Sources: the contract `engine-fairness-api` 1.0.0 (Contraste
`docs/contracts/engine/fairness-api.md`, sections `validation.fairness` and Policy thresholds); dossier 06 section C.7 l.1029-1040 and
I.6 l.1268-1270; dossier 04 J.3 l.897-905; the SDD l.335 (size and power on a planted defect); `riskvalidation.core.policy`. Primary
texts: 29 CFR 1607.4(D), https://www.law.cornell.edu/cfr/text/29/1607.4; Hardt, Price and Srebro (2016), arXiv:1610.02413; Morris, White
and Crowther (2019), doi:10.1002/sim.8086 (the simulation design and the Monte Carlo error of a rate).

Each function returns one `TestResult` (`TestResult.to_dict()` is a row of `common/test-result`) with a `fair.` test id, a `reference`,
a null hypothesis, an alternative, a policy version and the tolerances in `extras["thresholds"]`. A test that cannot be evaluated
(one group, a group without both classes, a group of no rows, no outcome, a policy without its row) returns the light
`not_evaluated` with its reason in `notes` and **no number** (statistic NaN, null once serialised; metric and p-value None). A
malformed input raises `ValueError` naming the field. The engine states **no legal claim** and never red-lights an indicator that
the contract labels one.

## The twelve tests

Symbols: $m$ the metric of the result; "inclusive" means the light is amber at $m \ge$ the amber cut and red at $m \ge$ the red cut;
"strict" means a value equal to the tolerance is green (the cut is stored one float above, `math.nextafter`). A red cut of 2.0 is a
sentinel above the range of a metric that is at most 1: that light is never red.

$$\text{light}=\begin{cases}\text{red}& p<\alpha_{red}\\ \text{amber}& p<\alpha_{amber}\\ \text{green}&\text{otherwise}\end{cases}\ (\text{p-value rows}),\qquad
\text{light}=\begin{cases}\text{red}& m\ge m_{red}\\ \text{amber}& m\ge m_{amber}\\ \text{green}&\text{otherwise}\end{cases}\ (\text{upper-bad rows})$$

| Test id | Function | Metric | Light and policy row | Label |
|---|---|---|---|---|
| `fair.demographic_parity` | `demographic_parity(model_id, rates, reference, policy)` | largest absolute rate gap to the reference | `upper_bad`, amber 0.05 strict, red 2.0: green up to the tolerance, amber above, never red | convention |
| `fair.adverse_impact` | `adverse_impact(model_id, rates, reference)` | lowest ratio to the highest-rate group | amber when `four_fifths_flag` flags any group on the exact counts, green otherwise; no policy row | rule, an employment benchmark |
| `fair.two_proportion_z` | `two_proportion_z(model_id, x1, n1, x2, n2, policy)` | rate difference (statistic: z, p-value) | `p_value`, amber 0.05, red 0.01, inclusive | convention |
| `fair.fisher_exact` | `fisher_exact(model_id, x1, n1, x2, n2, policy)` | rate difference (statistic: odds ratio, p-value) | `p_value`, amber 0.05, red 0.01, inclusive | convention |
| `fair.equalized_odds` | `equalized_odds(model_id, y, decision, group, policy)` | largest TPR or FPR gap | `upper_bad`, amber 0.05, red 0.10, inclusive; needs an outcome | convention |
| `fair.equal_opportunity` | `equal_opportunity(model_id, y, decision, group, policy)` | largest TPR gap | `upper_bad`, amber 0.05, red 0.10, inclusive; needs an outcome | convention |
| `fair.calibration_within_groups` | `calibration_within_groups(model_id, p, y, group, policy)` | largest ECE over the groups | `upper_bad`, amber 0.05, red 0.10, inclusive; needs an outcome | convention |
| `fair.auc_by_group` | `auc_by_group(model_id, score, y, group, policy)` | largest AUC gap between two groups | `upper_bad`, amber 0.05, red 0.10, inclusive; needs an outcome | convention |
| `fair.conditional_disparity` | `conditional_disparity(model_id, effect, controls, missing_controls)` | odds ratio (statistic: Wald z, p-value) | amber when the 95 percent interval excludes 0, never red; no policy row | convention |
| `fair.rate_spread_disparity` | `rate_spread_disparity(model_id, effect, controls, missing_controls)` | coefficient in percentage points (statistic: Wald z, p-value) | same rule; no policy row | convention |
| `fair.replication` | `replication(model_id, replication)` | size ratio, confirm over fit | amber when the sign differs between year t and t+1; no policy row | rule: sign and size |
| `fair.situation_testing` | `situation_testing(model_id, result, policy)` | significant share of the complainants | `upper_bad`, amber 0.20, red 2.0, inclusive: amber at the cut, never red | convention |

The rows are `FAIR_ROWS` of `riskvalidation.fairness.policy`; `fairness_default_policy()` is the packaged default policy with them added
where it holds none, and a `policy` argument of None means it. A policy a caller supplies is used as given: a policy that lacks the key
returns `not_evaluated`, never a colour. The four outcome-conditioned tests return `not_evaluated` with the reason when `y` is None (a
dataset that declares no outcome, HMDA); with an outcome they report the omitted groups counted in `extras`. The two interval tests
recompute the Wald statistic from the interval of the effect and name the controls and the missing controls in `notes` and `extras`:
they are an approximation of comparative evidence and never a finding of a violation. The four-fifths line is labelled an
**employment benchmark** in `reference` and `notes`, with `is_credit_law_threshold` False in `extras`, and is compared with the
highest-rate group; the parity ratios to the chosen reference group are in `extras["parity_ratio_to_reference"]`.

Golden values: the four-fifths pair 60 of 100 against 75 of 100 is green and 5999 of 10000 is amber; 55 of 100 against 50 of 100 is a
demographic-parity gap of exactly 0.05 and is green; a gap of 0.10 is amber and a gap of 1.0 is still amber; a share of 19 of 100
significant complainants is green, 20 of 100 amber and 100 of 100 amber (the amber cut is 0.20). A 'never red' light
(`fair.demographic_parity`, `fair.situation_testing`) is capped at amber whatever the policy: a caller policy whose red cut is 0.10
gives amber, and the notes say "never red by contract; the policy's red cut is not applied". The gate is `tests/fairness/test_fairness_validation.py` and
`tests/fairness/test_fairness_policy.py`.

## Size and power

Measured with `harness.run.simulate` at 4,000 repetitions on the planted bias of [the planted-bias page](05_planted-bias.md), one
seed per study and an independent stream per repetition (Morris, White and Crowther 2019); each rate is a Monte Carlo estimate with
standard error $\sqrt{\hat r(1-\hat r)/4000}$ (at most 0.0079). The tests with a p-value are read at $p<0.05$; the others by their own
rule, a light that is not green (amber or red). The gate is `tests/fairness/test_fairness_size_power.py`; the measured figures
are the ones the test reproduces (it asserts them with margins, and `FAIRNESS_SIZE_POWER_OUT` writes them as JSON).

| Test | Null (size) | Planted bias | Power |
|---|---|---|---|
| `fair.two_proportion_z` | 0.054 (se 0.004), threshold plant gap 0, n 2,000 | threshold gap 0.3 (true rate gap 0.091) | 0.990 |
| `fair.fisher_exact` | 0.046 (se 0.003), same | same | 0.992 |
| `fair.demographic_parity` | 0.015 (se 0.002), same | threshold gap 0.5 (true rate gap 0.156) | 1.000 |
| `fair.adverse_impact` | 0.000, same | threshold gap 0.8 (true rate gap 0.255, ratio 0.64) | 1.000 |
| `fair.equalized_odds` | 0.013 (se 0.002), label-noise plant noise 0, n 8,000 | label noise 0.25; threshold gap 0.5 | 1.000; 1.000 |
| `fair.equal_opportunity` | 0.000, same; 0.000 on the threshold plant at gap 0 | threshold gap 0.5; label noise 0.25 | 1.000; 0.516 |
| `fair.calibration_within_groups` | 0.000, same | label noise 0.25 | 1.000 |
| `fair.auc_by_group` | 0.000, same | label noise 0.25 | 1.000 |
| `fair.conditional_disparity` | 0.048 (se 0.003), proxy plant strength 0, n 3,000; 0.052 with the proxy controlled at strength 0.7 | proxy strength 0.7, proxy not controlled | 1.000 |
| `fair.rate_spread_disparity` | 0.052 (se 0.004), proxy plant strength 0 | proxy strength 0.7 | 1.000 |
| `fair.replication` | false alarm 0.000 on a stable effect (strength 0.7) | the second year recoded in reverse | 1.000 |
| `fair.situation_testing` | 0.0395 (se 0.0031) at the amber cut 0.20 (4,000 repetitions, n 400, k 30, no bias); at the former cut 0.10 0.194 (U14-E2), 0.191 for a share strictly above 0.10 and 0.088 above 0.15 (this round) | negative-decision bias 0.4 | 1.000 at every cut |

What the figures say. The z-test holds its 5 percent level within Monte Carlo error and Fisher's exact test is conservative, as an
exact test is. The two interval tests hold the 5 percent level of a 95 percent interval; the conditional disparity falls to that level
once the proxy is a control. Equal opportunity reads only the true positive rate, which label noise of 0.25 moves by about 0.06 (the
flipped defaults enter the positive class), so its power against label noise is weak (0.52) while equalized odds, which also reads the
false positive rate, detects it every time. The replication rule has no size to speak of when there is no true effect: the sign is a
coin flip (amber rate 0.495 at strength 0); it is a stability rule for an effect that exists. **The situation-testing amber cut is 0.20 (the main
session's decision, 2026-10-19; it was 0.10)**: at 0.10 the rule raised a false alarm on about one null draw in five at n 400 and
k 30, because the complainants share neighbours and the significant share is more variable than a binomial count; at 0.20 the
false-alarm rate is 0.0395 (se 0.0031), inside the default size bound 0.05 plus two standard errors, at power 1.000. The cut is a
convention, and this measured size is the evidence for it. The size and power module carries the pytest marker `sizepower`
(about 208 s per Python version): the engine CI deselects it with `-m "not sizepower"` and the local release gate runs it
(`python -m pytest -m sizepower`).

Limits: every figure is for its planted bias at its sample size; a different sample, base rate or correlation structure gives different
rates, and the plants are idealisations. The cut points are conventions, never regulatory thresholds. A green light is not a finding of
fairness and an amber one is not a finding of discrimination: the engine states no legal claim. The outcome-conditioned tests never run
on HMDA. UNVERIFIED: no dossier prints a value for any of these cut points (the Taiwan battery rows, the situation-testing share) or a
golden value for these tests; they are proposals, and the four-fifths line is an employment benchmark, not a credit-law threshold.
