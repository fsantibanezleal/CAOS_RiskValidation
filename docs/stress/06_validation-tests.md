# 06. The validation tests

`riskvalidation.validation.stress` holds the six tests of dossier 05 H.7 (l.422, what validators test in a stress model). Each returns one
`TestResult` that serialises (`TestResult.to_dict()`) into a row validating against `common/test-result`; the parameters of a light rule
that are not alphas travel in `extras["thresholds"]` beside `alpha_amber` and `alpha_red` (null here). The thresholds are policy
parameters proposed by the contract of the case, Contraste internal gates and none regulatory. All take `model_id` and `segment` last. A
test that cannot be evaluated returns the light `not_evaluated` with its reason in `notes` and a null statistic, metric and p-value, never a
number. The supervisory replication test of the unit sizing is void with decision E1 (C19 stays planned).

| Test id | Statistic | Metric | Light |
|---|---|---|---|
| `stress.satellite_backcast` | RMSE of predicted against realised | peak timing error in quarters | green: size deviation within `peak_amber` (0.35) and timing within `timing_quarters` (2); amber: size deviation within `peak_red` (0.60); red otherwise |
| `stress.sensitivity_monotonicity` | number of coefficients with the wrong sign | null | red when any, green otherwise |
| `stress.static_balance_sheet_rules` | number of violated rules among three | null | red when any, green otherwise |
| `stress.bottom_up_vs_top_down` | $\text{bottom-up} / \text{top-down} - 1$ | equal to the statistic | green at or below 0.25, amber at or below 0.50, red above |
| `stress.scenario_severity` | $\text{path peak} / \text{history peak}$ | null | green at or above 0.75, amber below (no red) |
| `stress.reverse_solution_check` | $\lvert L(x^*) - \ell\rvert / \ell$ | $\min(\text{starts}) / d^* - 1$ | red above the tolerance (1e-6) or when a start sits closer, green otherwise |

## Back-cast

Satellites estimated before an episode must back-cast it: dossier 05 H.7 asks to back-cast the satellites through 2008 to 2009 and 2020
with realised macro paths, out of sample (estimated on data ending before the episode), and to compare the predicted and realised
loss-rate peaks and their timing. `satellite_backcast` takes the episode's predicted and realised rates, their quarters and `fit_end`;

$$
\mathrm{RMSE} = \sqrt{\tfrac1n \sum_t (\hat y_t - y_t)^2},\qquad \rho = \frac{\max_t \hat y_t}{\max_t y_t},\qquad \Delta = \operatorname{argmax}\hat y - \operatorname{argmax} y.
$$

It returns the light `error` with its reason when `fit_end` is not before the first quarter of the episode (the fit window overlaps the
episode it predicts: leakage) and `not_evaluated` when the realised peak is not positive. Golden values: predicted peak $1.0$ against realised
$1.2$ (ratio $0.8333$, deviation $0.1667$) one quarter apart is green; $1.0$ and $1.2$ three quarters apart is amber. The contract's golden "predicted
peak $0.5$ against $1.2$: red" does not follow from its own rule at the default gates: the deviation is $0.5833$, within `peak_red = 0.60`, so the
light is amber; red needs a deviation above 0.60 (for example $0.4$ against $1.2$, deviation $0.6667$) or `peak_red` of 0.55 or less. The engine
implements the rule as written, and the discrepancy is reported to the integrator.

## Sensitivity monotonicity

Every macro sensitivity has its expected sign (`positive`, `negative` or `none`), so the projected loss rate is monotone in the severity of the
path. An estimate of zero has no sign and counts as wrong for `positive` and `negative`; `none` is never wrong. Golden values:
`((0.82, -0.045, 0.06), ("positive", "negative", "positive"))` gives statistic 0 and green; flipping the second estimate gives statistic 1 and red.

## Static balance sheet rules

The EBA paragraphs 33 and 90 rules: the total is constant (the largest and smallest period totals differ by at most `tolerance`, absolute), stage 3
never falls (no cure) and no balance is negative. Golden values: the output of `capital.eba_static_balance_sheet((80, 15, 5), (2, 3))` is green;
the same rows with stage 3 of $(7, 6)$ are red (two violated rules, the total and the cure).

## Bottom-up against top-down

The stressed loss of the bottom-up result is compared with that of a top-down challenger (dossier 05 H.7: for example a panel quantile
model). Golden values: bottom-up 120 against top-down 100 gives statistic $0.2$ and green. The contract's golden "160: amber" does not follow from
its own rule: $160/100 - 1 = 0.6$ is above the red gate $0.50$ and is red (so is 200); 140 gives $0.4$, amber. A top-down loss that is not positive gives
`not_evaluated`.

## Scenario severity

A stress path is at least as severe as history (the EBA documents severity against the GFC: dossier 05 l.422). The inputs are the peak deviation
from the jump-off of a scenario variable (unemployment in percentage points, GDP in percentage points) and the same deviation in the global
financial crisis; the result maps onto the document block `outputs.severity`. A baseline path (`is_stress = False`) gives `not_evaluated` with the
reason "not a stress path"; a history peak that is not positive gives `not_evaluated`. Golden values: unemployment peak $+5.5$ against a GFC peak
$+5.5$ gives $1.0$ and green; $+3.0$ against $+5.5$ gives $0.5455$ and amber. There is no red: a mild stress is a finding about the scenario, not a failure.

## Reverse solution check

The loss at the reverse stress solution equals the threshold and no start finds a closer point (dossier 05 H.7, the reverse stress check):

$$
\text{statistic} = \frac{\lvert L(x^*) - \ell\rvert}{\ell},\qquad \text{metric} = \frac{\min_i d_i}{d^*} - 1,\qquad \text{red if } \min_i d_i < d^*(1 - 10^{-6}).
$$

Fewer than three starts gives `not_evaluated`. Golden values: loss $5.0$, threshold $5.0$, distance $2.2361$ and starts $(2.2361, 2.2361, 2.2362)$ are green; a start
at $1.5$ is red.

## Limits

- The lights are Contraste internal gates stated in each docstring and recorded in `extras["thresholds"]`; none is a regulatory threshold,
  and a green light means a statistic fell inside a gate, nothing more. The engine states no compliance claim.
- The back-cast compares peaks and their timing, not the shape of the whole episode; a single peak can hide an early or late miss, which the
  RMSE only partly shows. It needs the episode's realised macro paths, which the caller supplies.
- Two goldens of the contract (the back-cast 0.5 against 1.2 and the bottom-up 160 against 100) contradict the contract's own light rules at
  the default gates; the engine follows the rules and the tests state the pairs that agree with them. The outcome of that amendment is for the
  contract owner (UNVERIFIED which of rule or golden was intended).
- The severity gate compares one peak per variable against a reference episode the caller supplies; the history peaks of the case fixtures
  are fixture values.

## Sources

- Dossier 05 H.7 and H.4 (EBA 2025 EU-wide stress test), dossier 01 J.4 (reverse stress) of the Contraste initiative research.
- EBA, 2025 EU-wide stress test, Methodological Note of 2025-01-20, paragraphs 33, 34 and 90,
  https://www.eba.europa.eu/sites/default/files/2025-01/0246e2f3-fa57-47d1-99f2-7a5b80cae509/2025%20EU-wide%20stress%20test%20-%20Methodological%20Note.pdf
- EBA, 2025 EU-wide stress test results (2025-08-01),
  https://www.eba.europa.eu/sites/default/files/2025-08/0178b9c5-2f0d-42ee-8226-6fa0a87c0d6c/2025%20EU-wide%20stress%20test%20-%20Results.pdf
