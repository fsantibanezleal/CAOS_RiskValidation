# 05. The validation tests

Module `riskvalidation.validation.oprisk`. Source: dossier 05, "Implications for Contraste": the operational risk ladder, its
acceptance tests and case O1 (fit on simulated years 1 to K, evaluate the capital stability on independent replications, never
re-use the replications used for tuning).

Each function returns one `TestResult`. These tests are **descriptive**: `p_value`, `alpha_amber` and `alpha_red` are null, and the
lights compare a metric against tolerances the caller supplies, which are Contraste internal gates, not regulatory thresholds. The
tolerances travel in `extras["thresholds"]` and in `policy_version`, so that a view can explain a light. A test that cannot be
evaluated returns `not_evaluated` with its reason in `notes` and a null statistic, metric and p-value, never a placeholder number.
The tests state no compliance claim. Symbols: $\ell$ the light, $m$ the metric, $\mathrm{tol}$ a tolerance.

| Test | Metric | Light (Contraste internal gate) |
|---|---|---|
| `oprisk.benchmark_reproduction` | the largest deviation over its tolerance | green $m \le 1$, amber $m \le 2$, red above |
| `oprisk.threshold_stability` | the range of the finite fitted tail indices | green $m \le$ `tol_green` (0.10), amber $\le$ `tol_amber` (0.25) |
| `oprisk.capital_stability` | the dispersion $(p_{75} - p_{25})/\text{truth}$ and the bias | the worse of the two |
| `oprisk.sma_ilm_response` | the ILM jump after one added loss | green $\le$ 0.05, amber $\le$ 0.15 |
| `oprisk.scenario_influence` | the largest absolute scenario influence | green $\le$ 0.10, amber $\le$ 0.25 |

## `benchmark_reproduction`

The engine reproduces Shevchenko's benchmark (page 02): $E[Z] = 738.9056$, $\mathrm{Var}[Z] = 298095.7987$, skewness 40.3428, and a
99.9 percent quantile of 5849 to 5850 (the step-1 discretised quantile) by Panjer and FFT and 5853.1 (the continuous quantile,
Shevchenko Table 3 at step 0.0625 and Table 7) by Monte Carlo. A moment's deviation is $|v - v_{ref}|/v_{ref}$; a Panjer or FFT
quantile's deviation is 0 inside $[5849, 5850]$ and otherwise the distance to the nearest end over 5849.5; a Monte Carlo quantile's
deviation is $|q - 5853.1|/5853.1$.

$$m = \max\Big(\frac{\max\text{ moment deviations}}{\text{moment\_rel\_tol}},\ \frac{\max\text{ quantile deviations}}{\text{quantile\_rel\_tol}}\Big)$$

`statistic` equals `metric`; `extras["deviations"]` lists every deviation by name and `extras["max_quantile_rel_dev"]` the largest
quantile deviation. A missing method gives `not_evaluated`. Golden: the moments above, quantiles Panjer 5849, FFT 5850, Monte Carlo 5900,
tolerances $10^{-6}$ and 0.01 give a Monte Carlo deviation $46.9/5853.1 = 0.008013$, $m = 0.801285$, green.

## `threshold_stability`

The fitted GPD tail index is stable across thresholds (the threshold-stability plot of the ladder). $m = \max - \min$ of the finite
fitted tail indices, `statistic` equals `metric`, `n` the number of finite values; fewer than 2 gives `not_evaluated`;
`extras["n_infinite_mean"]` counts the values at or above 1 (an infinite-mean fit, the ladder's failure mode). Golden:
$\hat\xi = [0.90, 0.95, 1.00, 0.97]$ with tolerances 0.05 and 0.15 gives $m = 0.10$, amber, one infinite-mean fit.

## `capital_stability`

The 99.9 percent capital is stable across independent histories and unbiased against the truth known by construction. With
$r = \text{capitals}/\text{truth}$ and percentiles by linear interpolation:

$$m = p_{75}(r) - p_{25}(r),\qquad \text{statistic} = \operatorname{median}(r) - 1 .$$

`extras` carries `ratio_p05`, `ratio_p25`, `ratio_median`, `ratio_p75`, `ratio_p95` and `share_underestimating` (the share of capitals
strictly below the truth); fewer than 4 capitals give `not_evaluated`. The light is the worse of dispersion (green $\le$ 0.25, amber
$\le$ 0.50) and bias ($|\text{statistic}|$: green $\le$ 0.10, amber $\le$ 0.25). Golden: capitals $[90, 100, 110, 120]$ and truth 100 give
$p_{25} = 97.5$, median 105, $p_{75} = 112.5$, $m = 0.15$, statistic 0.05, green.

## `sma_ilm_response`

The ILM does not jump when one large loss is added to the history (the SMA's response to a single large loss). `added_loss` is added
to the latest year; $m = \mathrm{ILM}_{after} - \mathrm{ILM}_{before}$, `statistic` $= \mathrm{ILM}_{after}$; fewer than 5 years give
`not_evaluated` (OPE25.15: no ILM on under 5 years). `extras` carries `ilm_before`, `ilm_after`, `lc_before`, `lc_after`, `orc_before`,
`orc_after`. Golden: BIC 150, ten annual losses of 10 and an added loss of 90 give $LC$ from 150 to 285, ILM from 1.0 to 1.220648,
$m = 0.220648$, red.

## `scenario_influence`

No single scenario point moves the capital by more than the tolerance (dossier 05 F.5). $m = \max|\text{influences}|$ over the output of
`scenarios.scenario_influence`; `extras` carries the influences and the index `argmax`; no influence gives `not_evaluated`. Golden:
influences $[0.05, 0.0, 0.2]$ give $m = 0.2$, amber.

## Limits

- The lights are gates on descriptive metrics; they are not p-values and not regulatory verdicts. A green light means a metric fell
  inside a tolerance the caller chose.
- Capital stability is a statement about independent replications of a generator with a known truth; it does not transfer to a bank
  whose true distribution is unknown.
- The benchmark test checks the engine against a published example, not a model; the Monte Carlo reference (5853.1) is the continuous
  quantile of that example.

Sources: dossier 05 (ladder and O1); Shevchenko (2010), arXiv 1008.1108, https://arxiv.org/abs/1008.1108; Basel Framework OPE25,
https://www.bis.org/committees/bcbs/basel-framework/standard/ope/25/inforce/2023-01-01/published/2024-07-05.
