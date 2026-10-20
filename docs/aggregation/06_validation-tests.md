# 06. The validation tests

Module `riskvalidation.validation.aggregation`. Sources: dossier 05 section G.5 (validation tests for an economic capital model:
benchmark the aggregation against the simple sum and the variance-covariance formula, the copula-choice sensitivity, worst-case
bounds, the stress of the inter-risk correlation, the sign of every diversification benefit, one horizon and confidence level),
G.2 (Rosenberg and Schuermann 2006, doi:10.1016/j.jfineco.2005.03.001), G.3 (Embrechts, Puccetti and Ruschendorf 2013,
doi:10.1016/j.jbankfin.2013.03.014) and dossier 01 l.837-839. Each function returns one `TestResult` (`riskvalidation.core.result`) that
serialises into a row of `common/test-result`. The Euler additivity test of the band is the reused
`validation.portfolio.euler_additivity` (`test_id = "portfolio.euler_additivity"`).

Acceptance gates are **Contraste internal gates**, stated as such, not regulatory thresholds. Each function records its parameters in
`policy_version` and in `extras["thresholds"]`, so that a view can explain a light; `alpha_amber` and `alpha_red` are null (these tests
are not hypothesis tests) and `p_value` is null. A test that cannot be evaluated returns the light `not_evaluated` with its reason in
`notes` and a null statistic, metric and p-value once serialised, never a fabricated number; a non-finite input raises `ValueError`
naming the field. The tests are plain functions that do not read the registry or the default policy. They make no compliance claim:
a green light means a statistic fell inside a gate, nothing more.

| Test | Statistic (metric) | Light rule (Contraste internal gate) | Golden |
|---|---|---|---|
| `simple_sum_and_variance_covariance` (`aggregation.simple_sum_and_variance_covariance`) | aggregate / variance-covariance (aggregate / simple sum) | red when aggregate > simple sum $(1+10^{-9})$; amber when $\lvert$statistic $-1\rvert>0.10$; green otherwise | EC (3, 4), $R=I$: 5.0 green (metric 0.7143), 8.0 red, 5.6 amber (1.12) |
| `copula_choice_sensitivity` (`aggregation.copula_choice_sensitivity`) | $\max_\nu\lvert t/g-1\rvert$ (signed value at the smallest $\nu$) | green at most 0.05, amber at most 0.15, red above | Gaussian 100, t capitals (4, 107), (8, 103.5), (30, 101): 0.07 amber |
| `worst_case_bracket` (`aggregation.worst_case_bracket`) | worst $-$ lower (that over the largest row) | red when a row is outside [lower $-$ tol, worst $+$ tol] or lower > worst; green otherwise (no amber) | lower 4.0, rows (5.0, 6.0828, 7.0), worst 8.0: 4.0 green; worst 6.5 red |
| `correlation_stress` (`aggregation.correlation_stress`) | stressed / base $-1$ (the same) | red below $-10^{-9}$; amber above 0.10; green otherwise | 100 to 105: 0.05 green; 112 amber; 95 red |
| `diversification_sign` (`aggregation.diversification_sign`) | max of the factors (1 $-$ the smallest) | red when a factor exceeds $1+10^{-9}$; green otherwise | (0.87, 0.9, 0.93) green; (0.87, 1.02) red |
| `horizon_confidence_consistency` (`aggregation.horizon_confidence_consistency`) | the count of marginals off the expected horizon (1 year) or level (0.999) | red when above 0, green otherwise | four consistent marginals: 0 green; one level 0.99: 1 red |
| `elliptical_mc_agreement` (`aggregation.elliptical_mc_agreement`) | (MC $-$ VC) / standard error (MC / VC $-1$) | green when $\lvert$statistic$\rvert\le3$, red above; not_evaluated for a nonpositive standard error | MC 100.4, se 0.2, VC 100: 2.0 green; MC 101.0: 5.0 red |
| `rosenberg_schuermann_reference` (`aggregation.rosenberg_schuermann_reference`) | simple sum / Gaussian $-1$ (the same) | always green: a reference, never a gate; the notes say inside or outside [0.30, 0.40] | the benchmark run below: 0.2403, outside |

## Equations

$$\text{statistic}_1=\frac{\text{aggregate}}{\sqrt{EC^\top R\,EC}},\qquad
\text{statistic}_2=\max_\nu\Big|\frac{EC_{t,\nu}}{EC_{Ga}}-1\Big|,\qquad
\text{statistic}_7=\frac{EC_{MC}-EC_{VC}}{\mathrm{se}_{MC}}$$

Symbols: $EC_{Ga}$ and $EC_{t,\nu}$ the capital of the Gaussian and t copula rows, $EC_{MC}$ and $EC_{VC}$ the Monte Carlo and
variance-covariance capitals, $\mathrm{se}_{MC}$ the Monte Carlo standard error, $R$ the correlation matrix.

## The reference run

Rosenberg and Schuermann's benchmark (dossier 05 G.2, staff report 185, Table 3 note): risk weights market 4.1, credit 26.4,
operational 69.5 percent, normal copula with market-credit correlation 0.5 and market and credit with operational 0.2. Then
$\sqrt{w^\top R\,w}=0.80624$ and the additive overestimate is $1/0.80624-1=0.2403$ (computed): outside the published 30 to 40 percent,
which is the point of the reference. The benchmark weights are the weights of simulated return components, not capital. The published
range itself and the Kuritzkes, Schuermann and Weiner inter-risk correlation ranges are UNVERIFIED in their originals (dossier 05 G.2),
and the Joint Forum and Embrechts, McNeil and Straumann wordings are not used as values. The test `rosenberg_schuermann_reference`
carries `extras["role"] = "reference-not-a-gate"`.

## What these tests do not claim

- A test comparing an aggregate with the simple sum and the variance-covariance formula shows agreement or disagreement with two
  benchmarks, not that the aggregate is right: the variance-covariance formula is exact only for elliptical losses.
- The correlation stress compares two capitals the caller computed; it does not choose the stressed correlation (0.8 is the dossier's
  example, an upper end of published ranges, UNVERIFIED).
- A diversified row above the simple sum is a true result (VaR is not subadditive): the test reports it, red, and nothing repairs it.
- The Rosenberg and Schuermann row is never a gate. The thresholds named here are policy parameters proposed by the contract; none is
  a regulatory value.
