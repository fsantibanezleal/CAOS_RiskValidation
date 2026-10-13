# 06. The LGD and EAD validation tests

Module `riskvalidation.validation.lgd_ead`. Source: dossier 04 F.3 (Takeaway) and G.3 (validation specifics). Each test returns
one `TestResult` that serialises into a row of `common/test-result`. The acceptance gates are **Contraste internal gates**,
not regulatory thresholds: each function states its light rule, and the parameters travel in `policy_version`, `alpha_amber`
and `alpha_red`; the parameters of a light rule that are not alphas (`rho_green`, `rho_amber`, `gap_amber`, `gap_red`,
`tolerance`) travel in `extras["thresholds"]`. A test with no data returns `not_evaluated` with its reason in `notes` and a
null statistic, metric and p-value in the serialised row, never a number. LGD calibration is judged by the mean realised against the mean predicted LGD
by segment and over time, not by $R^2$ alone (Loterman et al. 2012, doi:10.1016/j.ijforecast.2011.01.006; Qi and Zhao 2011,
doi:10.1016/j.jbankfin.2011.03.011).

| Test | Statistic | Light (Contraste internal gate) |
|---|---|---|
| `lgd_ead.lgd_calibration_by_segment` | paired $t=\bar d/(s_d/\sqrt N)$ on $d=\text{realised}-\text{predicted}$, $N-1$ degrees of freedom, two-sided p | green when $p\ge\alpha_{amber}$ (0.05), amber when $\alpha_{red}\le p<\alpha_{amber}$ (0.01), red below |
| `lgd_ead.lgd_calibration_over_time` | the largest $\lvert t\rvert$ over the periods tested; $p=\min(1,K\,p_{min})$ (Bonferroni) | as above, on the adjusted p |
| `lgd_ead.lgd_discrimination` | Spearman $\rho$ (ties by average rank), two-sided p | green when $\rho\ge0.30$, amber when $0.10\le\rho<0.30$, red below |
| `lgd_ead.downturn_coverage` | one-sided $t=(\bar x-\hat{\mathrm{LGD}})/(s/\sqrt N)$, upper p, metric the shortfall | as the calibration test, on the one-sided p |
| `lgd_ead.ead_backtest_by_band` | relative gap of the predicted against the realised EAD per utilisation band; metric the largest $\lvert\cdot\rvert$ | green when the metric $\le0.10$, amber when $\le0.25$, red above |
| `lgd_ead.ccf_estimator_bias` | bias $\bar e$ against the known CCF, $t=\bar e/(s_e/\sqrt R)$, two-sided p | green when $\lvert\bar e\rvert\le$ tolerance, amber when $\le2\times$ tolerance, red above |

The statistic common to the calibration, coverage and bias tests is the Student $t$ of a sample of gaps $g_1, \dots, g_n$:

$$t=\frac{\bar g}{s_g/\sqrt{n}},\qquad p_{\text{two-sided}}=2\,\Pr(T_{n-1}>\lvert t\rvert),\qquad
p_{\text{Bonferroni}}=\min(1,\,K\,p_{min}).$$

Symbols: $\bar d$ the mean gap, $s_d$ its standard deviation with ddof 1, $N$ the observations, $K$ the periods tested,
$R$ the replications of the generator, $e=\text{estimate}-\text{truth}$. The paired $t$ with two degrees of freedom has the
closed form $p=1-t/\sqrt{t^2+2}$ (two-sided); the calibration golden is $t=3.464102$, $p=0.074180$.

## Golden values

| Test | Inputs | Result |
|---|---|---|
| by segment | realised `[0.4, 0.5, 0.6]`, predicted 0.3 | $t=3.464102$, $p=0.074180$, metric 0.2, green |
| over time | period A gaps `[0.1, 0.2, 0.3]`, period B `[-0.1, 0, 0.1]` | statistic 3.464102, $p=0.148360$, metric 0.2, green |
| discrimination | realised `[0.1, 0.4, 0.6, 0.9]`, predicted `[0.2, 0.3, 0.7, 0.8]`; reversed | $\rho=1.0$ green; $\rho=-1.0$ red |
| downturn coverage | estimate 0.55, realised `[0.5, 0.6, 0.7]` | $t=0.866025$, $p=0.238884$, metric 0.05, green |
| EAD back-test | predicted `[900, 500]`, realised `[800, 600]`, utilisation `[0.2, 0.8]`, edges `[0, 0.5, 1]` | gaps 0.125 and -0.166667, statistic 0, metric 0.166667, amber |
| CCF bias | estimates `[0.61, 0.63, 0.65]`, truth 0.60, tolerance 0.05 | metric 0.03, $t=2.598076$, $p=0.121690$, green |

## What each test refuses

A non-finite input raises. A zero spread is a spread below $10^{-12}$ times the larger of 1 and the largest absolute gap, so
that rounding noise in identical gaps is not read as a spread; a zero mean is judged on the same scale.

- `not_evaluated`: fewer than 2 observations (3 for the rank correlation); a constant input to the rank correlation; no period
  with at least 2 observations and a spread and no period with a certain gap; no observation, or no realised EAD, in the
  back-test; no replication.
- No spread in the gaps of the calibration test: **red** with a null statistic and p-value and the gap in the metric when the
  mean gap is not zero (a certain gap), **green** when it is zero. The same holds for the downturn coverage (a shortfall is red,
  none is green) and, per period, for the calibration over time (a period with no spread and a nonzero gap makes the light red;
  the periods not tested are listed in `extras["excluded"]` with their reason, and their gaps enter the metric).
- The CCF bias test keeps its light for one replication or no spread (it depends on the bias against the tolerance only); its
  statistic and p-value are then null.
- The back-test accepts a negative predicted EAD (a utilisation-change prediction is not clipped) and refuses a negative
  realised EAD.
- A band of the back-test without observation has null values; a band whose realised EAD is 0 lists its gap and has a null
  relative gap; neither enters the metric.

Realised LGD values outside [0, 1] are kept and counted (`n_below_zero`, `n_above_one`).

## Limits

The lights are Contraste internal gates; they are not regulatory thresholds and the engine makes no compliance claim. LGD rank
power is modest in the literature (average $R^2$ between 4% and 43% in Loterman et al. 2012), which is why the discrimination
gate is low. The downturn coverage test judges a supplied estimate against a supplied downturn period; it claims no EBA approach
(EBA/GL/2019/03 is a cross-reference only, UNVERIFIED in dossier 04). A statistically significant CCF bias inside the tolerance
stays green; the view shows both the bias and the p-value.
