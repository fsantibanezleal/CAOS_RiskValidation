# 02. Discrimination

Discrimination asks whether the model ranks defaulters as riskier than non-defaulters. It says nothing about
calibration: multiplying every PD by two leaves every ranking statistic unchanged.

![ROC and CAP curves and the identity AR = 2 AUC - 1](../assets/roc-cap.svg)

Inputs are risk-oriented: a higher value means a riskier obligor (a PD, a risk score, or a grade number where
higher is worse). With $A$ the defaulters and $B$ the non-defaulters,

$$u_{a,b} = \begin{cases}1 & \text{risk}(a) > \text{risk}(b)\\ \tfrac12 & \text{risk}(a) = \text{risk}(b)\\ 0 & \text{risk}(a) < \text{risk}(b)\end{cases},\qquad \mathrm{AUC} = \frac{1}{|A|\,|B|}\sum_{a\in A}\sum_{b \in B} u_{a,b}.$$

This is the Mann-Whitney statistic scaled to [0, 1]: the probability that a randomly chosen defaulter is ranked
riskier than a randomly chosen non-defaulter (Hanley and McNeil 1982, Radiology 143(1), 29 to 36,
[doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)). The ECB states the same
statistic with ratings where higher means better quality; the orientation is reversed, the value is identical.

## AUC with the ECB standard error (`disc.auc`, descriptive)

The structural components (DeLong placements) are

$$V_{10}(a) = \frac{1}{|B|}\sum_{b} u_{a,b},\qquad V_{01}(b) = \frac{1}{|A|}\sum_{a} u_{a,b},$$

both averaging to the AUC, and the variance used by the ECB (Annex 3.1 of the 2019 instructions) is

$$s^2 = \frac{\widehat{\mathrm{var}}(V_{10})}{|A|} + \frac{\widehat{\mathrm{var}}(V_{01})}{|B|}$$

with unbiased sample variances. The package computes the placements in $O(n \log n)$ by sorting. The result also
reports the Gini coefficient (the accuracy ratio) $2\,\mathrm{AUC} - 1$ and a normal 95% interval.

**The standard error against the exact one.** For binormal scores (defaulters $N(\delta, 1)$, survivors $N(0, 1)$) the
AUC is $\Phi(\delta/\sqrt 2)$ and the variance of the Mann-Whitney AUC is exact (Hanley and McNeil 1982): $[A(1-A) +
(n_1 - 1)(Q_1 - A^2) + (n_0 - 1)(Q_2 - A^2)]/(n_1 n_0)$ with $Q_1 = Q_2 = \Phi_2(a, a; 1/2)$, $A = \Phi(a)$ ([scored
samples](../generators/03_scores-and-portfolios.md)). Measured with the performance measures of Morris, White and
Crowther (2019) over 4,000 samples of AUC 0.80: the ECB standard error averages the exact one (0.01793 against 0.01792
for 150 defaulters and 4,850 survivors), the estimator is unbiased, and the normal 95% interval covers 94.5% of the time
(93.0% with 50 defaulters, 95.3% with 500): a little short when defaulters are few, where the AUC's sampling
distribution is skewed to the left.

**AUC and the accuracy ratio.** BCBS WP14: "the area A under the ROC curve and the CAP accuracy ratio AR are
connected by means of the linear transformation AR = 2 A - 1. As a consequence, neither A nor AR depend on the
proportion of defaulters in the sample." The identity holds when both are computed on the same ranking with ties
treated consistently: half credit in the AUC and linear interpolation inside tied groups in the CAP. The package's
`cap_accuracy_ratio` computes AR from the CAP directly, and the test `test_auc_matches_pairwise_and_cap` checks
the identity to $10^{-12}$ on data with ties. Somers' D of the score with respect to the default indicator is the
same number (WP14; Somers 1962, [doi:10.2307/2090408](https://doi.org/10.2307/2090408)).

## AUC against the development AUC (`disc.auc_vs_initial`)

ECB 2019, section 2.5.4.1: with the AUC at initial validation treated as fixed,

$$S = \frac{\mathrm{AUC}_{init} - \mathrm{AUC}_{curr}}{s},\qquad p = 1 - \Phi(S).$$

- $H_0$: the AUC at development is smaller than the AUC in the observation period.
- Reject (discrimination deteriorated) when $p < \alpha$.
- For low-default portfolios the ECB allows pooling the current and two preceding periods, replacing each
  period's rating by its within-period cumulative rank fraction; callers pool before calling the test.

**The development AUC is an estimate too (measured in 0.03.000).** The statistic's variance counts the current sample
only. When the development AUC is the true one the test holds its size, a little conservatively: 4.12% at the 5% level
(150 defaulters and 4,850 survivors, AUC 0.80, 4,000 repetitions; the variance estimate rises when the estimate falls,
which shrinks the lower tail). When the development AUC was estimated on a sample $m$ times the current one, the
difference has about $1 + 1/m$ times the variance the test assumes, and under an unchanged AUC:

| Development sample | Size at 5% | Size at 1% | $P(Z > 1.645/\sqrt{1 + 1/m})$ |
|---|---|---|---|
| the true AUC (a known constant) | 4.12% | 0.35% | 5.0% |
| 16 times the current | 4.40% | 0.73% | 5.5% |
| 4 times the current | 6.75% | 1.43% | 7.1% |
| as large as the current | 10.88% | 3.62% | 12.2% |

A validator applying the test to a development AUC measured on a sample of similar size should widen $s$ by the
development sample's own variance, or read the p-value as an early warning.

## DeLong test (`disc.delong`)

For a champion and a challenger scored on the same obligors (DeLong, DeLong and Clarke-Pearson 1988, Biometrics
44(3), 837 to 845, [doi:10.2307/2531595](https://doi.org/10.2307/2531595)): with $S_{10}$ and $S_{01}$ the sample
covariance matrices of the two models' placements across defaulters and non-defaulters,

$$\mathbf{S} = \frac{S_{10}}{|A|} + \frac{S_{01}}{|B|},\qquad z = \frac{\widehat{\mathrm{AUC}}_1 - \widehat{\mathrm{AUC}}_2}{\sqrt{\mathbf{S}_{11} + \mathbf{S}_{22} - 2\mathbf{S}_{12}}},$$

asymptotically standard normal under $H_0$: equal AUCs; two-sided. When the two rankings are identical the
variance is zero and the result is "not evaluated". The ECB test above compares against a fixed number; DeLong
compares two models on one sample, which is the challenger question.

**Size, and the nested misuse.** On two different scores of equal AUC (0.80, correlation 0.5) scored on an independent
sample, the test holds its size: 5.15% at the 5% level for 150 defaulters and 4,850 survivors, 5.55% for 621 and 7,640.
Comparing nested models fitted on the sample they are compared on is different. Demler, Pencina and D'Agostino (2012,
Statistics in Medicine 31(23), 2577 to 2587, [doi:10.1002/sim.5328](https://doi.org/10.1002/sim.5328)) show that "if the
added predictor is not statistically significantly associated with the outcome, the null distribution is non-normal,
contrary to the assumption of DeLong test", and in their Framingham-like simulation the DeLong rejection rate at 5% is
0.001. The harness reproduces it on their design (8,261 observations, 621 cases, the fifth predictor under the null,
linear discriminants fitted on each sample, 1,000 repetitions): 1 rejection in 1,000. "This problem does not exist in
cases of predictors that are associated with the outcome and for non-nested models"; a champion and a challenger of
different families, validated on a holdout, are the safe case.

## Kolmogorov-Smirnov separation (`disc.ks`, descriptive)

$\mathrm{KS} = \sup_s |F_D(s) - F_{ND}(s)|$ between the risk distributions of defaulters and non-defaulters. Its
p-value tests identical distributions, that is no discrimination at all, which is rarely the question in
validation; the measure is reported as a separation statistic.

## Intervals

`bootstrap_ci` resamples defaulters and non-defaulters separately (stratified), so every replicate keeps the
observed default count, and returns a percentile interval for any statistic. No regulatory source prescribes the
number of replicates; the default is 2,000 and the seed makes the interval reproducible.
