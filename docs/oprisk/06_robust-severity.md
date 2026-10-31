# 06. Robust severity estimators and the mean-adjusted SLA

Module `riskvalidation.oprisk.robust` (added in 1.1.0). Source: Opdyke and Cavallo (2012, *Estimating
operational risk capital: the challenges of truncation, the hazards of maximum likelihood estimation, and the promise of robust
statistics*, Journal of Operational Risk 7(3)), who recommend the two estimators below with the mean-adjusted single loss
approximation for the truncated severity of an operational risk loss distribution. Their primary text was not read (it is behind a
paywall); the estimators are implemented from their textbook definitions and the SLA form from a secondary source, and the result
record says so (`status = "SECONDARY_SOURCE"`).

## What the module adds

| Name | What it is |
|---|---|
| `cramer_von_mises_distance` | $W^2$ of the losses against a severity law truncated at the collection threshold |
| `fit_severity_cvm` | the minimum $W^2$ fit (lognormal, Weibull, GPD) |
| `fit_severity_obre` | Hampel's optimally bias-robust fit at a requested asymptotic efficiency |
| `RobustSeverityFit` | the fit record: parameters, $W^2$, bound $c$, achieved efficiency, the final weights |
| `sla_var_mean_adjusted` | $F^{-1}(1-(1-\alpha)/E[N]) + (E[N]-1)\,E[X]$ |
| `MeanAdjustedSla` | its result record |

## Cramer-von Mises

With the left-truncated cdf $u = (F(x) - F(H))/(1 - F(H))$ and the sorted $u_{(1)} \le \dots \le u_{(n)}$,

$$W^2 = \frac{1}{12 n} + \sum_{i=1}^{n}\Big(u_{(i)} - \frac{2i-1}{2n}\Big)^2 ,$$

and `fit_severity_cvm` minimises it by Nelder-Mead from the truncated maximum likelihood fit. Golden values in the tests: one loss at
the median gives $1/12$; two losses at the quartiles give $1/24$. The statistic is bounded in each observation, so a gross loss
moves it by at most a constant; this is why the estimator resists contamination.

## Optimally bias-robust estimator

With $s(x;\theta)$ the score of the truncated density, the OBRE solves $\sum_i (s_i - a)\,W_i = 0$ with
$W_i = \min\big(1,\ c / \lVert A (s_i - a)\rVert\big)$, where the centre $a$ and the symmetric matrix $A$ satisfy, under the model,
$E[(s-a)W] = 0$ and $E[A (s-a) W s^\top] = I$ (Hampel, Ronchetti, Rousseeuw and Stahel 1986, section 4.2). The expectations are
taken at 2,000 quantile midpoints of the truncated model and the conditions solved by Levenberg-Marquardt on scores whitened by
the Fisher information (the plain fixed point on $A$ diverges). The bound $c$ is calibrated, by bisection, so that the asymptotic
efficiency $\mathrm{tr}(I^{-1})/\mathrm{tr}(E[\psi\psi^\top])$ at the Cramer-von Mises estimate equals the requested value (0.95 by
default); the record states the efficiency achieved at the fit (the tests check 0.80 and 0.95 within 0.01). An efficiency near 1
approaches the maximum likelihood fit (tested), a lower one discounts more. The final weights $W_i$ mark the discounted losses: in
the test with 4 % of the losses multiplied by 400, at least 30 of the 40 smallest weights are gross losses.

## The mean-adjusted SLA

$\mathrm{VaR}_\alpha \approx F^{-1}\big(1 - (1-\alpha)/E[N]\big) + (E[N]-1)\,E[X]$. It is one mean smaller than the UNVERIFIED
`sla.sla_var_mean_corrected` ($+\,E[N]E[X]$); the test pins the difference at exactly $E[X]$. A severity without a mean (tail index
1 or more) and $E[N] < 1$ raise. The form is the one stated in arXiv 1607.02319 (the single loss approximation of the Basel
Committee's standardised approach analysis, attributing the second-order term to Degen 2010); `status` is `SECONDARY_SOURCE` until
the primary text is checked.

## The study

`scripts/oprisk_robust_study.py` draws histories of 250 losses above EUR 20,000 from generators of known truth, fits five
estimators and divides each capital (mean-adjusted SLA at 0.999, observed rate 40 per year, frequency rescaled by the fitted
law) by the same functional of the true law, so the ratio isolates estimation. Generators: `lognormal` (10, 1.6), the family the
lognormal estimators assume; `splice` (lognormal(10, 1.2) body, GPD tail with $\xi = 0.6$, $\beta = 60{,}000$ above EUR 100,000), which
they misspecify; and each with 3 % of the losses multiplied by 50. The splice maximum likelihood fit places its threshold at the
90th percentile of the history. Seed 20261028; history $r$ uses seed $+\,r$. `obre-80` (80 % efficiency) was added to the run before
it was made, beside the usual 95 %. A ratio of 1 is unbiased.

**Size actually run.** A first run of 500 histories per scenario did not finish in 52 minutes on a loaded machine and was stopped
with no result. The reported run is the second and last one, capped at 900 seconds per scenario with 200 histories requested:
**165 (lognormal), 160 (lognormal-contaminated), 170 (splice) and 148 (splice-contaminated) histories**. The intervals are
therefore wide; each cell has an uncertainty of order a tenth in the median ratio. `n_defined` is the number of histories in which
the method returned a finite capital (a fit that fails or a fitted tail index of 1 or more has none).

Median ratio of capital to truth (interquartile range in brackets; share of histories below half of the truth):

| Scenario (histories) | MLE lognormal | Cramer-von Mises | OBRE 95 % | OBRE 80 % | MLE splice |
|---|---|---|---|---|---|
| lognormal (165) | 0.94 (0.54; 1 %) | 1.01 (0.69; 7 %) | 0.97 (0.59; 2 %) | 0.99 (0.69; 6 %) | 0.74 (1.64; 30 %), 143 defined |
| lognormal, 3 % gross (160) | 7.70 (7.45; 0 %) | 1.94 (1.98; 1 %) | 3.90 (4.70; 0 %) | 2.24 (2.73; 1 %) | 17.07 (35.7; 0 %), 74 defined |
| splice (170) | 0.30 (0.13; 92 %) | 0.26 (0.10; 94 %) | 0.27 (0.10; 98 %) | 0.26 (0.10; 97 %) | 0.46 (1.01; 53 %), 136 defined |
| splice, 3 % gross (148) | 4.52 (6.10; 0 %) | 0.38 (0.21; 73 %) | 0.71 (0.73; 30 %) | 0.42 (0.30; 66 %) | 9.09 (10.2; 0 %), 19 defined |

(In the last row the four lognormal-based methods have a finite capital in 143 of the 148 histories; the other five have none.)

Median absolute log ratio, the typical multiplicative error (0 is exact; 0.69 is a factor of 2):

| Scenario | MLE lognormal | Cramer-von Mises | OBRE 95 % | OBRE 80 % | MLE splice |
|---|---|---|---|---|---|
| lognormal | 0.29 | 0.32 | 0.30 | 0.31 | 0.76 |
| lognormal, 3 % gross | 2.04 | 0.67 | 1.36 | 0.81 | 2.83 |
| splice | 1.21 | 1.34 | 1.31 | 1.36 | 0.99 |
| splice, 3 % gross | 1.51 | 0.98 | 0.62 | 0.88 | 2.21 |

## Verdict on the hypotheses

The study asks for the capital's bias and spread of the robust estimators reported beside the MLE splice, on
generators of known truth; it states no numeric threshold. The three statements below were written into the study before the
final run and are operationalised here; each is reported whether or not it holds.

1. **Under 3 % gross losses the robust fits have a smaller typical error than the lognormal MLE.** Holds in both contaminated
   scenarios: Cramer-von Mises 0.67 against 2.04 and OBRE 95 % 1.36 against 2.04 (lognormal truth); 0.98 and 0.62 against 1.51
   (splice truth). The effect is large for Cramer-von Mises and for OBRE at 80 %; OBRE at the usual 95 % efficiency still
   overstates capital by a factor of 3.9 (median) on the lognormal truth: a 95 % efficient estimator discounts these gross losses
   only partly.
2. **On clean data from the assumed family the MLE has the smallest spread.** Holds: interquartile range 0.54 against 0.59 (OBRE
   95 %), 0.69 (OBRE 80 % and Cramer-von Mises). The price of robustness there is small, 9 to 28 % more spread.
3. **The robust lognormal fits beat the MLE splice in typical error in every scenario.** Does **not** hold. They do in three
   (0.29 to 0.32 against 0.76 on clean lognormal data; 0.67 to 1.36 against 2.83 and 0.62 to 0.98 against 2.21 under contamination),
   but on the clean splice truth the lognormal-based fits, robust or not, understate capital by about 70 % (median ratio 0.26 to
   0.30, 92 to 98 % of histories below half), and the splice MLE has the smaller typical error (0.99 against 1.21 to 1.36), with a
   much wider spread (interquartile range 1.01 against about 0.1). Robustness protects against gross losses; it does not repair a
   misspecified family.

What is not claimed: the generators draw from the lognormal family or from the splice the fits include, which flatters the
fits that match (the caveat of the C17 study); 3 % of losses multiplied by 50 is one contamination design; the runs are 148 to
170 histories, not the 500 requested; the SLA is an approximation, so the truth is the SLA functional of the true law, not the
exact aggregate quantile. The splice MLE's `n_defined` shows its other weakness: a fitted tail index of 1 or more leaves no
capital in 13 to 87 % of histories.

## Reproducing

    python scripts/oprisk_robust_study.py --scenario lognormal-contaminated --reps 200 --max-seconds 900 --out result.json

`tests/oprisk/test_robust.py` runs a 12-history version of the `lognormal` scenario and the closed-form, truth-recovery,
contamination and efficiency tests.
