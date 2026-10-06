# 03. Scored samples and portfolios

`riskvalidation.generators.BinormalScores`, `LogisticPortfolio`, `NestedLda`

Discrimination tests need scores whose true AUC is known; obligor-level calibration tests need obligors whose true PD
is known. Three generators provide them.

## 1. Binormal scores

Defaulters score $N(\delta, 1)$ and survivors $N(0, 1)$, higher meaning riskier. The difference of a defaulter's and a
survivor's score is $N(\delta, 2)$, so

$$\mathrm{AUC} = P(X_D > X_N) = \Phi\!\left(\frac{\delta}{\sqrt 2}\right).$$

The empirical AUC is the Mann-Whitney statistic, and its variance on $n_1$ defaulters and $n_0$ survivors is exact
(Hanley and McNeil 1982, Radiology 143(1), 29 to 36,
[doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)):

$$\mathrm{Var}(\widehat{\mathrm{AUC}}) = \frac{A(1-A) + (n_1 - 1)(Q_1 - A^2) + (n_0 - 1)(Q_2 - A^2)}{n_1 n_0},$$

$Q_1$ the probability that two defaulters both outrank one survivor and $Q_2$ that one defaulter outranks two
survivors. With equal class variances $\sigma^2$, $X - Y$ and $X' - Y$ are normal with correlation 1/2 (they share
$Y$), so $Q_1 = Q_2 = \Phi_2(a, a; 1/2)$ with $a = \delta/(\sigma\sqrt 2)$ and $A = \Phi(a)$ (`binormal_auc_variance`).

Options, each a validator's question:

- **Discrimination decay**: `separation_development` above `separation` gives a current AUC below the development
  AUC (the alternative of `disc.auc_vs_initial`).
- **An estimated development AUC**: `development_size=m` draws a development sample $m$ times the current one and
  passes its empirical AUC as `auc_initial`; `None` passes the true development AUC, the known constant the ECB test
  assumes. Measured: with $m = 1$ the ECB test rejects 10.9% of the time at the 5% level when the AUC has not changed
  ([discrimination](../tests/02_discrimination.md)).
- **A challenger**: `challenger_separation` and `challenger_correlation` add a second score, correlated within each
  class, with AUC $\Phi(\delta_2/\sqrt 2)$; with equal separations it is a DeLong null of two non-nested models.
- **A leaked feature**: with `leak` $= \lambda$ the model scores the genuine score plus a feature that in development
  is $N(\lambda, 1)$ for defaulters and $N(0, 1)$ for survivors (it carries the outcome) and at decision time is
  $N(0, 1)$ for everyone. The model score's development AUC is $\Phi((\delta + \lambda)/2)$, its current AUC
  $\Phi(\delta/2)$, below the genuine score's $\Phi(\delta/\sqrt 2)$: the feature is now noise. The feature binned at
  the deciles of $N(0, 1)$ feeds `stability.csi`, with the exact bin probabilities of both laws in the truth; `leak=0`
  keeps the feature without the leak (the null of the leakage family).

`verify()` checks the empirical AUC's mean and variance against the exact ones (and the challenger's and the
development AUC) within four Monte Carlo standard errors.

**Estimator performance** (Morris et al. 2019, Table 6; 4,000 samples of AUC 0.80): the ECB standard error averages
the exact one (0.01793 against 0.01792 for 150 defaulters and 4,850 survivors); the estimator is unbiased; the normal
95% interval covers 94.5% (93.0% with 50 defaulters, 95.3% with 500).

## 2. Nested discriminants fitted in-sample (Demler et al. 2012)

Demler, Pencina and D'Agostino, *Misuse of DeLong test to compare AUCs for nested models*, Statistics in Medicine
31(23), 2577 to 2587 (2012), [doi:10.1002/sim.5328](https://doi.org/10.1002/sim.5328), section 2.3: multivariate
normal data with the means and correlations of a Framingham Heart Study sample, "A total of 8261 observations ...
cases (621 observations)"; effect sizes 0.65, -0.47, 0.48 and 0.62 for age, total cholesterol, HDL and systolic
pressure (as printed); one correlation matrix for both groups; a fifth predictor "simulated under the null ... by
making its means equal in the two subgroups and ... setting nondiagonal elements in the last column of the
correlation matrix to zero"; "Linear discriminant analysis was fitted to each simulated dataset, first for the reduced
model and then for the full model"; 1,000 repetitions.

`NestedLda` draws exactly that design and fits both discriminants on the drawn sample (pooled covariance), returning
their scores on the same sample. The population AUC of a discriminant is $\Phi(\sqrt{M^2/2})$, $M^2$ the squared
Mahalanobis distance (their equation 3), equal for the two models under the null (0.808). Their result: "the
significance level for the F-test is close to the nominal 0.05 level, whereas it is 0.001 for the DeLong test". The
harness: 1 rejection in 1,000 at 5% (`test_delong_nested_and_non_nested`). On two non-nested scores of equal AUC
scored on an independent sample, the same test holds its size (5.15% and 5.55% at 5%).

## 3. The logistic portfolio

$n$ obligors with standard normal covariates of equicorrelation $r$ (shifted by `shift` for a drifted population), a
true PD logistic in them, $\mathrm{logit}\,\pi = \beta_0 + \beta' x$, and defaults Bernoulli in the true PD (or in its
one-factor conditional version with correlation `rho`, one systematic factor per sample). The model under test gives

$$\mathrm{logit}\,\widehat{\mathrm{PD}} = a + b\,(\beta_0 + \beta' x),$$

so $a = 0, b = 1$ is a right model, $a = -\ln k$ divides every odds by $k$ (a PD underestimated about $k$ times when
PDs are small), and $b \ne 1$ is a slope miscalibration. `quadratic` adds $q(x_1^2 - 1)$ to the true logit, a term the
model omits: the misspecification Hosmer, Hosmer, le Cessie and Lemeshow (1997) used to compare goodness-of-fit tests.

**Truths by quadrature.** The linear predictor is normal, $\eta \sim N(\beta_0 + \beta'\mu,\; \beta'\Sigma\beta)$, so the
mean true PD $E[\mathrm{expit}(\eta)]$, the mean model PD, the expected Brier score of the true PDs $E[\pi(1-\pi)]$ and of
the model ($E[\widehat{PD}^2 - 2\,\widehat{PD}\,\pi + \pi]$) are one-dimensional Gauss-Hermite integrals (two-dimensional,
over $x_1$ and the rest of $\eta$, with a quadratic term). The AUC of $\eta$ is

$$\mathrm{AUC} = \frac{1}{\bar\pi(1-\bar\pi)}\int \mathrm{expit}(u)\,\varphi_\eta(u)\int_{-\infty}^{u}\big(1 - \mathrm{expit}(v)\big)\,\varphi_\eta(v)\,dv\,du,$$

by Simpson's rule on a grid of 40,001 points; a sample of 200,000 obligors reproduces it within four of its standard
errors (tested). Measured on 5,000 obligors with a mean PD of 5%, the Hosmer-Lemeshow test with $G$ degrees of freedom
and the Spiegelhalter test hold their size, and $G - 2$ does not ([PD calibration](../tests/01_pd-calibration.md)).

## 4. Caveats

- Binormal scores have equal class variances; real score distributions are skewed and their ROC curves asymmetric.
- In the logistic portfolio the true model is in the model's family unless a quadratic term is planted; a validator's
  model is usually misspecified in ways no generator enumerates.
- An in-sample AUC is biased upwards by the fit; `NestedLda.verify` exposes the bias with enough draws.
