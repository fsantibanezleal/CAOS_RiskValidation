# Design: U3, known-truth generators and the size and power harness

## Why

A validation test is evidence only as far as its size and power are known. With real outcomes the truth is unknown,
so neither can be measured; with a generator whose parameters are known, both can (dossier 08, G.1). The unit gives
the package what a validator needs to judge its own instruments: data-generating mechanisms with exact truths, a
harness that measures rejection rates with their Monte Carlo error, exact rejection probabilities where a formula
exists, and the published simulation studies reproduced as the harness's oracle.

The study design follows Morris, White and Crowther (2019): aims, data-generating mechanisms, estimands, methods and
performance measures (ADEMP), one seed per study, one stream per repetition, a Monte Carlo standard error on every
estimated performance measure.

## Modules

| Module | Contents |
|---|---|
| `generators/_base.py` | `Generator` protocol (`name`, `config()`, `draw(rng)`, `truth()`), `Check` and `verify()` (a generator reproduces its truth on its own stream) |
| `generators/counts.py` | `DefaultCounts`: years by grades, WP14's one-factor model with factor correlation across years; `default_correlation(pd, rho)` |
| `generators/scores.py` | `BinormalScores` (one or two scores, a development AUC known or estimated, a leaked feature), `LogisticPortfolio` (covariates, true logistic PDs, a planted calibration intercept and slope, an omitted quadratic term, covariate shift, one-factor dependence) |
| `generators/drift.py` | `TwoSample`: Yurdakul and Naranjo's design, population or sample binning |
| `generators/ratings.py` | `Migrations`, `decaying_matrix`, `break_monotonicity`; `GradeFrequencies`, `concentrate` |
| `generators/framingham.py` | `NestedLda`: Demler et al.'s Framingham-like design, LDA scores of the reduced and the full model fitted on the same sample |
| `harness/rates.py` | `Rate` (count, rate, Morris SE, Wilson interval), `wilson_interval`, `size_bound`, `within_size`, `agrees` |
| `harness/run.py` | `simulate(test, generator, feed, n_rep, seed, ...)` returning a `Simulation`; decision rules `p_below`, `metric_above` |
| `harness/exact.py` | `count_pmf`, `critical_count`, `rejection_probability` for the count tests |
| `harness/performance.py` | `estimator_performance` (bias, empirical SE, model SE, coverage, with Monte Carlo SEs) |

## Contracts

- **A generator** is a frozen dataclass. `config()` returns its parameters with the generator's name and the package
  version (the configuration written next to the data, dossier 07); `draw(rng)` returns a dict of NumPy arrays and
  scalars named after the arguments of the tests it feeds (`n`, `d`, `pd`, `defaults`, `risk`, `risk_1`, `risk_2`,
  `expected_counts`, `actual_counts`, `development`, `current`, `counts`, `freq_current`, `freq_initial`,
  `default_rates`, `auc_initial`); `truth()` returns exact quantities, JSON-safe.
- **A feed** maps a draw to a test's keyword arguments. The default feed passes the draw's keys that the test accepts;
  a scenario that needs a view (one year, one grade, the portfolio) passes its own function.
- **A simulation** records the test id, the generator configuration and truth, `n_rep`, the seed, the levels, the rate
  per level and rule, the undefined count, and the p-values and metrics of every repetition (kept in memory; a product
  serialises a histogram).

## Streams and seeds

`SeedSequence(seed).spawn(n_rep)`: repetition $i$ uses `default_rng(children[i])` for its draw. A test with a `seed`
argument (the Monte Carlo p-value of `pd.default_profile`) receives `int(rng.integers(2**63 - 1))` drawn from the
repetition's generator after the dataset, unless the caller fixes it. Results of deterministic tests are cached by
their inputs when every input is a scalar or a short array (at most 256 values): the same inputs give the same result,
so the cache changes speed, never numbers.

## Decision rules

Rejection at level $\alpha$ is $p < \alpha$, the engine's light rule (amber below 5%, red below 1%); Morris et al. write
$p \le \alpha$, identical for continuous p-values and noted for the discrete ones. A rule is a function of the
`TestResult`; `metric_above(0.10)` gives the PSI rule of thumb.

## Exact rejection probabilities

A count test's p-value decreases in the count, so the test rejects exactly when $D \ge k^*$; `critical_count` finds
$k^*$ by bisection on the registered function itself, and `rejection_probability` returns
$P_{\pi,\rho}(D \ge k^*)$ from `default_count_sf` (binomial, or the one-factor mixture by 256-node Gauss-Hermite). The
harness's simulated rate must agree with it (RV-307): the exact value checks the generator, the harness and the test
together.

## Truths

| Generator | Exact truth |
|---|---|
| `DefaultCounts` | PDs, asset and factor correlations, expected counts, the variance of each count $n\pi(1-\pi) + n(n-1)(\pi_2 - \pi^2)$ with $\pi_2 = \Phi_2(\Phi^{-1}(\pi), \Phi^{-1}(\pi); \rho)$, the default correlation $(\pi_2 - \pi^2)/(\pi(1-\pi))$ |
| `BinormalScores` | AUC $\Phi(\delta/\sqrt2)$; the variance of the Mann-Whitney AUC $[A(1-A) + (n_1-1)(Q_1 - A^2) + (n_0-1)(Q_2-A^2)]/(n_1 n_0)$ with $Q_1 = Q_2 = \Phi_2(\delta/\sqrt2, \delta/\sqrt2; 1/2)$ for unit variances; with a leak $\lambda$, the development AUC of the model score $\Phi((\delta+\lambda)/2)$ and the current $\Phi(\delta/2)$ |
| `LogisticPortfolio` | the linear predictor is normal, $\eta \sim N(\beta_0 + \beta'\mu, \beta'\Sigma\beta)$, so the mean PD, the mean of $\pi(1-\pi)$ (the expected Brier score of the true PDs) and the true AUC of $\eta$ are one-dimensional integrals, by 256-node Gauss-Hermite |
| `TwoSample` | bin probabilities of base and target under the population edges; population PSI |
| `Migrations`, `GradeFrequencies` | expected counts; population CV and Herfindahl index |
| `NestedLda` | the AUCs of the reduced and full population discriminants $\Phi(\sqrt{M^2/2})$, equal under the null |

## The Hosmer-Lemeshow default

`pd.hosmer_lemeshow` gains `fitted: bool = False`. The degrees of freedom are the number of groups unless `fitted` is
true ($G-2$, the in-sample statistic) or `dof` is given. The change is decided by measurement (RV-312) and recorded in
the changelog; the result's `extras` carry `dof` and `fitted`.

## The traffic-lights table

The colour profiles of $T$ years, ordered lexicographically by $(A_g, A_y, A_o, A_r)$, and their cumulative multinomial
probabilities are computed once per $(T, q)$ and cached; the p-value is a lookup. Identical to the enumeration for
$T \le 9$ (RV-313), and fast enough for 25,000-run reproductions.

## Docs

`docs/generators.md` (index) and `docs/generators/01_size-and-power.md` (the ADEMP design, the Monte Carlo SE, the
Wilson interval, exact rejection probabilities, the size bound), `02_default-counts.md` (WP14's model and the
binomial test under correlation), `03_scores-and-portfolios.md` (binormal and logistic generators, AUC truths, leakage,
DeLong nested), `04_drift-and-ratings.md` (the two-sample design, migrations, concentration), each with its figures
drawn by the package (`scripts/make_doc_figures.py`) and the published reproductions with their tolerances.
