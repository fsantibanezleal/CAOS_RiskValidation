# 08. Planted defects, size and power

Modules `riskvalidation.ml.plants` and `tests/ml/ml_sims.py`; run by `tests/ml/test_ml_size_power.py` with `harness.run.simulate`.
Sources: dossier 07 D pattern 6, l.611-616 (synthetic data with a known ground truth, a verifier and a difficulty knob; gen-fraud-graph
`verify.py` and `--hardness`); the product SDD Method ladder (the size and power of each test on its planted defect, l.335);
Morris, White and Crowther (2019), Statistics in Medicine 38(11), doi:10.1002/sim.8086, Table 6 (the Monte Carlo error of a
rejection rate); the contract `engine-ml-api` 1.0.0 (module `plants.py` and the Check line of every validation function).

## The generators

Each plant is a frozen `GeneratorBase` configuration with `config()`, a seeded `draw(rng)`, an exact `truth()` and a `verify()` whose
large draw reproduces the truth within Monte Carlo error (4 standard errors). The plant seed is the default seed of `sample()` and,
for `SeedUnstablePlant`, part of every fit random stream. The design constants below are the plants' own and not regulatory values.

| Plant | Construction | Truth |
|---|---|---|
| `LeakPlant(n, leak_strength, seed)` | `LogisticPortfolio` (3 features, about 20 percent defaults) plus a fourth column $y\,s+N(0,1)$ | the leak column, and the single-feature AUC of the leak exactly $\Phi(s/\sqrt2)$ (0.5 at strength 0, the null) |
| `CovariateShiftPlant(n, shift, features, seed)` | a development and an evaluation sample from a stronger portfolio (about 50 percent defaults, AUC about 0.85); a quarter of the evaluation rows are shifted by `shift` sd on `features`; the true logit loses 2 per unit of $x_0$ beyond 1.5 (slope 1.6 reverses to -0.4), the linear model `predict` does not | the shifted features, the share 0.25, the evaluation means; shift 0 is the null |
| `NoiseSensitivityPlant(n, sensitivity, seed)` | prevalence 0.3; signal column $\delta y+N(0,1)$ with $\delta=$ `sensitivity`, a pure-noise column; the model is monotone in the signal | the AUC at scale $\epsilon$ is exactly $\Phi\big(\delta/\sqrt{2(1+\epsilon^2(1+\delta^2\pi(1-\pi)))}\big)$ |
| `SeedUnstablePlant(n, instability, seed)` | prevalence 0.3, separation 1.5; the model of seed $s$ is $\mathrm{expit}(x_0+\tau\xi_s x_1)$ with $\xi_s\sim N(0,1)$ from the stream (plant seed, $s$) | the AUC of seed $s$ is exactly $\Phi(1.5/\sqrt{2(1+\tau^2\xi_s^2)})$; the mean and the standard deviation across seeds by Gauss-Hermite quadrature |
| `ExchangeabilityBreakPlant(n, drift, seed)` | PDs Beta(2, 5); a calibration window of $10n$ and a test window of $n$; the test labels have log-odds shifted by `drift` | at $\alpha=0.10$ the population quantile, the coverage in each window and the coverage loss, by exact quadrature (drift 0.5: loss 0.0439; drift 1: 0.0990) |

`sensitivity` of `NoiseSensitivityPlant` is the standardised separation of the fragile column, not "the AUC loss per unit eps" of the
contract: under noise proportional to the column spread no model keeps its AUC, and the loss per unit eps is an output of
`truth()`. UNVERIFIED reading of the contract wording; the amendment, if wanted, goes through the main session.

## Method

Each test is fed, through the engine pipeline it belongs to, by a generator with a null and a planted defect; `simulate` draws
4,000 seeded repetitions per condition (one `SeedSequence` stream per repetition, Morris, White and Crowther 2019) and counts how
often the light is not green (an alarm). The rate is the **size** under the null and the **power** under the defect, with its Monte
Carlo standard error $\sqrt{r(1-r)/n}$ and the Wilson interval. The five planted-defect generators serve the tests they name
(`robustness`, `shift_resilience`, `seed_stability`, `conformal_coverage`, `leakage_screen`); the other tests are fed by small array
generators of `tests/ml/ml_sims.py` (an attribution table with refit noise, an importance vector with a related or independent effect
range, the exact Shapley values of a linear model with a tampered column, binormal train and test scores, Brier terms of calibrated
slices with one slice that predicts $1-p$, a NaN rate, two forecasts with extra noise). The design bounds of the study, stated before
the run, are a size of at most 0.05 and a power of at least 0.90, with two stated exceptions: the conformal coverage test (size at
most 0.10) and the worst-subsample test (size at most 0.15, finding F4).

## Measured size and power (4,000 repetitions per condition, seed 20261019 for the null and 20261020 for the defect)

| Test | Null | Size | se | Wilson 95% | Defect | Power | se | Wilson 95% |
|---|---|---|---|---|---|---|---|---|
| `ml.robustness` | sensitivity 0.25 | 0.0278 | 0.0026 | [0.0231, 0.0333] | sensitivity 2.5 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.shift_resilience` | shift 0 | 0.1055 | 0.0049 | [0.0964, 0.1154] | a quarter of the rows shifted 3 sd on feature 0 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.seed_stability` | instability 0.2 | 0.0000 | 0.0000 | [0.0000, 0.0010] | instability 1.0 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.reason_code_stability` | refit noise 0.1 | 0.0000 | 0.0000 | [0.0000, 0.0010] | refit noise 1.5 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.effect_agreement` | effect range follows importance | 0.0135 | 0.0018 | [0.0104, 0.0176] | independent | 0.9870 | 0.0018 | [0.9830, 0.9901] |
| `ml.additivity` | exact Shapley values | 0.0000 | 0.0000 | [0.0000, 0.0010] | first column off by 1e-6 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.overfit` | equal separation | 0.0127 | 0.0018 | [0.0097, 0.0167] | train separation 2, test 1 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.weak_spot` | calibrated slices | 0.0000 | 0.0000 | [0.0000, 0.0010] | slice 0 predicts 1 - p | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.conformal_coverage` | drift 0 | 0.0620 | 0.0038 | [0.0549, 0.0699] | log-odds drift 0.5 | 0.9995 | 0.0004 | [0.9982, 0.9999] |
| `ml.leakage_screen` | leak strength 0 | 0.0000 | 0.0000 | [0.0000, 0.0010] | leak strength 4 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.data_quality` | NaN rate 0.2 percent | 0.0000 | 0.0000 | [0.0000, 0.0010] | NaN rate 3 percent | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.challenger_brier` | equal noise | 0.0302 | 0.0027 | [0.0254, 0.0360] | challenger noise 0.15 above | 0.9952 | 0.0011 | [0.9926, 0.9970] |

Reading the table:

- Several nulls give size 0 and several defects give power 1: the null and the defect of those studies are far apart (for example the
  seed plant at instability 0.2 against 1.0, the additivity gap 0 against 1e-6), so the study shows the light can tell them apart, not
  where its boundary sits.
- **Finding F4, the worst-subsample test.** The size is 0.105, not below 0.05, and it does not fall with the sample size (an
  exploratory run of 100 repetitions at 8,000 rows gave 0.19 with a weaker model, about 0.20 at 3,000 rows with the same weaker model,
  and about 0.10 at 3,000 rows with the stronger model used here). Under no shift the worst subsample by shift score is not a random
  subsample: it is a region of the feature space (a half-space in a random direction), and the model is judged there on a narrower range
  of its own score, so its AUC is lower than on the whole sample by 0.013 to 0.020 on average with a spread of 0.027 to 0.036, which puts
  the amber cut point 0.05 within reach of chance. The false-alarm rate depends on the strength of the model; the cut points 0.05 and
  0.10 are conventions with no dossier value.
- The conformal coverage test is slightly oversized (0.062 at a nominal 0.05): the calibration quantile is itself random (the
  calibration window is 10 times the test window), a randomness the binomial test does not model.
- `ml.cbpe_estimate` has no light: its study is the bias of the estimate against the labelled AUC over 4,000 samples of 600
  probabilities (`test_cbpe_estimate_tracks_the_labelled_auc_when_calibrated_and_not_when_it_is_not`): unbiased within 0.01 when the
  probabilities are calibrated, more than 0.03 away when the labels follow $p^2$.

## What these studies do not claim

- The size and power are those of the planted defects at these sample sizes, not of a portfolio: a different model strength, sample
  size or defect changes them (finding F4 shows by how much).
- A rate here says how often a test light is not green on the generator, nothing about compliance.
- The engine computes and makes no compliance claim.

## Sources

- Dossier 07 D pattern 6 l.611-616; the product SDD Method ladder l.335; Morris, White and Crowther (2019), doi:10.1002/sim.8086;
  the harness `riskvalidation.harness.run.simulate` and `riskvalidation.harness.rates`.

## Limits

The study seeds are fixed in `tests/ml/ml_sims.py`; the measured rates in the table are those of the run the unit was verified with
and the test asserts the design bounds, not the table. UNVERIFIED: the exploratory runs quoted in finding F4 (100 repetitions each)
were not kept as tests; the table is the 4,000-repetition study.
