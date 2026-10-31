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
| `CovariateShiftPlant(n, shift, features, seed)` | (the size study uses n = 300) a development and an evaluation sample from a stronger portfolio (about 50 percent defaults, AUC about 0.85); a quarter of the evaluation rows are shifted by `shift` sd on `features`; the true logit loses 2 per unit of $x_0$ beyond 1.5 (slope 1.6 reverses to -0.4), the linear model `predict` does not | the shifted features, the share 0.25, the evaluation means; shift 0 is the null |
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
the first run, were a size of at most 0.05 and a power of at least 0.90, with two exceptions: the conformal coverage test (size at
most 0.10) and the worst-subsample test (size at most 0.15). **History, stated plainly.** The 0.10 and 0.15 bounds were not stated
before the run: the first design measured a size of 0.0620 for the conformal test and 0.1055 for the worst-subsample test (finding
F4), and the bounds were set after those runs. The review of the unit refused them in two major findings, and the designs were
replaced, not the bounds: the Beta-Binomial null (page 05) and the development-only reference (page 01). Every test now answers to
the default bound, a size of at most 0.05 plus two Monte Carlo standard errors (0.0569 at 4,000 repetitions), and a power of at least
0.90, with no exception. A first pooled-permutation reference for the shift test had a power of 0.28 and a power floor of 0.15 was set after that
measurement; the main session replaced the reference (decision after the first return) and the floor was removed: the pre-stated 0.90 applies.

## Measured size and power (4,000 repetitions per condition, seed 20261019 for the null and 20261020 for the defect)

Engine 1.1.0, run by `pytest tests/ml/test_ml_size_power.py -s` (marker `sizepower`).

| Test | Null | Size | se | Wilson 95% | Defect | Power | se | Wilson 95% |
|---|---|---|---|---|---|---|---|---|
| `ml.robustness` | sensitivity 0.25 | 0.0278 | 0.0026 | [0.0230, 0.0333] | sensitivity 2.5 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.shift_resilience` (development 2,000, evaluation 1,000, R = 99, share 0.25) | shift 0 | 0.0418 | 0.0032 | [0.0359, 0.0484] | a quarter of the rows shifted 3 sd on feature 0 | 0.9720 | 0.0026 | [0.9664, 0.9766] |
| `ml.seed_stability` | instability 0.2 | 0.0000 | 0.0000 | [0.0000, 0.0009] | instability 1.0 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.reason_code_stability` | refit noise 0.1 | 0.0000 | 0.0000 | [0.0000, 0.0009] | refit noise 1.5 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.effect_agreement` | effect range follows importance | 0.0135 | 0.0018 | [0.0103, 0.0175] | independent | 0.9870 | 0.0018 | [0.9829, 0.9900] |
| `ml.additivity` | exact Shapley values | 0.0000 | 0.0000 | [0.0000, 0.0009] | first column off by 1e-6 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.overfit` | equal separation | 0.0127 | 0.0018 | [0.0097, 0.0167] | train separation 2, test 1 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.weak_spot` | calibrated slices | 0.0000 | 0.0000 | [0.0000, 0.0009] | slice 0 predicts 1 - p | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.conformal_coverage` (n_cal 20,000, m 2,000) | drift 0 | 0.0475 | 0.0034 | [0.0413, 0.0545] | log-odds drift 0.5 | 0.9995 | 0.0004 | [0.9981, 0.9998] |
| `ml.conformal_coverage` (n_cal = m = 2,000) | drift 0 | 0.0493 | 0.0034 | [0.0429, 0.0563] | log-odds drift 0.5 | 0.9845 | 0.0020 | [0.9801, 0.9878] |
| `ml.conformal_coverage` (n_cal 1,000, m 4,000) | drift 0 | 0.0522 | 0.0035 | [0.0457, 0.0595] | log-odds drift 0.5 | 0.9545 | 0.0033 | [0.9475, 0.9605] |
| `ml.leakage_screen` | leak strength 0 | 0.0000 | 0.0000 | [0.0000, 0.0009] | leak strength 4 | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.data_quality` | NaN rate 0.2 percent | 0.0000 | 0.0000 | [0.0000, 0.0009] | NaN rate 3 percent | 1.0000 | 0.0000 | [0.9990, 1.0000] |
| `ml.challenger_brier` | equal noise | 0.0302 | 0.0027 | [0.0254, 0.0360] | challenger noise 0.15 above | 0.9952 | 0.0011 | [0.9925, 0.9969] |

Run time (default device, other sessions loading the CPU): `ml.shift_resilience` 17,495 s (about 2.2 s per repetition at 2,000 development rows, 1,000 evaluation rows and R = 99; the 4,000-repetition studies of the null and the defect), the next two 98 s and 89 s.

Reading the table:

- Several nulls give size 0 and several defects give power 1: the null and the defect of those studies are far apart (for example the
  seed plant at instability 0.2 against 1.0, the additivity gap 0 against 1e-6), so the study shows the light can tell them apart, not
  where its boundary sits.
- **Finding F4, the worst-subsample test, closed by the development-only reference.** The first design read the raw drop against fixed
  cut points and had a size of 0.1055 under no shift: the worst subsample by shift score is a region of the feature space on which the
  model's AUC is lower by range restriction, so the amber cut point 0.05 was within reach of chance. A second design, a permutation of
  the domain labels of the pooled rows, was exact in size (0.0432) but had a power of 0.28 at any sample size, because the pooled null
  contained the shifted rows. The light now reads the observed drop against the drops of random development subsets of the evaluation
  size (page 01): size 0.0418 (se 0.0032), power 0.9720 (se 0.0026).
- **Sample sizes of the shift study.** The reference needs at least twice as many development rows as evaluation rows. The plant draws
  `n` rows for each sample and the pipeline keeps the first `n / 2` evaluation rows. Exploratory power at 40 to 60 repetitions
  (UNVERIFIED as tests, not kept): 0.70 at 600 development and 300 evaluation rows, 0.90 at 1,000 and 500, 0.975 at 2,000 and 1,000;
  the study uses 2,000 and 1,000, the smallest size that left a margin over the 0.90 design power.
- The conformal coverage test is exact under exchangeability at the three split ratios (sizes 0.0475, 0.0493 and 0.0522 against the
  bound 0.0569). For comparison the dossier's binomial p-value, kept in `p_value_binomial`, has sizes 0.062, 0.1625 and 0.380 at the
  same ratios (4,000 repetitions, the same draws; one-off study, not a test): the oversizing the review found, which the
  Beta-Binomial null removes. The power falls as the calibration window shrinks against the test window (0.9995, 0.9845, 0.9545).
- `ml.cbpe_estimate` is always `not_evaluated` (an estimate, not a test): its study is the bias of the estimate against the labelled AUC over 4,000 samples of 600
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
and the test asserts the design bounds, not the table. UNVERIFIED: the exploratory 100-repetition power runs quoted for the shift
resilience test were not kept as tests; the table is the 4,000-repetition study. The module carries the `sizepower` marker: the
engine CI deselects it (`-m "not sizepower"`, the integrator registers the marker in `pyproject.toml`; until then pytest warns of an
unknown marker) and the local release gate runs it.
