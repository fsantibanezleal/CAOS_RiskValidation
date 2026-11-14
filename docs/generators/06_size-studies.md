# 06. Named size and power studies of the p-value tests

Every test with a p-value has a measured size, and (since U16-E2, BL-088) every one of them is a **named study**: an importable
configuration of a null scenario and of one alternative, with its seeds and the fixed arguments of the test.
`riskvalidation.generators.size_studies` holds the registry; the `sizepower` suites under `tests/` and Contraste's C22 read the same
names, so a scenario is never rebuilt. The registry holds 21 tests (the nine that had no study, the twelve whose scenario was inline in a suite).

## The registry

- `STUDIES`: `STUDIES[test_id]` is the tuple of the studies of a test (two for `portfolio.tail_backtest`, one per level; four for
  `market.dynamic_quantile_test`). `study(test_id, index=0)` returns the `index`-th.
- `Study`: one study. Its fields are the test id, a `label`, the `null` generator (at the boundary of the test's null with its
  assumptions met), the `alternative` generator (the same with one stated departure, described in `alternative_text`), the seeds
  (`seed`, `power_seed`), the fixed arguments `test_kwargs`, the repetitions (`n_rep`, `power_n_rep`), a `feed` that turns a draw into the
  arguments of the test, optional runners for a test that needs a pipeline, and the `rule` of rejection.
- `RULES`: the rejection rules a study may name: `p` (reject when p is below 0.05), `p_le` (at or below 0.05, for a Monte Carlo p that can
  equal it), `red` (the light is red, the back-tests' 5 percent red level) and `alarm` (the light is not green).
  `rejection_label(study)` is the label of the rate of the study's own rule (`p<0.05`, `p<=0.05`, `red` or `alarm`).
- `N_REP`: the default repetitions of a study, 4,000; a study of a costly test names fewer in its own `n_rep` and `power_n_rep`.
- `simulate_study(study, which="null", *, n_rep=None, seed=None)`: runs the null or the alternative with `harness.simulate`, using the
  study's generator, seed, feed, arguments and rule. It is the one entry point of the suites and of C22.
- `scripts/size_study_sharded.py`: runs the repetitions of a registered study in slices on a process pool and sums the counts
  (`python scripts/size_study_sharded.py --study <test id> --reps <n> --workers <k> --out <file>`). Each repetition keeps its own
  spawned seed, so the counts, the p-values and the rates equal `simulate_study`'s exactly for any number of workers; a local
  tool for the `sizepower` hours, outside CI. Test: `tests/test_size_study_sharded.py`.
- `STAGE_MATRIX`, `STAGE_SHIFT`, `STAGE_ROWS`: the predicted three-stage migration matrix of `ifrs9.stage_migration_backtest`, the
  departure of the alternative (1.5 points of the stage 1 accounts move to stage 2) and the accounts per starting stage (4,000, 1,500, 500:
  the smallest expected count is 10, the chi-square assumption).

## The generators

Each is a frozen configuration with an exact truth and a verifier, as the other generators of this package (page 01).

| Generator | Module | Mechanism |
|---|---|---|
| `StageTransitions` | `generators.credit` | one-year transitions of three IFRS 9 stages: counts drawn row by row from a multinomial with the predicted matrix plus a stated `shift` (zero is the null) |
| `LgdCalibration` | `generators.credit` | realised LGDs, Beta with mean `predicted + bias` in `n_periods` default years of `n_per_period` workouts (a bias in `biased_periods` only) |
| `DownturnLgd` | `generators.credit` | the realised LGDs of the defaults of a downturn period, Beta with mean `estimate + shortfall` |
| `PfeSeries` | `generators.exceedances` | a realised mark-to-market against a PFE forecast at quantile `q`; the forecast is the true quantile times `forecast_factor` |
| `EsResiduals` | `generators.exceedances` | normal losses with their true VaR and ES forecasts; the ES forecast is the true one times `es_factor` |
| `LhpLossRates` | `generators.exceedances` | annual loss rates of a large homogeneous portfolio, one-factor Gaussian model with the true correlation, handed to the tail back-test next to the fitted `pd` and `rho` |
| `VarHits` | `generators.hits` | VaR exceedances at level `alpha`: independent Bernoulli hits under the null, `rate_factor` scales the hit probability and `cluster` makes a hit follow a hit |
| `EsLosses` | `generators.hits` | standard normal losses against the true VaR and ES of level `alpha`; `scale` above 1 understates the risk |
| `PitSeries` | `generators.hits` | probability integral transforms of the realised losses, uniform under the null; `tail_shape` above 1 puts too much mass in the lower tail |
| `McQuantile` | `generators.montecarlo` | the quantile of the loss rate of a large homogeneous portfolio from `scenarios` draws, with the standard error of the sample quantile; the anchor is the closed form of Vasicek (2002) |
| `CreditRiskPlusSample` | `generators.montecarlo` | a Monte Carlo sample of the Poisson-Gamma losses of a one-sector CreditRisk+ portfolio next to the pmf of the stable recursion |
| `lhp_quantile_closed_form` | `generators.montecarlo` | the `level`-quantile of the large-portfolio loss rate, straight from the normal functions (Vasicek 2002) |
| `sample_quantile_with_se` | `generators.montecarlo` | the sample quantile and its standard error by Woodruff's method (1952): the half width of the distribution-free interval over z |

These classes are exported by `riskvalidation.generators`; the modules are also public by path.

## The pipelines

`riskvalidation.generators.pipelines` is the glue between a plant and the ML tests that read engine outputs and not raw arrays.
`run_shift(plant, share, shift_permutations=99)` returns a runner that feeds the worst-subsample performance and its permutation
reference to `ml.shift_resilience` and returns its result; `run_conformal(n_alpha, n_cal)` returns a runner that feeds the split-conformal
quantile and the empirical coverage to `ml.conformal_coverage` (`n_cal` keeps the first `n_cal` calibration points, so the three split
ratios share one generator). `SHIFT_N` (2,000 rows) and
`SHIFT_PERMUTATIONS` (99) are the size of the shift plant and the permutations of its reference. The pipelines are public by path; the
package `generators` does not import them (they import the ML family).

The measured sizes and powers of each study are on the page of its test (for example the eight market back-tests on
[market/07_backtests](../market/07_backtests.md)).
