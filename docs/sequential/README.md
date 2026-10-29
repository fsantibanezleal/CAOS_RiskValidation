# Sequential monitors: e-processes for back-tests and calibration (BS-15)

`riskvalidation.validation.sequential` reads a back-test or a calibration every day, or every vintage, without losing its
size. A fixed-window test (Kupiec, Hosmer-Lemeshow, the Basel traffic light) is valid when it is read once on a window
fixed in advance. Read every day on a growing or a sliding window it rejects a correct model far more often than its
nominal level, which is what a monitoring desk does. The monitors here keep the size.

## The idea

An *e-process* `E_t` is a non-negative process that starts at 1 and whose expected value under the null is at most 1 at
every stopping time. Ville's inequality: `P(sup_t E_t >= 1/a) <= a`. The monitor therefore rejects the first time
`E_t >= 1/a` (20 for `a = 0.05`) and has size at most `a` at any time, however often it is read and whenever the analyst
stops (anytime-valid size). `min(1, 1/sup_t E_t)` is a p-value valid at any stopping time. The statistics are kept as
`log E_t` (dimensionless).

Each monitor is a *bet against the null*: at every step the monitor stakes on one alternative, the likelihood ratio of
the alternative against the null is a non-negative martingale with mean 1 under the null, and a fixed mixture of such
bets over a set of alternatives is again one. The mixture is what makes the monitor useful when the size of the defect is
unknown, and it is also the price: the mixture spreads the bet, so the power at a fixed horizon is below the likelihood
ratio test that knows the alternative.

## Public names

| Name | Module | What it does |
|---|---|---|
| `var_hit_eprocess(hits, alpha)` | `var_hits` | VaR exceedance rate against `alpha`; alternatives `m alpha`, default `m` = 0.25, 0.5, 1.5, 2, 3, 4, 6, 10, equal weights; `direction` keeps one side |
| `var_hit_clustering_eprocess(hits, alpha)` | `var_hits` | a hit after a hit: first-order Markov alternatives `P(hit given hit) = m alpha`, default `m` = 3, 6, 12, 25, 50 |
| `pd_grade_eprocess(defaults, exposures, predicted_pd)` | `pd_grades` | observed defaults against the predicted PD of each grade by step; multiplicative miscalibration `m p`, 16-point log grid in [1/4, 4], Jeffreys-type weights; grades combined by `"product"` or `"average"` |
| `eprocess_battery(components)` | `battery` | several monitors of the same steps as one verdict: `"average"` (always valid) or `"product"` (independent components) |
| `EProcessResult`, `GradeEProcessResult` | `monitor` | the record: `log_e`, `rejection_time`, `rejected`, `anytime_p_value`, `threshold`, `to_test_result()` |
| `detection_time(log_e, alpha_level)`, `anytime_p_value(log_e)` | `monitor` | the first crossing (1-based) and the anytime-valid p-value of any path |

All are plain functions: they read no registry and no policy, register no test and state no compliance claim.
`to_test_result()` maps a record to the engine's `TestResult` (red when the monitor rejected, green otherwise,
`not_evaluated` for an empty series) with the policy label `contraste-sequential-monitors 1.0.0`.

## Assumptions and what is refused

- Hits are independent Bernoulli(`alpha`) under the null; the clustering monitor tests exactly that independence.
- Defaults of a grade are Binomial(obligors, predicted PD) given the past, the assumption of the per-grade binomial
  test. `combine="product"` also needs the grades to be independent given the past; with asset correlation across
  grades use `combine="average"`, which is an e-process whatever the dependence, at a power cost when one grade carries
  the signal. Neither monitor models the asset correlation inside a grade: that correlation is the reason the
  Vasicek-corrected tests exist, and a correlated portfolio would be over-rejected by the plain binomial null used here.
- The mixture weights are fixed before any outcome (the grid, and for the PD the mean predicted PD of the grade), which
  keeps the process an e-process. The PD may be a vector (constant over the steps) or an array by step and grade.
- A NaN, a non-flag in `hits`, a fractional or negative count, more defaults than obligors, a PD at 0 or 1, shapes that
  disagree, a battery of components of different lengths, an `alpha_level` outside (0, 1) are `ValueError`; a
  non-record in a battery is `TypeError`. An empty series gives `n = 0` and the `not_evaluated` form. A step with no
  obligor in a grade contributes a factor 1.

## Evaluation

Script: `scripts/sequential_monitors_evaluation.py` (seed 20261028, 2000 series per scenario, deterministic); its
output is committed as [`evaluation-result.json`](evaluation-result.json). Reproduce with

    python scripts/sequential_monitors_evaluation.py --reps 2000 --contraste-data <Contraste data/derived> --out result.json

The verdict rules were written in the script before the run (below). The monitors were not retuned after the run.

### Part 1: VaR hits (generator `VarHits`), alpha = 1 %, horizon 1000 days

Fractions of the 2000 series. "By 250" is the rejection by day 250 (the Basel window); "day" is the median first
rejection day over all series (`-` when more than half never reject). Kupiec trailing is the test on the last 250 days
read every day from day 250; Kupiec cumulative the test on all days so far read every day from day 50; the traffic light
is the first day with 5 or more exceptions in the last 250 days (the amber zone at 99 %); Kupiec one shot is the single
test on days 1 to 250.

| Scenario | E rate: ever / by 250 / day | Kupiec one shot (power) | Kupiec trailing: ever / day | Kupiec cumulative: ever / day | Traffic light: ever / day |
|---|---|---|---|---|---|
| null (size) | 0.022 / 0.015 / - | 0.096 | 0.655 / 729 | 0.336 / - | 0.608 / 788 |
| rate x2 | 0.556 / 0.172 / 867 | 0.248 | 0.890 / 399 | 0.875 / 280 | 0.994 / 250 |
| rate x3 | 0.980 / 0.504 / 239 | 0.626 | 0.998 / 250 | 0.999 / 111 | 1.000 / 250 |
| rate x4 | 1.000 / 0.785 / 129 | 0.866 | 1.000 / 250 | 1.000 / 66 | 1.000 / 250 |

Clustering at the right rate (P(hit given hit) = 0.1, 0.2, 0.4):

| Scenario | E cluster: ever / day | E battery: ever / day | Independence trailing: ever / day | E rate: ever |
|---|---|---|---|---|
| null | 0.004 / - | 0.009 / - | 0.076 / - | 0.022 |
| cluster 0.1 | 0.249 / - | 0.239 / - | 0.572 / 817 | 0.048 |
| cluster 0.2 | 0.531 / 939 | 0.516 / 958 | 0.787 / 471 | 0.086 |
| cluster 0.4 | 0.780 / 544 | 0.793 / 540 | 0.912 / 299 | 0.183 |

Reading. The e-process holds its size (2.2 % over a 1000-day horizon read every day; the null battery 0.9 %) where the
Kupiec test read the same way rejects a correct model in 65.5 % (trailing window) and 33.6 % (growing window) of the
series, and the traffic light leaves the green zone at some day in 60.8 %. The e-process is conservative (2.2 % against
the 5 % bound): that is the mixture and the discrete jumps of a rare event, not a defect. At 4 times the hit rate it
detects in 129 days (median), inside the 250-day window; at 3 times, in 239 days, just inside it. It is slower than an
unadjusted test read every day (Kupiec cumulative: 111 and 66 days), which is the price of the size those tests do not
have. For clustering the repeated independence test is faster (299 days at 0.4) and its null size is 7.6 %; the
clustering e-process detects 78 % of the series by 1000 days at 0.4 and is weak at 0.1 (25 %). The clustering monitor
is at 0.4 % under the null, so it is conservative too.

### Part 2: defaults against PD by grade

Five grades (PD 0.5, 1, 2, 5, 10 %), 2000 obligors per grade and year, ten yearly steps, every grade's PD multiplied by
the factor, 2000 series. The comparator is the chi-square test across the grades on the cumulative counts, read once at
year 10 (power) and read every year (the repeated reading). "Day" in `evaluation-result.json` is the year here. The
factors 1.05 and 1.1 were added after a first run in which every factor from 1.25 up was detected in the first two years
by every method; the other scenarios and all results of that run are unchanged.

| PD factor | E product: ever / by year 3 / median year | E average: ever / by year 3 / median year | Chi-square at year 10 (power) | Chi-square every year: ever / median year |
|---|---|---|---|---|
| 1 (size) | 0.000 / 0.000 / - | 0.005 / 0.004 / - | 0.051 | 0.201 / - |
| 1.05 | 0.000 / 0.000 / - | 0.038 / 0.009 / - | 0.683 | 0.791 / 6 |
| 1.1 | 0.242 / 0.029 / - | 0.611 / 0.143 / 8 | 1.000 | 1.000 / 2 |
| 1.25 | 1.000 / 0.997 / 2 | 1.000 / 0.997 / 2 | 1.000 | 1.000 / 1 |
| 1.5 | 1.000 / 1.000 / 1 | 1.000 / 1.000 / 1 | 1.000 | 1.000 / 1 |
| 2 | 1.000 / 1.000 / 1 | 1.000 / 1.000 / 1 | 1.000 | 1.000 / 1 |

Reading. The size is held (0.0 % and 0.5 %) where the chi-square read every year rejects a correct PD in 20.1 % of the
series. The power cost is large for a small miscalibration: at a 10 % relative error the single chi-square at year 10
detects it every time, the product e-process in 24 % of the series within ten years, the average in 61 %; at 5 % the
e-process detects almost nothing (the chi-square: 68 %). From 25 % relative error the e-process detects in the first two
years, as the repeated chi-square does. At the factors where both detect (1.25 and above) the two combinations are equal; at 1.1 the average detects more than the
product (61 % against 24 %), although every grade drifts by the same factor, so the product is not the better combination
here, and it is also the one that assumes independence across grades. The average is the default to prefer under
cross-grade dependence.

### Part 3: real data (Contraste C11, C03)

C11, alpha 1 %, the 17 models of the ladder, the committed exceedance days; the e-process is `var_hit_eprocess` with its
defaults.

| Window | Days | Models flagged by the traffic light | by the e-process | Median day, e-process (flagged models) | Median day, traffic light |
|---|---|---|---|---|---|
| equity-2007-09 | 504 | 17 | 17 | 24 | 250 |
| fx-2008 | 251 | 10 | 8 | 68 | 250 |

On fx-2008 the e-process misses two models the traffic light flags (`caviar-sav` and `caviar-es`: 5 exceptions in 251
days, the amber threshold, but not enough evidence for a factor-20 bet); neither it nor the light flags the 7 models with 3 or 4
exceedances. `hs-250` on equity-2007-09: e-process on day 24 (19 exceedances in 504 days); the traffic light and the
trailing Kupiec cannot speak before day 250 (the first full window), and the cumulative Kupiec, which has no
size guarantee, speaks on day 50. The comparison with the traffic light is therefore structurally in the e-process's
favour: the light reads nothing before day 250. The fairer comparison, with a test that reads earlier, is the cumulative
Kupiec of Part 1.

C03 (charge-offs by year of age against the predicted cumulative PD, per vintage; `pd_grade_eprocess` with the
upper direction, steps = the five years of age, columns = the vintages of a cohort, the PD of each interval the increment
of the predicted cumulative PD; the interval counts are `n` times the increments of the observed curve, an approximation
that ignores attrition and treats the multinomial intervals as independent binomials):

| Cohort | First age step rejected | Reading |
|---|---|---|
| FY2006-07 | 1 (12 months) | the level gap is visible at the first year |
| FY2008-10 | 1 | idem (the 2010 vintage at step 2) |
| FY2011-14 | none | the model over-predicts; `log E` falls to -1095 |
| FY2015-17 | none | idem |
| FY2018-20 | none as a cohort (the 2018 vintage alone at step 1) | `log E` -673 |

The sign of the evidence is the point: the monitor rejects when the observed defaults exceed the predicted ones and
stays below 1 when they do not. Two or three vintages per cohort are few; the C03 run is an illustration on the committed
curves, not a validation of power.

### Verdict on the hypothesis of the plan (BS-15)

| Rule (written before the run) | Measured | Holds |
|---|---|---|
| H1 size: E rate rejects at most 5 % plus 3 standard errors (6.46 %) of correct series, and the repeated Kupiec rejects more | E 2.2 %; Kupiec trailing 65.5 %, cumulative 33.6 % | yes |
| H2 delay: at hit-rate factor 3 and 4 the median detection day is below 250 and below the traffic light's | factor 3: 238.5 against 250; factor 4: 129 against 250 | yes (the traffic light's 250 is its first possible day) |
| H3 power cost: Kupiec power at 250 days minus E power by day 250 is below 10 points at factor 3 and 4 | factor 3: 62.65 % - 50.4 % = 12.2 points; factor 4: 86.6 % - 78.5 % = 8.1 points | **no at factor 3**, yes at factor 4 |
| H4 real data: on C11 equity-2007-09 and fx-2008 the median e-process day among the traffic-light-flagged models is below the light's | 24 against 250; 68 against 250 | yes (same structural caveat) |

The refutation clause of the plan (the delay not lower than the traffic light's on equity-2007-09 and fx-2008) did not
fire. The hypothesis as a whole does **not** hold as stated: the power cost at hit-rate factor 3 is 12.2 points, above
the 10 allowed. The result is reported as measured and the monitors are not retuned. The honest summary: anytime-valid
size and an earlier signal than the 250-day light on real exceedances are delivered; at a fixed 250-day horizon the
e-process loses 12 points of power at factor 3 (8 at factor 4), more for a small PD miscalibration (Part 2), and for
clustering the repeated independence test is faster at a size above 5 %.

## Sources

- Ville (1939), Etude critique de la notion de collectif, Gauthier-Villars (the maximal inequality).
- Ramdas, Grunwald, Vovk and Shafer (2023), Game-theoretic statistics and safe anytime-valid inference, Statistical Science
  38(4):576-601, arXiv 2210.01948 (e-processes, mixtures, combination of e-values).
- Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991 (e-values for VaR and ES back-tests).
- Kupiec (1995), J. Derivatives 3(2):73-84, doi:10.3905/jod.1995.407942; Christoffersen (1998), Int. Economic Review
  39(4):841-862, doi:10.2307/2527341 (the fixed-window comparators); BCBS MAR32.9 (the traffic light).
- Jeffreys (1946), Proc. R. Soc. A 186:453-461, doi:10.1098/rspa.1946.0056 (the prior of the PD mixture).
- The application to PD calibration by vintage and to a validator's battery is, to the author's knowledge, not published
  (UNVERIFIED: a literature search was not run in this unit; the research brief records the same status).

## The registered monitors

Each monitor is in the test catalogue (`riskvalidation.core.registry`); its `TestResult` (`EProcessResult.to_test_result`) names the
id, the null hypothesis and the reference below, and decides on the anytime-valid p-value at the monitor's size.

| Test id | Monitor | Null hypothesis |
|---|---|---|
| `sequential.var_hits` | `var_hit_eprocess` | the hits are independent Bernoulli(alpha): the VaR exceedance probability equals alpha |
| `sequential.var_hit_clustering` | `var_hit_clustering_eprocess` | no hit is more likely the day after a hit |
| `sequential.pd_grades` | `pd_grade_eprocess` | the defaults of every grade are Binomial(obligors, predicted PD) given the past |
| `sequential.battery` | `eprocess_battery` | every component null holds (the intersection of the component nulls) |
