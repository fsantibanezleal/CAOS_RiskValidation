# 08. The scale-and-level complement to the PLA test

Module `riskvalidation.market.pla_complement` (added in 1.1.0). It adds three statistics
and a calibrated decision to the regulatory P&L attribution test of [page 06](06_frtb-tests.md). The complement is added to the
regulatory zone, never substituted for it, and has no regulatory standing: `pla_complement` returns the zone of
`frtb.pla_zone_exact` unchanged beside its own decision.

## Why

The regulatory PLA reads the Spearman correlation of the levels of HPL and RTPL and the Kolmogorov-Smirnov distance of their
distributions. On the Contraste C13 desk generator (two-factor spread, log-AR(1) volatility known at the start of the day, true
risk model known) a risk model that uses a 20-day-old volatility passes both: 0.0 % of 1,000 replications amber or red.
The model has the right correlation structure and the right marginal distribution over a year; what it gets wrong is when the
scale moves.

## The statistics

All on the 250-day pair (MAR32.20), HPL and RTPL in the same unit (a decimal of the desk notional).

$$HPL_t=a+b\,RTPL_t+e_t,\qquad W=(\hat\beta-(0,1))'\,\hat V_{HC3}^{-1}\,(\hat\beta-(0,1))$$

1. `pla_mz_wald`: the Mincer-Zarnowitz regression and the joint Wald test of `(a, b) = (0, 1)` with the HC3 covariance
   (MacKinnon and White 1985), because the P&L volatility is time-varying. It sees a mis-scaled or shifted model. The returned
   p-value is the asymptotic chi-square(2) tail, a reference only.
2. `pla_abs_spearman`: the Spearman correlation of `|HPL|` and `|RTPL|`, which sees a model that misreads the size of the day's
   move.
3. `pla_rolling_abs_spearman`: the Spearman correlation of the 20-day rolling means of `|HPL|` and `|RTPL|` (`window` from 2 to
   247 days). The daily shock is shared by HPL and RTPL, so the rolling mean removes it and leaves the two scales; a model fed a
   volatility `L` days old moves its scale `L` days late, and the statistic falls as `L` grows toward the window.

`pla_complement_statistics` returns the three on one pair. Golden checks in `tests/market/test_pla_complement.py`: the Wald
statistic equals an independent HC3 sandwich computation, the two Spearman statistics equal `scipy.stats.spearmanr` on the
transformed series, the Spearman statistics are unchanged by scaling RTPL and the Wald statistic is not.

## The decision and its size

`pla_complement_calibrate(null, size, correction)` takes the statistics of `N` desk-years of a model with nothing planted
(simulated from the caller's own generator) and sets the critical values: the Wald statistic flags above its value, the Spearman
statistics below theirs. With `k = floor(p N)` the critical value is the `(k+1)`-th largest (Wald) or smallest (Spearman) null
value, so at most `k <= p N` null desk-years are flagged by each statistic; `correction="bonferroni"` uses `p = size / 3`, so the
union of the three flags has a size of at most `size`; `"none"` uses `p = size` for each. `ValueError` when the null cannot
resolve `p` (`N < 1 / p`). `pla_complement_decide` applies the thresholds; `pla_complement` adds the regulatory zone.

Missing or degenerate input (not 250 finite observations, a series with no spread, a constant absolute series, a window out of
range) raises `ValueError`, as the `frtb` statistics do. Nothing here draws random numbers.

## The evaluation

Script `scripts/pla_complement_eval.py` (needs a CAOS_Contraste checkout for the C13 generator, read only). Fixed before the run
and not tuned: window 20 days, size 1 %, thresholds calibrated on 10,000 independent null desk-years (the `none` variant under
ten case seeds, 5013 to 5022, whose seeds meet none of the committed ones), then applied unchanged to the 6,000 committed
replications (case seed 13, 1,000 per variant) and to the lag ladder (the committed stale-volatility seeds with the lag set to
5, 10, 20 and 40 days). The run took 386 seconds; it is deterministic.

Critical values (10,000 null desk-years): at 1 % per statistic Wald above 9.833, `|.|` Spearman below 0.9353, rolling Spearman
below 0.9619; at 1 % for the union (Bonferroni, 0.333 % each) Wald above 12.007, `|.|` Spearman below 0.9263, rolling Spearman
below 0.9493.

Rejection rates in percent of 1,000 replications. "PLA" is amber or red under MAR32.42; "KS" is the KS statistic alone at 0.09
or more (not KS-green); "union" is any of the three statistics at 1 % for the union.

| Variant | PLA | KS alone | Wald | `abs` Spearman | Rolling Spearman | Union (family 1 %) | Union (95 % Wilson) |
|---|---|---|---|---|---|---|---|
| none | 0.0 | 0.0 | 0.8 | 1.6 | 1.2 | 0.9 | 0.5 to 1.7 |
| missing-basis | 96.0 | 86.7 | 0.9 | 100.0 | 99.9 | 100.0 | 99.6 to 100 |
| stale-volatility | 0.0 | 0.0 | 14.2 | 76.7 | 99.9 | 99.9 | 99.4 to 100 |
| wrong-correlation | 100.0 | 100.0 | 100.0 | 8.7 | 4.5 | 100.0 | 99.6 to 100 |
| omitted-factor | 90.1 | 44.8 | 0.6 | 100.0 | 98.7 | 100.0 | 99.6 to 100 |
| time-misalignment | 100.0 | 0.0 | 100.0 | 100.0 | 42.3 | 100.0 | 99.6 to 100 |

The three statistic columns are at 1 % per statistic. The union at 1 % per statistic is 3.3 % on the null variant (Wilson 2.4 to
4.6 %), as three statistics at 1 % each allow up to 3 %; the Bonferroni column holds the union to 1 %: 0.9 % on the null variant,
Wilson 0.5 to 1.7 %.

Lag ladder (stale-volatility variant, same seeds, percent): rolling Spearman at 1 % per statistic flags 57.6 % at 5 days, 97.5 %
at 10, 99.9 % at 20 and 100.0 % at 40 days; the regulatory zone flags 0.0 %, 0.0 %, 0.0 % and 0.2 % (KS alone the same). The
Wald statistic flags 2.4, 6.0, 14.2 and 28.7 % and the `abs` Spearman 4.5, 25.6, 76.7 and 94.1 %. At 1 % for the union the
rolling statistic flags 36.5, 93.4, 99.7 and 100.0 %.

Power against size on the stale-volatility variant (the held-out null variant in brackets; percent): at 0.1 % per statistic the
rolling Spearman flags 99.1 (0.1), at 0.5 % 99.9 (0.3), at 1 % 99.9 (1.2), at 2 % 100.0 (1.9) and at 5 % 100.0 (4.3); the
Wald statistic flags 6.9, 11.1, 14.2, 17.6 and 26.3.

Size under a different null (the thresholds above applied unchanged to 1,000 desk-years of a null with less persistent
volatility, phi 0.90, and with less dispersed volatility, sd 0.20): the union at 1 % per statistic is 1.0 % and 2.1 %, the
union at 1 % for the union 0.1 % and 0.5 %. The size does not rise when the volatility is smoother.

## Verdict

**Hypothesis of the plan: holds.** At a size of at most 1 %, the stale-volatility model is flagged in at least 30 % of
replications; measured 99.9 % (999 of 1,000, Wilson 99.4 to 100.0 %) by the rolling-mean Spearman statistic at 1 % per statistic
(held-out size 1.2 %, Wilson 0.7 to 2.1 %), and 99.7 % (997 of 1,000) at 1 % for the union of the three statistics, whose
held-out size is 0.9 %; the regulatory test flags 0.0 %. The run is 10,000 null desk-years for calibration and 1,000 desk-years
of the committed variant. The refutation bound (below 10 %) is far from the measurement.

**Secondary clause of the research brief, in part not supported.** Statistic (a), the Wald test, flags wrong-correlation at
100 % (the KS statistic alone: 100 % not green, 91.3 % red), so it holds there. It does not flag missing-basis: 0.9 % against
86.7 % for the KS statistic alone and 43.7 % red, the nominal rate. The reason is structural: the basis noise sits in the
dependent variable HPL and is independent of RTPL, so the regression of HPL on RTPL keeps the intercept 0 and slope 1; the
`abs` Spearman statistic is the one that sees it (100 %). The Wald statistic also flags stale-volatility only weakly (14.2 %).

**What the complement does not do.** The rolling statistic is weak against time-misalignment by one day (42.3 % at 1 %, where the
regulatory zone flags 100 %) and against wrong-correlation (4.5 %, where the regulatory zone flags 100 %): the complement is a
supplement to the regulatory zones, not a replacement. The union with the regulatory zone (`any_or_pla` in the script
output, 1 % per statistic) flags 100 % on the five planted variants and 59.4 % and 97.9 % on stale-volatility at lags of 5 and
10 days.

## Limits and sources

- The thresholds are specific to the null generator that produced them; a desk with a different volatility process needs its
  own null. The size under the two changed nulls above did not exceed the nominal rate, which is evidence, not a guarantee.
- The cross-check of size on the Contraste C22 case named in the plan could not be run: C22's generators (G1 to G7) produce
  default counts, scores, migrations and samples, none a P&L pair. The size was checked instead on the 1,000 held-out null
  desk-years of C13 and on the two changed nulls above.
- Mincer and Zarnowitz (1969), "The evaluation of economic forecasts", in Mincer (ed.), Economic Forecasts and Expectations,
  NBER (the regression); MacKinnon and White (1985), Journal of Econometrics 29(3):305-325 (the HC3 covariance): both cited
  from memory of the standard references, not re-read for this unit (UNVERIFIED). The power study of the regulatory thresholds,
  Montoro, Spinaci and Georgi (Risk.net), is paywalled and was not read.
- The regulatory standing of an added test is none; it does not change MAR32.42, MAR32.43 or MAR32.44.
