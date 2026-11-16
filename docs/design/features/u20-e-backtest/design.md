# U20 E-backtests of VaR and expected shortfall, design

Source: Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991v7. Every definition below cites its equation, theorem or
remark in that paper; the engine's conventions (positive losses, `alpha` the exceedance probability, `alpha_level` the size of the
monitor) are stated in `requirements.md`.

## Module

`riskvalidation.validation.sequential.e_backtest` (re-exported from `riskvalidation.validation.sequential`), NumPy and SciPy
only. It reuses the record of the existing monitors: `EBacktestResult` subclasses `EProcessResult`
(`monitor.py`), so `rejection_time`, `rejected`, `anytime_p_value`, `threshold`, `e_process` and `to_test_result` are the
engine's own, and the input checks `check_alpha_level`, `check_open_unit` and `frozen` come from `sequential/_core.py`.

| Name | What | Paper |
|---|---|---|
| `var_backtest_e_statistic` | `1{loss > var} / alpha` | equation (1), Example 3, Theorem 4 |
| `es_backtest_e_statistic` | `(loss - var)+ / (alpha (es - var))`, 0/0 = 1, 1/0 = infinity | equation (3), Theorem 1, Theorem 5 |
| `betting_process` | the bets of GREE, GREL or GREM | equations (13), (14), (15), Section 5.1 |
| `e_backtest_var`, `e_backtest_es` | the e-process of equation (7) as an `EBacktestResult` | equation (7), Theorem 2, Section 4.2 |

## Computation

- Everything is in log space: `log M_t = cumsum(log1p(lambda_t (X_t - 1)))`. A day with `lambda_t = 0` contributes 0 whatever
  `X_t` (an infinite e-statistic that was not bet on is not paid), a day with `lambda_t > 0` and `X_t = infinity` makes the path
  `+infinity` for good (`gamma < 1`, so no later factor is 0).
- **GREE** (equation (13)) uses the e-statistic of every past day on that day's own forecast, so the closed form (15) is a
  difference of cumulative sums of `e - 1` and `(e - 1)^2` (linear in the number of days; a rolling window is the difference of
  the cumulative sums at its two ends; infinite e-statistics are counted apart so that no `inf - inf` arises).
- **GREL** (equation (14)) re-values the past losses at the forecast of the day being bet on, an `n x n` triangular
  computation done in row blocks of about two million cells. When both forecast series are constant it is the GREE computation
  itself, which is the paper's statement that the two are equal then and keeps the common case linear.
- **GREM** is the equal mixture of the GREE and GREL wealth in log space (`logaddexp`), with the betting process of the paper
  `(M_{t-1}^GREE lambda^GREE + M_{t-1}^GREL lambda^GREL) / (M_{t-1}^GREE + M_{t-1}^GREL)` from the weights
  `expit(log M^GREE - log M^GREL)` (one half each when both are infinite).
- **The exact solver** maximises `mean log(1 + lambda (e - 1))` over `[0, gamma]`. The function is concave, its derivative
  `mean (e - 1) / (1 + lambda (e - 1))` is decreasing and `1 + lambda (e - 1) >= 1 - gamma > 0`, so 56 bisections of
  `[0, gamma]` reach the optimum; a derivative at least 0 at `gamma` gives `gamma`, one at most 0 at 0 gives 0. This is the
  "convex programming" of the paper's equations (13) and (14) without a solver dependency.

## Choices the paper leaves open

Listed in `requirements.md` and on `docs/sequential/02_e-backtest.md` with their reason: the first day bets 0; a sample with mean
e-statistic at most 1 bets 0; an infinite e-statistic in the sample bets the cap; `gamma` defaults to 1/2 (Remark 7);
GREM is the default (Sections 5.2 and 8); an ES forecast below its VaR forecast is refused. The paper's Remark 5 (the other side
of the VaR test by the negated loss) and a weighted GREM are not offered: the first is a call with the negated loss, the second
is out of this unit's scope.

## Registry and the record

Both functions are registered (`sequential.e_backtest_var`, `sequential.e_backtest_es`, family `sequential`, decision `metric`,
like the existing monitors) with the null hypotheses of the paper's equations (8) and (9) and the same reference string the
result carries; the session-wide registry check (`tests/test_registry_complete.py`) reads the `TestResult` they build. The
result's `inputs_hash` covers the losses, the forecasts, `alpha`, the level and every option, so a changed option changes it.

## Tolerances

The e-statistics are exact in floating point on the tests' values. The VaR e-process equals the manuscript's reference to
1e-12 on a 750-day path (the two compute the same sums by different routes: cumulative sums of `e - 1` here, of the hit flag
there); GREE and GREL equal the loops to 1e-13 in the bets and 1e-11 in the path; the exact solver equals a bounded scalar
minimiser to 1e-7 and the closed Kelly fraction to 1e-9; the mean of `M_T` by enumeration is 1 to 1e-12.
