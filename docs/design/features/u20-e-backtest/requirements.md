# U20 E-backtests of VaR and expected shortfall: the backtest e-statistics and the betting processes of Wang, Wang and Ziegel (2022) (BL-121)

Sources: Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991 (version 7, read in full: Definitions 1 and 2, equation (1) and
Example 3, equation (3) and Theorem 1, Proposition 1, equation (7) and Theorem 2, Section 5.1 equations (12) to (15) and the GREM mixture,
Theorem 3, Remarks 6 to 8, Theorems 4 and 5, Section 7 and Appendix A). The decisions of 2026-11-16 (CAOS_MANAGE `plans/contraste/backlog.md`
BL-121): the published comparator of the Contraste e-process monitors becomes an engine method, so that the ladder row M-39 is run by the
engine and taught by a notebook, never hand-rolled in a product. The oracle for the VaR case is the reference function `wwz_log_e` of the
Contraste manuscript `e-process-sequential-monitors` (`manuscripts/e-process-sequential-monitors/experiments/common.py`, CAOS_Contraste),
copied into the test with its source; every definition is the paper's.

EARS (Mavin et al., RE'09, doi:10.1109/RE.2009.9). Every requirement names the gate that fails when it is violated. Conventions: losses are
positive numbers and a hit is `loss > forecast` (the engine's `market` convention); `alpha` is the exceedance probability, `alpha = 1 - p`
for the paper's level `p` (so `alpha = 0.025` is a 97.5% VaR or ES); `alpha_level` is the size of the monitor (threshold `1 / alpha_level`;
the paper's alert thresholds 2, 5 and 10 are `alpha_level` 0.5, 0.2 and 0.1). Randomness is `numpy.random.Generator(PCG64(seed))`. Existing
monitors are unchanged (`docs/api-policy.md`: names are added, none changed).

Choices the paper leaves open, each stated in the docstrings and on the page `docs/sequential/02_e-backtest.md`: the first day bets nothing
(no past sample); a past sample whose mean e-statistic is at most 1 bets nothing (Proposition 2 (i)); a past e-statistic equal to infinity
makes the growth rate infinite for every positive bet and the bet is the cap `gamma`; the default betting process is GREM (Section 5.1: "the
GREM method is recommended as a default choice"), with the closed form (15) the paper uses in its simulations (Sections 7 and 8) and the
growth-rate maximisation of (13) and (14) selectable; an ES forecast below its VaR forecast, for which the paper sets the e-statistic to
infinity as "a case of no relevance", is refused.

## Backtest e-statistics

| ID | Requirement | Gate |
|---|---|---|
| RV-2101 | THE `var_backtest_e_statistic` SHALL be the paper's equation (1), `1{loss > var} / alpha` (strict inequality), per day, on hand-computed values, and SHALL refuse a NaN or infinite loss or forecast, arrays of different lengths or more than one dimension, and `alpha` outside (0, 1), naming the field. | `tests/test_sequential_e_backtest.py::test_var_e_statistic_values_and_refusals` |
| RV-2102 | THE `es_backtest_e_statistic` SHALL be the paper's equation (3), `(loss - var)+ / (alpha (es - var))` with the paper's conventions 0/0 = 1 and 1/0 = infinity when `es = var`, on hand-computed values, and SHALL refuse an ES forecast below the VaR forecast and the inputs refused by RV-2101. | `tests/test_sequential_e_backtest.py::test_es_e_statistic_values_conventions_and_refusals` |
| RV-2103 | THE e-statistics SHALL have expectation at most 1 under the null and more than 1 under an under-forecast (Definitions 1 and 2): the expectation of the ES e-statistic of a standard normal loss SHALL be 1 to 1e-9 by quadrature at the true VaR and ES (equality in Theorem 1), below 1 for an over-forecast of the ES and above 1 for an under-forecast; that of the VaR e-statistic SHALL be 1 at the true quantile (Example 3); and a Monte Carlo of 200,000 draws SHALL put each mean within four standard errors of 1 (a run of 20,000,000 under `sizepower`). | `tests/test_sequential_e_backtest.py::test_expectations_by_quadrature_and_by_monte_carlo`, `tests/test_sequential_e_backtest.py::test_expectation_monte_carlo_large` |

## Betting processes and the e-process

| ID | Requirement | Gate |
|---|---|---|
| RV-2104 | THE e-process SHALL be the paper's equation (7), `M_t = prod_s (1 - lambda_s + lambda_s X_s)` with `X_s` the e-statistic of day `s`, returned as `log M_t` per day, with `lambda_1 = 0`, every `lambda_t` in `[0, gamma]` and `gamma` in (0, 1) (default 1/2, Remark 7); on a hand-computed series the bets and the path SHALL equal exact fractions to 1e-14. | `tests/test_sequential_e_backtest.py::test_the_e_process_on_a_hand_computed_series` |
| RV-2105 | THE betting process GREE SHALL be the paper's (13) by the closed form (15) with the forecasts of each past day, and GREL the paper's (14) by (15) with the forecasts of the current day, each on all past days, each equal to a brute-force loop written from the equations on VaR and on ES series with time-varying forecasts; GREE and GREL SHALL be equal when the forecasts are constant (the paper, after (14)). | `tests/test_sequential_e_backtest.py::test_gree_and_grel_equal_a_loop_from_the_equations`, `tests/test_sequential_e_backtest.py::test_gree_equals_grel_for_constant_forecasts` |
| RV-2106 | THE VaR e-process by GREL on constant forecasts (equal to GREE) SHALL equal the Contraste manuscript's `wwz_log_e` to 1e-12 on seeded series at three hit rates and two levels, and the closed form of Appendix A for VaR SHALL equal the engine's bet. | `tests/test_sequential_e_backtest.py::test_the_var_e_process_equals_the_manuscripts_reference` |
| RV-2107 | THE `solver="exact"` option SHALL maximise the empirical growth rate of (13) and (14) over `[0, gamma]`: on a hit series it SHALL equal the closed form `(h - alpha) / (1 - alpha)` clipped to the interval (h the hit share), SHALL equal a bounded scalar minimiser to 1e-8 on ES series, and SHALL reach an empirical growth rate at least that of the closed form (15). | `tests/test_sequential_e_backtest.py::test_the_exact_solver_maximises_the_growth_rate` |
| RV-2108 | THE method GREM SHALL be the equal mixture of the GREE and GREL wealth, `M = (M_GREE + M_GREL) / 2`, with the betting process `(M_{t-1}^GREE lambda^GREE + M_{t-1}^GREL lambda^GREL) / (M_{t-1}^GREE + M_{t-1}^GREL)` of the paper, and the product (7) of that betting process SHALL reproduce the mixture's path to 1e-10. | `tests/test_sequential_e_backtest.py::test_grem_is_the_wealth_mixture_with_the_paper_betting_process` |
| RV-2109 | THE option `window` SHALL compute the empirical sample of GREE and GREL on the last `window` past days only (Section 5.1, Remark 8, Section 8), equal to a loop, and a window at least as long as the series SHALL give the unwindowed result. | `tests/test_sequential_e_backtest.py::test_the_rolling_window` |

## Validity and power

| ID | Requirement | Gate |
|---|---|---|
| RV-2110 | THE e-process SHALL be a martingale with mean exactly 1 at every day under the null (Theorem 2): by full enumeration of all 2^10 hit sequences (2^8 for the exact solver) for the VaR e-statistic and of all 3^7 loss sequences of a three-point loss law with `ES = 3` and `VaR = 1` at level 0.9 for the ES e-statistic, `E[M_T]` SHALL equal 1 to 1e-12 for GREE, GREL and GREM, closed form and exact solver, with and without a window; and an over-forecast SHALL give `E[M_T]` below 1. | `tests/test_sequential_e_backtest.py::test_the_e_process_has_mean_one_by_enumeration` |
| RV-2111 | THE monitor SHALL hold its size at any stopping time (Ville, Theorem 2): the share of null series whose path ever reaches `1 / alpha_level` SHALL not exceed `alpha_level` plus three binomial standard errors at `alpha_level` 0.5, 0.2, 0.1 and 0.05, for the VaR and the ES e-process, constant and time-varying forecasts (a CI run of 400 series; 4,000 series of 1,000 days under `sizepower`). | `tests/test_sequential_e_backtest.py::test_size_at_any_stopping_time`, `tests/test_sequential_e_backtest.py::test_size_at_any_stopping_time_large` |
| RV-2112 | THE monitor SHALL detect a planted under-forecast (a VaR at 70% of the true value, an ES and a VaR each at 70%) before the end of 1,500 days in at least 90% of 100 seeded series, and SHALL not reject an over-forecast of 130%; and under the paper's Proposition 1 the ES e-process SHALL not reject when both forecasts are multiplied by the same factor above 1. | `tests/test_sequential_e_backtest.py::test_a_planted_under_forecast_is_detected_and_an_over_forecast_is_not` |

## Edge cases, record and surface

| ID | Requirement | Gate |
|---|---|---|
| RV-2113 | THE functions SHALL treat a series of length 0 (an empty path, `n = 0`, the not-evaluated form), of length 1 (`log M = 0`, no bet), with no hit (every bet 0, `M = 1`, Remark 8), with a hit every day (`lambda = alpha / (1 - alpha)`, `M_t = 2^(t-1)`) and an ES forecast equal to the VaR forecast (the conventions of RV-2102) as stated, and SHALL refuse a NaN, an infinite value, a `gamma` outside (0, 1), an `alpha_level` outside (0, 1), a non-positive or non-integer `window`, and an unknown `method` or `solver`. | `tests/test_sequential_e_backtest.py::test_edge_cases`, `tests/test_sequential_e_backtest.py::test_refusals` |
| RV-2114 | THE result `EBacktestResult` SHALL be a frozen `EProcessResult` with the e-statistic and the betting process per day (read-only arrays) and the wealth of GREE and GREL under GREM, carry its parameters in `extras`, be deterministic (the same inputs give the same `inputs_hash` and path), map to a `TestResult` of the registered ids `sequential.e_backtest_var` and `sequential.e_backtest_es` (red when the monitor rejected, green otherwise, `not_evaluated` for an empty series), and the new names SHALL be public by `__all__` and listed in `docs/api-policy.md`. | `tests/test_sequential_e_backtest.py::test_the_record_and_the_test_result`, `tests/test_public_api.py::test_the_page_is_what_the_generator_writes` |
| RV-2115 | THE page `docs/sequential/02_e-backtest.md` SHALL carry the display equations of the e-statistics, the e-process and the betting processes with the paper's equation numbers, every choice of this unit's header, the assumptions and what is refused, and be linked from the sequential index; the registered ids SHALL be named on a user-facing page. | `tests/test_docs.py::test_every_registered_test_is_on_a_user_facing_page`, `tests/test_sequential_e_backtest.py::test_the_page_carries_the_equations_and_the_choices` |
