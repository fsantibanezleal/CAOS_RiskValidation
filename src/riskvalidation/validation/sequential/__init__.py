"""Sequential, anytime-valid monitors of calibration and back-tests (BS-15): betting e-processes.

A fixed-window test (Kupiec, Hosmer-Lemeshow, the Basel traffic light) is read once per window; read every day it
loses its size. An e-process is a non-negative process with expected value at most 1 under the null; rejecting when it
reaches `1 / alpha_level` keeps the size at `alpha_level` at any stopping time (Ville's inequality).

- `var_hits.var_hit_eprocess`, `var_hits.var_hit_clustering_eprocess`: VaR exceedances against `alpha`.
- `pd_grades.pd_grade_eprocess`: defaults against the predicted PD of each grade, by step.
- `battery.eprocess_battery`: several monitors as one verdict.
- `e_backtest`: the e-backtests of VaR and of expected shortfall of Wang, Wang and Ziegel (2022): the backtest
  e-statistics, the betting processes GREE, GREL and GREM and the e-process, one-sided (under-forecasts).
- `monitor`: the record `EProcessResult`, `GradeEProcessResult`, `detection_time` and `anytime_p_value`.

Plain functions: they read no registry and no policy, state no compliance claim and register no test. Dimensionless
log-scale statistics; times are 1-based counts of the caller's steps.
"""

from riskvalidation.validation.sequential import battery, e_backtest, monitor, pd_grades, var_hits
from riskvalidation.validation.sequential.battery import eprocess_battery
from riskvalidation.validation.sequential.e_backtest import (
    EBacktestResult,
    betting_process,
    e_backtest_es,
    e_backtest_var,
    es_backtest_e_statistic,
    var_backtest_e_statistic,
)
from riskvalidation.validation.sequential.monitor import (
    EProcessResult,
    GradeEProcessResult,
    anytime_p_value,
    detection_time,
)
from riskvalidation.validation.sequential.pd_grades import pd_grade_eprocess
from riskvalidation.validation.sequential.var_hits import var_hit_clustering_eprocess, var_hit_eprocess

__all__ = [
    "EBacktestResult",
    "EProcessResult",
    "GradeEProcessResult",
    "anytime_p_value",
    "battery",
    "betting_process",
    "detection_time",
    "e_backtest",
    "e_backtest_es",
    "e_backtest_var",
    "eprocess_battery",
    "es_backtest_e_statistic",
    "monitor",
    "pd_grade_eprocess",
    "pd_grades",
    "var_backtest_e_statistic",
    "var_hit_clustering_eprocess",
    "var_hit_eprocess",
    "var_hits",
]
