# U20 E-backtests of VaR and expected shortfall, tasks

| # | Task | Requirement | State |
|---|---|---|---|
| 1 | Read the paper itself (arXiv 2209.00991v7) and write the requirements, one per behaviour, each naming its test | all | done |
| 2 | `tests/test_sequential_e_backtest.py`, red before the module exists | RV-2101 to RV-2115 | done (the first commit of the unit imports a module that does not exist) |
| 3 | `sequential/e_backtest.py`: the e-statistics, GREE, GREL, GREM, both solvers, the window, the record, the registration; the exports | RV-2101 to RV-2114 | done |
| 4 | The page `docs/sequential/02_e-backtest.md`, the sequential index, `docs/api-policy.md` (generated), the SDD row, the CHANGELOG `[Unreleased]` entry | RV-2114, RV-2115 | done |
| 5 | The release gate: `pytest -m sizepower tests/test_sequential_e_backtest.py` (size at 4,000 series of 1,000 days and of 250 days with moving forecasts; the expectation by Monte Carlo at 20,000,000 draws) | RV-2103, RV-2111 | run on the unit's branch; the release repeats it |
| 6 | The version bump, the release notes and the notebook that teaches the method (Contraste ladder row M-39) | none | out of this unit (the release bumps the version; the notebook is Contraste's) |

## Convergence verdict (2026-11-16)

Converged. Every requirement's gate exists and passes; the module is the paper's definitions, each equation cited in its
docstring and on its page, and the VaR case is equal to the Contraste manuscript's reference to 1e-12.
