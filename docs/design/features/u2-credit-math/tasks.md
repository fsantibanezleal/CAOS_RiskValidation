# U2 credit risk mathematics, tasks

| # | Task | Requirement | State |
|---|---|---|---|
| 1 | The CRE99 Table 1 fixture, verified cell by cell by recomputation | RV-201 | done (`tests/fixtures/cre99_table1.json`) |
| 2 | `regulatory/irb.py`: the four asset-class functions, the regimes (read-only, serialisable), floors, adjustments, defaulted exposures | RV-201, RV-202, RV-203 | done |
| 3 | Expected loss against eligible provisions | RV-204 | done |
| 4 | `engines/vasicek.py`: conditional PD, the LHP distribution, quantile and density, the implied systematic factor | RV-205 | done |
| 5 | `engines/low_default.py`: most prudent estimation, independent and correlated | RV-206 | done |
| 6 | `engines/pd_curve.py`: the model identities, quasi moment matching, cases 1 to 3, improper models refused, the profile chi-square test | RV-207 to RV-210, RV-213 | done |
| 7 | `pd.default_profile`, the Monte Carlo chi-square test of the default profile, with its policy threshold | RV-212 | done |
| 8 | `docs/regulatory/01_irb.md`, `docs/engines/01_vasicek-lhp.md`, `02_low-default.md`, `03_pd-curve-calibration.md`, the index pages, four computed and two hand-authored figures (theme-aware) | all | done |
| 9 | `examples/credit_math.py`, catalogue entries, CHANGELOG, version 0.02.000 | RV-211, RV-214 | done |

## Convergence verdict (2026-10-05)

Converged. Every requirement's gate exists and passes: 39 tests, lint clean, every guard. The golden values are the
primary sources' own tables, recomputed from their inputs: CRE99 Table 1 (144 cells, 141 exact), Pluto and Tasche
Tables 1 to 7 (one misprint found: Table 4 at 75%), Tasche (2013) Tables 2 to 9 (Table 5: 15 of 17 exact, the rest one
unit off; Table 7's 136 forecasts and 8 p-values; Table 8's 8 forecasts and 4 p-values; Table 9 up to the paper's solver
offset). Found and fixed on the way: the QRRE floor applies to revolvers only (CRE99 tabulates QRRE at 0.05%); a shared
regime table could be mutated through a result, now read-only. Not reproducible from the paper alone: the Moody's
backtest of Tasche (2013) section 5 (needs Moody's yearly grade frequencies).
