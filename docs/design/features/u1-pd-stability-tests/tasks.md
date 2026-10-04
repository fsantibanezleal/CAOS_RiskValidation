# U1 PD, discrimination, rating-system and stability tests, tasks

| # | Task | Requirement |
|---|---|---|
| 1 | Jeffreys per grade and portfolio | RV-101 |
| 2 | binomial and Vasicek mixture tail, critical counts | RV-102 |
| 3 | chi-square over grades, Hosmer-Lemeshow, Spiegelhalter, multi-period normal | RV-107 |
| 4 | traffic lights by exact multinomial enumeration | RV-108 |
| 5 | Brier with Murphy decomposition, ECE | RV-109 |
| 6 | placements, AUC with ECB variance, CAP accuracy ratio | RV-104 |
| 7 | AUC against development, DeLong | RV-105, RV-106 |
| 8 | stratified bootstrap | RV-114 |
| 9 | MWB, migration z-tests with Holm, Herfindahl test | RV-110, RV-111 |
| 10 | PSI and CSI with chi-square benchmark, chi-square, KS, Wasserstein | RV-103, RV-112 |
| 11 | registry entries and default-policy thresholds with sources | RV-113 |
| 12 | docs: catalogue, four family pages, five figures; runnable quickstart | all |

## Convergence verdict (2026-10-04)

| Requirement | Gate | Result |
|---|---|---|
| RV-101 | `tests/test_pd_calibration.py::test_jeffreys_matches_beta_cdf` | pass |
| RV-102 | `tests/test_pd_calibration.py::test_wp14_critical_values` | pass (19, 35, 49, 63, 77 reproduced) |
| RV-103 | `tests/test_stability.py::test_psi_chi2_benchmark` | pass (0.338, 0.169, 0.085, 0.034 reproduced) |
| RV-104 | `tests/test_discrimination.py::test_auc_matches_pairwise_and_cap` | pass (AR = 2 AUC - 1 to 1e-12 with ties) |
| RV-105 | `tests/test_discrimination.py::test_auc_vs_initial_ecb_statistic` | pass |
| RV-106 | `tests/test_discrimination.py::test_delong_matches_bruteforce` | pass |
| RV-107 | `tests/test_pd_calibration.py::test_chi2_spiegelhalter_hl_normal_formulas` | pass |
| RV-108 | `tests/test_pd_calibration.py::test_traffic_lights_multinomial` | pass (equals 4^T sequence enumeration) |
| RV-109 | `tests/test_pd_calibration.py::test_brier_murphy_identity` | pass |
| RV-110 | `tests/test_rating_system.py::test_mwb_and_migration_ztests` | pass |
| RV-111 | `tests/test_rating_system.py::test_hhi_cv_ecb` | pass |
| RV-112 | `tests/test_stability.py::test_shift_tests_match_scipy` | pass |
| RV-113 | `tests/test_catalogue.py::test_u1_catalogue_complete` | pass |
| RV-114 | `tests/test_discrimination.py::test_bootstrap_ci_seeded` | pass |

Also run: ruff (clean), the three guards (OK), the wheel build, and `examples/quickstart.py`
(`tests/test_examples.py::test_quickstart_runs`). Converged.
