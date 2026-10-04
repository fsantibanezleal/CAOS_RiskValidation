# U1 PD, discrimination, rating-system and stability tests, requirements

Sources: research dossier 06 (sections C.1 to C.4) of the Contraste initiative, transcribed into
`docs/tests/`. Every test returns a `TestResult` whose `reference` names the primary source.

| ID | Requirement | Gate |
|---|---|---|
| RV-101 | THE Jeffreys test SHALL return the CDF of Beta(D + 1/2, N - D + 1/2) at the PD applied, per grade and for the portfolio. | `tests/test_pd_calibration.py::test_jeffreys_matches_beta_cdf` |
| RV-102 | THE binomial and Vasicek-corrected binomial tests SHALL reproduce the BCBS WP14 critical default counts 19, 35, 49, 63 and 77 for PD 1%, N 1,000, confidence 99% and asset correlation 0%, 5%, 10%, 15%, 20%. | `tests/test_pd_calibration.py::test_wp14_critical_values` |
| RV-103 | THE PSI test SHALL report the Yurdakul-Naranjo chi-square benchmark, reproducing 0.338, 0.169, 0.085 and 0.034 for ten bins at 5% with equal samples of 100, 200, 400 and 1,000, beside the conventional bands. | `tests/test_stability.py::test_psi_chi2_benchmark` |
| RV-104 | THE AUC SHALL equal the pairwise Mann-Whitney count with half credit for ties, its variance SHALL follow the ECB structural-component formula, and the accuracy ratio from the CAP SHALL equal 2 AUC - 1 under consistent tie handling. | `tests/test_discrimination.py::test_auc_matches_pairwise_and_cap` |
| RV-105 | THE comparison against the development AUC SHALL compute S = (AUC_init - AUC_curr) / s with p-value 1 - Phi(S). | `tests/test_discrimination.py::test_auc_vs_initial_ecb_statistic` |
| RV-106 | THE DeLong paired test SHALL equal a brute-force computation of the structural components, and IF its variance is not positive, THEN THE result SHALL be not evaluated. | `tests/test_discrimination.py::test_delong_matches_bruteforce` |
| RV-107 | THE chi-square over grades SHALL use as many degrees of freedom as grades, and the Spiegelhalter, Hosmer-Lemeshow and multi-period normal tests SHALL follow their documented formulas. | `tests/test_pd_calibration.py::test_chi2_spiegelhalter_hl_normal_formulas` |
| RV-108 | THE traffic-lights test SHALL compute the probability of a colour profile no better than the observed one by exact multinomial enumeration. | `tests/test_pd_calibration.py::test_traffic_lights_multinomial` |
| RV-109 | THE Brier score SHALL equal reliability minus resolution plus uncertainty exactly when forecasts are constant within groups, and the ECE SHALL report its bins. | `tests/test_pd_calibration.py::test_brier_murphy_identity` |
| RV-110 | THE matrix weighted bandwidth and the migration z-tests SHALL follow the ECB formulas, with a Holm-adjusted family-wise p-value across cells. | `tests/test_rating_system.py::test_mwb_and_migration_ztests` |
| RV-111 | THE Herfindahl test SHALL give 0 for a uniform grade distribution, 1 for a single grade, and the ECB p-value for a change in the coefficient of variation. | `tests/test_rating_system.py::test_hhi_cv_ecb` |
| RV-112 | THE KS, chi-square homogeneity and Wasserstein shift measures SHALL equal their SciPy reference computations. | `tests/test_stability.py::test_shift_tests_match_scipy` |
| RV-113 | THE catalogue SHALL register every U1 test with a reference and a null hypothesis, and the default policy SHALL give each p-value test thresholds with their source while descriptive measures stay unthresholded. | `tests/test_catalogue.py::test_u1_catalogue_complete` |
| RV-114 | THE stratified bootstrap interval SHALL be identical for the same seed and SHALL resample defaulters and non-defaulters separately. | `tests/test_discrimination.py::test_bootstrap_ci_seeded` |
