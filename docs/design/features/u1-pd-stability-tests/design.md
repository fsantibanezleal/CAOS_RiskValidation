# U1 PD, discrimination, rating-system and stability tests, design

## Modules

| Module | Tests | Shared helpers |
|---|---|---|
| `validation/calibration.py` | `pd.jeffreys` (+ `pd_jeffreys_grades`), `pd.binomial`, `pd.binomial_vasicek`, `pd.chi2_grades`, `pd.hosmer_lemeshow`, `pd.spiegelhalter`, `pd.normal_multiperiod`, `pd.traffic_lights`, `pd.brier`, `pd.ece` | `default_count_sf` (binomial and Vasicek mixture tail), `critical_default_count` |
| `validation/discrimination.py` | `disc.auc`, `disc.auc_vs_initial`, `disc.delong`, `disc.ks` | `structural_components` (O(n log n) placements), `auc_with_variance`, `cap_accuracy_ratio`, `bootstrap_ci` |
| `validation/rating_system.py` | `rating.mwb`, `rating.migration_ztests`, `rating.hhi` | `matrix_weighted_bandwidth`, `concentration`, Holm adjustment |
| `validation/stability.py` | `stability.psi`, `stability.csi`, `stability.chi2`, `stability.ks`, `stability.wasserstein` | `quantile_edges`, `bin_counts`, `psi_benchmark` |
| `validation/_common.py` | | input validation (binary targets, PDs in (0, 1), integer counts) |

## Decisions

- **Orientation.** Risk inputs are "higher is riskier"; the ECB states AUC with "higher is better" ratings, and
  the value is identical under the reversed orientation.
- **Vasicek mixture.** Gauss-Hermite quadrature with 256 probabilists' nodes; reproduces the WP14 table exactly.
- **Critical count.** $k^* = \min\{k : P(D \ge k) \le 1 - q\}$, rejection at $D \ge k^*$: the definition that
  reproduces WP14's published counts (the 99% quantile is one lower).
- **Undefined statistics** (zero variance, uniform grades) return `not_evaluated` with a note, never a number that
  looks like evidence.
- **Multiple cells** (migration z-tests): every cell reported; Holm-adjusted minimum as the family p-value, labelled
  as the package's addition.
- **PSI empty bins**: floored at half an observation of their own sample; counted in the result.
- **Thresholds**: every p-value test gets 5% and 1% in the default policy with the source of that convention;
  descriptive measures stay unthresholded and are listed as such in the policy file.

## Data flow

Arrays in, `TestResult` out, through `make_result` (U0). Per-grade wrappers return lists. `run_suite` composes any
of them under one policy.
