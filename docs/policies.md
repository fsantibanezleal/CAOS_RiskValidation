# Packaged policies

A policy turns a statistic into a traffic light. It lives in a versioned JSON file shipped with the package, is
shown to users as "policy" and travels with every result as `policy_version`. The thresholds are policy choices:
conventions and internal gates, never regulatory thresholds (ECB 2019 sets none). The default policy is
`policies/default.json` (`riskvalidation-default 2026.10.1`, `default_policy()`), unchanged. Since unit U16-E3 each
Contraste validation battery has a file of its own.

## The label rule

`policy_version` of a result row is the packaged label `<policy_id> <version>`, and the file is
`policies/<policy_id>-<version>.json`.

| Inline head (before U16) | Policy id | Version | Label | File |
|---|---|---|---|---|
| `contraste-market-battery-1.0.0` | `contraste-market-battery` | `1.0.0` | `contraste-market-battery 1.0.0` | `contraste-market-battery-1.0.0.json` |
| `contraste-c12-1.0.0` | `contraste-c12` | `1.0.0` | `contraste-c12 1.0.0` | `contraste-c12-1.0.0.json` |

The eleven batteries are aggregation, c12, ccr, ifrs9, irrbb, lgd-ead, liquidity, market, oprisk, portfolio and
stress. Every `_version` call site of `riskvalidation.validation.*` writes its battery label and nothing else: the
parameters left the string. A test's own arguments (the PFE `quantile`, `n_sim`, `seed`, `lags`, `compared_at`,
`rho`, `estimator`, `basis`, ...) and the numbers measured on a case (the simulated critical values of the PIT
back-test) are in the row's `extras`. The old inline form `contraste-<name>-<version>; key=value; ...` keeps
resolving in `riskvalidation.governance.policy.resolve_policy` (hyphenated families such as `lgd-ead` and the
parameterless `contraste-c12-1.0.0` included), so documents baked before U16 stay readable.

## The file format

```json
{
  "policy_id": "contraste-portfolio-battery",
  "version": "1.0.0",
  "description": "...",
  "thresholds": {
    "portfolio.euler_additivity": {
      "kind": "upper_bad", "amber": 1e-06, "red": 1e-06,
      "source": "convention: ...",
      "rule": "upper_le:metric",
      "by": "extras.basis",
      "cases": {
        "exact-allocation": {"amber": 1e-06, "red": 1e-06, "tolerance": 1e-06},
        "kernel-var": {"amber": 0.01, "red": 0.01, "tolerance": 0.01}
      }
    }
  }
}
```

| Field | Meaning |
|---|---|
| `kind` | `p_value`, `upper_bad` or `lower_bad`: the direction in which the value gets worse (the direction reported by `governance.policy` where the rule is a plain gate) |
| `amber`, `red` | the two cut points, with the semantics of `Threshold` |
| `source` | always starts `convention:`; names the Contraste producer cases, the engine function whose light rule it transcribes and the dossier reference |
| `parameters` (optional) | the cut points beyond the pair, by the engine's own argument name (for example `k_se`, `minimum`, `power_green`); a mapping of finite numbers, frozen once loaded |
| `rule` (optional) | the name of a classifier of `riskvalidation.policies.rules` for a light that is not two cut points on one value; `name` or `name:field` |
| `by`, `cases` (optional, together) | the entry is keyed by a field of the row, `segment` or `extras.<key>`; `cases` maps each field value to the `amber`, `red` and parameters that apply to it |

`riskvalidation.core.policy.packaged_policy(label)` loads a packaged policy by its label and raises `KeyError`
naming the label and the packaged labels when it is absent (`packaged_labels()` lists them).
`Policy.classify_row(test_id, row)` gives the light of a result row (a mapping or a `TestResult`) through the entry's
rule and its keyed case; a test without an entry, and a keyed entry whose field has no case for the row, give
`not_evaluated`. `Policy.classify` and `load_policy` are unchanged.

## The rules

A rule reads only the fields of the row (`statistic`, `p_value`, `metric`, `segment`, `extras`), never the inputs of
the test, so a committed document can be re-classified. Cut points come from the entry. Generic rules: `upper_le:F`
(green at or below `amber`, amber at or below `red`, red above, with the 1e-12 rounding tolerance of the test
functions), `lower_ge:F`, `abs_upper_le:F`, `red_if_positive:F`, `red_if_nonempty:F`, `always_green`. A p-value entry
with no rule is read by `Threshold.classify`. The named rules transcribe one test function each
(`aggregation_sum_vc`, `ccr_path_convergence`, `ccr_pit_simulated`, `ifrs9_lifetime_pd`, `oprisk_capital_stability`, and
so on; see the module docstrings). The engine functions keep their own comparison code; the reproduction test
(`tests/test_packaged_policies.py`) proves the rule gives the same light on all 2,450 committed Contraste rows.

Two decisions of D16-17 are visible in the files:

- A per-case measured number is data. `ccr.pit_exposure_backtest` holds no critical value: its rule reads the row's
  `extras["simulated_critical_values"]` (`p05`, `p01`) and its `effective_sample_size`.
- A keyed entry carries one parameter set per value of a row field: the Euler tolerance by `extras.basis`
  (`exact-allocation` 1e-06 for ES and aggregation allocations, `kernel-var` 0.01 for the kernel VaR contributions;
  `euler_additivity(..., basis=...)` writes it) and the desk exception limit by `segment` (`alpha=0.01` 12,
  `alpha=0.025` 30).

Two rows are partly outside what a row can state: `ifrs9.lifetime_pd_backtest` reads `extras["observed"]` (written by
the engine since U16-E3; the committed documents predate it and the reproduction test derives it from the stored Brier
score and signed gap), and `liq.htm_loss_effect` sees the survival period only as the flags
`extras["survives_horizon"]`, so a lengthening between two finite periods is not visible to its rule.

## The batteries

Amber and red are the pair `Threshold` holds; the parameters repeat every cut point of the old inline string under the
engine's argument name, so each old string maps to its entry by name.

### `contraste-aggregation-battery 1.0.0`

File `policies/contraste-aggregation-battery-1.0.0.json`. Producer cases: c18.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `aggregation.copula_choice_sensitivity` | upper_bad | `upper_le:statistic` | 0.05 | 0.15 | amber=0.05, red=0.15 |  |
| `aggregation.correlation_stress` | upper_bad | `aggregation_corr_stress` | 0.1 | 0.1 | amber=0.1, tolerance=1e-09 |  |
| `aggregation.diversification_sign` | upper_bad | `aggregation_diversification` | 1e-09 | 1e-09 | tolerance=1e-09 |  |
| `aggregation.elliptical_mc_agreement` | upper_bad | `abs_upper_le:statistic` | 3 | 3 | z=3 |  |
| `aggregation.horizon_confidence_consistency` | upper_bad | `red_if_positive:statistic` | 0 | 0 | expected_horizon=1, expected_level=0.999 |  |
| `aggregation.rosenberg_schuermann_reference` | upper_bad | `always_green` | 0.3 | 0.4 | high=0.4, low=0.3 |  |
| `aggregation.simple_sum_and_variance_covariance` | upper_bad | `aggregation_sum_vc` | 0.1 | 0.1 | amber=0.1, tolerance=1e-09 |  |
| `aggregation.worst_case_bracket` | upper_bad | `red_if_positive:extras.rows_outside` | 1e-09 | 1e-09 | tolerance=1e-09 |  |

### `contraste-c12 1.0.0`

File `policies/contraste-c12-1.0.0.json`. Producer cases: c12/reconcile.py.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `ccr.ore_cva_agreement` | upper_bad | `upper_le:statistic` | 2 | 2 | k_se=2 |  |
| `ccr.ore_dim_agreement` | upper_bad | `upper_le:statistic` | 2 | 2 | k_se=2 |  |
| `ccr.ore_eepe_agreement` | upper_bad | `upper_le:statistic` | 2 | 2 | k_se=2 |  |
| `ccr.quantlib_bond_call` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | rel_tol=1e-06 |  |
| `ccr.quantlib_bond_put` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | rel_tol=1e-06 |  |
| `ccr.quantlib_swaption_payer` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | rel_tol=1e-06 |  |
| `ccr.quantlib_swaption_receiver` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | rel_tol=1e-06 |  |

### `contraste-ccr-battery 1.0.0`

File `policies/contraste-ccr-battery-1.0.0.json`. Producer cases: c12.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `ccr.cva_recomputed` | upper_bad | `upper_le:metric` | 1e-09 | 1e-09 | rel_tol=1e-09 |  |
| `ccr.ee_analytic_vs_simulated` | upper_bad | `upper_le:statistic` | 3 | 3 | k_se=3 |  |
| `ccr.eepe_recomputed` | upper_bad | `upper_le:metric` | 1e-09 | 1e-09 | rel_tol=1e-09 |  |
| `ccr.martingale_discounted_bonds` | upper_bad | `upper_le:statistic` | 3 | 3 | k_se=3 |  |
| `ccr.path_convergence` | upper_bad | `ccr_path_convergence` | 3 | 6 | k_se=3 |  |
| `ccr.pfe_exceedance_backtest` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  |
| `ccr.pit_exposure_backtest` | upper_bad | `ccr_pit_simulated` | 0 | 0 |  |  |
| `ccr.reconcile_ee_profile` | upper_bad | `upper_le:statistic` | 2 | 2 | k_se=2 |  |

### `contraste-ifrs9-battery 1.0.0`

File `policies/contraste-ifrs9-battery-1.0.0.json`. Producer cases: c06.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `ifrs9.ecl_backtest` | upper_bad | `ifrs9_ecl_backtest` | 1.25 | 2 | r_amber_high=2, r_amber_low=0.5, r_green_high=1.25, r_green_low=0.8 |  |
| `ifrs9.lifetime_pd_backtest` | upper_bad | `ifrs9_lifetime_pd` | 0.25 | 0.5 | alpha_amber=0.05, alpha_red=0.01, gap_amber=0.25, gap_red=0.5 |  |
| `ifrs9.nonlinearity` | upper_bad | `ifrs9_nonlinearity` | 1e-09 | 1e-09 | gauss_hermite_tolerance=1e-09 |  |
| `ifrs9.overlay_diagnostics` | upper_bad | `ifrs9_overlay` | 0.1 | 0.25 | share_amber=0.1, share_red=0.25 |  |
| `ifrs9.reconciliation_identities` | upper_bad | `red_if_nonempty:extras.failing_identities` | 1e-06 | 1e-06 | tolerance=1e-06 |  |
| `ifrs9.scenario_sensitivity` | upper_bad | `ifrs9_scenario_sensitivity` | 1 | 3 | nl_amber=1, nl_red=3 |  |
| `ifrs9.sicr_effectiveness` | lower_bad | `ifrs9_sicr_effectiveness` | 2 | 1 | amber_ratio=1, direct_share_amber=0.5, green_ratio=2 |  |
| `ifrs9.sicr_threshold_stability` | upper_bad | `ifrs9_sicr_stability` | 1 | 1 | amber_count=1, red_share_of_periods=0.5 |  |
| `ifrs9.stage_migration_backtest` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  |

### `contraste-irrbb-battery 1.0.0`

File `policies/contraste-irrbb-battery-1.0.0.json`. Producer cases: c14, c15.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `irrbb.behavioural_change_at_breach` | upper_bad | `red_if_positive:statistic` | 0 | 0 |  |  |
| `irrbb.cmf_annex1_goldens` | upper_bad | `red_if_nonempty:extras.failed` | 1e-12 | 1e-12 | tolerance=1e-12 |  |
| `irrbb.eve_sum_of_parts` | upper_bad | `upper_le:metric` | 1e-09 | 1e-09 | rel_tol=1e-09 |  |
| `irrbb.eve_with_without_behavioural` | upper_bad | `irrbb_eve_with_without` | 0.05 | 0.15 | diff_amber=0.05, diff_red=0.15 |  |
| `irrbb.nii_out_of_sample` | upper_bad | `upper_le:statistic` | 0.1 | 0.25 | err_amber=0.25, err_green=0.1 |  |
| `irrbb.nmd_parameter_stability` | upper_bad | `upper_le:statistic` | 0.1 | 0.25 | gap_amber=0.1, gap_red=0.25 |  |
| `irrbb.srp31_92_goldens` | upper_bad | `irrbb_srp31_92` | 0.0005 | 0.05 | tolerance_bp=0.05, tolerance_scalar=0.0005 |  |

### `contraste-lgd-ead-battery 1.0.0`

File `policies/contraste-lgd-ead-battery-1.0.0.json`. Producer cases: c08, c09.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `lgd_ead.ccf_estimator_bias` | upper_bad | `abs_upper_le:metric` | 0.02 | 0.04 | alpha_amber=0.05, alpha_red=0.01, tolerance=0.02 |  |
| `lgd_ead.downturn_coverage` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  |
| `lgd_ead.ead_backtest_by_band` | upper_bad | `upper_le:metric` | 0.1 | 0.25 | gap_amber=0.1, gap_red=0.25 |  |
| `lgd_ead.lgd_calibration_by_segment` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  |
| `lgd_ead.lgd_calibration_over_time` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  |
| `lgd_ead.lgd_discrimination` | lower_bad | `lower_ge:statistic` | 0.3 | 0.1 | rho_amber=0.1, rho_green=0.3 |  |

### `contraste-liquidity-battery 1.0.0`

File `policies/contraste-liquidity-battery-1.0.0.json`. Producer cases: c16.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `liq.cbc_haircut_rules` | upper_bad | `liquidity_haircut` | 1e-12 | 1e-12 | tolerance=1e-12 |  |
| `liq.cliff_effect_diagnostic` | lower_bad | `lower_ge:statistic` | -0.01 | -0.03 | amber=-0.01, red=-0.03 |  |
| `liq.htm_loss_effect` | upper_bad | `liquidity_htm_loss` | 0 | 0 | tolerance=0 |  |
| `liq.htm_revaluation_vs_reported` | upper_bad | `upper_le:statistic` | 0.05 | 0.15 | amber=0.15, green=0.05 |  |
| `liq.list_compounding_golden` | upper_bad | `upper_le:statistic` | 5e-07 | 5e-07 | tolerance=5e-07 |  |
| `liq.nlp_identity` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | tolerance=1e-06 |  |
| `liq.reverse_stress_round_trip` | upper_bad | `upper_le:statistic` | 0 | 0 | tolerance_days=0 |  |
| `liq.runoff_adequacy_back_test` | lower_bad | `lower_ge:statistic` | 0.8 | 0.5 | ratio_amber=0.5, ratio_green=0.8 |  |
| `liq.runoff_vs_generator_truth` | upper_bad | `upper_le:statistic` | 2 | 3 | se_green=2, se_red=3 |  |
| `liq.survival_monotone_in_severity` | upper_bad | `upper_le:statistic` | 0 | 0 | tolerance_days=0 |  |
| `liq.uninsured_share_sensitivity` | upper_bad | `upper_le:statistic` | 0 | 0 | tolerance_days=0 |  |

### `contraste-market-battery 1.0.0`

File `policies/contraste-market-battery-1.0.0.json`. Producer cases: c11, c13.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `market.acerbi_szekely` | p_value | two cut points | 0.05 | 0.0001 | alpha_amber=0.05, alpha_red=0.0001 |  |
| `market.christoffersen_conditional_coverage` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  |
| `market.christoffersen_independence` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  |
| `market.desk_backtest` | upper_bad | `upper_le:statistic` | 12 | 12 |  | `segment`: `alpha=0.01` -> limit=12; `alpha=0.025` -> limit=30 |
| `market.duration_test` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  |
| `market.dynamic_quantile_test` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  |
| `market.kupiec_pof` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  |
| `market.pla_test` | lower_bad | `market_pla_zone` | 0.8 | 0.7 | ks_green=0.09, ks_red=0.12, spearman_green=0.8, spearman_red=0.7 |  |

### `contraste-oprisk-battery 1.0.0`

File `policies/contraste-oprisk-battery-1.0.0.json`. Producer cases: c17.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `oprisk.benchmark_reproduction` | upper_bad | `upper_le:metric` | 1 | 2 | moment_rel_tol=1e-05, quantile_rel_tol=0.01 |  |
| `oprisk.capital_stability` | upper_bad | `oprisk_capital_stability` | 0.25 | 0.5 | bias_amber=0.25, bias_green=0.1, dispersion_amber=0.5, dispersion_green=0.25 |  |
| `oprisk.scenario_influence` | upper_bad | `upper_le:metric` | 0.1 | 0.25 | influence_amber=0.1, influence_red=0.25 |  |
| `oprisk.sma_ilm_response` | upper_bad | `upper_le:metric` | 0.05 | 0.15 | jump_amber=0.15, jump_green=0.05 |  |
| `oprisk.threshold_stability` | upper_bad | `upper_le:metric` | 0.1 | 0.25 | tol_amber=0.25, tol_green=0.1 |  |

### `contraste-portfolio-battery 1.0.0`

File `policies/contraste-portfolio-battery-1.0.0.json`. Producer cases: c10.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `portfolio.correlation_size_power` | p_value | `portfolio_size_power` | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01, power_amber=0.5, power_green=0.8 |  |
| `portfolio.crp_recursion_vs_mc` | p_value | `p_value_negative_red` | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  |
| `portfolio.euler_additivity` | upper_bad | `upper_le:metric` | 1e-06 | 1e-06 |  | `extras.basis`: `exact-allocation` -> tolerance=1e-06; `kernel-var` -> tolerance=0.01 |
| `portfolio.mc_convergence` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  |
| `portfolio.tail_backtest` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  |

### `contraste-stress-battery 1.0.0`

File `policies/contraste-stress-battery-1.0.0.json`. Producer cases: c18.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by |
|---|---|---|---|---|---|---|
| `stress.reverse_solution_check` | upper_bad | `stress_reverse_solution` | 1e-06 | 1e-06 | tolerance=1e-06 |  |
| `stress.satellite_backcast` | upper_bad | `stress_satellite` | 0.35 | 0.6 | peak_amber=0.35, peak_red=0.6, timing_quarters=2 |  |
| `stress.scenario_severity` | lower_bad | `stress_severity` | 0.75 | 0.75 | minimum=0.75 |  |
| `stress.sensitivity_monotonicity` | upper_bad | `red_if_positive:statistic` | 0 | 0 |  |  |
| `stress.static_balance_sheet_rules` | upper_bad | `red_if_positive:statistic` | 0.006 | 0.006 | tolerance=0.006 |  |
