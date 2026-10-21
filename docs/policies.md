# Packaged policies

A policy turns a statistic into a traffic light. It lives in a versioned JSON file shipped with the package, is
shown to users as "policy" and travels with every result as `policy_version`. The thresholds are policy choices:
conventions and internal gates (ECB 2019 sets none), except the values the Basel Framework sets, whose `source` says
`regulatory:` and names the paragraph. The default policy is `policies/default.json` (`riskvalidation-default
2026.10.1`, `default_policy()`), unchanged. Since unit U16-E3 each Contraste validation battery has a file of its own.

This page is enough to read a policy file and classify a result row without the engine's code: the format, the rules
and every entry are stated here, and a JSON Schema (`riskvalidation/policies/policy.schema.json`, shipped in the
wheel) validates the files.

## The label rule

`policy_version` of a result row is the packaged label `<policy_id> <version>`, and the file is
`policies/<policy_id>-<version>.json`.

| Inline head (before U16) | Policy id | Version | Label | File |
|---|---|---|---|---|
| `contraste-market-battery-1.0.0` | `contraste-market-battery` | `1.0.0` | `contraste-market-battery 1.0.0` | `contraste-market-battery-1.0.0.json` |
| `contraste-c12-1.0.0` | `contraste-c12` | `1.0.0` | `contraste-c12 1.0.0` | `contraste-c12-1.0.0.json` |

The eleven batteries are aggregation, c12, ccr, ifrs9, irrbb, lgd-ead, liquidity, market, oprisk, portfolio and
stress. Every call site of `riskvalidation.validation.*` that writes a battery label (77 of them: the code calls
`battery_label(POLICY_LABEL, <test id>, <cut points>)`) writes the label and nothing else: the parameters left the string.
A test's own arguments (the PFE `quantile`, `n_sim`, `seed`, `lags`, `compared_at`, `rho`, `estimator`, `basis`, ...)
and the numbers measured on a case (the simulated critical values of the PIT back-test) are in the row's `extras`. The
old inline form `contraste-<name>-<version>; key=value; ...` keeps resolving in
`riskvalidation.governance.policy.resolve_policy` (hyphenated families such as `lgd-ead` and the parameterless
`contraste-c12-1.0.0` included), so documents baked before U16 stay readable.

**A row computed with cut points that differ from its packaged entry** carries the packaged label plus `+custom`
(`contraste-market-battery 1.0.0+custom`), and the cut points it used are in `extras["policy_override"]`
(`{"label", "test_id", "kind", "parameters"}`: the entry's parameters with the caller's values). A caller that passes the
default values gets the bare label. `resolve_policy(label, test_id, override=...)` resolves it with `source`
`packaged-policy-overridden` and the override's parameters; without the override it returns the reason code
`policy.override_missing`. `ml_policy(**parameters)` does the same for the `ml.` rows: its label is
`riskvalidation-default 2026.10.1+custom` and its `override` holds the rows that differ from the packaged default.

## The file format

```json
{
  "format_version": 1,
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
| `format_version` (optional, 1) | the version of this format; an engine refuses a file whose version it does not read, so an older reader never misclassifies a newer file |
| `kind` | `p_value`, `upper_bad` or `lower_bad`: the direction in which the value gets worse (the direction reported by `governance.policy` where the rule is a plain gate) |
| `amber`, `red` | the two cut points; their boundary meaning is the one of the entry's `rule` (below), or of `Threshold` when it names none |
| `source` | starts `convention:` (a Contraste internal gate, not a regulatory threshold) or `regulatory:` (a value the Basel Framework sets, with the paragraph); then the engine function whose light rule it transcribes and the dossier reference |
| `parameters` (optional) | the cut points beyond the pair, by the engine's own argument name (for example `k_se`, `minimum`, `power_green`); finite numbers. No parameter reuses the name `amber` or `red` for a different value than the pair's |
| `rule` (optional) | the name of a classifier of `riskvalidation.policies.rules`, `name` or `name:field`; a name the engine does not know is refused when the file is loaded |
| `by`, `cases` (optional, together) | the entry is keyed by a field of the row, `segment` or `extras.<key>`; `cases` maps each field value to the `amber`, `red` and parameters that apply to it |

`riskvalidation.core.policy.packaged_policy(label)` loads a packaged policy by its label and raises `KeyError` naming the
label and the packaged labels when it is absent (`packaged_labels()` lists them). A value that is not a number in a file
raises `ValueError` naming the policy and the entry.

## Classifying a row

- `Policy.classify_row(test_id, row)` gives the light of a result row (a mapping or a `TestResult`) through the entry's
  `rule` and its keyed case. A test without an entry, and a keyed entry whose field has no case for the row, give
  `not_evaluated`.
- `Policy.classify(test_id, p_value=..., metric=..., row=...)` reads the two cut points on one value, which is the whole
  classification only for an entry that names no rule and is not keyed. For an entry that names a rule or is keyed it
  needs the row (`row=`) and then gives the light `classify_row` gives; without the row it raises `ValueError` naming
  `classify_row`. It never returns a light the rule would not give.
- **Two boundary meanings of `amber`.** Without a rule, an `upper_bad` entry is amber from `amber` (`m >= amber`), as
  `Threshold` states. Under the rule `upper_le`, a value equal to `amber` is green (green when `m <= amber`, with a 1e-12
  tolerance). The rule names the comparison: read the rules table for the entry's rule.

## The rules

A rule reads only the fields of the row (`statistic`, `p_value`, `metric`, `segment`, `extras`), never the inputs of the
test, so a committed document can be re-classified. Cut points come from the entry (`amber`, `red` and the parameters);
`F` is the field after the colon of `name:F`. A missing field the rule needs gives `not_evaluated`.

<!-- rules:begin -->

| Rule | Reads | Cut points | The light |
|---|---|---|---|
| `abs_upper_le:F` | the field F | `amber`, `red` | `upper_le` applied to the absolute value of F. |
| `aggregation_corr_stress` | `statistic` (the change of capital) | `tolerance`, `amber` | red when the statistic is below -`tolerance`; amber when above `amber`; green otherwise. |
| `aggregation_diversification` | `statistic` (the largest factor) | `tolerance` | red when the statistic is above 1 + `tolerance`, green otherwise. |
| `aggregation_sum_vc` | `metric` (the ratio of the sum), `statistic` (the variance-covariance ratio) | `tolerance`, `amber` | red when the ratio is above 1 + `tolerance`; amber when |statistic - 1| is above `amber`; green otherwise. |
| `always_green` | nothing | none | green: a reference row shown for context, never a gate. |
| `always_not_evaluated` | nothing | none | not evaluated: an estimate, never a colour. |
| `amber_if_false:F` | the field F (a boolean) | none | amber when F is false, green when true; never red. |
| `amber_if_nonempty:F` | the field F (a list of flagged groups) | none | amber when the list is not empty, green when empty; never red. |
| `ccr_path_convergence` | `statistic` (EE change), `metric` (CVA change) | `k_se` | the larger of the two is green at most `k_se`, amber at most 2 `k_se` (1e-12 tolerance), red above. |
| `ccr_pit_simulated` | `statistic` (the Anderson-Darling W2), `extras.simulated_critical_values` (`p05`, `p01`) | none (the critical values are data of the case) | green when W2 is at most `p05`, amber when at most `p01`, red above (1e-12 tolerance). |
| `fair_interval_excludes_zero` | `extras.effect.lo`, `extras.effect.hi` | none | amber when the interval excludes zero, green otherwise; never red. |
| `ifrs9_ecl_backtest` | `metric` (R_H) | `r_green_low`, `r_green_high`, `r_amber_low`, `r_amber_high` | green when R_H is inside the green range, amber inside the amber range, red outside (ranges closed). |
| `ifrs9_lifetime_pd` | `extras.observed` with the band (`band_low`, `band_high`, `band_low_red`, `band_high_red`), or `extras.relative_gap` | `gap_amber`, `gap_red` (the band alphas are in the row) | with a band: red when an observed rate is outside the red band, amber outside the amber band, green inside; without one: the worst relative gap is green below `gap_amber`, amber below `gap_red`, red otherwise (strict comparisons). |
| `ifrs9_nonlinearity` | `metric`, `extras.gauss_hermite_error` | `gauss_hermite_tolerance` | green when the Gauss-Hermite error is at most the tolerance and the metric is at least 0, red otherwise. |
| `ifrs9_overlay` | `metric` (the overlay share), `extras.direction_consistent` | `share_amber`, `share_red` | red above `share_red`; amber above `share_amber` or when the direction is inconsistent; green otherwise. |
| `ifrs9_scenario_sensitivity` | `metric` (the nonlinearity index) | `nl_amber`, `nl_red` | red above `nl_red`, amber above `nl_amber` or below 0, green otherwise. |
| `ifrs9_sicr_effectiveness` | `metric` (default-rate ratio), `extras.share_stage3_direct_from_stage1` | `green_ratio`, `amber_ratio`, `direct_share_amber` | green when the ratio is at least `green_ratio`, amber when at least `amber_ratio`, red below; a green becomes amber when the direct Stage 3 share is above `direct_share_amber`. |
| `ifrs9_sicr_stability` | `extras.loosening_periods`, `extras.periods` | `red_share_of_periods` | green with no loosening period; red when their count is at least `periods` x `red_share_of_periods`; amber otherwise; not evaluated with fewer than two periods. |
| `irrbb_eve_with_without` | `statistic`, `extras.hidden_breach` | `diff_amber`, `diff_red` | red when the behavioural assumptions hide a breach; otherwise `upper_le` on |statistic| with `diff_amber` and `diff_red`. |
| `irrbb_srp31_92` | `extras.deviations` (by name) | `tolerance_scalar` (for `alpha_short`), `tolerance_bp` (for the shocks) | green when every deviation is at most its tolerance (1e-12 tolerance), red otherwise. |
| `liquidity_haircut` | `statistic`, `extras.wrong_labels` | `tolerance` | green when the largest deviation is at most `tolerance` and no label is wrong; red otherwise. |
| `liquidity_htm_loss` | `statistic`, `extras.survives_horizon` | none | green when the unrealised loss does not raise the counterbalancing capacity (statistic at least 0) and does not make a surviving scenario survive the horizon; red otherwise. |
| `lower_ge:F` | the field F | `amber`, `red` | green when F is at least `amber`, amber when at least `red` (1e-12 tolerance), red below. |
| `market_basel_zone` | `statistic` (exceptions k), `extras.window`, `extras.coverage` | `amber_level` (0.95), `red_level` (0.9999) | with X ~ Binomial(window, 1 - coverage): amber from the smallest count whose P(X <= k) reaches `amber_level`, red from the one that reaches `red_level`, green below. 250 days at 99 percent: green 0 to 4, amber 5 to 9, red 10 or more (MAR32.9 Table 1). |
| `market_pla_zone` | `statistic` (Spearman), `metric` (KS) | `spearman_green` 0.80, `spearman_red` 0.70, `ks_green` 0.09, `ks_red` 0.12 | red when Spearman is below `spearman_red` or KS above `ks_red`; green when Spearman is above `spearman_green` and KS below `ks_green`; amber otherwise (MAR32.42; strict comparisons). |
| `ml_interval_above_zero` | `extras.lo` | none | amber when the lower end of the paired-Brier interval is above 0, green otherwise. |
| `oprisk_capital_stability` | `metric` (dispersion), `statistic` (bias) | `dispersion_green`, `dispersion_amber`, `bias_green`, `bias_amber` | the worse of `upper_le` on the dispersion and `upper_le` on |bias|. |
| `p_value_negative_red` | `p_value`, `extras.negative_entries` | `amber`, `red` of the entry | red when the recursion has a negative entry; otherwise green when p is at least `amber`, amber when at least `red`, red below. |
| `portfolio_size_power` | `extras.role`, `p_value`, `metric` | `amber`, `red` (size cells), `power_green`, `power_amber` (power cells) | a size cell: green when p is at least `amber`, amber when at least `red`, red below; a power cell (role `power`): `lower_ge` on the rejection rate with `power_green` and `power_amber`. |
| `red_if_nonempty:F` | the field F (a list of failures) | none | red when the list is not empty, green when it is empty. |
| `red_if_positive:F` | the field F (a count or a violation) | none | red when F is above 0, green otherwise. |
| `stress_reverse_solution` | `statistic`, `extras.closer_start` | `tolerance` | red when the statistic is above `tolerance` or a start is closer; green otherwise. |
| `stress_satellite` | `extras.peak_ratio`, `metric` (timing error) | `peak_amber`, `peak_red`, `timing_quarters` | green when |1 - peak ratio| is at most `peak_amber` and |timing| at most `timing_quarters`; amber when |1 - peak ratio| is at most `peak_red`; red otherwise. |
| `stress_severity` | `statistic` | `minimum` | green when the statistic is at least `minimum`, amber below; never red. |
| `upper_le:F` | the field F | `amber`, `red` of the entry | green when F is at most `amber`, amber when at most `red` (both with the 1e-12 tolerance), red above. A value equal to `amber` is green. |
| `worst_of_screens` | `extras.screens` | the entries `ml.leakage_screen.*` | the worst light of the screens, each read by the entry of its own key; needs the policy (`Policy.classify_row`). |

<!-- rules:end -->

Decisions of D16-17 visible in the files:

- A per-case measured number is data. `ccr.pit_exposure_backtest` holds no critical value: its rule reads the row's
  `extras["simulated_critical_values"]` (`p05`, `p01`).
- A keyed entry carries one parameter set per value of a row field: the Euler tolerance by `extras.basis`
  (`exact-allocation` 1e-06 for ES and aggregation allocations, `kernel-var` 0.01 for the kernel VaR contributions) and the
  desk exception limit by `segment` (`alpha=0.01` 12, `alpha=0.025` 30).

Two rows are partly outside what a row can state: `ifrs9.lifetime_pd_backtest` reads `extras["observed"]` (written by the
engine since U16-E3; the committed documents predate it and the reproduction test derives it from the stored Brier score
and signed gap), and `liq.htm_loss_effect` sees the survival period only as the flags `extras["survives_horizon"]`, so a
lengthening between two finite periods is not visible to its rule. The committed `portfolio.euler_additivity` rows (no
`extras.basis`) and `ifrs9.lifetime_pd_backtest` rows (no `extras.observed`) classify `not_evaluated` through the
packaged files until the cases are re-baked; the re-bake must check that they reproduce.

## Sources: convention and regulation

A `source` that starts `convention:` is a Contraste internal gate, never a regulatory threshold. A source that starts
`regulatory:` is a value the Basel Framework sets (BCBS, in force 2023-01-01), read from the BIS text, which Contraste applies
as an internal gate:

- `market.desk_backtest` (MAR32.19): a desk with more than 12 exceptions at the 99th percentile, or 30 at the 97.5th, in the
  most recent 12 months is capitalised under the standardised approach. The paragraph's wording for the second limit has no
  'more than'; the engine reads both limits as 'more than' (red above the limit).
- `market.pla_test` (MAR32.42): green zone when Spearman is above 0.80 and KS below 0.09 (p-value 0.264); red zone when
  Spearman is below 0.7 or KS above 0.12 (p-value 0.055); amber otherwise.
- `market.basel_traffic_light` (MAR32.9 Table 1): 250 observations at 99 percent coverage, green 0 to 4 exceptions, amber 5
  to 9, red 10 or more. For another window or coverage the engine uses the 95 and 99.99 percent cumulative-probability
  levels of the BCBS 1996 design, an extension that is a convention of Contraste.

## The packaged default with parameters appended

Before U16-E3 the fairness (`fair.*`) and ML (`ml.*`) tests wrote the default label with their parameters appended
(`riskvalidation-default 2026.10.1; alpha_amber=0.05; alpha_red=0.01`, and for ml `...; ml-battery-1.1.0; ...`): 673 rows of
the committed Contraste documents. They now write the bare label `riskvalidation-default 2026.10.1`; the parameters are in
`extras["thresholds"]` (and, for ml, the battery version in `extras["battery"]`). `governance.policy` still resolves the old
form: it reads the label, then the parameters (`identity["parameters"]`) and the battery token (`identity["battery"]`).

`policies/default.json` keeps its label and every threshold; it gains `parameters` on 12 entries, the cut points a row named
but the pair does not hold under the engine's argument name (`tolerance`, `alpha_amber` and `alpha_red`,
`additivity_tolerance`, `overfit_tolerance`, `perturbation_tolerance`, `weak_spot_multiple`): the strict rows store their amber
one float above the tolerance (`0.05000000000000001`), so the tolerance itself is a parameter. No entry was added: the
fairness and ML contracts state no policy row for `fair.adverse_impact`, `fair.conditional_disparity`,
`fair.rate_spread_disparity`, `fair.replication`, `ml.challenger_brier`, `ml.cbpe_estimate` and `ml.leakage_screen`, whose rule
is the whole classification. `Policy.classify_row` classifies them by `rules.RULE_ONLY` (rule and constants, for example
`four_fifths = 0.8`) whatever the policy holds; `ml.leakage_screen` reads the four `ml.leakage_screen.*` entries.

## Reproduction

`tests/test_packaged_policies.py` proves, over the committed Contraste documents (`CONTRASTE_ROOT`, read-only), that every
cut point of every inline string maps to an entry and that the entry reproduces the committed light of all 3,123 inline rows
(2,450 battery rows and 673 default rows with parameters), through `Policy.classify_row` and through `Policy.classify` with the
row. It skips when `CONTRASTE_ROOT` is unset.

## The batteries

Amber and red are the pair `Threshold` holds; the parameters repeat every cut point of the old inline string under the
engine's argument name, so each old string maps to its entry by name.

<!-- batteries:begin -->

### `contraste-aggregation-battery 1.0.0`

File `policies/contraste-aggregation-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `aggregation.copula_choice_sensitivity` | upper_bad | `upper_le:statistic` | 0.05 | 0.15 | amber=0.05, red=0.15 |  | convention |
| `aggregation.correlation_stress` | upper_bad | `aggregation_corr_stress` | 0.1 | 0.1 | amber=0.1, tolerance=1e-09 |  | convention |
| `aggregation.diversification_sign` | upper_bad | `aggregation_diversification` | 1e-09 | 1e-09 | tolerance=1e-09 |  | convention |
| `aggregation.elliptical_mc_agreement` | upper_bad | `abs_upper_le:statistic` | 3 | 3 | z=3 |  | convention |
| `aggregation.horizon_confidence_consistency` | upper_bad | `red_if_positive:statistic` | 0 | 0 | expected_horizon=1, expected_level=0.999 |  | convention |
| `aggregation.rosenberg_schuermann_reference` | upper_bad | `always_green` | 0.3 | 0.4 | high=0.4, low=0.3 |  | convention |
| `aggregation.simple_sum_and_variance_covariance` | upper_bad | `aggregation_sum_vc` | 0.1 | 0.1 | amber=0.1, tolerance=1e-09 |  | convention |
| `aggregation.worst_case_bracket` | upper_bad | `red_if_positive:extras.rows_outside` | 1e-09 | 1e-09 | tolerance=1e-09 |  | convention |

### `contraste-c12 1.0.0`

File `policies/contraste-c12-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `ccr.ore_cva_agreement` | upper_bad | `upper_le:statistic` | 2 | 2 | k_se=2 |  | convention |
| `ccr.ore_dim_agreement` | upper_bad | `upper_le:statistic` | 2 | 2 | k_se=2 |  | convention |
| `ccr.ore_eepe_agreement` | upper_bad | `upper_le:statistic` | 2 | 2 | k_se=2 |  | convention |
| `ccr.quantlib_bond_call` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | rel_tol=1e-06 |  | convention |
| `ccr.quantlib_bond_put` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | rel_tol=1e-06 |  | convention |
| `ccr.quantlib_swaption_payer` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | rel_tol=1e-06 |  | convention |
| `ccr.quantlib_swaption_receiver` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | rel_tol=1e-06 |  | convention |

### `contraste-ccr-battery 1.0.0`

File `policies/contraste-ccr-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `ccr.cva_recomputed` | upper_bad | `upper_le:metric` | 1e-09 | 1e-09 | rel_tol=1e-09 |  | convention |
| `ccr.ee_analytic_vs_simulated` | upper_bad | `upper_le:statistic` | 3 | 3 | k_se=3 |  | convention |
| `ccr.eepe_recomputed` | upper_bad | `upper_le:metric` | 1e-09 | 1e-09 | rel_tol=1e-09 |  | convention |
| `ccr.martingale_discounted_bonds` | upper_bad | `upper_le:statistic` | 3 | 3 | k_se=3 |  | convention |
| `ccr.path_convergence` | upper_bad | `ccr_path_convergence` | 3 | 6 | k_se=3 |  | convention |
| `ccr.pfe_exceedance_backtest` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  | convention |
| `ccr.pit_exposure_backtest` | upper_bad | `ccr_pit_simulated` | 0 | 0 |  |  | convention |
| `ccr.reconcile_ee_profile` | upper_bad | `upper_le:statistic` | 2 | 2 | k_se=2 |  | convention |

### `contraste-ifrs9-battery 1.0.0`

File `policies/contraste-ifrs9-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `ifrs9.ecl_backtest` | upper_bad | `ifrs9_ecl_backtest` | 1.25 | 2 | r_amber_high=2, r_amber_low=0.5, r_green_high=1.25, r_green_low=0.8 |  | convention |
| `ifrs9.lifetime_pd_backtest` | upper_bad | `ifrs9_lifetime_pd` | 0.25 | 0.5 | alpha_amber=0.05, alpha_red=0.01, gap_amber=0.25, gap_red=0.5 |  | convention |
| `ifrs9.nonlinearity` | upper_bad | `ifrs9_nonlinearity` | 1e-09 | 1e-09 | gauss_hermite_tolerance=1e-09 |  | convention |
| `ifrs9.overlay_diagnostics` | upper_bad | `ifrs9_overlay` | 0.1 | 0.25 | share_amber=0.1, share_red=0.25 |  | convention |
| `ifrs9.reconciliation_identities` | upper_bad | `red_if_nonempty:extras.failing_identities` | 1e-06 | 1e-06 | tolerance=1e-06 |  | convention |
| `ifrs9.scenario_sensitivity` | upper_bad | `ifrs9_scenario_sensitivity` | 1 | 3 | nl_amber=1, nl_red=3 |  | convention |
| `ifrs9.sicr_effectiveness` | lower_bad | `ifrs9_sicr_effectiveness` | 2 | 1 | amber_ratio=1, direct_share_amber=0.5, green_ratio=2 |  | convention |
| `ifrs9.sicr_threshold_stability` | upper_bad | `ifrs9_sicr_stability` | 1 | 1 | amber_count=1, red_share_of_periods=0.5 |  | convention |
| `ifrs9.stage_migration_backtest` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  | convention |

### `contraste-irrbb-battery 1.0.0`

File `policies/contraste-irrbb-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `irrbb.behavioural_change_at_breach` | upper_bad | `red_if_positive:statistic` | 0 | 0 |  |  | convention |
| `irrbb.cmf_annex1_goldens` | upper_bad | `red_if_nonempty:extras.failed` | 1e-12 | 1e-12 | tolerance=1e-12 |  | convention |
| `irrbb.eve_sum_of_parts` | upper_bad | `upper_le:metric` | 1e-09 | 1e-09 | rel_tol=1e-09 |  | convention |
| `irrbb.eve_with_without_behavioural` | upper_bad | `irrbb_eve_with_without` | 0.05 | 0.15 | diff_amber=0.05, diff_red=0.15 |  | convention |
| `irrbb.nii_out_of_sample` | upper_bad | `upper_le:statistic` | 0.1 | 0.25 | err_amber=0.25, err_green=0.1 |  | convention |
| `irrbb.nmd_parameter_stability` | upper_bad | `upper_le:statistic` | 0.1 | 0.25 | gap_amber=0.1, gap_red=0.25 |  | convention |
| `irrbb.srp31_92_goldens` | upper_bad | `irrbb_srp31_92` | 0.0005 | 0.05 | tolerance_bp=0.05, tolerance_scalar=0.0005 |  | convention |

### `contraste-lgd-ead-battery 1.0.0`

File `policies/contraste-lgd-ead-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `lgd_ead.ccf_estimator_bias` | upper_bad | `abs_upper_le:metric` | 0.02 | 0.04 | alpha_amber=0.05, alpha_red=0.01, tolerance=0.02 |  | convention |
| `lgd_ead.downturn_coverage` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  | convention |
| `lgd_ead.ead_backtest_by_band` | upper_bad | `upper_le:metric` | 0.1 | 0.25 | gap_amber=0.1, gap_red=0.25 |  | convention |
| `lgd_ead.lgd_calibration_by_segment` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  | convention |
| `lgd_ead.lgd_calibration_over_time` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  | convention |
| `lgd_ead.lgd_discrimination` | lower_bad | `lower_ge:statistic` | 0.3 | 0.1 | rho_amber=0.1, rho_green=0.3 |  | convention |

### `contraste-liquidity-battery 1.0.0`

File `policies/contraste-liquidity-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `liq.cbc_haircut_rules` | upper_bad | `liquidity_haircut` | 1e-12 | 1e-12 | tolerance=1e-12 |  | convention |
| `liq.cliff_effect_diagnostic` | lower_bad | `lower_ge:statistic` | -0.01 | -0.03 | amber=-0.01, red=-0.03 |  | convention |
| `liq.htm_loss_effect` | upper_bad | `liquidity_htm_loss` | 0 | 0 | tolerance=0 |  | convention |
| `liq.htm_revaluation_vs_reported` | upper_bad | `upper_le:statistic` | 0.05 | 0.15 | reval_amber=0.15, reval_green=0.05 |  | convention |
| `liq.list_compounding_golden` | upper_bad | `upper_le:statistic` | 5e-07 | 5e-07 | tolerance=5e-07 |  | convention |
| `liq.nlp_identity` | upper_bad | `upper_le:statistic` | 1e-06 | 1e-06 | tolerance=1e-06 |  | convention |
| `liq.reverse_stress_round_trip` | upper_bad | `upper_le:statistic` | 0 | 0 | tolerance_days=0 |  | convention |
| `liq.runoff_adequacy_back_test` | lower_bad | `lower_ge:statistic` | 0.8 | 0.5 | ratio_amber=0.5, ratio_green=0.8 |  | convention |
| `liq.runoff_vs_generator_truth` | upper_bad | `upper_le:statistic` | 2 | 3 | se_green=2, se_red=3 |  | convention |
| `liq.survival_monotone_in_severity` | upper_bad | `upper_le:statistic` | 0 | 0 | tolerance_days=0 |  | convention |
| `liq.uninsured_share_sensitivity` | upper_bad | `upper_le:statistic` | 0 | 0 | tolerance_days=0 |  | convention |

### `contraste-market-battery 1.0.0`

File `policies/contraste-market-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `market.acerbi_szekely` | p_value | two cut points | 0.05 | 0.0001 | alpha_amber=0.05, alpha_red=0.0001 |  | convention |
| `market.basel_traffic_light` | upper_bad | `market_basel_zone` | 0.95 | 0.9999 | amber_level=0.95, red_level=0.9999 |  | regulatory |
| `market.christoffersen_conditional_coverage` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  | convention |
| `market.christoffersen_independence` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  | convention |
| `market.desk_backtest` | upper_bad | `upper_le:statistic` | 12 | 12 |  | `segment`: `alpha=0.01` -> amber=12, red=12, limit=12; `alpha=0.025` -> amber=30, red=30, limit=30 | regulatory |
| `market.du_escanciano_conditional` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  | convention |
| `market.du_escanciano_unconditional` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  | convention |
| `market.duration_test` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  | convention |
| `market.dynamic_quantile_test` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  | convention |
| `market.kupiec_pof` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  | convention |
| `market.mcneil_frey_residual_test` | p_value | two cut points | 0.1 | 0.05 | alpha_amber=0.1, alpha_red=0.05 |  | convention |
| `market.pla_test` | lower_bad | `market_pla_zone` | 0.8 | 0.7 | ks_green=0.09, ks_red=0.12, spearman_green=0.8, spearman_red=0.7 |  | regulatory |

### `contraste-oprisk-battery 1.0.0`

File `policies/contraste-oprisk-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `oprisk.benchmark_reproduction` | upper_bad | `upper_le:metric` | 1 | 2 | moment_rel_tol=1e-05, quantile_rel_tol=0.01 |  | convention |
| `oprisk.capital_stability` | upper_bad | `oprisk_capital_stability` | 0.25 | 0.5 | bias_amber=0.25, bias_green=0.1, dispersion_amber=0.5, dispersion_green=0.25 |  | convention |
| `oprisk.scenario_influence` | upper_bad | `upper_le:metric` | 0.1 | 0.25 | influence_amber=0.1, influence_red=0.25 |  | convention |
| `oprisk.sma_ilm_response` | upper_bad | `upper_le:metric` | 0.05 | 0.15 | jump_amber=0.15, jump_green=0.05 |  | convention |
| `oprisk.threshold_stability` | upper_bad | `upper_le:metric` | 0.1 | 0.25 | tol_amber=0.25, tol_green=0.1 |  | convention |

### `contraste-portfolio-battery 1.0.0`

File `policies/contraste-portfolio-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `portfolio.correlation_size_power` | p_value | `portfolio_size_power` | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01, power_amber=0.5, power_green=0.8 |  | convention |
| `portfolio.crp_recursion_vs_mc` | p_value | `p_value_negative_red` | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  | convention |
| `portfolio.euler_additivity` | upper_bad | `upper_le:metric` | 1e-06 | 1e-06 |  | `extras.basis`: `exact-allocation` -> amber=1e-06, red=1e-06, tolerance=1e-06; `kernel-var` -> amber=0.01, red=0.01, tolerance=0.01 | convention |
| `portfolio.mc_convergence` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  | convention |
| `portfolio.tail_backtest` | p_value | two cut points | 0.05 | 0.01 | alpha_amber=0.05, alpha_red=0.01 |  | convention |

### `contraste-stress-battery 1.0.0`

File `policies/contraste-stress-battery-1.0.0.json`.

| Test id | Kind | Rule | Amber | Red | Parameters | Keyed by | Source |
|---|---|---|---|---|---|---|---|
| `stress.bottom_up_vs_top_down` | upper_bad | `abs_upper_le:statistic` | 0.25 | 0.5 | amber=0.25, red=0.5 |  | convention |
| `stress.reverse_solution_check` | upper_bad | `stress_reverse_solution` | 1e-06 | 1e-06 | tolerance=1e-06 |  | convention |
| `stress.satellite_backcast` | upper_bad | `stress_satellite` | 0.35 | 0.6 | peak_amber=0.35, peak_red=0.6, timing_quarters=2 |  | convention |
| `stress.scenario_severity` | lower_bad | `stress_severity` | 0.75 | 0.75 | minimum=0.75 |  | convention |
| `stress.sensitivity_monotonicity` | upper_bad | `red_if_positive:statistic` | 0 | 0 |  |  | convention |
| `stress.static_balance_sheet_rules` | upper_bad | `red_if_positive:statistic` | 0.006 | 0.006 | tolerance=0.006 |  | convention |


<!-- batteries:end -->
