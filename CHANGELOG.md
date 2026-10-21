# Changelog

All notable changes are recorded here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use the display form `X.XX.XXX`; the package
metadata carries the PEP 440 form (`0.01.000` is `0.1.0`).

## [0.09.000] - 2026-10-21

Model governance (Contraste wave 3: unit U15).

- New `riskvalidation.governance` (U15-E, contract `engine-governance-api` 1.0.0): `scope` (the B.1 scope test over the profiles `us-2026`, `pra`,
  `osfi-2027` and `ecb`), `inventory` (entries keyed `case_id/fit_id` with the forty B.2 attributes as value, status and basis, components,
  edges, `assign_variant`, `build_inventory` refusing a repeated key, an absent entry and a cycle named in dependency order, Kahn generations,
  depth, fan-in and fan-out, completeness), `tiering` (exposure band, materiality, complexity as the mean rounded half up in integer
  arithmetic or the maximum, the 4 by 4 matrix, the override with its rationale and approver, `tier_sensitivity` under the four readings,
  reproducing the tiers of the nineteen built cases), `lifecycle` (the twelve B.4 states, use before validation only with an exception
  record), `findings` (severity and status maps, deadline, due date, ageing and overdue at `due = as_of`, the seven states and their
  transitions, the quantitative anchor, the RWEA share, register counts), `schedule` (calendar months with the day clamped, the next review,
  revalidation and monitoring dates, the ECB minimums), `opinion` (the level as the maximum of the severity level and the RWEA level with the
  ECB 5 and 10 percent cut points, so an anchor can only raise a level; the recommendation labelled the rule's output), `appetite` (the seven
  B.6 metrics counting distinct findings with the register rows beside them), `policy` (`resolve_policy`: a packaged label or an inline battery
  string, the kind from the registry, the direction null with "no policy file (BL-085)") and `report` (the twelve sections of the I.3
  validation report, filled or absent with their reason, bilingual, deterministic). Every tier count, band, deadline, frequency and
  threshold is a named convention with the label `convention` (`riskvalidation.governance.CONVENTIONS`), never a regulatory value; no text
  states an approval or a certification. `riskvalidation.validation.governance` returns the five `gov.` tests (`gov.inventory_completeness`,
  `gov.findings_overdue`, `gov.validation_due`, `gov.tier_override_documented`, `gov.edges_acyclic`) registered in the family `governance`,
  with their rows in `GOV_ROWS` and `gov_default_policy()` until `policies/default.json` holds them. The four contract fixtures (C01/german,
  C07/banco-andino, C18/banco-andino, C22/generators) are reproduced field by field, the 81 register rows included. Documentation in
  `docs/governance/`.
- Additive to the contract (U15-E, to be taken into `engine-governance-api` 1.1.0): `build_inventory(..., related=())` validates the related
  links named by the golden `related link A and Z: unknown entry Z`; `anchor_from_test_row(..., document="")` prefixes the basis pointer; a test
  row may state its own `unit`; `appetite_population`; `find_cycle` and `nodes_on_cycles`; the `gov.` functions accept `policy`, `model_id`
  and `segment` keywords so that `run_suite` can call them.
- U15-E fix round after the Opus review: a null, numeric or blank rationale or approver of a tier override is undocumented (`apply_override`
  refuses it, `gov.tier_override_documented` counts it); `import riskvalidation.validation` registers the `gov.` tests; a non-finite RWEA
  share, amount or total is refused; `anchor_from_test_row` stops at the first row that matches the citation; `resolve_policy` gives a kind
  `None` for an unregistered test (no fallback to a threshold) and resolves a `+gov-` label through `gov_default_policy`; `executive_counts`
  is tested with the `error` light. No golden changes.
- Engine 0.09.000 integration of the governance unit under the 0.08.000 catalogue rules: the five `gov.` tests (`gov.inventory_completeness`,
  `gov.findings_overdue`, `gov.validation_due`, `gov.tier_override_documented`, `gov.edges_acyclic`; all decided by a metric gate, so none
  needs a size suite) are in the registry inventory (family `governance`, prefix `gov`), and the five `gov.` rows are in the packaged
  `policies/default.json`, written from `GOV_ROWS` and `gov_threshold` (30 lines added, no existing row changed, label 2026.10.1 unchanged).
  `gov_default_policy()` now returns the default policy itself; the `+gov-` label branch of `resolve_policy` stays and is harmless.
  The governance report keeps reproducing the Contraste fixtures except the `kind` and `kind_reason` of 11 (C07) and 17 (C18) report rows
  whose test ids the fixture wrote as unregistered at the batch 3 head; the engine registers them since 0.08.000 and states the decision.

## [0.08.001] - 2026-10-20

A correction of 0.08.000's fairness rows, and the PyPI publication route.

- The outcome-conditioned fairness tests (`fair.equalized_odds`, `fair.equal_opportunity`, `fair.calibration_within_groups`,
  `fair.auc_by_group`) decide their light on a metric, so their `alpha_amber` and `alpha_red` are null, as the `ml.` metric rows
  and the three fairness rows of `engine-fairness-api` 1.2.0 already are; the metric's cut points stay in
  `extras["thresholds"]`. The alpha fields are p-value cut points, and a consumer that read them as such misread these rows.
  The statistic, the metric and the light are unchanged.
- The publish workflow tests the built wheel as CI tests the source (the `dev` and `market` extras, `-m "not sizepower"`), and
  the session-wide registry check recognises engine frames by the imported package's folder, so it holds on an installed wheel.
  0.8.0 was published to PyPI from this route.

## [0.08.000] - 2026-10-20

Liquidity, aggregation and stress testing, the regulatory calculators and counterparty credit risk, machine-learning model validation and fair lending (Contraste wave 2 batch 3: units U11, U13, U9 and U14).

- **New packages:** `riskvalidation.liquidity`, `riskvalidation.aggregation`, `riskvalidation.stress`, `riskvalidation.regulatory`, `riskvalidation.ccr`, `riskvalidation.ml` and `riskvalidation.fairness`, each with its validation module.
- **Contracts:** each package reproduces the golden values of its Contraste contract: `engine-liquidity-api` 1.2.0, `engine-aggregation-api` 1.1.0, `engine-stress-api` 2.0.0, `engine-regulatory-api` 2.2.0, `engine-ccr-api` 2.0.0, `engine-ml-api` 2.0.0 and `engine-fairness-api` 1.2.0. Each was reviewed by Opus, with its fixes applied.
- **Test catalogue:** it lists every validation test, with the size coverage of each p-value test stated (BL-088).
- **Reconciliation and Monte Carlo error:** the cohort estimator is reconciled with `transitionMatrix` (BL-083), and the IRRBB Monte Carlo error takes antithetic batches (BL-074).
- **Packaged policy:** it gains the `fair.` and `ml.` rows. No existing row changes, so its label stays 2026.10.1.

No existing engine result changes.

- `simulation.ear_evar_mc_error` takes an optional `antithetic=False`. With `antithetic=True` the batches are unions of
  whole mirror pairs of an antithetic path set (as `hw1f_scenarios` returns): no batch holds a path whose mirror sits in
  another batch, so the batches are independent, as the batch-means error assumes. `P` must be even and
  `P / 2 >= n_batches`. The result for `antithetic=False` is unchanged (BL-074).
- Measured on a Hull-White NII proxy (the 99 percent quantile, 20 batches, against its spread over independent seeds), the
  two batchings agree within noise: 0.961 and 0.951 of the truth at 10 000 paths, 1.04 and 1.07 at 40 000. At 2 000
  paths both sit near 0.78, a small-batch effect of the estimator. The C15 review measured a larger gap on its own paths
  (25.5 against 33.7); the C14 and C15 re-bakes show each case's effect.
- The cohort estimator is reconciled with the open-source `transitionMatrix` 0.5.1 (Apache-2.0), the optional extra
  `transitions`, in the tests only (`tests/test_transitions_reconciliation.py`, RV-415, BL-083): `pooled_cohort` and `cohort`
  equal the package's `CohortEstimator` element-wise (largest difference 0.0, tolerance 1e-12) on a seeded panel with
  withdrawals, for the `state` treatment directly and the `exclude` treatment through the package's renormalised matrix.
  No engine result changes.
- Findings pinned with both values: the package double counts the last record of its last entity (up to 1.6e-3 on the test
  panel), fills an empty row with zeros (the engine: NaN, listed), keeps a reinstated default as a migration (the engine:
  absorbing), has no maximum likelihood generator from paths, and divides the Aalen-Johansen increment by the risk set
  after the exits (1/3 where the engine and the textbook give 1/4).
- The thirteen `ml.` validation tests (`ml.robustness`, `ml.shift_resilience`, `ml.seed_stability`,
  `ml.reason_code_stability`, `ml.effect_agreement`, `ml.additivity`, `ml.overfit`, `ml.weak_spot`,
  `ml.conformal_coverage`, `ml.leakage_screen`, `ml.data_quality`, `ml.cbpe_estimate`, `ml.challenger_brier`) are
  registered in the test catalogue (family `ml`) with the `h0` and the `reference` their row carries; the two p-value
  tests (`ml.shift_resilience`, `ml.conformal_coverage`) are named to their size suite. No result changes.
- `import riskvalidation.validation` imports `validation.ml` too, so the package alone registers every validation test, the thirteen
  `ml.` ids included (the package-alone count test counts the whole inventory again).
- The test catalogue (`riskvalidation.core.registry`) now lists every validation test of the aggregation, ccr, fairness, ifrs9,
  irrbb, lgd_ead, liquidity, market, oprisk, portfolio and stress modules (89 test ids), each with its family, name, hypothesis,
  reference and decision (`p_value` where the light is decided by a p-value against alphas, `metric` otherwise), so that
  `spec(test_id)` answers for every row the engine writes. No function body, signature or result changes. `import
  riskvalidation.validation` registers them all (the package imports the eleven modules; no optional dependency is loaded).
  `ccr.pfe_exceedance_backtest` holds its hypothesis with `{q}` and `{1 - q}` where the row writes the quantile and its
  complement; `liq.reverse_stress_round_trip` writes a second hypothesis on its threshold branch.
- The engine's own tests check, over the whole session, that every `TestResult` an engine function builds names a registered test
  with the `h0` and `reference` of its spec; the size rule now accepts a measurement in `NULL_SCENARIOS` or in a named `sizepower`
  suite (`SIZE_SUITES`) and pins the registered p-value tests that have none (`SIZE_GAPS`).
- U11-E: liquidity (contract `engine-liquidity-api` 1.2.0). New `riskvalidation.liquidity` (modules imported by path, nothing re-exported):
  `runoff` (constant daily compounding `1 - (1 - q)^n` and its inverse, contractual roll-off, the transcribed ECB LiST 2019 Tables 3.2 and
  3.3 and drawdown rates as `LIST_PARAMETERS`, BCBS d555 Table A.1 as `TABLE_A1`), `cbc` (the LiST para 74 haircut rule, liquidity value,
  monetisation, the CBC stack, the HTM loss), `ladder` (contractual gaps, behavioural outflows, the net liquidity position, the survival
  period, the minimum NLP, the cliff effect), `scenarios` (families and phases, the LiST parameter sets, the fast-run calibration, the
  anchor, the uninsured-share and HTM-loss grids), `reverse` (the survival period against the run-off multiplier, the multiplier that
  reaches a target by `scipy.optimize.brentq`, refusals with their reason) and `deposits` (a seeded depositor generator with an exact
  truth by the transfer recursion).
- A survival period is never a number when the net liquidity position stays non-negative to the horizon: `SurvivalResult.survives` is true,
  `day` is `None` and `horizon` is the horizon. The reverse stress refuses an unbracketed target with its reason. A rate of exactly 0 or 1 is
  legitimate for a segment.
- New `riskvalidation.regulatory.lcr_nsfr`: HQLA with the LCR30.39 adjustments and caps, the LCR outflows, inflows, the 75 percent inflow
  cap and the ratio, the NSFR ASF, RSF and ratio, each parameter with its paragraph and the in-force date 2019-12-15. The trade finance rate
  and the NSFR derivative add-on rate (within 0.05 to 0.20) are caller-supplied, never defaults. The LCR is a Basel formula on aggregates,
  never the bank's reported or regulatory LCR; the goldens are constructed, no official worked example exists.
- New `riskvalidation.validation.liquidity`: eleven `TestResult` functions (`liq.list_compounding_golden`, `liq.survival_monotone_in_severity`,
  `liq.cbc_haircut_rules`, `liq.nlp_identity`, `liq.runoff_adequacy_back_test`, `liq.runoff_vs_generator_truth`,
  `liq.cliff_effect_diagnostic`, `liq.htm_loss_effect`, `liq.uninsured_share_sensitivity`, `liq.reverse_stress_round_trip`,
  `liq.htm_revaluation_vs_reported`) with the policy label `contraste-liquidity-battery-1.0.0`, thresholds in `extras["thresholds"]` and
  `not_evaluated` carrying no number. No existing engine result changes.
- Reading choices of the contract's open points, labelled in `docs/liquidity/`: `fast_run_scenario` takes an additive keyword `scored` to
  refuse the 2023 rows of Table A.1; a fast-run parameter set holds the one class `concentrated-uninsured`; the depositor chain is shared,
  so the segments of a configuration must carry the same `p_enter_stress` and `p_exit_stress`.
- Fix round after the Opus review (contract `engine-liquidity-api` 1.2.0): `net_cash_outflows` refuses a `cap` above 0.75 (LCR40.1);
  `solve_multiplier` returns the threshold multiplier when the survival period steps past the target (a monetisation CBC), with the new
  last field `SolveResult.survival_just_below`, and `validation.liquidity.reverse_stress_round_trip` takes the keyword-only
  `survival_just_below` to test the threshold property (without it the 1.1.0 equality check is kept); the HQLA level shares of an empty
  stock are NaN; `NSFR_RSF_FACTORS` gains `operational_deposits_held` at 0.50; `fast_run_scenario` takes `scored` as keyword-only; the
  module tables, `ListScenario` dicts and scenario rate paths are read-only; a negative tolerance is refused.
- Added `riskvalidation.aggregation` (contract `engine-aggregation-api` 1.0.0, U13-E1): `simple` (simple sum, correlation check,
  variance-covariance, hybrid quantile, diversification factor, Euler diversification benefit, standard deviation contributions),
  `copulas` (marginal quantile functions, Gaussian and Student t copula draws, t and Clayton tail dependence, Clayton copula, the
  copula aggregate), `bounds` (comonotonic VaR, the rearrangement algorithm, the best and worst VaR bounds, reproducing Tables 1 and 4
  of Embrechts, Puccetti and Ruschendorf 2013), `band` (the dependence-uncertainty band on the economic capital basis VaR less EL) and
  `model_risk` (Barrieu and Scandolo AM and RM, the Glasserman and Xu worst case). Vine copulas, Clayton sampling and the BVAR stay
  deferred (BL-078).
- Added `riskvalidation.validation.aggregation`: eight tests (simple sum and variance-covariance, copula choice sensitivity, worst-case
  bracket, correlation stress, diversification sign, horizon and confidence consistency, elliptical Monte Carlo agreement, the
  Rosenberg and Schuermann reference). No existing engine result changes.
- Added `docs/aggregation/` (six pages and an index) and the requirements `RV-1301` to `RV-1326` under
  `docs/design/features/u13-aggregation/`.
- Hardened `riskvalidation.aggregation` after the review of U13-E1: `worst_var_bounds` refuses a non-finite bound, `hybrid_quantile`
  refuses a negative column quantile, and `dependence_band` takes `ec_consistency_tolerance` (default 0.01) and refuses an `ec` that
  contradicts its marginals; the test gaps on the band EC basis, the upper range width and the random start are closed.
- New package `riskvalidation.stress` and module `riskvalidation.validation.stress` (contract `engine-stress-api` 1.0.0, requirements `RV-1301` to `RV-1324`, pages `docs/stress/`), imported by path. `stress.satellites`: the probit, logit and identity links, the OLS satellite fit with standard errors on an explicit window (the identity link for rates that are zero or negative), the scenario projection. `stress.paths`: transposition, reversion to the baseline, annual to quarterly, deviation paths, the vintage check, compared with C06's committed output. `stress.capital`: the loss-only CET1 path, the PPNR row, the internal capital target (the 4 percent cap on `p2r`), the Chilean stack, the three-year plan, the P2G proxy, the 34-row Annex 3 summary, the EBA static balance sheet. `stress.reverse`: the minimum Mahalanobis scenario (closed form and SLSQP) with its plausibility. `stress.scenario_models`: the VAR and its conditional forecast, the skewed t, growth-at-risk. `validation.stress`: six `TestResult` tests (back-cast, sensitivity signs, static balance sheet rules, bottom-up against top-down, scenario severity, reverse solution check).
- `stress.scenario_models.gar_quantile_regressions` reads `statsmodels`, the optional extra `stress` (the `pyproject.toml` entry is the integrator's); every other function needs NumPy and SciPy only. No existing engine result changes.
- Two goldens of the contract contradicted its own light rules at the default gates (the back-cast "0.5 against 1.2: red", whose
  deviation 0.5833 is within `peak_red` 0.60, and the bottom-up "160: amber", above the 0.50 red gate). The engine implements the rules
  as written, and the contract's goldens were corrected to 0.4 against 1.2 (red) and 140 (amber), keeping the approved thresholds in
  `engine-stress-api` 2.0.0 (Contraste U13 decision Z8).
- Review fixes: `InternalTarget` gains `p2r` (default 0.0) and Annex 3 rows 7, 7.2 and 8 read it, not `c_p2`; `annex3_summary` raises `ValueError` naming both values when the RWA of `rwa_by_risk` differs from the capital plan's beyond 1e-9 relative; the `not_evaluated` rows of `validation.stress` carry `model_id` and `segment`; the `con_jac` docstring of `reverse_stress` fits the line length.
- U14-E1: `riskvalidation.ml` (engine-ml-api 1.0.0): perturbation robustness and the shift score, the worst subsample and out-of-time
  windows, seed stability, reason-code stability and the explainer agreement, the Shapley additivity, sign and monotonicity checks,
  partial dependence, ICE and accumulated local effects, the overfit gap and the weak spots, split conformal prediction, the leakage
  screen, six data-quality counts, the label-free performance estimate (CBPE), the paired Brier bootstrap and the grade
  disagreement, and five planted-defect generators (leak, covariate shift, noise sensitivity, seed instability, exchangeability
  break). NumPy and SciPy only; the model enters as a predict callable on arrays.
- U14-E1: `riskvalidation.validation.ml`: twelve `TestResult` functions with ids `ml.robustness`, `ml.shift_resilience`,
  `ml.seed_stability`, `ml.reason_code_stability`, `ml.effect_agreement`, `ml.additivity`, `ml.overfit`, `ml.weak_spot`,
  `ml.conformal_coverage`, `ml.leakage_screen`, `ml.data_quality`, `ml.cbpe_estimate` and `ml.challenger_brier`. The lights read the
  contract's Policy thresholds rows, kept in `riskvalidation.ml.policy` (`ml_thresholds`, `ml_policy`) and added to a base policy
  only where it holds no row; `policies/default.json` is not edited (the integrator can move the rows there as they stand).
- U14-E1: requirements RV-1401 to RV-1439 (`docs/design/features/u14-ml/`), pages `docs/ml/`, the size and power of every test on its
  planted defect at 4,000 repetitions, and the MAPIE 1.5.0 reconciliation of the split conformal quantile (optional extra `ml`:
  `pytest.importorskip("mapie")`; `pyproject.toml` is the integrator's). Findings against the contract: the 5th percentile of the
  reason-code overlap is not bounded by the mean; the float product of the conformal rank misrounds for pairs such as
  n = 149 and alpha = 0.18, not for the contract's two examples; MAPIE's classification quantile is the k-th or the (k + 1)-th order
  statistic; the worst-subsample test has a false-alarm rate of about 10 percent under no shift.
- U14-E1 fix round (engine-ml-api 1.1.0, after the Opus review): `conformal_coverage` takes the calibration size as the required
  keyword `n_cal` and reads the covered count against the Beta-Binomial(m, k, n_cal + 1 - k) law (the binomial p-value stays in
  `p_value_binomial`, beside it); `ml.conformal_coverage` is `not_evaluated` when k exceeds n_cal. `ml.shift_resilience` reads an
  development-only reference (`shift_development_reference`, `ShiftReference`, default 199 draws) with the p-value cut points
  0.05 and 0.01, and is `not_evaluated` without development labels. `ml.cbpe_estimate` is always `not_evaluated` (an estimate, not a
  test), so it never moves the worst light. `reason_code_overlap` ranks reasons by signed attribution toward the event. The overfit
  gap and the AUC loss are rounded to 12 decimals before they are compared. `ml.policy.threshold_rows()` gives the `ml.` rows as
  plain mappings for `policies/default.json`. The size and power study carries the `sizepower` marker.
- The metric tests of `riskvalidation.validation.ml` (`robustness` and the tests built by `_simple`: the upper_bad and lower_bad
  rows) carry null `alpha_amber` and `alpha_red`, because they have no p-value threshold; their cut points stay in
  `extras['thresholds']`. The p-value tests (`shift_resilience`, `conformal_coverage`) keep 0.05 and 0.01. No light changes.
- New `riskvalidation.fairness` (U14-E2, contract `engine-fairness-api` 1.0.0): `metrics` (selection rates, demographic parity
  difference and ratio, equalized odds and equal opportunity, the adverse impact ratio, calibration and AUC within groups, and
  `four_fifths_flag`, decided on the counts in exact integer arithmetic against the highest-rate group, labelled an employment
  benchmark and never a credit-law threshold), `inference` (the pooled two-proportion z-test and Fisher's exact test), `conditional`
  (the logistic decision model by IRLS and the linear rate-spread model, the protected effect with its Wald interval), `replication`
  (an estimate on year t against year t+1 by sign and size), `situation` (situation testing after arXiv:2502.01267: the Gower
  distance under `range` or `standard`, k nearest neighbours, the statistic, the one-sided Wald interval with the exact quantile,
  ceteris paribus and mutatis mutandis, `bonferroni_alpha`) and `plants` (three planted-bias generators with exact truths). The
  published ST and CST counts of the loan-application experiment reproduce exactly under `standard` (a gated reconciliation, the data
  file not stored). `riskvalidation.validation.fairness` returns the twelve `fair.` tests as `TestResult` rows with their lights from
  `Policy` rows (`fairness.policy`), outcome-conditioned tests never run without an outcome. Each test has its size and power
  measured at 4,000 repetitions; documented in `docs/fairness/`. The engine core stays NumPy and SciPy; fairlearn and statsmodels
  are optional references of the tests. The `fair.` policy rows are carried by `fairness_default_policy()` until the integrator adds
  them to `policies/default.json`.
- U14-E2 fix round after the Opus review (`engine-fairness-api` 1.1.0): `demographic_parity` and `situation_testing` cap a red light from
  any policy at amber and say so in `notes`; `protected_effect` refuses index 0 (the intercept; protected indicator j is index 1 + j);
  `situation_test` takes `categorical`, a boolean mask over the columns of `X` (overlap for a categorical column, normalised Manhattan
  for the others, also under `standard`; `None` keeps 1.0.0); the `fair.situation_testing` amber cut is 0.20 (size 0.0395, power 1.000
  at n 400, k 30, 4,000 repetitions); `fairness_default_policy()` returns the default policy unchanged once it holds every `fair.` key; the
  size and power module carries the `sizepower` pytest marker (the integrator registers it and deselects it in the engine CI).
- `engine-fairness-api` 1.2.0 (the U14-P3 review, deviation 4): `fair.conditional_disparity`, `fair.rate_spread_disparity` and
  `fair.situation_testing` carry null `alpha_amber` and `alpha_red`. They have no p-value threshold: the interval level and the
  share cut point stay in `extras.thresholds` and in `policy_version`. No statistic, metric or light changes.
- U9 regulatory calculators (`riskvalidation.regulatory`, contract `engine-regulatory-api` 1.0.0): five new modules, each function of the
  contract with its golden values and invariants. `standardised` holds the CRE20 risk weights, credit conversion factors and the lookup as
  data, every entry a `RegulatoryParameter(value, paragraph, table, in_force)` (`RiskWeight` is an alias); `output_floor` the RBC90.1
  phase-in and `floored_rwa` (RBC20.13: 101.5, and 95 with the RBC90.2 cap); `sa_ccr` the CRE52 functions with every intermediate returned
  (the five CRE99 netting sets from their trades give EAD 569, 381, 5,406, 936 and 1,879 exactly after rounding); `ba_cva` the MAR50.14 to
  MAR50.25 formulas and the MAR50.16 Table 1 weights; `frtb` the SBM bucket and cross-bucket aggregation with the MAR21.4(5)(b) fallback,
  the GIRR delta parameters (MAR21.42 to MAR21.50, with the sqrt(2) division of MAR21.44 stated), curvature, `sbm_capital` and the MAR33.4
  liquidity-horizon cascade with the MAR33.12 horizons.
- The GIRR delta sensitivity of `girr_delta_charge` is the change in value for a 1 bp shift divided by 0.0001 (MAR21.19), not the PV01 per
  basis point. `entity_aggregate` takes one correlation per entity (CRE99 netting set 2 uses 0.5 and 0.8). Stated limits: no vega, no
  SA-CVA, DRC or RRAO, no verified BA-CVA numeric example (BL-080). Pages in
  `docs/regulatory/u9/`; requirements RV-901 to RV-925 in `docs/design/features/u9-regulatory/requirements.md`. NumPy and SciPy only.
- Fix round after the review: `curvature_charge` squares the cross-bucket `gamma` as it does `rho` (MAR21.101, transcribed) and, when K_b+ equals
  K_b-, selects the upward scenario only if sum CVR+ exceeds sum CVR- (MAR21.5(3)(a)(iii)); the curvature shock for consumers is
  `curvature_shock_undivided` (the divided value is a derived reading); a forward-starting trade is pinned to its end-date bucket in a test.
- Additive terms for the regulatory parity bake (contract `engine-regulatory-api` 2.1.0): `sa_ccr.credit_addon_terms` and `equity_addon_terms` return
  `AddonTerms(systematic, idiosyncratic, addon)`, the two CRE52.61 terms under the root and the add-on (`entity_aggregate_terms` for explicit add-ons);
  `frtb.cross_bucket_charge_detail` returns `CrossBucket(charge, fallback_used)`, true exactly when the MAR21.4(5)(b) fallback applied. The existing
  functions keep their signatures and results and call the new ones. RV-926 and RV-927; pages 03 and 05 of `docs/regulatory/u9/`.
- Trade effective notionals (contract `engine-regulatory-api` 2.1.0): `sa_ccr.credit_trade_effective_notional` (delta x notional x supervisory duration x maturity factor, CRE52.34)
  and `commodity_trade_effective_notional` (delta x adjusted notional x maturity factor), so the parity bake computes none of them. RV-928.
- New package `riskvalidation.ccr` (U9-E2, contract `engine-ccr-api` 1.0.0): `hw1f` (the Hull-White one-factor fit to today's curve,
  the affine bond price, the exact simulation of the state and of the short rate, the zero-coupon bond option, the Jamshidian
  swaption and the AR(1) calibration with standard errors), `swaps` (fixed-float swaps valued at any date, including between two
  resets from the path's last fixing), `exposure` (EE and ENE, the effective EE and EPE, PFE with a seeded bootstrap error, netting,
  the effective maturity, the analytic EE of a single swap, rung 1 as a wrapper of `regulatory.sa_ccr`), `collateral` (the credit
  support amount, the delivery amount, the CRE53 MPoR floors, the close-out exposure, the regression dynamic initial margin, a
  proxy and never SIMM) and `cva` (CVA, DVA and the Hull and White wrong-way hazard model with its sequential calibration).
- New `riskvalidation.validation.ccr`: eight tests (`ccr.ee_analytic_vs_simulated`, `ccr.martingale_discounted_bonds`,
  `ccr.eepe_recomputed`, `ccr.cva_recomputed`, `ccr.path_convergence`, `ccr.reconcile_ee_profile`, `ccr.pfe_exceedance_backtest`,
  `ccr.pit_exposure_backtest`) and the helpers of the PIT backtest (`pit_from_simulated`, `anderson_darling_w2`,
  `effective_sample_size`, `simulate_pit_critical_values`, `CriticalValues`). A PIT backtest without simulated critical values is
  `not_evaluated` with no number.
- Additions to the contract's names, reported for its next revision: `hw1f.curve_functions`, `hw1f.bank_account`,
  `hw1f.bond_option_vol`, `hw1f.ar1_to_hw1f`, `swaps.reset_times`, `swaps.simulation_grid` and `validation.ccr.IID_CRITICAL_VALUES`,
  and one optional trailing argument, `nominal`, of `hw1f.swaption_jamshidian` (the contract's signature cannot carry the notional
  of the coupon bond equation).
- Documentation in `docs/ccr/` and requirements `RV-950` to `RV-986` in `docs/design/features/u9-ccr/requirements.md`. No existing
  engine result changes.
- Fix round after the Opus review of U9-E2: `hw1f.bank_account(grid, short_rate, a, sigma, p0, forward)` integrates phi exactly (the
  trapezoid on r biased every discounted exposure by up to 0.2 percent at five years on a semiannual grid); `wwr_cva` validates `dt`
  also at b = 0; the residual spread of `calibrate_ar1` uses ddof = 1 as the source's listing; `validation.ccr.path_convergence`
  takes the Monte Carlo standard errors as required keyword arguments (`ee_se_by_path_count`, `cva_se_by_path_count`) and its band is
  `k_se` standard errors of the smaller count.

## [0.07.000] - 2026-10-14

Portfolio credit capital, market risk, interest rate risk in the banking book and operational risk (Contraste units U7, U8, U10 and U12): `riskvalidation.portfolio` with `riskvalidation.validation.portfolio`, `riskvalidation.market` with `riskvalidation.validation.market` (the GARCH family through the optional extra `market`), `riskvalidation.irrbb` with `riskvalidation.validation.irrbb`, and `riskvalidation.oprisk` with `riskvalidation.validation.oprisk`, each reproducing the golden values of its Contraste contract (`engine-portfolio-api` 1.0.0, `engine-market-api` 1.0.0, `engine-irrbb-api` 1.1.0, `engine-oprisk-api` 1.0.0) and each reviewed by Opus with its fixes applied. No existing engine result changes.

- engine U7-E: new package `riskvalidation.portfolio` (the portfolio record and its EL, VaR, ES and EC with the Acerbi-Tasche sample
  estimators and the large homogeneous portfolio measures; the Gaussian and Student-t copula Monte Carlo, seeded and chunked to a stated
  memory bound, with batch-means standard errors; CreditRisk+ with Gamma sectors by the stable recursion of Haaf, Reiss and Schoenmakers
  (2004) and the Panjer comparison; the Gordy and Lutkebohmert (2013) granularity adjustment as a share of EAD from the caller's UL capital
  K_i, with `k_creditrisk_plus` for eq. 13; importance sampling for the Gaussian copula accepted by its unbiasedness against plain
  Monte Carlo (the twist is UNVERIFIED in dossier 04 H.7); Euler ES contributions that add up exactly, kernel VaR contributions, the
  asymptotic one-factor anchor, HHI and the effective number of names; the fit of the Vasicek distribution to yearly loss rates by
  maximum likelihood and moments with the profile interval and a parametric bootstrap, and the implied factors) and
  `riskvalidation.validation.portfolio` (the tail back-test, Monte Carlo convergence, Euler additivity, CreditRisk+ recursion against
  Monte Carlo, and the size and power of the correlation estimate), contract `engine-portfolio-api` 1.0.0, requirements RV-701 to RV-729,
  documentation in `docs/portfolio/`. Existing engine results are unchanged.
- engine U7-E review fixes: `importance.weighted_var_es` builds the quantile and the ES on the upper tail probability
  sum(w 1{L > x}) / S normalised by the scenario count, not by the realised sum of the weights (which gave a VaR of 35
  against 21.0 at the plain 99.9 percent VaR; blocker B1) and refuses a level whose tail mass the weights cannot
  account for; tests for the production threshold, for the size of the batch-means standard errors against the spread over
  replications and for the moments estimator on an asymmetric series; documentation of the lattice case.
- U8 market risk (RV-801 to RV-830): `riskvalidation.market` with the sign convention (`to_loss_numbers`, `var_es_from_sample`,
  `var_es_from_distribution`), historical, age-weighted, filtered and volatility-updated simulation, delta-normal, Cornish-Fisher
  and EWMA, GARCH/GJR/EGARCH through `arch` (optional extra `market`), conditional EVT, CAViaR with the Taylor ES extension
  and the dynamic quantile statistic, the GAS one-factor, two-factor and hybrid FZ0 models, the FZ0 and tick losses,
  Diebold-Mariano and the Model Confidence Set, and the FRTB statistics (MAR32.9 table, zones, multipliers, desk limits, PLA
  Spearman, exact KS step count and zones).
- U8 back-tests (`riskvalidation.validation.market`): Kupiec, Christoffersen independence and conditional coverage, the
  duration test, the Basel traffic light, Acerbi-Szekely, Du-Escanciano unconditional and conditional, McNeil-Frey, the
  dynamic quantile test, the PLA test and the desk back-test, each a `TestResult` with its size checked by simulation.
- Documentation under `docs/market/` (equations, goldens, sources, decisions D-1 to D-9, limits) and the feature SDD
  `docs/design/features/u8-market/requirements.md`.
- U8-E review fixes: the age-weighted HS now has the two knots per return of Boudoukh, Richardson and Whitelaw (1998) and
  reproduces their worked example (VaR 2.63% and 2.34%); the dynamic quantile test takes a Monte Carlo p-value (`n_mc`, `seed`;
  size between 0.03 and 0.07 at (0.01, 250), (0.025, 250) and (0.01, 750)); the CI installs `.[dev,market]`; the ES-CAViaR is
  declared two-step (decision D-10, `caviar_es_fit`); the duration test censors the first spell only when the sample does not
  start with a hit and reports a capped shape; the PLA zone is decided in exact arithmetic (`frtb.compare_spearman`,
  `frtb.pla_zone_exact`).
- U10 IRRBB (U10-E): the new subpackage `riskvalidation.irrbb` with `curves` (zero rates, discount factors, the par-yield bootstrap, Nelson-Siegel, Svensson, the Diebold-Li dynamic Nelson-Siegel loadings, fit and AR(1) factor forecast), `shocks` (the six SRP31 shapes, the 2024 and 2016 size tables and the RAN 21-13 Annex 1 Hoja 9 table as data with their source, `shock_size` with its origin and the Contraste-estimate label for a currency without an entry, the post-shock floors, the EU floor), `slotting` (the 19 buckets of SRP31.96 Table 3, notional cash-flow slotting, the non-maturity deposit split with the SRP31 Table 4 caps applied and reported, the prepayment and term-deposit multipliers, SMM, CPR and PSA), `eve_nii` (the repricing and duration gaps, the standardised EVE and its risk measure, delta NII, the automatic-option add-on, the CLP and UF pair with the 50 percent offset on both sign branches, the Annex 1 delta NII and delta EVE, the 2.5 percent indexation charge, the outlier and prioritised-bank tests), `behavioural` (the static replicating portfolio as the global optimum, the weighted average maturity, the deposit beta, the prepayment logit and its maximum-likelihood fit) and `simulation` (the dynamic NII projection, Hull-White one-factor and dynamic Nelson-Siegel scenarios, EaR and EVaR); a positive delta EVE or delta NII is a loss and discounting is continuously compounded.
- U10 IRRBB (U10-E): the new module `riskvalidation.validation.irrbb` with seven validation tests that each return one `TestResult`: the SRP31.92 golden values, the RAN 21-13 Annex 1 formulas, the EVE sum-of-parts reconciliation, the NMD parameter stability across estimation windows, the EVE with and without behavioural assumptions (red whatever the size when the behavioural measure hides an outlier breach), the behavioural change at a limit breach and the out-of-sample NII check; a test that cannot be evaluated is `not_evaluated` with a null statistic, metric and p-value. The acceptance gates are Contraste internal gates, not regulatory thresholds.
- U10 IRRBB (U10-E): documentation pages under `docs/irrbb/` (curves, shocks, slotting, EVE and NII, Annex 1, behavioural models, simulation, the validation tests) with the formulas, the golden values, the sources and the UNVERIFIED items, and requirements `RV-1001` to `RV-1032` in `docs/design/features/u10-irrbb/requirements.md`, each naming its test. The engine core stays NumPy and SciPy; QuantLib is only an optional test cross-check of the discount factors, skipped when its wheel is not installed. No existing engine result changes.
- U10 IRRBB (U10-E review fixes): `simulation.ear_evar_mc_error` returns the Monte Carlo standard error of the EaR and of the EVaR at a stated level (batch means over contiguous batches, with an exact golden and a closed-form check), so the C14 documents carry an engine number; `eve_risk_measure` refuses a matrix without exactly six shock rows (a transposed matrix is no longer accepted silently); `forecast_dns_factors` keeps the innovation spread of a factor whose lagged series is flat; `cmf_annex1_goldens` counts a raising Annex 1 EVE function as a failed case and states the printed-precision tolerances of the two Hoja 10 values in its thresholds and policy version; four test gaps found by mutation are closed (the `cmf_pair` share on the first branch, the logit separation guard, the Hoja 10 tolerances).
- Operational risk (U12-E): `riskvalidation.oprisk` with the frequency and severity laws and their truncated maximum likelihood fits (lognormal, Weibull, GPD, the lognormal body and GPD tail splice, the tail-index profile), the compound annual loss by Monte Carlo (seeded, chunked), Panjer recursion and FFT with Shevchenko's benchmark reproduced (Table 2: H_5847 0.998999329, H_5849 0.999000217, quantile 5849), the quantile, expected shortfall and exceedance read from the pmf, the single loss approximation with its UNVERIFIED mean-corrected form (refused for an infinite mean), the scenario combination with the Shevchenko and Wuthrich Gamma-Pareto credibility, and the OPE25 standardised approach (business indicator, BIC 5.37bn at BI 35bn, loss component, ILM, the five-year BIC-only rule); contract `engine-oprisk-api` 1.0.0.
- Operational risk validation tests (U12-E): `riskvalidation.validation.oprisk` with the benchmark reproduction, the threshold stability of the tail index, the capital stability across independent histories, the SMA ILM response to one large loss and the scenario influence, each returning one `TestResult` (descriptive: no p-value, tolerances in `extras["thresholds"]`).
- Documentation: `docs/oprisk/` (five pages and an index) and the feature requirements `RV-1201` to `RV-1225` under `docs/design/features/u12-oprisk/`.
- Review fixes (U12-E): `es_for_severity` and the `infinite_mean` keyword of `es_from_pmf` refuse an expected shortfall when the severity has no finite mean on any grid (a coarse grid passes the missing-mass guard); `gamma_prior_from_scenarios` refuses points worth less than one expected event in `basis_years` and any refit on the search bound, and `fit_tail_to_scenarios` reports `converged = False` on a bound; `benchmark_reproduction` documents `moment_rel_tol` of at least 1e-5 for engine moments; the not_evaluated threshold-stability result counts infinite-mean fits; `severity_cdf` and `truncated_severity_cdf` refuse a NaN loss.

## [0.06.000] - 2026-10-13

Loss given default and exposure at default (Contraste unit U6): `riskvalidation.lgd_ead` (workout LGD, the LGD model rungs, downturn LGD, CCF and EAD by reference-date design, the IRB floors) and `riskvalidation.validation.lgd_ead` (LGD calibration and discrimination, downturn coverage, EAD back-testing by utilisation band, CCF estimator bias), each function reproducing the golden values of Contraste's `engine-lgd-ead-api` contract (reviewed: 74 of 74 recomputed independently).

- engine U6-E: new package `riskvalidation.lgd_ead` (workout LGD with the EU discount rate convention, range flags,
  incomplete workout treatment and the LGD distribution; the LGD rungs OLS, fractional logit, beta, hurdle and Tobit with
  `predict_lgd`; the cohort downturn add-on; the EU-convention CCF with three reference-date designs and four treatments,
  the direct EAD and utilisation change models; the IRB LGD and EAD floors, the partly secured floor formula documented for the caller and the
  supervisory CCFs) and the six tests of `riskvalidation.validation.lgd_ead`, reproducing every golden value of the contract
  `engine-lgd-ead-api` 1.0.0 (requirements RV-601 to RV-625, pages under `docs/lgd_ead/`).

## [0.05.000] - 2026-10-12

Provisions (Contraste unit U5): the IFRS 9 expected credit loss engine and its validation battery, and the Chilean
(CMF CNCB B-1) and Spanish (BdE Circular 4/2017 Annex 9) statutory provisions, implementing the provisions API
contract of CAOS_Contraste. Reviewed by Opus: every golden value and transcribed cell reproduces.

### Added

- Added `riskvalidation.provisions.term_structure`: `TermStructure` and the constructors `from_hazard`, `from_marginal` and
  `from_cumulative` (survival, marginal and cumulative PD identities to 1e-12), `competing_risks` (default incidence next to
  prepayment, never 1 minus the survival of both exits), `forward_pd`, `annualise_pd`, the one-factor `ttc_to_pit` and
  `implied_z`, `weighted_pit_pd` and the Gauss-Hermite cross check `gauss_hermite_pit_pd`.
- Added `riskvalidation.provisions.staging`: `StagingPolicy` (the threefold ratio, the 30 and 90 days past due thresholds and
  the probation are parameters, not constants), `sicr_quantitative`, `allocate_stages` with the low credit risk exemption
  suppressing only the two quantitative triggers, `apply_cure_probation` and the report-only `collective_transfer`.
- Added `riskvalidation.provisions.ecl`: `discount_factors`, `ecl_scenario`, `probability_weighted_ecl`,
  `non_linearity_index`, `weight_sensitivity`, `stage3_ecl`, `poci_allowance`, `simplified_matrix_ecl`, `commitment_ead` and
  `portfolio_allowance` (model ECL by the four stage keys; overlays added to the total only).
- Added `riskvalidation.provisions.overlays` (overlays as separate lines with a stage) and
  `riskvalidation.provisions.reconciliation` (the IFRS 7 35H and 35I roll-forwards, `check_identities` and
  `identity_residuals` on one tolerance basis, 1e-6 times max(1, G)).
- Added `riskvalidation.validation.ifrs9`: nine tests of dossier 02 I.2 (lifetime PD backtest with the Vasicek band, SICR
  effectiveness and threshold stability, stage migration and ECL backtests, scenario sensitivity, non-linearity, overlay
  diagnostics, reconciliation identities), each a `TestResult` with its reference, null hypothesis and policy parameters.
- Added documentation pages `docs/provisions/ecl/` and requirements RV-501 to RV-519 with golden values from the provisions API
  contract; the new modules are imported by path and are not re-exported from the package root.
- Engine review fixes: `portfolio_allowance` and `probability_weighted_ecl` refuse a negative model ECL for stages 1 to 3 (POCI may
  be negative); `lifetime_pd_backtest` refuses `n_events` above `n_at_risk` at an observed horizon; `reconciliation_identities`
  returns `not_evaluated` with its reason (never an infinite metric) when the waterfall has no opening or closing line or a stage
  lacks its stage ECL, and a movement without a waterfall line counts as a line of zero; the Vasicek band has a value assertion.
- Contract change (decision C2): `apply_cure_probation(stage_now, stage_previous, months_since_cure, policy, now)` takes the
  current `StageResult` as a required argument `now`; the trigger `unspecified` no longer exists, so every unchanged Stage 2 or 3
  account keeps a trigger of `TRIGGERS`. `riskvalidation.provisions.__init__` carries one docstring naming both halves, identical
  on both U5 branches.
- Added `riskvalidation.provisions.local_cl`: CMF CNCB Chapter B-1 (the individual categories A1 to B4, the default portfolio C1 to C6,
  the Circular 2.346 standard-method tables for mortgages, commercial leasing, student loans, generic commercial and consumer loans
  with every cell transcribed, the group provision with guarantor substitution, the 0.50% minimum on the individually assessed
  normal portfolio, the max rule and additional provisions as their own line). B-1 is kept apart from IFRS 9.
- Added `riskvalidation.provisions.local_es`: Banco de Espana Circular 4/2017 Annex 9 coverage tables (point 144 and point 140 as
  replaced by Circular 6/2021), the uncovered amount and the category-to-stage map.
- Added documentation pages `docs/provisions/local/` and requirements RV-550 to RV-567 with the transcribed tables as fixtures.
- Fix (engine review F1): `local_cl` and `local_es` refuse non-finite money, rates, buckets, days and amounts (`inf` as well as
  NaN) with `ValueError`; `b1_group_provision(inf, 0.02, ...)` no longer returns NaN.
- `riskvalidation.provisions.__init__` carries one docstring naming both halves, identical on both U5 branches (F6).

## [0.04.003] - 2026-10-07

### Fixed

- `generators.paths.RatingPaths` refuses a generator with a non-finite rate. Its checks compared entries, and every
  comparison is False on NaN: a NaN rate passed and made a draw run without end; an infinite rate balancing an
  infinite diagonal passed the row sums. Found by the adversarial review of Contraste's case C04.

### Documented

- `docs/transitions/03_markov-tests.md`: which form of the order test holds its size depends on the design, measured on
  two seven-grade chains (Contraste C04's S&P-like chain: the chi-square over-rejects, the likelihood ratio holds; a
  hand-made chain: the reverse at 5%). The default stays the chi-square, chosen on the size study's chain.

## [0.04.002] - 2026-10-07

### Added

- `harness.estimator_performance` reports the mean squared error with its Monte Carlo SE (Morris, White and Crowther
  2019, Table 6) and its root, whose Monte Carlo SE is the MSE's carried by the delta method (the package's step:
  Table 6 gives none for the root). Contraste's case C04 computed both itself.
- `transitions.intervals.exact_coverage(method, n, pd)`: the exact coverage and expected length of the Wald,
  Agresti-Coull or Jeffreys interval for a binomial count, by enumeration, for one or many true PDs; and
  `overlap_probability(method, n, pd_a, pd_b)`: the probability that two grades' intervals overlap, summed over every
  pair of counts. The engine's own tests enumerated the coverage by hand, and C04 re-implemented both.

## [0.04.001] - 2026-10-07

### Fixed

- `regulatory.capital_requirement` refused a PD of exactly 0 although it floors the PD at the regime's floor: the
  long-run average of a grade that never defaulted (an agency's AAA over CEREP's cohorts, in Contraste's case C04) is
  the case the floor exists for (CRE32.4), and every caller had to apply the floor itself, a second copy of the regime's
  value. A PD of 0 now takes the floor wherever one applies (the QRRE revolver floor for a revolver), with the usual
  note, and is still refused for a class the regime exempts from the floor (sovereigns under Basel III final, CRR3 and
  Basel II), where the formula has no value at 0. A negative PD, a PD of 1 and NaN are refused as before.

## [0.04.000] - 2026-10-06

### Added

- `riskvalidation.transitions`: rating transitions.
  - `estimators`: the cohort matrix (Anderson and Goodman's (2.9), pooled (2.8)) with withdrawals removed, the industry
    treatment, or kept as an absorbing state, as EBA/GL/2017/16 paragraph 76 keeps them in the default rate; the
    duration generator $K_{ij}/S_i$; the Aalen-Johansen product integral.
  - `embedding`: Israel, Rosenthal and Wei's log series with its convergence condition, the three non-existence
    conditions of their Theorem 3, the diagonal, weighted and JLT generators with the L1 distance of $\exp(Q)$ to $P$,
    and stochastic monotonicity.
  - `em`: the maximum likelihood generator of snapshot counts by EM, with Smith and dos Reis's closed-form E-step
    (Van Loan block exponentials).
  - `intervals`: Wald (2.2), Agresti-Coull (3.3) with the dependence correction (3.4) of Schuermann and Hanson, the
    Jeffreys interval, and the duration PD's bootstrap by resampling histories or parametric (Christensen, Hansen and
    Lando).
  - `mobility`: Jafry and Schuermann's $M_{SVD}$ and the trace index.
  - `ttc`: Engelmann's through-the-cycle portfolio (the fixed point of his (10) under Theorem 1, primitivity checked)
    and the projection (9) for one matrix or a stressed path.
- Four registered tests (family `transitions`, policy 2026.10.1): `rating.time_homogeneity`, `rating.markov_order`,
  `rating.matrix_reference` (Anderson and Goodman 1957) and `rating.momentum` (dos Reis, Pfeuffer and Smith 2020), each
  in its size study.
- `generators.paths.RatingPaths`: exact continuous-time rating paths from a generator, with dos Reis et al.'s
  self-exciting downgrade momentum (Ogata thinning), regimes, informative withdrawals and snapshots in ESMA CEREP's
  stock model; truth and verifier.
- Published results reproduced: Israel, Rosenthal and Wei (2001) section 4 and Engelmann (2024) section 4, read from
  the papers in the device data root (`RISKVALIDATION_DATA`, the tests skip without them: the matrices are agency data
  reprinted there); Schuermann and Hanson (2004) Table 5.
- Docs: the rating-transitions section (five pages), the rating-paths generator page, seven figures (six computed,
  theme-aware), `examples/rating_transitions.py` (run by the suite), and `tests/test_docs.py`, which checks every
  relative link, every figure and that every registered test is documented on a user-facing page.

### Changed

- The size study gives every test a fixed seed (`tests/test_size.py`, `SEEDS`): the seed used to depend on the
  number of tests sorted before it, so adding a test moved other tests' measured sizes, which their pages quote. The
  existing tests keep their seeds and measurements; `scripts/make_doc_figures.py` reads the same seeds and can draw
  named figures only.
- Each default form of the transition tests was chosen by its measured size on sparse rating tables (the
  time-homogeneity chi-square (3.8), the order chi-square on states grouped by direction, the reference likelihood
  ratio); the size study requires the test's page to state its measured size at 5% and at 1%.

## [0.03.002] - 2026-10-06

### Fixed

- `harness.wilson_interval` returns its bounds exactly at 0 and at n successes: the closed form's subtraction left a
  residue of about 1e-20 above a zero rate for some n (5.4e-20 at n = 4,000), so an interval did not contain its own
  rate (#18). The suite checks the containment for every k over seven values of n.

## [0.03.001] - 2026-10-06

### Fixed

- Docs: the reproduction of Yurdakul and Naranjo's Table 4 now states its small systematic offset (the harness sits
  about 0.7 standard errors above the paper on average, so how many of the 54 cells pass depends on the seed: 53 with
  the suite's, 50 with another), which "53 of 54" alone hid; the suite holds the mean z below 1.2. No code changed.

## [0.03.000] - 2026-10-05

### Added

- `riskvalidation.generators`: known-truth data-generating mechanisms, each a frozen configuration that serialises
  itself, draws one dataset per seed, returns its exact truth and verifies itself on its own stream:
  `DefaultCounts` (BCBS WP14's one-factor model with factor correlation across years; the default correlation
  reproduces WP14's examples), `BinormalScores` (exact AUC and AUC variance; decay, an estimated development AUC, a
  challenger, a leaked feature), `LogisticPortfolio` (true PDs, a planted calibration intercept and slope, an omitted
  quadratic term, covariate shift, one-factor dependence), `TwoSample` (Yurdakul and Naranjo's design, population or
  sample binning), `Migrations` and `decaying_matrix` with `break_monotonicity`, `GradeFrequencies` with `concentrate`
  (and a development sample for two-sample tests of the counts), and `NestedLda` (Demler et al.'s Framingham-like
  nested discriminants).
- `riskvalidation.harness`: `simulate` runs a registered test over repetitions with one independent stream per
  repetition (a test with its own randomness gets a fresh seed), counts undefined results, and reports rates per level
  and rule with the Monte Carlo SE of Morris, White and Crowther (2019) and the Wilson interval; `size_bound`,
  `within_size`, `agrees`, `agrees_with_published`; exact rejection probabilities for the count tests
  (`rejection_probability`, `critical_count`, `count_pmf`) and for the traffic-lights test with independent years;
  WP14's large-portfolio true confidence; `estimator_performance` (bias, empirical SE, model SE, coverage).
- Published simulations reproduced by the suite: BCBS WP14 Tables 7 and 8 (143 of 144 cells, 25,000 runs), Yurdakul
  and Naranjo (2020) Table 4 (53 of 54, 10,000 runs), Demler, Pencina and D'Agostino (2012) (1 rejection in 1,000);
  the two cells that do not reproduce are listed with their evidence.
- The size of every registered test with a p-value, measured under its null (`tests/test_size.py`).
- `disc.auc_vs_initial(auc_initial_variance=...)`: when the development AUC is itself an estimate, its variance is
  added to the statistic (an extension of the ECB formula, stated in the result); the ECB form rejects 11.6% of the
  time at 5% with equal development and current samples, the extended form 5.4%.
- `pd.traffic_lights(ties=...)`: the colour of a year on a boundary; `"lower"` (default) is WP14's printed mapping,
  `"upper"` the rule its own simulation must have used. `traffic_light_colours` and `traffic_profile_cdf` expose the
  colour rule and the profile table.
- Docs: the generators section (four pages), the measured sizes and the findings on every test family's page, five
  computed theme-aware figures, `examples/size_power.py` (run by the suite).

### Changed

- `pd.hosmer_lemeshow` uses as many degrees of freedom as groups by default (PDs given before the outcomes, the
  validation case) and G - 2 only with `fitted=True`. Measured on independent samples with the true PDs, G - 2
  rejects 11.7% of the time at the 5% level and G 5.1%. Results computed with the old default on validation samples
  had p-values that were too small.
- The traffic-lights p-value is a lookup in a cached profile table, and the normal test, the traffic lights and the
  PSI call SciPy's special functions directly: identical numbers, several times faster.
- A standardised default rate within 1e-12 of a traffic-lights boundary is on the boundary (it was coloured by
  rounding noise).

### Documented (found by the harness)

- The normal test over five years rejects 8.5% of the time at the 5% level, as WP14's own Table 7 shows, while WP14's
  summary calls it conservative.
- The chi-square test over grades exceeds its level when a grade expects under one default.
- The ECB AUC test treats the development AUC as known; estimated on a sample as large as the current one, it rejects
  10.9% of the time at the 5% level (the extension above holds the size).
- The ECB concentration statistic is bounded in the current CV: with ten grades and an initial CV of 0.6 it cannot
  reject at 5%, whatever the current distribution.
- The Jeffreys test's exact size for 1,000 obligors at PD 1% is 1.38% at the 1% level.

## [0.02.000] - 2026-10-05

### Added

- `riskvalidation.regulatory.irb`: the IRB risk-weight functions (corporate with the firm-size and
  financial-institution adjustments, sovereign, bank, HVCRE, SME, residential mortgage, QRRE, other retail), defaulted
  exposures, expected loss against eligible provisions, and four regimes as data with paragraph references and in-force
  dates: Basel III final, EU CRR3, Basel II (the 1.06 factor, 0.03% floors) and CMF RAN 21-6 (its own correlations, the
  61.3% consumer LGD). Reproduces all 144 cells of BCBS CRE99 Table 1 within 0.01 percentage points (141 exact).
- `riskvalidation.engines.vasicek`: the one-factor conditional PD, the large homogeneous portfolio's distribution,
  quantile and density, and the systematic factor implied by an observed default rate.
- `riskvalidation.engines.low_default`: most prudent estimation (Pluto and Tasche), independent and with asset
  correlation by Gauss-Hermite quadrature, scaled to a central tendency or to the portfolio's upper bound (section 5),
  and over several years with intertemporally correlated factors by seeded Monte Carlo (section 6). Reproduces the
  paper's Tables 1 to 14 and documents where its own values deviate (Table 4 at 75%, Table 8, Table 12, Tables 13
  and 14).
- `riskvalidation.engines.pd_curve`: the rating model's identities (Tasche 2013, section 3, theorem 3.3), quasi moment
  matching (appendix A), and every calibration approach of the paper: invariant default profile, invariant accuracy
  ratio, scaled PDs, scaled likelihood ratio (case 1), invariant likelihood ratio (cases 2 and 3), invariant PD curve
  and invariant conditional profiles by least squares or least chi-square (case 3), with improper models refused.
  Reproduces Tables 2 to 9 of the paper from the raw S&P counts.
- `pd.default_profile`: the chi-square test of the implied default profile with a seeded Monte Carlo p-value; it
  reproduces the p-values of Tasche (2013) Table 7. Policy threshold 5% and 1%, with its source.
- Docs: the regulatory and engines sections (four pages with the formulas from the primary sources, the
  reproductions and what not to claim), four computed and two hand-authored theme-aware figures, and
  `examples/credit_math.py`, run by the test suite.

### Fixed

- The QRRE PD floor of 0.10% applies to revolvers only; transactors take the retail 0.05% (CRE32.58), which is what
  CRE99 Table 1 tabulates.

## [0.01.000] - 2026-10-04

### Added

- Core contracts: the `TestResult` record with `Light` (green, amber, red, not evaluated, error), versioned
  `Policy` files with three threshold kinds, the test registry, `make_result`, a canonical inputs hash, and
  `run_suite`, which records a failing test and finishes the battery.
- Packaged default policy file, with no thresholds until the test catalogue lands.
- Guards: content standards (no em-dash, no emoji), CI budget (ADR-0074 shape), SDD gate (every requirement names
  a gate that exists).
- CI and the PyPI trusted-publishing workflow; the `docs/` wiki with the SDD, the U0 feature SDD and three
  architecture pages.
- 22 validation tests, each reproducing a published worked example or an independent reference:
  - PD calibration: Jeffreys per grade and portfolio (ECB 2019), binomial and Vasicek-corrected binomial with
    the BCBS WP14 critical counts (19, 35, 49, 63, 77), chi-square over grades, Hosmer-Lemeshow, Spiegelhalter,
    multi-period normal test, traffic lights by exact multinomial enumeration, Brier score with the Murphy
    decomposition, expected calibration error.
  - Discrimination: AUC with the ECB standard error and Gini, AUC against the development AUC (ECB 2019), the
    DeLong paired test, KS separation, a CAP accuracy ratio that equals 2 AUC - 1, a stratified bootstrap.
  - Rating system: matrix weighted bandwidth, migration z-tests with a Holm view, Herfindahl concentration test
    (ECB 2019).
  - Stability: PSI and CSI with the Yurdakul-Naranjo chi-square benchmark beside the conventional bands,
    chi-square homogeneity, two-sample KS, Wasserstein distance.
- Default policy thresholds (5% and 1%) for every p-value test, each with its stated source; descriptive
  measures stay unthresholded.
- Docs: the test catalogue and four family pages with formulas, null hypotheses, decision rules and primary
  sources; five figures, two computed by `scripts/make_doc_figures.py`.

### Changed

- Line length 120 for this repository.
