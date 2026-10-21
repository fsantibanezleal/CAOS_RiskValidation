# The public API and its policy

From 1.0.0 `riskvalidation` follows semantic versioning, and the public API below is what that promise covers. The
page is tested: `tests/test_public_api.py` fails when a subpackage exports a name this page does not list, when this
page lists a name the code does not export, when a listed name cannot be imported or has no docstring, and when
`from riskvalidation.<subpackage> import *` imports anything but `__all__`.

## Semantic versioning

The version is `MAJOR.MINOR.PATCH`.

- A **patch** release fixes a defect and changes no public name, signature, result field or documented behaviour
  beyond the defect.
- A **minor** release adds: a name, a function, a keyword argument with a default, a field of a result, a test of the
  catalogue. It deprecates, it never removes.
- A **major** release may remove a deprecated name, change a signature, or change what a documented function computes.
  The CHANGELOG states every such change under its own heading, and the engine API contract of the subpackage takes a
  major version in step.

A number that a function computes is part of its behaviour: a change of a published value (a corrected constant, a
changed convention) is announced in the CHANGELOG with the old and the new value, and is a major when a documented
output of a public function changes beyond its stated tolerance.

## Deprecation

A public name is never removed in the release that announces it. The rule:

1. A minor release deprecates the name. It keeps working and emits a `DeprecationWarning` that names the version that
   deprecated it, the major that removes it and the replacement. The helper is `deprecated(since, removal, use)` in
   `riskvalidation.core.deprecation`; it refuses a `removal` that is not a later major `X.0.0`.
2. The name stays through at least one minor release: the removal is in a later major than the one that deprecated it.
3. The documentation page of the name and the CHANGELOG carry the deprecation from the day it lands.

Nothing is deprecated in 1.0.0. The one signature change the freeze allowed, the move of `p_enter_stress` and
`p_exit_stress` from `DepositorSegment` to `DepositorConfig` (the liquidity depositor generator, BL-082), happened before
the freeze, in the release that cut 1.0.0, and is recorded in the CHANGELOG and in the liquidity API contract.

## What the public API is

1. The top-level exports of `riskvalidation` (the first section below).
2. Every name in a subpackage's `__all__` (the sections below, one per subpackage). A name that is a module is public by
   path: `from riskvalidation.liquidity import runoff`. The functions and classes of such a module whose names do not
   start with an underscore are the public functions that the `engine-<subpackage>-api` contract of the subpackage states.
3. The documented fields of the result records and the policy file format (`TestResult`, `Policy`, the packaged
   `default.json` keys), which are versioned by the CHANGELOG as above.

The package ships `py.typed` (PEP 561): the annotations of the public API are part of it, and a change that makes an
annotation incompatible is a signature change.

## Bilingual reasons (additive in 1.0.0)

The governance functions `scope_test`, `assign_variant`, `anchor_from_test_row`, `rwea_share_of_anchor` and `resolve_policy`
return a reason as an English string, as the `engine-governance-api` contract states, and that string is part of the
public API. The Spanish form is added beside them, by one pattern: `riskvalidation.governance.reasons.bilingual(english)`
takes the English reason a function returned and returns `{"en": ..., "es": ...}`, in RAE orthography. The English reason is
the key: fixed sentences are in `REASONS_ES`, sentences that carry a value (a source list, a label, a test id) are matched by
`TEMPLATES`, and a reason in neither raises `ValueError`. Adding the Spanish of a new reason is a minor change; changing an
existing English reason is a major one, because it is the key.

## `riskvalidation`

The top-level exports: the result record, the policy, the registry and the suite runner, the test catalogue and the version.

Modules, imported by path (`from riskvalidation import <module>`): `validation`.

Names, imported from the package (`from riskvalidation import <name>`): `Light`, `Policy`, `SuiteRun`, `TestResult`, `TestSpec`, `Threshold`, `__version__`, `catalogue`, `default_policy`, `load_policy`, `run_suite`.

## `riskvalidation.aggregation`

Risk aggregation: the ladder from the simple sum to the worst-case bounds, the dependence band and model risk.

Modules, imported by path (`from riskvalidation.aggregation import <module>`): `band`, `bounds`, `copulas`, `model_risk`, `simple`.

## `riskvalidation.ccr`

Counterparty credit risk: Hull-White paths, swap exposure, collateral, CVA, DVA and wrong-way risk.

Modules, imported by path (`from riskvalidation.ccr import <module>`): `collateral`, `cva`, `exposure`, `hw1f`, `swaps`.

## `riskvalidation.core`

The core contracts: the result record, the policy, the registry, hashing, the suite runner and the deprecation helper.

Modules, imported by path (`from riskvalidation.core import <module>`): `build`, `deprecation`, `hashing`, `policy`, `registry`, `result`, `suite`.

## `riskvalidation.engines`

The reference engines: the one-factor model, most prudent estimation for low-default portfolios and PD curve calibration.

Modules, imported by path (`from riskvalidation.engines import <module>`): `low_default`, `pd_curve`, `vasicek`.

## `riskvalidation.fairness`

Fairness: group metrics, inference on selection rates, conditional disparity, replication and situation testing.

Modules, imported by path (`from riskvalidation.fairness import <module>`): `conditional`, `inference`, `metrics`, `plants`, `policy`, `replication`, `situation`.

## `riskvalidation.generators`

The known-truth generators and the helpers that feed their data to the validation tests.

Names, imported from the package (`from riskvalidation.generators import <name>`): `DEMLER`, `HANLEY_MCNEIL`, `WP14`, `YURDAKUL_NARANJO`, `BinormalScores`, `Check`, `DefaultCounts`, `GeneratorBase`, `GradeFrequencies`, `LogisticPortfolio`, `Migrations`, `NestedLda`, `TwoSample`, `binormal_auc`, `binormal_auc_variance`, `break_monotonicity`, `bvn_equal`, `concentrate`, `decaying_matrix`, `default_correlation`, `feed_cell`, `feed_grades`, `feed_portfolio`, `feed_series`, `joint_default_probability`, `jsonable`, `lda_auc`, `monotone_cells`, `population_psi`.

## `riskvalidation.governance`

Model governance: the scope test, the inventory, tiering, lifecycle, findings, review schedule, opinion, risk appetite metrics and the validation report. Every tier count, band and deadline is a stated convention.

Modules, imported by path (`from riskvalidation.governance import <module>`): `appetite`, `findings`, `inventory`, `lifecycle`, `opinion`, `policy`, `reasons`, `report`, `schedule`, `scope`, `tiering`, `conventions`.

Names, imported from the package (`from riskvalidation.governance import <name>`): `CONVENTIONS`.

## `riskvalidation.harness`

The size and power harness: simulation, rejection probabilities, exact critical values and the agreement rules.

Names, imported from the package (`from riskvalidation.harness import <name>`): `COUNT_TESTS`, `MORRIS`, `Rate`, `Rule`, `Simulation`, `agrees`, `agrees_with_published`, `count_pmf`, `critical_count`, `estimator_performance`, `light_not_green`, `metric_above`, `p_below`, `rejection_probability`, `simulate`, `size_bound`, `traffic_lights_rejection_probability`, `wilson_interval`, `within_size`, `wp14_true_confidence`.

## `riskvalidation.irrbb`

Interest-rate risk in the banking book: curves, shocks, slotting, EVE and NII, the behavioural models and the simulation.

Modules, imported by path (`from riskvalidation.irrbb import <module>`): `behavioural`, `curves`, `eve_nii`, `shocks`, `simulation`, `slotting`.

## `riskvalidation.lgd_ead`

LGD and EAD: workout LGD, the LGD rungs, downturn LGD, CCF and the IRB floors.

Modules, imported by path (`from riskvalidation.lgd_ead import <module>`): `ccf`, `downturn`, `floors`, `lgd_models`, `workout`.

## `riskvalidation.liquidity`

Liquidity: run-off, the counterbalancing capacity, ladders, scenarios, reverse stress and the depositor generator.

Modules, imported by path (`from riskvalidation.liquidity import <module>`): `cbc`, `deposits`, `ladder`, `reverse`, `runoff`, `scenarios`.

## `riskvalidation.market`

Market risk: VaR and ES estimators, the scoring and comparison of forecasts and the FRTB statistics.

Modules, imported by path (`from riskvalidation.market import <module>`): `caviar`, `conventions`, `evt`, `frtb`, `garch`, `gas_fz`, `historical`, `parametric`, `scoring`.

## `riskvalidation.ml`

ML model validation: robustness, stability, explanations, effects, generalisation, conformal prediction, leakage and data quality.

Modules, imported by path (`from riskvalidation.ml import <module>`): `cbpe`, `challenger`, `conformal`, `dataquality`, `effects`, `explanations`, `generalisation`, `leakage`, `plants`, `policy`, `robustness`, `seeds`.

## `riskvalidation.oprisk`

Operational risk: the loss distribution approach (distributions, aggregation, the single loss approximation, scenarios) and the standardised approach (SMA).

Modules, imported by path (`from riskvalidation.oprisk import <module>`): `aggregate`, `distributions`, `scenarios`, `sla`, `sma`.

## `riskvalidation.policies`

The packaged policy data. It exports no Python name; the policy is loaded through `riskvalidation.core.policy`.

The list is empty.

## `riskvalidation.portfolio`

Portfolio credit capital: the loss distribution of a credit portfolio by the rungs of dossier 04 and its Euler contributions.

Modules, imported by path (`from riskvalidation.portfolio import <module>`): `asrf_fit`, `copula`, `creditrisk_plus`, `euler`, `granularity`, `importance`, `portfolio`.

## `riskvalidation.provisions`

IFRS 9 expected credit loss and the local statutory provision regimes (CMF B-1 of Chile, Banco de Espana Circular 4/2017).

Modules, imported by path (`from riskvalidation.provisions import <module>`): `ecl`, `local_cl`, `local_es`, `overlays`, `reconciliation`, `staging`, `term_structure`.

## `riskvalidation.regulatory`

The regulatory calculators: published formulas that turn parameters into capital, provisions or liquidity metrics. The IRB names are exported by name; the other calculators are modules.

Modules, imported by path (`from riskvalidation.regulatory import <module>`): `ba_cva`, `frtb`, `irb`, `lcr_nsfr`, `output_floor`, `sa_ccr`, `standardised`.

Names, imported from the package (`from riskvalidation.regulatory import <name>`): `REGIMES`, `IRBResult`, `Regime`, `capital_requirement`, `correlation`, `el_versus_provisions`.

## `riskvalidation.stress`

Macro stress testing: satellites, scenario paths, the capital engine, reverse stress and the scenario models.

Modules, imported by path (`from riskvalidation.stress import <module>`): `capital`, `paths`, `reverse`, `satellites`, `scenario_models`.

## `riskvalidation.transitions`

Rating transitions: estimators, the embedding problem and the EM generator, intervals, mobility and the through-the-cycle portfolio.

Modules, imported by path (`from riskvalidation.transitions import <module>`): `em`, `embedding`, `estimators`, `intervals`, `mobility`, `ttc`.

## `riskvalidation.validation`

The validation tests, one TestResult function each, grouped by family modules; the family modules are public by path and the test functions by name.

Modules, imported by path (`from riskvalidation.validation import <module>`): `calibration`, `discrimination`, `governance`, `rating_system`, `stability`, `transitions`.

Names, imported from the package (`from riskvalidation.validation import <name>`): `bin_counts`, `bootstrap_ci`, `cap_accuracy_ratio`, `critical_default_count`, `disc_auc`, `disc_auc_vs_initial`, `disc_delong`, `disc_ks`, `pd_binomial`, `pd_binomial_vasicek`, `pd_brier`, `pd_chi2_grades`, `pd_default_profile`, `pd_ece`, `pd_hosmer_lemeshow`, `pd_jeffreys`, `pd_jeffreys_grades`, `pd_normal_multiperiod`, `pd_spiegelhalter`, `pd_traffic_lights`, `psi_benchmark`, `quantile_edges`, `rating_hhi`, `rating_migration_ztests`, `rating_mwb`, `stability_chi2`, `stability_csi`, `stability_ks`, `stability_psi`, `stability_wasserstein`, `rating_markov_order`, `rating_matrix_reference`, `rating_momentum`, `rating_time_homogeneity`.

## What is not public

- Any name that starts with an underscore, and the private modules `_checks`, `_logistic` and `_base` of a subpackage.
- Any module or name that is not in the `__all__` of its subpackage and is not stated by the engine API contract of the
  subpackage, even when it can be imported.
- The `scripts/`, `examples/` and `tests/` folders, and the files under `docs/design/`.
- The exact text of an error message, beyond the field it names, and the order of keys in a JSON document.
- The optional extras (`market`, `ml`, `stress`, `transitions`): the third-party libraries they install are used by the
  tests and by the functions that import them inside, and their versions are pinned by the extra, not promised by the API.
