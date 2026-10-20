"""RV-1500: the catalogue lists every validation test of the eleven engine modules that were not registered before
(aggregation, ccr, fairness, ifrs9, irrbb, lgd_ead, liquidity, market, oprisk, portfolio, stress), with the identity,
hypothesis, reference and decision that the function itself writes on its row.

Three rules hold the catalogue complete:

* the inventory below is every test id a public function of those modules writes (found by reading the source, so a
  new function cannot be forgotten), and each id has a spec whose family is the module;
* every `TestResult` an engine function builds during the whole test session names a registered test and carries the
  `h0` and the `reference` of its spec (the session-wide check, which runs after every other test; the recording is in
  `conftest.py`, never in `src/`);
* the size rule of `test_size.py` and the docs rule of `test_docs.py` read the catalogue, so a registered id is
  measured or named there.

`validation/ml.py` registers in its own unit and is not part of this inventory."""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from conftest import PRODUCED
from riskvalidation.core.registry import catalogue, spec

# module -> the id prefix its functions write; the family of a registered test is the module's name
PREFIX = {
    "aggregation": "aggregation", "ccr": "ccr", "fairness": "fair", "ifrs9": "ifrs9", "irrbb": "irrbb",
    "lgd_ead": "lgd_ead", "liquidity": "liq", "market": "market", "oprisk": "oprisk", "portfolio": "portfolio",
    "stress": "stress",
}

# test id -> (module, function, decision as the function's own code decides it): `p_value` where the light is decided
# by a p-value against alphas, `metric` otherwise (a p-value carried beside a metric gate, as `lgd_discrimination` does
# beside its rank-correlation gates, does not decide the light)
INVENTORY = {
    # aggregation
    "aggregation.simple_sum_and_variance_covariance": ("aggregation", "simple_sum_and_variance_covariance", "metric"),
    "aggregation.copula_choice_sensitivity": ("aggregation", "copula_choice_sensitivity", "metric"),
    "aggregation.worst_case_bracket": ("aggregation", "worst_case_bracket", "metric"),
    "aggregation.correlation_stress": ("aggregation", "correlation_stress", "metric"),
    "aggregation.diversification_sign": ("aggregation", "diversification_sign", "metric"),
    "aggregation.horizon_confidence_consistency": ("aggregation", "horizon_confidence_consistency", "metric"),
    "aggregation.elliptical_mc_agreement": ("aggregation", "elliptical_mc_agreement", "metric"),
    "aggregation.rosenberg_schuermann_reference": ("aggregation", "rosenberg_schuermann_reference", "metric"),
    # ccr
    "ccr.ee_analytic_vs_simulated": ("ccr", "ee_analytic_vs_simulated", "metric"),
    "ccr.martingale_discounted_bonds": ("ccr", "martingale_discounted_bonds", "metric"),
    "ccr.eepe_recomputed": ("ccr", "eepe_recomputed", "metric"),
    "ccr.cva_recomputed": ("ccr", "cva_recomputed", "metric"),
    "ccr.path_convergence": ("ccr", "path_convergence", "metric"),
    "ccr.reconcile_ee_profile": ("ccr", "reconcile_ee_profile", "metric"),
    "ccr.pfe_exceedance_backtest": ("ccr", "pfe_exceedance_backtest", "p_value"),
    "ccr.pit_exposure_backtest": ("ccr", "pit_exposure_backtest", "metric"),
    # fairness
    "fair.demographic_parity": ("fairness", "demographic_parity", "metric"),
    "fair.adverse_impact": ("fairness", "adverse_impact", "metric"),
    "fair.two_proportion_z": ("fairness", "two_proportion_z", "p_value"),
    "fair.fisher_exact": ("fairness", "fisher_exact", "p_value"),
    "fair.equalized_odds": ("fairness", "equalized_odds", "metric"),
    "fair.equal_opportunity": ("fairness", "equal_opportunity", "metric"),
    "fair.calibration_within_groups": ("fairness", "calibration_within_groups", "metric"),
    "fair.auc_by_group": ("fairness", "auc_by_group", "metric"),
    "fair.conditional_disparity": ("fairness", "conditional_disparity", "metric"),
    "fair.rate_spread_disparity": ("fairness", "rate_spread_disparity", "metric"),
    "fair.replication": ("fairness", "replication", "metric"),
    "fair.situation_testing": ("fairness", "situation_testing", "metric"),
    # ifrs9
    "ifrs9.lifetime_pd_backtest": ("ifrs9", "lifetime_pd_backtest", "metric"),
    "ifrs9.sicr_effectiveness": ("ifrs9", "sicr_effectiveness", "metric"),
    "ifrs9.sicr_threshold_stability": ("ifrs9", "sicr_threshold_stability", "metric"),
    "ifrs9.stage_migration_backtest": ("ifrs9", "stage_migration_backtest", "p_value"),
    "ifrs9.ecl_backtest": ("ifrs9", "ecl_backtest", "metric"),
    "ifrs9.scenario_sensitivity": ("ifrs9", "scenario_sensitivity", "metric"),
    "ifrs9.nonlinearity": ("ifrs9", "nonlinearity_test", "metric"),
    "ifrs9.overlay_diagnostics": ("ifrs9", "overlay_diagnostics", "metric"),
    "ifrs9.reconciliation_identities": ("ifrs9", "reconciliation_identities", "metric"),
    # irrbb
    "irrbb.srp31_92_goldens": ("irrbb", "srp31_92_goldens", "metric"),
    "irrbb.cmf_annex1_goldens": ("irrbb", "cmf_annex1_goldens", "metric"),
    "irrbb.eve_sum_of_parts": ("irrbb", "eve_sum_of_parts", "metric"),
    "irrbb.nmd_parameter_stability": ("irrbb", "nmd_parameter_stability", "metric"),
    "irrbb.eve_with_without_behavioural": ("irrbb", "eve_with_without_behavioural", "metric"),
    "irrbb.behavioural_change_at_breach": ("irrbb", "behavioural_change_at_breach", "metric"),
    "irrbb.nii_out_of_sample": ("irrbb", "nii_out_of_sample", "metric"),
    # lgd_ead
    "lgd_ead.lgd_calibration_by_segment": ("lgd_ead", "lgd_calibration_by_segment", "p_value"),
    "lgd_ead.lgd_calibration_over_time": ("lgd_ead", "lgd_calibration_over_time", "p_value"),
    "lgd_ead.lgd_discrimination": ("lgd_ead", "lgd_discrimination", "metric"),
    "lgd_ead.downturn_coverage": ("lgd_ead", "downturn_coverage", "p_value"),
    "lgd_ead.ead_backtest_by_band": ("lgd_ead", "ead_backtest_by_band", "metric"),
    "lgd_ead.ccf_estimator_bias": ("lgd_ead", "ccf_estimator_bias", "metric"),
    # liquidity
    "liq.list_compounding_golden": ("liquidity", "list_compounding_golden", "metric"),
    "liq.survival_monotone_in_severity": ("liquidity", "survival_monotone_in_severity", "metric"),
    "liq.cbc_haircut_rules": ("liquidity", "cbc_haircut_rules", "metric"),
    "liq.nlp_identity": ("liquidity", "nlp_identity", "metric"),
    "liq.runoff_adequacy_back_test": ("liquidity", "runoff_adequacy_back_test", "metric"),
    "liq.runoff_vs_generator_truth": ("liquidity", "runoff_vs_generator_truth", "metric"),
    "liq.cliff_effect_diagnostic": ("liquidity", "cliff_effect_diagnostic", "metric"),
    "liq.htm_loss_effect": ("liquidity", "htm_loss_effect", "metric"),
    "liq.uninsured_share_sensitivity": ("liquidity", "uninsured_share_sensitivity", "metric"),
    "liq.reverse_stress_round_trip": ("liquidity", "reverse_stress_round_trip", "metric"),
    "liq.htm_revaluation_vs_reported": ("liquidity", "htm_revaluation_vs_reported", "metric"),
    # market
    "market.kupiec_pof": ("market", "kupiec_pof", "p_value"),
    "market.christoffersen_independence": ("market", "christoffersen_independence", "p_value"),
    "market.christoffersen_conditional_coverage": ("market", "christoffersen_conditional_coverage", "p_value"),
    "market.duration_test": ("market", "duration_test", "p_value"),
    "market.basel_traffic_light": ("market", "basel_traffic_light", "metric"),
    "market.acerbi_szekely": ("market", "acerbi_szekely", "p_value"),
    "market.du_escanciano_unconditional": ("market", "du_escanciano_unconditional", "p_value"),
    "market.du_escanciano_conditional": ("market", "du_escanciano_conditional", "p_value"),
    "market.mcneil_frey_residual_test": ("market", "mcneil_frey_residual_test", "p_value"),
    "market.dynamic_quantile_test": ("market", "dynamic_quantile_test", "p_value"),
    "market.pla_test": ("market", "pla_test", "metric"),
    "market.desk_backtest": ("market", "desk_backtest", "metric"),
    # oprisk
    "oprisk.benchmark_reproduction": ("oprisk", "benchmark_reproduction", "metric"),
    "oprisk.threshold_stability": ("oprisk", "threshold_stability", "metric"),
    "oprisk.capital_stability": ("oprisk", "capital_stability", "metric"),
    "oprisk.sma_ilm_response": ("oprisk", "sma_ilm_response", "metric"),
    "oprisk.scenario_influence": ("oprisk", "scenario_influence", "metric"),
    # portfolio
    "portfolio.tail_backtest": ("portfolio", "tail_backtest", "p_value"),
    "portfolio.mc_convergence": ("portfolio", "mc_convergence", "p_value"),
    "portfolio.euler_additivity": ("portfolio", "euler_additivity", "metric"),
    "portfolio.crp_recursion_vs_mc": ("portfolio", "crp_recursion_vs_mc", "p_value"),
    "portfolio.correlation_size_power": ("portfolio", "correlation_size_power", "metric"),
    # stress
    "stress.satellite_backcast": ("stress", "satellite_backcast", "metric"),
    "stress.sensitivity_monotonicity": ("stress", "sensitivity_monotonicity", "metric"),
    "stress.static_balance_sheet_rules": ("stress", "static_balance_sheet_rules", "metric"),
    "stress.bottom_up_vs_top_down": ("stress", "bottom_up_vs_top_down", "metric"),
    "stress.scenario_severity": ("stress", "scenario_severity", "metric"),
    "stress.reverse_solution_check": ("stress", "reverse_solution_check", "metric"),
}
assert len(INVENTORY) == 89

# ids whose row text is built from arguments: the spec holds the text with `{...}` where the argument goes, and a row
# matches when the rest is equal and the argument is a number
PARAMETERISED = {
    "ccr.pfe_exceedance_backtest": ("h0",),
}
# ids whose function writes a second, fixed text on another branch: the spec holds the first, the others are listed
H0_VARIANTS = {
    "liq.reverse_stress_round_trip": {
        "the solved multiplier is the threshold: the target is reached at it and not just below it"},
}

SRC = Path(__file__).resolve().parents[1] / "src" / "riskvalidation" / "validation"
ID = re.compile(r"^[a-z_0-9]+\.[a-z_0-9]+$")


def _import_all() -> None:
    for module in PREFIX:
        importlib.import_module(f"riskvalidation.validation.{module}")


def _written_ids(module: str) -> dict[str, str]:
    """Id -> public function, for every string constant of a public top-level function that has the module's prefix."""
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_"):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str) and ID.match(node.value)
                    and node.value.split(".")[0] == PREFIX[module]):
                found[node.value] = fn.name
    return found


def test_inventory_is_every_id_the_modules_write():
    found = {i: (m, fn) for m in PREFIX for i, fn in _written_ids(m).items()}
    listed = {i: (m, fn) for i, (m, fn, _d) in INVENTORY.items()}
    assert found == listed, sorted(set(found) ^ set(listed))


def test_every_inventory_id_is_registered_with_its_family_function_and_decision():
    _import_all()
    problems = []
    for tid, (module, fname, decision) in sorted(INVENTORY.items()):
        try:
            s = spec(tid)
        except KeyError:
            problems.append(f"{tid}: not registered")
            continue
        func = getattr(importlib.import_module(f"riskvalidation.validation.{module}"), fname)
        if s.family != module:
            problems.append(f"{tid}: family {s.family!r}, expected {module!r}")
        if s.func is not func:
            problems.append(f"{tid}: registered on {s.func.__qualname__}, expected {fname}")
        if s.decision != decision:
            problems.append(f"{tid}: decision {s.decision!r}, expected {decision!r}")
        if not (s.name.strip() and s.h0.strip() and s.reference.strip()):
            problems.append(f"{tid}: empty name, h0 or reference")
    assert not problems, problems


def test_the_catalogue_holds_the_inventory_beside_the_older_families():
    _import_all()
    ids = {s.test_id for s in catalogue()}
    assert set(INVENTORY) <= ids
    assert len(set(INVENTORY)) == 89
    for module in PREFIX:
        assert {s.test_id for s in catalogue(module)} == {i for i, v in INVENTORY.items() if v[0] == module}, module
    # no family of the inventory reaches an older module's ids, and the older 27 tests are still there
    older = {s.test_id for s in catalogue() if s.family not in PREFIX and s.family != "selftest"}
    assert len(older) >= 27 and not older & set(INVENTORY)


def _matches(text: str, spec_text: str, parameterised: bool) -> bool:
    if text == spec_text:
        return True
    if not parameterised or "{" not in spec_text:
        return False
    pattern = re.escape(spec_text)
    pattern = re.sub(r"\\{[^}]*\\}", r"[-+0-9.eE]+", pattern)
    return re.fullmatch(pattern, text) is not None


def _exercise_one_engine_function() -> None:
    """So that this file records something even when it runs alone."""
    from riskvalidation.validation.market import basel_traffic_light

    basel_traffic_light(np.zeros(250, dtype=bool))


def test_session_wide_check_is_wired():
    _import_all()
    before = sum(PRODUCED.values())
    _exercise_one_engine_function()
    assert sum(PRODUCED.values()) > before
    assert any(k[0] == "market.basel_traffic_light" for k in PRODUCED)


def test_every_result_names_a_registered_test_session_wide():
    """Every `TestResult` that any engine function built during this session names a registered test and carries the
    `h0` and the `reference` of its spec. Runs after every other test (conftest.py), so it sees all of them."""
    _import_all()
    unregistered, mismatched = set(), []
    for tid, h0, reference in sorted(PRODUCED):
        try:
            s = spec(tid)
        except KeyError:
            unregistered.add(tid)
            continue
        h0_ok = _matches(h0, s.h0, tid in PARAMETERISED) or h0 in H0_VARIANTS.get(tid, set())
        ref_ok = _matches(reference, s.reference, False)
        if not (h0_ok and ref_ok):
            mismatched.append((tid, "h0" if not h0_ok else "reference", h0 if not h0_ok else reference))
    assert not unregistered, f"unregistered test ids: {sorted(unregistered)}"
    assert not mismatched, f"h0 or reference differ from the spec: {mismatched}"


def test_parameterised_and_variant_ids_are_registered_and_listed():
    _import_all()
    for tid in PARAMETERISED:
        assert "{" in spec(tid).h0, tid
    for tid, variants in H0_VARIANTS.items():
        assert variants and spec(tid).h0 not in variants
    assert (set(PARAMETERISED) | set(H0_VARIANTS)) <= set(INVENTORY)


def test_importing_the_validation_package_alone_registers_the_inventory():
    """A consumer that imports only `riskvalidation.validation` can call `spec(test_id)` for any row (a fresh
    interpreter, with no optional dependency of the extras imported)."""
    code = (
        "import sys, riskvalidation.validation; "
        "from riskvalidation.core.registry import catalogue; "
        "print(len(catalogue())); "
        "print(sorted(m for m in ('arch', 'statsmodels', 'pandas', 'fairlearn', 'mapie') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.splitlines()
    assert int(out[0]) == len(INVENTORY) + 27
    assert out[1] == "[]"


def test_the_row_text_comparison_is_strict():
    template = spec("ccr.pfe_exceedance_backtest").h0
    assert _matches("the PFE forecast at 0.95 is exceeded with frequency 0.05", template, True)
    assert not _matches("the PFE forecast at 0.95 is exceeded with frequency", template, True)
    assert not _matches("the PFE forecast at 0.95 is exceeded with frequency 0.05", template, False)
    assert not _matches("another text", "the text", True)
