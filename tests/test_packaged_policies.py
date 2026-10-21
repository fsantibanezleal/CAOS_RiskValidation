"""The packaged Contraste battery policies (unit U16-E3, decisions D16-16 and D16-17, backlog BL-085).

One versioned policy file per Contraste battery, `policies/contraste-<name>-<version>.json`, label `contraste-<name>
<version>`. Every cut point of every committed inline battery string resolves to an entry, and the entry (through its
`rule` and its keyed case) reproduces the light of every committed row. The committed-document tests read the
Contraste case documents read-only and are skipped when `CONTRASTE_ROOT` is unset.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import math
import os
from collections import Counter
from importlib import resources
from pathlib import Path

import pytest

from riskvalidation.core.policy import (
    Threshold,
    default_policy,
    load_policy,
    packaged_labels,
    packaged_policy,
    policy_from_mapping,
)
from riskvalidation.core.result import Light
from riskvalidation.policies.rules import RULES, classify_rule

BATTERY_LABELS = [
    "contraste-aggregation-battery 1.0.0",
    "contraste-c12 1.0.0",
    "contraste-ccr-battery 1.0.0",
    "contraste-ifrs9-battery 1.0.0",
    "contraste-irrbb-battery 1.0.0",
    "contraste-lgd-ead-battery 1.0.0",
    "contraste-liquidity-battery 1.0.0",
    "contraste-market-battery 1.0.0",
    "contraste-oprisk-battery 1.0.0",
    "contraste-portfolio-battery 1.0.0",
    "contraste-stress-battery 1.0.0",
]
# sha256 of policies/default.json at 0.09.000: the default policy is unchanged by this unit
DEFAULT_SHA256 = "0f930dff38c6fe6208c882d9aeb3e63e53ef8a82ed4db210dcf03315b95959ed"

# the keys of an inline string that are cut points of the light; the others are the test's own arguments (quantile,
# n_sim, seed, lags, compared_at, rho, estimator, ...), measured numbers of the case or descriptions: they stay in
# the row
CUT_KEYS = frozenset(
    "alpha_amber alpha_red amber red green tolerance tolerance_days rel_tol k_se gap_amber gap_red diff_amber diff_red "
    "err_green err_amber ratio_green ratio_amber se_green se_red rho_green rho_amber dispersion_green dispersion_amber "
    "bias_green bias_amber jump_green jump_amber influence_amber influence_red tol_green tol_amber moment_rel_tol "
    "quantile_rel_tol peak_amber peak_red timing_quarters minimum z low high expected_horizon expected_level nl_amber "
    "nl_red r_green r_amber share_amber share_red green_ratio amber_ratio direct_share_amber amber_count "
    "red_share_of_periods gauss_hermite_tolerance spearman_green spearman_red ks_green ks_red power_green power_amber "
    "limit tolerance_scalar tolerance_bp".split()
)


# ---------------------------------------------------------------------------------------------------------------
# the files


def test_packaged_labels_list_the_default_and_the_eleven_batteries():
    assert set(packaged_labels()) == {default_policy().label, *BATTERY_LABELS}


@pytest.mark.parametrize("label", BATTERY_LABELS)
def test_every_packaged_file_loads_and_has_its_label(label):
    policy = packaged_policy(label)
    assert policy.label == label
    name = label.replace(" ", "-") + ".json"
    path = Path(str(resources.files("riskvalidation.policies").joinpath(name)))
    assert path.is_file() and load_policy(path).label == label
    assert policy.thresholds, "a battery policy holds at least one entry"


@pytest.mark.parametrize("label", BATTERY_LABELS)
def test_every_entry_states_its_convention_source_and_a_known_rule(label):
    for tid, th in packaged_policy(label).thresholds.items():
        assert th.source.startswith("convention:"), tid
        assert "dossier reference:" in th.source and "pipeline/cases/" in th.source, tid
        assert "is regulatory" not in th.source, tid
        if th.rule:
            assert th.rule.partition(":")[0] in RULES, (tid, th.rule)
        elif th.kind != "p_value":
            pytest.fail(f"{tid}: a non-p-value entry names its rule")
        if th.by:
            assert th.rule and th.cases, tid


def test_packaged_policy_raises_keyerror_naming_the_label_and_the_packaged_labels():
    with pytest.raises(KeyError) as exc:
        packaged_policy("contraste-nothing-battery 9.9.9")
    text = str(exc.value)
    assert "contraste-nothing-battery 9.9.9" in text and "contraste-market-battery 1.0.0" in text
    for bad in ("", "contraste-market-battery", "../default 1", "contraste-market-battery 2.0.0"):
        with pytest.raises(KeyError):
            packaged_policy(bad)


def test_packaged_policy_returns_the_default_for_its_own_label():
    assert packaged_policy(default_policy().label) is default_policy()


def test_default_json_is_unchanged():
    text = resources.files("riskvalidation.policies").joinpath("default.json").read_bytes()
    assert hashlib.sha256(text).hexdigest() == DEFAULT_SHA256
    policy = default_policy()
    assert policy.label == "riskvalidation-default 2026.10.1"
    assert all(not t.parameters and not t.rule and not t.by and not t.cases for t in policy.thresholds.values())


def test_parameters_round_trip_through_the_json_format():
    data = {
        "policy_id": "p",
        "version": "1",
        "thresholds": {
            "t.a": {"kind": "upper_bad", "amber": 1.0, "red": 2.0, "parameters": {"k_se": 3.0, "minimum": 0.75}},
            "t.b": {"kind": "p_value", "amber": 0.05, "red": 0.01},
        },
    }
    policy = policy_from_mapping(copy.deepcopy(data))
    assert dict(policy.thresholds["t.a"].parameters) == {"k_se": 3.0, "minimum": 0.75}
    assert dict(policy.thresholds["t.b"].parameters) == {}
    with pytest.raises(TypeError):
        policy.thresholds["t.a"].parameters["k_se"] = 1.0  # frozen
    rebuilt = policy_from_mapping(
        {
            "policy_id": "p",
            "version": "1",
            "thresholds": {
                tid: {"kind": t.kind, "amber": t.amber, "red": t.red, "parameters": dict(t.parameters)}
                for tid, t in policy.thresholds.items()
            },
        }
    )
    assert rebuilt == policy
    # an existing threshold keeps its behavior
    plain = Threshold("p_value", 0.05, 0.01)
    assert plain.classify(0.03) is Light.AMBER and plain.parameters == {}


def test_a_keyed_entry_needs_its_field_and_its_cases():
    with pytest.raises(ValueError):
        Threshold("upper_bad", 1.0, 1.0, by="segment")
    with pytest.raises(ValueError):
        Threshold("upper_bad", 1.0, 1.0, cases={"a": {"limit": 1.0}})
    with pytest.raises(ValueError):
        Threshold("upper_bad", 1.0, 1.0, parameters={"x": float("nan")})


def test_a_keyed_entry_picks_the_case_by_the_row_field():
    policy = packaged_policy("contraste-market-battery 1.0.0")
    row = {"segment": "alpha=0.025", "statistic": 31.0, "metric": 30.0, "extras": {}}
    light, eff = policy.classify_row("market.desk_backtest", row)
    assert light is Light.RED and eff.parameters["limit"] == 30.0
    row = {"segment": "alpha=0.01", "statistic": 12.0}
    assert policy.classify_row("market.desk_backtest", row)[0] is Light.GREEN
    # a value without a case, or a row without the field, is not evaluated
    unknown = {"segment": "alpha=0.5", "statistic": 1.0}
    assert policy.classify_row("market.desk_backtest", unknown)[0] is Light.NOT_EVALUATED
    assert policy.classify_row("market.desk_backtest", {"statistic": 1.0})[0] is Light.NOT_EVALUATED
    assert policy.classify_row("market.not_a_test", row) == (Light.NOT_EVALUATED, None)
    euler = packaged_policy("contraste-portfolio-battery 1.0.0")
    es = {"metric": 5e-7, "extras": {"basis": "exact-allocation"}}
    var = {"metric": 5e-3, "extras": {"basis": "kernel-var"}}
    assert euler.classify_row("portfolio.euler_additivity", es)[0] is Light.GREEN
    assert euler.classify_row("portfolio.euler_additivity", var)[0] is Light.GREEN
    # the same gap passes at the kernel tolerance and fails at the exact one
    assert euler.classify_row("portfolio.euler_additivity", var)[0] is Light.GREEN
    wrong = {"metric": 5e-3, "extras": {"basis": "exact-allocation"}}
    assert euler.classify_row("portfolio.euler_additivity", wrong)[0] is Light.RED


def test_the_pit_policy_holds_no_simulated_critical_value():
    th = packaged_policy("contraste-ccr-battery 1.0.0").thresholds["ccr.pit_exposure_backtest"]
    assert not th.parameters and th.rule == "ccr_pit_simulated"
    row = {"statistic": 3.0, "extras": {"simulated_critical_values": {"p05": 2.0, "p01": 4.0}}}
    assert classify_rule(th.rule, th, row) is Light.AMBER
    assert classify_rule(th.rule, th, {**row, "statistic": 5.0}) is Light.RED
    assert classify_rule(th.rule, th, {**row, "statistic": 1.0}) is Light.GREEN
    assert classify_rule(th.rule, th, {"statistic": 1.0, "extras": {}}) is Light.NOT_EVALUATED


def test_an_unknown_rule_is_a_keyerror_naming_the_rules():
    with pytest.raises(KeyError) as exc:
        classify_rule("no_such_rule", Threshold("upper_bad", 1.0, 1.0), {})
    assert "upper_le" in str(exc.value)


# ---------------------------------------------------------------------------------------------------------------
# the committed documents (read-only)

ROOT = os.environ.get("CONTRASTE_ROOT")
needs_root = pytest.mark.skipif(not ROOT, reason="CONTRASTE_ROOT is unset")


def _documents():
    for path in sorted(glob.glob(str(Path(ROOT) / "data" / "derived" / "C*" / "*.json"))):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if isinstance(doc, dict) and "tests" in doc:
            rel = Path(path).relative_to(Path(ROOT) / "data" / "derived").as_posix()
            for row in doc["tests"]:
                yield rel, row


def _inline_rows():
    return [(rel, row) for rel, row in _documents() if row["policy_version"].startswith("contraste")]


def _label(pv: str) -> str:
    head = pv.split("; ")[0]
    stem, _, version = head.rpartition("-")
    return f"{stem} {version}"


def _cut_params(pv: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in pv.split("; ")[1:]:
        key, value = part.split("=", 1)
        if key in ("r_green", "r_amber"):
            low, high = value.split("-")
            out[key + "_low"], out[key + "_high"] = float(low), float(high)
        elif key in CUT_KEYS:
            out[key] = float(value)
    return out


def _view(row: dict) -> dict:
    """The row as the policy reads it. Two fields the committed documents predate are derived from the document:
    the Euler basis from the segment (`var-*` is the kernel VaR, everything else an exact allocation), and the
    observed rates of a lifetime-PD back-test from the stored Brier score and signed gap."""
    view = copy.deepcopy(row)
    extras = view.setdefault("extras", {})
    if row["test_id"] == "portfolio.euler_additivity" and "basis" not in extras:
        extras["basis"] = "kernel-var" if str(row.get("segment") or "").startswith("var-") else "exact-allocation"
    if row["test_id"] == "ifrs9.lifetime_pd_backtest" and "band_low" in extras and "observed" not in extras:
        obs = []
        for brier, gap in zip(extras["brier"], extras["signed_gap"], strict=True):
            disc = (1 - 2 * gap) ** 2 - 4 * (brier - gap)
            p = ((1 - 2 * gap) - math.sqrt(max(disc, 0.0))) / 2.0  # the smaller root: a PD below one half
            o = p + gap
            obs.append(0.0 if abs(o) < 1e-7 else o)  # a zero rate carries the rounding of the stored 9 digits
        extras["observed"] = obs
    return view


@needs_root
def test_every_inline_string_maps_to_a_packaged_entry_with_its_cut_points():
    rows = _inline_rows()
    assert len(rows) == 2450
    checked = Counter()
    for rel, row in rows:
        policy = packaged_policy(_label(row["policy_version"]))
        th = policy.threshold(row["test_id"])
        assert th is not None, (rel, row["test_id"])
        eff = th.for_row(_view(row))
        assert eff is not None, (rel, row["test_id"], th.by)
        cut = _cut_params(row["policy_version"])
        known = {"amber": eff.amber, "red": eff.red, **eff.parameters}  # parameters keep the engine's argument names
        for key, value in cut.items():
            assert key in known and known[key] == value, (rel, row["test_id"], key, value, known)
        if row["test_id"] == "ccr.pit_exposure_backtest":  # the per-case numbers are data in the row's extras
            part = dict(p.split("=", 1) for p in row["policy_version"].split("; ")[1:])
            crit = row["extras"]["simulated_critical_values"]
            assert float(part["simulated_p05"]) == crit["p05"] and float(part["simulated_p01"]) == crit["p01"]
            assert float(part["effective_sample_size"]) == row["extras"]["effective_sample_size"]
        checked[row["test_id"]] += 1
    assert sum(checked.values()) == 2450


@needs_root
def test_the_packaged_entry_reproduces_the_committed_light_of_every_inline_row():
    reproduced, failures = 0, []
    for rel, row in _inline_rows():
        policy = packaged_policy(_label(row["policy_version"]))
        light, _ = policy.classify_row(row["test_id"], _view(row))
        if light.value != row["light"]:
            failures.append((rel, row["test_id"], row.get("segment"), row["light"], light.value))
        else:
            reproduced += 1
    print(f"reproduced rows: {reproduced}")
    assert not failures, f"{len(failures)} rows not reproduced, first: {failures[:8]}"
    assert reproduced == 2450


@needs_root
def test_the_inline_battery_heads_keep_resolving_with_the_packaged_equivalent():
    from riskvalidation.governance.policy import resolve_policy

    for rel, row in _inline_rows():
        identity, reason = resolve_policy(row["policy_version"], row["test_id"], row["alpha_amber"], row["alpha_red"])
        assert reason is None and identity["source"] == "inline-battery", (rel, row["policy_version"], reason)


def test_battery_results_carry_the_packaged_label_and_it_resolves():
    """Every `_version` call site writes the packaged battery label, and each label is a packaged policy."""
    import numpy as np

    from riskvalidation.core.registry import spec
    from riskvalidation.governance.policy import resolve_policy
    from riskvalidation.validation import aggregation, liquidity, market, oprisk, portfolio, stress  # noqa: F401

    results = [
        portfolio.euler_additivity(np.array([0.5, 0.5]), 1.0, basis="exact-allocation"),
        portfolio.mc_convergence(1.0, 0.1, 1.0),
        liquidity.list_compounding_golden(),
        market.desk_backtest(np.zeros(250), np.ones(250), 0.01),
    ]
    for result in results:
        assert result.policy_version in BATTERY_LABELS and "=" not in result.policy_version
        policy = packaged_policy(result.policy_version)
        assert policy.threshold(result.test_id) is not None, result.test_id
        identity, reason = resolve_policy(result.policy_version, result.test_id)
        assert reason is None and identity["source"] == "packaged-policy"
        assert spec(result.test_id).test_id == result.test_id
        light, _ = policy.classify_row(result.test_id, result.to_dict())
        assert light is result.light, (result.test_id, light, result.light)
