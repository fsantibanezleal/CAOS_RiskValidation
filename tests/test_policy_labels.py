"""U16 second fix round (review N3, N4): a row computed with cut points that differ from its packaged entry says so
(`<label>+custom` and `extras["policy_override"]`); a test fails when a call site writes a label whose file lacks the
test id, or passes a cut point the entry does not name."""

from __future__ import annotations

import ast
import pickle
from pathlib import Path

import numpy as np
import pytest

from riskvalidation.core.policy import Policy, default_policy, packaged_labels, packaged_policy
from riskvalidation.governance.policy import resolve_policy
from riskvalidation.governance.report import _policy_row
from riskvalidation.validation import ccr, liquidity, market, portfolio, stress
from riskvalidation.validation import ml as vml

VALIDATION = Path(__file__).resolve().parents[1] / "src" / "riskvalidation" / "validation"
BATTERY_MODULES = "aggregation ccr ifrs9 irrbb lgd_ead liquidity market oprisk portfolio stress".split()


def _sites():
    """Every ``battery_label(POLICY_LABEL, "<test id>", ...)`` call of the battery modules: (module, test id, keys, lineno)."""
    out = []
    for m in BATTERY_MODULES:
        path = VALIDATION / f"{m}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        label = next(
            n.value.value for n in tree.body
            if isinstance(n, ast.Assign) and n.targets[0].id == "POLICY_LABEL"  # type: ignore[attr-defined]
        )
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "battery_label":
                tid = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else None
                out.append((m, label, tid, [k.arg for k in n.keywords], n.lineno))
        bare = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id == "POLICY_LABEL"
        ]
        calls = {a.lineno for a in ast.walk(tree) if isinstance(a, ast.Call) and getattr(a.func, "id", "") == "battery_label"}
        out.append((m, label, "<bare>", [ln for ln in bare if not any(c <= ln for c in calls)], 0))
    return out


SITES = [s for s in _sites() if s[2] != "<bare>"]


def test_there_are_77_call_sites_and_none_writes_a_bare_label():
    """The count of call sites that write a battery label (the earlier reports said 79; the code has 77)."""
    assert len(SITES) == 77, len(SITES)
    assert {m for m, *_ in SITES} == set(BATTERY_MODULES)


def test_no_site_writes_the_bare_label_outside_battery_label():
    """Every use of ``POLICY_LABEL`` is an argument of ``battery_label`` (the one exception per module is its definition)."""
    for m, _label, tid, lines, _ in _sites():
        if tid == "<bare>":
            assert len(lines) <= 1, (m, lines)  # the definition `POLICY_LABEL = ...`


@pytest.mark.parametrize("module,label,test_id,keys,lineno", [s for s in SITES if s[2]])
def test_every_site_writes_a_label_whose_file_holds_its_test_id_and_names_only_cut_points(
    module, label, test_id, keys, lineno
):
    th = packaged_policy(label).threshold(test_id)
    assert th is not None, f"{module}.py:{lineno}: {label} holds no entry for {test_id}"
    names = {"amber", "red", *th.parameters} | {k for case in th.cases.values() for k in case}
    unknown = [k for k in keys if k not in names]
    assert not unknown, f"{module}.py:{lineno}: {test_id} passes {unknown}, which the entry does not name {sorted(names)}"


def test_the_five_ids_of_the_review_have_their_entries():
    market_policy, stress_policy = packaged_policy("contraste-market-battery 1.0.0"), packaged_policy(
        "contraste-stress-battery 1.0.0")
    for tid in ("market.basel_traffic_light", "market.du_escanciano_unconditional", "market.du_escanciano_conditional",
                "market.mcneil_frey_residual_test"):
        assert market_policy.threshold(tid) is not None, tid
    assert stress_policy.threshold("stress.bottom_up_vs_top_down") is not None


# ---- the two paths of N3: default cut points and caller cut points ----------------------------------------------------

def _exceedances(n_exc: int = 3, n: int = 250) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[:n_exc] = True
    return out


def test_default_cut_points_write_the_bare_packaged_label():
    results = [
        market.kupiec_pof(_exceedances(), 0.01),
        liquidity.list_compounding_golden(),
        stress.bottom_up_vs_top_down(1.0, 1.1),
        market.basel_traffic_light(_exceedances(6)),
        market.desk_backtest(np.zeros(250), np.ones(250), 0.025),
        portfolio.euler_additivity(np.array([0.5, 0.5]), 1.0, basis="exact-allocation"),
        ccr.ee_analytic_vs_simulated(np.ones(4), np.ones(4), np.ones(4)),
    ]
    for r in results:
        assert type(r.policy_version) is str and "+custom" not in r.policy_version, r.test_id
        assert "policy_override" not in r.extras, r.test_id
        light, _ = packaged_policy(r.policy_version).classify_row(r.test_id, r.to_dict())
        assert light is r.light, (r.test_id, light, r.light)


CUSTOM = [
    ("market.kupiec_pof", lambda: market.kupiec_pof(_exceedances(), 0.01, alpha_amber=0.9, alpha_red=0.5),
     {"alpha_amber": 0.9, "alpha_red": 0.5}),
    ("liq.list_compounding_golden", lambda: liquidity.list_compounding_golden(tol=1e-3), {"tolerance": 1e-3}),
    ("stress.bottom_up_vs_top_down", lambda: stress.bottom_up_vs_top_down(1.0, 1.1, amber=0.05, red=0.08),
     {"amber": 0.05, "red": 0.08}),
    ("ccr.ee_analytic_vs_simulated", lambda: ccr.ee_analytic_vs_simulated(np.ones(4), np.ones(4), np.ones(4), k_se=1.0),
     {"k_se": 1.0}),
    ("portfolio.euler_additivity",
     lambda: portfolio.euler_additivity(np.array([0.5, 0.5]), 1.0, tolerance=0.5, basis="kernel-var"),
     {"tolerance": 0.5}),
]


@pytest.mark.parametrize("test_id,make,used", CUSTOM, ids=[c[0] for c in CUSTOM])
def test_caller_cut_points_write_plus_custom_and_the_cut_points_they_used(test_id, make, used):
    r = make()
    assert type(r.policy_version) is str and r.policy_version.endswith("+custom"), r.policy_version
    override = r.extras["policy_override"]
    assert override["test_id"] == test_id and override["label"] == r.policy_version[: -len("+custom")]
    for key, value in used.items():
        assert override["parameters"][key] == value, (key, override)
    # the label names the packaged policy and the override carries the difference: governance resolves both
    d = r.to_dict()
    identity, reason = resolve_policy(d["policy_version"], d["test_id"], d["alpha_amber"], d["alpha_red"],
                                      override=d["extras"]["policy_override"])
    assert reason is None and identity["source"] == "packaged-policy-overridden"
    assert identity["label"] == r.policy_version
    for key, value in used.items():
        assert identity["parameters"][key] == value
    row = _policy_row(d)
    assert row["source"] == "packaged-policy-overridden" and row["reason"] is None
    # without the override the identity cannot be built, and says why
    none, why = resolve_policy(d["policy_version"], d["test_id"])
    assert none is None and "policy_override" in why


def test_a_customised_row_survives_pickling_and_the_label_is_a_plain_string():
    r = market.kupiec_pof(_exceedances(), 0.01, alpha_amber=0.9, alpha_red=0.5)
    again = pickle.loads(pickle.dumps(r))
    assert again == r and type(again.policy_version) is str


def test_ml_policy_with_changed_cut_points_labels_its_results_and_resolves():
    from riskvalidation.ml.policy import ml_policy

    assert ml_policy().label == default_policy().label and not ml_policy().override
    custom = ml_policy(overfit_tolerance=0.10)
    assert isinstance(custom, Policy) and custom.label == default_policy().label + "+custom"
    assert set(custom.override) == {"ml.overfit"} and custom.override["ml.overfit"]["amber"] != \
        default_policy().threshold("ml.overfit").amber
    ident, reason = resolve_policy(custom.label, "ml.overfit", override=custom.override)
    assert reason is None and ident["source"] == "packaged-policy-overridden"
    assert ident["parameters"]["amber"] == custom.override["ml.overfit"]["amber"]
    # a result computed under it carries the override
    from riskvalidation.ml.generalisation import overfit_gap

    res = vml.overfit("m", overfit_gap(0.80, 0.72, 0.10), policy=custom)
    assert res.policy_version == custom.label and type(res.policy_version) is str
    assert res.extras["policy_override"] == dict(custom.override)
    plain = vml.overfit("m", overfit_gap(0.80, 0.72, 0.05))
    assert "+custom" not in plain.policy_version and "policy_override" not in plain.extras


def test_every_packaged_label_still_resolves_after_the_fix():
    for label in packaged_labels():
        assert packaged_policy(label).label == label
