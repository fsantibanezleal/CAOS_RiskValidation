"""U16 second fix round (review N7 and the sources): the policy format is readable without the engine's code. A JSON
Schema validates every packaged file; the page states every rule; a file the engine cannot read is refused at load."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from riskvalidation.core.policy import (
    FORMAT_VERSION,
    packaged_labels,
    packaged_policy,
    policy_from_mapping,
)
from riskvalidation.policies.rules import GENERIC, NAMED, RULES

ROOT = Path(__file__).resolve().parents[1]
POLICIES = resources.files("riskvalidation.policies")
FILES = sorted(p.name for p in POLICIES.iterdir() if p.name.endswith(".json") and not p.name.endswith(".schema.json"))
SCHEMA = json.loads(POLICIES.joinpath("policy.schema.json").read_text(encoding="utf-8"))
PAGE = ROOT / "docs" / "policies.md"


def test_the_schema_is_not_taken_for_a_policy():
    assert "policy.schema.json" not in {f"{label.split()[0]}.schema.json" for label in packaged_labels()}
    assert len(packaged_labels()) == len(FILES) == 12


@pytest.mark.parametrize("name", FILES)
def test_every_packaged_file_validates_against_the_schema(name):
    jsonschema = pytest.importorskip("jsonschema")
    doc = json.loads(POLICIES.joinpath(name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(SCHEMA).validate(doc)
    for tid, entry in doc["thresholds"].items():
        rule = entry.get("rule", "")
        if rule:
            assert rule.partition(":")[0] in RULES, (name, tid, rule)


@pytest.mark.parametrize("name", [f for f in FILES if f.startswith("contraste-")])
def test_every_battery_file_declares_its_format_version(name):
    doc = json.loads(POLICIES.joinpath(name).read_text(encoding="utf-8"))
    assert doc["format_version"] == FORMAT_VERSION == 1


def test_the_schema_rejects_what_the_engine_would_misread():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(SCHEMA)
    good = {"policy_id": "p", "version": "1", "thresholds": {"t.a": {"kind": "p_value", "amber": 0.05, "red": 0.01}}}
    validator.validate(good)
    for mutate in (
        lambda d: d.update(format_version=2),
        lambda d: d["thresholds"]["t.a"].update(kind="sideways"),
        lambda d: d["thresholds"]["t.a"].update(by="segment"),  # a key without cases
        lambda d: d["thresholds"]["t.a"].update(source=7),
        lambda d: d["thresholds"]["t.a"].update(parameters={"x": "1"}),
        lambda d: d["thresholds"]["t.a"].update(unknown_key=1),
    ):
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(bad)


def test_a_file_the_engine_cannot_read_is_refused_when_it_is_loaded():
    base = {"policy_id": "p", "version": "1", "thresholds": {"t.a": {"kind": "upper_bad", "amber": 1, "red": 2}}}
    policy_from_mapping(base)  # the format version is absent: 1
    with pytest.raises(ValueError, match="format_version"):
        policy_from_mapping({**base, "format_version": 2})
    typo = copy.deepcopy(base)
    typo["thresholds"]["t.a"]["rule"] = "uper_le:statistic"
    with pytest.raises(ValueError, match=r"policy p 1, entry t\.a.*unknown rule"):
        policy_from_mapping(typo)
    nan = copy.deepcopy(base)
    nan["thresholds"]["t.a"]["parameters"] = {"k": "x"}
    with pytest.raises(ValueError, match=r"policy p 1, entry t\.a"):
        policy_from_mapping(nan)
    missing = copy.deepcopy(base)
    del missing["thresholds"]["t.a"]["red"]
    with pytest.raises(ValueError, match="red"):
        policy_from_mapping(missing)


def test_no_parameter_reuses_the_name_of_the_pair_with_another_value():
    for label in packaged_labels():
        for tid, th in packaged_policy(label).thresholds.items():
            for key, pair in (("amber", th.amber), ("red", th.red)):
                if key in th.parameters:
                    assert th.parameters[key] == pair, (label, tid, key)


# ---- docs/policies.md ---------------------------------------------------------------------------------------------

def test_every_rule_is_stated_in_the_page():
    text = PAGE.read_text(encoding="utf-8")
    for name in sorted({*GENERIC, *NAMED}):
        assert f"`{name}" in text, f"docs/policies.md does not state the rule {name}"
    for sentence in ("A value equal to `amber` is green", "never returns a light the rule would not give",
                     "format_version", "policy.schema.json", "regulatory:", "77"):
        assert sentence in text, sentence


def test_the_page_is_what_the_generator_writes():
    if not (ROOT / "scripts" / "make_policies_doc.py").is_file():
        pytest.skip("the generator is not in the tree this test runs from")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_policies_doc.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_regulatory_sources_name_the_paragraph_and_the_rest_are_conventions():
    reg = {"market.desk_backtest": "MAR32.19", "market.pla_test": "MAR32.42", "market.basel_traffic_light": "MAR32.9"}
    seen = set()
    for label in packaged_labels():
        if label.startswith("riskvalidation-default"):
            continue
        for tid, th in packaged_policy(label).thresholds.items():
            if tid in reg:
                assert th.source.startswith(f"regulatory: BCBS {reg[tid]}"), tid
                assert "not a regulatory threshold" not in th.source, tid
                seen.add(tid)
            else:
                assert th.source.startswith("convention: "), tid
    assert seen == set(reg)
