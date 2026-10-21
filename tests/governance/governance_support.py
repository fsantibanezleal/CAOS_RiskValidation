"""Helpers of the governance tests (the fixtures are in conftest.py): loaders for the contract fixtures and
the builders of entries, edges and report inputs."""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import Any

from riskvalidation.governance.inventory import Component, Edge, Entry

FIX = Path(__file__).resolve().parent / "fixtures"
SCHEMAS = FIX / "schemas"
AS_OF = date(2026, 12, 31)
ATTR_IDS = [f"A{i:02d}" for i in range(1, 41)]


def load(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def attributes(**overrides: dict) -> dict[str, dict]:
    """Forty valid attributes (all `convention`, A32 and A33 `not-applicable`); `overrides` replace attributes by id."""
    out = {a: {"value": f"value of {a}", "status": "convention", "basis": f"a stated basis for {a}"} for a in ATTR_IDS}
    out["A32"] = {
        "value": None,
        "status": "not-applicable",
        "basis": "Contraste approves no model (SDD non-goal 1): no approver exists",
    }
    out["A33"] = {
        "value": None,
        "status": "not-applicable",
        "basis": "Contraste approves no model (SDD non-goal 1): no approval date",
    }
    out.update(overrides)
    return out


def make_entry(
    case: str = "C01",
    fit: str = "german",
    components: tuple[Component, ...] | None = None,
    variants: tuple[tuple[str, str], ...] = (("baseline", "the single models document of the case"),),
    attrs: dict | None = None,
) -> Entry:
    comps = components if components is not None else (Component(case, fit, "P0", "benchmark"),)
    return Entry(case, fit, attrs if attrs is not None else attributes(), variants, comps)


def entries_of(doc: dict) -> tuple[Entry, ...]:
    out = []
    for e in doc["entries"]:
        comps = tuple(Component(e["case_id"], e["fit_id"], c["model_id"], c["role"]) for c in e["components"])
        variants = tuple((v["variant_id"], v["basis"]) for v in e["variants"])
        out.append(Entry(e["case_id"], e["fit_id"], copy.deepcopy(e["attributes"]), variants, comps))
    return tuple(out)


def edges_of(doc: dict) -> tuple[Edge, ...]:
    return tuple(Edge(x["upstream"], x["downstream"], x["kind"], x["stated_source"]) for x in doc["edges"])


def variants_of(extract: dict, case: str) -> list[dict]:
    """The variant documents of a case as `build_report` reads them: the default variant first, then the others in
    order; each carries its path, its test rows and the provenance the report states (code version, engine
    version, the manifest's seed)."""
    cd = extract["cases"][case]
    seed = cd["manifest"]["seed"]
    docs = [
        {
            "variant_id": v["variant_id"],
            "path": v["path"],
            "tests": v["tests"],
            "code_version": v["code_version"],
            "riskvalidation_version": v["riskvalidation_version"],
            "seed": seed,
        }
        for v in cd["variants"]
    ]
    docs.sort(key=lambda d: d["variant_id"] != cd["default_variant"])  # a stable sort: the default first
    return docs


def report_inputs(
    extract: dict, inventory_doc: dict, findings_doc: dict, key: str
) -> tuple[dict, dict, list[dict], list[dict]]:
    case = key.split("/")[0]
    entry = next(e for e in inventory_doc["entries"] if e["key"] == key)
    register = [r for r in findings_doc["rows"] if r["entry_key"] == key]
    return entry, inventory_doc, register, variants_of(extract, case)
