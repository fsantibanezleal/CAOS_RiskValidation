"""Fixtures of the governance tests: the four fixtures of the contract `engine-governance-api` 1.0.0 (copied from
CAOS_Contraste `task/u15-s` at 5dce395, see fixtures/README.md), the facts they were built from, and the contract's
checker."""

from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader

import pytest

from .governance_support import FIX, SCHEMAS, load


@pytest.fixture(scope="session")
def inventory_doc() -> dict:
    return load("inventory-fixture.json")


@pytest.fixture(scope="session")
def findings_doc() -> dict:
    return load("findings-fixture.json")


@pytest.fixture(scope="session")
def appetite_doc() -> dict:
    return load("appetite-fixture.json")


@pytest.fixture(scope="session")
def extract() -> dict:
    return load("source-extract.json")


@pytest.fixture(scope="session")
def report_docs() -> dict:
    return {
        "C07/banco-andino": load("report-C07-banco-andino-fixture.json"),
        "C18/banco-andino": load("report-C18-banco-andino-fixture.json"),
    }


@pytest.fixture(scope="session")
def checker():
    """The contract's cross-field checker (`semantics.py` of the contract tests), loaded from its .pysrc copy."""
    path = FIX / "semantics.pysrc"
    loader = SourceFileLoader("semantics_u15s", str(path))
    mod = module_from_spec(spec_from_loader("semantics_u15s", loader))
    loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def schemas() -> dict:
    return {p.name.split(".")[0]: json.loads(p.read_text(encoding="utf-8")) for p in SCHEMAS.glob("*.schema.json")}


@pytest.fixture(scope="session")
def validate_schema(schemas):
    """`validate_schema(doc, "governance-report")`: validate against a governance schema (jsonschema is a test-only
    reference; the test is skipped with its reason when it is absent). The relative `$ref` to the common text
    schema resolves through its `$id`."""
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    from referencing.jsonschema import DRAFT202012

    text = json.loads((SCHEMAS / "common" / "text.schema.json").read_text(encoding="utf-8"))
    registry = referencing.Registry().with_resource(text["$id"], referencing.Resource.from_contents(text, DRAFT202012))

    def check(doc: dict, name: str) -> None:
        jsonschema.Draft202012Validator(schemas[name], registry=registry).validate(doc)

    return check
