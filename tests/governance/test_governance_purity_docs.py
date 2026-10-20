"""Requirements RV-1550 to RV-1581 (docs/design/features/u15-governance/requirements.md): the governance modules are
pure and standard-library only, every convention carries its label, no text the package writes states an approval,
and the documentation names every function and every `gov.` test."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

from riskvalidation.governance import CONVENTIONS
from riskvalidation.governance.conventions import LIFECYCLE_GRAPH

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "riskvalidation"
MODULES = sorted((SRC / "governance").glob("*.py")) + [SRC / "validation" / "governance.py"]
PAGES = ROOT / "docs" / "governance"
NAMES = (
    "README.md",
    "01_scope-and-inventory.md",
    "02_tiering-and-lifecycle.md",
    "03_findings-and-schedule.md",
    "04_opinion-and-appetite.md",
    "05_policy-and-report.md",
    "06_validation-tests.md",
)
GOV_IDS = (
    "gov.inventory_completeness",
    "gov.findings_overdue",
    "gov.validation_due",
    "gov.tier_override_documented",
    "gov.edges_acyclic",
)
HEAVY = (
    "pandas",
    "sklearn",
    "statsmodels",
    "torch",
    "lightgbm",
    "xgboost",
    "matplotlib",
    "fairlearn",
    "shap",
    "QuantLib",
    "networkx",
    "jsonschema",
)
CONVENTION_NAMES = (
    "EXPOSURE_BAND_CUTS",
    "DEADLINE_DAYS",
    "FREQUENCY",
    "APPETITE_THRESHOLDS",
    "REGULATORY_USE_GRADES",
    "COMPLEXITY_AGGREGATION",
    "ADOPTED_READING",
    "LIFECYCLE_GRAPH",
    "FINDING_GRAPH",
    "RECOMMENDATION_RULE",
)
FORBIDDEN_ATTRS = {
    "now",
    "today",
    "utcnow",
    "getenv",
    "environ",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "getcwd",
}
FORBIDDEN_NAMES = {"open", "input"}


def test_modules_import_only_the_standard_library_numpy_and_scipy():
    imports = ",".join(
        f"riskvalidation.governance.{p.stem}" for p in sorted((SRC / "governance").glob("*.py")) if p.stem != "__init__"
    )
    code = (
        f"import sys, riskvalidation.governance, {imports}, riskvalidation.validation.governance;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules]; print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported: {out.stdout.strip()} {out.stderr}"


def test_no_function_reads_the_clock_the_environment_or_a_file():
    offenders = []
    for path in MODULES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                offenders.append(f"{path.name}:{node.lineno} .{node.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
                offenders.append(f"{path.name}:{node.lineno} {node.func.id}()")
    assert not offenders, offenders


def test_no_governance_module_imports_os_sys_pathlib_or_the_clock():
    bad = []
    for path in MODULES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            bad += [
                f"{path.name}: {m}"
                for m in mods
                if m.split(".")[0] in ("os", "sys", "pathlib", "time", "io", "subprocess", "glob")
            ]
    assert not bad, bad


def test_every_convention_carries_its_label_and_none_calls_itself_regulatory(checker):
    assert set(CONVENTIONS) >= set(CONVENTION_NAMES)
    for name, conv in CONVENTIONS.items():
        assert conv["label"] == "convention", name
        assert conv["basis"].strip(), name
        assert checker.regulatory_claims(conv["basis"]) == [], name


def _runtime_strings(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


def test_no_text_the_package_writes_states_an_approval_or_a_regulatory_convention(checker):
    """A string that is exactly a lifecycle state name (`approved`, `approved_with_conditions`) is a B.4 identifier
    the contract fixes, not a statement about a model: it is the only exemption."""
    hits = []
    for path in MODULES:
        for line, text in _runtime_strings(path):
            if text in LIFECYCLE_GRAPH:
                continue
            for claim in checker.approval_claims(text) + checker.regulatory_claims(text):
                hits.append(f"{path.name}:{line} {claim!r}")
    assert not hits, hits


def test_the_conventions_are_declared_in_one_module():
    text = (SRC / "governance" / "conventions.py").read_text(encoding="utf-8")
    for name in CONVENTION_NAMES:
        assert re.search(rf"\b{name}\b", text), name


# ---- the documentation


def _text(name: str) -> str:
    return (PAGES / name).read_text(encoding="utf-8")


def test_governance_pages_exist_link_each_other_and_the_docs_index():
    for name in NAMES:
        assert (PAGES / name).is_file(), name
    readme = _text("README.md")
    for name in NAMES[1:]:
        assert f"({name})" in readme, name
    assert "(governance/README.md)" in (ROOT / "docs" / "README.md").read_text(encoding="utf-8")


def test_every_gov_test_id_is_named_in_backticks_on_a_page():
    text = "\n".join(_text(n) for n in NAMES)
    for test_id in GOV_IDS:
        assert f"`{test_id}`" in text, test_id


def test_every_function_of_the_contract_is_named_in_the_pages():
    text = "\n".join(_text(n) for n in NAMES)
    functions = (
        "scope_test",
        "scope_record",
        "assign_variant",
        "build_inventory",
        "topological_order",
        "graph_metrics",
        "attribute_completeness",
        "exposure_band",
        "materiality",
        "complexity",
        "matrix_tier",
        "apply_override",
        "tier_sensitivity",
        "tier_counts",
        "allowed_next",
        "transition",
        "severity_to_ecb",
        "map_product_status",
        "deadline_days",
        "due_date",
        "ageing_days",
        "is_overdue",
        "transition_finding",
        "anchor_from_test_row",
        "rwea_share_of_anchor",
        "register_counts",
        "add_months",
        "next_dates",
        "finding_level",
        "opinion_level",
        "recommendation",
        "light",
        "tier1_validated_on_time",
        "open_high_severity",
        "overdue_remediation",
        "entries_under_exception",
        "entries_in_breach",
        "pma_magnitude",
        "owner_and_complete_record",
        "appetite_metrics",
        "resolve_policy",
        "executive_counts",
        "build_report",
    )
    missing = [f for f in functions if f"`{f}`" not in text and f"`{f}(" not in text]
    assert not missing, missing


def test_conventions_are_labelled_as_such_and_the_what_not_to_claim_list_is_stated():
    text = "\n".join(_text(n) for n in NAMES)
    for name in CONVENTION_NAMES:
        assert name in text or name.lower().replace("_", " ") in text.lower(), name
    assert text.lower().count("convention") >= 20
    for claim in ("approves no model", "certif", "item 30", "SS1/23", "UNVERIFIED"):
        assert claim in text, claim
    assert "What the engine does not claim" in _text("README.md")


def test_pages_carry_their_sources_and_stated_goldens():
    assert "dossier 06" in _text("01_scope-and-inventory.md") and "B.2" in _text("01_scope-and-inventory.md")
    assert "12/6/0/1" in _text("02_tiering-and-lifecycle.md") and "18/0/0/1" in _text("02_tiering-and-lifecycle.md")
    assert "ECB" in _text("04_opinion-and-appetite.md") and "5 percent" in _text("04_opinion-and-appetite.md")
    assert "BL-085" in _text("05_policy-and-report.md")
    assert "0.7125" in _text("06_validation-tests.md")
