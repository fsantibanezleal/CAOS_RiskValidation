"""Count-noun agreement in the validation report (Contraste backlog BL-097, engine 1.2.0).

Every sentence of ``governance/report.py`` that prints a count beside a noun agrees in number in English and in
Spanish: one is singular, zero and two or more are plural. The sentences are (report.py, ``build_report``):

1. section 1, the scope: ``N test rows`` and ``N variants`` (``N filas de prueba``, ``N variantes``);
2. section 2, the description: ``N model records`` (``N registros de modelo``);
3. section 6, the missing element: ``U of M rows are of tests that are not`` (noun by M, verb by U) and ``null for
   D rows`` (``U de M filas son de pruebas``, ``nula en D filas``);
4. section 7, the input reference: ``the N variants of the entry``;
5. section 9, the limits: ``N register rows are stated limits`` (``N filas del registro son límites declarados``);
6. section 12, the input reference: ``the N documents read for the entry``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re

import pytest

from riskvalidation.governance.report import _agree, _pick, build_report

from .governance_support import AS_OF, report_inputs


def _inputs(extract, inventory_doc, findings_doc, key="C18/banco-andino"):
    return copy.deepcopy(report_inputs(extract, inventory_doc, findings_doc, key))


def _section(report: dict, number: int) -> dict:
    return report["sections"][number - 1]


def _text(section: dict, index: int = 0) -> dict:
    return section["blocks"][index]["text"]


# ---- the helpers


@pytest.mark.parametrize(
    ("n", "en", "es"),
    [
        (0, "0 variants", "0 variantes"),
        (1, "1 variant", "1 variante"),
        (2, "2 variants", "2 variantes"),
        (11, "11 variants", "11 variantes"),
    ],
)
def test_agree_gives_the_count_and_the_noun_in_both_languages(n, en, es):
    assert _agree(n, ("variant", "variants"), ("variante", "variantes")) == (en, es)


def test_pick_takes_the_singular_only_for_one():
    assert [_pick(n, "is", "are") for n in (0, 1, 2)] == ["are", "is", "are"]


# ---- 1. section 1: the scope


@pytest.mark.parametrize("n_var", [1, 2])
def test_scope_sentence_agrees_the_variants(extract, inventory_doc, findings_doc, n_var):
    entry, inv, reg, variants = _inputs(extract, inventory_doc, findings_doc)
    text = _text(_section(build_report(entry, inv, reg, variants[:n_var], AS_OF), 1), 1)
    if n_var == 1:
        assert text["en"].endswith("findings of its 1 variant.")
        assert text["es"].endswith("hallazgos de su 1 variante.")
    else:
        assert text["en"].endswith("findings of its 2 variants.")
        assert text["es"].endswith("hallazgos de sus 2 variantes.")


@pytest.mark.parametrize("n_rows", [1, 2])
def test_scope_sentence_agrees_the_test_rows(extract, inventory_doc, findings_doc, n_rows):
    entry, inv, reg, variants = _inputs(extract, inventory_doc, findings_doc)
    variants[0]["tests"] = variants[0]["tests"][:n_rows]
    text = _text(_section(build_report(entry, inv, reg, variants, AS_OF), 1), 1)
    en = "(1 test row)" if n_rows == 1 else "(2 test rows)"
    es = "(1 fila de prueba)" if n_rows == 1 else "(2 filas de prueba)"
    assert en in text["en"] and es in text["es"]


# ---- 2. section 2: the model records


@pytest.mark.parametrize("n", [1, 2])
def test_description_sentence_agrees_the_model_records(extract, inventory_doc, findings_doc, n):
    entry, inv, reg, variants = _inputs(extract, inventory_doc, findings_doc)
    entry["components"] = entry["components"][:n]
    text = _text(_section(build_report(entry, inv, reg, variants, AS_OF), 2))
    assert (f"{n} model record in the fit" if n == 1 else f"{n} model records in the fit") in text["en"]
    assert (f"{n} registro de modelo en el ajuste" if n == 1 else f"{n} registros de modelo en el ajuste") in text["es"]


# ---- 3. section 6: the missing element


def _with_unregistered(variants, unregistered: int, registered: int):
    rows = variants[0]["tests"]
    bad = [{**r, "test_id": "zz.not_registered"} for r in rows[:unregistered]]
    variants[0]["tests"] = bad + rows[unregistered : unregistered + registered]
    return variants


@pytest.mark.parametrize(
    ("unregistered", "registered", "en", "es"),
    [
        (
            1,
            0,
            "(1 of 1 row is of a test that is not: spec() raises KeyError)",
            "(1 de 1 fila es de una prueba que no lo está: spec() lanza KeyError)",
        ),
        (
            1,
            1,
            "(1 of 2 rows is of a test that is not: spec() raises KeyError)",
            "(1 de 2 filas es de una prueba que no lo está: spec() lanza KeyError)",
        ),
        (
            2,
            0,
            "(2 of 2 rows are of tests that are not: spec() raises KeyError)",
            "(2 de 2 filas son de pruebas que no lo están: spec() lanza KeyError)",
        ),
        (
            0,
            1,
            "(0 of 1 row are of tests that are not: spec() raises KeyError)",
            "(0 de 1 fila son de pruebas que no lo están: spec() lanza KeyError)",
        ),
    ],
)
def test_outcomes_missing_agrees_the_rows_and_the_verb(
    extract, inventory_doc, findings_doc, unregistered, registered, en, es
):
    entry, inv, reg, variants = _inputs(extract, inventory_doc, findings_doc)
    variants = _with_unregistered(variants, unregistered, registered)
    missing = _section(build_report(entry, inv, reg, variants, AS_OF), 6)["missing"][0]["reason"]
    assert en in missing["en"] and es in missing["es"]


@pytest.mark.parametrize(
    ("registered", "en", "es"),
    [
        (1, "the direction is null for 1 row for an inline", "la dirección es nula en 1 fila de una cadena"),
        (2, "the direction is null for 2 rows for an inline", "la dirección es nula en 2 filas de una cadena"),
    ],
)
def test_outcomes_missing_agrees_the_rows_without_direction(
    extract, inventory_doc, findings_doc, registered, en, es
):
    entry, inv, reg, variants = _inputs(extract, inventory_doc, findings_doc)
    variants = _with_unregistered(variants, 0, registered)
    missing = _section(build_report(entry, inv, reg, variants, AS_OF), 6)["missing"][0]["reason"]
    assert en in missing["en"] and es in missing["es"]


# ---- 4. section 7: the input reference


@pytest.mark.parametrize("n_var", [1, 2])
def test_variants_reference_agrees(extract, inventory_doc, findings_doc, n_var):
    entry, inv, reg, variants = _inputs(extract, inventory_doc, findings_doc)
    sec = _section(build_report(entry, inv, reg, variants[:n_var], AS_OF), 7)
    want = "the 1 variant of the entry" if n_var == 1 else "the 2 variants of the entry"
    assert sec["inputs"][0]["ref"] == want


# ---- 5. section 9: the stated limits


def _accepted(register, n):
    rows = copy.deepcopy(register)
    for i, r in enumerate(rows):
        r["status"] = "risk_accepted" if i < n else "validated_closed"
    return rows


@pytest.mark.parametrize("n", [0, 1, 2])
def test_limits_sentence_agrees_the_register_rows(extract, inventory_doc, findings_doc, n):
    entry, inv, reg, variants = _inputs(extract, inventory_doc, findings_doc, "C07/banco-andino")
    text = _text(_section(build_report(entry, inv, _accepted(reg, n), variants, AS_OF), 9))
    if n == 1:
        assert text["en"].startswith("1 register row is a stated limit of the synthetic case")
        assert text["es"].startswith("1 fila del registro es un límite declarado del caso sintético")
    else:
        assert text["en"].startswith(f"{n} register rows are stated limits of the synthetic case")
        assert text["es"].startswith(f"{n} filas del registro son límites declarados del caso sintético")


# ---- 6. section 12: the input reference


def _paths(basis: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"(\S+?)#/", basis)}


@pytest.mark.parametrize("n", [1, 2])
def test_documents_reference_agrees(extract, inventory_doc, findings_doc, n):
    entry, inv, reg, variants = _inputs(extract, inventory_doc, findings_doc)
    wanted = {v["path"] for v in variants}
    for a in entry["attributes"].values():
        wanted |= _paths(str(a.get("basis", "")))
    inv["provenance"]["documents_read"] = [d for d in inv["provenance"]["documents_read"] if d["path"] in wanted][:n]
    assert len(inv["provenance"]["documents_read"]) == n
    sec = _section(build_report(entry, inv, reg, variants, AS_OF), 12)
    noun = "1 document" if n == 1 else "2 documents"
    want = f"provenance of the {noun} read for the entry"
    assert sec["inputs"][0]["ref"] == want


# ---- the plural cases stay as they were (no byte moves for counts of two or more)

GOLDEN = {
    "C07/banco-andino": "f4a5b0ce59cc3095aa98ca292837f2cccbe89e9a19a0136c1e25ed75e37bf89d",
    "C18/banco-andino": "6775a77362e08bc45c0c1f4ab13d194c1a11b3b6248e478c073e6935a9408ec8",
    "C22/generators": "2636e8f48eea28e873b92cab45aa8b79b83611c2bff805849e03c5267d6c1d4a",
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_reports_with_plural_counts_are_byte_identical_to_1_1_0(extract, inventory_doc, findings_doc, key):
    """The three entries built with the 1.1.0 report: their counts are all two or more, so agreement moves nothing."""
    report = build_report(*report_inputs(extract, inventory_doc, findings_doc, key), AS_OF)
    digest = hashlib.sha256(json.dumps(report, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert digest == GOLDEN[key]
