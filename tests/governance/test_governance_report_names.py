"""Display names in the validation report (Contraste backlog BL-098, engine 1.2.0).

``build_report`` reads an optional ``names`` block of the inventory entry,
``{"fit": {"en", "es"}, "sources": {<source id>: {"en", "es"}}}``, and prints the names of the fit (sections 1 and 2)
and of the sources (section 4) where it has them, the ids otherwise. A report built without the block is the report of
1.1.0, byte for byte.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from riskvalidation.governance.report import build_report

from .governance_support import AS_OF, report_inputs

GOLDEN = {
    "C07/banco-andino": "f4a5b0ce59cc3095aa98ca292837f2cccbe89e9a19a0136c1e25ed75e37bf89d",
    "C18/banco-andino": "6775a77362e08bc45c0c1f4ab13d194c1a11b3b6248e478c073e6935a9408ec8",
    "C22/generators": "2636e8f48eea28e873b92cab45aa8b79b83611c2bff805849e03c5267d6c1d4a",
}
KEY = "C18/banco-andino"
SOURCES = ["fed-stress-tests", "fed-chargeoff", "eba-esrb-stress-2025", "ngfs-phase-v", "bcch-bde"]


def _digest(report: dict) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _build(extract, inventory_doc, findings_doc, names=..., key=KEY):
    entry, inv, reg, variants = copy.deepcopy(report_inputs(extract, inventory_doc, findings_doc, key))
    if names is not ...:
        entry["names"] = names
    return build_report(entry, inv, reg, variants, AS_OF)


def _scope(report):
    return report["sections"][0]["blocks"][1]["text"]


def _description(report):
    return report["sections"][1]["blocks"][0]["text"]


def _sources(report):
    return report["sections"][3]["blocks"][0]["text"]


NAMES = {
    "fit": {"en": "Andean bank stress fit", "es": "Ajuste de estrés del banco andino"},
    "sources": {
        "fed-stress-tests": {"en": "Federal Reserve stress tests", "es": "Pruebas de estrés de la Reserva Federal"},
        "bcch-bde": {"en": "Central bank of Chile and Bank of Spain", "es": "Banco Central de Chile y Banco de España"},
    },
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_a_report_without_the_block_is_the_1_1_0_report_byte_for_byte(extract, inventory_doc, findings_doc, key):
    assert _digest(_build(extract, inventory_doc, findings_doc, key=key)) == GOLDEN[key]


@pytest.mark.parametrize("empty", [None, {}, {"fit": None, "sources": None}, {"fit": {}, "sources": {}}, "text", []])
def test_an_absent_or_empty_block_changes_no_byte(extract, inventory_doc, findings_doc, empty):
    assert _digest(_build(extract, inventory_doc, findings_doc, empty)) == GOLDEN[KEY]


def test_the_names_are_printed_where_the_ids_were(extract, inventory_doc, findings_doc):
    plain = _build(extract, inventory_doc, findings_doc)
    named = _build(extract, inventory_doc, findings_doc, NAMES)
    assert "the fit Andean bank stress fit of case C18" in _scope(named)["en"]
    assert "el ajuste Ajuste de estrés del banco andino del caso C18" in _scope(named)["es"]
    assert "in the fit Andean bank stress fit; tier" in _description(named)["en"]
    assert "en el ajuste Ajuste de estrés del banco andino; nivel" in _description(named)["es"]
    assert _sources(named)["en"] == (
        "Sources: Federal Reserve stress tests, fed-chargeoff, eba-esrb-stress-2025, ngfs-phase-v, "
        "Central bank of Chile and Bank of Spain."
    )
    assert _sources(named)["es"] == (
        "Fuentes: Pruebas de estrés de la Reserva Federal, fed-chargeoff, eba-esrb-stress-2025, ngfs-phase-v, "
        "Banco Central de Chile y Banco de España."
    )
    assert _sources(plain)["en"] == "Sources: " + ", ".join(SOURCES) + "."


def test_only_the_named_sentences_differ_from_the_plain_report(extract, inventory_doc, findings_doc):
    plain = _build(extract, inventory_doc, findings_doc)
    named = _build(extract, inventory_doc, findings_doc, NAMES)
    for report in (plain, named):
        report["sections"][0]["blocks"][1] = report["sections"][1]["blocks"][0] = None
        report["sections"][3]["blocks"][0] = None
    assert plain == named


def test_a_missing_language_or_id_falls_back_to_the_id_never_fails(extract, inventory_doc, findings_doc):
    names = {
        "fit": {"en": "Andean bank stress fit"},  # no Spanish
        "sources": {
            "fed-stress-tests": {"es": "Reserva Federal"},  # no English
            "fed-chargeoff": {"en": "", "es": None},  # empty and null
            "ngfs-phase-v": "not a mapping",
            "unknown-source": {"en": "never cited", "es": "nunca citada"},  # an id the entry does not cite
        },
    }
    report = _build(extract, inventory_doc, findings_doc, names)
    assert "the fit Andean bank stress fit of case C18" in _scope(report)["en"]
    assert "el ajuste banco-andino del caso C18" in _scope(report)["es"]
    assert _sources(report)["en"] == "Sources: " + ", ".join(SOURCES) + "."
    assert _sources(report)["es"] == (
        "Fuentes: Reserva Federal, fed-chargeoff, eba-esrb-stress-2025, ngfs-phase-v, bcch-bde."
    )
    assert "never cited" not in json.dumps(report)


def test_a_block_with_only_one_part_names_only_that_part(extract, inventory_doc, findings_doc):
    only_fit = _build(extract, inventory_doc, findings_doc, {"fit": NAMES["fit"]})
    assert "Andean bank stress fit" in _scope(only_fit)["en"]
    assert _sources(only_fit)["en"] == "Sources: " + ", ".join(SOURCES) + "."
    only_sources = _build(extract, inventory_doc, findings_doc, {"sources": NAMES["sources"]})
    assert "the fit banco-andino of case C18" in _scope(only_sources)["en"]
    assert "Federal Reserve stress tests" in _sources(only_sources)["en"]


def test_a_generated_book_without_sources_keeps_its_phrase(extract, inventory_doc, findings_doc):
    named = _build(extract, inventory_doc, findings_doc, NAMES, key="C07/banco-andino")
    assert _sources(named)["en"] == "Sources: none (a generated book)."
    assert _sources(named)["es"] == "Fuentes: ninguna (un libro generado)."


def test_the_entry_is_not_mutated_and_the_names_keep_their_accents(extract, inventory_doc, findings_doc):
    entry, inv, reg, variants = copy.deepcopy(report_inputs(extract, inventory_doc, findings_doc, KEY))
    entry["names"] = copy.deepcopy(NAMES)
    before = copy.deepcopy(entry)
    report = build_report(entry, inv, reg, variants, AS_OF)
    assert entry == before
    assert "España" in json.dumps(report, ensure_ascii=False)
    assert "ñ" not in json.dumps(report, ensure_ascii=True)  # escaped, not lost
