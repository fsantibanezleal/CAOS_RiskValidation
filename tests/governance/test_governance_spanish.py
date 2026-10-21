"""U16-E1 (RV-1706): the Spanish the governance package writes follows the RAE orthography (accents and the eñe)."""

from __future__ import annotations

import re

from riskvalidation.governance import appetite, report

# unaccented spellings of words the package writes; each has an accented form
UNACCENTED = re.compile(
    r"\b(descripcion|implementacion|verificacion|analisis|estres|opinion|recomendacion|ningun|algun|critico|maximo|"
    r"convencion|proporcion|bateria|seccion|limites?|operacion|diseno|peldanos?|evaluacion|calificacion|version|codigo|"
    r"replicacion|declaracion|politica|catalogo|remediacion|validacion|excepcion|ejecucion|linea|sintetico|leidos?|"
    r"direccion|mas|estan)\b",
    re.IGNORECASE,
)


def _spanish_texts() -> list[str]:
    texts = [es for _, es in report._TITLES] + [es for _, es in report._LEVEL_LABEL.values()]
    texts += [report._LABEL_ES, report._RULE[1], *report._REC_ES.values()]
    for name, _kind, _unit, reason in appetite._META.values():
        texts.append(name["es"])
        if reason:
            texts.append(reason["es"])
    return texts


def test_the_engine_spanish_constants_are_accented():
    bad = [(t, m.group(0)) for t in _spanish_texts() for m in UNACCENTED.finditer(t)]
    assert not bad, bad
    joined = " ".join(_spanish_texts())
    for word in ("Descripción", "Implementación", "Análisis", "estrés", "Opinión", "recomendación", "ningún", "crítico",
                 "máximo", "convención", "Proporción", "Remediación", "excepción"):
        assert word in joined, word


def test_the_built_report_spanish_is_accented(extract, inventory_doc, findings_doc):
    from .test_governance_report import _build

    doc = _build(extract, inventory_doc, findings_doc, "C07/banco-andino")
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if set(node) == {"en", "es"} and isinstance(node["es"], str):
                found.append(node["es"])
            else:
                for v in node.values():
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    assert found
    engine = [t for t in found if t.startswith(("Informe de", "Alcance:", "Opinión", "Nivel de opinión", "Semilla"))]
    assert engine and not [t for t in engine if UNACCENTED.search(t)], engine
