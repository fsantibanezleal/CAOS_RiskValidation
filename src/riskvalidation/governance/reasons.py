"""The Spanish of the reasons that the governance functions return in English only.

``scope_test``, ``assign_variant``, ``anchor_from_test_row``, ``rwea_share_of_anchor`` and ``resolve_policy`` return
a reason as an English string, as the contract ``engine-governance-api`` states, and that does not change. This
module adds the bilingual form beside them, one pattern for all: :func:`bilingual` takes the English reason a
function returned and gives ``{"en": ..., "es": ...}``, the Spanish in RAE orthography. The English reason is the
key: a fixed sentence is looked up in :data:`REASONS_ES`, and a sentence that carries a value (a source list, a
label, a test id) is matched by one of :data:`TEMPLATES`. A reason that is in neither raises ``ValueError``: a Spanish
text is never invented and an English reason is never passed on as Spanish.
"""

from __future__ import annotations

import re

#: fixed English reasons and their Spanish
REASONS_ES: dict[str, str] = {
    # scope_test
    "In scope: it applies statistical or economic theory, its output is an estimate and it is not generative or "
    "agentic (SR 26-2).": "En alcance: aplica teoría estadística o económica, su resultado es una estimación y no es "
    "generativo ni agéntico (SR 26-2).",
    "Out of scope: it is a generative or agentic AI model (SR 26-2).": "Fuera de alcance: es un modelo de inteligencia "
    "artificial generativo o agéntico (SR 26-2).",
    "Out of scope: it applies no statistical or economic theory (SR 26-2).": "Fuera de alcance: no aplica teoría "
    "estadística ni económica (SR 26-2).",
    "Out of scope: it applies no statistical or economic theory and it is a deterministic rule set (SR 26-2).": "Fuera "
    "de alcance: no aplica teoría estadística ni económica y es un conjunto de reglas deterministas (SR 26-2).",
    "Out of scope: its output is not an estimate (SR 26-2).": "Fuera de alcance: su resultado no es una estimación "
    "(SR 26-2).",
    "In scope: it applies a quantitative method (SS1/23 Principle 1.1).": "En alcance: aplica un método cuantitativo "
    "(SS1/23 Principio 1.1).",
    "Out of scope: it applies no statistical, economic, financial or mathematical theory "
    "(SS1/23 Principle 1.1).": "Fuera "
    "de alcance: no aplica teoría estadística, económica, financiera ni matemática (SS1/23 Principio 1.1).",
    "In scope: it applies theory and its output is an estimate (OSFI E-23).": "En alcance: aplica teoría y su "
    "resultado es una estimación (OSFI E-23).",
    "Out of scope: it applies no statistical, economic, financial or mathematical theory (OSFI E-23).": "Fuera de "
    "alcance: no aplica teoría estadística, económica, financiera ni matemática (OSFI E-23).",
    "Out of scope: its output is not an estimate (OSFI E-23).": "Fuera de alcance: su resultado no es una estimación "
    "(OSFI E-23).",
    "In scope: it applies theory and its output is an estimate (ECB).": "En alcance: aplica teoría y su resultado es "
    "una estimación (BCE).",
    "Out of scope: it applies no statistical, economic, financial or mathematical theory (ECB).": "Fuera de alcance: "
    "no aplica teoría estadística, económica, financiera ni matemática (BCE).",
    "Out of scope: its output is not an estimate (ECB).": "Fuera de alcance: su resultado no es una estimación (BCE).",
    # assign_variant
    "the case has no fit to assign the variant to": "el caso no tiene un ajuste al que asignar la variante",
    "the single models document of the case": "el único documento de modelos del caso",
    "the variant's provenance names no source, so no fit can be determined": "la procedencia de la variante no "
    "nombra fuente alguna, por lo que no se puede determinar el ajuste",
    # anchor_from_test_row and rwea_share_of_anchor
    "the evidence is not a test_id@model row (it is a design:, contract:, rate: or bare test id), so no statistic is "
    "read": "la evidencia no es una fila test_id@modelo (es design:, contract:, rate: o un id de prueba sin modelo), "
    "por lo que no se lee estadístico",
    "the cited test row holds no statistic or metric (or no row matches), so the anchor is null": "la fila de prueba "
    "citada no tiene estadístico ni métrica (o ninguna fila coincide), por lo que el ancla es nula",
    "the anchor is not stated as a share of RWEA or as an RWEA amount, so no RWEA level applies": "el ancla no se "
    "expresa como proporción de RWEA ni como monto de RWEA, por lo que no aplica nivel RWEA",
    # resolve_policy
    "no policy file (BL-085)": "sin archivo de política (BL-085)",
    "unresolved identity": "identidad no resuelta",
    "a p-value threshold has no direction": "un umbral de valor p no tiene dirección",
    "the test id is not in the engine registry at the batch 3 head (spec() raises KeyError), so it has no "
    "decision": "el "
    "id de la prueba no está en el registro del motor en la cabeza del lote 3 (spec() lanza KeyError), por lo que no "
    "tiene decisión",
}

#: English reasons that carry a value: a pattern with named groups and the Spanish template that reuses them
TEMPLATES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"the variant's provenance reads (?P<s>.*), the sources of the fit (?P<f>[^,]*)"),
        "la procedencia de la variante lee {s}, las fuentes del ajuste {f}",
    ),
    (
        re.compile(r"no fit holds all of the variant's sources \((?P<s>.*)\)"),
        "ningún ajuste contiene todas las fuentes de la variante ({s})",
    ),
    (
        re.compile(r"(?P<n>\d+) fits hold all of the variant's sources \((?P<s>.*?)\): (?P<f>.*)"),
        "{n} ajustes contienen todas las fuentes de la variante ({s}): {f}",
    ),
    (
        re.compile(r"the packaged policy (?P<label>.*) holds no packaged threshold for (?P<t>\S+)"),
        "la política empaquetada {label} no contiene umbral empaquetado para {t}",
    ),
    (
        re.compile(
            r"the label (?P<text>.*) is not the packaged policy (?P<label>.*), so no threshold can be read for it "
            r"\(a packaged policy file per battery is backlog BL-085\)"
        ),
        "la etiqueta {text} no es la política empaquetada {label}, por lo que no se puede leer umbral para ella "
        "(un archivo de política empaquetado por batería es el ítem BL-085 del backlog)",
    ),
    (
        re.compile(
            r"the policy string (?P<text>.*) matches neither a packaged label nor an inline battery string "
            r"\(contraste-<family>-battery-<version>; key=value; \.\.\.\)"
        ),
        "la cadena de política {text} no coincide con una etiqueta empaquetada ni con una cadena de batería en línea "
        "(contraste-<family>-battery-<version>; key=value; ...)",
    ),
)


def bilingual(english: str) -> dict[str, str]:
    """``{"en": english, "es": <the Spanish>}`` for an English reason the governance functions return.

    Raises ``ValueError`` for a reason that is in neither :data:`REASONS_ES` nor :data:`TEMPLATES`."""
    if english in REASONS_ES:
        return {"en": english, "es": REASONS_ES[english]}
    for pattern, template in TEMPLATES:
        m = pattern.fullmatch(english)
        if m:
            return {"en": english, "es": template.format(**m.groupdict())}
    raise ValueError(f"reason: no Spanish text is held for {english!r}")


__all__ = ["REASONS_ES", "TEMPLATES", "bilingual"]
