"""The reason codes of the governance functions, and the Spanish of their English reasons.

``scope_test``, ``assign_variant``, ``anchor_from_test_row``, ``rwea_share_of_anchor`` and ``resolve_policy`` return
a reason as an English string, and that return value does not change. The **stable key** of a reason is its
**code** (``"anchor.no_cited_row"``, ``"policy.no_threshold"``, ...): codes are public API, and the English
sentence is human text that a minor release may reword. Each code has an English template and a Spanish template
(RAE orthography); a template of a reason that carries a value names it in braces (``{sources}``, ``{test_id}``).

:func:`bilingual` gives ``{"en": ..., "es": ...}`` for a code (with its values as keywords) or, as a convenience,
for the English sentence a function returned. :func:`code_of` gives ``(code, values)`` of an English sentence. A
reason with no code, and a code with a missing value, raise ``ValueError``: a Spanish text is never invented and an
English reason is never passed on as Spanish.
"""

from __future__ import annotations

import re

#: code -> (English template, Spanish template)
REASONS: dict[str, tuple[str, str]] = {
    # scope_test
    "scope.us.in": (
        "In scope: it applies statistical or economic theory, its output is an estimate and it is not generative or "
        "agentic (SR 26-2).",
        "En alcance: aplica teoría estadística o económica, su resultado es una estimación y no es generativo ni "
        "agéntico (SR 26-2).",
    ),
    "scope.us.generative": (
        "Out of scope: it is a generative or agentic AI model (SR 26-2).",
        "Fuera de alcance: es un modelo de inteligencia artificial generativo o agéntico (SR 26-2).",
    ),
    "scope.us.no_theory": (
        "Out of scope: it applies no statistical or economic theory (SR 26-2).",
        "Fuera de alcance: no aplica teoría estadística ni económica (SR 26-2).",
    ),
    "scope.us.no_theory_deterministic": (
        "Out of scope: it applies no statistical or economic theory and it is a deterministic rule set (SR 26-2).",
        "Fuera de alcance: no aplica teoría estadística ni económica y es un conjunto de reglas deterministas "
        "(SR 26-2).",
    ),
    "scope.us.not_estimate": (
        "Out of scope: its output is not an estimate (SR 26-2).",
        "Fuera de alcance: su resultado no es una estimación (SR 26-2).",
    ),
    "scope.pra.in": (
        "In scope: it applies a quantitative method (SS1/23 Principle 1.1).",
        "En alcance: aplica un método cuantitativo (SS1/23 Principio 1.1).",
    ),
    "scope.pra.no_theory": (
        "Out of scope: it applies no statistical, economic, financial or mathematical theory (SS1/23 Principle 1.1).",
        "Fuera de alcance: no aplica teoría estadística, económica, financiera ni matemática (SS1/23 Principio 1.1).",
    ),
    "scope.osfi.in": (
        "In scope: it applies theory and its output is an estimate (OSFI E-23).",
        "En alcance: aplica teoría y su resultado es una estimación (OSFI E-23).",
    ),
    "scope.osfi.no_theory": (
        "Out of scope: it applies no statistical, economic, financial or mathematical theory (OSFI E-23).",
        "Fuera de alcance: no aplica teoría estadística, económica, financiera ni matemática (OSFI E-23).",
    ),
    "scope.osfi.not_estimate": (
        "Out of scope: its output is not an estimate (OSFI E-23).",
        "Fuera de alcance: su resultado no es una estimación (OSFI E-23).",
    ),
    "scope.ecb.in": (
        "In scope: it applies theory and its output is an estimate (ECB).",
        "En alcance: aplica teoría y su resultado es una estimación (BCE).",
    ),
    "scope.ecb.no_theory": (
        "Out of scope: it applies no statistical, economic, financial or mathematical theory (ECB).",
        "Fuera de alcance: no aplica teoría estadística, económica, financiera ni matemática (BCE).",
    ),
    "scope.ecb.not_estimate": (
        "Out of scope: its output is not an estimate (ECB).",
        "Fuera de alcance: su resultado no es una estimación (BCE).",
    ),
    # assign_variant
    "variant.no_fit": (
        "the case has no fit to assign the variant to",
        "el caso no tiene un ajuste al que asignar la variante",
    ),
    "variant.single_document": ("the single models document of the case", "el único documento de modelos del caso"),
    "variant.no_source": (
        "the variant's provenance names no source, so no fit can be determined",
        "la procedencia de la variante no nombra fuente alguna, por lo que no se puede determinar el ajuste",
    ),
    "variant.reads": (
        "the variant's provenance reads {sources}, the sources of the fit {fit}",
        "la procedencia de la variante lee {sources}, las fuentes del ajuste {fit}",
    ),
    "variant.no_holder": (
        "no fit holds all of the variant's sources ({sources})",
        "ningún ajuste contiene todas las fuentes de la variante ({sources})",
    ),
    "variant.many_holders": (
        "{n} fits hold all of the variant's sources ({sources}): {fits}",
        "{n} ajustes contienen todas las fuentes de la variante ({sources}): {fits}",
    ),
    # anchor_from_test_row and rwea_share_of_anchor
    "anchor.no_cited_row": (
        "the evidence is not a test_id@model row (it is a design:, contract:, rate: or bare test id), so no statistic "
        "is read",
        "la evidencia no es una fila test_id@modelo (es design:, contract:, rate: o un id de prueba sin modelo), por "
        "lo que no se lee estadístico",
    ),
    "anchor.no_value_in_row": (
        "the cited test row holds no statistic or metric (or no row matches), so the anchor is null",
        "la fila de prueba citada no tiene estadístico ni métrica (o ninguna fila coincide), por lo que el ancla es "
        "nula",
    ),
    "anchor.not_rwea": (
        "the anchor is not stated as a share of RWEA or as an RWEA amount, so no RWEA level applies",
        "el ancla no se expresa como proporción de RWEA ni como monto de RWEA, por lo que no aplica nivel RWEA",
    ),
    # resolve_policy
    "policy.no_policy_file": (
        "no policy file for an inline battery string",
        "sin archivo de política para una cadena de batería en línea",
    ),
    "policy.unresolved_identity": ("unresolved identity", "identidad no resuelta"),
    "policy.p_value_no_direction": (
        "a p-value threshold has no direction",
        "un umbral de valor p no tiene dirección",
    ),
    "policy.kind_not_registered": (
        "the test id is not in the engine registry (spec() raises KeyError), so it has no decision",
        "el id de la prueba no está en el registro del motor (spec() lanza KeyError), por lo que no tiene decisión",
    ),
    "policy.no_threshold": (
        "the packaged policy {label} holds no packaged threshold for {test_id}",
        "la política empaquetada {label} no contiene umbral empaquetado para {test_id}",
    ),
    "policy.rule_not_single_direction": (
        "the light of {test_id} is the rule {rule}, not a single direction",
        "la luz de {test_id} es la regla {rule}, no una sola dirección",
    ),
    "policy.label_not_packaged": (
        "the label {text} is not the packaged policy {label}, so no threshold can be read for it",
        "la etiqueta {text} no es la política empaquetada {label}, por lo que no se puede leer umbral para ella",
    ),
    "policy.override_missing": (
        "the label {label} is a customised packaged policy but the row carries no extras.policy_override",
        "la etiqueta {label} es una política empaquetada personalizada pero la fila no trae extras.policy_override",
    ),
    "policy.string_unmatched": (
        "the policy string {text} matches neither a packaged label nor an inline battery string "
        "(contraste-<family>-battery-<version>; key=value; ...)",
        "la cadena de política {text} no coincide con una etiqueta empaquetada ni con una cadena de batería en línea "
        "(contraste-<family>-battery-<version>; key=value; ...)",
    ),
}

# the part of a sentence a named value may match; the default is any text
_FIELD = {"n": r"\d+", "test_id": r"\S+", "fit": r"[^,]*", "sources": r".*?"}
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _compile(template: str) -> re.Pattern[str]:
    out, last = [], 0
    for m in _PLACEHOLDER.finditer(template):
        out.append(re.escape(template[last : m.start()]))
        out.append(f"(?P<{m.group(1)}>{_FIELD.get(m.group(1), '.*?')})")
        last = m.end()
    out.append(re.escape(template[last:]))
    return re.compile("".join(out))


_PATTERNS: dict[str, re.Pattern[str]] = {code: _compile(en) for code, (en, _) in REASONS.items()}


def codes() -> tuple[str, ...]:
    """Every reason code, in the order of the table."""
    return tuple(REASONS)


def code_of(english: str) -> tuple[str, dict[str, str]]:
    """``(code, values)`` of an English reason a governance function returned.

    Raises ``ValueError`` for a sentence that has no code."""
    for code, pattern in _PATTERNS.items():
        m = pattern.fullmatch(english)
        if m:
            return code, m.groupdict()
    raise ValueError(f"reason: no code is held for {english!r}")


def bilingual(reason: str, **values: object) -> dict[str, str]:
    """``{"en": ..., "es": ...}`` for a reason code, with the values of its template as keywords.

    As a convenience ``reason`` may instead be the English sentence a function returned (its values are read from
    it). Raises ``ValueError`` for a reason with no code and for a code whose value is missing."""
    if reason in REASONS:
        code = reason
    else:
        code, found = code_of(reason)
        values = {**found, **values}
    en, es = REASONS[code]
    try:
        return {"en": en.format(**values), "es": es.format(**values)}
    except KeyError as exc:
        raise ValueError(f"reason: the code {code!r} needs the value {exc.args[0]!r}") from exc


__all__ = ["REASONS", "bilingual", "code_of", "codes"]
