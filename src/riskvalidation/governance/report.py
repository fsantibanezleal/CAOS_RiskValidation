"""The validation report (dossier 06 I.3 l.1219-1237) as a structured document of twelve sections, each filled from
the inputs it names or absent with its reason, validating against the schema ``governance-report``.

The report states the rule's output on the evidence and never an approval, a certification or a compliance
(dossier 06 I.6). The texts are bilingual (English and Spanish, the Spanish in standard RAE orthography, with its
accents and the eñe); every number in them is computed here.

Inputs, all mappings read and never mutated:

- ``entry``: the inventory entry in its document form (``key``, ``case_id``, ``fit_id``, ``title``,
  ``attributes``, ``components``, ``tier_block``);
- ``inventory``: the inventory document (its ``provenance.documents_read``, ``as_of_basis``);
- ``register``: the findings register rows of the entry;
- ``variants``: the variant documents of the entry, the first being the default one; each holds ``variant_id``,
  ``path``, ``tests`` (the test rows, as ``TestResult.to_dict()`` writes them) and, where known, ``seed``,
  ``code_version`` and ``riskvalidation_version``.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from riskvalidation.core.result import Light
from riskvalidation.governance.findings import OPEN_STATES
from riskvalidation.governance.opinion import LABEL, opinion_level, recommendation, rwea_level
from riskvalidation.governance.policy import resolve_policy

METHODOLOGY_PAGE = "frontend/src/pages/Methodology.tsx (Methodology page)"
SECTION_KEYS = (
    "executive_summary",
    "model_description",
    "conceptual_soundness",
    "data",
    "implementation",
    "outcomes_analysis",
    "sensitivity_robustness_stress",
    "ongoing_monitoring",
    "limitations_and_controls",
    "findings_register",
    "opinion_and_recommendation",
    "appendices",
)
_TITLES = (
    ("Executive summary", "Resumen ejecutivo"),
    ("Model description and intended use", "Descripción del modelo y uso previsto"),
    ("Conceptual soundness", "Solidez conceptual"),
    ("Data", "Datos"),
    ("Implementation and process verification", "Implementación y verificación del proceso"),
    ("Outcomes analysis", "Análisis de resultados"),
    ("Sensitivity, robustness and stress", "Sensibilidad, robustez y estrés"),
    ("Ongoing monitoring", "Monitoreo continuo"),
    ("Limitations and compensating controls", "Limitaciones y controles compensatorios"),
    ("Findings register", "Registro de hallazgos"),
    ("Overall opinion and recommendation", "Opinión global y recomendación"),
    ("Appendices", "Anexos"),
)
_LEVEL_LABEL = {
    1: ("adequate, no deficiencies", "adecuado, sin deficiencias"),
    2: ("adequate, minor deficiencies", "adecuado, deficiencias menores"),
    3: ("major deficiencies", "deficiencias mayores"),
    4: ("severe deficiencies", "deficiencias severas"),
}
_LABEL_ES = "resultado de la regla sobre la evidencia; Contraste no aprueba ningún modelo"
_REC_ES = {
    "approve": "Aprobar: sin hallazgos abiertos. Es el resultado de la regla sobre la evidencia; Contraste no "
    "aprueba ningún modelo.",
    "approve with conditions": (
        "Aprobar con condiciones: solo hay hallazgos menores abiertos. Es el resultado de la regla sobre la evidencia; "
        "Contraste no aprueba ningún modelo."
    ),
    "remediate before use": (
        "Remediar antes del uso: hay un hallazgo significativo o crítico abierto. Es el resultado de la regla sobre la "
        "evidencia; Contraste no aprueba ningún modelo."
    ),
}
_RULE = (
    "Level 1 with no open finding; 2 with S4 or S3 only; 3 with any S2; 4 with any S1 (open and overdue findings, "
    "not risk_accepted); "
    "where an anchor is a share of RWEA, 5 to 10 percent gives level 3 and 10 percent or more level 4, and an "
    "anchor can only raise a level "
    "(the maximum of the two). A convention (item 30).",
    "Nivel 1 sin hallazgos abiertos; 2 con solo S4 o S3; 3 con algún S2; 4 con algún S1 (hallazgos abiertos y "
    "vencidos, no risk_accepted); "
    "si el ancla es una proporción de RWEA, de 5 a 10 por ciento da nivel 3 y 10 por ciento o más da nivel 4, y un "
    "ancla solo puede subir un "
    "nivel (el máximo de ambos). Una convención (punto 30).",
)
_PATH = re.compile(r"(\S+?)#/")
_POINTER = re.compile(r"\S+#/\S+")
_LIGHTS = tuple(light.value for light in Light)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _t(en: str, es: str) -> dict[str, str]:
    return {"en": en, "es": es}


def executive_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """The counts of the lights ``green``, ``amber``, ``red``, ``not_evaluated`` and ``error`` (the counting of
    ``SuiteRun.counts``).
    """
    counts = dict.fromkeys(_LIGHTS, 0)
    for row in rows:
        light = row["light"]
        if light not in counts:
            raise ValueError(f"unknown light {light!r}; expected one of {_LIGHTS}")
        counts[light] += 1  # type: ignore[index]
    return counts


def _section(
    number: int,
    inputs: Sequence[tuple[str, str]],
    blocks: list[dict],
    missing: Sequence[tuple[tuple[str, str], tuple[str, str]]] = (),
    absent: tuple[str, str] | None = None,
    **extra: Any,
) -> dict:
    filled = absent is None and bool(inputs) and bool(blocks)
    section = {
        "number": number,
        "key": SECTION_KEYS[number - 1],
        "title": _t(*_TITLES[number - 1]),
        "status": "filled" if filled else "absent",
        "inputs": [{"ref": ref, "kind": kind} for ref, kind in inputs] if filled else [],
        "absent_reason": None
        if filled
        else _t(*(absent or ("no input exists for this section", "no existe insumo para esta sección"))),
        "blocks": blocks if filled else [],
        "missing": [{"element": _t(*e), "reason": _t(*r)} for e, r in missing] if filled else [],
    }
    section.update(extra)
    return section


def _text(en: str, es: str) -> dict:
    return {"kind": "text", "text": _t(en, es)}


def _table(caption: tuple[str, str], columns: Sequence[tuple[str, str]], rows: list[list]) -> dict:
    return {"kind": "table", "caption": _t(*caption), "columns": [_t(*c) for c in columns], "rows": rows}


def _first_path(attribute: Mapping[str, Any] | None) -> str | None:
    m = _PATH.search(str((attribute or {}).get("basis", "")))
    return m.group(1) if m else None


def _all_paths(attributes: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {m.group(1) for a in attributes.values() for m in _PATH.finditer(str(a.get("basis", "")))}


def _bilingual(value: Any, fallback: tuple[str, str]) -> dict[str, str]:
    if isinstance(value, Mapping) and value.get("en") and value.get("es"):
        return _t(str(value["en"]), str(value["es"]))
    return _t(*fallback)


def _policy_row(row: Mapping[str, Any]) -> dict:
    version = row.get("policy_version")
    identity, reason = resolve_policy(
        version if isinstance(version, str) else "", row["test_id"], row.get("alpha_amber"), row.get("alpha_red")
    )
    if identity is not None:
        return {**identity, "reason": None}
    return {
        "label": version if isinstance(version, str) else "",
        "kind": None,
        "kind_reason": "unresolved identity",
        "alpha_amber": row.get("alpha_amber"),
        "alpha_red": row.get("alpha_red"),
        "direction": None,
        "direction_reason": "no policy file (BL-085)",
        "parameters": {},
        "version": None,
        "source": "unresolved",
        "reason": reason,
    }


def _outcome_rows(tests: Sequence[Mapping[str, Any]]) -> list[dict]:
    return [
        {
            "test_id": r["test_id"],
            "model_id": r.get("model_id"),
            "segment": r.get("segment"),
            "family": r["test_id"].split(".")[0],
            "statistic": r.get("statistic"),
            "p_value": r.get("p_value"),
            "metric": r.get("metric"),
            "light": r["light"],
            "h0": r.get("h0", ""),
            "reference": r.get("reference", ""),
            "policy": _policy_row(r),
        }
        for r in tests
    ]


def _check_as_of(entry: Mapping[str, Any], register: Sequence[Mapping[str, Any]], as_of: date) -> None:
    dates = [str(r["raised"]) for r in register if r.get("raised")]
    last = (entry.get("schedule") or {}).get("last_validation", {}).get("date")
    if last:
        dates.append(str(last))
    newest = max(dates, default=None)
    if newest is not None and as_of < date.fromisoformat(newest):
        raise ValueError(f"as_of {as_of.isoformat()} is earlier than the newest raised or validation date {newest}")


def build_report(
    entry: Mapping,
    inventory: Mapping,
    register: Sequence[Mapping],
    variants: Sequence[Mapping],
    as_of: date,
    *,
    methodology_page: str = METHODOLOGY_PAGE,
) -> dict:
    """The validation report of one inventory entry: the twelve sections of I.3, deterministic, validating against
    ``governance-report``.

    ``variants`` lists the variant documents of the entry, the first being the default one: section 6 lists its
    test rows with their definition and policy identity. Section 11 is :func:`opinion_level` and
    :func:`recommendation` over the entry's open findings.
    """
    if not variants:
        raise ValueError("the entry has no variant document: the report needs the default variant's test rows")
    _check_as_of(entry, register, as_of)
    key, fit = entry["key"], entry["fit_id"]
    attrs = entry["attributes"]
    tb = entry["tier_block"]
    default = variants[0]
    tests = list(default["tests"])
    title = _bilingual(entry.get("title"), (key, key))
    components = list(entry.get("components", []))
    adopted = next(r for r in tb["tier_sensitivity"] if r["reading"] == tb["adopted_reading"])
    manifest_path = _first_path(attrs.get("A02"))
    models_path = _first_path(attrs.get("A01")) or default["path"]
    n_var = len(variants)

    counts = executive_counts(tests)
    open_rows = [r for r in register if r["status"] in OPEN_STATES]
    pairs = [(r["severity"], r.get("rwea_share")) for r in open_rows]
    level = opinion_level(pairs)
    outcome, rec_en, _ = recommendation(level)
    label_en, label_es = _LEVEL_LABEL[level]
    outcomes = _outcome_rows(tests)

    secs: list[dict] = []
    # 1 executive summary
    question = _bilingual(
        (attrs.get("A04") or {}).get("value"), (f"Validation of {title['en']}", f"Validación de {title['es']}")
    )
    secs.append(
        _section(
            1,
            [
                (f"{manifest_path or 'the case manifest'} (manifest: question)", "document"),
                (f"findings register rows of {key}", "document"),
                (f"test rows of {default['path']}", "document"),
            ],
            [
                {"kind": "text", "text": question},
                _text(
                    f"Scope: the fit {fit} of case {entry['case_id']}; the full battery of the default variant "
                    f"({len(tests)} test rows) and the "
                    f"findings of its {n_var} variants.",
                    f"Alcance: el ajuste {fit} del caso {entry['case_id']}; la batería completa de la variante por "
                    f"defecto ({len(tests)} filas de "
                    f"prueba) y los hallazgos de sus {n_var} variantes.",
                ),
                _table(
                    ("Test lights of the default variant", "Luces de las pruebas de la variante por defecto"),
                    [("Light", "Luz"), ("Rows", "Filas")],
                    [[k, v] for k, v in counts.items()],
                ),
                _text(
                    f"Overall opinion: level {level} of 4 on the ECB scale ({label_en}); see section 11.",
                    f"Opinión global: nivel {level} de 4 en la escala del BCE ({label_es}); ver sección 11.",
                ),
            ],
            [
                (
                    ("Conditions of use", "Condiciones de uso"),
                    (
                        "no structured list of conditions of use exists (A D5 section 1)",
                        "no existe una lista estructurada de condiciones de uso (A D5 sección 1)",
                    ),
                )
            ],
            opinion_level=level,
        )
    )
    # 2 model description
    mc = f"M {adopted['m']}, C {adopted['c']}, reading {tb['adopted_reading']}"
    mc_es = f"M {adopted['m']}, C {adopted['c']}, lectura {tb['adopted_reading']}"
    secs.append(
        _section(
            2,
            [
                (f"{models_path} (models document)", "document"),
                (f"inventory entry {key} (attributes and components)", "document"),
                ("tier block of the inventory entry (tiering.matrix_tier)", "engine"),
            ],
            [
                _text(
                    f"{title['en']}: {len(components)} model records in the fit {fit}; tier {tb['tier']} ({mc}).",
                    f"{title['es']}: {len(components)} registros de modelo en el ajuste {fit}; nivel {tb['tier']} "
                    f"({mc_es}).",
                ),
                _table(
                    ("Components", "Componentes"),
                    [("Model", "Modelo"), ("Role", "Rol"), ("Rung", "Peldaño")],
                    [[c["model_id"], c["role"], c.get("rung")] for c in components],
                ),
                _table(
                    ("Tier under the four readings", "Nivel bajo las cuatro lecturas"),
                    [("Reading", "Lectura"), ("Tier", "Nivel")],
                    [[s["reading"], s["tier"]] for s in tb["tier_sensitivity"]],
                ),
            ],
            [
                (
                    ("Intended use and operating boundaries", "Uso previsto y límites de operación"),
                    (
                        "no intended-use field exists; the boundaries are not stated for this case (A D5 section 2)",
                        "no existe campo de uso previsto; los límites no se declaran para este caso (A D5 sección 2)",
                    ),
                )
            ],
        )
    )
    # 3 conceptual soundness
    ids = ", ".join(c["model_id"] for c in components)
    secs.append(
        _section(
            3,
            [(f"{models_path} (models document: the rungs)", "document"), (methodology_page, "document")],
            [
                _text(
                    f"The design alternatives are the rungs of the ladder in the models document: {ids}.",
                    f"Las alternativas de diseño son los peldaños de la escalera del documento de modelos: {ids}.",
                )
            ],
            [
                (
                    (
                        "Written soundness assessment and explainability rating",
                        "Evaluación escrita de solidez y calificación de explicabilidad",
                    ),
                    (
                        "no written assessment and no rating exist per case (A D5 section 3)",
                        "no existe evaluación escrita ni calificación por caso (A D5 sección 3)",
                    ),
                )
            ],
        )
    )
    # 4 data
    a14 = (attrs.get("A14") or {}).get("value")
    sources = list(a14.get("sources", [])) if isinstance(a14, Mapping) else []
    secs.append(
        _section(
            4,
            [
                (f"{manifest_path or 'the case manifest'} (manifest: sources)", "document"),
                (f"provenance inputs of {default['path']}", "document"),
            ],
            [
                _text(
                    "Sources: " + (", ".join(sources) if sources else "none (a generated book)") + ".",
                    "Fuentes: " + (", ".join(sources) if sources else "ninguna (un libro generado)") + ".",
                )
            ],
            [
                (
                    ("Representativeness and bias examination", "Representatividad y examen de sesgo"),
                    (
                        "only C01-style drift tests and C21 examine them (A D5 section 4)",
                        "solo las pruebas de deriva estilo C01 y C21 las examinan (A D5 sección 4)",
                    ),
                )
            ],
        )
    )
    # 5 implementation
    seed, code_v, rv_v = default.get("seed"), default.get("code_version"), default.get("riskvalidation_version")
    facts = [
        (f"Seed {seed}", f"Semilla {seed}") if seed is not None else None,
        (f"code version {code_v}", f"versión de código {code_v}") if code_v else None,
        (f"riskvalidation {rv_v}", f"riskvalidation {rv_v}") if rv_v else None,
    ]
    facts = [f for f in facts if f]
    block5 = (
        [
            _text(
                _cap(", ".join(f[0] for f in facts)) + "; every document records them.",
                _cap(", ".join(f[1] for f in facts)) + "; cada documento las registra.",
            )
        ]
        if facts
        else []
    )
    secs.append(
        _section(
            5,
            [(f"provenance of {models_path}: seed, code_version, riskvalidation_version", "document")],
            block5,
            [
                (
                    ("Independent replication of the pipeline", "Replicación independiente del pipeline"),
                    (
                        "no independent replication statement exists (A D5 section 5)",
                        "no existe declaración de replicación independiente (A D5 sección 5)",
                    ),
                )
            ],
            absent=(
                "the variant documents record no seed, code version or engine version",
                "los documentos de variante no registran semilla, versión de código ni versión del motor",
            )
            if not facts
            else None,
        )
    )
    # 6 outcomes analysis
    unresolved = sum(1 for r in outcomes if r["policy"]["kind"] is None)
    no_direction = sum(1 for r in outcomes if r["policy"]["direction"] is None)
    missing6 = []
    if unresolved or no_direction:
        missing6.append(
            (
                ("Threshold kind for rows without a policy file", "Tipo de umbral para filas sin archivo de política"),
                (
                    f"the kind is spec(test_id).decision where the test is registered ({unresolved} of "
                    f"{len(outcomes)} rows are of tests "
                    f"that are not: spec() raises KeyError), and the direction is null for {no_direction} rows "
                    f"until a policy file exists "
                    f"(BL-085)",
                    f"el tipo es spec(test_id).decision donde la prueba está registrada ({unresolved} de "
                    f"{len(outcomes)} filas son de "
                    f"pruebas que no lo están: spec() lanza KeyError), y la dirección es nula en {no_direction} "
                    f"filas hasta que exista "
                    f"un archivo de política (BL-085)",
                ),
            )
        )
    secs.append(
        _section(
            6,
            [
                (f"test rows of {default['path']}", "document"),
                ("policy identity of each row (policy.resolve_policy)", "engine"),
            ],
            [{"kind": "outcomes", "variant_id": default["variant_id"], "rows": outcomes}] if outcomes else [],
            missing6,
            absent=("the default variant holds no test row", "la variante por defecto no tiene filas de prueba")
            if not outcomes
            else None,
        )
    )
    # 7 sensitivity, robustness and stress
    secs.append(
        _section(
            7,
            [(f"the {n_var} variants of the entry", "document")],
            [
                _table(
                    ("Variants as perturbations", "Variantes como perturbaciones"),
                    [("Variant", "Variante"), ("Test rows", "Filas"), ("Red rows", "Filas rojas")],
                    [
                        [v["variant_id"], len(v["tests"]), sum(1 for t in v["tests"] if t["light"] == "red")]
                        for v in variants
                    ],
                )
            ],
        )
    )
    # 8 ongoing monitoring (the PMA overlays where the entry holds them)
    a26 = attrs.get("A26") or {}
    pma = a26.get("value") if a26.get("status") == "present" else None
    if isinstance(pma, Mapping) and pma.get("overlays_total") is not None and pma.get("model_output") is not None:
        pointer = (_POINTER.search(str(a26.get("basis", ""))) or [str(a26.get("basis", ""))])[0]
        cur = pma.get("currency", "")
        secs.append(
            _section(
                8,
                [(f"PMA overlays_total at {pointer}", "document")],
                [
                    _text(
                        f"PMAs in force: overlays of {pma['overlays_total']:.0f} {cur} on a model output of "
                        f"{pma['model_output']:.2f} {cur}.",
                        f"PMA vigentes: ajustes por {pma['overlays_total']:.0f} {cur} sobre un resultado del modelo "
                        f"de {pma['model_output']:.2f} {cur}.",
                    )
                ],
                [
                    (
                        ("Monitoring pack and overrides analysis", "Paquete de monitoreo y análisis de overrides"),
                        (
                            "no monitoring pack exists and the word override appears in no committed document (A D5 "
                            "section 8)",
                            "no existe paquete de monitoreo y la palabra override no aparece en ningún documento "
                            "confirmado (A D5 sección 8)",
                        ),
                    )
                ],
            )
        )
    else:
        secs.append(
            _section(
                8,
                [],
                [],
                absent=(
                    "no monitoring pack, no overrides analysis and no PMA exists for this entry (A D5 section 8)",
                    "no existe paquete de monitoreo, análisis de overrides ni PMA para esta entrada (A D5 sección 8)",
                ),
            )
        )
    # 9 limitations and controls
    accepted = [r for r in register if r["status"] == "risk_accepted"]
    secs.append(
        _section(
            9,
            [(f"findings register rows of {key} with status risk_accepted", "document")],
            [
                _text(
                    f"{len(accepted)} register rows are stated limits of the synthetic case (risk_accepted); use "
                    f"restrictions are the SDD non-goals.",
                    f"{len(accepted)} filas del registro son límites declarados del caso sintético (risk_accepted); "
                    f"las restricciones de uso son los "
                    f"no-objetivos del SDD.",
                )
            ],
            [
                (
                    (
                        "Compensating-control catalogue and use restrictions as data",
                        "Catálogo de controles compensatorios y restricciones de uso como datos",
                    ),
                    ("neither exists as data (A D5 section 9)", "ninguno existe como datos (A D5 sección 9)"),
                )
            ],
        )
    )
    # 10 findings register
    secs.append(
        _section(
            10,
            [(f"findings register rows of {key}", "document")],
            [
                _table(
                    ("Findings of the entry", "Hallazgos de la entrada"),
                    [
                        ("Key", "Clave"),
                        ("Severity", "Severidad"),
                        ("ECB F", "F del BCE"),
                        ("Status", "Estado"),
                        ("Due", "Vence"),
                        ("Overdue", "Vencido"),
                    ],
                    [[r["key"], r["severity"], r["ecb_f"], r["status"], r["due"], r["overdue"]] for r in register],
                )
            ]
            if register
            else [],
            [
                (
                    ("Remediation plan text", "Texto del plan de remediación"),
                    (
                        "no remediation plan exists: the findings are open results or stated limits (W15-5)",
                        "no existe plan de remediación: los hallazgos son resultados abiertos o límites declarados "
                        "(W15-5)",
                    ),
                )
            ],
            absent=(
                "the findings register holds no row for this entry",
                "el registro de hallazgos no tiene filas para esta entrada",
            )
            if not register
            else None,
        )
    )
    # 11 opinion and recommendation
    opinion = {
        "level": level,
        "level_label": _t(label_en, label_es),
        "rule": {"id": "W15-7", "text": _t(*_RULE)},
        "rwea_cut_points_applied": any(rwea_level(r.get("rwea_share")) > 0 for r in open_rows),
        "open_findings": [
            {"key": r["key"], "severity": r["severity"], "rwea_share": r.get("rwea_share")} for r in open_rows
        ],
        "recommendation": {"outcome": outcome, "text": _t(rec_en, _REC_ES[outcome])},
        "label": _t(LABEL, _LABEL_ES),
    }
    secs.append(
        _section(
            11,
            [
                ("opinion rule W15-7 (opinion.opinion_level) over the open findings", "engine"),
                ("recommendation text (opinion.recommendation)", "engine"),
            ],
            [
                _text(
                    f"Opinion level {level}: {label_en}. {rec_en}",
                    f"Nivel de opinión {level}: {label_es}. {_REC_ES[outcome]}",
                )
            ],
            opinion=opinion,
        )
    )
    # 12 appendices: the sha256 of every document read for the entry
    wanted = _all_paths(attrs) | {v["path"] for v in variants}
    docs = [d for d in inventory.get("provenance", {}).get("documents_read", []) if d["path"] in wanted]
    secs.append(
        _section(
            12,
            [(f"provenance of the {len(docs)} documents read for the entry", "document")],
            [
                _table(
                    ("Documents read", "Documentos leídos"),
                    [("Path", "Ruta"), ("sha256", "sha256")],
                    [[d["path"], d["sha256"]] for d in docs],
                )
            ]
            if docs
            else [],
            [
                (
                    (
                        "Policy-file version and content hash for the inline battery strings",
                        "Versión y hash del archivo de política de las cadenas en línea",
                    ),
                    (
                        "the contraste-*-battery strings have no policy file (backlog BL-085, done at the U16 re-bake)",
                        "las cadenas contraste-*-battery no tienen archivo de política (BL-085, en la re-ejecución "
                        "de U16)",
                    ),
                )
            ],
            absent=(
                "no document read is recorded for the entry",
                "no se registra ningún documento leído para la entrada",
            )
            if not docs
            else None,
        )
    )

    return {
        "schema": "contraste.governance.report/v1",
        "case_id": "GOVERNANCE",
        "governance_id": "report",
        "schema_version": "1.0.0",
        "as_of": as_of.isoformat(),
        "as_of_basis": copy.deepcopy(inventory["as_of_basis"]),
        "provenance": copy.deepcopy(inventory["provenance"]),
        "entry_key": key,
        "title": _t(f"Validation report: {title['en']}", f"Informe de validación: {title['es']}"),
        "tier": tb["tier"],
        "test_counts": counts,
        "sections": secs,
    }
