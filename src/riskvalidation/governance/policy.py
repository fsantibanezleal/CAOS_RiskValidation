"""The policy identity of a test row (decisions W15-6 and W15-12): the source of every threshold the report states.

A row's ``policy_version`` is either a packaged label (``riskvalidation-default 2026.10.1``), which resolves
through the packaged policy file (:func:`riskvalidation.core.policy.default_policy`), or an inline battery string
(``contraste-<family>-battery-<version>; key=value; ...``), which is parsed into its version and parameters, or a
packaged battery label (``contraste-<family>-battery <version>``, U16-E3: one packaged file per Contraste battery,
``riskvalidation.core.policy.packaged_policy``), which resolves through its file like the default. The
threshold ``kind`` is ``spec(test_id).decision`` (``p_value`` or ``metric``), null with its reason where the test
is not registered; the ``direction`` (``upper_bad`` or ``lower_bad``) is read from the policy file where one
resolves and is null with the reason ``NO_POLICY_FILE`` for an inline battery string. The inline form keeps
resolving so the committed documents stay readable; a packaged battery label resolves through its file.
"""

from __future__ import annotations

import re

from riskvalidation.core.policy import default_policy, packaged_policy
from riskvalidation.core.registry import spec

NO_POLICY_FILE = "no policy file for an inline battery string"
KIND_NOT_REGISTERED = "the test id is not in the engine registry (spec() raises KeyError), so it has no decision"
_PACKAGED = re.compile(r"(?P<id>[A-Za-z][\w.-]*) (?P<version>\d[\w.]*(?:\+[\w.-]+)?)")
_INLINE = re.compile(
    r"contraste-(?P<family>(?!battery-)[a-z0-9]+(?:-[a-z0-9]+)*?)(?:-battery)?-(?P<version>\d+\.\d+\.\d+)"
    r"(?P<params>(?:; [a-z0-9_]+=[^;]+)*)"
)
# the form before U16-E3: the packaged label with the parameters of the test appended, `<label>; key=value; ...`
# (and, for the ml tests, the battery token `ml-battery-1.1.0`); it resolves as the bare label with the parameters read
_APPENDED = re.compile(r"(?P<label>[A-Za-z][\w.-]* \d[\w.]*(?:\+[\w.-]+)?)(?P<rest>(?:; [^;]+)+)")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _kind(test_id: str) -> tuple[str | None, str | None]:
    """(kind, reason): the registry's decision, else null with the reason (no fallback to a threshold)."""
    try:
        return spec(test_id).decision, None
    except KeyError:
        return None, KIND_NOT_REGISTERED


def _parameters(text: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for part in [p for p in text.split("; ") if p]:
        key, value = part.split("=", 1)
        out[key] = float(value) if _NUMBER.fullmatch(value) else value
    return out


def _packaged_identity(policy, text: str, test_id: str, alpha_amber: float | None, alpha_red: float | None) -> dict:
    """The identity of a row whose label is a packaged battery policy: the direction is read from the file."""
    threshold = policy.threshold(test_id)
    kind, kind_reason = _kind(test_id)
    if threshold is None:
        direction = None
        direction_reason = f"the packaged policy {policy.label} holds no packaged threshold for {test_id}"
    elif threshold.kind == "p_value":
        direction, direction_reason = None, "a p-value threshold has no direction"
    elif threshold.rule.partition(":")[0] not in ("", "upper_le", "lower_ge"):
        direction = None
        direction_reason = f"the light of {test_id} is the rule {threshold.rule!r}, not a single direction"
    else:
        direction, direction_reason = threshold.kind, None
    return {
        "label": text,
        "kind": kind,
        "kind_reason": kind_reason,
        "direction": direction,
        "direction_reason": direction_reason,
        "alpha_amber": alpha_amber,
        "alpha_red": alpha_red,
        "parameters": dict(threshold.parameters) if threshold is not None else {},
        "version": policy.version,
        "source": "packaged-policy",
    }


def resolve_policy(
    policy_version: str, test_id: str, alpha_amber: float | None = None, alpha_red: float | None = None
) -> tuple[dict | None, str | None]:
    """The policy identity of a test row, ``(identity, reason)``, exactly one of them ``None``.

    An identity is ``{"label", "kind", "kind_reason", "direction", "direction_reason", "alpha_amber", "alpha_red",
    "parameters", "version", "source"}`` with ``source`` one of ``packaged-policy`` and ``inline-battery``. An
    identity that is neither a packaged label nor an inline battery string returns ``(None, reason)``.
    """
    text = (policy_version or "").strip()
    appended = _APPENDED.fullmatch(text)
    if appended and not _INLINE.fullmatch(text):
        identity, reason = resolve_policy(appended.group("label"), test_id, alpha_amber, alpha_red)
        if identity is None:
            return None, reason
        parts = [p for p in appended.group("rest").split("; ") if p]
        params = _parameters("; ".join(p for p in parts if "=" in p))
        battery = [p for p in parts if "=" not in p]
        return {**identity, "label": text, "parameters": params, **({"battery": battery[0]} if battery else {})}, None
    inline = _INLINE.fullmatch(text)
    if inline:
        kind, kind_reason = _kind(test_id)
        return {
            "label": text,
            "kind": kind,
            "kind_reason": kind_reason,
            "direction": None,
            "direction_reason": NO_POLICY_FILE,
            "alpha_amber": alpha_amber,
            "alpha_red": alpha_red,
            "parameters": _parameters(inline.group("params")),
            "version": inline.group("version"),
            "source": "inline-battery",
        }, None
    packaged = _PACKAGED.fullmatch(text)
    if packaged:
        base = default_policy()
        version = packaged.group("version")
        if packaged.group("id") != base.policy_id:
            try:
                battery = packaged_policy(text)
            except KeyError:
                battery = None
            if battery is not None:
                return _packaged_identity(battery, text, test_id, alpha_amber, alpha_red), None
        # a label with the +gov- suffix is the governance policy: the packaged default plus the gov. rows
        # (until they sit in the file)
        if "+gov-" in version:
            from riskvalidation.validation.governance import gov_default_policy

            policy = gov_default_policy()
        else:
            policy = base
        if packaged.group("id") != base.policy_id or version.split("+")[0] != base.version:
            return None, (
                f"the label {text!r} is not the packaged policy {policy.label!r}, so no threshold can be read for it"
            )
        threshold = policy.threshold(test_id)
        kind, kind_reason = _kind(test_id)
        if threshold is None:
            direction, direction_reason = (
                None,
                f"the packaged policy {policy.label} holds no packaged threshold for {test_id}",
            )
        elif threshold.kind == "p_value":
            direction, direction_reason = None, "a p-value threshold has no direction"
        else:
            direction, direction_reason = threshold.kind, None
        return {
            "label": text,
            "kind": kind,
            "kind_reason": kind_reason,
            "direction": direction,
            "direction_reason": direction_reason,
            "alpha_amber": alpha_amber,
            "alpha_red": alpha_red,
            "parameters": {},
            "version": version,
            "source": "packaged-policy",
        }, None
    return None, (
        f"the policy string {text!r} matches neither a packaged label nor an inline battery string "
        f"(contraste-<family>-battery-<version>; key=value; ...)"
    )

__all__ = [
    "resolve_policy",
]
