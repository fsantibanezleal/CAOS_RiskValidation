"""The policy identity of a test row (decisions W15-6 and W15-12): the source of every threshold the report states.

A row's ``policy_version`` is either a packaged label (``riskvalidation-default 2026.10.1``), which resolves
through the packaged policy file (:func:`riskvalidation.core.policy.default_policy`), or an inline battery string
(``contraste-<family>-battery-<version>; key=value; ...``), which is parsed into its version and parameters. The
threshold ``kind`` is ``spec(test_id).decision`` (``p_value`` or ``metric``), null with its reason where the test
is not registered; the ``direction`` (``upper_bad`` or ``lower_bad``) is read from the policy file where one
resolves and is null with the reason "no policy file (BL-085)" for an inline battery string. A packaged policy
file per battery is backlog item BL-085, done when the cases are next re-baked (U16).
"""

from __future__ import annotations

import re

from riskvalidation.core.policy import default_policy
from riskvalidation.core.registry import spec

NO_POLICY_FILE = "no policy file (BL-085)"
KIND_NOT_REGISTERED = (
    "the test id is not in the engine registry at the batch 3 head (spec() raises KeyError), so it has no decision"
)
_PACKAGED = re.compile(r"(?P<id>[A-Za-z][\w.-]*) (?P<version>\d[\w.]*(?:\+[\w.-]+)?)")
_INLINE = re.compile(
    r"contraste-(?P<family>[a-z0-9]+)-battery-(?P<version>\d+\.\d+\.\d+)(?P<params>(?:; [a-z0-9_]+=[^;]+)*)"
)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _kind(test_id: str, packaged_kind: str | None) -> tuple[str | None, str | None]:
    """(kind, reason): the registry's decision, else the packaged threshold's, else null with the reason."""
    try:
        return spec(test_id).decision, None
    except KeyError:
        pass
    if packaged_kind is not None:
        return ("p_value" if packaged_kind == "p_value" else "metric"), None
    return None, KIND_NOT_REGISTERED


def _parameters(text: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for part in [p for p in text.split("; ") if p]:
        key, value = part.split("=", 1)
        out[key] = float(value) if _NUMBER.fullmatch(value) else value
    return out


def resolve_policy(
    policy_version: str, test_id: str, alpha_amber: float | None = None, alpha_red: float | None = None
) -> tuple[dict | None, str | None]:
    """The policy identity of a test row, ``(identity, reason)``, exactly one of them ``None``.

    An identity is ``{"label", "kind", "kind_reason", "direction", "direction_reason", "alpha_amber", "alpha_red",
    "parameters", "version", "source"}`` with ``source`` one of ``packaged-policy`` and ``inline-battery``. An
    identity that is neither a packaged label nor an inline battery string returns ``(None, reason)``.
    """
    text = (policy_version or "").strip()
    inline = _INLINE.fullmatch(text)
    if inline:
        kind, kind_reason = _kind(test_id, None)
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
        policy = default_policy()
        version = packaged.group("version")
        if packaged.group("id") != policy.policy_id or version.split("+")[0] != policy.version:
            return None, (
                f"the label {text!r} is not the packaged policy {policy.label!r}, so no threshold can be read for it "
                f"(a packaged policy file per battery is backlog BL-085)"
            )
        threshold = policy.threshold(test_id)
        kind, kind_reason = _kind(test_id, threshold.kind if threshold is not None else None)
        if threshold is None:
            direction, direction_reason = (
                None,
                f"the packaged policy {policy.label} holds no threshold for {test_id} (a descriptive test)",
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
