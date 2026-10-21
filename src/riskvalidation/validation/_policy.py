"""The policy label of a battery result: the packaged label, or the label plus ``+custom`` when the caller's cut points
differ from the packaged entry.

Every Contraste battery function writes ``battery_label(POLICY_LABEL, test_id, <the cut points it used>)``. When each
cut point equals the packaged entry's (for a keyed entry, the case the row selects) the label is the packaged label;
otherwise it is ``<label>+custom`` and the cut points it used travel in ``extras["policy_override"]``
(:class:`~riskvalidation.core.result.TestResult` moves them there). A key that is not a cut point of the entry
(``n_sim``, ``seed``, ``lags``, ...) is the test's own argument and is ignored.
"""

from __future__ import annotations

import math
from typing import Any

from riskvalidation.core.policy import _CustomLabel, packaged_policy


def battery_label(label: str, test_id: str, row: Any = None, /, **used: Any) -> str:
    """``label``, or ``label + "+custom"`` carrying the cut points that differ from the packaged entry of ``test_id``.

    ``row`` (a mapping with ``segment`` and ``extras``) selects the case of a keyed entry. ``used`` are the cut
    points the function was called with, by the entry's own names."""
    th = packaged_policy(label).threshold(test_id)
    if th is None:
        return label
    eff = th.for_row(row or {}) or th
    names: dict[str, float] = {"amber": eff.amber, "red": eff.red, **eff.parameters}
    changed: dict[str, float] = {}
    for key, value in used.items():
        if key not in names or isinstance(value, bool):
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isclose(v, names[key], rel_tol=1e-12, abs_tol=1e-15):
            changed[key] = v
    if not changed:
        return label
    return _CustomLabel(
        f"{label}+custom",
        {"label": label, "test_id": test_id, "kind": eff.kind, "parameters": {**dict(eff.parameters), **changed}},
    )
