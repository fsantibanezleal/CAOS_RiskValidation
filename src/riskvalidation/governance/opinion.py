"""The overall opinion on the ECB four-level scale (dossier 06 l.620-623) and the recommendation rule (W15-7,
W15-11).

The level of an entry follows its open findings (``open`` and ``overdue`` states, not ``risk_accepted`` and not
``validated_closed``): the maximum of the findings' levels, 1 with none. A finding's level is the maximum of its
severity level (S1 4, S2 3, S3 and S4 2) and its RWEA level (3 from 5 percent of RWEA, 4 from 10 percent, none
below 5 percent; the cut points are the ECB's, Instructions of February 2019, section 2.4.2), so an anchor can
only raise a level and a critical finding is never downgraded by a small share. The rule that turns findings into
a level is a **convention** (item 30). The recommendation is the rule's output on the evidence; the engine
approves no model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from riskvalidation.governance.conventions import RECOMMENDATION_RULE

LABEL = "the rule's output on the evidence; Contraste approves no model"
RWEA_LEVEL_3 = 0.05
RWEA_LEVEL_4 = 0.10
_SEVERITY_LEVEL = {"S1": 4, "S2": 3, "S3": 2, "S4": 2}
_TEXT = {
    "approve": "Approve: no open finding. This is the rule's output on the evidence; Contraste approves no model.",
    "approve with conditions": (
        "Approve with conditions: only minor findings are open. This is the rule's output on the evidence; "
        "Contraste approves no model."
    ),
    "remediate before use": (
        "Remediate before use: a significant or critical finding is open. This is the rule's output on the evidence; "
        "Contraste approves no model."
    ),
}


def rwea_level(rwea_share: float | None) -> int:
    """The level an RWEA share gives: 3 from 5 percent, 4 from 10 percent, 0 (none) below 5 percent or without a
    share.
    """
    if rwea_share is None:
        return 0
    if not math.isfinite(rwea_share):
        raise ValueError(f"rwea_share must be finite, got {rwea_share}")
    if rwea_share < 0:
        raise ValueError(f"rwea_share must not be negative, got {rwea_share}")
    return 4 if rwea_share >= RWEA_LEVEL_4 else 3 if rwea_share >= RWEA_LEVEL_3 else 0


def finding_level(severity: str, rwea_share: float | None = None) -> int:
    """The level 2 to 4 that one open finding gives: the maximum of its severity level and its RWEA level."""
    if severity not in _SEVERITY_LEVEL:
        raise ValueError(f"unknown severity {severity!r}; expected one of S1, S2, S3, S4")
    return max(_SEVERITY_LEVEL[severity], rwea_level(rwea_share))


def opinion_level(open_findings: Sequence[tuple[str, float | None]]) -> int:
    """The level 1 to 4 over the open findings given as (severity, RWEA share or ``None``) pairs; 1 with none."""
    return max((finding_level(severity, share) for severity, share in open_findings), default=1)


def recommendation(level: int) -> tuple[str, str, str]:
    """(outcome, text, label) of a level 1 to 4; the text states the rule's output and never an approval."""
    if level not in RECOMMENDATION_RULE:
        raise ValueError(f"level must be 1 to 4, got {level!r}")
    outcome = RECOMMENDATION_RULE[level]
    return outcome, _TEXT[outcome], LABEL
