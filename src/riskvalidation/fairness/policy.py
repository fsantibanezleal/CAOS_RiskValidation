"""The `fair.` rows of the validation policy: the cut points of the fairness lights (contract `engine-fairness-api`
1.0.0, section Policy thresholds).

The values are conventions, never regulatory thresholds (dossier 06 I.6 l.1268-1270): each is a named parameter for the
main session's approval. A row marked strict states a tolerance that 'exceeds' triggers: the cut point is stored one
float above the tolerance (`math.nextafter`), so a metric equal to the tolerance is not a finding and is green
(`Threshold.classify` is inclusive at its cut points). A red cut of 2.0 is a sentinel above the range of a metric that
is at most 1: that light is never red.

`fairness_default_policy()` is the packaged default policy with these rows added where it holds none; once the rows
sit in `policies/default.json` the two are the same. A policy a caller supplies is used as given: a policy that lacks a
key gives the `not_evaluated` light, never a colour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from riskvalidation.core.policy import Policy, Threshold, default_policy

POLICY_SUFFIX = "fair-1.0.0"
NEVER_RED = 2.0


@dataclass(frozen=True)
class FairRow:
    """One row of the contract's Policy thresholds table."""

    kind: str
    amber: float
    red: float
    strict: bool
    source: str


FAIR_ROWS: dict[str, FairRow] = {
    "fair.demographic_parity": FairRow(
        "upper_bad", 0.05, NEVER_RED, True,
        "convention: the amber tolerance 0.05 on the largest absolute selection-rate gap is proposed, no dossier "
        "value; "
        "strict (a gap equal to the tolerance is green); red 2.0 is a sentinel above the metric's range: never red"),
    "fair.two_proportion_z": FairRow(
        "p_value", 0.05, 0.01, False,
        "convention: the engine's default p-value cut points (5% and 1%); dossier 06 C.7 l.1036-1039"),
    "fair.fisher_exact": FairRow(
        "p_value", 0.05, 0.01, False,
        "convention: the engine's default p-value cut points (5% and 1%); dossier 06 C.7 l.1036-1039"),
    "fair.equalized_odds": FairRow(
        "upper_bad", 0.05, 0.10, False,
        "convention: proposed for the Taiwan battery, no dossier value (06 C.7 l.1029-1033)"),
    "fair.equal_opportunity": FairRow(
        "upper_bad", 0.05, 0.10, False,
        "convention: proposed for the Taiwan battery, no dossier value (06 C.7 l.1029-1033)"),
    "fair.calibration_within_groups": FairRow(
        "upper_bad", 0.05, 0.10, False,
        "convention: proposed for the Taiwan battery, no dossier value (04 J.3 l.904-905)"),
    "fair.auc_by_group": FairRow(
        "upper_bad", 0.05, 0.10, False,
        "convention: proposed for the Taiwan battery, no dossier value (04 J.3 l.904-905)"),
    "fair.situation_testing": FairRow(
        "upper_bad", 0.10, NEVER_RED, False,
        "convention: the amber share 0.10 of significant complainants is proposed, no source value; red 2.0 is a "
        "sentinel above the metric's range: never red"),
}
FAIR_KEYS = tuple(FAIR_ROWS)


def fair_threshold(key: str) -> Threshold:
    """The `Threshold` of a `fair.` row, the strict amber cut stored one float above the tolerance."""
    row = FAIR_ROWS[key]
    amber = math.nextafter(row.amber, math.inf) if row.strict else row.amber
    return Threshold(kind=row.kind, amber=amber, red=row.red, source=row.source)


@lru_cache(maxsize=1)
def fairness_default_policy() -> Policy:
    """The packaged default policy with the `fair.` rows added where it holds none."""
    base = default_policy()
    thresholds = dict(base.thresholds)
    for key in FAIR_KEYS:
        thresholds.setdefault(key, fair_threshold(key))
    return Policy(
        policy_id=base.policy_id,
        version=f"{base.version}+{POLICY_SUFFIX}",
        description=f"{base.description} Fairness rows ({POLICY_SUFFIX}): conventions for the main session's approval, "
        "never regulatory thresholds.",
        thresholds=thresholds,
    )


__all__ = ["FAIR_KEYS", "FAIR_ROWS", "FairRow", "NEVER_RED", "fair_threshold", "fairness_default_policy"]
