"""The named convention parameters of the governance package (contract `engine-governance-api` 1.0.0, section
Conventions).

Every table below is a **convention**: a named parameter that carries the label `convention` and its basis, passed
to the function that uses it as an argument with the default stated here. None is presented as a value taken from
a regulation (dossier 06, item 30 of the UNVERIFIED list, l.1321-1322). The one exception in kind is the pair of
ECB cut points of an RWEA anchor in `opinion` (Instructions of February 2019, section 2.4.2), which are the ECB's;
the rule that turns findings into a level is a convention here as well.

This module imports nothing: it is the leaf every other governance module reads its defaults from.
"""

from __future__ import annotations

CONVENTION = "convention"

# 06 B.3: M is banded at "configurable cut-offs" and fixes none; the share of the stated balance at 25, 10 and 2
# percent (A D2 3.2, W15-3).
EXPOSURE_BAND_CUTS: tuple[float, float, float] = (0.25, 0.10, 0.02)

# 06 B.7 names no days: days from the raised date, severity S1 to S4 by tier 1 to 4 (W15-5).
DEADLINE_DAYS: dict[str, tuple[int, int, int, int]] = {
    "S1": (30, 45, 60, 90),
    "S2": (60, 90, 120, 180),
    "S3": (90, 120, 180, 270),
    "S4": (180, 270, 365, 365),
}

# 06 B.8 l.636-640: tier to (review, full revalidation, monitoring) in calendar months; tier 4 is an attestation.
FREQUENCY: dict[int, tuple[int, int | None, int | None]] = {
    1: (12, 36, 3),
    2: (12, 36, 6),
    3: (24, None, 12),
    4: (36, None, None),
}

# 06 B.6 l.606-612: (direction, green limit, amber limit) per metric id; the limits are inclusive on the good side.
APPETITE_THRESHOLDS: dict[str, tuple[str, float, float]] = {
    "tier1-validated-on-time": ("higher-is-better", 0.9, 0.75),
    "open-high-severity": ("lower-is-better", 0, 3),
    "overdue-remediation": ("lower-is-better", 0, 3),
    "entries-under-exception": ("lower-is-better", 0, 1),
    "entries-in-breach": ("lower-is-better", 0, 1),
    "pma-magnitude": ("lower-is-better", 0.01, 0.03),
    "owner-and-complete-record": ("higher-is-better", 0.9, 0.5),
}

# W15-3: the grades of regulatory use; the literal reading is never below the graded one.
REGULATORY_USE_GRADES = {
    "graded": {"reported_figure": 4, "feeding_parameter": 3, "compliance_monitoring": 2, "none": 1},
    "literal": {"feeds_capital_provisioning_or_reporting": 4, "compliance_monitoring": 2, "none": 1},
}

# 06 l.567: C is the mean of the factor scores; the rounding is not in the text, half up is used (A D2 3.3).
COMPLEXITY_AGGREGATION = {"aggregation": "mean", "rounding": "half-up"}

# W15-3: regulatory use graded, complexity the mean rounded half up; the other three readings are shown beside it.
ADOPTED_READING = "B"

# 06 B.4 l.576-583 synthesised from SS1/23 para 2.10, OSFI E-23 and SR 11-7: the allowed transitions of the twelve
# states.
LIFECYCLE_GRAPH: dict[str, tuple[str, ...]] = {
    "proposed": ("in_development",),
    "in_development": ("submitted_for_validation",),
    "submitted_for_validation": ("in_validation",),
    "in_validation": ("approved", "approved_with_conditions", "in_development"),
    "approved": ("in_use",),
    "approved_with_conditions": ("in_use",),
    "in_use": ("under_monitoring_alert", "material_change_pending", "in_validation", "restricted", "decommissioned"),
    "under_monitoring_alert": ("in_use", "restricted", "in_remediation"),
    "restricted": ("in_remediation", "in_use", "decommissioned"),
    "in_remediation": ("in_validation", "restricted", "decommissioned"),
    "material_change_pending": ("submitted_for_validation", "in_use"),
    "decommissioned": (),
}

# 06 l.624-628 names the seven states and no transition: the graph is this contract's proposal.
FINDING_GRAPH: dict[str, tuple[str, ...]] = {
    "open": ("action_plan_agreed", "risk_accepted", "overdue"),
    "action_plan_agreed": ("in_progress", "risk_accepted", "overdue"),
    "in_progress": ("submitted_for_closure", "risk_accepted", "overdue"),
    "submitted_for_closure": ("validated_closed", "in_progress", "overdue"),
    "validated_closed": (),
    "overdue": ("in_progress", "submitted_for_closure", "risk_accepted"),
    "risk_accepted": ("open",),
}

# W15-7: the level of an entry follows its open findings; the recommendation is the rule's text for the level.
RECOMMENDATION_RULE = {
    1: "approve",
    2: "approve with conditions",
    3: "remediate before use",
    4: "remediate before use",
}

CONVENTIONS: dict[str, dict[str, str]] = {
    "EXPOSURE_BAND_CUTS": {
        "label": CONVENTION,
        "basis": "share of the stated balance at least 25 percent is band 4, 10 percent 3, 2 percent 2, below that "
        "1; dossier 06 B.3 fixes no "
        "cut-offs (item 30 of the UNVERIFIED list; A D2 3.2, W15-3)",
    },
    "DEADLINE_DAYS": {
        "label": CONVENTION,
        "basis": "days from the raised date by severity S1 to S4 and tier 1 to 4; no source prescribes days (06 "
        "B.7), so the table is a "
        "proposal (item 30 of the UNVERIFIED list; W15-5)",
    },
    "FREQUENCY": {
        "label": CONVENTION,
        "basis": "06 B.8 l.636-640: Tier 1 annual review, full revalidation every 3 years, quarterly monitoring; "
        "Tier 2 annual, 3 years, "
        "semi-annual; Tier 3 every 2 years with annual monitoring; Tier 4 an attestation every 3 years; "
        "configurable, and the "
        "ECB minimums of 12 and 36 months replace it for the profile `ecb` (item 30)",
    },
    "APPETITE_THRESHOLDS": {
        "label": CONVENTION,
        "basis": "06 B.6 l.606-612 names the metrics and no limits; each limit is a proposal with a direction, a "
        "green limit and an amber "
        "limit, inclusive on the good side (item 30)",
    },
    "REGULATORY_USE_GRADES": {
        "label": CONVENTION,
        "basis": "W15-3: graded 4 where the output is itself the reported or binding figure, 3 where it is a "
        "parameter or a back-test that "
        "feeds one, 2 for compliance monitoring, 1 for none; literal 4 whenever the output feeds capital, "
        "provisioning or "
        "reporting; the literal grade is never below the graded one (A D2 3.2)",
    },
    "COMPLEXITY_AGGREGATION": {
        "label": CONVENTION,
        "basis": "06 l.567: C is the mean of the factor scores; the rounding is not in the text, half up is used "
        "and the maximum is the "
        "alternative reading (A D2 3.3)",
    },
    "ADOPTED_READING": {
        "label": CONVENTION,
        "basis": "W15-3: the reading B (regulatory use graded, complexity the mean rounded half up) sets the tier; "
        "the other three readings "
        "are shown beside it by `tier_sensitivity`",
    },
    "LIFECYCLE_GRAPH": {
        "label": CONVENTION,
        "basis": "06 B.4 l.576-583, a synthesis of SS1/23 para 2.10, OSFI E-23 and SR 11-7; use before validation "
        "only with an exception "
        "record (type, approver, expiry date)",
    },
    "FINDING_GRAPH": {
        "label": CONVENTION,
        "basis": "06 l.624-628 names the seven states and no transition; the graph is this contract's proposal "
        "(item 30)",
    },
    "RECOMMENDATION_RULE": {
        "label": CONVENTION,
        "basis": "W15-7: level 1 with no open finding, 2 with S4 or S3 only, 3 with any S2, 4 with any S1; an RWEA "
        "anchor can only raise a "
        "level (the 5 and 10 percent cut points are the ECB's, Instructions of February 2019, section 2.4.2); the "
        "recommendation is the rule's output on the evidence (item 30)",
    },
}
