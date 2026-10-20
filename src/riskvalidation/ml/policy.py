"""The cut points of the ``ml.`` lights: the "Policy thresholds" table of the contract ``engine-ml-api`` 1.1.0, as
entries of the engine's ``Policy`` (``Threshold(kind, amber, red)`` keyed by test id).

``Threshold.classify`` is inclusive at its cut points (``upper_bad``: amber at or above ``amber``, red at or above
``red``; ``lower_bad``: amber at or below ``amber``, red at or below ``red``; ``p_value``: red below ``red``, amber
below ``amber``). A rule stated as "exceeds" (a tolerance) keeps a value equal to the tolerance in the better colour, so
such a row stores ``math.nextafter(cut, math.inf)`` as its cut point: the light and the strict boolean
(``robustness_finding``, ``overfit_gap``) agree at the boundary. Every value is a convention, never a regulatory
threshold (dossier 06 I.6 l.1268-1270). The rows are the contract's, kept here because ``policies/default.json`` is the
integrator's file (the same rows can be moved there as they stand); ``ml_policy`` adds them to a base policy only where
the base holds no row for the key.
"""

from __future__ import annotations

import math

from riskvalidation.core.policy import Policy, Threshold, default_policy

CONVENTION = "convention (engine-ml-api 1.1.0, Policy thresholds)"
RULE = "rule, not a convention (engine-ml-api 1.1.0, Policy thresholds)"

ML_KEYS = (
    "ml.robustness",
    "ml.shift_resilience",
    "ml.seed_stability",
    "ml.reason_code_stability",
    "ml.effect_agreement",
    "ml.additivity",
    "ml.overfit",
    "ml.weak_spot",
    "ml.conformal_coverage",
    "ml.leakage_screen.single_feature_auc",
    "ml.leakage_screen.resemblance_auc",
    "ml.leakage_screen.temporal_order",
    "ml.leakage_screen.entity_overlap",
    "ml.data_quality",
)


def strict(cut: float) -> float:
    """The cut point of a rule stated as "exceeds": one float above, so that a value equal to the cut stays green."""
    return math.nextafter(float(cut), math.inf)


def ml_thresholds(
    perturbation_tolerance: float = 0.02,
    overfit_tolerance: float = 0.05,
    weak_spot_multiple: float = 2.0,
    additivity_tolerance: float = 1e-9,
) -> dict[str, Threshold]:
    """The fourteen rows. The four named parameters are the approved values of decision V6 (perturbation tolerance 0.02
    AUC points, overfit tolerance 0.05, weak-spot multiple 2, numeric additivity tolerance 1e-9); the other cut points
    are the contract's proposals (no dossier value) for the main session's approval."""
    tol, over, k, add = perturbation_tolerance, overfit_tolerance, weak_spot_multiple, additivity_tolerance
    return {
        "ml.robustness": Threshold(
            "upper_bad",
            strict(tol),
            strict(2.0 * tol),
            f"AUC degradation at the preset scale: amber above perturbation_tolerance ({tol}), red above "
            f"twice it, a delta equal to the tolerance green; {CONVENTION}",
        ),
        "ml.shift_resilience": Threshold(
            "p_value",
            0.05,
            0.01,
            "Monte Carlo p-value of the AUC drop on the worst subsample by shift score against the drop of random "
            "development subsets of the evaluation size (06 C.7 l.1013-1015 gives the construction "
            f"only): the engine default p-value cut points (alpha_amber 0.05, alpha_red 0.01); {CONVENTION}",
        ),
        "ml.seed_stability": Threshold(
            "upper_bad",
            0.01,
            0.02,
            "standard deviation of the AUC across seeds: proposed, about the sampling standard error "
            f"of a Taiwan holdout AUC (0.008); {CONVENTION}",
        ),
        "ml.reason_code_stability": Threshold(
            "lower_bad",
            0.80,
            0.60,
            f"mean top-four reason overlap across refits: proposed, no dossier value (04 J.2 l.881-883); {CONVENTION}",
        ),
        "ml.effect_agreement": Threshold(
            "lower_bad",
            0.70,
            0.40,
            "Spearman rho of mean absolute SHAP against the range of the ALE curve: proposed, no "
            f"dossier value (06 C.7 l.1006-1008); {CONVENTION}",
        ),
        "ml.additivity": Threshold(
            "upper_bad",
            strict(add),
            strict(add),
            f"maximum Shapley additivity gap: red above the numeric tolerance ({add}); {CONVENTION}",
        ),
        "ml.overfit": Threshold(
            "upper_bad",
            strict(over),
            strict(over),
            f"train minus holdout AUC gap: red above overfit_tolerance ({over}); {CONVENTION}",
        ),
        "ml.weak_spot": Threshold(
            "upper_bad",
            strict(k),
            strict(1e12),
            f"error ratio of the worst slice to the average error: amber above weak_spot_multiple ({k}); "
            f"red 1e12 is a sentinel no ratio reaches, so never red; {CONVENTION}",
        ),
        "ml.conformal_coverage": Threshold(
            "p_value",
            0.05,
            0.01,
            "Beta-Binomial p-value of the coverage against 1 - alpha given the calibration size: the engine default "
            f"p-value cut points (alpha_amber 0.05, alpha_red 0.01); {CONVENTION}",
        ),
        "ml.leakage_screen.single_feature_auc": Threshold(
            "upper_bad",
            strict(0.95),
            strict(0.95),
            f"largest single-feature max(AUC, 1 - AUC): near-perfect above 0.95, no dossier value; {CONVENTION}",
        ),
        "ml.leakage_screen.resemblance_auc": Threshold(
            "upper_bad",
            strict(0.70),
            strict(0.70),
            f"cross-fitted development against out-of-time AUC: proposed, no dossier value; {CONVENTION}",
        ),
        "ml.leakage_screen.temporal_order": Threshold(
            "upper_bad",
            1.0,
            1.0,
            f"feature values dated after the observation date: one is a leak by definition (06 C.7 l.1048-1051); "
            f"{RULE}",
        ),
        "ml.leakage_screen.entity_overlap": Threshold(
            "upper_bad",
            1.0,
            1.0,
            f"any test entity present in training (the indicator 1 when the overlap share is above 0): "
            f"a leak by definition; "
            f"{RULE}",
        ),
        "ml.data_quality": Threshold(
            "upper_bad",
            0.01,
            0.05,
            "largest share of failed checks over the dq_ counts: proposed, no dossier value "
            f"(06 C.7 l.1045-1047); {CONVENTION}",
        ),
    }


def threshold_rows(**parameters: float) -> dict[str, dict[str, float | str]]:
    """The ``ml.`` rows of ``ml_thresholds`` as plain mappings (``kind``, ``amber``, ``red``, ``source``), the form the
    integrator writes into ``policies/default.json`` as it stands: the strict rows hold their ``nextafter`` floats. The
    keyword arguments are those of ``ml_thresholds``."""
    return {
        key: {"kind": t.kind, "amber": float(t.amber), "red": float(t.red), "source": t.source}
        for key, t in ml_thresholds(**parameters).items()
    }


def ml_policy(base: Policy | None = None, **parameters: float) -> Policy:
    """``base`` (the packaged default policy when None) with the ``ml.`` rows of the named parameters. The keyword
    arguments are those of ``ml_thresholds``. A row the base does not hold is added; a row the base holds at its
    packaged default (the packaged policy carries the ``ml.`` rows, so a default row is no customisation) is replaced
    by the parameterised one; a row the base customised is kept as it stands."""
    b = default_policy() if base is None else base
    packaged = ml_thresholds()
    rows = dict(b.thresholds)
    for key, row in ml_thresholds(**parameters).items():
        if key not in rows or rows[key] == packaged[key]:
            rows[key] = row
    return Policy(
        policy_id=b.policy_id,
        version=b.version,
        description=(f"{b.description} The ml. rows are those of engine-ml-api 1.1.0 where it holds none or holds the "
                     "packaged default.").strip(),
        thresholds=rows,
    )
