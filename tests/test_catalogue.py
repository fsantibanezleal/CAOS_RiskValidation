"""RV-113: every U1 test is registered with its source, and the default policy states each threshold's origin."""

from __future__ import annotations

import json
from importlib import resources

from riskvalidation import catalogue, default_policy

U1 = {
    "pd.jeffreys", "pd.binomial", "pd.binomial_vasicek", "pd.chi2_grades", "pd.hosmer_lemeshow",
    "pd.spiegelhalter", "pd.normal_multiperiod", "pd.traffic_lights", "pd.brier", "pd.ece",
    "disc.auc", "disc.auc_vs_initial", "disc.delong", "disc.ks",
    "rating.mwb", "rating.migration_ztests", "rating.hhi",
    "stability.psi", "stability.csi", "stability.chi2", "stability.ks", "stability.wasserstein",
}


def test_u1_catalogue_complete():
    specs = {s.test_id: s for s in catalogue() if not s.test_id.startswith("selftest.")}
    assert U1 <= set(specs)
    policy = default_policy()
    raw = json.loads(resources.files("riskvalidation.policies").joinpath("default.json").read_text("utf-8"))
    descriptive = set(raw["descriptive_without_thresholds"])
    for tid in U1:
        s = specs[tid]
        assert s.reference.strip() and s.h0.strip(), tid
        th = policy.threshold(tid)
        if s.decision == "p_value":
            assert th is not None and th.source.strip(), f"{tid} has no sourced threshold"
            assert tid not in descriptive
        else:
            assert th is None, f"descriptive {tid} must stay unthresholded"
            assert tid in descriptive
