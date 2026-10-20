"""The ML validation tests (dossier 06 C.7 l.979-1058 and I.1 l.1184-1200; dossier 04 J.2 l.881-883): each returns one
``TestResult`` that validates against ``common/test-result``; contract ``engine-ml-api`` 1.1.0, module
``validation.ml``.

Test ids are prefixed ``ml.``. The lights come from the engine's ``Policy`` (``Threshold(kind, amber, red)`` keyed by
test id) with a ``policy`` argument whose None means the packaged default policy completed, for the keys it does not
hold, by the rows of ``riskvalidation.ml.policy`` (the contract's Policy thresholds table). A policy that lacks a key
returns the light ``not_evaluated``, never a colour. A rule stated as "exceeds" keeps a value equal to the tolerance in
the better colour (the row stores one float above the cut), so a light and the strict boolean of the pure function agree
at the boundary. The tolerances, grids and cut points are conventions, never regulatory thresholds (06 I.6 l.1268-1270);
each function records its parameters in ``policy_version`` and in ``extras["thresholds"]``. A test that cannot be
evaluated returns ``not_evaluated`` with its reason in ``notes`` and a null statistic, metric and p-value once
serialised, never a number. The tests compute and state no compliance claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.policy import Policy, Threshold
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.ml import policy as _policy
from riskvalidation.ml.cbpe import CbpeAuc
from riskvalidation.ml.challenger import PairedBrier
from riskvalidation.ml.conformal import CoverageResult
from riskvalidation.ml.dataquality import DataQualityCount
from riskvalidation.ml.explanations import OverlapResult
from riskvalidation.ml.generalisation import OverfitResult, WeakSpot
from riskvalidation.ml.leakage import AucResult, SingleFeatureAuc
from riskvalidation.ml.robustness import RobustnessCurve, ShiftReference, SubsampleResult
from riskvalidation.ml.seeds import SeedDispersion

NAN = float("nan")
BATTERY = "ml-battery-1.1.0"

DOSSIER_06_C7 = (
    "dossier 06 (model risk management and validation) C.7 ML/AI validation and I.1, the validation engine "
    "catalogue; contract engine-ml-api 2.0.0"
)
SR_11_7 = (
    "SR 11-7 sensitivity analysis (historical reference text, superseded in the US by SR 26-2; dossier 06 A.1): "
    "unexpectedly large changes in outputs for small changes in inputs"
)
ECB_43 = (
    "ECB guide to internal models, paragraph 43 (randomness of the training process, overfitting) and 59 (stored seeds)"
)
EBA_ML = "EBA/REP/2023/28 on machine learning for IRB models, recommendation 3; ECB paragraph 43(d), 62 and 64 to 66"
ANGELOPOULOS = (
    "Angelopoulos and Bates, arXiv 2107.07511; Vovk, Gammerman and Shafer (2022), doi:10.1007/978-3-031-06649-8"
)
NANNYML = "NannyML CBPE, https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html"


def _isfinite(*values: float | None) -> bool:
    return all(v is not None and math.isfinite(float(v)) for v in values)


def _resolve(policy: Policy | None) -> Policy:
    return policy if policy is not None else _policy.ml_policy()


def _version(pol: Policy, **params: Any) -> str:
    return "; ".join([pol.label, BATTERY, *(f"{k}={v}" for k, v in params.items())])


def _not_evaluated(
    test_id: str,
    h0: str,
    alternative: str,
    reference: str,
    version: str,
    reason: str,
    *,
    n: int | None = None,
    model_id: str | None = None,
    extras: Mapping[str, Any] | None = None,
    inputs: Sequence[Any] = (),
) -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alternative,
        reference=reference,
        policy_version=version,
        light=Light.NOT_EVALUATED,
        statistic=NAN,
        notes=f"not evaluated: {reason}",
        n=n,
        model_id=model_id,
        inputs_hash=inputs_hash(*inputs),
        extras=dict(extras or {}),
    )


def _cut_matches(th: Threshold, tolerance: float, name: str, *, red: float | None = None) -> None:
    """The policy's cut points must state the tolerance the caller names (the strict one-float-above rule)."""
    want_amber = _policy.strict(tolerance)
    want_red = want_amber if red is None else _policy.strict(red)
    if th.amber != want_amber or th.red != want_red:
        raise ValueError(
            f"{name} {tolerance} does not match the cut points of the policy (amber {th.amber}, "
            f"red {th.red}); pass a policy "
            f"built with riskvalidation.ml.policy.ml_policy({name}=...) or whose row states it"
        )


# ----------------------------------------------------------------------------------------------------------------
# ml.robustness


@register(
    "ml.robustness",
    family="ml",
    name="Perturbation robustness",
    h0="the AUC at the preset noise scale loses no more than the tolerance against the clean AUC",
    reference=f"{DOSSIER_06_C7} (C.7 l.1009-1012, I.1 l.1189); {SR_11_7}",
    decision="metric",
)
def robustness(
    model_id: str, curve: RobustnessCurve, tolerance: float, preset_eps: float, policy: Policy | None = None
) -> TestResult:
    """Perturbation robustness (06 C.7 l.1009-1012; SR 11-7). statistic = metric = the AUC degradation ``-auc_delta`` at
    the preset noise scale (AUC points; negative when the AUC rose). Light (policy key ``ml.robustness``, upper_bad,
    strict): green when the degradation is at most ``tolerance`` (the named parameter ``perturbation_tolerance``), amber
    above it up to twice the tolerance, red above twice it; a degradation equal to the tolerance is green and not a
    ``robustness_finding``. ``tolerance`` must be the one the policy's cut points state
    (``ml_policy(perturbation_tolerance=...)``), else it raises."""
    test_id = "ml.robustness"
    h0 = "the AUC at the preset noise scale loses no more than the tolerance against the clean AUC"
    alt = "the AUC degrades by more than the tolerance under input noise at the preset scale"
    ref = f"{DOSSIER_06_C7} (C.7 l.1009-1012, I.1 l.1189); {SR_11_7}"
    pol = _resolve(policy)
    tol = float(tolerance)
    eps = float(preset_eps)
    hits = [i for i, g in enumerate(curve.grid) if math.isclose(g, eps, rel_tol=0.0, abs_tol=1e-12)]
    if not hits:
        raise ValueError(f"preset_eps {eps} is not a point of the grid {curve.grid}")
    deg = round(-curve.auc_delta[hits[0]], 12)  # 12 decimals: a loss that is the tolerance in decimals stays green
    version = _version(pol, perturbation_tolerance=tol, preset_eps=eps)
    ins = (list(curve.grid), list(curve.auc), list(curve.brier), tol, eps)
    th = pol.threshold(test_id)
    if th is None:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            f"policy {pol.label} holds no threshold for {test_id}",
            model_id=model_id,
            inputs=ins,
        )
    if not _isfinite(deg, tol) or tol < 0.0:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            "the AUC at the preset scale or the tolerance is not finite",
            model_id=model_id,
            inputs=ins,
        )
    _cut_matches(th, tol, "tolerance", red=2.0 * tol)
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=ref,
        policy_version=version,
        light=th.classify(deg),
        statistic=deg,
        metric=deg,
        alpha_amber=th.amber if th.kind == "p_value" else None,
        alpha_red=th.red if th.kind == "p_value" else None,
        model_id=model_id,
        inputs_hash=inputs_hash(*ins),
        extras={
            "thresholds": {"perturbation_tolerance": tol, "amber_cut": th.amber, "red_cut": th.red},
            "preset_eps": eps,
            "grid": list(curve.grid),
            "auc": list(curve.auc),
            "brier": list(curve.brier),
        },
    )


# ----------------------------------------------------------------------------------------------------------------
# ml.shift_resilience, ml.seed_stability, ml.reason_code_stability, ml.effect_agreement


def _simple(
    test_id: str,
    model_id: str,
    value: float | None,
    n: int | None,
    policy: Policy | None,
    *,
    h0: str,
    alt: str,
    ref: str,
    params: Mapping[str, Any],
    extras: Mapping[str, Any],
    inputs: Sequence[Any],
    statistic: float | None = None,
) -> TestResult:
    """A test whose light is one threshold of the policy read on one metric."""
    pol = _resolve(policy)
    version = _version(pol, **params)
    th = pol.threshold(test_id)
    if th is None:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            f"policy {pol.label} holds no threshold for {test_id}",
            n=n,
            model_id=model_id,
            inputs=inputs,
        )
    if value is None or not math.isfinite(value):
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            "the metric is not finite (missing data)",
            n=n,
            model_id=model_id,
            inputs=inputs,
        )
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=ref,
        policy_version=version,
        light=th.classify(value),
        statistic=value if statistic is None else statistic,
        metric=value,
        alpha_amber=th.amber if th.kind == "p_value" else None,
        alpha_red=th.red if th.kind == "p_value" else None,
        n=n,
        model_id=model_id,
        inputs_hash=inputs_hash(*inputs),
        extras={"thresholds": {"amber_cut": th.amber, "red_cut": th.red, **params}, **extras},
    )


@register(
    "ml.shift_resilience",
    family="ml",
    name="Resilience to distribution shift",
    h0=(
        "the AUC drop on the worst subsample by shift score is that of a random development subset of the same size"
    ),
    reference=(
        f"{DOSSIER_06_C7} (C.7 l.1013-1015, I.1 l.1190); PiML resilience; SS1/23 operating boundaries; "
        "Monte Carlo test against development-only draws (Phipson and Smyth 2010, doi:10.2202/1544-6115.1585)"
    ),
    decision="p_value",
)
def shift_resilience(
    model_id: str, result: SubsampleResult, reference: ShiftReference, policy: Policy | None = None
) -> TestResult:
    """Resilience to distribution shift (06 C.7 l.1013-1015), read against an exact permutation reference. statistic =
    metric = the AUC of the full sample minus the AUC of the worst subsample (AUC points; n = the subsample rows);
    p_value = ``reference.p_value``, ``(1 + #{null drops >= observed drop}) / (R + 1)`` over the drops of random
    pseudo-evaluation subsets of the development rows
    (``riskvalidation.ml.robustness.shift_development_reference``). Under no shift the raw drop is positive (range
    restriction on the worst subsample, finding F4), so the drop alone is not read against a cut point. Light
    (``ml.shift_resilience``, p_value kind): red below ``alpha_red`` (0.01), amber below ``alpha_amber`` (0.05), green
    otherwise; conventions. The null drops are summarised in ``extras`` (their mean and their 95th percentile). Without
    development labels the test is not_evaluated. ``reference.observed_drop`` must be the drop of ``result``, else it
    raises."""
    test_id = "ml.shift_resilience"
    h0 = "the AUC drop on the worst subsample by shift score is that of a random development subset of the same size"
    alt = "the model loses more AUC on the rows that resemble the development sample least than chance explains"
    ref = (
        f"{DOSSIER_06_C7} (C.7 l.1013-1015, I.1 l.1190); PiML resilience; SS1/23 operating boundaries; "
        "Monte Carlo test against development-only draws (Phipson and Smyth 2010, doi:10.2202/1544-6115.1585)"
    )
    pol = _resolve(policy)
    drop = float(result.auc_full - result.auc)
    version = _version(pol, shift_permutations=reference.n_permutations, share=reference.share)
    ins = (
        result.n,
        result.auc,
        result.brier,
        result.auc_full,
        result.brier_full,
        reference.observed_drop,
        list(reference.null_drops),
        reference.p_value,
        reference.n_permutations,
        reference.n_undefined,
        reference.enough_development,
        reference.share,
    )
    extras = {
        "auc_full": result.auc_full,
        "auc_worst": result.auc,
        "brier_full": result.brier_full,
        "brier_worst": result.brier,
        "shift_permutations": reference.n_permutations,
        "n_undefined": reference.n_undefined,
        "share": reference.share,
    }
    n = int(result.n)
    if reference.labelled and _isfinite(drop, reference.observed_drop) and abs(reference.observed_drop - drop) > 1e-9:
        raise ValueError(
            f"reference.observed_drop {reference.observed_drop} is not the drop {drop} of result: build the reference "
            "from the same predict, rows, score and share"
        )
    th = pol.threshold(test_id)
    if th is None:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            f"policy {pol.label} holds no threshold for {test_id}",
            n=n,
            model_id=model_id,
            extras=extras,
            inputs=ins,
        )
    if reference.labelled and not reference.enough_development:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            "the development set holds fewer than twice the evaluation rows: no development-only reference",
            n=n,
            model_id=model_id,
            extras=extras,
            inputs=ins,
        )
    if not reference.labelled:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            "no development labels: the reference draws pseudo-evaluation sets from the labelled development rows",
            n=n,
            model_id=model_id,
            extras=extras,
            inputs=ins,
        )
    if not _isfinite(drop, reference.p_value) or not reference.null_drops:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            "the AUC drop or its permutation p-value is not finite (missing data)",
            n=n,
            model_id=model_id,
            extras=extras,
            inputs=ins,
        )
    nulls = np.asarray(reference.null_drops, dtype=float)
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=ref,
        policy_version=version,
        light=th.classify(reference.p_value),
        statistic=drop,
        p_value=float(reference.p_value),
        metric=drop,
        alpha_amber=th.amber,
        alpha_red=th.red,
        n=n,
        model_id=model_id,
        inputs_hash=inputs_hash(*ins),
        extras={
            "thresholds": {"alpha_amber": th.amber, "alpha_red": th.red},
            **extras,
            "null_mean": float(nulls.mean()),
            "null_p95": float(np.percentile(nulls, 95.0)),
        },
    )


@register(
    "ml.seed_stability",
    family="ml",
    name="Seed stability",
    h0="the AUC does not depend on the training seed",
    reference=f"{DOSSIER_06_C7} (I.1 l.1191); {ECB_43}",
    decision="metric",
)
def seed_stability(model_id: str, dispersion: SeedDispersion, policy: Policy | None = None) -> TestResult:
    """Seed stability (06 I.1 l.1191; ECB 43 and 59). metric = the standard deviation of the AUC across seeds (AUC
    points; n = the seeds). Light (``ml.seed_stability``, upper_bad, inclusive): green below 0.01, amber from it, red
    from 0.02; conventions (about the sampling standard error of a Taiwan holdout AUC, 0.008)."""
    return _simple(
        "ml.seed_stability",
        model_id,
        float(dispersion.sd),
        int(dispersion.n_seeds),
        policy,
        h0="the AUC does not depend on the training seed",
        alt="the AUC varies across retraining seeds",
        ref=f"{DOSSIER_06_C7} (I.1 l.1191); {ECB_43}",
        params={},
        extras={"mean": dispersion.mean, "minimum": dispersion.minimum, "maximum": dispersion.maximum},
        inputs=(dispersion.n_seeds, dispersion.mean, dispersion.sd, dispersion.minimum, dispersion.maximum),
    )


@register(
    "ml.reason_code_stability",
    family="ml",
    name="Reason-code stability",
    h0="the top reasons of an obligor do not change across bootstrap refits",
    reference=f"dossier 04 (credit risk models) J.2 l.881-883 and ladder P4 l.968; {DOSSIER_06_C7}",
    decision="metric",
)
def reason_code_stability(model_id: str, overlap: OverlapResult, policy: Policy | None = None) -> TestResult:
    """Reason-code stability (04 J.2 l.881-883; ladder P4 l.968). metric = the mean top-k reason overlap across refits
    (share; n = the refits). Light (``ml.reason_code_stability``, lower_bad, inclusive): green above 0.80, amber at or
    below it, red at or below 0.60; conventions, no dossier value."""
    return _simple(
        "ml.reason_code_stability",
        model_id,
        float(overlap.mean),
        int(overlap.n_refits),
        policy,
        h0="the top reasons of an obligor do not change across bootstrap refits",
        alt="the top reasons change across refits",
        ref=f"dossier 04 (credit risk models) J.2 l.881-883 and ladder P4 l.968; {DOSSIER_06_C7}",
        params={"top_k": overlap.top_k},
        extras={"p05": overlap.p05},
        inputs=(overlap.top_k, overlap.n_refits, overlap.mean, overlap.p05),
    )


@register(
    "ml.effect_agreement",
    family="ml",
    name="Explainer agreement",
    h0="the explainers rank the features alike (SHAP against ALE or PDP)",
    reference=f"{DOSSIER_06_C7} (C.7 l.1006-1008, I.1 l.1193); {EBA_ML}",
    decision="metric",
)
def effect_agreement(model_id: str, rho: float, n_features: int, policy: Policy | None = None) -> TestResult:
    """Explainer agreement (06 C.7 l.1006-1008; I.1 l.1193; ECB 43(d), 66). metric = the Spearman rho of the mean
    absolute SHAP against the range of the ALE or PDP curve (n = the features). Light (``ml.effect_agreement``,
    lower_bad, inclusive): green above 0.70, amber at or below it, red at or below 0.40; conventions. A rho that is not
    finite (a constant input) or fewer than two features is not evaluated."""
    ok = _isfinite(rho) and n_features >= 2
    return _simple(
        "ml.effect_agreement",
        model_id,
        float(rho) if ok else None,
        int(n_features),
        policy,
        h0="the explainers rank the features alike (SHAP against ALE or PDP)",
        alt="the explainers disagree on the feature ranking",
        ref=f"{DOSSIER_06_C7} (C.7 l.1006-1008, I.1 l.1193); {EBA_ML}",
        params={},
        extras={"n_features": int(n_features)},
        inputs=(rho, n_features),
    )


# ----------------------------------------------------------------------------------------------------------------
# ml.additivity, ml.overfit, ml.weak_spot


@register(
    "ml.additivity",
    family="ml",
    name="Shapley additivity",
    h0="the attributions plus the base value reproduce the prediction",
    reference=f"{DOSSIER_06_C7} (C.7 l.990: f(x) = phi_0 + sum phi_i); Lundberg and Lee (2017), arXiv 1705.07874",
    decision="metric",
)
def additivity(model_id: str, gap: float, tolerance: float = 1e-9, policy: Policy | None = None) -> TestResult:
    """Shapley additivity (06 C.7 l.990). metric = the maximum over rows of the additivity gap (prediction units). Light
    (``ml.additivity``, upper_bad, strict): red when the gap exceeds the numeric ``tolerance``, green otherwise (never
    amber); ``tolerance`` must be the one the policy's cut points state."""
    test_id = "ml.additivity"
    pol = _resolve(policy)
    th = pol.threshold(test_id)
    if th is not None and _isfinite(gap):
        _cut_matches(th, float(tolerance), "tolerance")
    return _simple(
        test_id,
        model_id,
        float(gap) if _isfinite(gap) and gap >= 0.0 else None,
        None,
        pol,
        h0="the attributions plus the base value reproduce the prediction",
        alt="the attributions do not add up to the prediction",
        ref=f"{DOSSIER_06_C7} (C.7 l.990: f(x) = phi_0 + sum phi_i); Lundberg and Lee (2017), arXiv 1705.07874",
        params={"additivity_tolerance": float(tolerance)},
        extras={},
        inputs=(gap, tolerance),
    )


@register(
    "ml.overfit",
    family="ml",
    name="Overfit gap",
    h0="the model performs on the holdout as on the training data",
    reference=f"{DOSSIER_06_C7} (C.7 l.1016-1018, I.1 l.1187); {ECB_43}(a); EBA recommendation (i)",
    decision="metric",
)
def overfit(model_id: str, result: OverfitResult, policy: Policy | None = None) -> TestResult:
    """The overfit gap (06 C.7 l.1016-1018; I.1 l.1187; ECB 43(a)). metric = the train minus holdout gap (metric units).
    Light (``ml.overfit``, upper_bad, strict): red when the gap exceeds ``overfit_tolerance`` (0.05), green otherwise
    (never amber); the result's tolerance must be the one the policy's cut points state."""
    pol = _resolve(policy)
    th = pol.threshold("ml.overfit")
    if th is not None and _isfinite(result.gap, result.tolerance):
        _cut_matches(th, float(result.tolerance), "tolerance")
    return _simple(
        "ml.overfit",
        model_id,
        round(float(result.gap), 12) if _isfinite(result.gap) else None,  # 12 decimals, as overfit_gap
        None,
        pol,
        h0="the model performs on the holdout as on the training data",
        alt="the train-holdout performance gap exceeds the overfit tolerance",
        ref=f"{DOSSIER_06_C7} (C.7 l.1016-1018, I.1 l.1187); {ECB_43}(a); EBA recommendation (i)",
        params={"overfit_tolerance": float(result.tolerance)},
        extras={"flag": bool(result.flag)},
        inputs=(result.gap, result.tolerance, result.flag),
    )


@register(
    "ml.weak_spot",
    family="ml",
    name="Weak spots",
    h0="no slice of the data has an error above the weak-spot multiple of the average error",
    reference=f"{DOSSIER_06_C7} (C.7 l.1016-1018, I.1 l.1188); PiML WeakSpot",
    decision="metric",
)
def weak_spot(
    model_id: str, spots: Sequence[WeakSpot], k: float, n_slices: int, policy: Policy | None = None
) -> TestResult:
    """Weak spots (06 C.7 l.1016-1018; I.1 l.1188). metric = the error ratio of the worst slice to the average error (0
    when no slice qualifies; n = the slices examined). Light (``ml.weak_spot``, upper_bad, strict): amber when the worst
    ratio exceeds ``k`` (``weak_spot_multiple``), green otherwise, never red (the red cut point is a sentinel no ratio
    reaches); ``k`` must be the one the policy's cut points state."""
    test_id = "ml.weak_spot"
    pol = _resolve(policy)
    th = pol.threshold(test_id)
    if th is not None and _isfinite(k):
        _cut_matches(th, float(k), "k", red=1e12)
    worst = max(spots, key=lambda s: s.ratio) if spots else None
    return _simple(
        test_id,
        model_id,
        float(worst.ratio) if worst is not None else 0.0,
        int(n_slices),
        pol,
        h0="no slice of the data has an error above the weak-spot multiple of the average error",
        alt="a slice of the data has an error above the weak-spot multiple of the average error",
        ref=f"{DOSSIER_06_C7} (C.7 l.1016-1018, I.1 l.1188); PiML WeakSpot",
        params={"weak_spot_multiple": float(k)},
        extras={
            "n_spots": len(spots),
            "worst_label": None if worst is None else worst.label,
            "worst_n": None if worst is None else worst.n,
        },
        inputs=(k, n_slices, [(s.label, s.n, s.mean_error, s.ratio) for s in spots]),
    )


# ----------------------------------------------------------------------------------------------------------------
# ml.conformal_coverage, ml.leakage_screen, ml.data_quality


@register(
    "ml.conformal_coverage",
    family="ml",
    name="Conformal coverage",
    h0="the coverage of the split conformal set equals the target 1 - alpha (exchangeability holds)",
    reference=f"{DOSSIER_06_C7} (C.7 l.1019-1028, I.1 l.1194); {ANGELOPOULOS}",
    decision="p_value",
)
def conformal_coverage(model_id: str, result: CoverageResult, policy: Policy | None = None) -> TestResult:
    """Conformal coverage (06 C.7 l.1019-1028; I.1 l.1194). p_value = the two-sided Beta-Binomial p-value of the covered
    count given the calibration size (``result.p_value``; the coverage of a split conformal set is Beta(k, n_cal + 1 -
    k) distributed, so the count of m test points is Beta-Binomial(m, k, n_cal + 1 - k)); metric = the empirical
    coverage; statistic = coverage minus target. The dossier's binomial p-value is in ``extras["p_value_binomial"]``
    beside it and never drives the light. Light (``ml.conformal_coverage``, p_value kind): red below ``alpha_red``
    (0.01), amber below ``alpha_amber`` (0.05), green otherwise. When ``k > n_cal`` the set is every label and the
    coverage is 1 by construction: not_evaluated. A coverage loss is read as an exchangeability violation, never as
    proof."""
    test_id = "ml.conformal_coverage"
    pol = _resolve(policy)
    h0 = "the coverage of the split conformal set equals the target 1 - alpha (exchangeability holds)"
    alt = "the coverage differs from the target: the calibration and evaluation samples are not exchangeable"
    ref = f"{DOSSIER_06_C7} (C.7 l.1019-1028, I.1 l.1194); {ANGELOPOULOS}"
    version = _version(pol, target=result.target, n_cal=result.n_cal)
    ins = (result.n, result.coverage, result.target, result.p_value, result.n_cal, result.k, result.p_value_binomial)
    extras = {"target": result.target, "n_cal": int(result.n_cal), "k": int(result.k)}
    th = pol.threshold(test_id)
    if th is None:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            f"policy {pol.label} holds no threshold for {test_id}",
            n=int(result.n),
            model_id=model_id,
            inputs=ins,
        )
    if result.k > result.n_cal:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            f"k = {result.k} exceeds n_cal = {result.n_cal}: the set is every label, the coverage is 1 by construction",
            n=int(result.n),
            model_id=model_id,
            extras=extras,
            inputs=ins,
        )
    if not _isfinite(result.p_value, result.coverage, result.target):
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            "the coverage or its p-value is not finite",
            n=int(result.n),
            model_id=model_id,
            inputs=ins,
        )
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=ref,
        policy_version=version,
        light=th.classify(result.p_value),
        statistic=result.coverage - result.target,
        p_value=float(result.p_value),
        metric=float(result.coverage),
        alpha_amber=th.amber,
        alpha_red=th.red,
        n=int(result.n),
        model_id=model_id,
        inputs_hash=inputs_hash(*ins),
        extras={
            "thresholds": {"alpha_amber": th.amber, "alpha_red": th.red},
            **extras,
            "p_value_binomial": float(result.p_value_binomial),
        },
    )


_LEAK = {
    "temporal": "ml.leakage_screen.temporal_order",
    "single": "ml.leakage_screen.single_feature_auc",
    "overlap": "ml.leakage_screen.entity_overlap",
    "resemblance": "ml.leakage_screen.resemblance_auc",
}
_ORDER = {Light.GREEN: 0, Light.AMBER: 1, Light.RED: 2}


@register(
    "ml.leakage_screen",
    family="ml",
    name="Leakage screen",
    h0=(
        "no feature leaks the outcome: none is dated after the observation date, none is near-perfect alone, "
        "no entity is in "
        "both slices and development and out-of-time rows are alike"
    ),
    reference=(
        f"{DOSSIER_06_C7} (C.7 l.1048-1051, I.1 l.1197); dossier 07 D pattern 3 l.597-600; "
        "ING probatus sample_similarity"
    ),
    decision="metric",
)
def leakage_screen(
    model_id: str,
    temporal: int,
    single: SingleFeatureAuc,
    overlap: float,
    resemblance: AucResult,
    policy: Policy | None = None,
) -> TestResult:
    """The leakage screen (06 C.7 l.1048-1051; I.1 l.1197; dossier 07 D pattern 3). Four screens, each read on its own
    policy key: ``temporal`` the number of feature values dated after the observation date (a rule: one is a leak),
    ``single`` the largest single-feature max(AUC, 1 - AUC) (strictly above 0.95 trips), ``overlap`` the share of test
    entities present in training read as the indicator of any overlap (a rule: an entity in both slices is a leak),
    ``resemblance`` the cross-fitted AUC separating development from out-of-time rows (strictly above 0.70 trips).
    Light: the worst of the four; red when any screen trips, green otherwise. statistic = the number of screens that
    tripped; ``extras["screens"]`` holds the value and light of each; n = the rows of the resemblance sample. Any
    missing policy key or non-finite value is not evaluated."""
    test_id = "ml.leakage_screen"
    pol = _resolve(policy)
    h0 = (
        "no feature leaks the outcome: none is dated after the observation date, none is near-perfect alone, "
        "no entity is in "
        "both slices and development and out-of-time rows are alike"
    )
    alt = (
        "at least one screen trips: a future-dated feature, a near-perfect single feature, an entity overlap or a "
        "resemblance AUC beyond its limit"
    )
    ref = (
        f"{DOSSIER_06_C7} (C.7 l.1048-1051, I.1 l.1197); dossier 07 D pattern 3 l.597-600; "
        "ING probatus sample_similarity"
    )
    version = _version(pol)
    n = int(resemblance.n)
    values = {
        "temporal": float(temporal),
        "single": float(single.max_auc),
        "overlap": 1.0 if overlap > 0.0 else 0.0,
        "resemblance": float(resemblance.auc),
    }
    ins = (temporal, single.max_auc, single.feature, overlap, resemblance.auc, resemblance.variance, resemblance.n)
    missing = [key for key in _LEAK.values() if pol.threshold(key) is None]
    if missing:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            f"policy {pol.label} holds no threshold for {missing[0]}",
            n=n,
            model_id=model_id,
            inputs=ins,
        )
    if not _isfinite(*values.values(), overlap) or not 0.0 <= overlap <= 1.0 or temporal < 0:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            ref,
            version,
            "a screen value is not finite or out of range",
            n=n,
            model_id=model_id,
            inputs=ins,
        )
    screens: dict[str, dict[str, Any]] = {}
    worst = Light.GREEN
    for name, key in _LEAK.items():
        th = pol.threshold(key)
        assert th is not None
        light = th.classify(values[name])
        screens[name] = {
            "key": key,
            "value": int(temporal) if name == "temporal" else values[name],
            "light": light.value,
            "amber_cut": th.amber,
            "red_cut": th.red,
        }
        if _ORDER[light] > _ORDER[worst]:
            worst = light
    tripped = sum(1 for s in screens.values() if s["light"] != "green")
    screens["single"]["feature"] = int(single.feature)
    screens["overlap"]["share"] = float(overlap)
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=ref,
        policy_version=version,
        light=worst,
        statistic=float(tripped),
        n=n,
        model_id=model_id,
        inputs_hash=inputs_hash(*ins),
        extras={
            "thresholds": {s["key"]: {"amber_cut": s["amber_cut"], "red_cut": s["red_cut"]} for s in screens.values()},
            "screens": screens,
        },
    )


@register(
    "ml.data_quality",
    family="ml",
    name="Data quality",
    h0="every counted data-quality check passes",
    reference=f"{DOSSIER_06_C7} (C.7 l.1045-1047, I.1 l.1198); ECB paragraph 52(h)(vii) and Overarching paragraph 11",
    decision="metric",
)
def data_quality(model_id: str, counts: Sequence[DataQualityCount], policy: Policy | None = None) -> TestResult:
    """Data quality (06 C.7 l.1045-1047; I.1 l.1198; ECB 52(h)(vii)). metric = the largest share of failed checks over
    the ``dq_`` counts (``1 - share_ok``); n and n_events = the cells or rows checked and failed, summed. Light
    (``ml.data_quality``, upper_bad, inclusive): green below 0.01, amber from it, red from 0.05; conventions, no dossier
    value. No count is not evaluated."""
    test_id = "ml.data_quality"
    if not counts:
        pol = _resolve(policy)
        return _not_evaluated(
            test_id,
            "data quality counts hold no failure",
            "a counted data-quality check fails",
            f"{DOSSIER_06_C7} (C.7 l.1045-1047, I.1 l.1198)",
            _version(pol),
            "no data-quality count supplied",
            model_id=model_id,
        )
    # equal to 1 - share_ok, without the rounding of the subtraction
    shares = [c.n_failed / c.n_checked for c in counts]
    res = _simple(
        test_id,
        model_id,
        max(shares),
        sum(c.n_checked for c in counts),
        policy,
        h0="every counted data-quality check passes",
        alt="at least one counted data-quality check fails for a share of its cells or rows above the cut points",
        ref=f"{DOSSIER_06_C7} (C.7 l.1045-1047, I.1 l.1198); ECB paragraph 52(h)(vii) and Overarching paragraph 11",
        params={},
        extras={
            "checks": [
                {
                    "check": c.check,
                    "n_checked": c.n_checked,
                    "n_failed": c.n_failed,
                    "share_failed": c.n_failed / c.n_checked,
                }
                for c in counts
            ]
        },
        inputs=[(c.check, c.n_checked, c.n_failed, c.share_ok) for c in counts],
    )
    return _with_events(res, sum(c.n_failed for c in counts))


def _with_events(res: TestResult, n_events: int) -> TestResult:
    return replace(res, n_events=int(n_events))


# ----------------------------------------------------------------------------------------------------------------
# ml.cbpe_estimate, ml.challenger_brier (no policy threshold)


@register(
    "ml.cbpe_estimate",
    family="ml",
    name="Label-free AUC estimate",
    h0=(
        "the label-free AUC estimate equals the labelled AUC (calibrated probabilities, no concept drift, "
        "no extrapolation)"
    ),
    reference=f"{DOSSIER_06_C7} (C.7 l.1052-1058, I.1 l.1199); {NANNYML}",
    decision="metric",
)
def cbpe_estimate(model_id: str, estimate: CbpeAuc, labels_supplied: bool = False) -> TestResult:
    """The label-free AUC estimate (06 C.7 l.1052-1058; I.1 l.1199). An estimate, not a test: the row is always
    ``not_evaluated`` with a null statistic, metric and p-value, so that it never moves ``SuiteRun.worst()``. The reason
    is "an estimate, not a test: CBPE estimates performance without labels (06 C.7 l.1052-1058)" followed by the four
    assumptions (in ``notes``), or, when a label is supplied, "a label is supplied: read the labelled AUC, an estimate
    is not evidence when labels exist". The estimated AUC, its label ``estimate`` and the assumptions are in
    ``extras["estimate"]``."""
    test_id = "ml.cbpe_estimate"
    h0 = (
        "the label-free AUC estimate equals the labelled AUC (calibrated probabilities, no concept drift, "
        "no extrapolation)"
    )
    alt = "the estimate departs from the labelled AUC when an assumption fails"
    ref = f"{DOSSIER_06_C7} (C.7 l.1052-1058, I.1 l.1199); {NANNYML}"
    version = _version(_resolve(None), policy_threshold="none")
    ins = (estimate.estimated_auc, estimate.label, list(estimate.assumptions), labels_supplied)
    extras = {
        "estimate": {
            "estimated_auc": float(estimate.estimated_auc),
            "label": estimate.label,
            "assumptions": list(estimate.assumptions),
        }
    }
    if labels_supplied:
        reason = "a label is supplied: read the labelled AUC, an estimate is not evidence when labels exist"
    else:
        reason = (
            "an estimate, not a test: CBPE estimates performance without labels (06 C.7 l.1052-1058); it assumes "
            + ", ".join(estimate.assumptions)
            + "; it complements, never replaces, back-testing"
        )
    return _not_evaluated(test_id, h0, alt, ref, version, reason, model_id=model_id, extras=extras, inputs=ins)


@register(
    "ml.challenger_brier",
    family="ml",
    name="Champion and challenger Brier",
    h0="the champion and the challenger have the same Brier score",
    reference=f"{DOSSIER_06_C7} (C.7 l.1041-1044, I.1 l.1196); SS1/23 principle 3.3(b)(iii); ECB guide paragraph 60",
    decision="metric",
)
def challenger_brier(model_id: str, result: PairedBrier) -> TestResult:
    """Champion and challenger by the paired Brier bootstrap (06 C.7 l.1041-1044; SS1/23 3.3(b)(iii); ECB 60). No policy
    threshold. metric = statistic = the Brier difference challenger minus champion with its 95 percent percentile
    interval in ``extras``. Light: amber when the interval excludes zero in the champion's favour (the interval lies
    above zero: the challenger's Brier is higher), green when it does not (a better challenger is not a finding against
    the champion)."""
    test_id = "ml.challenger_brier"
    h0 = "the champion and the challenger have the same Brier score"
    alt = "the challenger's Brier score is higher than the champion's (the interval excludes zero)"
    ref = f"{DOSSIER_06_C7} (C.7 l.1041-1044, I.1 l.1196); SS1/23 principle 3.3(b)(iii); ECB guide paragraph 60"
    version = _version(_resolve(None), policy_threshold="none", interval_level=0.95)
    ins = (result.diff, result.lo, result.hi, result.n_boot)
    if not _isfinite(result.diff, result.lo, result.hi):
        return _not_evaluated(
            test_id, h0, alt, ref, version, "the paired Brier interval is not finite", model_id=model_id, inputs=ins
        )
    light = Light.AMBER if result.lo > 0.0 else Light.GREEN
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=ref,
        policy_version=version,
        light=light,
        statistic=float(result.diff),
        metric=float(result.diff),
        model_id=model_id,
        inputs_hash=inputs_hash(*ins),
        extras={
            "thresholds": {"policy_threshold": "none", "interval_level": 0.95},
            "lo": result.lo,
            "hi": result.hi,
            "level": 0.95,
            "n_boot": result.n_boot,
        },
    )
