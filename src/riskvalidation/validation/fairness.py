"""The fairness validation tests (dossier 06 C.7 l.1029-1040; dossier 04 J.3 l.897-905; contract `engine-fairness-api`
1.0.0, module validation.fairness): each returns one `TestResult` that validates against `common/test-result`, with a
test id prefixed `fair.`.

Every light comes from `core.policy.Policy` rows keyed by test id (`riskvalidation.fairness.policy`), or from the rule
a function states in its docstring; the tolerances are conventions, never regulatory thresholds (dossier 06 I.6
l.1268-1270), and every function records them in `extras["thresholds"]` and in the policy version. A `policy` argument
of None means the packaged default policy with the `fair.` rows. A test that cannot be evaluated returns the light
`not_evaluated` with its reason in `notes` and no number (the statistic is NaN, null once serialised; the metric and
the p-value are None), never a placeholder. A malformed input raises `ValueError` naming the field. The
outcome-conditioned tests need an outcome and return `not_evaluated` when none is supplied (HMDA has no outcome). The
four-fifths line is an employment benchmark (29 CFR 1607.4(D)) and never a credit-law threshold. The engine states no
legal claim: the controlled disparity is an approximation of comparative evidence, never a finding of a violation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.policy import Policy, Threshold
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.fairness import metrics as _metrics
from riskvalidation.fairness._checks import InsufficientData, binary, group_labels, group_names, same_rows
from riskvalidation.fairness.conditional import Effect
from riskvalidation.fairness.inference import fisher_exact_test, two_proportion_z_test
from riskvalidation.fairness.metrics import (
    EMPLOYMENT_BENCHMARK,
    FOUR_FIFTHS_THRESHOLD,
    SelectionRate,
    adverse_impact_ratio,
    demographic_parity_difference,
    demographic_parity_ratio,
    four_fifths_flag,
)
from riskvalidation.fairness.policy import FAIR_ROWS, fairness_default_policy
from riskvalidation.fairness.replication import Replication
from riskvalidation.fairness.situation import SituationResult

NAN = float("nan")
NO_LEGAL_CLAIM = "no legal claim"

DOSSIER_06_C7 = "dossier 06 (Contraste research) section C.7 l.1029-1040"
DOSSIER_04_J3 = "dossier 04 section J.3 l.897-905"
HARDT = "Hardt, Price and Srebro (2016), Equality of opportunity in supervised learning, arXiv:1610.02413"
CFR_1607 = "29 CFR 1607.4(D) (the four-fifths rule of the Uniform Guidelines on Employee Selection Procedures)"
COMPANION = ("Alvarez and Ruggieri, Counterfactual Situation Testing, arXiv:2502.01267v2, JAIR 82, "
             "doi:10.1613/jair.1.17935; Mutatis Mutandis, arXiv:2405.13693v4, doi:10.1111/coin.70238")


def _policy(policy: Policy | None) -> Policy:
    return policy if policy is not None else fairness_default_policy()


def _version(pol: Policy, **params: Any) -> str:
    return "; ".join([pol.label, *(f"{k}={v}" for k, v in params.items())])


def _not_evaluated(test_id: str, h0: str, alternative: str, reference: str, version: str, reason: str, *,
                   model_id: str | None, n: int | None = None, extras: Mapping[str, Any] | None = None) -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=version,
                      light=Light.NOT_EVALUATED, statistic=NAN, notes=f"not evaluated: {reason}", n=n,
                      model_id=model_id, extras=dict(extras or {}))


def _threshold_extras(test_id: str, th: Threshold) -> dict[str, Any]:
    """The cut points as the contract states them (a strict row is reported at its tolerance, not one float above)."""
    row = FAIR_ROWS.get(test_id)
    amber = row.amber if row is not None and row.strict and th.amber == math.nextafter(row.amber, math.inf) \
        else th.amber
    return {"kind": th.kind, "amber": amber, "red": th.red,
            "strict": bool(row is not None and row.strict and amber != th.amber), "source": th.source}


def _classified(pol: Policy, test_id: str, *, p_value: float | None = None, metric: float | None = None
                ) -> tuple[Light, Threshold | None]:
    return pol.classify(test_id, p_value=p_value, metric=metric)


NEVER_RED_NOTE = "; never red by contract; the policy's red cut is not applied"


def _cap_at_amber(light: Light) -> tuple[Light, str]:
    """A 'never red' rule: a red light from any policy is amber (F1), and the notes say so."""
    return (Light.AMBER, NEVER_RED_NOTE) if light is Light.RED else (light, "")


def _check_text(name: str, v: Any) -> list[str]:
    if isinstance(v, (str, bytes)) or not isinstance(v, Sequence) or not all(isinstance(s, str) for s in v):
        raise ValueError(f"{name} must be a sequence of strings (not a single string), got {v!r}")
    return list(v)


def _rates_view(rates: Mapping[str, SelectionRate]) -> dict[str, dict[str, Any]]:
    return {g: {"n": s.n, "favourable": s.favourable, "rate": s.rate} for g, s in rates.items()}


def _check_rates(rates: Any, reference: str) -> None:
    if not isinstance(rates, Mapping) or not rates:
        raise ValueError("rates must be a non-empty mapping of group to SelectionRate")
    if reference not in rates:
        raise ValueError(f"reference {reference!r} is not one of the groups {list(rates)}")


# ----------------------------------------------------------------------------------------------------------------
# Selection-rate tests


@register(
    "fair.demographic_parity",
    family="fairness",
    name="Demographic parity",
    h0="descriptive: every group's selection rate equals the reference group's rate",
    reference=f"{DOSSIER_06_C7}: demographic (statistical) parity difference and ratio; {HARDT}",
    decision="metric",
)
def demographic_parity(model_id: str, rates: Mapping[str, SelectionRate], reference: str,
                       policy: Policy | None = None) -> TestResult:
    """Demographic (statistical) parity as an indicator. metric = the largest absolute difference of a group's
    selection rate from the reference group's rate, computed on the counts. Light (policy key `fair.demographic_parity`,
    upper_bad, strict): green when the gap is at most the tolerance (0.05, a convention), amber above it; never red (the
    red cut is a sentinel above the metric's range, and a red light from a caller policy is capped at amber).
    p_value None."""
    test_id = "fair.demographic_parity"
    h0 = "descriptive: every group's selection rate equals the reference group's rate"
    alt = "a group's selection rate differs from the reference group's rate"
    ref = f"{DOSSIER_06_C7}: demographic (statistical) parity difference and ratio; {HARDT}"
    pol = _policy(policy)
    _check_rates(rates, reference)
    version = _version(pol)
    if len(rates) < 2:
        return _not_evaluated(test_id, h0, alt, ref, version, "fewer than two groups: there is nothing to compare",
                              model_id=model_id)
    diffs = demographic_parity_difference(rates, reference)
    ratios = demographic_parity_ratio(rates, reference)
    metric = max(abs(v) for v in diffs.values())
    light, th = _classified(pol, test_id, metric=metric)
    if th is None:
        return _not_evaluated(test_id, h0, alt, ref, version, f"policy {pol.label} holds no threshold for {test_id}",
                              model_id=model_id, extras={"thresholds": {}})
    light, capped = _cap_at_amber(light)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=ref,
        policy_version=_version(pol, tolerance=_threshold_extras(test_id, th)["amber"]), light=light, metric=metric,
        n=sum(s.n for s in rates.values()), n_events=sum(s.favourable for s in rates.values()), model_id=model_id,
        inputs_hash=inputs_hash({g: (s.n, s.favourable) for g, s in rates.items()}, reference),
        notes=f"indicator of statistical parity against {reference!r}; the tolerance is a convention, never a "
              f"regulatory threshold; {NO_LEGAL_CLAIM}{capped}",
        extras={"reference_group": reference, "rates": _rates_view(rates), "differences": diffs, "ratios": ratios,
                "thresholds": _threshold_extras(test_id, th)})


@register(
    "fair.adverse_impact",
    family="fairness",
    name="Adverse impact",
    h0="no group's selection rate is below four-fifths of the highest-rate group's",
    reference=(
        f"{CFR_1607}, labelled an {EMPLOYMENT_BENCHMARK}, never a credit-law threshold; {DOSSIER_06_C7} "
        "l.1033-1037 and I.6 l.1268-1270"),
    decision="metric",
)
def adverse_impact(model_id: str, rates: Mapping[str, SelectionRate], reference: str) -> TestResult:
    """The four-fifths flag on the exact counts. metric = the lowest ratio of a group's selection rate to the
    highest-rate group's. Light: amber when `four_fifths_flag` raises a flag for any group (5 x_g n_h < 4 x_h n_g,
    against the highest-rate group, in exact arithmetic), green otherwise. The line is an employment benchmark
    (29 CFR 1607.4(D)), never a credit-law threshold; the parity ratios to the chosen reference group are in `extras`.
    No policy threshold: the rule is the whole classification. p_value None."""
    test_id = "fair.adverse_impact"
    h0 = "no group's selection rate is below four-fifths of the highest-rate group's"
    alt = "a group's selection rate is below four-fifths of the highest-rate group's"
    ref = (f"{CFR_1607}, labelled an {EMPLOYMENT_BENCHMARK}, never a credit-law threshold; {DOSSIER_06_C7} "
           "l.1033-1037 and I.6 l.1268-1270")
    pol = fairness_default_policy()
    _check_rates(rates, reference)
    version = _version(pol, four_fifths=FOUR_FIFTHS_THRESHOLD, compared_with="highest-rate-group")
    if len(rates) < 2:
        return _not_evaluated(test_id, h0, alt, ref, version, "fewer than two groups: there is nothing to compare",
                              model_id=model_id)
    ff = four_fifths_flag(rates)
    if all(v is None for v in ff.ratio_to_highest.values()):
        return _not_evaluated(test_id, h0, alt, ref, version,
                              "no group has a favourable decision: the ratio to the highest-rate group is undefined",
                              model_id=model_id, n=sum(s.n for s in rates.values()))
    ratios = [v for v in ff.ratio_to_highest.values() if v is not None]
    flagged = [g for g, f in ff.flag.items() if f]
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=ref, policy_version=version,
        light=Light.AMBER if flagged else Light.GREEN, metric=min(ratios),
        n=sum(s.n for s in rates.values()), n_events=sum(s.favourable for s in rates.values()), model_id=model_id,
        inputs_hash=inputs_hash({g: (s.n, s.favourable) for g, s in rates.items()}, reference),
        notes=f"the four-fifths line is an {EMPLOYMENT_BENCHMARK} ({CFR_1607}), compared with the highest-rate group "
              f"and never a credit-law threshold; decided on the exact counts; {NO_LEGAL_CLAIM}",
        extras={"highest_rate_group": ff.highest_rate_group, "ratio_to_highest": ff.ratio_to_highest, "flag": ff.flag,
                "flagged_groups": flagged, "reference_group": reference,
                "parity_ratio_to_reference": adverse_impact_ratio(rates, reference), "rates": _rates_view(rates),
                "label": ff.label, "is_credit_law_threshold": ff.is_credit_law_threshold,
                "thresholds": {"four_fifths": ff.threshold, "rule": "flag when 5 x_g n_h < 4 x_h n_g, exact counts",
                               "compared_with": ff.compared_with}})


def _two_group_result(test_id: str, model_id: str, x1: int, n1: int, x2: int, n2: int, policy: Policy | None, *,
                      h0: str, alt: str, ref: str, run: Any) -> TestResult:
    pol = _policy(policy)
    version = _version(pol)
    try:
        statistic, p_value, note = run()
    except InsufficientData as exc:
        return _not_evaluated(test_id, h0, alt, ref, version, str(exc), model_id=model_id)
    n1_, n2_ = int(n1), int(n2)
    x1_, x2_ = int(x1), int(x2)
    light, th = _classified(pol, test_id, p_value=p_value)
    if th is None:
        return _not_evaluated(test_id, h0, alt, ref, version, f"policy {pol.label} holds no threshold for {test_id}",
                              model_id=model_id, n=n1_ + n2_, extras={"thresholds": {}})
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=ref,
        policy_version=_version(pol, alpha_amber=th.amber, alpha_red=th.red), light=light, statistic=statistic,
        p_value=p_value, metric=(x1_ * n2_ - x2_ * n1_) / (n1_ * n2_), alpha_amber=th.amber, alpha_red=th.red,
        n=n1_ + n2_, n_events=x1_ + x2_, model_id=model_id, inputs_hash=inputs_hash(x1_, n1_, x2_, n2_),
        notes="; ".join(s for s in (note, "the rate difference is group 1 minus group 2; smaller differences can "
                                    f"matter when statistically significant; {NO_LEGAL_CLAIM}") if s),
        extras={"rate_1": x1_ / n1_, "rate_2": x2_ / n2_, "n1": n1_, "n2": n2_,
                "thresholds": _threshold_extras(test_id, th)})


@register(
    "fair.two_proportion_z",
    family="fairness",
    name="Two-proportion z-test",
    h0="the favourable-decision rate of group 1 equals that of group 2",
    reference=f"{DOSSIER_06_C7} l.1036-1039: two-sample test of proportions",
    decision="p_value",
)
def two_proportion_z(model_id: str, x1: int, n1: int, x2: int, n2: int, policy: Policy | None = None) -> TestResult:
    """The pooled two-sided z-test on the favourable-decision rates of two groups (dossier 06 C.7 l.1038-1039).
    statistic = z (group 1 minus group 2), metric = the rate difference. Light (policy key `fair.two_proportion_z`,
    p_value): red below alpha_red (0.01), amber below alpha_amber (0.05), green otherwise."""
    def run() -> tuple[float, float, str | None]:
        r = two_proportion_z_test(x1, n1, x2, n2)
        return r.statistic, r.p_value, r.note

    return _two_group_result(
        "fair.two_proportion_z", model_id, x1, n1, x2, n2, policy,
        h0="the favourable-decision rate of group 1 equals that of group 2",
        alt="the rates differ (two-sided)", ref=f"{DOSSIER_06_C7} l.1036-1039: two-sample test of proportions", run=run)


@register(
    "fair.fisher_exact",
    family="fairness",
    name="Fisher exact test",
    h0="the favourable-decision rate of group 1 equals that of group 2",
    reference=f"{DOSSIER_06_C7} l.1038-1039: Fisher exact test",
    decision="p_value",
)
def fisher_exact(model_id: str, x1: int, n1: int, x2: int, n2: int, policy: Policy | None = None) -> TestResult:
    """Fisher's exact test on the same 2 x 2 table (dossier 06 C.7 l.1038-1039). statistic = the sample odds ratio
    (None when undefined), metric = the rate difference. Light (policy key `fair.fisher_exact`, p_value): red below
    alpha_red (0.01), amber below alpha_amber (0.05), green otherwise."""
    def run() -> tuple[float | None, float, str | None]:
        r = fisher_exact_test(x1, n1, x2, n2)
        return r.odds_ratio, r.p_value, None

    return _two_group_result(
        "fair.fisher_exact", model_id, x1, n1, x2, n2, policy,
        h0="the favourable-decision rate of group 1 equals that of group 2",
        alt="the rates differ (two-sided)", ref=f"{DOSSIER_06_C7} l.1038-1039: Fisher exact test", run=run)


# ----------------------------------------------------------------------------------------------------------------
# Outcome-conditioned tests: they need an outcome and never run on a dataset that has none


NO_OUTCOME = ("no outcome was supplied: the dataset declares no outcome (outcome: none), and the outcome-conditioned "
              "tests never run on it")


def _outcome_result(test_id: str, model_id: str, metric: float, pol: Policy, *, h0: str, alt: str, ref: str, n: int,
                    n_events: int, extras: dict[str, Any], notes: str) -> TestResult:
    light, th = _classified(pol, test_id, metric=metric)
    version = _version(pol)
    if th is None:
        return _not_evaluated(test_id, h0, alt, ref, version, f"policy {pol.label} holds no threshold for {test_id}",
                              model_id=model_id, n=n, extras={**extras, "thresholds": {}})
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=ref,
        policy_version=_version(pol, alpha_amber=th.amber, alpha_red=th.red), light=light, metric=metric,
        # a metric light: the alpha fields are p-value cut points and stay null; the metric's cut points are in extras
        alpha_amber=None, alpha_red=None, n=n, n_events=n_events, model_id=model_id,
        notes=f"{notes}; the cut points are conventions, never regulatory thresholds; {NO_LEGAL_CLAIM}",
        extras={**extras, "thresholds": _threshold_extras(test_id, th)})


def _error_rate_test(test_id: str, model_id: str, y: Any, decision: Any, group: Any, policy: Policy | None, *,
                     metric_fn: Any, h0: str, alt: str, ref: str, what: str) -> TestResult:
    pol = _policy(policy)
    version = _version(pol)
    if y is None:
        return _not_evaluated(test_id, h0, alt, ref, version, NO_OUTCOME, model_id=model_id)
    yy = binary(y, "y")
    dd = binary(decision, "decision")
    same_rows(yy, "y", dd, "decision")
    gg = group_labels(group, yy.shape[0], "group")
    try:
        metric = metric_fn(yy, dd, gg)
    except InsufficientData as exc:
        return _not_evaluated(test_id, h0, alt, ref, version, str(exc), model_id=model_id, n=int(yy.shape[0]))
    return _outcome_result(
        test_id, model_id, metric, pol, h0=h0, alt=alt, ref=ref, n=int(yy.shape[0]), n_events=int(yy.sum()),
        extras={"rates_by_group": _metrics.outcome_rates(yy, dd, gg), "groups": group_names(gg)},
        notes=f"{what}; y = 1 is the outcome the favourable decision should predict")


@register(
    "fair.equalized_odds",
    family="fairness",
    name="Equalized odds",
    h0="descriptive: the true and the false positive rates are equal across groups",
    reference=f"{HARDT}; {DOSSIER_06_C7} l.1029-1033",
    decision="metric",
)
def equalized_odds(model_id: str, y: np.ndarray | None, decision: np.ndarray, group: np.ndarray,
                   policy: Policy | None = None) -> TestResult:
    """Equalized odds (Hardt, Price and Srebro 2016): metric = the larger of the true positive rate gap and the false
    positive rate gap across groups. Outcome-conditioned: `not_evaluated` with its reason when no outcome is supplied or
    a group lacks a class; never run on HMDA. Light (policy key `fair.equalized_odds`, upper_bad, inclusive): green
    below the amber cut (0.05), amber from it, red from the red cut (0.10)."""
    return _error_rate_test(
        "fair.equalized_odds", model_id, y, decision, group, policy, metric_fn=_metrics.equalized_odds_difference,
        h0="descriptive: the true and the false positive rates are equal across groups",
        alt="the true or the false positive rate differs between groups",
        ref=f"{HARDT}; {DOSSIER_06_C7} l.1029-1033", what="the largest gap in the true or the false positive rate")


@register(
    "fair.equal_opportunity",
    family="fairness",
    name="Equal opportunity",
    h0="descriptive: the true positive rate is equal across groups",
    reference=f"{HARDT}; {DOSSIER_06_C7} l.1029-1033",
    decision="metric",
)
def equal_opportunity(model_id: str, y: np.ndarray | None, decision: np.ndarray, group: np.ndarray,
                      policy: Policy | None = None) -> TestResult:
    """Equal opportunity (Hardt, Price and Srebro 2016): metric = the true positive rate gap across groups.
    Outcome-conditioned like `equalized_odds`; policy key `fair.equal_opportunity` (upper_bad, inclusive, 0.05 and
    0.10)."""
    return _error_rate_test(
        "fair.equal_opportunity", model_id, y, decision, group, policy,
        metric_fn=_metrics.equal_opportunity_difference,
        h0="descriptive: the true positive rate is equal across groups",
        alt="the true positive rate differs between groups",
        ref=f"{HARDT}; {DOSSIER_06_C7} l.1029-1033", what="the largest gap in the true positive rate")


def _omitted(groups: np.ndarray, kept: Mapping[str, float]) -> list[str]:
    return [g for g in group_names(groups) if g not in kept]


@register(
    "fair.calibration_within_groups",
    family="fairness",
    name="Calibration within groups",
    h0="descriptive: the PD is as well calibrated within each group as within any other",
    reference=f"{DOSSIER_04_J3}; Naeini, Cooper and Hauskrecht (2015), doi:10.1609/aaai.v29i1.9602",
    decision="metric",
)
def calibration_within_groups(model_id: str, p: np.ndarray, y: np.ndarray | None, group: np.ndarray,
                              policy: Policy | None = None) -> TestResult:
    """Calibration within groups (dossier 04 J.3 l.904-905): metric = the largest expected calibration error (`pd_ece`,
    10 quantile bins) over the groups. Groups with fewer rows than the bins are not evaluated and are counted in
    `extras`. Outcome-conditioned like `equalized_odds`; policy key `fair.calibration_within_groups` (upper_bad,
    inclusive, 0.05 and 0.10)."""
    test_id = "fair.calibration_within_groups"
    h0 = "descriptive: the PD is as well calibrated within each group as within any other"
    alt = "the expected calibration error of some group is large"
    ref = f"{DOSSIER_04_J3}; Naeini, Cooper and Hauskrecht (2015), doi:10.1609/aaai.v29i1.9602"
    pol = _policy(policy)
    if y is None:
        return _not_evaluated(test_id, h0, alt, ref, _version(pol), NO_OUTCOME, model_id=model_id)
    yy = binary(y, "y")
    pp = np.array(p, dtype=float)
    if pp.ndim != 1 or pp.shape[0] != yy.shape[0]:
        raise ValueError(f"p must be one-dimensional with the {yy.shape[0]} rows of y, got shape {pp.shape}")
    gg = group_labels(group, yy.shape[0], "group")
    eces = _metrics.calibration_within_groups(pp, yy, gg)
    omitted = _omitted(gg, eces)
    extras = {"ece_by_group": eces, "omitted_groups": omitted, "n_omitted_groups": len(omitted), "n_bins": 10}
    if not eces:
        return _not_evaluated(test_id, h0, alt, ref, _version(pol), "no group has enough rows for 10 calibration bins",
                              model_id=model_id, n=int(yy.shape[0]), extras=extras)
    return _outcome_result(test_id, model_id, max(eces.values()), pol, h0=h0, alt=alt, ref=ref,
                           n=int(yy.shape[0]), n_events=int(yy.sum()), extras=extras,
                           notes="the largest expected calibration error over the groups; y = 1 is a default")


@register(
    "fair.auc_by_group",
    family="fairness",
    name="AUC by group",
    h0="descriptive: the score discriminates as well within each group as within any other",
    reference=f"{DOSSIER_04_J3}; Hanley and McNeil (1982), doi:10.1148/radiology.143.1.7063747",
    decision="metric",
)
def auc_by_group(model_id: str, score: np.ndarray, y: np.ndarray | None, group: np.ndarray,
                 policy: Policy | None = None) -> TestResult:
    """AUC by group (dossier 04 J.3 l.904-905): metric = the largest AUC gap between two groups (`disc_auc` within each
    group). Single-class groups are omitted and counted in `extras`; fewer than two groups left is `not_evaluated`.
    Outcome-conditioned like `equalized_odds`; policy key `fair.auc_by_group` (upper_bad, inclusive, 0.05 and 0.10)."""
    test_id = "fair.auc_by_group"
    h0 = "descriptive: the score discriminates as well within each group as within any other"
    alt = "the AUC of some group differs from that of another"
    ref = f"{DOSSIER_04_J3}; Hanley and McNeil (1982), doi:10.1148/radiology.143.1.7063747"
    pol = _policy(policy)
    if y is None:
        return _not_evaluated(test_id, h0, alt, ref, _version(pol), NO_OUTCOME, model_id=model_id)
    yy = binary(y, "y")
    ss = np.array(score, dtype=float)
    if ss.ndim != 1 or ss.shape[0] != yy.shape[0]:
        raise ValueError(f"score must be one-dimensional with the {yy.shape[0]} rows of y, got shape {ss.shape}")
    gg = group_labels(group, yy.shape[0], "group")
    aucs = _metrics.auc_by_group(ss, yy, gg)
    omitted = _omitted(gg, aucs)
    extras = {"auc_by_group": aucs, "omitted_groups": omitted, "n_omitted_groups": len(omitted)}
    if len(aucs) < 2:
        return _not_evaluated(test_id, h0, alt, ref, _version(pol),
                              "fewer than two groups have both classes of the outcome", model_id=model_id,
                              n=int(yy.shape[0]), extras=extras)
    return _outcome_result(test_id, model_id, max(aucs.values()) - min(aucs.values()), pol, h0=h0, alt=alt, ref=ref,
                           n=int(yy.shape[0]), n_events=int(yy.sum()), extras=extras,
                           notes="the largest AUC gap between two groups; y = 1 is the class the score ranks higher")


# ----------------------------------------------------------------------------------------------------------------
# Controlled disparity, replication and situation testing


def _interval_test(test_id: str, model_id: str, effect: Effect | None, controls: Sequence[str],
                   missing_controls: Sequence[str], *, metric_of: Any, unit: str, h0: str, alt: str,
                   ref: str) -> TestResult:
    ctrl = _check_text("controls", controls)
    miss = _check_text("missing_controls", missing_controls)
    pol = fairness_default_policy()
    if effect is not None and not isinstance(effect, Effect):
        raise ValueError(f"effect must be an Effect, got {type(effect).__name__}")
    extras: dict[str, Any] = {"controls": ctrl, "missing_controls": miss}
    version = _version(pol, interval_level=0.95)
    if effect is None:
        return _not_evaluated(test_id, h0, alt, ref, version,
                              "no effect was supplied: the model was not fitted or did not converge",
                              model_id=model_id, extras=extras)
    vals = (effect.coefficient, effect.odds_ratio, effect.lo, effect.hi, effect.level)
    if not all(math.isfinite(v) for v in vals):
        return _not_evaluated(test_id, h0, alt, ref, version, "the effect or its interval is not finite",
                              model_id=model_id, extras=extras)
    if not 0.0 < effect.level < 1.0:
        return _not_evaluated(test_id, h0, alt, ref, version, f"the interval level {effect.level} is invalid",
                              model_id=model_id, extras=extras)
    z_level = float(stats.norm.ppf(0.5 + effect.level / 2.0))
    se = (effect.hi - effect.lo) / (2.0 * z_level)
    if not se > 0.0:
        return _not_evaluated(test_id, h0, alt, ref, version,
                              "the interval has zero width: the Wald statistic is undefined", model_id=model_id,
                              extras=extras)
    z = effect.coefficient / se
    excludes = effect.lo > 0.0 or effect.hi < 0.0
    miss_text = ", ".join(miss) if miss else "none stated"
    extras.update({"effect": {"coefficient": effect.coefficient, "odds_ratio": effect.odds_ratio, "lo": effect.lo,
                              "hi": effect.hi, "level": effect.level},
                   "standard_error": se, "unit": unit,
                   "thresholds": {"interval_level": effect.level,
                                  "rule": "amber when the interval excludes 0; never red"}})
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=ref,
        policy_version=_version(pol, interval_level=effect.level),
        light=Light.AMBER if excludes else Light.GREEN, statistic=z, p_value=float(2.0 * stats.norm.sf(abs(z))),
        metric=metric_of(effect), model_id=model_id,
        inputs_hash=inputs_hash(vals, ctrl, miss),
        notes=f"an approximation of comparative evidence controlling for {', '.join(ctrl) or 'no stated control'}; "
              f"missing controls: {miss_text}, so the estimate may be confounded by them; never a finding of a "
              f"violation; {NO_LEGAL_CLAIM}",
        extras=extras)


@register(
    "fair.conditional_disparity",
    family="fairness",
    name="Conditional disparity",
    h0="the protected indicator has no effect on the decision once the permitted controls are held fixed",
    reference=(
        f"decision Q4 of the Contraste U14 sizing: logistic decision model with Wald interval; {COMPANION}; "
        "the Interagency Fair Lending Examination Procedures (gross HMDA disparities are not an adverse impact "
        f"analysis); {NO_LEGAL_CLAIM}"),
    decision="metric",
)
def conditional_disparity(model_id: str, effect: Effect, controls: Sequence[str],
                          missing_controls: Sequence[str]) -> TestResult:
    """The disparity of the decision that persists after the permitted controls (dossier 08 l.839; decision Q4): the
    protected effect of `fit_decision_model` with its interval. statistic = the Wald z recomputed from the interval,
    p_value its two-sided p-value, metric = the odds ratio. Light: amber when the interval (95 percent) excludes 0,
    green otherwise; never red, never a finding of a violation. The controls and the missing controls are named in
    `notes` and `extras`. No policy threshold: the rule is the whole classification."""
    return _interval_test(
        "fair.conditional_disparity", model_id, effect, controls, missing_controls, metric_of=lambda e: e.odds_ratio,
        unit="odds ratio",
        h0="the protected indicator has no effect on the decision once the permitted controls are held fixed",
        alt="the log-odds of the favourable decision differ with the protected indicator given the controls",
        ref=f"decision Q4 of the Contraste U14 sizing: logistic decision model with Wald interval; {COMPANION}; "
            "the Interagency Fair Lending Examination Procedures (gross HMDA disparities are not an adverse impact "
            f"analysis); {NO_LEGAL_CLAIM}")


@register(
    "fair.rate_spread_disparity",
    family="fairness",
    name="Rate spread disparity",
    h0="the protected indicator has no effect on the rate spread once the permitted controls are held fixed",
    reference=f"decision Q4 of the Contraste U14 sizing: linear rate-spread model with Wald interval; {NO_LEGAL_CLAIM}",
    decision="metric",
)
def rate_spread_disparity(model_id: str, effect: Effect, controls: Sequence[str],
                          missing_controls: Sequence[str]) -> TestResult:
    """The pricing disparity on the rate spread by the same construction (a linear model, `fit_rate_spread_model`):
    metric = the coefficient in percentage points; the light, the statistic and the rule are those of
    `conditional_disparity`. No policy threshold."""
    return _interval_test(
        "fair.rate_spread_disparity", model_id, effect, controls, missing_controls, metric_of=lambda e: e.coefficient,
        unit="percentage points of rate spread",
        h0="the protected indicator has no effect on the rate spread once the permitted controls are held fixed",
        alt="the rate spread differs with the protected indicator given the controls",
        ref=f"decision Q4 of the Contraste U14 sizing: linear rate-spread model with Wald interval; {NO_LEGAL_CLAIM}")


@register(
    "fair.replication",
    family="fairness",
    name="Replication",
    h0="the estimate fitted on year t has the same sign on year t+1",
    reference="decision Q4 of the Contraste U14 sizing; dossier 08 l.839 (is the finding stable year to year)",
    decision="metric",
)
def replication(model_id: str, replication: Replication) -> TestResult:
    """Replication of an estimate on the next year by sign and size (decision Q4). metric = the size ratio, confirm
    over fit. Light: amber when the sign differs between year t and year t+1, green otherwise; `not_evaluated` when the
    fit coefficient is 0 (the ratio is undefined). No policy threshold: sign and size are the replication test."""
    test_id = "fair.replication"
    h0 = "the estimate fitted on year t has the same sign on year t+1"
    alt = "the sign of the estimate differs between year t and year t+1"
    ref = "decision Q4 of the Contraste U14 sizing; dossier 08 l.839 (is the finding stable year to year)"
    if not isinstance(replication, Replication):
        raise ValueError(f"replication must be a Replication, got {type(replication).__name__}")
    pol = fairness_default_policy()
    version = _version(pol, rule="sign_and_size")
    extras = {"fit_coefficient": replication.fit_coefficient, "confirm_coefficient": replication.confirm_coefficient,
              "same_sign": replication.same_sign, "size_ratio": replication.size_ratio}
    if replication.size_ratio is None:
        return _not_evaluated(test_id, h0, alt, ref, version, "the fit coefficient is 0: the size ratio is undefined",
                              model_id=model_id, extras=extras)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=ref, policy_version=version,
        light=Light.GREEN if replication.same_sign else Light.AMBER, metric=replication.size_ratio, model_id=model_id,
        inputs_hash=inputs_hash(replication.fit_coefficient, replication.confirm_coefficient),
        notes=f"replication by sign and size only, no verdict beyond them; {NO_LEGAL_CLAIM}",
        extras={**extras, "thresholds": {"rule": "amber when the sign differs between year t and year t+1"}})


@register(
    "fair.situation_testing",
    family="fairness",
    name="Situation testing",
    h0="descriptive: no complainant of the protected group is treated worse than a comparable one of the others",
    reference=f"{COMPANION}; dossier 07 A.3 l.158-177",
    decision="metric",
)
def situation_testing(model_id: str, result: SituationResult, policy: Policy | None = None) -> TestResult:
    """Situation testing as an individual-level indicator (dossier 07 A.3 l.158-177). metric = n_significant over
    n_complainants. Light (policy key `fair.situation_testing`, upper_bad, inclusive): amber when the significant share
    reaches the cut point (0.20, a convention); never red (a red
    light from a caller policy is capped at amber); no legal claim. `not_evaluated` without complainants."""
    test_id = "fair.situation_testing"
    h0 = "descriptive: no complainant of the protected group is treated worse than a comparable one of the others"
    alt = "complainants of the protected group are treated worse than their comparators"
    ref = f"{COMPANION}; dossier 07 A.3 l.158-177"
    if not isinstance(result, SituationResult):
        raise ValueError(f"result must be a SituationResult, got {type(result).__name__}")
    pol = _policy(policy)
    version = _version(pol)
    extras = {"k": result.k, "tau": result.tau, "alpha": result.alpha, "n_cases": result.n_cases,
              "n_significant": result.n_significant, "n_complainants": result.n_complainants}
    if result.n_complainants == 0:
        return _not_evaluated(test_id, h0, alt, ref, version, "there are no complainants", model_id=model_id, n=0,
                              extras=extras)
    metric = result.n_significant / result.n_complainants
    light, th = _classified(pol, test_id, metric=metric)
    if th is None:
        return _not_evaluated(test_id, h0, alt, ref, version, f"policy {pol.label} holds no threshold for {test_id}",
                              model_id=model_id, n=result.n_complainants, extras={**extras, "thresholds": {}})
    light, capped = _cap_at_amber(light)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=ref,
        policy_version=_version(pol, alpha_amber=th.amber, k=result.k, tau=result.tau, alpha=result.alpha),
        light=light, metric=metric, n=result.n_complainants,
        n_events=result.n_significant, model_id=model_id,
        inputs_hash=inputs_hash(result.n_complainants, result.n_cases, result.n_significant, result.k, result.tau,
                                result.alpha),
        notes=f"individual-level indicator over the complainants of the protected group, centers excluded; the "
              f"cut point is a convention, never a regulatory threshold, and sits in extras.thresholds, not in the "
              f"p-value alphas; never red; {NO_LEGAL_CLAIM}{capped}",
        extras={**extras, "thresholds": _threshold_extras(test_id, th)})


__all__ = [
    "adverse_impact",
    "auc_by_group",
    "calibration_within_groups",
    "conditional_disparity",
    "demographic_parity",
    "equal_opportunity",
    "equalized_odds",
    "fisher_exact",
    "rate_spread_disparity",
    "replication",
    "situation_testing",
    "two_proportion_z",
]
