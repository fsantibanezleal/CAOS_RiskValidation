"""The liquidity validation tests (dossier 05 E.1 to E.5 and its acceptance rows; dossier 08 G.2 and G.4): each
returns one
`TestResult` that validates against `common/test-result`.

Acceptance gates are Contraste internal gates, stated as such, never regulatory thresholds (decision Q14: the named
parameters of the case documents). Each function states its light rule in its docstring and records its parameters in
`policy_version` and in `extras["thresholds"]`, so that a view can explain a light. A test that cannot be evaluated
returns
the light `not_evaluated` with its reason in `notes` and a null statistic, metric and p-value once serialised, never a
fabricated number. The tests are plain functions that do not read the registry or the default policy: their rules are
the
contract's (`engine-liquidity-api` 1.2.0, module validation.liquidity). They state no compliance claim. The tests of
`regulatory.lcr_nsfr` are constructed tests and invariants (`tests/liquidity/test_lcr_nsfr.py`), not `TestResult`
functions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.liquidity import cbc as _cbc
from riskvalidation.liquidity import ladder as _ladder
from riskvalidation.liquidity import runoff as _runoff
from riskvalidation.liquidity._checks import check_increasing_grid, check_unit_share
from riskvalidation.liquidity.ladder import SurvivalResult

POLICY_LABEL = "contraste-liquidity-battery-1.0.0"
NAN = float("nan")
EPS = 1e-12             # tolerance on a gate comparison, so that a value exactly on a gate is not moved by rounding

LIST_NOTE = ("ECB Banking Supervision, Sensitivity analysis of liquidity risk, stress test 2019, methodological note "
             "(2019-02-06) and results annex (2019-10-07), https://www.bankingsupervision.europa.eu/")
REF_COMPOUNDING = (
    "dossier 05 E.1 line 237 (constant daily compounding on relevant days, Table 3.3 and para 107 of the LiST note) "
    f"and the SDD acceptance row of the unit sizing; {LIST_NOTE}")
REF_MONOTONE = (
    "dossier 05 E.1 line 240 (median survival periods 176 and 122 days) and line 480, SDD line 329: the survival "
    f"period does not lengthen as the severity rises; {LIST_NOTE}")
REF_HAIRCUTS = (
    "dossier 05 E.1 line 238 and line 480 (LiST note para 74, Table 2.1: the haircut order, the equity table, "
    f"30 and 50 percent); {LIST_NOTE}")
REF_NLP = ("dossier 05 E.1 line 234: NLP(t) = CBC(t) - sum over s <= t of (O(s) - I(s)), "
           "checked on hand-computed ladders")
REF_ADEQUACY = (
    "dossier 08 lines 566 to 568 and dossier 06 lines 971 to 973: a Basel outflow assumption back-tested against "
    "the observed run rate (BCBS d555 Box A, Table A.1, https://www.bis.org/bcbs/publ/d555.htm); "
    "a descriptive score on the real event, one observation, no p-value (decision Q5)")
REF_GENERATOR = (
    "dossier 08 line 698 and G.4 lines 739 to 742: the run-off estimate against the depositor generator's exact "
    "truth, in Monte Carlo standard errors (the generator test of decision Q5)")
REF_CLIFF = (
    "dossier 05 E.1 line 236 defines the cliff effect (NLP(35) - NLP(30)) / TA as a diagnostic of LCR window "
    f"optimisation and no limit, so the lights are conventions (decision Q14); {LIST_NOTE}")
REF_HTM = (
    "dossier 05 lines 263, 278 and 480: the CBC and the survival period with and without held-to-maturity "
    "unrealised losses (Jiang, Matvos, Piskorski and Seru, doi:10.1016/j.jfineco.2024.103899)")
REF_UNINSURED = ("dossier 05 line 278: the survival period does not lengthen as the uninsured share of deposits "
                 "rises")
REF_REVERSE = ("dossier 05 E.5 rung 5, line 279: the reverse liquidity stress round trip (the survival period at "
               "the solved multiplier equals the target)")
REF_REVALUATION = (
    "dossier 05 line 263 and decision Q11: the HTM book revalued from its maturity profile under the Treasury "
    "curve through the engine's irrbb curves against the reported fair value SCHF (FDIC BankFind fields, "
    "addendum L3)")


def _version(**params: Any) -> str:
    return "; ".join([POLICY_LABEL, *(f"{k}={v}" for k, v in params.items())])


def _not_evaluated(test_id: str, h0: str, alternative: str, reference: str, version: str, reason: str, *,
                   n: int | None = None, segment: str | None = None,
                   extras: Mapping[str, Any] | None = None) -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=version,
                      light=Light.NOT_EVALUATED, statistic=NAN, notes=f"not evaluated: {reason}", n=n, segment=segment,
                      extras=dict(extras or {}))


def _scalar(x: Any, name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: must be numeric, got {x!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name}: must be finite, got {x!r}")
    return v


def _nonneg_tol(x: Any, name: str) -> float:
    """A tolerance: finite and not negative (a negative tolerance would turn a shortening into a failure)."""
    v = _scalar(x, name)
    if v < 0.0:
        raise ValueError(f"{name}: must be non-negative, got {x!r}")
    return v


def _vec(x: Any, name: str) -> np.ndarray:
    a = np.array(x, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a non-empty one-dimensional array of finite values, got shape {a.shape}")
    return a


def _same_length(**arrays: np.ndarray) -> None:
    sizes = {k: v.shape[0] for k, v in arrays.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"{', '.join(sizes)} must have the same length, got {sizes}")


def _days(values: Sequence[float | None], name: str) -> np.ndarray:
    """Survival periods in days with `None` ("survives the horizon") as infinity, longer than any day."""
    out = []
    for v in values:
        out.append(math.inf if v is None else _scalar(v, name))
    return np.array(out, dtype=float)


def _largest_increase(days: np.ndarray) -> float:
    """The largest lengthening of the survival period along the order, in days, at least 0; a finite period followed by
    "survives the horizon" is an unbounded lengthening (infinity)."""
    worst = 0.0
    for a, b in zip(days[:-1], days[1:], strict=True):
        if math.isinf(b) and not math.isinf(a):
            return math.inf
        if math.isinf(a):
            continue
        worst = max(worst, float(b - a))
    return worst


def _three_light_high(value: float, green: float, amber: float) -> Light:
    """Green at or above `green`, amber at or above `amber`, red below (with a rounding tolerance)."""
    if value >= green - EPS:
        return Light.GREEN
    return Light.AMBER if value >= amber - EPS else Light.RED


def _three_light_low(value: float, green: float, amber: float) -> Light:
    """Green at or below `green`, amber at or below `amber`, red above (with a rounding tolerance)."""
    if value <= green + EPS:
        return Light.GREEN
    return Light.AMBER if value <= amber + EPS else Light.RED


# ----------------------------------------------------------------------------------------------------------------
# The goldens and identities


@register(
    "liq.list_compounding_golden",
    family="liquidity",
    name="LiST compounding golden",
    h0="the engine reproduces the LiST compounding golden of dossier 05 line 237 within the printed precision",
    reference=REF_COMPOUNDING,
    decision="metric",
)
def list_compounding_golden(q: float = 0.00103258, n: int = 125, expected: float = 0.121148,
                            tol: float = 5e-7) -> TestResult:
    """Reproduces the LiST compounding of dossier 05 line 237: stable retail sight deposits at `q = 0.103258`
    percent per
    relevant day for `n = 125` relevant days give a cumulative outflow of 12.11 percent.

    statistic = `abs(1 - (1 - q)^n - expected)`, metric = the computed cumulative outflow, p_value None. Light (a
    Contraste internal gate): green when the statistic is at most `tol` (the named parameter
    `list_compounding_tolerance`), red otherwise; `extras["thresholds"]` holds the tolerance. At the defaults the metric
    is
    `0.1211484` (`0.12114837`) and the statistic `3.7e-7`."""
    test_id = "liq.list_compounding_golden"
    h0 = "the engine reproduces the LiST compounding golden of dossier 05 line 237 within the printed precision"
    alt = "the engine departs from the printed cumulative outflow"
    rate = float(check_unit_share(q, "q")[0])
    days = int(_scalar(n, "n"))
    exp, tolerance = _scalar(expected, "expected"), _scalar(tol, "tol")
    if tolerance < 0.0:
        raise ValueError(f"tol: must be non-negative, got {tol}")
    value = float(_runoff.cumulative_outflow_constant(rate, days))
    stat = abs(value - exp)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_COMPOUNDING,
        policy_version=_version(tolerance=tolerance),
        light=Light.GREEN if stat <= tolerance + EPS else Light.RED, statistic=stat, metric=value, n=days,
        inputs_hash=inputs_hash(np.array([rate, days, exp])), extras={"thresholds": {"tolerance": tolerance}})


@register(
    "liq.survival_monotone_in_severity",
    family="liquidity",
    name="Survival monotone in severity",
    h0="the survival period does not lengthen as the scenario severity rises",
    reference=REF_MONOTONE,
    decision="metric",
)
def survival_monotone_in_severity(survival_days: Sequence[float | None], tol_days: float = 0.0) -> TestResult:
    """The survival period does not lengthen as the severity rises (dossier 05 line 480; SDD l.329).

    `survival_days` is ordered from the mildest to the most severe scenario, in relevant days; `None` is "survives the
    horizon" and counts as longer than any day. statistic = the largest increase of the survival period along the order,
    in days (at least 0; a finite period followed by "survives the horizon" is an unbounded increase, null once
    serialised). Light (a Contraste internal gate): green when no step lengthens the survival period by more than
    `tol_days` (the named parameter `monotonicity_tolerance_days`), red otherwise; fewer than two scenarios:
    not_evaluated. Golden: `[None, 176, 122]` (the LiST 2019 median survival periods, 05 line 240) is green with
    statistic 0; `[100, 120]` is red with statistic 20."""
    test_id = "liq.survival_monotone_in_severity"
    h0 = "the survival period does not lengthen as the scenario severity rises"
    alt = "a more severe scenario survives longer than a milder one"
    tol = _nonneg_tol(tol_days, "tol_days")
    days = _days(survival_days, "survival_days")
    version = _version(tolerance_days=tol)
    thresholds = {"tolerance_days": tol}
    if days.size < 2:
        return _not_evaluated(test_id, h0, alt, REF_MONOTONE, version, "fewer than 2 scenarios", n=int(days.size),
                              extras={"thresholds": thresholds})
    stat = _largest_increase(days)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_MONOTONE, policy_version=version,
        light=Light.GREEN if stat <= tol + EPS else Light.RED, statistic=stat, n=int(days.size),
        inputs_hash=inputs_hash(np.where(np.isinf(days), -1.0, days)),
        extras={"thresholds": thresholds, "survives_horizon": [bool(math.isinf(d)) for d in days]},
        notes="a finite survival period is followed by 'survives the horizon': an unbounded lengthening"
        if math.isinf(stat) else "")


_HAIRCUT_CASES: tuple[tuple[str, dict[str, Any], float, str], ...] = (
    ("equity", {"equity_rating": "AAA-A-"}, 0.20, "ecb-equity-table"),
    ("equity", {"equity_rating": "BBB+-BBB-"}, 0.35, "ecb-equity-table"),
    ("equity", {"equity_rating": "lower-or-unrated"}, 0.50, "ecb-equity-table"),
    ("non-equity", {"hqla_level": "level_1"}, 0.30, "ecb-non-equity-hqla"),
    ("non-equity", {}, 0.50, "fifty-percent-fallback"),
    ("non-equity", {"central_bank_eligible": True, "central_bank_haircut": 0.04}, 0.04, "central-bank-framework"),
    ("non-equity", {"central_bank_eligible": True, "hqla_level": "level_2a"}, 0.15, "lcr-haircut-hqla"),
    ("non-equity", {"central_bank_eligible": True}, 0.50, "fifty-percent-fallback"),
    ("cash", {}, 0.0, "nominal-value"),
)


@register(
    "liq.cbc_haircut_rules",
    family="liquidity",
    name="CBC haircut rules",
    h0="the engine's haircut rules reproduce the order and values of LiST note para 74 on the nine printed cases",
    reference=REF_HAIRCUTS,
    decision="metric",
)
def cbc_haircut_rules(tol: float = 1e-12) -> TestResult:
    """Runs `cbc.haircut_rule` on the nine cases of dossier 05 line 238 (the three equity bands, non-equity HQLA
    and not,
    central bank given, eligible and undetermined HQLA and not, cash) and compares the haircut and the rule label with
    the
    printed ones.

    statistic = the largest absolute haircut deviation, n = 9. Light (a Contraste internal gate): green when every
    haircut
    and label equals the expected one (deviation at most `tol`), red otherwise (a wrong label alone is red; a case that
    raises counts as failed with an infinite deviation, null once serialised); `extras["thresholds"]` holds the
    tolerance."""
    test_id = "liq.cbc_haircut_rules"
    h0 = "the engine's haircut rules reproduce the order and values of LiST note para 74 on the nine printed cases"
    alt = "a haircut or a rule label departs from the note"
    tolerance = _scalar(tol, "tol")
    if tolerance < 0.0:
        raise ValueError(f"tol: must be non-negative, got {tol}")
    deviations, wrong_labels, cases = [], 0, []
    for kind, kwargs, haircut, label in _HAIRCUT_CASES:
        try:
            got_h, got_label = _cbc.haircut_rule(kind, **kwargs)
        except ValueError as exc:
            got_h, got_label = math.inf, f"raised: {exc}"
        deviations.append(abs(got_h - haircut))
        wrong_labels += int(got_label != label)
        cases.append({"kind": kind, "expected": [haircut, label], "got": [got_h, got_label]})
    stat = max(deviations)
    ok = stat <= tolerance + EPS and wrong_labels == 0
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_HAIRCUTS, policy_version=_version(tolerance=tolerance),
        light=Light.GREEN if ok else Light.RED, statistic=stat, n=len(_HAIRCUT_CASES),
        inputs_hash=inputs_hash(np.array(deviations)), extras={"thresholds": {"tolerance": tolerance}, "cases": cases,
                                                               "wrong_labels": wrong_labels},
        notes="" if ok else f"{wrong_labels} wrong label(s); the largest haircut deviation is {stat}")


@register(
    "liq.nlp_identity",
    family="liquidity",
    name="NLP identity",
    h0="the net liquidity position equals the CBC less the cumulated net outflows on every day",
    reference=REF_NLP,
    decision="metric",
)
def nlp_identity(cbc: np.ndarray, outflows: np.ndarray, inflows: np.ndarray, nlp: np.ndarray,
                 tol: float = 1e-6) -> TestResult:
    """The net liquidity position identity of dossier 05 E.1 line 234 on a ladder: `NLP(t) = CBC(t) - sum over s <= t of
    (O(s) - I(s))`. Money (H,) arrays in USD thousands.

    statistic = the largest absolute residual of `nlp - (cbc - cumsum(outflows - inflows))`. Light (a Contraste internal
    gate): green when the statistic is at most `tol` (the named parameter `nlp_identity_tolerance`), red otherwise;
    `extras["thresholds"]` holds the tolerance. Golden: the hand ladder `cbc = [100]*4`, outflows `[30, 40, 50, 60]`,
    inflows `[0, 10, 0, 0]`, `nlp = [70, 40, -10, -70]` gives statistic 0 and green."""
    test_id = "liq.nlp_identity"
    h0 = "the net liquidity position equals the CBC less the cumulated net outflows on every day"
    alt = "the net liquidity position departs from the identity"
    c, o, i, p = _vec(cbc, "cbc"), _vec(outflows, "outflows"), _vec(inflows, "inflows"), _vec(nlp, "nlp")
    _same_length(cbc=c, outflows=o, inflows=i, nlp=p)
    tolerance = _scalar(tol, "tol")
    residual = p - (c - np.cumsum(o - i))
    stat = float(np.max(np.abs(residual)))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_NLP, policy_version=_version(tolerance=tolerance),
        light=Light.GREEN if stat <= tolerance + EPS else Light.RED, statistic=stat, n=int(c.size),
        inputs_hash=inputs_hash(c, o, i, p), extras={"thresholds": {"tolerance": tolerance}})


@register(
    "liq.runoff_adequacy_back_test",
    family="liquidity",
    name="Runoff adequacy back test",
    h0="the modelled outflow share is at least as large as the realised one",
    reference=REF_ADEQUACY,
    decision="metric",
)
def runoff_adequacy_back_test(modelled_share: float, realised_share: float, ratio_green: float = 0.8,
                              ratio_amber: float = 0.5, compared_at: str = "event-horizon",
                              denominator: str | None = None) -> TestResult:
    """A Basel outflow assumption back-tested against the observed rate (dossier 08 lines 566 to 568; 06 lines 971
    to 973):
    a descriptive score on the real event, one observation, no p-value (decision Q5). Shares of the deposit base.

    statistic = the ratio `modelled_share / realised_share`, metric = the shortfall `realised_share - modelled_share`,
    `segment` = the denominator id. Light (a Contraste internal gate): green when the ratio is at least `ratio_green`,
    amber when at least `ratio_amber`, red below (the named parameters `scoring_ratio_green` and `scoring_ratio_amber`);
    not_evaluated when `realised_share` is 0. `compared_at` is `"event-horizon"` or `"model-horizon"`;
    `extras["thresholds"]` holds both ratios and `compared_at`. Golden: modelled 0.013723 against 0.85 gives ratio
    0.016145, shortfall 0.836277 and red."""
    test_id = "liq.runoff_adequacy_back_test"
    h0 = "the modelled outflow share is at least as large as the realised one"
    alt = "the modelled outflow share falls short of the realised one"
    modelled = float(check_unit_share(modelled_share, "modelled_share")[0])
    realised = float(check_unit_share(realised_share, "realised_share")[0])
    green, amber = _scalar(ratio_green, "ratio_green"), _scalar(ratio_amber, "ratio_amber")
    if not 0.0 <= amber <= green:
        raise ValueError(f"ratio_amber and ratio_green: must satisfy 0 <= ratio_amber <= ratio_green, got {amber} and "
                         f"{green}")
    if compared_at not in ("event-horizon", "model-horizon"):
        raise ValueError(f"compared_at: must be 'event-horizon' or 'model-horizon', got {compared_at!r}")
    thresholds = {"ratio_green": green, "ratio_amber": amber, "compared_at": compared_at}
    version = _version(ratio_green=green, ratio_amber=amber, compared_at=compared_at)
    if realised == 0.0:
        return _not_evaluated(test_id, h0, alt, REF_ADEQUACY, version,
                              "realised_share is 0, so the ratio is undefined", n=1, segment=denominator,
                              extras={"thresholds": thresholds})
    ratio = modelled / realised
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_ADEQUACY, policy_version=version,
        light=_three_light_high(ratio, green, amber), statistic=ratio, metric=realised - modelled, n=1,
        segment=denominator, inputs_hash=inputs_hash(np.array([modelled, realised])),
        extras={"thresholds": thresholds})


@register(
    "liq.runoff_vs_generator_truth",
    family="liquidity",
    name="Runoff vs generator truth",
    h0="the run-off estimate reproduces the generator's exact truth within Monte Carlo error",
    reference=REF_GENERATOR,
    decision="metric",
)
def runoff_vs_generator_truth(estimated: np.ndarray, truth: np.ndarray, mc_se: np.ndarray, se_green: float = 2.0,
                              se_red: float = 3.0) -> TestResult:
    """The run-off estimate against the depositor generator's exact truth (dossier 08 line 698, G.4 lines 739 to 742).
    `estimated`, `truth` and `mc_se` are `(K,)` by segment and horizon; `mc_se` is the Monte Carlo standard error of the
    generator's verification.

    statistic = the largest `abs(estimated - truth) / mc_se`, in standard errors, `n = K`. Light (a Contraste internal
    gate): green when the statistic is at most `se_green`, amber when at most `se_red`, red above (the named parameters
    `generator_truth_se_green` and `generator_truth_se_red`); a zero or negative `mc_se` raises `ValueError`;
    `extras["thresholds"]` holds both. Golden: `estimated = truth` gives 0 and green; an entry off by `2.5 x mc_se`
    gives
    2.5 and amber; by `3.5 x mc_se` gives 3.5 and red."""
    test_id = "liq.runoff_vs_generator_truth"
    h0 = "the run-off estimate reproduces the generator's exact truth within Monte Carlo error"
    alt = "the run-off estimate departs from the generator's truth by more than Monte Carlo error"
    est, tru, se = _vec(estimated, "estimated"), _vec(truth, "truth"), _vec(mc_se, "mc_se")
    _same_length(estimated=est, truth=tru, mc_se=se)
    if np.any(se <= 0.0):
        raise ValueError(f"mc_se: every standard error must be positive, got {float(se.min())}")
    green, red = _scalar(se_green, "se_green"), _scalar(se_red, "se_red")
    if not 0.0 <= green <= red:
        raise ValueError(f"se_green and se_red: must satisfy 0 <= se_green <= se_red, got {green} and {red}")
    z = np.abs(est - tru) / se
    stat = float(z.max())
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_GENERATOR,
        policy_version=_version(se_green=green, se_red=red),
        light=_three_light_low(stat, green, red), statistic=stat, n=int(est.size),
        inputs_hash=inputs_hash(est, tru, se),
        extras={"thresholds": {"se_green": green, "se_red": red}, "z": z.tolist()})


@register(
    "liq.cliff_effect_diagnostic",
    family="liquidity",
    name="Cliff effect diagnostic",
    h0="the net liquidity position does not drop between the near and far day by more than the convention allows",
    reference=REF_CLIFF,
    decision="metric",
)
def cliff_effect_diagnostic(nlp: np.ndarray, total_assets: float, near: int = 30, far: int = 35, amber: float = -0.01,
                            red: float = -0.03) -> TestResult:
    """The cliff effect of dossier 05 E.1 line 236: a diagnostic of LCR window optimisation. `nlp` is `(H,)` money, the
    cliff a share of total assets.

    statistic = `ladder.cliff_effect(nlp, total_assets, near, far)`, metric = `NLP(far) - NLP(near)` in money. Light (a
    convention, decision Q14, as 05 E.1 line 236 defines no limit): green when the statistic is at least `amber`, amber
    when at least `red`, red below (the named parameters `cliff_amber_threshold` and `cliff_red_threshold`);
    not_evaluated with the reason in `notes` when the horizon is shorter than `far`; `extras["thresholds"]` holds both.
    Golden: `NLP(30) = 100`, `NLP(35) = 40`, `TA = 1000` gives `-0.06` and red."""
    test_id = "liq.cliff_effect_diagnostic"
    h0 = "the net liquidity position does not drop between the near and far day by more than the convention allows"
    alt = "the net liquidity position drops between the near and far day (an LCR window effect)"
    x = _vec(nlp, "nlp")
    ta = _scalar(total_assets, "total_assets")
    a, r = _scalar(amber, "amber"), _scalar(red, "red")
    if r > a:
        raise ValueError(f"amber and red: must satisfy red <= amber, got amber = {a} and red = {r}")
    thresholds = {"amber": a, "red": r}
    version = _version(near=int(near), far=int(far), amber=a, red=r)
    if x.size < far:
        return _not_evaluated(test_id, h0, alt, REF_CLIFF, version,
                              f"the horizon of {x.size} days is shorter than far = {int(far)}", n=int(x.size),
                              extras={"thresholds": thresholds})
    stat = _ladder.cliff_effect(x, ta, int(near), int(far))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_CLIFF, policy_version=version,
        light=_three_light_high(stat, a, r), statistic=stat, metric=float(x[int(far) - 1] - x[int(near) - 1]),
        n=int(x.size), inputs_hash=inputs_hash(x, np.array([ta, near, far])), extras={"thresholds": thresholds})


@register(
    "liq.htm_loss_effect",
    family="liquidity",
    name="HTM loss effect",
    h0="the held-to-maturity unrealised loss does not raise the CBC and does not lengthen the survival period",
    reference=REF_HTM,
    decision="metric",
)
def htm_loss_effect(cbc_without: float, cbc_with: float, survival_without: float | None,
                    survival_with: float | None) -> TestResult:
    """The CBC and the survival period with and without the held-to-maturity unrealised loss (dossier 05 lines 263, 278
    and 480). Money for the CBC; survival periods in relevant days with `None` as "survives the horizon" (longer than
    any
    day).

    statistic = the CBC removed, `cbc_without - cbc_with`, metric = its share of `cbc_without`. Light (a Contraste
    internal
    gate): green when the unrealised loss does not raise the CBC (statistic at least 0) and does not lengthen the
    survival
    period, red otherwise; not_evaluated when `cbc_without` is 0; `extras["thresholds"] = {"tolerance": 0.0}`. Golden:
    `(72, 58.5, None, 48)` gives 13.5, 0.1875 and green; `(72, 80, 40, 60)` gives red."""
    test_id = "liq.htm_loss_effect"
    h0 = "the held-to-maturity unrealised loss does not raise the CBC and does not lengthen the survival period"
    alt = "the unrealised loss raises the CBC or lengthens the survival period"
    without, with_loss = _scalar(cbc_without, "cbc_without"), _scalar(cbc_with, "cbc_with")
    sw = math.inf if survival_without is None else _scalar(survival_without, "survival_without")
    sl = math.inf if survival_with is None else _scalar(survival_with, "survival_with")
    thresholds = {"tolerance": 0.0}
    version = _version(tolerance=0.0)
    if without == 0.0:
        return _not_evaluated(test_id, h0, alt, REF_HTM, version, "cbc_without is 0, so its share is undefined", n=2,
                              extras={"thresholds": thresholds})
    removed = without - with_loss
    ok = removed >= -EPS and sl <= sw
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_HTM, policy_version=version,
        light=Light.GREEN if ok else Light.RED, statistic=removed, metric=removed / without, n=2,
        inputs_hash=inputs_hash(np.array([without, with_loss, -1.0 if math.isinf(sw) else sw,
                                          -1.0 if math.isinf(sl) else sl])),
        extras={"thresholds": thresholds, "survives_horizon": {"without": math.isinf(sw), "with": math.isinf(sl)}})


@register(
    "liq.uninsured_share_sensitivity",
    family="liquidity",
    name="Uninsured share sensitivity",
    h0="the survival period does not lengthen as the uninsured share rises",
    reference=REF_UNINSURED,
    decision="metric",
)
def uninsured_share_sensitivity(shares: Sequence[float], survival_days: Sequence[float | None],
                                tol_days: float = 0.0) -> TestResult:
    """The survival period does not lengthen as the uninsured share rises (dossier 05 line 278). `shares` are increasing
    shares of domestic deposits (the named parameter `uninsured_share_grid`); `survival_days` is the survival period
    under
    each share in relevant days, `None` surviving the horizon.

    statistic = the largest increase of the survival period along the grid, in days. Light (a Contraste internal gate):
    green when no step lengthens the survival period by more than `tol_days`, red otherwise; fewer than two shares:
    not_evaluated; `extras["thresholds"] = {"tolerance_days": tol_days}`. Golden: `[0.5, 0.9]` with `[None, 67]` is
    green
    with statistic 0; `[40, 67]` is red with statistic 27."""
    test_id = "liq.uninsured_share_sensitivity"
    h0 = "the survival period does not lengthen as the uninsured share rises"
    alt = "a higher uninsured share survives longer than a lower one"
    tol = _nonneg_tol(tol_days, "tol_days")
    grid = check_increasing_grid(check_unit_share(shares, "shares"), "shares")
    days = _days(survival_days, "survival_days")
    if days.size != grid.size:
        raise ValueError(f"shares and survival_days: must have the same length, got {grid.size} and {days.size}")
    thresholds = {"tolerance_days": tol}
    version = _version(tolerance_days=tol)
    if days.size < 2:
        return _not_evaluated(test_id, h0, alt, REF_UNINSURED, version, "fewer than 2 shares", n=int(days.size),
                              extras={"thresholds": thresholds})
    stat = _largest_increase(days)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_UNINSURED, policy_version=version,
        light=Light.GREEN if stat <= tol + EPS else Light.RED, statistic=stat, n=int(days.size),
        inputs_hash=inputs_hash(grid, np.where(np.isinf(days), -1.0, days)),
        extras={"thresholds": thresholds, "survives_horizon": [bool(math.isinf(d)) for d in days]},
        notes="a finite survival period is followed by 'survives the horizon': an unbounded lengthening"
        if math.isinf(stat) else "")


@register(
    "liq.reverse_stress_round_trip",
    family="liquidity",
    name="Reverse stress round trip",
    h0="the survival period at the solved multiplier equals the target",
    reference=REF_REVERSE,
    decision="metric",
)
def reverse_stress_round_trip(solved_multiplier: float | None, survival_at_multiplier: int | None, target_days: int,
                              tol_days: float = 0.0, *,
                              survival_just_below: SurvivalResult | float | None = None) -> TestResult:
    """The reverse stress round trip (dossier 05 line 279): the solved multiplier is the threshold at which the survival
    period reaches the target.

    Without `survival_just_below` (the 1.1.0 call): statistic = `abs(survival_at_multiplier - target_days)`, metric =
    the multiplier, green when the statistic is at most `tol_days` (the named parameter `round_trip_tolerance_days`),
    red otherwise. With `survival_just_below` (the survival period at `m* - 1e-6`: a `SurvivalResult`, a number of days,
    or `math.inf` for "survives the horizon"), the threshold property is tested: statistic = the larger of the shortfall
    `max(survival_at_multiplier - target_days, 0)` (the multiplier fails to reach the target) and the overshoot
    `max(target_days + 1 - survival_just_below, 0)` (the multiplier is not the smallest one, so the period just below it
    is already at or under the target); green when the statistic is at most `tol_days`, red otherwise. Overshoot is
    measured against `target_days + 1`, the shortest period that does not reach the target. When the curve stepped
    (`survival_at_multiplier` below the target, because the CBC grows over time) the notes give the gap
    `target - SP(m*)` and the light stays green. Not_evaluated with the reason when the multiplier is `None` (a
    refused solve); `extras["thresholds"] = {"tolerance_days": tol_days}`. Golden: `(2.085164, 5, 5)` gives 0 and
    green; `(2.0, 7, 5)` gives 2 and red; `(None, None, 30)` gives not_evaluated;
    `(1.5, 1, 5, survival_just_below=math.inf)` is green with the gap 4 in the notes."""
    test_id = "liq.reverse_stress_round_trip"
    h0 = "the survival period at the solved multiplier equals the target"
    alt = "the survival period at the solved multiplier differs from the target"
    tol = _nonneg_tol(tol_days, "tol_days")
    target = int(_scalar(target_days, "target_days"))
    thresholds = {"tolerance_days": tol}
    version = _version(tolerance_days=tol)
    if solved_multiplier is None or survival_at_multiplier is None:
        return _not_evaluated(test_id, h0, alt, REF_REVERSE, version,
                              "the solve was refused, so there is no multiplier to round trip", n=1,
                              extras={"thresholds": thresholds, "target_days": target})
    mult = _scalar(solved_multiplier, "solved_multiplier")
    surv = _scalar(survival_at_multiplier, "survival_at_multiplier")
    notes = ""
    if survival_just_below is None:
        stat = abs(surv - target)
        below_hash = -2.0
    else:
        h0 = "the solved multiplier is the threshold: the target is reached at it and not just below it"
        alt = "the solved multiplier does not reach the target, or a smaller multiplier already does"
        if isinstance(survival_just_below, SurvivalResult):
            below = math.inf if survival_just_below.survives else float(survival_just_below.day)
        elif isinstance(survival_just_below, float) and math.isinf(survival_just_below):
            below = math.inf
        else:
            below = _scalar(survival_just_below, "survival_just_below")
        stat = max(surv - target, 0.0, target + 1 - below)
        below_hash = -1.0 if math.isinf(below) else below
        if surv < target:
            notes = (f"the survival period steps past the target: it is {int(surv)} days at the threshold multiplier, "
                     f"a gap of {int(target - surv)} days to the target of {target} days")
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_REVERSE, policy_version=version,
        light=Light.GREEN if stat <= tol + EPS else Light.RED, statistic=stat, metric=mult, n=1,
        inputs_hash=inputs_hash(np.array([mult, surv, target, below_hash])),
        extras={"thresholds": thresholds, "target_days": target}, notes=notes)


@register(
    "liq.htm_revaluation_vs_reported",
    family="liquidity",
    name="HTM revaluation vs reported",
    h0=(
        "the engine's revaluation of the HTM book reproduces the reported fair value within the stated share "
        "of the loss"),
    reference=REF_REVALUATION,
    decision="metric",
)
def htm_revaluation_vs_reported(model_fair_value: float | None, reported_fair_value: float, amortised_cost: float,
                                green: float = 0.05, amber: float = 0.15) -> TestResult:
    """The HTM book revalued from its maturity profile under the Treasury curve through the engine's `irrbb` curves
    (decision Q11) against the reported fair value `SCHF`. Money in USD thousands.

    statistic = `abs(model - reported) / abs(reported - amortised)`, the share of the reported unrealised loss. Light (a
    Contraste internal gate): green when the statistic is at most `green`, amber when at most `amber`, red above (the
    named parameters `htm_revaluation_green` and `htm_revaluation_amber`); not_evaluated with the reason when
    `model_fair_value` is `None` or the reported unrealised loss is zero; `extras["thresholds"]` holds both. Golden:
    `reported = 80`, `amortised = 100`: `model = 81` gives 0.05 and green; 82 gives 0.10 and amber; 85 gives 0.25
    and red."""
    test_id = "liq.htm_revaluation_vs_reported"
    h0 = ("the engine's revaluation of the HTM book reproduces the reported fair value within the stated share "
          "of the loss")
    alt = "the revaluation departs from the reported fair value by more than the stated share of the reported loss"
    g, a = _scalar(green, "green"), _scalar(amber, "amber")
    if not 0.0 <= g <= a:
        raise ValueError(f"green and amber: must satisfy 0 <= green <= amber, got {g} and {a}")
    reported = _scalar(reported_fair_value, "reported_fair_value")
    cost = _scalar(amortised_cost, "amortised_cost")
    thresholds = {"green": g, "amber": a}
    version = _version(green=g, amber=a)
    if model_fair_value is None:
        return _not_evaluated(test_id, h0, alt, REF_REVALUATION, version,
                              "model_fair_value is None: no revaluation was run", n=1,
                              extras={"thresholds": thresholds})
    model = _scalar(model_fair_value, "model_fair_value")
    loss = abs(reported - cost)
    if loss == 0.0:
        return _not_evaluated(test_id, h0, alt, REF_REVALUATION, version,
                              "the reported unrealised loss is zero, so its share is undefined", n=1,
                              extras={"thresholds": thresholds})
    stat = abs(model - reported) / loss
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_REVALUATION, policy_version=version,
        light=_three_light_low(stat, g, a), statistic=stat, metric=model - reported, n=1,
        inputs_hash=inputs_hash(np.array([model, reported, cost])), extras={"thresholds": thresholds})


__all__ = ["POLICY_LABEL", "cbc_haircut_rules", "cliff_effect_diagnostic", "htm_loss_effect",
           "htm_revaluation_vs_reported", "list_compounding_golden", "nlp_identity", "reverse_stress_round_trip",
           "runoff_adequacy_back_test", "runoff_vs_generator_truth", "survival_monotone_in_severity",
           "uninsured_share_sensitivity"]
