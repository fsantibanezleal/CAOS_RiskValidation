"""The aggregation validation tests (dossier 05 G.5): each returns one `TestResult` that validates against
`common/test-result`.

Acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds. Each function states its light
rule in its docstring and records its parameters in `policy_version` and in `extras["thresholds"]`, so that a view can
explain a light (`alpha_amber` and `alpha_red` are null: these tests are not hypothesis tests). A test that cannot be
evaluated returns the light `not_evaluated` with its reason in `notes` and a null statistic, metric and p-value once
serialised, never a fabricated number; a non-finite input raises `ValueError` naming the field. The tests are plain
functions that do not read the registry or the default policy: their rules are the contract's (`engine-aggregation-api`,
module validation.aggregation). They state no compliance claim. The Euler additivity test of the band is the reused
`riskvalidation.validation.portfolio.euler_additivity`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from riskvalidation.aggregation import simple as _simple
from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.lgd_ead._checks import scalar, vec
from riskvalidation.validation._policy import battery_label

POLICY_LABEL = "contraste-aggregation-battery 1.0.0"
NAN = float("nan")
EPS = 1e-12             # tolerance on a gate comparison, so that a value exactly on a gate is not moved by rounding

G5 = ("dossier 05 G.5 (validation tests of an economic capital model: benchmark the aggregation against the simple sum "
      "and the variance-covariance formula, the copula-choice sensitivity, worst-case bounds, the correlation "
      "stress, the sign of every diversification benefit, one horizon and confidence level)")
G1 = "dossier 05 G.1 (the aggregation ladder: simple sum, variance-covariance, copulas)"
G3 = ("dossier 05 G.3 (VaR is not subadditive; worst-case VaR bounds by the rearrangement algorithm of Embrechts, "
      "Puccetti and Ruschendorf 2013, doi:10.1016/j.jbankfin.2013.03.014)")
G2 = ("Rosenberg and Schuermann (2006), A general approach to integrated risk management with skewed, "
      "fat-tailed risks, J. Financial Economics 79(3):569-614, doi:10.1016/j.jfineco.2005.03.001 (dossier 05 G.2)")


def _not_evaluated(test_id: str, h0: str, alternative: str, reference: str, version: str, reason: str, *,
                   thresholds: Mapping[str, Any], model_id: str | None = None, segment: str | None = None,
                   n: int | None = None, extras: Mapping[str, Any] | None = None) -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(
        test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=version,
        light=Light.NOT_EVALUATED, statistic=NAN, notes=f"not evaluated: {reason}", model_id=model_id, segment=segment,
        n=n, extras={**(extras or {}), "thresholds": dict(thresholds)})


def _gate(value: float, name: str) -> float:
    v = scalar(value, name)
    if v < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return v


def _three_light(value: float, green: float, amber: float) -> Light:
    """Green at or below `green`, amber at or below `amber`, red above (with a rounding tolerance)."""
    if value <= green + EPS:
        return Light.GREEN
    return Light.AMBER if value <= amber + EPS else Light.RED


@register(
    "aggregation.simple_sum_and_variance_covariance",
    family="aggregation",
    name="Simple sum and variance covariance",
    h0="the aggregate is not above the simple sum and agrees with the variance-covariance formula",
    reference=f"{G5}; {G1}",
    decision="metric",
)
def simple_sum_and_variance_covariance(ec: np.ndarray, corr: np.ndarray, aggregate: float, amber: float = 0.10,
                                       tolerance: float = 1e-9, model_id: str | None = None,
                                       segment: str | None = None) -> TestResult:
    """The aggregate against the simple sum and the variance-covariance formula (dossier 05 G.5).

    h0: the aggregate is not above the simple sum and agrees with the variance-covariance formula. statistic =
    aggregate /
    variance_covariance(ec, corr), metric = aggregate / simple_sum(ec), p_value null; extras carry `simple_sum`,
    `variance_covariance` and thresholds = {"amber": amber, "tolerance": tolerance}. Light (a Contraste internal gate):
    red when aggregate > simple_sum (1 + tolerance) (a diversified figure above the sum), amber when |statistic - 1| >
    amber, green otherwise. not_evaluated when the simple sum or the variance-covariance anchor is zero.
    """
    test_id = "aggregation.simple_sum_and_variance_covariance"
    h0 = "the aggregate is not above the simple sum and agrees with the variance-covariance formula"
    alt = "the aggregate is above the simple sum or departs from the variance-covariance formula"
    reference = f"{G5}; {G1}"
    agg = scalar(aggregate, "aggregate")
    am, tol = _gate(amber, "amber"), _gate(tolerance, "tolerance")
    capital = vec(ec, "ec")
    ss = _simple.simple_sum(capital)
    vc = _simple.variance_covariance(capital, corr)
    thresholds = {"amber": am, "tolerance": tol}
    version = battery_label(POLICY_LABEL, "aggregation.simple_sum_and_variance_covariance", amber=am, tolerance=tol)
    extras = {"simple_sum": ss, "variance_covariance": vc}
    kw = dict(model_id=model_id, segment=segment, n=int(capital.size))
    if not (ss > 0.0 and vc > 0.0):
        return _not_evaluated(test_id, h0, alt, reference, version, "the simple sum or the "
                              "variance-covariance anchor is zero: no ratio", thresholds=thresholds,
                              extras=extras, **kw)
    statistic = agg / vc
    if agg > ss * (1.0 + tol):
        light = Light.RED
    elif abs(statistic - 1.0) > am + EPS:
        light = Light.AMBER
    else:
        light = Light.GREEN
    return TestResult(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version, light=light,
                      statistic=statistic, metric=agg / ss, inputs_hash=inputs_hash(capital, np.asarray(corr), agg),
                      extras={**extras, "thresholds": thresholds}, **kw)


@register(
    "aggregation.copula_choice_sensitivity",
    family="aggregation",
    name="Copula choice sensitivity",
    h0=(
        "the choice between the Gaussian and the t copula over the stated nu set moves the capital by less than the "
        "thresholds"),
    reference=f"{G5}; {G2}",
    decision="metric",
)
def copula_choice_sensitivity(gaussian: float, t_capitals: Sequence[tuple[float, float]], amber: float = 0.05,
                              red: float = 0.15, model_id: str | None = None,
                              segment: str | None = None) -> TestResult:
    """The sensitivity of the capital to the choice between the Gaussian and the t copula (dossier 05 G.5; Rosenberg and
    Schuermann find the choice has "a more modest effect on risk", G.2).

    h0: the choice over the stated nu set moves the capital by less than the thresholds. `t_capitals` are pairs (nu,
    capital). statistic = max over nu of |t / gaussian - 1|, metric = the signed value at the smallest nu, p_value null;
    extras carry the per-nu differences (`differences`, keyed by nu) and thresholds = {"amber": amber, "red": red}.
    Light (a
    Contraste internal gate): green when statistic <= amber, amber when <= red, red above. not_evaluated for no pair
    or a
    nonpositive Gaussian capital.
    """
    test_id = "aggregation.copula_choice_sensitivity"
    h0 = ("the choice between the Gaussian and the t copula over the stated nu set moves the capital by less than the "
          "thresholds")
    alt = "the copula choice moves the capital by more than the thresholds"
    reference = f"{G5}; {G2}"
    g = scalar(gaussian, "gaussian")
    am, rd = _gate(amber, "amber"), _gate(red, "red")
    if am > rd:
        raise ValueError(f"amber and red must satisfy 0 <= amber <= red, got {amber} and {red}")
    thresholds = {"amber": am, "red": rd}
    version = battery_label(POLICY_LABEL, "aggregation.copula_choice_sensitivity", amber=am, red=rd)
    kw = dict(model_id=model_id, segment=segment)
    pairs = np.array(t_capitals, dtype=float).reshape(-1, 2) if len(t_capitals) else np.empty((0, 2))
    if not np.all(np.isfinite(pairs)):
        raise ValueError("t_capitals contains non-finite values")
    if pairs.shape[0] == 0:
        return _not_evaluated(test_id, h0, alt, reference, version, "t_capitals holds no (nu, capital) pair",
                              thresholds=thresholds, n=0, **kw)
    if not g > 0.0:
        return _not_evaluated(test_id, h0, alt, reference, version, "the Gaussian capital is not positive: no ratio",
                              thresholds=thresholds, n=int(pairs.shape[0]), **kw)
    diff = pairs[:, 1] / g - 1.0
    statistic = float(np.max(np.abs(diff)))
    first = int(np.argmin(pairs[:, 0]))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=_three_light(statistic, am, rd), statistic=statistic, metric=float(diff[first]), n=int(pairs.shape[0]),
        inputs_hash=inputs_hash(pairs, g),
        extras={"differences": {f"{nu:g}": float(d) for nu, d in zip(pairs[:, 0], diff, strict=True)},
                "thresholds": thresholds}, **kw)


@register(
    "aggregation.worst_case_bracket",
    family="aggregation",
    name="Worst case bracket",
    h0="every row of the band lies between the lower bound and the worst case",
    reference=f"{G5}; {G3}",
    decision="metric",
)
def worst_case_bracket(lower: float, rows: Sequence[float], worst: float, tolerance: float = 1e-9,
                       model_id: str | None = None, segment: str | None = None) -> TestResult:
    """Every row of the band lies between the lower bound and the worst case (dossier 05 G.3 and G.5: report worst-case
    bounds).

    statistic = worst - lower, metric = (worst - lower) / max(rows) (null when no row is positive), p_value null; extras
    carry thresholds = {"tolerance": tolerance}. Light (a Contraste internal gate, no amber: a bracket holds or it does
    not): red when any row is outside [lower - tolerance, worst + tolerance] or lower > worst, green otherwise.
    not_evaluated for no row.
    """
    test_id = "aggregation.worst_case_bracket"
    h0 = "every row of the band lies between the lower bound and the worst case"
    alt = "a row of the band lies outside the bracket of the lower bound and the worst case"
    reference = f"{G5}; {G3}"
    lo, hi = scalar(lower, "lower"), scalar(worst, "worst")
    tol = _gate(tolerance, "tolerance")
    r = vec(rows, "rows", allow_empty=True)
    thresholds = {"tolerance": tol}
    version = battery_label(POLICY_LABEL, "aggregation.worst_case_bracket", tolerance=tol)
    kw = dict(model_id=model_id, segment=segment)
    if r.size == 0:
        return _not_evaluated(test_id, h0, alt, reference, version, "rows holds no row", thresholds=thresholds, n=0,
        **kw)
    outside = int(np.sum((r < lo - tol) | (r > hi + tol)))
    red = lo > hi or outside > 0
    top = float(r.max())
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=Light.RED if red else Light.GREEN, statistic=hi - lo, metric=(hi - lo) / top if top > 0.0 else None,
        n=int(r.size), inputs_hash=inputs_hash(r, lo, hi),
        extras={"rows_outside": outside, "thresholds": thresholds},
        notes="" if not red else "a row lies outside the bracket, or the lower bound is above the worst case", **kw)


@register(
    "aggregation.correlation_stress",
    family="aggregation",
    name="Correlation stress",
    h0="raising the market-credit correlation does not lower the capital and raises it by less than the amber gate",
    reference=G5,
    decision="metric",
)
def correlation_stress(base: float, stressed: float, amber: float = 0.10, tolerance: float = 1e-9,
                       model_id: str | None = None, segment: str | None = None) -> TestResult:
    """Raising the market-credit correlation to 0.8 (dossier 05 G.5) does not lower the capital and raises it by less
    than
    `amber`.

    statistic = stressed / base - 1, metric = statistic, p_value null; extras carry thresholds = {"amber": amber,
    "tolerance": tolerance}. Light (a Contraste internal gate): red when statistic < -tolerance (the capital falls
    when a
    correlation rises), amber when statistic > amber, green otherwise. not_evaluated for a nonpositive base capital.
    """
    test_id = "aggregation.correlation_stress"
    h0 = "raising the market-credit correlation does not lower the capital and raises it by less than the amber gate"
    alt = "the capital falls when the correlation rises, or rises by more than the amber gate"
    reference = G5
    b, s = scalar(base, "base"), scalar(stressed, "stressed")
    am, tol = _gate(amber, "amber"), _gate(tolerance, "tolerance")
    thresholds = {"amber": am, "tolerance": tol}
    version = battery_label(POLICY_LABEL, "aggregation.correlation_stress", amber=am, tolerance=tol)
    kw = dict(model_id=model_id, segment=segment)
    if not b > 0.0:
        return _not_evaluated(test_id, h0, alt, reference, version, "the base capital is not positive: "
                              "no relative change", thresholds=thresholds, **kw)
    statistic = s / b - 1.0
    if statistic < -tol:
        light = Light.RED
    elif statistic > am + EPS:
        light = Light.AMBER
    else:
        light = Light.GREEN
    return TestResult(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version, light=light,
                      statistic=statistic, metric=statistic, inputs_hash=inputs_hash(np.array([b, s])),
                      extras={"thresholds": thresholds}, **kw)


@register(
    "aggregation.diversification_sign",
    family="aggregation",
    name="Diversification sign",
    h0="every diversification benefit has the right sign (DF <= 1)",
    reference=f"{G5}; {G3}",
    decision="metric",
)
def diversification_sign(factors: Sequence[float], tolerance: float = 1e-9, model_id: str | None = None,
                         segment: str | None = None) -> TestResult:
    """Every diversification benefit has the right sign, DF <= 1 (dossier 05 G.5: "the sign of every diversification
    benefit"). `factors` are the DF of each diversified row.

    statistic = max(factors), metric = 1 - min(factors), p_value null; extras carry `count_above_one` (the rows above
    1 +
    tolerance) and thresholds = {"tolerance": tolerance}. Light (a Contraste internal gate): red when max(factors) > 1 +
    tolerance, green otherwise. not_evaluated for no factor.
    """
    test_id = "aggregation.diversification_sign"
    h0 = "every diversification benefit has the right sign (DF <= 1)"
    alt = "a diversified row is above the simple sum (DF > 1)"
    reference = f"{G5}; {G3}"
    f = vec(factors, "factors", allow_empty=True)
    tol = _gate(tolerance, "tolerance")
    thresholds = {"tolerance": tol}
    version = battery_label(POLICY_LABEL, "aggregation.diversification_sign", tolerance=tol)
    kw = dict(model_id=model_id, segment=segment)
    if f.size == 0:
        return _not_evaluated(test_id, h0, alt, reference, version, "factors holds no row", thresholds=thresholds,
        n=0, **kw)
    top = float(f.max())
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=Light.RED if top > 1.0 + tol else Light.GREEN, statistic=top, metric=1.0 - float(f.min()), n=int(f.size),
        inputs_hash=inputs_hash(f), extras={"count_above_one": int(np.sum(f > 1.0 + tol)), "thresholds": thresholds},
        **kw)


@register(
    "aggregation.horizon_confidence_consistency",
    family="aggregation",
    name="Horizon confidence consistency",
    h0="one risk horizon and one confidence level apply across the marginals",
    reference=f"{G5}; dossier 01 l.837-839 (one horizon and one confidence level for the aggregation)",
    decision="metric",
)
def horizon_confidence_consistency(horizons: Sequence[float], levels: Sequence[float], expected_horizon: float = 1.0,
                                   expected_level: float = 0.999, model_id: str | None = None,
                                   segment: str | None = None) -> TestResult:
    """One risk horizon and one confidence level across the marginals (dossier 05 G.5; dossier 01 l.837-839).

    statistic = the number of marginals that differ from the expected horizon or level, metric null, p_value null;
    extras
    carry `offending_indices` and thresholds = {"expected_horizon": ..., "expected_level": ...}. Light (a Contraste
    internal
    gate): red when statistic > 0, green otherwise. not_evaluated for no marginal. Unequal lengths raise.
    """
    test_id = "aggregation.horizon_confidence_consistency"
    h0 = "one risk horizon and one confidence level apply across the marginals"
    alt = "a marginal is at another horizon or confidence level"
    reference = f"{G5}; dossier 01 l.837-839 (one horizon and one confidence level for the aggregation)"
    h = vec(horizons, "horizons", allow_empty=True)
    lv = vec(levels, "levels", allow_empty=True)
    if h.size != lv.size:
        raise ValueError(f"horizons and levels must have the same length, got {h.size} and {lv.size}")
    eh, el = scalar(expected_horizon, "expected_horizon"), scalar(expected_level, "expected_level")
    thresholds = {"expected_horizon": eh, "expected_level": el}
    version = battery_label(
            POLICY_LABEL, "aggregation.horizon_confidence_consistency", expected_horizon=eh, expected_level=el)
    kw = dict(model_id=model_id, segment=segment)
    if h.size == 0:
        return _not_evaluated(test_id, h0, alt, reference, version, "no marginal to compare", thresholds=thresholds,
        n=0, **kw)
    bad = np.flatnonzero((np.abs(h - eh) > EPS) | (np.abs(lv - el) > EPS))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=Light.RED if bad.size else Light.GREEN, statistic=float(bad.size), n=int(h.size),
        inputs_hash=inputs_hash(h, lv), extras={"offending_indices": [int(i) for i in bad], "thresholds": thresholds},
        **kw)


@register(
    "aggregation.elliptical_mc_agreement",
    family="aggregation",
    name="Elliptical MC agreement",
    h0=(
        "the Monte Carlo capital agrees with the variance-covariance anchor within z standard errors in the "
        "elliptical case"),
    reference=f"{G5}; {G1}",
    decision="metric",
)
def elliptical_mc_agreement(mc_capital: float, mc_se: float, vc_capital: float, z: float = 3.0,
                            model_id: str | None = None, segment: str | None = None) -> TestResult:
    """In the elliptical case (normal marginals, Gaussian copula) the Monte Carlo capital agrees with the
    variance-covariance anchor within `z` standard errors (exact there for a common quantile scaling: dossier 05 G.1).

    statistic = (mc_capital - vc_capital) / mc_se, metric = mc_capital / vc_capital - 1, p_value null; extras carry
    thresholds = {"z": z}. Light (a Contraste internal gate): green when |statistic| <= z, red above. not_evaluated when
    mc_se or vc_capital is not positive.
    """
    test_id = "aggregation.elliptical_mc_agreement"
    h0 = ("the Monte Carlo capital agrees with the variance-covariance anchor within z standard errors in the "
          "elliptical case")
    alt = "the Monte Carlo capital departs from the variance-covariance anchor by more than z standard errors"
    reference = f"{G5}; {G1}"
    mc, se, vc = scalar(mc_capital, "mc_capital"), scalar(mc_se, "mc_se"), scalar(vc_capital, "vc_capital")
    zz = scalar(z, "z")
    if not zz > 0.0:
        raise ValueError(f"z must be positive, got {z}")
    thresholds = {"z": zz}
    version = battery_label(POLICY_LABEL, "aggregation.elliptical_mc_agreement", z=zz)
    kw = dict(model_id=model_id, segment=segment)
    if not se > 0.0:
        return _not_evaluated(test_id, h0, alt, reference, version, "the Monte Carlo standard error is not positive",
                              thresholds=thresholds, **kw)
    if not vc > 0.0:
        return _not_evaluated(test_id, h0, alt, reference, version, "the variance-covariance capital is not positive",
                              thresholds=thresholds, **kw)
    statistic = (mc - vc) / se
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=Light.GREEN if abs(statistic) <= zz + EPS else Light.RED, statistic=statistic, metric=mc / vc - 1.0,
        inputs_hash=inputs_hash(np.array([mc, se, vc])), extras={"thresholds": thresholds}, **kw)


@register(
    "aggregation.rosenberg_schuermann_reference",
    family="aggregation",
    name="Rosenberg schuermann reference",
    h0="none to reject: the additive approximation typically overestimates risk by about 30 to 40 percent",
    reference=f"{G2}; the figure is a reference beside the result, never a gate",
    decision="metric",
)
def rosenberg_schuermann_reference(simple_sum: float, gaussian: float, low: float = 0.30, high: float = 0.40,
                                   model_id: str | None = None, segment: str | None = None) -> TestResult:
    """The additive approximation overestimates risk by about 30 to 40 percent (Rosenberg and Schuermann 2006; dossier
    05
    G.2), reported beside the result as a reference, never a gate.

    h0: none to reject. statistic = simple_sum / gaussian - 1, metric = statistic, p_value null; the light is always
    green
    (the row exists to be shown) and `notes` states whether the figure lies inside or outside [low, high]; extras carry
    `role = "reference-not-a-gate"` and thresholds = {"low": low, "high": high}. not_evaluated for a nonpositive
    Gaussian
    capital. The published range itself is UNVERIFIED in its original (dossier 05 G.2).
    """
    test_id = "aggregation.rosenberg_schuermann_reference"
    h0 = "none to reject: the additive approximation typically overestimates risk by about 30 to 40 percent"
    alt = "a reference, not a hypothesis test"
    reference = f"{G2}; the figure is a reference beside the result, never a gate"
    ss, g = scalar(simple_sum, "simple_sum"), scalar(gaussian, "gaussian")
    lo, hi = scalar(low, "low"), scalar(high, "high")
    if not 0.0 <= lo <= hi:
        raise ValueError(f"low and high must satisfy 0 <= low <= high, got {low} and {high}")
    thresholds = {"low": lo, "high": hi}
    version = battery_label(POLICY_LABEL, "aggregation.rosenberg_schuermann_reference", low=lo, high=hi)
    kw = dict(model_id=model_id, segment=segment)
    role = {"role": "reference-not-a-gate"}
    if not g > 0.0:
        return _not_evaluated(test_id, h0, alt, reference, version, "the Gaussian capital is not positive: no ratio",
                              thresholds=thresholds, extras=role, **kw)
    statistic = ss / g - 1.0
    where = "inside" if lo - EPS <= statistic <= hi + EPS else "outside"
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version, light=Light.GREEN,
        statistic=statistic, metric=statistic, inputs_hash=inputs_hash(np.array([ss, g])),
        extras={**role, "thresholds": thresholds},
        notes=f"the additive overestimate {statistic:.4f} lies {where} the published range [{lo}, {hi}]", **kw)

__all__ = [
    "copula_choice_sensitivity",
    "correlation_stress",
    "diversification_sign",
    "elliptical_mc_agreement",
    "horizon_confidence_consistency",
    "rosenberg_schuermann_reference",
    "simple_sum_and_variance_covariance",
    "worst_case_bracket",
]
