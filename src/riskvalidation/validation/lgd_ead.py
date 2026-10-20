"""The LGD and EAD validation tests (dossier 04 F.3 and G.3): each returns one `TestResult`.

Acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds. Each function states its light
rule in its docstring and records its parameters in `policy_version`, in `alpha_amber` and `alpha_red` where a
p-value is
used, and in `extras["thresholds"]` for the other parameters (`rho_green`, `rho_amber`, `gap_amber`, `gap_red`,
`tolerance`), so that a view can explain a light. A test that cannot be evaluated returns the light `not_evaluated` with
its reason in `notes` and a null statistic, p-value and metric once serialised, never a fabricated number. LGD
calibration
is judged by mean realised against mean predicted by segment and over time, not by R2 alone.

The tests are plain functions that do not read the registry or the default policy: their rules are the contract's
(`engine-lgd-ead-api`, module validation.lgd_ead). They state no compliance claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.lgd_ead._checks import same_length, scalar, vec

POLICY_LABEL = "contraste-lgd-ead-battery-1.0.0"
NAN = float("nan")
ZERO_SPREAD = 1e-12     # a spread or a mean below this (relative to max(1, the largest value)) is zero
EPS = 1e-12             # tolerance on a gate comparison, so that a value exactly on a gate is not moved by rounding

LOTERMAN_2012 = (
    "Loterman, Brown, Martens, Mues and Baesens (2012), Benchmarking regression algorithms for loss given default "
    "modeling, Int. J. Forecasting 28(1):161-170, doi:10.1016/j.ijforecast.2011.01.006"
)
QI_ZHAO_2011 = (
    "Qi and Zhao (2011), Comparison of modeling methods for Loss Given Default, J. Banking and Finance "
    "35(11):2842-2855, doi:10.1016/j.jbankfin.2011.03.011"
)
TONG_2016 = (
    "Tong, Mues, Brown and Thomas (2016), Exposure at default models with and without the credit conversion factor, "
    "EJOR 252(3):910-920, doi:10.1016/j.ejor.2016.01.054"
)
QI_2009 = (
    "Qi (2009), Exposure at default of unsecured credit cards, OCC Economics Working Paper 2009-2, "
    "https://www.occ.gov/publications-and-resources/publications/economics/working-papers-archived/"
    "pub-econ-working-paper-2009-2.pdf"
)
EBA_DOWNTURN = (
    "EBA/GL/2019/03 and EBA/RTS/2018/04 are cross-references only; the add-on is Contraste's cohort-based one "
    "(dossier 04 F.4)"
)


def _version(**params: Any) -> str:
    return "; ".join([POLICY_LABEL, *(f"{k}={v}" for k, v in params.items())])


def _not_evaluated(test_id: str, h0: str, alternative: str, reference: str, version: str, reason: str, *,
                   model_id: str | None = None, segment: str | None = None, n: int | None = None,
                   alpha_amber: float | None = None, alpha_red: float | None = None,
                   extras: Mapping[str, Any] | None = None) -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(
        test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=version,
        light=Light.NOT_EVALUATED, statistic=NAN, notes=f"not evaluated: {reason}", model_id=model_id,
        segment=segment, n=n, alpha_amber=alpha_amber, alpha_red=alpha_red, extras=dict(extras or {}),
    )


def _check_alphas(alpha_amber: float, alpha_red: float) -> None:
    if not (0.0 < alpha_red <= alpha_amber < 1.0):
        raise ValueError(f"alpha_red and alpha_amber must satisfy 0 < alpha_red <= alpha_amber < 1, got "
                         f"{alpha_red} and {alpha_amber}")


def _p_light(p: float, alpha_amber: float, alpha_red: float) -> Light:
    if p >= alpha_amber:
        return Light.GREEN
    return Light.AMBER if p >= alpha_red else Light.RED


def _scale(d: np.ndarray) -> float:
    return ZERO_SPREAD * max(1.0, float(np.max(np.abs(d))))


def _no_spread(d: np.ndarray) -> bool:
    return float(np.std(d, ddof=1)) <= _scale(d)


def _zero_mean(d: np.ndarray) -> bool:
    return abs(float(d.mean())) <= _scale(d)


def _paired_t(d: np.ndarray) -> tuple[float, float, float, float]:
    """(t, two-sided p, mean, standard error) of a sample of gaps with at least 2 values and a nonzero spread."""
    n = d.size
    mean = float(d.mean())
    se = float(np.std(d, ddof=1)) / math.sqrt(n)
    t = mean / se
    return t, float(2.0 * stats.t.sf(abs(t), n - 1)), mean, se


# ----------------------------------------------------------------------------------------------------------------
# LGD calibration and discrimination


@register(
    "lgd_ead.lgd_calibration_by_segment",
    family="lgd_ead",
    name="LGD calibration by segment",
    h0="the mean realised LGD equals the mean predicted LGD in the segment",
    reference=f"{LOTERMAN_2012}; {QI_ZHAO_2011}; dossier 04 F.3 (calibration by segment, not R2 alone)",
    decision="p_value",
)
def lgd_calibration_by_segment(realised: np.ndarray, predicted: np.ndarray, segment: str | None = None,
                               model_id: str | None = None, alpha_amber: float = 0.05,
                               alpha_red: float = 0.01) -> TestResult:
    """The mean realised LGD against the mean predicted LGD of one segment: a paired t of d = realised - predicted.

    statistic = mean(d) / (sd(d) / sqrt(N)) with N - 1 degrees of freedom (sd with ddof = 1), p_value two-sided,
    metric = mean(d). extras: mean_realised, mean_predicted, ci_low and ci_high (95% interval of the mean gap), and the
    counts of realised values below 0 and above 1 (kept, not clipped). N < 2: not_evaluated. No spread in d: red with a
    null statistic and p_value and the gap in the metric when mean(d) is not zero (a certain gap), green when it is
    zero.
    Light (a Contraste internal gate): green when p_value >= `alpha_amber`, amber when `alpha_red` <= p_value <
    `alpha_amber`, red below `alpha_red`.
    """
    test_id = "lgd_ead.lgd_calibration_by_segment"
    reference = f"{LOTERMAN_2012}; {QI_ZHAO_2011}; dossier 04 F.3 (calibration by segment, not R2 alone)"
    h0 = "the mean realised LGD equals the mean predicted LGD in the segment"
    alt = "the mean realised LGD differs from the mean predicted LGD"
    _check_alphas(alpha_amber, alpha_red)
    real, pred = vec(realised, "realised", allow_empty=True), vec(predicted, "predicted", allow_empty=True)
    same_length(realised=real, predicted=pred)
    version = _version(alpha_amber=alpha_amber, alpha_red=alpha_red)
    n = int(real.size)
    kw = dict(model_id=model_id, segment=segment, n=n, alpha_amber=alpha_amber, alpha_red=alpha_red)
    if n < 2:
        return _not_evaluated(test_id, h0, alt, reference, version, "fewer than 2 observations", **kw)
    d = real - pred
    mean = float(d.mean())
    extras: dict[str, Any] = {
        "mean_realised": float(real.mean()), "mean_predicted": float(pred.mean()),
        "n_below_zero": int((real < 0.0).sum()), "n_above_one": int((real > 1.0).sum()),
    }
    common = dict(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
                  inputs_hash=inputs_hash(real, pred), **kw)
    if _no_spread(d):
        extras.update(ci_low=mean, ci_high=mean, note="no spread in the gaps: the gap is certain")
        zero = _zero_mean(d)
        return TestResult(light=Light.GREEN if zero else Light.RED, metric=0.0 if zero else mean, extras=extras,
                          notes="no spread in the gaps (sd = 0): " + ("no gap" if zero else "a certain gap"), **common)
    t, p, mean, se = _paired_t(d)
    half = float(stats.t.ppf(0.975, n - 1)) * se
    extras.update(ci_low=mean - half, ci_high=mean + half)
    return TestResult(light=_p_light(p, alpha_amber, alpha_red), statistic=t, p_value=p, metric=mean, extras=extras,
                      **common)


@register(
    "lgd_ead.lgd_calibration_over_time",
    family="lgd_ead",
    name="LGD calibration over time",
    h0="in every period the mean realised LGD equals the mean predicted LGD",
    reference=f"{LOTERMAN_2012}; {QI_ZHAO_2011}; dossier 04 F.3 (calibration over time)",
    decision="p_value",
)
def lgd_calibration_over_time(realised: np.ndarray, predicted: np.ndarray, period: np.ndarray,
                              model_id: str | None = None, alpha_amber: float = 0.05,
                              alpha_red: float = 0.01) -> TestResult:
    """The paired t of the segment test per period (a default year or a cohort index) with at least 2 observations and a
    nonzero spread. statistic = the largest |t| over those periods, p_value = min(1, K p_min) (Bonferroni, K the periods
    tested), metric = the largest |mean(d)| over every period with at least one observation (tested or not).
    extras["periods"] lists period, n, mean_gap, t and p of each period (t and p null where not tested) and
    extras["excluded"] each period not tested with its period, n and reason. A period with zero spread and mean(d) not
    zero makes the light red (with a null statistic and p_value when no period was tested); no period tested and no such
    period: not_evaluated. Light (a Contraste internal gate): as `lgd_calibration_by_segment`, on the adjusted p_value.
    """
    test_id = "lgd_ead.lgd_calibration_over_time"
    reference = f"{LOTERMAN_2012}; {QI_ZHAO_2011}; dossier 04 F.3 (calibration over time)"
    h0 = "in every period the mean realised LGD equals the mean predicted LGD"
    alt = "in at least one period the mean realised LGD differs from the mean predicted LGD"
    _check_alphas(alpha_amber, alpha_red)
    real, pred = vec(realised, "realised", allow_empty=True), vec(predicted, "predicted", allow_empty=True)
    per = np.array(period)
    if per.ndim != 1 or (per.size and per.dtype.kind not in "iu"):
        raise ValueError("period must be a one-dimensional integer array")
    same_length(realised=real, predicted=pred, period=per)
    version = _version(alpha_amber=alpha_amber, alpha_red=alpha_red)
    n = int(real.size)
    kw = dict(model_id=model_id, n=n, alpha_amber=alpha_amber, alpha_red=alpha_red)
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    tested: list[dict[str, Any]] = []
    certain_gap = False
    d_all = real - pred
    for key in np.unique(per):
        d = d_all[per == key]
        row: dict[str, Any] = {"period": int(key), "n": int(d.size), "mean_gap": float(d.mean()), "t": None, "p": None}
        if d.size < 2:
            excluded.append({"period": int(key), "n": int(d.size), "reason": "fewer than 2 observations"})
        elif _no_spread(d):
            excluded.append({"period": int(key), "n": int(d.size), "reason": "zero spread"})
            certain_gap = certain_gap or not _zero_mean(d)
        else:
            t, p, mean, _ = _paired_t(d)
            row.update(t=t, p=p)
            tested.append(row)
        rows.append(row)
    extras: dict[str, Any] = {"periods": rows, "excluded": excluded, "n_periods_tested": len(tested)}
    if not tested and not certain_gap:
        return _not_evaluated(test_id, h0, alt, reference, version,
                              "no period with at least 2 observations and a nonzero spread", extras=extras, **kw)
    common = dict(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
                  inputs_hash=inputs_hash(real, pred, per), extras=extras, **kw)
    metric = max(abs(r["mean_gap"]) for r in rows)
    if not tested:
        return TestResult(light=Light.RED, metric=metric,
                          notes="no period with a spread; a period with zero spread has a certain gap", **common)
    p_adj = min(1.0, len(tested) * min(r["p"] for r in tested))
    light = Light.RED if certain_gap else _p_light(p_adj, alpha_amber, alpha_red)
    return TestResult(light=light, statistic=max(abs(r["t"]) for r in tested), p_value=p_adj, metric=metric,
                      notes="a period with zero spread has a certain gap" if certain_gap else "", **common)


@register(
    "lgd_ead.lgd_discrimination",
    family="lgd_ead",
    name="LGD rank discrimination",
    h0="predicted and realised LGD are rank-independent",
    reference=f"{LOTERMAN_2012}; Spearman (1904), doi:10.2307/1412159",
    decision="metric",
)
def lgd_discrimination(realised: np.ndarray, predicted: np.ndarray, segment: str | None = None,
                       model_id: str | None = None, rho_green: float = 0.30, rho_amber: float = 0.10) -> TestResult:
    """The Spearman rank correlation of predicted and realised LGD (ties by average rank).

    statistic = metric = rho in [-1, 1], p_value the two-sided p of the rank correlation. N < 3 or a constant input:
    not_evaluated. Light (a Contraste internal gate, not a regulatory one; LGD rank power is modest in the
    literature, R2
    between 4% and 43% in Loterman et al. 2012): green when rho >= `rho_green`, amber when `rho_amber` <= rho <
    `rho_green`, red below. extras["thresholds"] carries `rho_green` and `rho_amber`.
    """
    test_id = "lgd_ead.lgd_discrimination"
    reference = f"{LOTERMAN_2012}; Spearman (1904), doi:10.2307/1412159"
    h0 = "predicted and realised LGD are rank-independent"
    alt = "predicted and realised LGD are rank-correlated"
    if not rho_amber <= rho_green:
        raise ValueError(f"rho_amber must not exceed rho_green, got {rho_amber} and {rho_green}")
    real, pred = vec(realised, "realised", allow_empty=True), vec(predicted, "predicted", allow_empty=True)
    same_length(realised=real, predicted=pred)
    version = _version(rho_green=rho_green, rho_amber=rho_amber)
    n = int(real.size)
    kw = dict(model_id=model_id, segment=segment, n=n)
    extras = {"thresholds": {"rho_green": float(rho_green), "rho_amber": float(rho_amber)}}
    if n < 3:
        return _not_evaluated(test_id, h0, alt, reference, version, "fewer than 3 observations", extras=extras, **kw)
    if np.ptp(real) == 0.0 or np.ptp(pred) == 0.0:
        return _not_evaluated(test_id, h0, alt, reference, version, "constant input (no rank variation)",
                              extras=extras, **kw)
    res = stats.spearmanr(real, pred)
    rho, p = float(res.statistic), float(res.pvalue)
    if not math.isfinite(rho):
        return _not_evaluated(test_id, h0, alt, reference, version, "the rank correlation is undefined",
                              extras=extras, **kw)
    if rho >= rho_green - EPS:
        light = Light.GREEN
    elif rho >= rho_amber - EPS:
        light = Light.AMBER
    else:
        light = Light.RED
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version, light=light,
        statistic=rho, p_value=min(1.0, max(0.0, p)), metric=rho, inputs_hash=inputs_hash(real, pred), extras=extras,
        **kw,
    )


# ----------------------------------------------------------------------------------------------------------------
# downturn coverage


@register(
    "lgd_ead.downturn_coverage",
    family="lgd_ead",
    name="Downturn coverage",
    h0="the mean realised downturn LGD does not exceed the downturn LGD estimate",
    reference=f"{EBA_DOWNTURN}; {LOTERMAN_2012}",
    decision="p_value",
)
def downturn_coverage(estimate: float, realised_downturn: np.ndarray, segment: str | None = None,
                      model_id: str | None = None, alpha_amber: float = 0.05, alpha_red: float = 0.01) -> TestResult:
    """Does the downturn LGD estimate cover the realised LGD of the defaults in the downturn period?

    h0: the mean realised downturn LGD does not exceed the estimate; alternative: it does. statistic = (mean(realised) -
    estimate) / (sd / sqrt(N)), one-sided upper p_value with N - 1 degrees of freedom, metric = max(0, mean(realised) -
    estimate) (the shortfall of `downturn.coverage_shortfall`). N < 2: not_evaluated. No spread: red with a null
    statistic and p_value and the shortfall in the metric when there is a shortfall, green when there is none.
    Light (a Contraste internal gate): green when p_value >= `alpha_amber`, amber when `alpha_red` <= p_value <
    `alpha_amber`, red below `alpha_red`. The downturn period is the caller's (a supplied macro path); no regulatory
    approach is claimed.
    """
    test_id = "lgd_ead.downturn_coverage"
    reference = f"{EBA_DOWNTURN}; {LOTERMAN_2012}"
    h0 = "the mean realised downturn LGD does not exceed the downturn LGD estimate"
    alt = "the mean realised downturn LGD exceeds the estimate (the estimate does not cover the downturn losses)"
    _check_alphas(alpha_amber, alpha_red)
    est = scalar(estimate, "estimate")
    real = vec(realised_downturn, "realised_downturn", allow_empty=True)
    version = _version(alpha_amber=alpha_amber, alpha_red=alpha_red)
    n = int(real.size)
    kw = dict(model_id=model_id, segment=segment, n=n, alpha_amber=alpha_amber, alpha_red=alpha_red)
    if n < 2:
        return _not_evaluated(test_id, h0, alt, reference, version, "fewer than 2 observations", **kw)
    gap = float(real.mean()) - est
    extras = {"estimate": est, "mean_realised": float(real.mean())}
    common = dict(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
                  inputs_hash=inputs_hash(est, real), extras=extras, **kw)
    if _no_spread(real):
        shortfall = gap > _scale(real)
        return TestResult(light=Light.RED if shortfall else Light.GREEN, metric=max(0.0, gap) if shortfall else 0.0,
                          notes="no spread in the realised downturn LGD (sd = 0): "
                                + ("a certain shortfall" if shortfall else "no shortfall"), **common)
    t = gap / (float(np.std(real, ddof=1)) / math.sqrt(n))
    p = float(stats.t.sf(t, n - 1))
    return TestResult(light=_p_light(p, alpha_amber, alpha_red), statistic=t, p_value=p, metric=max(0.0, gap), **common)


# ----------------------------------------------------------------------------------------------------------------
# EAD back-test and CCF estimator bias


@register(
    "lgd_ead.ead_backtest_by_band",
    family="lgd_ead",
    name="EAD backtest by band",
    h0="the predicted EAD equals the realised EAD in every utilisation band",
    reference=f"{TONG_2016}; {QI_2009}; dossier 04 G.3 (back-test in currency by utilisation band)",
    decision="metric",
)
def ead_backtest_by_band(predicted_ead: np.ndarray, realised_ead: np.ndarray, utilisation: np.ndarray,
                         band_edges: np.ndarray, model_id: str | None = None, gap_amber: float = 0.10,
                         gap_red: float = 0.25) -> TestResult:
    """Predicted against realised EAD in currency, by utilisation band (not only as a CCF; dossier 04 G.3).

    Bands are [edge_i, edge_i+1), the last closed; a utilisation outside the edges raises. Per band gap =
    sum(predicted) -
    sum(realised) and relative_gap = gap / sum(realised); statistic = the relative gap of the total (null when the total
    realised EAD is 0), metric = max over the bands of |relative_gap|, p_value null; extras["bands"] lists lower,
    upper, n,
    predicted_ead, realised_ead, gap and relative_gap (null for a band with no observation, which does not enter the
    metric; null relative_gap and a listed gap for a band whose realised EAD is 0, which does not enter it either).
    Every band empty, or no band with a realised EAD: not_evaluated. Light (a Contraste internal gate): green when
    metric
    <= `gap_amber`, amber when metric <= `gap_red`, red above. extras["thresholds"] carries `gap_amber` and `gap_red`.
    """
    test_id = "lgd_ead.ead_backtest_by_band"
    reference = f"{TONG_2016}; {QI_2009}; dossier 04 G.3 (back-test in currency by utilisation band)"
    h0 = "the predicted EAD equals the realised EAD in every utilisation band"
    alt = "the predicted EAD differs from the realised EAD in at least one utilisation band"
    if not 0.0 <= gap_amber <= gap_red:
        raise ValueError(f"gap_amber and gap_red must satisfy 0 <= gap_amber <= gap_red, got {gap_amber} and {gap_red}")
    pred = vec(predicted_ead, "predicted_ead", allow_empty=True)
    real = vec(realised_ead, "realised_ead", allow_empty=True)
    util = vec(utilisation, "utilisation", allow_empty=True)
    same_length(predicted_ead=pred, realised_ead=real, utilisation=util)
    edges = vec(band_edges, "band_edges")
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("band_edges must hold at least 2 strictly increasing values")
    if np.any(real < 0.0):
        raise ValueError("realised_ead must be non-negative")   # a predicted EAD may be negative (utilisation change)
    if util.size and (util.min() < edges[0] or util.max() > edges[-1]):
        raise ValueError(f"utilisation must lie within the band edges [{edges[0]}, {edges[-1]}]")
    version = _version(gap_amber=gap_amber, gap_red=gap_red)
    n = int(pred.size)
    kw = dict(model_id=model_id, n=n)
    thresholds = {"thresholds": {"gap_amber": float(gap_amber), "gap_red": float(gap_red)}}
    if n == 0:
        return _not_evaluated(test_id, h0, alt, reference, version, "no observation", extras=thresholds, **kw)
    nb = edges.size - 1
    idx = np.minimum(np.searchsorted(edges, util, side="right") - 1, nb - 1)
    bands: list[dict[str, Any]] = []
    rels: list[float] = []
    for b in range(nb):
        m = idx == b
        row: dict[str, Any] = {"lower": float(edges[b]), "upper": float(edges[b + 1]), "n": int(m.sum()),
                               "predicted_ead": None, "realised_ead": None, "gap": None, "relative_gap": None}
        if m.any():
            p_sum, r_sum = float(pred[m].sum()), float(real[m].sum())
            row.update(predicted_ead=p_sum, realised_ead=r_sum, gap=p_sum - r_sum)
            if r_sum > 0.0:
                row["relative_gap"] = (p_sum - r_sum) / r_sum
                rels.append(abs(row["relative_gap"]))
        bands.append(row)
    extras = {"bands": bands, **thresholds}
    total_real = float(real.sum())
    if not rels:
        return _not_evaluated(test_id, h0, alt, reference, version, "the realised EAD is zero in every band",
                              extras=extras, **kw)
    metric = max(rels)
    light = Light.GREEN if metric <= gap_amber + EPS else Light.AMBER if metric <= gap_red + EPS else Light.RED
    statistic = (float(pred.sum()) - total_real) / total_real if total_real > 0.0 else NAN
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version, light=light,
        statistic=statistic, metric=metric, inputs_hash=inputs_hash(pred, real, util, edges), extras=extras, **kw,
    )


@register(
    "lgd_ead.ccf_estimator_bias",
    family="lgd_ead",
    name="CCF estimator bias",
    h0="the estimator is unbiased (the mean estimate equals the known CCF)",
    reference=f"{TONG_2016}; {QI_2009}; dossier 04 G.1 to G.3 (the CCF estimators and their reference-date designs)",
    decision="metric",
)
def ccf_estimator_bias(estimates: np.ndarray, truth: float, tolerance: float, model_id: str | None = None,
                       alpha_amber: float = 0.05, alpha_red: float = 0.01) -> TestResult:
    """The bias of a CCF estimator against the known CCF of the C09 generator, over R independent replications.

    With e = estimates - truth: metric = mean(e) (the bias, equal to `ccf.ccf_bias(mean(estimates), truth)`),
    statistic =
    mean(e) / (sd(e) / sqrt(R)), p_value two-sided with R - 1 degrees of freedom; extras: rmse, ci_low, ci_high
    (null for
    R < 2), within_tolerance and thresholds (`tolerance`). R < 2 or no spread leaves the statistic and p_value null; the
    light depends only on |metric| against the tolerance and is kept for R >= 1; R < 1 is not_evaluated. Light (a
    Contraste internal gate): green when |metric| <= `tolerance`, amber when |metric| <= 2 `tolerance`, red above; the
    p-value is informative (a statistically significant bias inside the tolerance stays green, and the view shows both).
    """
    test_id = "lgd_ead.ccf_estimator_bias"
    reference = f"{TONG_2016}; {QI_2009}; dossier 04 G.1 to G.3 (the CCF estimators and their reference-date designs)"
    h0 = "the estimator is unbiased (the mean estimate equals the known CCF)"
    alt = "the estimator is biased (the mean estimate differs from the known CCF)"
    _check_alphas(alpha_amber, alpha_red)
    est = vec(estimates, "estimates", allow_empty=True)
    tr = scalar(truth, "truth")
    tol = scalar(tolerance, "tolerance")
    if tol < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance!r}")
    version = _version(tolerance=tol, alpha_amber=alpha_amber, alpha_red=alpha_red)
    r = int(est.size)
    kw = dict(model_id=model_id, n=r, alpha_amber=alpha_amber, alpha_red=alpha_red)
    thresholds = {"tolerance": tol}
    if r < 1:
        return _not_evaluated(test_id, h0, alt, reference, version, "no replication",
                              extras={"thresholds": thresholds}, **kw)
    e = est - tr
    mean = float(e.mean())
    ab = abs(mean)
    light = Light.GREEN if ab <= tol + EPS else Light.AMBER if ab <= 2.0 * tol + EPS else Light.RED
    extras: dict[str, Any] = {"rmse": float(math.sqrt(np.mean(e * e))), "ci_low": None, "ci_high": None,
                              "within_tolerance": bool(ab <= tol + EPS), "truth": tr, "thresholds": thresholds}
    common = dict(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version, light=light,
                  metric=mean, inputs_hash=inputs_hash(est, tr, tol), **kw)
    if r < 2:
        return TestResult(extras=extras, notes="fewer than 2 replications: no statistic and no p-value", **common)
    if _no_spread(e):
        extras.update(ci_low=mean, ci_high=mean)
        return TestResult(extras=extras,
                          notes="no spread across the replications (sd = 0): no statistic and no p-value",
                          **common)
    t, p, mean, se = _paired_t(e)
    half = float(stats.t.ppf(0.975, r - 1)) * se
    extras.update(ci_low=mean - half, ci_high=mean + half)
    return TestResult(statistic=t, p_value=p, extras=extras, **common)
