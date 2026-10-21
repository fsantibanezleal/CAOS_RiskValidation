"""The operational risk validation tests of the ladder (dossier 05, "Implications for Contraste": the acceptance tests
of the operational risk ladder and case O1): each returns one `TestResult`.

The tests are descriptive: `p_value`, `alpha_amber` and `alpha_red` are null, and the lights compare a metric against
tolerances the caller supplies; those tolerances are Contraste internal gates, stated as such, never regulatory
thresholds, and travel in `extras["thresholds"]` (and in `policy_version`) so that a view can explain a light. A test
that cannot be evaluated returns the light `not_evaluated` with its reason in `notes` and a null statistic, metric and
p-value once serialised, never a fabricated number. Capital stability is judged on independent replications (fit on
simulated years
1 to K, evaluate on replications never used for tuning), as the ladder prescribes.

The tests are plain functions that do not read the registry or the default policy: their rules are the contract's
(`engine-oprisk-api`, module validation.oprisk). They state no compliance claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.oprisk._checks import positive, scalar, vec
from riskvalidation.oprisk.sma import OPE25_MINIMUM_YEARS, ilm, loss_component

POLICY_LABEL = "contraste-oprisk-battery 1.0.0"
NAN = float("nan")
EPS = 1e-12  # tolerance on a gate comparison, so that a value exactly on a gate is not moved by rounding

SHEVCHENKO_2010 = (
    "Shevchenko (2010), Implementing loss distribution approach for operational risk, J. Operational Risk 5(2):3-40, "
    "arXiv 1008.1108 (Example 9.4 and Table 2), as quoted in dossier 05 F.3"
)
LADDER = "dossier 05 (operational risk ladder and case O1, Implications for Contraste)"
OPE25 = "Basel Framework OPE25 (BCBS, in force 2023-01-01), dossier 01 section F"

BENCHMARK_MOMENTS = {"mean": 738.9056, "variance": 298095.7987, "skewness": 40.3428}
BENCHMARK_QUANTILE_RANGE = (5849.0, 5850.0)
BENCHMARK_QUANTILE_SCALE = 5849.5
# the continuous 99.9 percent quantile (Shevchenko Table 3 at step 0.0625: 5853.06; Table 7)
MONTE_CARLO_QUANTILE = 5853.1
METHODS = ("monte-carlo", "panjer", "fft")


def _not_evaluated(test_id: str, h0: str, reference: str, version: str, reason: str, *, model_id: str | None,
                   n: int | None = None, extras: Mapping[str, Any] | None = None) -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(test_id=test_id, h0=h0, reference=reference, policy_version=version, light=Light.NOT_EVALUATED,
                      statistic=NAN, notes=f"not evaluated: {reason}", model_id=model_id, n=n,
                      extras=dict(extras or {}))


def _gate_light(metric: float, green: float, amber: float) -> Light:
    if metric <= green + EPS:
        return Light.GREEN
    return Light.AMBER if metric <= amber + EPS else Light.RED


def _worse(a: Light, b: Light) -> Light:
    order = (Light.GREEN, Light.AMBER, Light.RED)
    return order[max(order.index(a), order.index(b))]


def _check_gates(green: float, amber: float, names: tuple[str, str]) -> None:
    if not (0.0 < green <= amber):
        raise ValueError(f"{names[0]} and {names[1]} must satisfy 0 < {names[0]} <= {names[1]}, "
                         f"got {green} and {amber}")


# ----------------------------------------------------------------------------------------------------------------
# Shevchenko's benchmark by the three aggregate methods


def _quantile_deviation(method: str, q: float) -> float:
    if method == "monte-carlo":  # a Monte Carlo sample estimates the continuous quantile, not the discretised one
        return abs(q - MONTE_CARLO_QUANTILE) / MONTE_CARLO_QUANTILE
    lo, hi = BENCHMARK_QUANTILE_RANGE
    if lo <= q <= hi:
        return 0.0
    return (lo - q if q < lo else q - hi) / BENCHMARK_QUANTILE_SCALE


@register(
    "oprisk.benchmark_reproduction",
    family="oprisk",
    name="Benchmark reproduction",
    h0=(
        "the engine reproduces Shevchenko's benchmark, Poisson(100) with LN(0, 2): E[Z] = 738.9056, "
        "Var[Z] = 298095.7987, skewness 40.3428, a 99.9 percent quantile of 5849 to 5850 by Panjer and FFT and "
        "5853.1 by Monte Carlo"),
    reference=f"{SHEVCHENKO_2010}; {LADDER}",
    decision="metric",
)
def benchmark_reproduction(moments: tuple[float, float, float], quantiles: Mapping[str, float],
                           moment_rel_tol: float = 1e-5, quantile_rel_tol: float = 0.01,
                           model_id: str | None = None) -> TestResult:
    """The engine reproduces Shevchenko's benchmark, Poisson(100) with LN(0, 2): E[Z] = 738.9056, Var[Z] = 298095.7987,
    skewness 40.3428, and a 99.9 percent quantile of 5849 to 5850 (the step-1 discretised quantile) by Panjer and FFT
    and of 5853.1 (the continuous quantile; Shevchenko Table 3 at step 0.0625 and Table 7) by Monte Carlo.

    `moments` is (mean, variance, skewness) computed by the engine; `quantiles` maps "monte-carlo", "panjer" and "fft"
    to the 99.9 percent quantile in loss units. A moment's deviation is |value - reference| / reference; a `panjer` or
    `fft` quantile's deviation is 0 inside [5849, 5850] and otherwise the distance to the nearest end over 5849.5; a
    `monte-carlo` quantile estimates the continuous quantile, so its deviation is |q - 5853.1| / 5853.1. metric =
    max(moment deviations / `moment_rel_tol`, quantile deviations / `quantile_rel_tol`) and statistic equals metric;
    extras carry `deviations` by name, `max_moment_rel_dev` and `max_quantile_rel_dev`. A missing method gives
    not_evaluated.
    The references carry four decimals (the skewness 40.3428 is the exact e^6 / 10 = 40.342879 cut, a relative error of
    1.97e-6), so `moment_rel_tol` must exceed that rounding when the moments come from the engine: use at least 1e-5
    (at 1e-6 the engine's exact moments read amber; 1e-5 is the default).
    Light (a Contraste internal gate): green when metric <= 1, amber when metric <= 2, red above.
    """
    test_id = "oprisk.benchmark_reproduction"
    h0 = ("the engine reproduces Shevchenko's benchmark, Poisson(100) with LN(0, 2): E[Z] = 738.9056, "
          "Var[Z] = 298095.7987, skewness 40.3428, a 99.9 percent quantile of 5849 to 5850 by Panjer and FFT and "
          "5853.1 by Monte Carlo")
    reference = f"{SHEVCHENKO_2010}; {LADDER}"
    m = vec(moments, "moments")
    if m.size != 3:
        raise ValueError(f"moments must be (mean, variance, skewness), got {m.size} values")
    mtol, qtol = positive(moment_rel_tol, "moment_rel_tol"), positive(quantile_rel_tol, "quantile_rel_tol")
    version = POLICY_LABEL
    thresholds = {"moment_rel_tol": mtol, "quantile_rel_tol": qtol}
    for name, value in quantiles.items():
        scalar(value, f"quantiles[{name!r}]")
    missing = [k for k in METHODS if k not in quantiles]
    if missing:
        return _not_evaluated(test_id, h0, reference, version, f"no 99.9 percent quantile for {', '.join(missing)}",
                              model_id=model_id, extras={"thresholds": thresholds})
    deviations: dict[str, float] = {}
    for (name, ref), value in zip(BENCHMARK_MOMENTS.items(), m, strict=True):
        deviations[name] = abs(float(value) - ref) / ref
    for name in ("panjer", "fft", "monte-carlo"):
        deviations[name] = _quantile_deviation(name, float(quantiles[name]))
    max_mom = max(deviations[k] for k in BENCHMARK_MOMENTS)
    max_q = max(deviations[k] for k in ("panjer", "fft", "monte-carlo"))
    metric = max(max_mom / mtol, max_q / qtol)
    return TestResult(
        test_id=test_id, h0=h0, alternative="at least one moment or quantile deviates from the benchmark",
        reference=reference, policy_version=version, light=_gate_light(metric, 1.0, 2.0), statistic=metric,
        metric=metric, model_id=model_id,
        inputs_hash=inputs_hash(m, np.array([float(quantiles[k]) for k in METHODS])),
        notes="descriptive: a Contraste internal gate on the relative deviations, no p-value",
        extras={"deviations": deviations, "max_quantile_rel_dev": max_q, "max_moment_rel_dev": max_mom,
                "thresholds": thresholds},
    )


# ----------------------------------------------------------------------------------------------------------------
# threshold stability of the fitted tail index


@register(
    "oprisk.threshold_stability",
    family="oprisk",
    name="Threshold stability",
    h0="the fitted tail index is stable across thresholds",
    reference=f"{LADDER}; Chavez-Demoulin, Embrechts and Neslehova (2006), doi:10.1016/j.jbankfin.2005.11.008",
    decision="metric",
)
def threshold_stability(thresholds: np.ndarray, xi_hat: np.ndarray, model_id: str | None = None,
                        tol_green: float = 0.10, tol_amber: float = 0.25) -> TestResult:
    """The fitted GPD tail index is stable across thresholds (the threshold-stability plot of dossier 05, ladder rungs 2
    and
    3). `thresholds` is strictly increasing; `xi_hat` holds the fitted tail indices (nan for a threshold that could not
    be fitted). metric = max - min over the finite values (the range) and statistic equals metric; fewer than 2 finite
    values give not_evaluated; `extras["n_infinite_mean"]` counts the values at or above 1 (an infinite-mean fit); n is
    the number of finite values. Light (a Contraste internal gate): green when metric <= `tol_green`, amber when <=
    `tol_amber`, red above.
    """
    test_id = "oprisk.threshold_stability"
    h0 = "the fitted tail index is stable across thresholds"
    reference = f"{LADDER}; Chavez-Demoulin, Embrechts and Neslehova (2006), doi:10.1016/j.jbankfin.2005.11.008"
    th = vec(thresholds, "thresholds")
    xi = np.array(xi_hat, dtype=float)
    if xi.ndim != 1 or xi.size != th.size:
        raise ValueError("xi_hat must be one-dimensional with the length of thresholds")
    if th.size > 1 and not np.all(np.diff(th) > 0.0):
        raise ValueError("thresholds must be strictly increasing")
    if np.any(np.isinf(xi)):
        raise ValueError("xi_hat contains infinite values")
    tg, ta = scalar(tol_green, "tol_green"), scalar(tol_amber, "tol_amber")
    _check_gates(tg, ta, ("tol_green", "tol_amber"))
    version = POLICY_LABEL
    gates = {"tol_green": tg, "tol_amber": ta}
    finite = xi[np.isfinite(xi)]
    if finite.size < 2:
        return _not_evaluated(test_id, h0, reference, version, "fewer than 2 thresholds with a fitted tail index",
                              model_id=model_id, n=int(finite.size), extras={"thresholds": gates,
                                      "n_infinite_mean": int((finite >= 1.0).sum())})
    metric = float(finite.max() - finite.min())
    return TestResult(
        test_id=test_id, h0=h0, alternative="the fitted tail index drifts with the threshold", reference=reference,
        policy_version=version, light=_gate_light(metric, tg, ta), statistic=metric, metric=metric, n=int(finite.size),
        model_id=model_id, inputs_hash=inputs_hash(th, xi),
        notes="descriptive: the range of the fitted tail index, no p-value",
        extras={"n_infinite_mean": int((finite >= 1.0).sum()), "thresholds": gates},
    )


# ----------------------------------------------------------------------------------------------------------------
# capital stability across independent histories


@register(
    "oprisk.capital_stability",
    family="oprisk",
    name="Capital stability",
    h0="the capital is stable across independent histories and unbiased against the truth",
    reference=f"{LADDER}; Shevchenko (2010), arXiv 1008.1108 (Monte Carlo quantile error O(K^-1/2))",
    decision="metric",
)
def capital_stability(capitals: np.ndarray, truth: float, model_id: str | None = None, dispersion_green: float = 0.25,
                      dispersion_amber: float = 0.50, bias_green: float = 0.10, bias_amber: float = 0.25) -> TestResult:
    """The 99.9 percent capital is stable across independent loss histories and unbiased against the truth. `capitals`
    holds the capital of each independent history (EUR), `truth > 0` the true 99.9 percent annual loss quantile (known
    by construction in C17). Percentiles by linear interpolation (the numpy default); metric = (p75 - p25) / truth (the
    dispersion), statistic = median / truth - 1 (the bias); extras carry `ratio_p05`, `ratio_p25`, `ratio_median`,
    `ratio_p75`, `ratio_p95` (percentiles of capitals / truth) and `share_underestimating`; fewer than 4 capitals give
    not_evaluated.
    Light (Contraste internal gates): the worse of dispersion (green when metric <= `dispersion_green`, amber when <=
    `dispersion_amber`, red above) and bias (the same on |statistic| with `bias_green` and `bias_amber`).
    """
    test_id = "oprisk.capital_stability"
    h0 = "the capital is stable across independent histories and unbiased against the truth"
    reference = f"{LADDER}; Shevchenko (2010), arXiv 1008.1108 (Monte Carlo quantile error O(K^-1/2))"
    cap = vec(capitals, "capitals")
    tr = positive(truth, "truth")
    dg, da = scalar(dispersion_green, "dispersion_green"), scalar(dispersion_amber, "dispersion_amber")
    bg, ba = scalar(bias_green, "bias_green"), scalar(bias_amber, "bias_amber")
    _check_gates(dg, da, ("dispersion_green", "dispersion_amber"))
    _check_gates(bg, ba, ("bias_green", "bias_amber"))
    version = POLICY_LABEL
    gates = {"dispersion_green": dg, "dispersion_amber": da, "bias_green": bg, "bias_amber": ba}
    if cap.size < 4:
        return _not_evaluated(test_id, h0, reference, version, f"fewer than 4 independent histories ({cap.size})",
                              model_id=model_id, n=int(cap.size), extras={"thresholds": gates})
    ratio = cap / tr
    p05, p25, med, p75, p95 = (float(v) for v in np.percentile(ratio, [5, 25, 50, 75, 95]))
    dispersion, bias = p75 - p25, med - 1.0
    light = _worse(_gate_light(dispersion, dg, da), _gate_light(abs(bias), bg, ba))
    return TestResult(
        test_id=test_id, h0=h0, alternative="the capital is dispersed across histories or biased against the truth",
        reference=reference, policy_version=version, light=light, statistic=bias, metric=dispersion, n=int(cap.size),
        model_id=model_id, inputs_hash=inputs_hash(cap, np.array([tr])),
        notes="descriptive: dispersion (p75 - p25) / truth and bias median / truth - 1, no p-value",
        extras={"ratio_p05": p05, "ratio_p25": p25, "ratio_median": med, "ratio_p75": p75, "ratio_p95": p95,
                "share_underestimating": float((cap < tr).mean()), "thresholds": gates},
    )


# ----------------------------------------------------------------------------------------------------------------
# the SMA's response to one large loss


@register(
    "oprisk.sma_ilm_response",
    family="oprisk",
    name="SMA ILM response to a large loss",
    h0="the ILM does not jump when one large loss is added to the loss history",
    reference=f"{OPE25} (OPE25.8, OPE25.9, OPE25.15); {LADDER}",
    decision="metric",
)
def sma_ilm_response(bic_value: float, annual_losses: np.ndarray, added_loss: float, model_id: str | None = None,
                     jump_green: float = 0.05, jump_amber: float = 0.15) -> TestResult:
    """The ILM does not jump when one large loss is added to the history (the SMA's response to a single large loss).
    `bic_value > 0`; `annual_losses` are the annual operational risk losses net of recoveries, oldest first; `added_loss
    >= 0` is added to the latest year. ILM_before and ILM_after come from `loss_component` and `ilm` on the history
    without and with the added loss; metric = ILM_after - ILM_before, statistic = ILM_after. Fewer than 5 years give
    not_evaluated with the reason (OPE25.15: no ILM on under 5 years). extras carry `ilm_before`, `ilm_after`,
    `lc_before`, `lc_after`, `orc_before`, `orc_after` and the tolerances. Light (a Contraste internal gate): green when
    metric <= `jump_green`, amber when <= `jump_amber`, red above.
    """
    test_id = "oprisk.sma_ilm_response"
    h0 = "the ILM does not jump when one large loss is added to the loss history"
    reference = f"{OPE25} (OPE25.8, OPE25.9, OPE25.15); {LADDER}"
    bic_v = positive(bic_value, "bic_value")
    losses = vec(annual_losses, "annual_losses")
    added = scalar(added_loss, "added_loss")
    if added < 0.0:
        raise ValueError(f"added_loss must not be negative, got {added}")
    jg, ja = scalar(jump_green, "jump_green"), scalar(jump_amber, "jump_amber")
    _check_gates(jg, ja, ("jump_green", "jump_amber"))
    version = POLICY_LABEL
    gates = {"jump_green": jg, "jump_amber": ja}
    lc_before = loss_component(losses)  # also refuses a negative loss
    if losses.size < OPE25_MINIMUM_YEARS:
        return _not_evaluated(test_id, h0, reference, version,
                              f"{losses.size} years of loss data, fewer than {OPE25_MINIMUM_YEARS} (OPE25.15): no ILM",
                              model_id=model_id, n=int(losses.size), extras={"thresholds": gates})
    after = losses.copy()
    after[-1] += added
    lc_after = loss_component(after)
    ilm_before, ilm_after = ilm(lc_before, bic_v), ilm(lc_after, bic_v)
    metric = ilm_after - ilm_before
    return TestResult(
        test_id=test_id, h0=h0, alternative="the ILM jumps when a large loss is added", reference=reference,
        policy_version=version, light=_gate_light(metric, jg, ja), statistic=ilm_after, metric=metric,
        n=int(losses.size), model_id=model_id, inputs_hash=inputs_hash(losses, np.array([bic_v, added])),
        notes="descriptive: the ILM jump after one added loss, no p-value",
        extras={"ilm_before": ilm_before, "ilm_after": ilm_after, "lc_before": lc_before, "lc_after": lc_after,
                "orc_before": bic_v * ilm_before, "orc_after": bic_v * ilm_after, "added_loss": added,
                "thresholds": gates},
    )


# ----------------------------------------------------------------------------------------------------------------
# scenario influence


@register(
    "oprisk.scenario_influence",
    family="oprisk",
    name="Scenario influence",
    h0="no single scenario point moves the 99.9 percent capital by more than the tolerance",
    reference=(
        "dossier 05 F.5 (scenario analysis and its combination with loss data); Shevchenko and Wuthrich "
        "(2006), J. Operational Risk 1(3):3-26, doi:10.21314/JOP.2006.016"),
    decision="metric",
)
def scenario_influence(influences: np.ndarray, model_id: str | None = None, influence_amber: float = 0.10,
                       influence_red: float = 0.25) -> TestResult:
    """No single scenario point moves the capital by more than the tolerance (dossier 05 F.5: validation must test the
    sensitivity of capital to each scenario point). `influences` is the output of `scenarios.scenario_influence`
    (dimensionless relative changes of the 99.9 percent capital). metric = max |influences| and statistic equals metric;
    extras carry the `influences` and the index `argmax` of the largest; no influence gives not_evaluated.
    Light (a Contraste internal gate): green when metric <= `influence_amber`, amber when <= `influence_red`, red above.
    """
    test_id = "oprisk.scenario_influence"
    h0 = "no single scenario point moves the 99.9 percent capital by more than the tolerance"
    reference = ("dossier 05 F.5 (scenario analysis and its combination with loss data); Shevchenko and Wuthrich "
                 "(2006), J. Operational Risk 1(3):3-26, doi:10.21314/JOP.2006.016")
    infl = vec(influences, "influences", allow_empty=True)
    ia, ir = scalar(influence_amber, "influence_amber"), scalar(influence_red, "influence_red")
    _check_gates(ia, ir, ("influence_amber", "influence_red"))
    version = POLICY_LABEL
    gates = {"influence_amber": ia, "influence_red": ir}
    if infl.size < 1:
        return _not_evaluated(test_id, h0, reference, version, "no scenario influence", model_id=model_id, n=0,
                              extras={"thresholds": gates})
    mag = np.abs(infl)
    metric = float(mag.max())
    return TestResult(
        test_id=test_id, h0=h0, alternative="a single scenario point moves the capital by more than the tolerance",
        reference=reference, policy_version=version, light=_gate_light(metric, ia, ir), statistic=metric, metric=metric,
        n=int(infl.size), model_id=model_id, inputs_hash=inputs_hash(infl),
        notes="descriptive: the largest absolute influence of one scenario point, no p-value",
        extras={"influences": [float(v) for v in infl], "argmax": int(np.argmax(mag)), "thresholds": gates},
    )

__all__ = [
    "benchmark_reproduction",
    "capital_stability",
    "scenario_influence",
    "sma_ilm_response",
    "threshold_stability",
]
