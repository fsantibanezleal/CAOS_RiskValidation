"""The stress validation tests (dossier 05 H.7, l.422): each returns one `TestResult` that validates against
`common/test-result`.

Acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds. Each function states its light
rule in its docstring and records its parameters in `policy_version` and in `extras["thresholds"]`, so that a view can
explain a light. A test that cannot be evaluated returns the light `not_evaluated` with its reason in `notes` and a null
statistic, metric and p-value once serialised, never a fabricated number. The tests are plain functions that do not read
the registry or the default policy: their rules are the contract's (`engine-stress-api`, module validation.stress). They
state no compliance claim. The supervisory replication test of the unit sizing is void with decision E1.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.lgd_ead._checks import mat, scalar, vec

POLICY_LABEL = "contraste-stress-battery 1.0.0"
NAN = float("nan")
EPS = 1e-12  # tolerance on a gate comparison, so that a value exactly on a gate is not moved by rounding
START_TOLERANCE = 1e-6  # a start counts as closer when its distance is below distance (1 - 1e-6) (the contract)

DOSSIER_05_H7 = (
    "dossier 05 H.7 (Contraste initiative research): back-cast the satellites through 2008-2009 and 2020 with data "
    "ending before the episode, check the sign of macro sensitivities, the static balance-sheet rules, bottom-up "
    "against top-down, "
    "severity against history"
)
EBA_NOTE = (
    "EBA 2025 EU-wide stress test, Methodological Note of 2025-01-20, paras 33, 34 and 90 (static balance sheet: "
    "constant total, no cure of stage 3); dossier 05 H.4 and H.7"
)
GFC_SEVERITY = (
    "dossier 05 H.7: document severity relative to history (GDP and unemployment paths against the GFC, "
    "as the EBA does)"
)
REVERSE_CHECK = (
    "dossier 01 J.4 and dossier 05 H.5 and H.7: the reverse stress solution, the minimum Mahalanobis distance scenario"
)


def _not_evaluated(
    test_id: str,
    h0: str,
    alternative: str,
    reference: str,
    version: str,
    reason: str,
    *,
    n: int | None = None,
    extras: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    segment: str | None = None,
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
        model_id=model_id,
        segment=segment,
        n=n,
        extras=dict(extras or {}),
    )


def _three_light(value: float, green: float, amber: float) -> Light:
    """Green at or below `green`, amber at or below `amber`, red above (with a rounding tolerance)."""
    if value <= green + EPS:
        return Light.GREEN
    return Light.AMBER if value <= amber + EPS else Light.RED


def _check_gates(low: float, high: float, low_name: str, high_name: str) -> None:
    if not (0.0 <= low <= high):
        raise ValueError(
            f"{low_name} and {high_name} must satisfy 0 <= {low_name} <= {high_name}, got {low} and {high}"
        )


def _quarter_number(iso: str, name: str) -> int:
    try:
        d = _dt.date.fromisoformat(str(iso))
    except ValueError as exc:
        raise ValueError(f"{name} {iso!r} is not an ISO date") from exc
    return d.year * 4 + (d.month - 1) // 3


# ----------------------------------------------------------------------------------------------------------------
# The satellites


@register(
    "stress.satellite_backcast",
    family="stress",
    name="Satellite backcast",
    h0=(
        "satellites estimated before the episode back-cast it: the predicted loss-rate peak is within the "
        "thresholds of the realised "
        "peak, in size and timing"),
    reference=DOSSIER_05_H7,
    decision="metric",
)
def satellite_backcast(
    predicted: np.ndarray,
    realised: np.ndarray,
    quarters: Sequence[str],
    fit_end: str,
    peak_amber: float = 0.35,
    peak_red: float = 0.60,
    timing_quarters: int = 2,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """Satellites estimated before an episode back-cast it: the predicted loss-rate peak is within the thresholds of the
    realised peak, in size and timing (05 l.422). `predicted` and `realised` are the rates of the episode's quarters
    (decimals), `quarters` their ISO quarter ends, `fit_end` the last quarter of the fit window.

    statistic = the RMSE of predicted against realised, metric = the peak timing error in quarters (predicted peak
    quarter less realised peak quarter); `extras["peak_ratio"]` = peak predicted / peak realised and
    `extras["thresholds"]` = peak_amber, peak_red and timing_quarters. Light (a Contraste internal gate): green when `|1
    - peak_ratio| <= peak_amber` and
    `|timing| <= timing_quarters`; amber when `|1 - peak_ratio| <= peak_red`; red otherwise. The light `error` with its
    reason when `fit_end` is not before the first quarter of the episode (leakage); `not_evaluated` when the realised
    peak is not positive (no
    size ratio)."""
    test_id = "stress.satellite_backcast"
    h0 = (
        "satellites estimated before the episode back-cast it: the predicted loss-rate peak is within the "
        "thresholds of the realised "
        "peak, in size and timing"
    )
    alt = "the predicted peak differs from the realised peak by more than the thresholds in size or timing"
    pred, real = vec(predicted, "predicted"), vec(realised, "realised")
    qs = [str(q) for q in quarters]
    if pred.size != real.size or pred.size != len(qs):
        raise ValueError(
            f"predicted ({pred.size}), realised ({real.size}) and quarters ({len(qs)}) must have the same length"
        )
    amber, red, tq = (
        scalar(peak_amber, "peak_amber"),
        scalar(peak_red, "peak_red"),
        scalar(timing_quarters, "timing_quarters"),
    )
    _check_gates(amber, red, "peak_amber", "peak_red")
    if tq < 0:
        raise ValueError("timing_quarters must be nonnegative")
    q_idx = [_quarter_number(q, "quarters") for q in qs]
    if any(b <= a for a, b in zip(q_idx, q_idx[1:], strict=False)):
        raise ValueError("quarters must be ascending")
    fit_q = _quarter_number(fit_end, "fit_end")
    thresholds = {"peak_amber": amber, "peak_red": red, "timing_quarters": int(tq) if tq == int(tq) else tq}
    version = POLICY_LABEL
    if fit_q >= q_idx[0]:
        return TestResult(
            test_id=test_id,
            h0=h0,
            alternative=alt,
            reference=DOSSIER_05_H7,
            policy_version=version,
            light=Light.ERROR,
            n=int(pred.size),
            model_id=model_id,
            segment=segment,
            notes=(
                f"fit_end {fit_end} is not before the first quarter of the episode {qs[0]}: the fit window "
                f"overlaps the episode it predicts (leakage)"
            ),
            extras={"thresholds": thresholds},
        )
    ip, ir = int(np.argmax(pred)), int(np.argmax(real))
    peak_p, peak_r = float(pred[ip]), float(real[ir])
    if peak_r <= 0.0:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            DOSSIER_05_H7,
            version,
            f"the realised peak {peak_r:.6g} is not positive: no size ratio",
            n=int(pred.size),
            extras={"thresholds": thresholds},
            model_id=model_id,
            segment=segment,
        )
    ratio = peak_p / peak_r
    timing = q_idx[ip] - q_idx[ir]
    rmse = float(np.sqrt(np.mean((pred - real) ** 2)))
    dev = abs(1.0 - ratio)
    if dev <= amber + EPS and abs(timing) <= tq:
        light = Light.GREEN
    elif dev <= red + EPS:
        light = Light.AMBER
    else:
        light = Light.RED
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=DOSSIER_05_H7,
        policy_version=version,
        light=light,
        statistic=rmse,
        metric=float(timing),
        n=int(pred.size),
        model_id=model_id,
        segment=segment,
        inputs_hash=inputs_hash(pred, real, qs, fit_end),
        extras={
            "thresholds": thresholds,
            "peak_ratio": ratio,
            "peak_predicted": peak_p,
            "peak_realised": peak_r,
            "peak_predicted_quarter": qs[ip],
            "peak_realised_quarter": qs[ir],
        },
    )


SIGNS = ("positive", "negative", "none")


@register(
    "stress.sensitivity_monotonicity",
    family="stress",
    name="Sensitivity monotonicity",
    h0="every macro sensitivity has its expected sign, so the projected loss rate is monotone in the path severity",
    reference=DOSSIER_05_H7,
    decision="metric",
)
def sensitivity_monotonicity(
    estimates: Sequence[float], expected_signs: Sequence[str], model_id: str | None = None, segment: str | None = None
) -> TestResult:
    """Every macro sensitivity has its expected sign, so the projected loss rate is monotone in the severity of the path
    (05 l.422).
    `expected_signs` are `positive`, `negative` or `none`.

    statistic = the number of coefficients with the wrong sign (an estimate of zero has no sign and is wrong for
    `positive` and `negative`; `none` is never wrong), metric null, `extras["offending_indices"]` lists them. Light: red
    when the statistic is
    above zero, green otherwise."""
    test_id = "stress.sensitivity_monotonicity"
    h0 = (
        "every macro sensitivity has its expected sign, so the projected loss rate is monotone in the path severity"
    )
    alt = "at least one macro sensitivity has the wrong sign"
    est = vec(estimates, "estimates")
    signs = [str(s) for s in expected_signs]
    if len(signs) != est.size:
        raise ValueError(f"expected_signs has {len(signs)} entries for {est.size} estimates")
    bad = [s for s in signs if s not in SIGNS]
    if bad:
        raise ValueError(f"expected_signs entry {bad[0]!r} is not one of {', '.join(SIGNS)}")
    offending = [
        i
        for i, (e, s) in enumerate(zip(est, signs, strict=False))
        if (s == "positive" and not e > 0.0) or (s == "negative" and not e < 0.0)
    ]
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=DOSSIER_05_H7,
        policy_version=POLICY_LABEL,
        light=Light.RED if offending else Light.GREEN,
        statistic=float(len(offending)),
        n=int(est.size),
        n_events=len(offending),
        model_id=model_id,
        segment=segment,
        inputs_hash=inputs_hash(est, signs),
        extras={"thresholds": {}, "offending_indices": offending},
    )


# ----------------------------------------------------------------------------------------------------------------
# The balance sheet and the comparison with a challenger


@register(
    "stress.static_balance_sheet_rules",
    family="stress",
    name="Static balance sheet rules",
    h0="the static balance sheet rules are applied: the total is constant, stage 3 never falls, no balance is negative",
    reference=EBA_NOTE,
    decision="metric",
)
def static_balance_sheet_rules(
    balances: np.ndarray, tolerance: float = 1e-9, model_id: str | None = None, segment: str | None = None
) -> TestResult:
    """The static balance sheet rules are applied: the total is constant, stage 3 never falls (no cure), no balance is
    negative
    (EBA Methodological Note paras 33 and 90). `balances` is (H, 3), the stage balances after each period.

    statistic = the number of violated rules among the three (`constant-total`, `no-cure`, `no-negative-balance`), named
    in `extras["violated"]`, with `extras["thresholds"] = {"tolerance": tolerance}` (absolute, in money). Light: red
    when the
    statistic is above zero, green otherwise."""
    test_id = "stress.static_balance_sheet_rules"
    h0 = (
        "the static balance sheet rules are applied: the total is constant, stage 3 never falls, no balance is negative"
    )
    alt = "a static balance sheet rule is violated"
    b = mat(balances, "balances")
    if b.shape[1] != 3:
        raise ValueError(f"balances must be (H, 3), the stage balances after each period, got shape {b.shape}")
    tol = scalar(tolerance, "tolerance")
    if tol < 0.0:
        raise ValueError("tolerance must be nonnegative")
    total = b.sum(axis=1)
    violated = []
    if np.max(total) - np.min(total) > tol:
        violated.append("constant-total")
    if np.any(np.diff(b[:, 2]) < -tol):
        violated.append("no-cure")
    if np.any(b < -tol):
        violated.append("no-negative-balance")
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=EBA_NOTE,
        policy_version=POLICY_LABEL,
        light=Light.RED if violated else Light.GREEN,
        statistic=float(len(violated)),
        n=int(b.shape[0]),
        model_id=model_id,
        segment=segment,
        inputs_hash=inputs_hash(b),
        extras={"thresholds": {"tolerance": tol}, "violated": violated},
    )


@register(
    "stress.bottom_up_vs_top_down",
    family="stress",
    name="Bottom up vs top down",
    h0="the bottom-up stressed loss agrees with a top-down challenger within the thresholds",
    reference=DOSSIER_05_H7,
    decision="metric",
)
def bottom_up_vs_top_down(
    bottom_up: float,
    top_down: float,
    amber: float = 0.25,
    red: float = 0.50,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """The bottom-up result agrees with a top-down challenger within the thresholds (05 l.422). Both are the stressed
    loss, in money.

    statistic = `bottom_up / top_down - 1`, metric equal to the statistic. Light (a Contraste internal gate): green when
    `|statistic| <= amber`, amber when at most `red`, red above; `extras["thresholds"] = {"amber": amber, "red": red}`.
    A top-down
    loss that is not positive gives `not_evaluated`."""
    test_id = "stress.bottom_up_vs_top_down"
    h0 = "the bottom-up stressed loss agrees with a top-down challenger within the thresholds"
    alt = "the bottom-up stressed loss differs from the top-down challenger by more than the thresholds"
    bu, td = scalar(bottom_up, "bottom_up"), scalar(top_down, "top_down")
    a, r = scalar(amber, "amber"), scalar(red, "red")
    _check_gates(a, r, "amber", "red")
    version = POLICY_LABEL
    thresholds = {"amber": a, "red": r}
    if td <= 0.0:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            DOSSIER_05_H7,
            version,
            f"top_down {td:.6g} is not positive: no ratio",
            extras={"thresholds": thresholds},
            model_id=model_id,
            segment=segment,
        )
    stat = bu / td - 1.0
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=DOSSIER_05_H7,
        policy_version=version,
        light=_three_light(abs(stat), a, r),
        statistic=stat,
        metric=stat,
        model_id=model_id,
        segment=segment,
        inputs_hash=inputs_hash(bu, td),
        extras={"thresholds": thresholds},
    )


@register(
    "stress.scenario_severity",
    family="stress",
    name="Scenario severity",
    h0="a stress path is at least as severe as history",
    reference=GFC_SEVERITY,
    decision="metric",
)
def scenario_severity(
    path_peak: float,
    history_peak: float,
    is_stress: bool = True,
    minimum: float = 0.75,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """A stress path is at least as severe as history (the EBA documents severity against the GFC: 05 l.422).
    `path_peak` is the peak deviation from the jump-off of a scenario variable (unemployment in percentage points, GDP
    in percentage points) and `history_peak` the same deviation in the global financial crisis; maps onto the document
    block `outputs.severity`.

    statistic = `path_peak / history_peak`; `is_stress = False` (the baseline) gives `not_evaluated` with the reason
    "not a stress path"; a history peak that is not positive gives `not_evaluated`. Light: green when `statistic >=
    minimum`, amber below (no red: a mild stress is a finding about the scenario, not a failure); `extras["thresholds"]
    = {"minimum":
    minimum}`."""
    test_id = "stress.scenario_severity"
    h0 = "a stress path is at least as severe as history"
    alt = "the stress path is milder than history by more than the minimum ratio allows"
    pp, hp = scalar(path_peak, "path_peak"), scalar(history_peak, "history_peak")
    mn = scalar(minimum, "minimum")
    if mn < 0.0:
        raise ValueError("minimum must be nonnegative")
    version = POLICY_LABEL
    thresholds = {"minimum": mn}
    if not is_stress:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            GFC_SEVERITY,
            version,
            "not a stress path",
            extras={"thresholds": thresholds},
            model_id=model_id,
            segment=segment,
        )
    if hp <= 0.0:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            GFC_SEVERITY,
            version,
            f"history_peak {hp:.6g} is not positive: no ratio",
            extras={"thresholds": thresholds},
            model_id=model_id,
            segment=segment,
        )
    stat = pp / hp
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=GFC_SEVERITY,
        policy_version=version,
        light=Light.GREEN if stat >= mn - EPS else Light.AMBER,
        statistic=stat,
        model_id=model_id,
        segment=segment,
        inputs_hash=inputs_hash(pp, hp),
        extras={"thresholds": thresholds},
    )


@register(
    "stress.reverse_solution_check",
    family="stress",
    name="Reverse solution check",
    h0="the loss at the reverse stress solution equals the threshold and no start finds a closer point",
    reference=REVERSE_CHECK,
    decision="metric",
)
def reverse_solution_check(
    loss_at_x: float,
    threshold: float,
    distance: float,
    start_distances: Sequence[float],
    tolerance: float = 1e-6,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """The loss at the reverse stress solution equals the threshold and no start finds a closer point (05 l.422, H.7).

    statistic = `|loss_at_x - threshold| / threshold`, metric = `min(start_distances) / distance - 1` (zero or positive
    when no start is closer); `extras["thresholds"] = {"tolerance": tolerance}`. Light: red when the statistic is above
    `tolerance` or any
    start distance is below `distance (1 - 1e-6)`, green otherwise; fewer than three starts gives `not_evaluated`."""
    test_id = "stress.reverse_solution_check"
    h0 = "the loss at the reverse stress solution equals the threshold and no start finds a closer point"
    alt = "the loss at the solution misses the threshold or a start finds a closer point"
    loss, thr, dist = scalar(loss_at_x, "loss_at_x"), scalar(threshold, "threshold"), scalar(distance, "distance")
    starts = vec(start_distances, "start_distances", allow_empty=True)
    tol = scalar(tolerance, "tolerance")
    if thr <= 0.0:
        raise ValueError("threshold must be positive")
    if dist <= 0.0:
        raise ValueError("distance must be positive")
    if tol < 0.0:
        raise ValueError("tolerance must be nonnegative")
    version = POLICY_LABEL
    thresholds = {"tolerance": tol}
    if starts.size < 3:
        return _not_evaluated(
            test_id,
            h0,
            alt,
            REVERSE_CHECK,
            version,
            f"{starts.size} starts: at least three are required",
            n=int(starts.size),
            extras={"thresholds": thresholds},
            model_id=model_id,
            segment=segment,
        )
    stat = abs(loss - thr) / thr
    closest = float(np.min(starts))
    closer = bool(np.any(starts < dist * (1.0 - START_TOLERANCE)))
    return TestResult(
        test_id=test_id,
        h0=h0,
        alternative=alt,
        reference=REVERSE_CHECK,
        policy_version=version,
        light=Light.RED if (stat > tol or closer) else Light.GREEN,
        statistic=stat,
        metric=closest / dist - 1.0,
        n=int(starts.size),
        model_id=model_id,
        segment=segment,
        inputs_hash=inputs_hash(loss, thr, dist, starts),
        extras={"thresholds": thresholds, "closer_start": closer},
    )
