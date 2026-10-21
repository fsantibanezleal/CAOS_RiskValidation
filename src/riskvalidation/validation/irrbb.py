"""The IRRBB validation tests (dossier 05 D.7 and the IRRBB ladder acceptance; dossier 01 H.2; dossier 03 F.2): each
returns one `TestResult` that validates against `common/test-result`.

Acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds. Each function states its
light rule in its docstring and records its parameters in `policy_version` and in `extras["thresholds"]`, so that a
view can explain a light. A test that cannot be evaluated returns the light `not_evaluated` with its reason in `notes`
and a null statistic, metric and p-value once serialised, never a fabricated number. The tests are plain functions
that do not read the registry or the default policy: their rules are the contract's (`engine-irrbb-api`, module
validation.irrbb). They state no compliance claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.irrbb import behavioural as _behavioural
from riskvalidation.irrbb import eve_nii as _eve_nii
from riskvalidation.irrbb import shocks as _shocks
from riskvalidation.lgd_ead._checks import mat, same_length, scalar, vec

POLICY_LABEL = "contraste-irrbb-battery 1.0.0"
NAN = float("nan")
EPS = 1e-12             # tolerance on a gate comparison, so that a value exactly on a gate is not moved by rounding

SRP31_92 = (
    "BCBS SRP31.90 to SRP31.92, Interest rate risk in the banking book, the six standardised shock scenarios and the "
    "worked example at t = 3.5 years (dossier 01 H.2), https://www.bis.org/basel_framework/chapter/SRP/31.htm"
)
RAN_21_13 = (
    "CMF Recopilacion Actualizada de Normas, capitulo 21-13, Anexo 1 (Hoja 9 to Hoja 12): CLP and UR shock sizes, the "
    "50 percent offset, the 2.5 percent indexation charge (dossier 03 F.2), https://www.cmfchile.cl"
)
BCBS_368 = (
    "BCBS d368 (2016), Standards: Interest rate risk in the banking book, https://www.bis.org/bcbs/publ/d368.htm; "
    "dossier 05 D.3 (weight stability across estimation windows)"
)
SRP31_82 = (
    "BCBS SRP31.82 to SRP31.83 and SRP31.129: the EVE supervisory outlier test (15 percent of Tier 1) and the "
    "standardised EVE; dossier 05 D.7: always report EVE with and without behavioural assumptions"
)
DOSSIER_05_D7 = ("dossier 05 D.7: flag any change in NMD behavioural parameters that coincides with a limit breach "
                 "(the SVB lesson)")
DOSSIER_05_OOS = "dossier 05, IRRBB acceptance: the 2022 hiking cycle as the out-of-sample test of an NII projection"


def _not_evaluated(test_id: str, h0: str, alternative: str, reference: str, version: str, reason: str, *,
                   n: int | None = None, extras: Mapping[str, Any] | None = None) -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=version,
                      light=Light.NOT_EVALUATED, statistic=NAN, notes=f"not evaluated: {reason}", n=n,
                      extras=dict(extras or {}))


def _three_light(value: float, green: float, amber: float) -> Light:
    """Green at or below `green`, amber at or below `amber`, red above (with a rounding tolerance)."""
    if value <= green + EPS:
        return Light.GREEN
    return Light.AMBER if value <= amber + EPS else Light.RED


def _check_gates(low: float, high: float, low_name: str, high_name: str) -> None:
    if not (0.0 <= low <= high):
        raise ValueError(
            f"{low_name} and {high_name} must satisfy 0 <= {low_name} <= {high_name}, got {low} and {high}")


# ----------------------------------------------------------------------------------------------------------------
# The goldens


@register(
    "irrbb.srp31_92_goldens",
    family="irrbb",
    name="SRP31.92 worked example goldens",
    h0="the engine reproduces the SRP31.92 worked example within the printed precision",
    reference=SRP31_92,
    decision="metric",
)
def srp31_92_goldens(tolerance_scalar: float = 5e-4, tolerance_bp: float = 5e-2) -> TestResult:
    """Reproduces the SRP31.92 worked example (dossier 01 H.2): at t = 3.5 years alpha_short = 0.417, a 100 bp short
    shock +41.7 bp, the steepener +25.4 bp and the flattener -1.6 bp (with the short and long sizes both 100 bp).

    statistic = the largest absolute deviation of the four values from the printed ones, each in its own unit (the
    scalar in its own, the three shocks in basis points); p_value None. Light (a Contraste internal gate): green when
    every deviation is within its tolerance, red otherwise; `extras["thresholds"]` holds the two tolerances. At the
    engine release the statistic is the flattener's deviation, 0.0393 bp (-1.6393 against the printed -1.6)."""
    test_id = "irrbb.srp31_92_goldens"
    h0 = "the engine reproduces the SRP31.92 worked example within the printed precision"
    alt = "the engine departs from the printed values"
    tol_s, tol_b = scalar(tolerance_scalar, "tolerance_scalar"), scalar(tolerance_bp, "tolerance_bp")
    if tol_s < 0.0 or tol_b < 0.0:
        raise ValueError("tolerances must be nonnegative")
    t = np.array([3.5])
    a_short, _ = _shocks.shock_scalars(t, 4.0)
    got = {
        "alpha_short": (float(a_short[0]), 0.417, tol_s),
        "short-up": (float(_shocks.shock_curve("short-up", 100, 100, 100, t)[0]), 41.7, tol_b),
        "steepener": (float(_shocks.shock_curve("steepener", 100, 100, 100, t)[0]), 25.4, tol_b),
        "flattener": (float(_shocks.shock_curve("flattener", 100, 100, 100, t)[0]), -1.6, tol_b),
    }
    dev = {k: abs(v - printed) for k, (v, printed, _) in got.items()}
    ok = all(dev[k] <= got[k][2] + EPS for k in got)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=SRP31_92, policy_version=POLICY_LABEL,
        light=Light.GREEN if ok else Light.RED, statistic=max(dev.values()), n=len(got),
        extras={"thresholds": {"tolerance_scalar": tol_s, "tolerance_bp": tol_b},
                "values": {k: v for k, (v, _, _) in got.items()}, "printed": {k: p for k, (_, p, _) in got.items()},
                "deviations": dev},
        notes="" if ok else "a value departs from the printed SRP31.92 precision")


@register(
    "irrbb.cmf_annex1_goldens",
    family="irrbb",
    name="CMF RAN 21-13 Annex 1 goldens",
    h0="the engine reproduces the RAN 21-13 Annex 1 formulas and table exactly",
    reference=RAN_21_13,
    decision="metric",
)
def cmf_annex1_goldens() -> TestResult:
    """Reproduces the Annex 1 formulas (dossier 03 F.2): the `cmf_pair` table on both signs (8 cases), the Hoja 9
    table through `shock_table("cmf-annex1")` (5 rows), the indexation charge (`indexation_charge(-200) = 5`) and the
    `annex1_delta_eve` golden (its 6 pairs, the measure and the worst shock: 3 cases) and the worked example of Annex 1
    Hoja 10 (`alpha_short(3.5) = 0.417` and `+41.7` bp for a 100 bp short shock, at the precision they are printed: 2
    cases).

    statistic = the number of failed cases (an exception counts as a failed case); p_value None. Light (a Contraste
    internal gate): green when every case is exact (floating-point equality to 1e-12; the two Hoja 10 values at their
    printed precision, 5e-4 and 5e-2 bp), red otherwise."""
    test_id = "irrbb.cmf_annex1_goldens"
    h0 = "the engine reproduces the RAN 21-13 Annex 1 formulas and table exactly"
    alt = "the engine departs from the Annex 1 formulas or table"
    pair_table = [(10, 20, 30), (-10, -20, 0), (30, -20, 20), (-20, 30, 20), (10, -40, 0), (-40, 10, 0), (0, 25, 25),
                  (25, 0, 25)]
    hoja9 = {"canasta-1": (200, 300, 150), "canasta-2": (400, 500, 300), "CLP": (300, 400, 200),
             "UR-nii": (300, None, None), "UR-eve": (200, 250, 150)}

    def run(name: str, check: Any) -> tuple[str, bool]:
        try:
            return name, bool(check())
        except Exception:  # a function that raises has failed its golden
            return name, False

    cases = [run(f"cmf_pair({a}, {b}) = {p}", lambda a=a, b=b, p=p: abs(_eve_nii.cmf_pair(a, b) - p) <= 1e-12)
             for a, b, p in pair_table]
    cases += [run(f"Hoja 9 {k}", lambda k=k, v=v: tuple(_shocks.shock_table("cmf-annex1")[k]) == v
                  and list(_shocks.shock_table("cmf-annex1")) == list(hoja9)) for k, v in hoja9.items()]
    cases.append(run("indexation_charge(-200) = 5", lambda: abs(_eve_nii.indexation_charge(-200.0) - 5.0) <= 1e-12))
    eve_clp, eve_ur = np.array([10, -20, 0, 5, -30, 40.0]), np.array([20, 30, -10, -2, 50, -15.0])

    def annex1():  # computed inside each case, so a function that raises fails the three cases instead of escaping
        return _eve_nii.annex1_delta_eve(eve_clp, eve_ur)

    cases.append(run("annex1 pairs", lambda: np.allclose(annex1().pairs, [30, 20, 0, 4, 35, 32.5], atol=1e-12, rtol=0)))
    cases.append(run("annex1 delta_eve = 35", lambda: abs(annex1().delta_eve - 35.0) <= 1e-12))
    cases.append(run("annex1 worst shock = short-up", lambda: annex1().worst_shock == "short-up"))
    # the worked example of Annex 1 Hoja 10 (dossier 03 G3), at the precision it is printed: 0.417 and +41.7 bp
    t35 = np.array([3.5])
    cases.append(run("Hoja 10 alpha_short(3.5) = 0.417",
                     lambda: abs(float(_shocks.shock_scalars(t35)[0][0]) - 0.417) <= 5e-4))
    cases.append(run("Hoja 10 short-up 100 bp at 3.5 = 41.7 bp",
                     lambda: abs(float(_shocks.shock_curve("short-up", 100, 100, 100, t35)[0]) - 41.7) <= 5e-2))
    failed = [name for name, ok in cases if not ok]
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=RAN_21_13, policy_version=POLICY_LABEL,
        light=Light.RED if failed else Light.GREEN, statistic=float(len(failed)), n=len(cases),
        n_events=len(failed),
        extras={"failed": failed,
                "thresholds": {"tolerance": 1e-12, "tolerance_printed": {"alpha_short": 5e-4, "bp": 5e-2}}},
        notes=f"failed: {'; '.join(failed)}" if failed else "")


# ----------------------------------------------------------------------------------------------------------------
# Reconciliation and stability


@register(
    "irrbb.eve_sum_of_parts",
    family="irrbb",
    name="EVE sum of parts",
    h0="the EVE of the balance sheet equals the sum of the EVEs of its items",
    reference=SRP31_82,
    decision="metric",
)
def eve_sum_of_parts(item_flows: np.ndarray, rates: np.ndarray, midpoints: np.ndarray, whole_eve: float,
                     rel_tol: float = 1e-9) -> TestResult:
    """The EVE of a balance sheet equals the sum of the EVEs of its items: `sum_i eve(item_i) = whole_eve`, where
    `whole_eve` is the EVE of the flows aggregated by the caller (`item_flows` is (I, K), one row per item).

    statistic = the absolute residual, metric = the relative residual `|residual| / |whole_eve|` (against the larger
    of the two EVEs when `whole_eve` is zero; zero when both are zero). Light (a Contraste internal gate): green when
    the relative residual is at most `rel_tol`, red above; `extras["thresholds"] = {"rel_tol": rel_tol}`. The identity
    is exact for the
    aggregated flows: a `whole_eve` rounded to six decimals needs `rel_tol` of 1e-6 or more."""
    test_id = "irrbb.eve_sum_of_parts"
    h0 = "the EVE of the balance sheet equals the sum of the EVEs of its items"
    alt = "the aggregated EVE differs from the sum of its parts"
    items, r, t = mat(item_flows, "item_flows"), vec(rates, "rates"), vec(midpoints, "midpoints")
    whole = scalar(whole_eve, "whole_eve")
    tol = scalar(rel_tol, "rel_tol")
    if tol < 0.0:
        raise ValueError("rel_tol must be nonnegative")
    if items.shape[1] != r.size or r.size != t.size:
        raise ValueError(f"item_flows has {items.shape[1]} columns, rates {r.size} and midpoints {t.size}")
    parts = float(sum(_eve_nii.eve(row, r, t) for row in items))
    resid = abs(parts - whole)
    scale = abs(whole) if whole != 0.0 else abs(parts)
    rel = 0.0 if resid == 0.0 else resid / scale
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=SRP31_82, policy_version=POLICY_LABEL,
        light=Light.GREEN if rel <= tol else Light.RED, statistic=resid, metric=rel, n=int(items.shape[0]),
        inputs_hash=inputs_hash(items, r, t, whole), extras={"thresholds": {"rel_tol": tol}, "sum_of_parts": parts,
                                                              "whole_eve": whole})


@register(
    "irrbb.nmd_parameter_stability",
    family="irrbb",
    name="NMD parameter stability",
    h0="the replicating-portfolio weights are stable across estimation windows",
    reference=BCBS_368,
    decision="metric",
)
def nmd_parameter_stability(window_weights: np.ndarray, maturities: np.ndarray, gap_amber: float = 0.10,
                            gap_red: float = 0.25) -> TestResult:
    """The stability of the replicating-portfolio weights re-estimated on W estimation windows (dossier 05 D.3).

    statistic = the largest sup-norm distance `max_j |w_aj - w_bj|` between consecutive windows, metric = the largest
    change of the weighted average maturity between consecutive windows, in years. Light (a Contraste internal gate):
    green when the statistic is at most `gap_amber`, amber when at most `gap_red`, red above; fewer than two windows:
    not_evaluated."""
    test_id = "irrbb.nmd_parameter_stability"
    h0 = "the replicating-portfolio weights are stable across estimation windows"
    alt = "the weights move between consecutive windows by more than the gap"
    _check_gates(scalar(gap_amber, "gap_amber"), scalar(gap_red, "gap_red"), "gap_amber", "gap_red")
    w = mat(window_weights, "window_weights")
    m = vec(maturities, "maturities")
    if w.shape[1] != m.size:
        raise ValueError(f"window_weights has {w.shape[1]} columns for {m.size} maturities")
    version = POLICY_LABEL
    thresholds = {"gap_amber": gap_amber, "gap_red": gap_red}
    if w.shape[0] < 2:
        return _not_evaluated(test_id, h0, alt, BCBS_368, version, "fewer than 2 estimation windows", n=int(w.shape[0]),
                              extras={"thresholds": thresholds})
    dist = np.abs(np.diff(w, axis=0)).max(axis=1)
    wam = np.array([_behavioural.weighted_average_maturity(m, row) for row in w])
    stat = float(dist.max())
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=BCBS_368, policy_version=version,
        light=_three_light(stat, gap_amber, gap_red), statistic=stat, metric=float(np.abs(np.diff(wam)).max()),
        n=int(w.shape[0]), inputs_hash=inputs_hash(w, m),
        extras={"thresholds": thresholds, "distances": dist.tolist(), "weighted_average_maturity": wam.tolist()})


@register(
    "irrbb.eve_with_without_behavioural",
    family="irrbb",
    name="EVE with without behavioural",
    h0="the behavioural assumptions do not change the EVE risk measure materially and hide no breach",
    reference=SRP31_82,
    decision="metric",
)
def eve_with_without_behavioural(measure_behavioural: float, measure_contractual: float, tier1: float,
                                 diff_amber: float = 0.05, diff_red: float = 0.15,
                                 threshold: float = 0.15) -> TestResult:
    """The two EVE risk measures of the same shocks, with and without the behavioural assumptions, always reported
    together (dossier 05 D.7). statistic = `(contractual - behavioural) / tier1`, metric = `contractual - behavioural`
    in money.

    Light (a Contraste internal gate): green when `|statistic| <= diff_amber`, amber when at most `diff_red`, red
    above; and red whatever the size when the behavioural measure passes the outlier test (ratio to Tier 1 not above
    `threshold`) and the contractual one breaches it: the behavioural assumptions hide a breach (the SVB lesson).
    `extras["thresholds"]` holds `diff_amber`, `diff_red`, `threshold` and both ratios."""
    test_id = "irrbb.eve_with_without_behavioural"
    h0 = "the behavioural assumptions do not change the EVE risk measure materially and hide no breach"
    alt = "the behavioural assumptions move the EVE risk measure materially or hide a breach"
    beh, con = scalar(measure_behavioural, "measure_behavioural"), scalar(measure_contractual, "measure_contractual")
    t1, th = scalar(tier1, "tier1"), scalar(threshold, "threshold")
    if t1 <= 0.0:
        raise ValueError("tier1 must be positive")
    _check_gates(scalar(diff_amber, "diff_amber"), scalar(diff_red, "diff_red"), "diff_amber", "diff_red")
    ob, oc = _eve_nii.outlier_test(beh, t1, th), _eve_nii.outlier_test(con, t1, th)
    stat = (con - beh) / t1
    hidden = (not ob.breached) and oc.breached
    light = Light.RED if hidden else _three_light(abs(stat), diff_amber, diff_red)
    thresholds = {"diff_amber": diff_amber, "diff_red": diff_red, "threshold": th, "ratio_behavioural": ob.ratio,
                  "ratio_contractual": oc.ratio}
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=SRP31_82,
        policy_version=POLICY_LABEL, light=light, statistic=stat,
        metric=con - beh, n=2, inputs_hash=inputs_hash(np.array([beh, con, t1])),
        extras={"thresholds": thresholds, "hidden_breach": hidden},
        notes="the behavioural assumptions hide an outlier breach" if hidden else "")


@register(
    "irrbb.behavioural_change_at_breach",
    family="irrbb",
    name="Behavioural change at breach",
    h0="no change of a behavioural parameter coincides with a limit breach",
    reference=DOSSIER_05_D7,
    decision="metric",
)
def behavioural_change_at_breach(values: np.ndarray, dates: np.ndarray, breach: np.ndarray) -> TestResult:
    """Flags any change of a behavioural parameter (a core share, a core maturity, a beta) that coincides with a limit
    breach (dossier 05 D.7). `values` (T,), `dates` (T,) `datetime64[D]` strictly increasing, `breach` (T,) booleans,
    true at a date where a limit was breached in the window ending there; a change is `values[t] != values[t - 1]`.

    statistic = the number of flagged changes (a change at a date with `breach[t]` true), n = the number of dates,
    n_events = the same count. Light (a Contraste internal gate): red when any change coincides with a breach, green
    otherwise; fewer than two dates: not_evaluated."""
    test_id = "irrbb.behavioural_change_at_breach"
    h0 = "no change of a behavioural parameter coincides with a limit breach"
    alt = "a behavioural parameter changed at a date where a limit was breached"
    v = vec(values, "values")
    d = np.asarray(dates)
    b = np.asarray(breach)
    if d.ndim != 1 or d.dtype.kind != "M":
        raise ValueError("dates must be a one-dimensional datetime64 array")
    if b.ndim != 1 or b.dtype.kind != "b":
        raise ValueError("breach must be a one-dimensional boolean array")
    d = d.astype("datetime64[D]")
    same_length(values=v, dates=d.astype(float), breach=b.astype(float))
    if np.any(np.diff(d.astype("int64")) <= 0):
        raise ValueError("dates must be strictly increasing")
    version = POLICY_LABEL
    if v.size < 2:
        return _not_evaluated(test_id, h0, alt, DOSSIER_05_D7, version, "fewer than 2 dates", n=int(v.size),
                              extras={"thresholds": {}})
    flagged = (v[1:] != v[:-1]) & b[1:]
    count = int(flagged.sum())
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=DOSSIER_05_D7, policy_version=version,
        light=Light.RED if count else Light.GREEN, statistic=float(count), n=int(v.size), n_events=count,
        inputs_hash=inputs_hash(v, d.astype("int64"), b),
        extras={"thresholds": {}, "flagged_dates": [str(x) for x in d[1:][flagged]]},
        notes=f"{count} change(s) coincide with a breach" if count else "")


@register(
    "irrbb.nii_out_of_sample",
    family="irrbb",
    name="NII out of sample",
    h0="the projected change of NII matches the realised change",
    reference=DOSSIER_05_OOS,
    decision="metric",
)
def nii_out_of_sample(projected_delta_nii: float, realised_delta_nii: float | None, err_green: float = 0.10,
                      err_amber: float = 0.25) -> TestResult:
    """The projected change of NII under the scenario against the realised change over the 2022 hiking cycle (dossier
    05, IRRBB acceptance). Money in one currency, loss-positive. statistic = the relative error
    `|projected - realised| / |realised|`, metric = `projected - realised` in money.

    Light (a Contraste internal gate): green when the relative error is at most `err_green`, amber when at most
    `err_amber`, red above; not_evaluated with the reason in `notes` when `realised_delta_nii` is None or zero;
    `extras["thresholds"]` holds both."""
    test_id = "irrbb.nii_out_of_sample"
    h0 = "the projected change of NII matches the realised change"
    alt = "the projected change of NII departs from the realised change"
    proj = scalar(projected_delta_nii, "projected_delta_nii")
    _check_gates(scalar(err_green, "err_green"), scalar(err_amber, "err_amber"), "err_green", "err_amber")
    version = POLICY_LABEL
    thresholds = {"err_green": err_green, "err_amber": err_amber}
    if realised_delta_nii is None:
        return _not_evaluated(test_id, h0, alt, DOSSIER_05_OOS, version, "the realised change in NII is not available",
                              extras={"thresholds": thresholds})
    real = scalar(realised_delta_nii, "realised_delta_nii")
    if real == 0.0:
        return _not_evaluated(test_id, h0, alt, DOSSIER_05_OOS, version,
                              "the realised change in NII is zero: a relative error is undefined",
                              extras={"thresholds": thresholds})
    stat = abs(proj - real) / abs(real)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=DOSSIER_05_OOS, policy_version=version,
        light=_three_light(stat, err_green, err_amber), statistic=stat, metric=proj - real, n=1,
        inputs_hash=inputs_hash(np.array([proj, real])), extras={"thresholds": thresholds})

__all__ = [
    "behavioural_change_at_breach",
    "cmf_annex1_goldens",
    "eve_sum_of_parts",
    "eve_with_without_behavioural",
    "nii_out_of_sample",
    "nmd_parameter_stability",
    "srp31_92_goldens",
]
