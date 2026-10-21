"""The IFRS 9 validation battery of dossier 02 I.2: each test returns one `TestResult`.

Acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds. Each function states its light
rule in its docstring and records its parameters in `policy_version` (and `alpha_amber`, `alpha_red` where a p-value or
a band is used), so that a view can explain a light. A test that cannot be evaluated returns the light `not_evaluated`
with its reason in `notes`; its statistic is NaN (undefined), never a fabricated number.

The tests are plain functions that do not read the registry or the default policy: their rules are the contract's
(`engine-provisions-api`, module ifrs9). They state no compliance claim and are not IFRS 7 disclosures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.harness.rates import wilson_interval
from riskvalidation.provisions import reconciliation as _rc
from riskvalidation.provisions._checks import months_array
from riskvalidation.provisions.ecl import non_linearity_index, probability_weighted_ecl, weight_sensitivity
from riskvalidation.provisions.overlays import TARGETS, Overlay, check_unique
from riskvalidation.provisions.reconciliation import RollForwardRow
from riskvalidation.provisions.term_structure import gauss_hermite_pit_pd, ttc_to_pit, weighted_pit_pd
from riskvalidation.validation._common import check_pd as _check_pd
from riskvalidation.validation._policy import battery_label
from riskvalidation.validation.calibration import default_count_sf
from riskvalidation.validation.discrimination import auc_with_variance

POLICY_LABEL = "contraste-ifrs9-battery 1.0.0"
NAN = float("nan")
Z95 = 1.959963984540054

EBA_2023 = (
    "EBA (2023), IFRS 9 implementation by EU institutions, 2023 Monitoring report, EBA/Rep/2023/36, "
    "doi:10.2853/269836"
)
D350 = "BCBS (2015), Guidance on credit risk and accounting for expected credit losses (d350)"
GPPC = (
    "GPPC (2016), The implementation of IFRS 9 impairment requirements by banks, 2016-06-17"
)
ECB_2024 = "ECB Banking Supervision (2024), IFRS 9 overlays and model improvements for novel risks, doi:10.2866/034127"
VASICEK = "Vasicek (2002), The distribution of loan portfolio value, Risk 15(12)"


def _not_evaluated(test_id: str, h0: str, alternative: str, reference: str, version: str, reason: str, *,
                   use_statistic: bool = False, model_id: str | None = None, segment: str | None = None,
                   n: int | None = None, alpha_amber: float | None = None, alpha_red: float | None = None,
                   extras: Mapping[str, Any] | None = None) -> TestResult:
    return TestResult(
        test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=version,
        light=Light.NOT_EVALUATED, statistic=NAN if use_statistic else None, metric=None if use_statistic else NAN,
        notes=f"not evaluated: {reason}", model_id=model_id, segment=segment, n=n, alpha_amber=alpha_amber,
        alpha_red=alpha_red, extras=dict(extras or {}),
    )


def _floats(x: Any, name: str, allow_nan: bool = False) -> np.ndarray:
    a = np.array(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    bad = ~np.isfinite(a) if not allow_nan else np.isinf(a)
    if np.any(bad):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _same_length(**arrays: np.ndarray) -> None:
    sizes = {k: v.shape[0] for k, v in arrays.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"{', '.join(sizes)} must have the same length, got {sizes}")


def _bools(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype != bool or a.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional boolean array")
    return a


# ----------------------------------------------------------------------------------------------------------------
# lifetime PD backtest

LIFETIME_H0 = "the observed cumulative default rate equals the predicted cumulative PD at every horizon"


def _mixture_quantile(q: float, n: int, pd: float, rho: float) -> int:
    """The smallest default count k with P(D <= k) >= q, D the defaults among n obligors with PD `pd` and asset
    correlation `rho` (binomial given the systematic factor, Vasicek's one-factor model)."""
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if 1.0 - default_count_sf(mid + 1, n, pd, rho) >= q:
            hi = mid
        else:
            lo = mid + 1
    return lo


@register(
    "ifrs9.lifetime_pd_backtest",
    family="ifrs9",
    name="Lifetime PD backtest",
    h0=LIFETIME_H0,
    reference=(
        f"{EBA_2023}, paras 114 to 118; {VASICEK}; Kaplan-Meier product-limit and Aalen-Johansen cumulative incidence "
        "estimators (dossier 02 B.2, C.2)"),
    decision="metric",
)
def lifetime_pd_backtest(months: np.ndarray, predicted_cumulative: np.ndarray, observed_cumulative: np.ndarray,
                         n_at_risk: np.ndarray, n_events: np.ndarray, rho: float | None = None,
                         competing_prepayment: bool = False,
                         model_id: str | None = None, segment: str | None = None, *,
                         gap_amber: float = 0.25, gap_red: float = 0.50,
                         alpha_amber: float = 0.05, alpha_red: float = 0.01) -> TestResult:
    """Observed cumulative default against the predicted cumulative PD by horizon (EBA/Rep/2023/36 paras 114 to 118).

    `observed_cumulative` is the Kaplan-Meier estimate, or the Aalen-Johansen cumulative incidence when
    `competing_prepayment` is true (the prediction then comes from `competing_risks`); NaN marks a horizon not yet
    observed. `n_at_risk` is the cohort size at each horizon and `n_events` the defaults observed up to it.

    metric: the largest `|observed - predicted|` over the observed horizons. Light: with `rho`, red if any observed
    horizon lies outside the two-sided Vasicek band at `alpha_red`, amber if outside at `alpha_amber`, else green (the
    band is the central interval of the default count among `n_at_risk` obligors with the PD `predicted` and asset
    correlation `rho`, as a rate). Without `rho`, the relative gap `|observed - predicted| / predicted` of the worst
    observed horizon: green below `gap_amber`, amber below `gap_red`, red otherwise. Contraste internal gates.
    """
    reference = (
        f"{EBA_2023}, paras 114 to 118; {VASICEK}; Kaplan-Meier product-limit and Aalen-Johansen cumulative incidence "
        "estimators (dossier 02 B.2, C.2)"
    )
    version = battery_label(
            POLICY_LABEL, "ifrs9.lifetime_pd_backtest", gap_amber=gap_amber, gap_red=gap_red,
            alpha_amber=alpha_amber, alpha_red=alpha_red)
    m = months_array(months)
    pred = _floats(predicted_cumulative, "predicted_cumulative")
    obs = _floats(observed_cumulative, "observed_cumulative", allow_nan=True)
    nar = _floats(n_at_risk, "n_at_risk")
    nev = _floats(n_events, "n_events")
    _same_length(months=m, predicted_cumulative=pred, observed_cumulative=obs, n_at_risk=nar, n_events=nev)
    _check_pd(pred, "predicted_cumulative")
    if np.any(obs[~np.isnan(obs)] < 0.0) or np.any(obs[~np.isnan(obs)] > 1.0):
        raise ValueError("observed_cumulative must lie in [0, 1] (NaN where the horizon is not observed)")
    if np.any(nar < 0.0) or np.any(nev < 0.0):
        raise ValueError("n_at_risk and n_events must be non-negative")
    if rho is not None and not 0.0 < float(rho) < 1.0:
        raise ValueError(f"rho must lie strictly between 0 and 1, got {rho!r}")
    estimator = "aalen-johansen" if competing_prepayment else "kaplan-meier"
    seen = ~np.isnan(obs)
    id_hash = inputs_hash(m, pred, obs, nar, nev, rho, competing_prepayment)
    if not seen.any():
        return _not_evaluated("ifrs9.lifetime_pd_backtest", LIFETIME_H0, "observed differs from predicted", reference,
                              version, "no horizon is observed yet", model_id=model_id, segment=segment,
                              extras={"estimator": estimator})
    if np.any(nar[seen] < 1.0):
        raise ValueError("n_at_risk must be at least 1 at every observed horizon")
    if np.any(nev[seen] > nar[seen]):
        raise ValueError("n_events must not exceed n_at_risk at an observed horizon")
    p, o, n_h = pred[seen], obs[seen], np.rint(nar[seen]).astype(np.int64)
    gap = np.abs(o - p)
    brier = o * (1.0 - p) ** 2 + (1.0 - o) * p**2
    extras: dict[str, Any] = {
        "estimator": estimator, "months": np.asarray(m)[seen].tolist(), "gap": gap.tolist(), "brier": brier.tolist(),
        "signed_gap": (o - p).tolist(), "horizons_excluded": np.asarray(m)[~seen].tolist(),
        "predicted": p.tolist(), "observed": o.tolist(), "rho": None if rho is None else float(rho),
    }
    if rho is not None:
        r = float(rho)
        bands = {}
        for tag, alpha in (("", alpha_amber), ("_red", alpha_red)):
            lows, highs = [], []
            for pd_h, n in zip(p, n_h, strict=True):
                lows.append(_mixture_quantile(alpha / 2.0, int(n), float(pd_h), r) / int(n))
                highs.append(_mixture_quantile(1.0 - alpha / 2.0, int(n), float(pd_h), r) / int(n))
            bands[tag] = (np.array(lows), np.array(highs))
        extras.update({"band_low": bands[""][0].tolist(), "band_high": bands[""][1].tolist(),
                       "band_low_red": bands["_red"][0].tolist(), "band_high_red": bands["_red"][1].tolist()})
        outside_red = np.any((o < bands["_red"][0]) | (o > bands["_red"][1]))
        outside_amber = np.any((o < bands[""][0]) | (o > bands[""][1]))
        light = Light.RED if outside_red else Light.AMBER if outside_amber else Light.GREEN
    else:
        worst = float(np.max(gap / p))
        extras["relative_gap"] = (gap / p).tolist()
        light = Light.GREEN if worst < gap_amber else Light.AMBER if worst < gap_red else Light.RED
    return TestResult(
        test_id="ifrs9.lifetime_pd_backtest", h0=LIFETIME_H0, alternative="observed differs from predicted",
        reference=reference, policy_version=version, light=light, metric=float(np.max(gap)),
        alpha_amber=alpha_amber if rho is not None else None, alpha_red=alpha_red if rho is not None else None,
        n=int(np.max(n_h)), n_events=int(np.max(nev[seen])), model_id=model_id, segment=segment,
        inputs_hash=id_hash, extras=extras,
    )


# ----------------------------------------------------------------------------------------------------------------
# SICR effectiveness and threshold stability

SICR_H0 = "Stage 2 accounts default at the same rate as Stage 1 accounts (the SICR flag carries no information)"


def _wilson(k: int, n: int) -> list[float] | None:
    return list(wilson_interval(int(k), int(n))) if n > 0 else None


@register(
    "ifrs9.sicr_effectiveness",
    family="ifrs9",
    name="SICR effectiveness",
    h0=SICR_H0,
    reference=f"{EBA_2023}, paras 94 to 99 and Figure 42; {D350}, A38; dossier 02 I.2",
    decision="metric",
)
def sicr_effectiveness(stage_at_origination_view: np.ndarray, defaulted_within_k: np.ndarray, k_months: int,
                       months_in_stage2_before_default: np.ndarray, stage3_from_stage1: np.ndarray,
                       backstop_only: np.ndarray, model_id: str | None = None, *,
                       green_ratio: float = 2.0, amber_ratio: float = 1.0, direct_share_amber: float = 0.5,
                       min_months: int = 1) -> TestResult:
    """Does the Stage 2 flag select the accounts that default (EBA/Rep/2023/36 paras 94 to 99; d350 A38)?

    `stage_at_origination_view` is the stage (1 or 2) at the reference date, `defaulted_within_k` the default within
    `k_months`. metric = DR_S2 / DR_S1 with a log-scale interval (Katz). extras: `roc_auc` (the stage as score, ties one
    half), `hit_rate` (the share of defaulters at least `min_months` months in Stage 2 before default),
    `share_stage3_direct_from_stage1` (of the defaulters), `mean_months_in_stage2` (of the defaulters that were in
    Stage 2), `share_stage2_backstop_only` (of Stage 2), each with its interval where defined.

    Light: green when metric >= `green_ratio`, amber when `amber_ratio` <= metric < `green_ratio`, red below
    `amber_ratio`; amber at least when the direct share exceeds `direct_share_amber`. No Stage 1 defaults, no Stage 1
    or no Stage 2 accounts: not_evaluated. Contraste internal gates.
    """
    reference = f"{EBA_2023}, paras 94 to 99 and Figure 42; {D350}, A38; dossier 02 I.2"
    version = battery_label(
            POLICY_LABEL, "ifrs9.sicr_effectiveness", green_ratio=green_ratio, amber_ratio=amber_ratio,
            direct_share_amber=direct_share_amber)
    stage = _floats(stage_at_origination_view, "stage_at_origination_view")
    dflt = _bools(defaulted_within_k, "defaulted_within_k")
    months = _floats(months_in_stage2_before_default, "months_in_stage2_before_default", allow_nan=True)
    direct = _bools(stage3_from_stage1, "stage3_from_stage1")
    backstop = _bools(backstop_only, "backstop_only")
    _same_length(stage_at_origination_view=stage, defaulted_within_k=dflt, months_in_stage2_before_default=months,
                 stage3_from_stage1=direct, backstop_only=backstop)
    if int(k_months) < 1:
        raise ValueError(f"k_months must be at least 1, got {k_months!r}")
    if np.any(~np.isin(stage, (1.0, 2.0))):
        raise ValueError("stage_at_origination_view must hold stage 1 or 2 only")
    if np.any(np.isnan(months[dflt])) or np.any(months[dflt] < 0.0):
        raise ValueError("months_in_stage2_before_default must be finite and non-negative for the defaulted accounts")
    s1, s2 = stage == 1.0, stage == 2.0
    n1, n2, d1, d2 = int(s1.sum()), int(s2.sum()), int((dflt & s1).sum()), int((dflt & s2).sum())
    n, d = int(stage.size), int(dflt.sum())
    h0, alt = SICR_H0, "Stage 2 defaults more often than Stage 1"
    common = {"model_id": model_id, "n": n}
    if n1 == 0 or n2 == 0:
        return _not_evaluated("ifrs9.sicr_effectiveness", h0, alt, reference, version,
                              "the default-rate ratio needs accounts in both Stage 1 and Stage 2", **common)
    if d1 == 0:
        return _not_evaluated("ifrs9.sicr_effectiveness", h0, alt, reference, version,
                              "no Stage 1 account defaulted, so DR_S1 is zero and the ratio is undefined", **common)
    ratio = (d2 / n2) / (d1 / n1)
    c = 0.5 if d2 == 0 else 0.0  # continuity correction for the interval only
    se = math.sqrt(1.0 / (d2 + c) - 1.0 / n2 + 1.0 / d1 - 1.0 / n1)
    centre = math.log((d2 + c) / n2 / (d1 / n1))
    interval = [0.0 if d2 == 0 else math.exp(centre - Z95 * se), math.exp(centre + Z95 * se)]
    extras: dict[str, Any] = {
        "metric_interval": interval, "default_rate_stage1": d1 / n1, "default_rate_stage2": d2 / n2,
        "k_months": int(k_months), "n_stage1": n1, "n_stage2": n2, "defaults_stage1": d1, "defaults_stage2": d2,
    }
    if 0 < d < n:
        auc, var, _, _ = auc_with_variance(stage, dflt.astype(int))
        half = Z95 * math.sqrt(var)
        extras["roc_auc"] = auc
        extras["roc_auc_interval"] = [max(0.0, auc - half), min(1.0, auc + half)]
    else:
        extras["roc_auc"], extras["roc_auc_interval"] = None, None
    hits = int((dflt & (months >= min_months)).sum()) if d else 0
    extras["hit_rate"] = hits / d
    extras["hit_rate_interval"] = _wilson(hits, d)
    direct_k = int((direct & dflt).sum())
    share_direct = direct_k / d
    extras["share_stage3_direct_from_stage1"] = share_direct
    extras["share_stage3_direct_from_stage1_interval"] = _wilson(direct_k, d)
    in_s2 = months[dflt & (months > 0.0)]
    if in_s2.size:
        mean = float(np.mean(in_s2))
        half = Z95 * float(np.std(in_s2, ddof=1)) / math.sqrt(in_s2.size) if in_s2.size > 1 else None
        extras["mean_months_in_stage2"] = mean
        extras["mean_months_in_stage2_interval"] = None if half is None else [mean - half, mean + half]
    else:
        extras["mean_months_in_stage2"], extras["mean_months_in_stage2_interval"] = None, None
    only = int((backstop & s2).sum())
    extras["share_stage2_backstop_only"] = only / n2
    extras["min_months"] = int(min_months)
    extras["share_stage2_backstop_only_interval"] = _wilson(only, n2)
    light = Light.GREEN if ratio >= green_ratio else Light.AMBER if ratio >= amber_ratio else Light.RED
    if share_direct > direct_share_amber and light is Light.GREEN:
        light = Light.AMBER
    return TestResult(
        test_id="ifrs9.sicr_effectiveness", h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=light, metric=ratio, n=n, n_events=d, model_id=model_id,
        inputs_hash=inputs_hash(stage, dflt, int(k_months), months, direct, backstop), extras=extras,
    )


@register(
    "ifrs9.sicr_threshold_stability",
    family="ifrs9",
    name="SICR threshold stability",
    h0="the SICR threshold does not loosen in the periods in which conditions worsen",
    reference=(
        f"{EBA_2023}, paras 29 to 33; EBA (2021), IFRS 9 implementation by EU institutions, Monitoring report, "
        "EBA/Rep/2021/35, doi:10.2853/631076"),
    decision="metric",
)
def sicr_threshold_stability(implied_multiplier: np.ndarray, condition_index: np.ndarray,
                             model_id: str | None = None) -> TestResult:
    """Thresholds must not loosen when conditions worsen (EBA/Rep/2023/36 paras 29 to 33; EBA/Rep/2021/35).

    metric: the number of reporting periods in which the implied multiplier rises (the threshold loosens) while the
    condition index rises (conditions worsen). Light: 0 green; 1 or more amber; red when the count is at least half of
    the periods (the length of the arrays). Contraste internal gates.
    """
    reference = (
        f"{EBA_2023}, paras 29 to 33; EBA (2021), IFRS 9 implementation by EU institutions, Monitoring report, "
        "EBA/Rep/2021/35, doi:10.2853/631076"
    )
    version = battery_label(POLICY_LABEL, "ifrs9.sicr_threshold_stability", amber_count=1, red_share_of_periods=0.5)
    h0 = "the SICR threshold does not loosen in the periods in which conditions worsen"
    alt = "the multiplier rises as the condition index rises"
    mult = _floats(implied_multiplier, "implied_multiplier")
    cond = _floats(condition_index, "condition_index")
    _same_length(implied_multiplier=mult, condition_index=cond)
    if np.any(mult < 1.0):
        raise ValueError("implied_multiplier must be at least 1")
    p = int(mult.size)
    if p < 2:
        return _not_evaluated("ifrs9.sicr_threshold_stability", h0, alt, reference, version,
                              "at least two reporting periods are needed", model_id=model_id, n=p)
    loosening = np.flatnonzero((np.diff(mult) > 0.0) & (np.diff(cond) > 0.0)) + 1
    count = int(loosening.size)
    light = Light.GREEN if count == 0 else Light.RED if count >= p / 2.0 else Light.AMBER
    return TestResult(
        test_id="ifrs9.sicr_threshold_stability", h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=light, metric=float(count), n=p, model_id=model_id, inputs_hash=inputs_hash(mult, cond),
        extras={"loosening_periods": loosening.tolist(), "periods": p},
    )


# ----------------------------------------------------------------------------------------------------------------
# stage migration and ECL backtests


@register(
    "ifrs9.stage_migration_backtest",
    family="ifrs9",
    name="Stage migration backtest",
    h0="the observed stage transitions are drawn from the predicted transition matrix",
    reference=f"{EBA_2023}, para 97; Anderson and Goodman (1957), doi:10.1214/aoms/1177707039, equation (3.1)",
    decision="p_value",
)
def stage_migration_backtest(predicted_matrix: np.ndarray, observed_counts: np.ndarray,
                             model_id: str | None = None, *, alpha_amber: float = 0.05,
                             alpha_red: float = 0.01) -> TestResult:
    """Predicted against observed one-year stage transitions, chi-square on the counts (EBA/Rep/2023/36 para 97).

    statistic = sum over the rows with observations and the cells with positive predicted probability of
    `(O - n_i p)^2 / (n_i p)`, with sum over rows of (positive cells - 1) degrees of freedom; rows with no observations
    are skipped and counted in `extras`. An observation in a cell predicted at zero is impossible under H0 (statistic
    infinite, p-value 0). Light: green when p >= `alpha_amber`, amber when `alpha_red` <= p < `alpha_amber`, red below.
    """
    reference = f"{EBA_2023}, para 97; Anderson and Goodman (1957), doi:10.1214/aoms/1177707039, equation (3.1)"
    version = battery_label(
            POLICY_LABEL, "ifrs9.stage_migration_backtest", alpha_amber=alpha_amber, alpha_red=alpha_red)
    h0 = "the observed stage transitions are drawn from the predicted transition matrix"
    alt = "the observed transitions differ from the predicted matrix"
    pm = np.array(predicted_matrix, dtype=float)
    oc = np.array(observed_counts, dtype=float)
    if pm.shape != (3, 3) or oc.shape != (3, 3):
        raise ValueError(f"predicted_matrix and observed_counts must be (3, 3), got {pm.shape} and {oc.shape}")
    if not np.all(np.isfinite(pm)) or np.any(pm < 0.0) or np.any(np.abs(pm.sum(axis=1) - 1.0) > 1e-9):
        raise ValueError("predicted_matrix must be row-stochastic (rows sum to 1 within 1e-9, entries non-negative)")
    if not np.all(np.isfinite(oc)) or np.any(oc < 0.0) or np.any(oc != np.floor(oc)):
        raise ValueError("observed_counts must be non-negative integers")
    row_n = oc.sum(axis=1)
    used = row_n > 0
    skipped = int((~used).sum())
    stat, dof, impossible = 0.0, 0, False
    for i in np.flatnonzero(used):
        pos = pm[i] > 0.0
        if np.any(oc[i][~pos] > 0.0):
            impossible = True
        expected = row_n[i] * pm[i][pos]
        stat += float(np.sum((oc[i][pos] - expected) ** 2 / expected))
        dof += int(pos.sum()) - 1
    total = int(oc.sum())
    extras = {"rows_skipped": skipped, "degrees_of_freedom": dof, "impossible_cell_observed": impossible}
    if impossible:
        stat, p_value, light = math.inf, 0.0, Light.RED
    elif dof <= 0:
        return _not_evaluated("ifrs9.stage_migration_backtest", h0, alt, reference, version,
                              "no row with observations and at least two possible destinations", use_statistic=True,
                              model_id=model_id, n=total, alpha_amber=alpha_amber, alpha_red=alpha_red,
                              extras=extras)
    else:
        p_value = float(stats.chi2.sf(stat, dof))
        light = Light.GREEN if p_value >= alpha_amber else Light.AMBER if p_value >= alpha_red else Light.RED
    return TestResult(
        test_id="ifrs9.stage_migration_backtest", h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=light, statistic=stat, p_value=p_value, alpha_amber=alpha_amber, alpha_red=alpha_red, n=total,
        model_id=model_id, inputs_hash=inputs_hash(pm, oc), extras=extras,
    )


@register(
    "ifrs9.ecl_backtest",
    family="ifrs9",
    name="ECL backtest",
    h0="the discounted realised losses within the horizon equal the model ECL for the same accounts",
    reference=f"IFRS 9 B5.5.52; {EBA_2023}, paras 100 to 104",
    decision="metric",
)
def ecl_backtest(realised_pv_losses: np.ndarray, ecl_horizon: np.ndarray, horizon_months: int,
                 model_id: str | None = None, *, r_green: tuple[float, float] = (0.8, 1.25),
                 r_amber: tuple[float, float] = (0.5, 2.0)) -> TestResult:
    """Like-for-like ECL backtest, overlays excluded (IFRS 9 B5.5.52; EBA/Rep/2023/36 paras 100 to 104).

    metric = R_H = sum(realised) / sum(ecl). Light: green when `r_green[0] <= R_H <= r_green[1]`, amber when R_H lies in
    `r_amber` but not in `r_green`, red outside. A zero total ECL: not_evaluated. Contraste internal gates.
    """
    reference = f"IFRS 9 B5.5.52; {EBA_2023}, paras 100 to 104"
    version = battery_label(POLICY_LABEL, "ifrs9.ecl_backtest", r_green_low=r_green[0], r_green_high=r_green[1],
                            r_amber_low=r_amber[0], r_amber_high=r_amber[1])
    h0 = "the discounted realised losses within the horizon equal the model ECL for the same accounts"
    alt = "realised losses differ from the ECL (R_H differs from 1)"
    real = _floats(realised_pv_losses, "realised_pv_losses")
    ecl_h = _floats(ecl_horizon, "ecl_horizon")
    if real.shape != ecl_h.shape:
        raise ValueError(
            f"realised_pv_losses and ecl_horizon must have the same shape, got {real.shape} and {ecl_h.shape}"
        )
    if int(horizon_months) < 1:
        raise ValueError(f"horizon_months must be at least 1, got {horizon_months!r}")
    if np.any(real < 0.0):
        raise ValueError("realised_pv_losses must be non-negative")
    if np.any(ecl_h < 0.0):
        raise ValueError("ecl_horizon must be non-negative")
    extras = {"horizon_months": int(horizon_months), "sum_realised": float(real.sum()), "sum_ecl": float(ecl_h.sum())}
    if float(ecl_h.sum()) <= 0.0:
        return _not_evaluated("ifrs9.ecl_backtest", h0, alt, reference, version, "the total ECL is zero",
                              model_id=model_id, n=int(real.size), extras=extras)
    r_h = float(real.sum()) / float(ecl_h.sum())
    if r_green[0] <= r_h <= r_green[1]:
        light = Light.GREEN
    elif r_amber[0] <= r_h <= r_amber[1]:
        light = Light.AMBER
    else:
        light = Light.RED
    return TestResult(
        test_id="ifrs9.ecl_backtest", h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=light, metric=r_h, n=int(real.size), model_id=model_id,
        inputs_hash=inputs_hash(real, ecl_h, int(horizon_months)), extras=extras,
    )


# ----------------------------------------------------------------------------------------------------------------
# scenario sensitivity, non-linearity, overlays, reconciliation


@register(
    "ifrs9.scenario_sensitivity",
    family="ifrs9",
    name="Scenario sensitivity",
    h0="the probability-weighted ECL equals the ECL at the probability-weighted mean path (a linear ECL)",
    reference=f"{D350}, para 36(a); {GPPC}, 2.8.1.2; IFRS 9 5.5.17(a), B5.5.42",
    decision="metric",
)
def scenario_sensitivity(ecl_by_scenario: np.ndarray, weights: np.ndarray, ecl_at_mean_path: float,
                         model_id: str | None = None, *, nl_amber: float = 1.0, nl_red: float = 3.0) -> TestResult:
    """The non-linearity index NL = ECL_weighted / ECL(mean path) - 1 (d350 para 36(a); GPPC 2.8.1.2).

    extras: the 100 per cent single-scenario ECLs and the weight sensitivity against each reference scenario; these are
    sensitivities, never the IFRS 9 measure (5.5.17(a), B5.5.42). Light: green when 0 <= NL <= `nl_amber`, amber when
    NL > `nl_amber` or NL < 0 (a concave ECL is unexpected), red when NL > `nl_red`. A rounding residue below 1e-12 in
    NL is reported as 0. Contraste internal gates.
    """
    reference = f"{D350}, para 36(a); {GPPC}, 2.8.1.2; IFRS 9 5.5.17(a), B5.5.42"
    version = battery_label(POLICY_LABEL, "ifrs9.scenario_sensitivity", nl_amber=nl_amber, nl_red=nl_red)
    h0 = "the probability-weighted ECL equals the ECL at the probability-weighted mean path (a linear ECL)"
    alt = "the weighted ECL exceeds the ECL at the mean path (convex ECL)"
    e = _floats(ecl_by_scenario, "ecl_by_scenario")
    weighted = float(probability_weighted_ecl(e, weights))
    extras = {
        "single_scenario_ecl": e.tolist(), "weighted_ecl": weighted, "ecl_at_mean_path": float(ecl_at_mean_path),
        "weight_sensitivity": [weight_sensitivity(e, k).tolist() for k in range(e.size)],
        "note": "the 100 per cent single-scenario ECLs are sensitivities, never the IFRS 9 measure "
                "(5.5.17(a), B5.5.42)",
    }
    if not float(ecl_at_mean_path) > 0.0:
        return _not_evaluated("ifrs9.scenario_sensitivity", h0, alt, reference, version,
                              "the ECL at the mean path is not positive", model_id=model_id, n=int(e.size),
                              extras=extras)
    nl = non_linearity_index(weighted, float(ecl_at_mean_path))
    if abs(nl) < 1e-12:
        nl = 0.0
    light = Light.RED if nl > nl_red else Light.AMBER if (nl > nl_amber or nl < 0.0) else Light.GREEN
    return TestResult(
        test_id="ifrs9.scenario_sensitivity", h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=light, metric=nl, n=int(e.size), model_id=model_id,
        inputs_hash=inputs_hash(e, np.asarray(weights, dtype=float), float(ecl_at_mean_path)),
        notes="the 100 per cent single-scenario ECLs in extras are sensitivities, never the IFRS 9 measure "
              "(5.5.17(a), B5.5.42)",
        extras=extras,
    )


@register(
    "ifrs9.nonlinearity",
    family="ifrs9",
    name="Non-linearity of the weighted PD",
    h0="the weighted PD equals the PD at the weighted mean factor (a linear PD in the systematic factor)",
    reference=(
        "dossier 02 B.4 and C.1 (BCBS 2005 explanatory note; Vasicek 2002; Gordy 2003, "
        "doi:10.1016/S1042-9573(03)00040-8)"),
    decision="metric",
)
def nonlinearity_test(pd_ttc: float, rho: float, z: np.ndarray, weights: np.ndarray,
                      model_id: str | None = None, *, gh_tolerance: float = 1e-9) -> TestResult:
    """The PD-level non-linearity index with the Gauss-Hermite cross check (dossier 02 B.4, C.1, I.1 item 6).

    metric = PD_weighted / PD(mean z) - 1, the weighted PD over the scenario factors `z` against the PD at their
    weighted mean. extras: `pd_weighted`, `pd_at_mean_z`, `gauss_hermite_error` (|E_Z[PD(Z)] - pd_ttc| at 64 nodes).
    Light: green when the error is at most `gh_tolerance` and metric >= 0, red otherwise (a numerical failure, not a
    model finding). Contraste internal gates.
    """
    reference = (
        "dossier 02 B.4 and C.1 (BCBS 2005 explanatory note; Vasicek 2002; Gordy 2003, "
        "doi:10.1016/S1042-9573(03)00040-8)"
    )
    version = battery_label(POLICY_LABEL, "ifrs9.nonlinearity", gauss_hermite_tolerance=gh_tolerance)
    zz = _floats(z, "z")
    w = np.array(weights, dtype=float)
    weighted = weighted_pit_pd(pd_ttc, rho, zz, w)  # validates the weights
    mean_z = float(np.sum(w * zz))
    at_mean = float(ttc_to_pit(pd_ttc, rho, mean_z))
    metric = non_linearity_index(weighted, at_mean)
    if abs(metric) < 1e-12:
        metric = 0.0
    gh_error = abs(gauss_hermite_pit_pd(pd_ttc, rho, 64) - float(pd_ttc))
    green = gh_error <= gh_tolerance and metric >= 0.0
    return TestResult(
        test_id="ifrs9.nonlinearity",
        h0="the weighted PD equals the PD at the weighted mean factor (a linear PD in the systematic factor)",
        alternative="the weighted PD exceeds the PD at the mean factor", reference=reference, policy_version=version,
        light=Light.GREEN if green else Light.RED, metric=metric,
        n=int(zz.size), model_id=model_id, inputs_hash=inputs_hash(float(pd_ttc), float(rho), zz, w),
        extras={"pd_weighted": weighted, "pd_at_mean_z": at_mean, "mean_z": mean_z,
                "gauss_hermite_error": gh_error, "gauss_hermite_nodes": 64},
    )


@register(
    "ifrs9.overlay_diagnostics",
    family="ifrs9",
    name="Overlay diagnostics",
    h0="overlays are a small, attributed, directionally consistent part of the allowance",
    reference=f"{ECB_2024}; ECB letter SSM-2020-0744 (2020-12-04); {D350}, paras 50 and 51",
    decision="metric",
)
def overlay_diagnostics(overlays: Sequence[Overlay], allowance_by_stage: Mapping[str, float],
                        scenario_direction: int, model_id: str | None = None, *,
                        share_amber: float = 0.10, share_red: float = 0.25) -> TestResult:
    """Overlays as a share of the total allowance (ECB July 2024; d350 paras 50, 51).

    `allowance_by_stage` is the model allowance by stage (overlays excluded); `scenario_direction` is +1 when the
    scenarios worsened, -1 when they improved, 0 otherwise. metric = overlay total / (model total + overlay total).
    extras: the share by stage (an overlay with no stage under `portfolio`) and by segment, the amounts by target,
    `parameter_level` (targets pd, lgd, ead), `staging_level` (target staging-share), `total_level` (overlays attached
    to no segment) and `direction_consistent` (false when a positive overlay stands in an improving scenario).
    Light: green when metric <= `share_amber` and the direction is consistent; red above `share_red`; amber
    otherwise. Contraste internal gates.
    """
    reference = f"{ECB_2024}; ECB letter SSM-2020-0744 (2020-12-04); {D350}, paras 50 and 51"
    version = battery_label(POLICY_LABEL, "ifrs9.overlay_diagnostics", share_amber=share_amber, share_red=share_red)
    h0 = "overlays are a small, attributed, directionally consistent part of the allowance"
    alt = "overlays are large or move against the direction of the scenarios"
    if int(scenario_direction) not in (-1, 0, 1):
        raise ValueError(f"scenario_direction must be -1, 0 or +1, got {scenario_direction!r}")
    check_unique(overlays)
    model_by_stage = {str(k): float(v) for k, v in allowance_by_stage.items()}
    if not all(math.isfinite(v) for v in model_by_stage.values()):
        raise ValueError("allowance_by_stage must be finite")
    overlay_sum = math.fsum(float(o.amount) for o in overlays)
    total = math.fsum(model_by_stage.values()) + overlay_sum
    if not total > 0.0:
        return _not_evaluated("ifrs9.overlay_diagnostics", h0, alt, reference, version,
                              "the total allowance is not positive", model_id=model_id, n=len(overlays))
    by_stage_amount: dict[str, float] = {}
    by_segment_amount: dict[str, float] = {}
    for o in overlays:
        key = o.stage if o.stage is not None else "portfolio"
        by_stage_amount[key] = by_stage_amount.get(key, 0.0) + float(o.amount)
        seg = o.segment_id if o.segment_id is not None else "portfolio"
        by_segment_amount[seg] = by_segment_amount.get(seg, 0.0) + float(o.amount)
    share_by_stage: dict[str, float | None] = {}
    for key in sorted(set(model_by_stage) | set(by_stage_amount)):
        amount = by_stage_amount.get(key, 0.0)
        denominator = model_by_stage.get(key, 0.0) + amount
        share_by_stage[key] = amount / denominator if denominator > 0.0 else None
    by_target = {t: math.fsum(float(o.amount) for o in overlays if o.target == t) for t in TARGETS}
    consistent = not (int(scenario_direction) == -1 and any(float(o.amount) > 0.0 for o in overlays))
    metric = overlay_sum / total
    if metric > share_red:
        light = Light.RED
    elif metric > share_amber or not consistent:
        light = Light.AMBER
    else:
        light = Light.GREEN
    return TestResult(
        test_id="ifrs9.overlay_diagnostics", h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=light, metric=metric, n=len(overlays), model_id=model_id,
        inputs_hash=inputs_hash(model_by_stage, [(o.id, o.amount, o.stage, o.segment_id, o.target) for o in overlays],
                                int(scenario_direction)),
        extras={
            "overlay_total": overlay_sum, "model_total": math.fsum(model_by_stage.values()), "total_allowance": total,
            "share_by_stage": share_by_stage,
            "share_by_segment": {k: v / total for k, v in sorted(by_segment_amount.items())},
            "by_target": by_target,
            "parameter_level": by_target["pd"] + by_target["lgd"] + by_target["ead"],
            "staging_level": by_target["staging-share"],
            "total_level": math.fsum(float(o.amount) for o in overlays if o.segment_id is None),
            "direction_consistent": bool(consistent), "scenario_direction": int(scenario_direction),
        },
    )


@register(
    "ifrs9.reconciliation_identities",
    family="ifrs9",
    name="Reconciliation identities",
    h0="opening plus movements equals closing, by stage, and the waterfall opens into the stage tables",
    reference="IFRS 7 35H and 35I; dossier 02 I.1 items 9 and 10 and I.2",
    decision="metric",
)
def reconciliation_identities(allowance: Sequence[RollForwardRow], gca: Sequence[RollForwardRow],
                              waterfall_lines: Sequence[dict], stage_ecl: Mapping[str, float],
                              model_id: str | None = None, *, tolerance: float = 1e-6) -> TestResult:
    """The IFRS 7 35H and 35I identities (dossier 02 I.2), judged on the tolerance basis `tolerance * max(1, G)`.

    metric: the largest absolute residual of the roll-forward identities (the two roll-forwards, the waterfall closing,
    the waterfall against the stage closings, the lines against the stage movements). The stage closing against the
    stage ECL (the overlays carried in a stage read from the waterfall's overlay lines) is a rounding comparison of
    two stated amounts; its largest gap is `stage_ecl_gap_max` and it is judged on the same tolerance. extras: the
    failing identity names, the absolute tolerance and G. Light: green when every identity closes, red otherwise;
    `not_evaluated` with no rows, with no opening or closing waterfall line or with a stage lacking its stage ECL (never
    an infinite metric). Contraste internal gate, not a regulatory threshold.
    """
    reference = "IFRS 7 35H and 35I; dossier 02 I.1 items 9 and 10 and I.2"
    version = battery_label(POLICY_LABEL, "ifrs9.reconciliation_identities", tolerance=tolerance)
    h0 = "opening plus movements equals closing, by stage, and the waterfall opens into the stage tables"
    alt = "an identity of the reconciliation does not close"
    if not allowance and not gca:
        return _not_evaluated("ifrs9.reconciliation_identities", h0, alt, reference, version,
                              "no roll-forward rows were given", model_id=model_id)
    g_total = _rc.gross_total(gca)
    limit = float(tolerance) * max(1.0, g_total)
    residuals = _rc.identity_residuals(allowance, gca, waterfall_lines, stage_ecl,
                                       carried=_rc.carried_from_lines(waterfall_lines))
    unevaluable = sorted(k for k, v in residuals.items() if not math.isfinite(v))
    if unevaluable:
        reason = ("the waterfall has no opening or closing line" if "waterfall-closes" in unevaluable else
                  "a stage has no stage ECL to compare with its allowance closing")
        return _not_evaluated("ifrs9.reconciliation_identities", h0, alt, reference, version,
                              f"{reason}; identities that cannot be evaluated: {', '.join(unevaluable)}",
                              model_id=model_id, extras={"unevaluable_identities": unevaluable})
    failing = [name for name, v in residuals.items() if not v <= limit]
    exact = [v for k, v in residuals.items() if not k.startswith("stage-closing-equals-ecl")]
    gaps = [v for k, v in residuals.items() if k.startswith("stage-closing-equals-ecl")]
    return TestResult(
        test_id="ifrs9.reconciliation_identities", h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=Light.RED if failing else Light.GREEN, metric=max(exact) if exact else 0.0,
        n=len(allowance) + len(gca), model_id=model_id,
        inputs_hash=inputs_hash([(r.stage, r.opening, r.movements, r.closing) for r in allowance],
                                [(r.stage, r.opening, r.movements, r.closing) for r in gca],
                                [sorted(line.items(), key=str) for line in waterfall_lines], dict(stage_ecl)),
        extras={"failing_identities": failing, "tolerance": limit, "gross_total": g_total,
                "stage_ecl_gap_max": max(gaps) if gaps else 0.0},
    )


__all__ = [
    "ecl_backtest", "lifetime_pd_backtest", "nonlinearity_test", "overlay_diagnostics", "reconciliation_identities",
    "scenario_sensitivity", "sicr_effectiveness", "sicr_threshold_stability", "stage_migration_backtest",
]
