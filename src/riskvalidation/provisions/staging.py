"""Stage allocation: relative and absolute PD tests, the 30 days past due backstop, forbearance, watch list, cure
probation and the collective transfer (dossier 02 B.5; IFRS 9 5.5.3, 5.5.5, 5.5.9 to 5.5.11, B5.5.37).

Only two presumptions are in IFRS 9: 30 days past due as a rebuttable backstop (5.5.11) and 90 days past due for
default (B5.5.37). The threefold ratio is a supervisory expectation (EBA/Rep/2023/36 para 32), not IFRS 9, and is a
configurable parameter of `StagingPolicy`; the multiplier schedule, the absolute floor, the probation length and the
low credit risk threshold are policy parameters the caller supplies.

Quantitative rule on annualised PDs (orig = origination forward PD, rep = current PD):

    SICR = [rep / orig >= m(orig) and rep - orig >= delta]  or  [rep / orig >= threefold_ratio]

with `m` read from the schedule of (upper bound of orig, multiplier) pairs. Every function acts row by row: row i of
the output depends on row i of the inputs only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from riskvalidation.provisions._checks import bool_array, closed_unit, open_unit, same_length

TRIGGERS = (
    "none", "relative-absolute", "threefold", "dpd-backstop", "forbearance", "watch-list", "default", "collective",
    "cure-probation",
)
_TRIGGER_DTYPE = "U20"


@dataclass(frozen=True)
class StagingPolicy:
    """The staging parameters; PDs are annualised decimals, days and months integers."""

    multiplier_schedule: tuple[tuple[float, float], ...]
    absolute_floor: float
    threefold_ratio: float = 3.0
    dpd_backstop: int = 30
    default_dpd: int = 90
    probation_months: int = 0
    low_credit_risk_pd: float | None = None

    def __post_init__(self) -> None:
        sched = self.multiplier_schedule
        if not sched:
            raise ValueError("multiplier_schedule must hold at least one (pd_origination_upper, multiplier) pair")
        prev_upper, prev_mult = 0.0, float("inf")
        for upper, mult in sched:
            if not 0.0 < upper <= 1.0:
                raise ValueError(f"multiplier_schedule upper bounds must lie in (0, 1], got {upper!r}")
            if upper <= prev_upper:
                raise ValueError("multiplier_schedule upper bounds must be strictly increasing")
            if mult < 1.0:
                raise ValueError(f"multiplier_schedule multipliers must be at least 1, got {mult!r}")
            if mult > prev_mult:
                raise ValueError("multiplier_schedule multipliers must not rise as the origination PD rises")
            prev_upper, prev_mult = upper, mult
        if not 0.0 <= self.absolute_floor <= 1.0:
            raise ValueError(f"absolute_floor must lie in [0, 1], got {self.absolute_floor!r}")
        if not self.threefold_ratio > 1.0:
            raise ValueError(f"threefold_ratio must be above 1, got {self.threefold_ratio!r}")
        if int(self.dpd_backstop) < 0:
            raise ValueError(f"dpd_backstop must be non-negative, got {self.dpd_backstop!r}")
        if int(self.default_dpd) <= int(self.dpd_backstop):
            raise ValueError("default_dpd must be above dpd_backstop")
        if int(self.probation_months) < 0:
            raise ValueError(f"probation_months must be non-negative, got {self.probation_months!r}")
        if self.low_credit_risk_pd is not None and not 0.0 < self.low_credit_risk_pd < 1.0:
            raise ValueError(f"low_credit_risk_pd must be None or lie in (0, 1), got {self.low_credit_risk_pd!r}")


@dataclass(frozen=True)
class StageResult:
    """Stage (A,) int8 in {1, 2, 3}; trigger (A,) str from `TRIGGERS`; lcr_exempt (A,) bool."""

    stage: np.ndarray
    trigger: np.ndarray
    lcr_exempt: np.ndarray


def _components(orig: np.ndarray, rep: np.ndarray, policy: StagingPolicy) -> tuple[np.ndarray, np.ndarray]:
    """(relative-absolute flag, threefold flag), elementwise."""
    o = open_unit(orig, "pd_origination_annualised")
    r = open_unit(rep, "pd_current_annualised")
    if o.ndim != 1 or r.ndim != 1:
        raise ValueError("pd_origination_annualised and pd_current_annualised must be one-dimensional (A,)")
    same_length({"pd_origination_annualised": o, "pd_current_annualised": r})
    uppers = np.array([u for u, _ in policy.multiplier_schedule])
    mults = np.array([m for _, m in policy.multiplier_schedule])
    if o.size and float(np.max(o)) > float(uppers[-1]):
        raise ValueError(
            f"pd_origination_annualised exceeds the last bound of multiplier_schedule ({uppers[-1]}): the schedule "
            "must cover the largest origination PD"
        )
    m = mults[np.searchsorted(uppers, o, side="left")] if o.size else np.zeros(0)
    ratio = r / o if o.size else np.zeros(0)
    rel_abs = (ratio >= m) & ((r - o) >= policy.absolute_floor)
    return rel_abs, ratio >= policy.threefold_ratio


def sicr_quantitative(pd_origination_annualised: np.ndarray, pd_current_annualised: np.ndarray,
                      policy: StagingPolicy) -> np.ndarray:
    """The quantitative SICR flag per account (bool, (A,)); never applies the low credit risk exemption."""
    rel_abs, threefold = _components(pd_origination_annualised, pd_current_annualised, policy)
    return rel_abs | threefold


def allocate_stages(pd_origination_annualised: np.ndarray, pd_current_annualised: np.ndarray, dpd: np.ndarray,
                    forborne: np.ndarray, watch_list: np.ndarray, defaulted: np.ndarray,
                    policy: StagingPolicy, collective: np.ndarray | None = None) -> StageResult:
    """Stage 3 iff defaulted or dpd >= default_dpd; else Stage 2 iff a trigger fires; else Stage 1.

    Precedence of the trigger: default, dpd-backstop, forbearance, watch-list, collective, relative-absolute,
    threefold. The low credit risk exemption (`current PD <= policy.low_credit_risk_pd`) suppresses only the two
    quantitative triggers (5.5.10); the others still apply (5.5.11, B5.5.37).
    """
    rel_abs, threefold = _components(pd_origination_annualised, pd_current_annualised, policy)
    n = rel_abs.shape[0]
    days = np.asarray(dpd)
    if days.dtype.kind not in "iu":
        raise ValueError(f"dpd must be an integer array of days, got dtype {days.dtype}")
    if days.ndim != 1 or np.any(days < 0):
        raise ValueError("dpd must be a one-dimensional array of non-negative days")
    flags = {
        "forborne": bool_array(forborne, "forborne"),
        "watch_list": bool_array(watch_list, "watch_list"),
        "defaulted": bool_array(defaulted, "defaulted"),
    }
    flags["collective"] = np.zeros(n, dtype=bool) if collective is None else bool_array(collective, "collective")
    same_length({"dpd": days, **flags, "pd_current_annualised": rel_abs})

    quantitative = rel_abs | threefold
    if policy.low_credit_risk_pd is None:
        exempt = np.zeros(n, dtype=bool)
    else:
        exempt = np.asarray(pd_current_annualised, dtype=float) <= policy.low_credit_risk_pd
    rel_abs_eff, threefold_eff = rel_abs & ~exempt, threefold & ~exempt

    is_default = flags["defaulted"] | (days >= policy.default_dpd)
    ordered = (
        ("default", is_default),
        ("dpd-backstop", days > policy.dpd_backstop),
        ("forbearance", flags["forborne"]),
        ("watch-list", flags["watch_list"]),
        ("collective", flags["collective"]),
        ("relative-absolute", rel_abs_eff),
        ("threefold", threefold_eff),
    )
    trigger = np.full(n, "none", dtype=_TRIGGER_DTYPE)
    unset = np.ones(n, dtype=bool)
    for name, fired in ordered:
        take = fired & unset
        trigger[take] = name
        unset &= ~take
    stage = np.where(trigger == "none", 1, np.where(trigger == "default", 3, 2)).astype(np.int8)
    return StageResult(stage=stage, trigger=trigger, lcr_exempt=(stage == 1) & quantitative)


def apply_cure_probation(stage_now: np.ndarray, stage_previous: np.ndarray, months_since_cure: np.ndarray,
                         policy: StagingPolicy, now: StageResult | None = None) -> StageResult:
    """Hold in Stage 2 (trigger `cure-probation`) an account that is Stage 1 now, was Stage 2 before and cured fewer
    than `policy.probation_months` months ago; every other account is unchanged.

    The contract signature carries no trigger for the unchanged accounts. The optional `now` (the `StageResult` of
    `allocate_stages` for `stage_now`) supplies the trigger and the exemption flag of the unchanged accounts; without
    it a Stage 1 account has trigger `none` and a Stage 2 or 3 account has trigger `unspecified`. This is an additive
    argument, raised as an open question to the contract owner.
    """
    s_now = np.asarray(stage_now)
    s_prev = np.asarray(stage_previous)
    months = np.asarray(months_since_cure)
    if months.dtype.kind not in "iu":
        raise ValueError(f"months_since_cure must be an integer array of months, got dtype {months.dtype}")
    for name, v in (("stage_now", s_now), ("stage_previous", s_prev)):
        if v.ndim != 1 or np.any(~np.isin(v, (1, 2, 3))):
            raise ValueError(f"{name} must be a one-dimensional array in {{1, 2, 3}}")
    n = same_length({"stage_now": s_now, "stage_previous": s_prev, "months_since_cure": months})
    if now is not None:
        if now.stage.shape != s_now.shape or np.any(now.stage != s_now):
            raise ValueError("now must be the StageResult of stage_now (its stage differs from stage_now)")
        trigger = now.trigger.astype(_TRIGGER_DTYPE).copy()
        lcr = now.lcr_exempt.copy()
    else:
        trigger = np.where(s_now == 1, "none", "unspecified").astype(_TRIGGER_DTYPE)
        lcr = np.zeros(n, dtype=bool)
    held = (s_now == 1) & (s_prev == 2) & (months < policy.probation_months)
    stage = np.where(held, 2, s_now).astype(np.int8)
    trigger[held] = "cure-probation"
    lcr[held] = False
    return StageResult(stage=stage, trigger=trigger, lcr_exempt=lcr)


def collective_transfer(gross_carrying_amount: np.ndarray, segment: np.ndarray,
                        share_by_segment: Mapping[str, float]) -> dict[str, float]:
    """The target money to be in Stage 2 per segment: amount_g = pi_g * sum(gca in g) (dossier 02 B.5, B5.5.6).

    It reports only: the amount is never added to any allowance. The caller turns each target into accounts (the
    `collective` mask of `allocate_stages`), so the ECL reaches the allowance only through the stage of those accounts.
    """
    g = np.array(gross_carrying_amount, dtype=float)
    if g.ndim != 1 or not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("gross_carrying_amount must be a one-dimensional array of finite non-negative money")
    seg = np.asarray(segment)
    if seg.ndim != 1:
        raise ValueError("segment must be a one-dimensional array")
    same_length({"gross_carrying_amount": g, "segment": seg})
    out: dict[str, float] = {}
    for name in sorted({str(s) for s in seg}):
        if name not in share_by_segment:
            raise KeyError(f"share_by_segment has no share for segment {name!r}")
        share = float(closed_unit(share_by_segment[name], f"share of segment {name!r}"))
        out[name] = share * float(np.sum(g[seg == name]))
    return out
