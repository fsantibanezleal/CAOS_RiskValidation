"""Measurement: ECL as a sum of discounted marginal losses, probability weighting, Stage 3, POCI, the simplified
approach and the portfolio allowance (dossier 02 B.3, B.4, B.5, B.7, A.5, A.6; IFRS 9 5.5.17 to 5.5.20,
B5.5.28 to B5.5.55).

    ECL_{i,s}(T) = sum_{t <= T} q_D(t) LGD(t) EAD(t) DF(t),   DF(t) = (1 + EIR)^(-delta(t) / 12)

with `delta(t)` the months from the reporting date to the loss date of period t (mid-period or end-of-period). The
discounting is at the original EIR to the reporting date, never at a risk-free rate and never to the default date.
Overlays are separate lines (see `overlays`) and never enter the model ECL. This module states no compliance claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from riskvalidation.provisions._checks import (
    closed_unit,
    float_array,
    months_array,
    non_negative,
    open_unit,
    same_length,
    weights_array,
)
from riskvalidation.provisions.overlays import Overlay, check_unique

CONVENTIONS = ("mid-period", "end-of-period")
STAGE_KEYS = ("stage1", "stage2", "stage3", "poci")


def discount_factors(eir: np.ndarray, months: np.ndarray, convention: str = "mid-period") -> np.ndarray:
    """DF (A, T) = (1 + eir)^(-delta / 12) to the reporting date; the first period's length is `months[0]`.

    `mid-period`: delta = month - Delta / 2 (the midpoint of the period); `end-of-period`: delta = month.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    r = non_negative(eir, "eir")
    if r.ndim != 1:
        raise ValueError(f"eir must be one-dimensional (A,), got shape {r.shape}")
    m = months_array(months)
    ends = m.astype(float)
    delta = ends if convention == "end-of-period" else ends - np.diff(m, prepend=0).astype(float) / 2.0
    return (1.0 + r[:, None]) ** (-delta[None, :] / 12.0)


def _per_period(x: object, name: str, n_accounts: int, n_periods: int, closed: bool) -> np.ndarray:
    """A scalar, (A,) or (A, T) parameter broadcast to (A, T)."""
    a = closed_unit(x, name) if closed else non_negative(x, name)
    if a.ndim == 0:
        return np.full((n_accounts, n_periods), float(a))
    if a.ndim == 1:
        if a.shape[0] != n_accounts:
            raise ValueError(f"{name} has {a.shape[0]} entries for {n_accounts} accounts")
        return np.repeat(a[:, None], n_periods, axis=1)
    if a.shape != (n_accounts, n_periods):
        raise ValueError(f"{name} must be a scalar, (A,) or (A, T) = {(n_accounts, n_periods)}, got shape {a.shape}")
    return a


def ecl_scenario(marginal_pd: np.ndarray, lgd: np.ndarray, ead: np.ndarray, eir: np.ndarray, months: np.ndarray,
                 horizon_months: int, convention: str = "mid-period") -> np.ndarray:
    """The ECL (A,) of one scenario: the sum over the periods with `months <= horizon_months` of q_D LGD EAD DF.

    `horizon_months` is 12 for the 12-month ECL and the expected life (5.5.19, 5.5.20) for the lifetime ECL; it must
    be a member of `months` or at least `months[-1]` (no proration of a straddling period).
    """
    m = months_array(months)
    q = np.array(marginal_pd, dtype=float)
    if q.ndim != 2 or q.shape[1] != m.size:
        raise ValueError(f"marginal_pd must have shape (A, {m.size}), got {q.shape}")
    open_unit(q, "marginal_pd")
    n_accounts = q.shape[0]
    h = int(horizon_months)
    if h not in m and h < int(m[-1]):
        raise ValueError(
            f"horizon_months={horizon_months} is neither a member of months {m.tolist()} nor at least months[-1]: "
            "a straddling period is never prorated"
        )
    lgd_t = _per_period(lgd, "lgd", n_accounts, m.size, closed=True)
    ead_t = _per_period(ead, "ead", n_accounts, m.size, closed=False)
    r = non_negative(eir, "eir")
    if r.shape != (n_accounts,):
        raise ValueError(f"eir must have shape ({n_accounts},), got {r.shape}")
    df = discount_factors(r, m, convention)
    inside = (m <= h)[None, :]
    return np.sum(np.where(inside, q * lgd_t * ead_t * df, 0.0), axis=1)


def probability_weighted_ecl(ecl_by_scenario: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """sum_s w_s ECL_s over at least two scenarios (5.5.17(a), B5.5.42); (S, A) or (S,) in, (A,) or scalar out."""
    e = float_array(ecl_by_scenario, "ecl_by_scenario")
    if e.ndim not in (1, 2):
        raise ValueError(f"ecl_by_scenario must be (S, A) or (S,), got shape {e.shape}")
    w = weights_array(weights, e.shape[0])
    return np.tensordot(w, e, axes=(0, 0))


def non_linearity_index(ecl_weighted: float, ecl_at_mean_path: float) -> float:
    """NL = ecl_weighted / ecl_at_mean_path - 1; raises when the denominator is not positive."""
    if not float(ecl_at_mean_path) > 0.0:
        raise ValueError(f"ecl_at_mean_path must be positive, got {ecl_at_mean_path!r}")
    return float(ecl_weighted) / float(ecl_at_mean_path) - 1.0


def weight_sensitivity(ecl_by_scenario: np.ndarray, reference_index: int) -> np.ndarray:
    """ECL_s - ECL_ref: the derivative dECL/dw_s when weight moves from the reference scenario to s (staging frozen)."""
    e = float_array(ecl_by_scenario, "ecl_by_scenario", ndim=1)
    ref = int(reference_index)
    if not 0 <= ref < e.size:
        raise ValueError(f"reference_index must lie in [0, {e.size - 1}], got {reference_index!r}")
    return e - e[ref]


def stage3_ecl(gross_carrying_amount: np.ndarray, expected_recoveries: np.ndarray, recovery_months: np.ndarray,
               eir: np.ndarray) -> np.ndarray:
    """GCA - sum_k CF_k (1 + eir)^(-months_k / 12), clipped at 0 (B5.5.33): the credit-impaired, non-POCI ECL."""
    gca = non_negative(gross_carrying_amount, "gross_carrying_amount")
    cf = non_negative(expected_recoveries, "expected_recoveries")
    mo = non_negative(recovery_months, "recovery_months")
    r = non_negative(eir, "eir")
    if gca.ndim != 1:
        raise ValueError("gross_carrying_amount must be one-dimensional (A,)")
    if cf.ndim != 2 or cf.shape[0] != gca.size:
        raise ValueError(f"expected_recoveries must have shape (A, K) with A = {gca.size}, got {cf.shape}")
    if mo.shape != cf.shape:
        raise ValueError(f"recovery_months must have the shape of expected_recoveries {cf.shape}, got {mo.shape}")
    if r.shape != gca.shape:
        raise ValueError(f"eir must have shape {gca.shape}, got {r.shape}")
    present_value = np.sum(cf * (1.0 + r[:, None]) ** (-mo / 12.0), axis=1)
    return np.maximum(gca - present_value, 0.0)


def poci_allowance(ecl_lifetime_now: np.ndarray, ecl_lifetime_initial: np.ndarray) -> np.ndarray:
    """now - initial, both lifetime ECLs at the credit-adjusted EIR (5.5.13); may be negative (5.5.14), not clipped."""
    now = float_array(ecl_lifetime_now, "ecl_lifetime_now", ndim=1)
    initial = float_array(ecl_lifetime_initial, "ecl_lifetime_initial", ndim=1)
    same_length({"ecl_lifetime_now": now, "ecl_lifetime_initial": initial})
    return now - initial


def simplified_matrix_ecl(gross_carrying_amount_by_bucket: np.ndarray, loss_rate_by_bucket: np.ndarray,
                          forward_looking_multiplier: float = 1.0) -> float:
    """min(sum(gca), m sum(gca rate)): the provision matrix of the simplified approach (B5.5.35)."""
    g = non_negative(gross_carrying_amount_by_bucket, "gross_carrying_amount_by_bucket")
    rate = closed_unit(loss_rate_by_bucket, "loss_rate_by_bucket")
    if g.ndim != 1 or rate.shape != g.shape:
        raise ValueError(
            "gross_carrying_amount_by_bucket and loss_rate_by_bucket must be one-dimensional and equal in shape"
        )
    m = float(forward_looking_multiplier)
    if not math.isfinite(m) or m < 0.0:
        raise ValueError(
            f"forward_looking_multiplier must be finite and non-negative, got {forward_looking_multiplier!r}"
        )
    return float(min(float(np.sum(g)), m * float(np.sum(g * rate))))


def commitment_ead(drawn: np.ndarray, undrawn: np.ndarray, ccf: np.ndarray) -> np.ndarray:
    """drawn + ccf * undrawn (B5.5.30, B5.5.31), between `drawn` and `drawn + undrawn`."""
    d = non_negative(drawn, "drawn")
    u = non_negative(undrawn, "undrawn")
    c = closed_unit(ccf, "ccf")
    return d + c * u


@dataclass(frozen=True)
class AllowanceBreakdown:
    """Model ECL by stage (the four keys), the model total, the overlay total and the total allowance."""

    by_stage: dict[str, float]
    model_total: float
    overlay_total: float
    total: float


def portfolio_allowance(stage: np.ndarray, ecl_12m: np.ndarray, ecl_lifetime: np.ndarray, ecl_stage3: np.ndarray,
                        poci: np.ndarray | None = None, poci_delta: np.ndarray | None = None,
                        overlays: Sequence[Overlay] = ()) -> AllowanceBreakdown:
    """LA = sum_S1 ECL_12m + sum_S2 ECL_life + sum_S3 ECL_D + sum_POCI delta + sum_k OVL_k (dossier 02 B.5).

    A POCI account is excluded from stages 1 to 3 whatever its `stage` value and counts once, in `poci`. Overlays are
    added to the total only and never enter `by_stage`.
    """
    st = np.asarray(stage)
    if st.ndim != 1:
        raise ValueError(f"stage must be one-dimensional (A,), got shape {st.shape}")
    e12 = np.array(ecl_12m, dtype=float)
    life = np.array(ecl_lifetime, dtype=float)
    s3 = np.array(ecl_stage3, dtype=float)
    n = same_length({"stage": st, "ecl_12m": e12, "ecl_lifetime": life, "ecl_stage3": s3})
    if poci is None:
        is_poci = np.zeros(n, dtype=bool)
    else:
        is_poci = np.asarray(poci)
        if is_poci.dtype != bool or is_poci.shape != (n,):
            raise ValueError(f"poci must be a boolean array of shape ({n},)")
    if np.any(~np.isin(st[~is_poci], (1, 2, 3))):
        raise ValueError("stage must lie in {1, 2, 3} for every non-POCI account")
    delta_total = 0.0
    if is_poci.any():
        if poci_delta is None:
            raise ValueError("poci_delta is required where poci is true")
        delta = np.asarray(poci_delta, dtype=float)
        if delta.shape != (n,):
            raise ValueError(f"poci_delta must have shape ({n},), got {delta.shape}")
        if not np.all(np.isfinite(delta[is_poci])):
            raise ValueError("poci_delta must be finite where poci is true")
        delta_total = math.fsum(delta[is_poci].tolist())

    def pick(values: np.ndarray, stage_no: int) -> float:
        chosen = values[(st == stage_no) & ~is_poci]
        if not np.all(np.isfinite(chosen)):
            raise ValueError(f"ECL of stage {stage_no} accounts must be finite")
        return math.fsum(chosen.tolist())

    by_stage = {"stage1": pick(e12, 1), "stage2": pick(life, 2), "stage3": pick(s3, 3), "poci": delta_total}
    check_unique(overlays)
    model_total = math.fsum(by_stage[k] for k in STAGE_KEYS)
    overlay_sum = math.fsum(float(o.amount) for o in overlays)
    return AllowanceBreakdown(by_stage=by_stage, model_total=model_total, overlay_total=overlay_sum,
                              total=model_total + overlay_sum)
