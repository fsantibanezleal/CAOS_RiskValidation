"""Anytime-valid monitors of VaR exceedances (BS-15): a betting e-process on the hit rate and one on hit clustering.

A VaR at level `alpha` produces a hit `I_t = 1{loss_t > VaR_t}` that is Bernoulli(alpha) and independent over time when
the model is right. Both monitors are likelihood-ratio mixtures against stated alternatives: a bet on the next hit
against the null is a non-negative martingale with mean 1, a fixed mixture of such bets is one too, and so the
monitor that rejects when the mixture reaches `1 / alpha_level` has size at most `alpha_level` at any stopping time
(Ville). It can be read every day without the multiple-testing inflation of a Kupiec test repeated on a growing or a
sliding window. The price is power at a fixed horizon: a mixture spreads its bet over alternatives.

- `var_hit_eprocess`: alternatives `p_k = m_k alpha` for a grid of multipliers `m_k` (default `0.25, 0.5` below and
  `1.5, 2, 3, 4, 6, 10` above, equal weights; `direction` keeps one side). `E_t = sum_k w_k prod_s (p_k/alpha)^{I_s}
  ((1-p_k)/(1-alpha))^{1-I_s}`. Alternatives with `p_k >= 1` are dropped.
- `var_hit_clustering_eprocess`: first-order Markov alternatives with `P(hit | hit yesterday) = q_k = m_k alpha`
  (default `3, 6, 12, 25, 50`), `P(hit | no hit yesterday) = alpha`. The factor of a day after a non-hit is 1 and the
  first day has no predecessor, so each factor has mean 1 given the past under the null; it reacts to hits that follow
  hits and not to the rate, which is the departure the independence tests (Christoffersen 1998) look for.

Hits are flags in the engine's convention (`market` back-tests); units are dimensionless counts of days. The record is
an `EProcessResult` (`log_e`, the first crossing, the anytime-valid p-value, `to_test_result`).

Sources: Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991; Ramdas, Grunwald, Vovk and Shafer (2023),
Statistical Science 38(4):576-601, arXiv 2210.01948; Ville (1939); Kupiec (1995), J. Derivatives 3(2):73-84,
doi:10.3905/jod.1995.407942 and Christoffersen (1998), Int. Economic Review 39(4):841-862, doi:10.2307/2527341 for the
fixed-window tests the monitors are compared with.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.validation.sequential._core import (
    check_alpha_level,
    check_open_unit,
    flags,
    log_mixture,
    multiplier_grid,
)
from riskvalidation.validation.sequential.monitor import EProcessResult

REFERENCE = (
    "Wang, Wang and Ziegel (2022), E-backtesting, arXiv 2209.00991; Ramdas, Grunwald, Vovk and Shafer (2023), "
    "Statistical Science 38(4):576-601, arXiv 2210.01948; Ville (1939)"
)
_HIT_LOW, _HIT_HIGH = (0.25, 0.5), (1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
_CLUSTER = (3.0, 6.0, 12.0, 25.0, 50.0)


@register(
    "sequential.var_hits",
    family="sequential",
    name="VaR hit-rate e-process",
    h0="the hits are independent Bernoulli(alpha): the VaR exceedance probability equals alpha",
    reference=REFERENCE,
    decision="metric",
)
def var_hit_eprocess(hits: object, alpha: float, *, alpha_level: float = 0.05, direction: str = "two-sided",
                     multipliers: Sequence[float] | None = None) -> EProcessResult:
    """The e-process of the VaR hit rate against `alpha`, the exceedance probability the VaR claims (a decimal, for
    example 0.01).

    `hits` are 0/1 flags in time order (NaN, other values or a non-1-D shape raise: a missing day is not a non-hit).
    `alpha_level` is the size of the monitor (default 0.05, threshold 20). `direction` is `"upper"` (a hit rate above
    `alpha`: the VaR understates the risk, the case a supervisor penalises), `"lower"` (too conservative) or
    `"two-sided"` (default); `multipliers` replaces the default grid by explicit factors of `alpha` (positive, not 1).
    Returns an `EProcessResult` with the path `log E_t`, `t = 1..n` (empty series: `n = 0`, not evaluated).
    """
    a = check_open_unit("alpha", alpha)
    level = check_alpha_level(alpha_level)
    h = flags(hits, "hits")
    m = multiplier_grid(multipliers, direction, _HIT_LOW, _HIT_HIGH)
    m = m[m * a < 1.0]
    if m.size == 0:
        raise ValueError(f"no alternative hit rate below 1 for alpha = {alpha}: lower the multipliers")
    p = m * a
    x = h.astype(float)
    step = x[None, :] * np.log(p / a)[:, None] + (1.0 - x)[None, :] * np.log((1.0 - p) / (1.0 - a))[:, None]
    log_e = log_mixture(np.cumsum(step, axis=1), np.full(m.size, -np.log(m.size))) if h.size else np.empty(0)
    return EProcessResult(
        monitor_id="sequential.var_hits", log_e=log_e, alpha_level=level,
        h0="the hits are independent Bernoulli(alpha): the VaR exceedance probability equals alpha",
        reference=REFERENCE, inputs_hash=inputs_hash(h, a, level, m),
        notes="anytime-valid: read it every day; power at a fixed horizon is below the Kupiec test (the price of "
              "optional stopping)",
        extras={"alpha": a, "direction": direction, "multipliers": m.tolist(), "hit_rates": p.tolist(),
                "weights": [1.0 / m.size] * int(m.size), "n_events": int(h.sum())})


@register(
    "sequential.var_hit_clustering",
    family="sequential",
    name="VaR hit-clustering e-process",
    h0="the hits are independent Bernoulli(alpha): no hit is more likely the day after a hit",
    reference=REFERENCE + "; Christoffersen (1998), doi:10.2307/2527341",
    decision="metric",
)
def var_hit_clustering_eprocess(hits: object, alpha: float, *, alpha_level: float = 0.05,
                                multipliers: Sequence[float] | None = None) -> EProcessResult:
    """The e-process of hit clustering: the null is independent Bernoulli(alpha); the alternatives are first-order
    Markov chains with `P(hit | hit yesterday) = m alpha` (default `m` = 3, 6, 12, 25, 50, equal weights; those with
    `m alpha >= 1` are dropped).

    `hits` are 0/1 flags in time order; `alpha` the VaR exceedance probability. The factor of a day that follows a
    non-hit is 1, so the process does not move on the hit rate alone. Same record and anytime-valid size as
    `var_hit_eprocess`; the first observation has no predecessor and contributes 1.
    """
    a = check_open_unit("alpha", alpha)
    level = check_alpha_level(alpha_level)
    h = flags(hits, "hits")
    m = multiplier_grid(multipliers, "upper", (), _CLUSTER)
    m = m[m * a < 1.0]
    if m.size == 0:
        raise ValueError(f"no alternative P(hit | hit) below 1 for alpha = {alpha}: lower the multipliers")
    q = m * a
    prev, cur = h[:-1].astype(float), h[1:].astype(float)
    inner = cur[None, :] * np.log(q / a)[:, None] + (1.0 - cur)[None, :] * np.log((1.0 - q) / (1.0 - a))[:, None]
    step = np.concatenate([np.zeros((m.size, 1)), prev[None, :] * inner], axis=1) if h.size else np.empty((m.size, 0))
    log_e = log_mixture(np.cumsum(step, axis=1), np.full(m.size, -np.log(m.size))) if h.size else np.empty(0)
    return EProcessResult(
        monitor_id="sequential.var_hit_clustering", log_e=log_e, alpha_level=level,
        h0="the hits are independent Bernoulli(alpha): no hit is more likely the day after a hit",
        reference=REFERENCE + "; Christoffersen (1998), doi:10.2307/2527341", inputs_hash=inputs_hash(h, a, level, m),
        notes="anytime-valid; reacts to a hit that follows a hit, not to the hit rate",
        extras={"alpha": a, "multipliers": m.tolist(), "p_hit_after_hit": q.tolist(),
                "weights": [1.0 / m.size] * int(m.size), "n_events": int(h.sum())})


__all__ = [
    "var_hit_clustering_eprocess",
    "var_hit_eprocess",
]
