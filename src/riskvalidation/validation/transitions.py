"""Tests on rating transitions: time homogeneity, the order of the chain, a matrix against a reference, momentum.

Anderson and Goodman (1957), Statistical inference about Markov chains, Annals of Mathematical Statistics 28(1),
89-110, doi:10.1214/aoms/1177707039 (read at the page images, sections 2 and 3):

- Time homogeneity (3.5): lambda = prod_t prod_{i,j} [p^_ij / p^_ij(t)]^{n_ij(t)}, -2 log lambda asymptotically
  chi-square with (T - 1)[m(m - 1)] degrees of freedom; the contingency form (3.8),
  sum_i sum_{t,j} n_i(t - 1) [p^_ij(t) - p^_ij]^2 / p^_ij, with the same degrees of freedom.
- Order (3.12): first against second order, lambda = prod_{i,j,k} (p^_jk / p^_ijk)^{n_ijk}, -2 log lambda
  asymptotically chi-square with m(m - 1)^2 degrees of freedom.
- Specified probabilities (3.1): sum_j n_i^* (p^_ij - p0_ij)^2 / p0_ij, chi-square with m - 1 degrees of freedom per
  row, summed over the rows.

Their asymptotics assume every p_ij > 0 (section 3.1), and rating tables are sparse. Which form is the default was
decided by measurement, on the rating-path generator under the null (a Markov chain, 1,000 obligors in each of five
grades, five annual snapshots; 4,000 repetitions at the seeds of ``tests/test_size.py``, where the bound is 6.06% at 5%
and 1.49% at 1%), rejection rates at 5% and at 1%:

- time homogeneity: the chi-square (3.8) 4.32% and 0.68%, the likelihood ratio (3.5) 7.12% and 1.68%; the chi-square
  is the default;
- order: on the full m x m x m table the chi-square (3.14) 13.08% and 8.88%, the likelihood ratio (3.12) 0.30% and
  0.03% (one far above its level, the other almost never rejecting); on the previous and next states grouped by
  direction (from a better grade, the same, a worse one; to a better, the same, a worse one), a 3 x 3 table per
  middle grade where every cell has data and the grouping momentum acts on, the chi-square 4.20% and 0.95% and the
  likelihood ratio 5.17% and 1.05%; the grouped chi-square is the default;
- a matrix against a reference: (3.1) as printed 7.12% and 2.90%, its likelihood-ratio analogue 3.23% and 0.38%;
  the likelihood ratio is the default;
- momentum: 4.95% and 1.18% downward, 4.40% and 1.10% upward.

Every form stays available and both are in ``extras``. Cells a source never fills carry no information: each row's
degrees of freedom count only the columns with a positive pooled (or reference) probability and only the periods (or
first states) in which the row was observed, the convention of a contingency table with empty margins removed.

Momentum (dos Reis, Pfeuffer and Smith 2020, arXiv 1809.09889, section 4.1, after Andersen et al. 1991 and Lando and
Skodeberg 2002): the hazard of leaving a grade lambda_n(t) = q_i(t) exp(c Z_n(t)), Z_n = 1 for an obligor downgraded
into its current grade, fitted by the partial likelihood stratified by grade on calendar time (each grade its own
baseline, an obligor at risk from its entry into the grade), the Markov null c = 0 tested by the likelihood ratio.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from riskvalidation.core.build import make_result
from riskvalidation.core.policy import Policy
from riskvalidation.core.registry import register
from riskvalidation.core.result import TestResult

ANDERSON_GOODMAN_1957 = (
    "Anderson and Goodman (1957), Statistical inference about Markov chains, Annals of Mathematical Statistics 28(1), "
    "doi:10.1214/aoms/1177707039"
)
DOS_REIS_2020 = (
    "dos Reis, Pfeuffer and Smith (2020), Capturing model risk and rating momentum in the estimation of probabilities "
    "of default and credit rating migrations, Quantitative Finance, arXiv 1809.09889, section 4.1; Andersen et al. "
    "(1991); Lando and Skodeberg (2002), doi:10.1016/S0378-4266(01)00228-X"
)


def _table(x: Any, ndim: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != ndim or any(s != a.shape[-1] for s in a.shape[-2:]):
        raise ValueError(f"{name} must have {ndim} dimensions with square trailing axes, got shape {a.shape}")
    if np.any(a < 0) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite and non-negative")
    return a


def _xlogy_ratio(n: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """sum n log(a / b) over the cells with n > 0."""
    m = n > 0
    return float(np.sum(n[m] * np.log(a[m] / b[m])))


@register(
    "rating.time_homogeneity",
    family="transitions",
    name="Time homogeneity of a migration matrix (Anderson-Goodman chi-square)",
    h0="the transition probabilities are the same in every period",
    reference=f"{ANDERSON_GOODMAN_1957}, section 3.2, equations (3.5) and (3.8)",
    decision="p_value",
)
def rating_time_homogeneity(
    counts_by_period: Sequence[Sequence[Sequence[float]]],
    *,
    form: str = "chi2",
    default_state: int | None = -1,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """The chi-square (3.8) across T periods' count matrices (rows the start state); ``form="lr"`` gives the likelihood
    ratio (3.5); both are in ``extras``. The default row (absorbing) carries no information and is left out."""
    if form not in ("chi2", "lr"):
        raise ValueError("form is 'chi2' or 'lr'")
    n = _table(counts_by_period, 3, "counts_by_period")
    t_, m, _ = n.shape
    rows = [i for i in range(m) if default_state is None or i != default_state % m]
    lr = chi2 = 0.0
    dof = 0
    per_row = []
    for i in rows:
        nit = n[:, i, :].sum(axis=1)  # n_i(t)
        seen = nit > 0
        if seen.sum() < 2:
            continue
        pooled = n[seen, i, :].sum(axis=0) / nit[seen].sum()
        cols = pooled > 0
        p_t = n[seen, i, :] / nit[seen, None]
        lr_i = 2.0 * _xlogy_ratio(n[seen, i, :], p_t, np.broadcast_to(pooled, p_t.shape))
        chi2_i = float(np.sum(nit[seen, None] * (p_t[:, cols] - pooled[cols]) ** 2 / pooled[cols]))
        dof_i = int((seen.sum() - 1) * (cols.sum() - 1))
        lr += lr_i
        chi2 += chi2_i
        dof += dof_i
        per_row.append({"row": i, "periods": int(seen.sum()), "columns": int(cols.sum()), "lr": lr_i, "chi2": chi2_i,
                        "dof": dof_i})
    if dof == 0:
        return make_result("rating.time_homogeneity", statistic=float("nan"), policy=policy, n=int(n.sum()),
                           model_id=model_id, segment=segment, notes="no row was observed in two periods")
    stat = chi2 if form == "chi2" else lr
    return make_result(
        "rating.time_homogeneity",
        statistic=stat,
        p_value=float(stats.chi2.sf(stat, dof)),
        policy=policy,
        alternative="the transition probabilities change between periods",
        n=int(n.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(n, form),
        notes=f"{'chi-square (3.8)' if form == 'chi2' else 'likelihood ratio (3.5)'}, {dof} degrees of freedom "
        "(empty cells and periods removed)",
        extras={"dof": dof, "form": form, "chi2": chi2, "chi2_p_value": float(stats.chi2.sf(chi2, dof)), "lr": lr,
                "lr_p_value": float(stats.chi2.sf(lr, dof)), "periods": t_, "rows": per_row},
    )


def _group_by_direction(n: np.ndarray, j: int) -> np.ndarray:
    """The 3 x 3 table of a middle state j: previous state better, same, worse; next state better, same, worse."""
    m = n.shape[0]
    g = np.sign(np.arange(m) - j) + 1  # 0 better (smaller index), 1 same, 2 worse
    out = np.zeros((3, 3))
    np.add.at(out, (g[:, None].repeat(m, 1), g[None, :].repeat(m, 0)), n[:, j, :])
    return out


def _order_stat(tables: list[np.ndarray]) -> tuple[float, float, int]:
    """Sum over middle states of the likelihood ratio (3.12) and the chi-square (3.14), with their degrees of
    freedom."""
    lr = x2 = 0.0
    dof = 0
    for tab in tables:
        rows = tab.sum(axis=1)
        seen = rows > 0
        if seen.sum() < 2:
            continue
        pk = tab[seen].sum(axis=0) / rows[seen].sum()
        cols = pk > 0
        p = tab[seen][:, cols] / rows[seen, None]
        lr += 2.0 * _xlogy_ratio(tab[seen][:, cols], p, np.broadcast_to(pk[cols], p.shape))
        x2 += float(np.sum(rows[seen, None] * (p - pk[cols]) ** 2 / pk[cols]))
        dof += int((seen.sum() - 1) * (cols.sum() - 1))
    return lr, x2, dof


@register(
    "rating.markov_order",
    family="transitions",
    name="First-order Markov chain against second order (Anderson-Goodman likelihood ratio)",
    h0="the next state depends on the current state only",
    reference=f"{ANDERSON_GOODMAN_1957}, section 3.3, equation (3.12)",
    decision="p_value",
)
def rating_markov_order(
    triplets: Sequence[Sequence[Sequence[float]]],
    *,
    grouping: str = "direction",
    form: str = "chi2",
    default_state: int | None = -1,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """The order test from n_ijk, the obligors in i, then j, then k at three consecutive dates: by default the
    chi-square (3.14) on the previous and next states grouped by direction; ``grouping="none"`` uses the full table and
    ``form="lr"`` the likelihood ratio (3.12). Both forms of both tables are in ``extras``."""
    if grouping not in ("direction", "none") or form not in ("chi2", "lr"):
        raise ValueError("grouping is 'direction' or 'none', form 'chi2' or 'lr'")
    n = _table(triplets, 3, "triplets")
    m = n.shape[0]
    middles = [j for j in range(m) if default_state is None or j != default_state % m]
    full = _order_stat([n[:, j, :] for j in middles])
    grouped = _order_stat([_group_by_direction(n, j) for j in middles])
    lr, x2, dof = grouped if grouping == "direction" else full
    stat = x2 if form == "chi2" else lr
    if dof == 0:
        return make_result("rating.markov_order", statistic=float("nan"), policy=policy, n=int(n.sum()),
                           model_id=model_id, segment=segment, notes="no middle state was reached from two states")
    return make_result(
        "rating.markov_order",
        statistic=stat,
        p_value=float(stats.chi2.sf(stat, dof)),
        policy=policy,
        alternative="the next state depends on the state before the current one",
        n=int(n.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(n, grouping, form),
        notes=f"{'chi-square (3.14)' if form == 'chi2' else 'likelihood ratio (3.12)'} on the "
        f"{'states grouped by direction' if grouping == 'direction' else 'full table'}, {dof} degrees of freedom",
        extras={"dof": dof, "grouping": grouping, "form": form,
                "grouped": {"lr": grouped[0], "chi2": grouped[1], "dof": grouped[2]},
                "full": {"lr": full[0], "chi2": full[1], "dof": full[2]}},
    )


@register(
    "rating.matrix_reference",
    family="transitions",
    name="A migration matrix against a reference matrix (Anderson-Goodman chi-square)",
    h0="the transition probabilities equal the reference matrix",
    reference=f"{ANDERSON_GOODMAN_1957}, section 3.1, equation (3.1)",
    decision="p_value",
)
def rating_matrix_reference(
    counts: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    *,
    form: str = "lr",
    default_state: int | None = -1,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """By default the likelihood ratio 2 sum n_ij log(p^_ij / p0_ij); ``form="chi2"`` gives (3.1),
    sum_i sum_j n_i (p^_ij - p0_ij)^2 / p0_ij; both over the reference's positive cells, both in ``extras``. A move the
    reference makes impossible (p0_ij = 0) that the counts show rejects outright (p = 0)."""
    if form not in ("lr", "chi2"):
        raise ValueError("form is 'lr' or 'chi2'")
    n = _table(counts, 2, "counts")
    p0 = _table(reference, 2, "reference")
    if p0.shape != n.shape:
        raise ValueError("counts and reference must have the same shape")
    m = n.shape[0]
    x2 = g2 = 0.0
    dof = 0
    impossible = []
    for i in range(m):
        if default_state is not None and i == default_state % m:
            continue
        ni = n[i].sum()
        if ni == 0:
            continue
        p = n[i] / ni
        pos = p0[i] > 0
        x2 += float(np.sum(ni * (p[pos] - p0[i, pos]) ** 2 / p0[i, pos]))
        g2 += 2.0 * _xlogy_ratio(n[i, pos], p[pos], p0[i, pos])
        dof += int(pos.sum() - 1)
        impossible += [(i, int(j)) for j in np.flatnonzero(~pos & (n[i] > 0))]
    if dof == 0:
        return make_result("rating.matrix_reference", statistic=float("nan"), policy=policy, n=int(n.sum()),
                           model_id=model_id, segment=segment, notes="no row with data and two positive cells")
    stat = g2 if form == "lr" else x2
    p_value = 0.0 if impossible else float(stats.chi2.sf(stat, dof))
    return make_result(
        "rating.matrix_reference",
        statistic=stat,
        p_value=p_value,
        policy=policy,
        alternative="the transition probabilities differ from the reference",
        n=int(n.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(n, p0, form),
        notes=f"{'likelihood ratio' if form == 'lr' else 'chi-square (3.1)'}, {dof} degrees of freedom"
        + (f"; {len(impossible)} observed move(s) the reference makes impossible" if impossible else ""),
        extras={"dof": dof, "form": form, "lr": g2, "chi2": x2, "chi2_p_value": float(stats.chi2.sf(x2, dof)),
                "impossible_moves": impossible},
    )


def _momentum_fit(at_risk_events: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    """Maximise the partial log-likelihood l(c) = sum_e [c z_e - log(n0_e + n1_e e^c)] (one binary covariate) by
    Newton; returns (c^, l(c^), l(0))."""

    def ll(c: float) -> tuple[float, float, float]:
        v, g, h = 0.0, 0.0, 0.0
        for z, n0, n1 in at_risk_events:
            s = n0 + n1 * math.exp(c)
            v += float(np.sum(c * z - np.log(s)))
            w = n1 * math.exp(c) / s
            g += float(np.sum(z - w))
            h -= float(np.sum(w * (1.0 - w)))
        return v, g, h

    c = 0.0
    l0 = ll(0.0)[0]
    for _ in range(100):
        v, g, h = ll(c)
        if h >= 0 or not math.isfinite(h):
            break
        step = g / h
        c_new = c - step
        if not math.isfinite(c_new) or abs(c_new) > 50:
            break
        c = c_new
        if abs(step) < 1e-10:
            break
    return c, ll(c)[0], l0


@register(
    "rating.momentum",
    family="transitions",
    name="Rating momentum (stratified hazard with a downgraded-into covariate, likelihood ratio)",
    h0="how an obligor entered its grade does not change its hazard of leaving it (Markov)",
    reference=DOS_REIS_2020,
    decision="p_value",
)
def rating_momentum(
    paths: Mapping[str, Any],
    *,
    direction: str = "down",
    default_state: int | None = -1,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """The likelihood ratio for c in lambda(t) = q_g(t) exp(c Z), Z = 1 for an obligor downgraded (``"down"``) or
    upgraded (``"up"``) into its current grade g, from the segment table of obligor paths (each stay in a grade from
    its entry to its exit or censoring). The first stay of an obligor (its initial rating) has Z = 0."""
    if direction not in ("down", "up"):
        raise ValueError("direction must be 'down' or 'up'")
    seg = {k: np.asarray(paths[k]) for k in ("obligor", "start", "stop", "state", "to")}
    ob, start, stop = seg["obligor"], seg["start"].astype(float), seg["stop"].astype(float)
    state, to = seg["state"].astype(int), seg["to"].astype(int)
    order = np.lexsort((start, ob))
    ob, start, stop, state, to = ob[order], start[order], stop[order], state[order], to[order]
    prev = np.full(state.size, -1)
    same = np.r_[False, ob[1:] == ob[:-1]]
    prev[same] = state[np.flatnonzero(same) - 1]
    z = (same & ((prev < state) if direction == "down" else (prev > state))).astype(float)
    m = int(state.max()) + 1 if state.size else 0
    d = None if default_state is None else (default_state % max(m, 1))
    groups = []
    n_events = 0
    for g in np.unique(state):
        if d is not None and g == d:
            continue
        sel = state == g
        st, sp, zz, ev = start[sel], stop[sel], z[sel], to[sel] >= 0
        if ev.sum() == 0 or zz.sum() == 0 or zz.sum() == zz.size:
            continue
        times = sp[ev]
        # at risk just before u: start < u <= stop, counted by covariate value
        n_at = []
        for flag in (0.0, 1.0):
            s_sorted = np.sort(st[zz == flag])
            e_sorted = np.sort(sp[zz == flag])
            n_at.append(np.searchsorted(s_sorted, times, side="left") - np.searchsorted(e_sorted, times, side="left"))
        groups.append((zz[ev], n_at[0].astype(float), n_at[1].astype(float)))
        n_events += int(ev.sum())
    if not groups:
        return make_result("rating.momentum", statistic=float("nan"), policy=policy, n=int(state.size),
                           model_id=model_id, segment=segment, notes="no grade has stays of both kinds with exits")
    c, l1, l0 = _momentum_fit(groups)
    lr = max(2.0 * (l1 - l0), 0.0)
    return make_result(
        "rating.momentum",
        statistic=lr,
        p_value=float(stats.chi2.sf(lr, 1)),
        policy=policy,
        alternative=f"obligors {direction}graded into a grade leave it at a different rate",
        n=int(state.size),
        n_events=n_events,
        model_id=model_id,
        segment=segment,
        notes="-2 log partial-likelihood ratio, chi-square with 1 degree of freedom; the coefficient is in extras",
        extras={"coefficient": c, "hazard_ratio": math.exp(c), "direction": direction},
    )
