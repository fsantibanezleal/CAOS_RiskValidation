"""Transition matrix estimators: cohort, pooled cohort, duration (the generator's maximum likelihood) and
Aalen-Johansen.

States are ordered best to worst with default last and absorbing; withdrawals are never a grade.

Cohort (Anderson and Goodman 1957, doi:10.1214/aoms/1177707039, equations 2.8 and 2.9): with N_i ratings in state i at
the start of a period and N_ij of them in j at its end, p^_ij = N_ij / N_i; pooled over periods,
p^_ij = sum_t n_ij(t) / sum_t sum_k n_ik(t), "add the corresponding entries in the two-way tables ... divided by the sum
of the entries in the ith row". Movements within the period are not seen. Ratings withdrawn by the end of the period
are, by "the method which has emerged as an industry standard", treated "as non-informative ... distributed among all
states in proportion to their values" (Schuermann and Hanson 2004, FRBNY Staff Report 190, footnote 4): removing them
and renormalising each row does exactly that. Keeping them instead as an absorbing state shows how much of each row
left the sample.

Duration (Lando and Skodeberg 2002, doi:10.1016/S0378-4266(01)00228-X): for a time-homogeneous chain observed
continuously, the likelihood is L(Q) = exp( sum_i [ sum_{j != i} log(q_ij) K_ij - S_i sum_{j != i} q_ij ] ), with K_ij
the jumps from i to j and S_i the time spent in i (Smith and dos Reis 2018, arXiv 1702.08867, citing Kuchler and
Sorensen 1997), maximised by q^_ij = K_ij / S_i. A withdrawn obligor contributes its time up to the withdrawal,
"firms which end the period in an NR status still contribute ... up until the date when they transition to NR"
(Schuermann and Hanson 2004).

Aalen-Johansen (Aalen and Johansen 1978): without time homogeneity, P^(s, t) = prod_{s < u <= t} (I + dA^(u)), the
product over the observed transition times of the identity plus the Nelson-Aalen increments dA^_ij(u) =
dN_ij(u) / Y_i(u-), dA^_ii(u) = -sum_{j != i} dA^_ij(u), Y_i(u-) the obligors in i just before u.

Paths are a segment table, one row per stay of an obligor in a state: ``obligor``, ``start``, ``stop``, ``state`` and
``to``, the state entered at ``stop`` or ``CENSORED_END`` (-1, the observation ended) or ``WITHDRAWN`` (-2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CENSORED_END = -1
WITHDRAWN = -2

ANDERSON_GOODMAN_1957 = (
    "Anderson and Goodman (1957), Statistical inference about Markov chains, doi:10.1214/aoms/1177707039"
)
SCHUERMANN_HANSON_2004 = "Schuermann and Hanson (2004), Estimating probabilities of default, FRBNY Staff Report 190"
LANDO_SKODEBERG_2002 = (
    "Lando and Skodeberg (2002), Analyzing rating transitions and rating drift with continuous observations, "
    "doi:10.1016/S0378-4266(01)00228-X"
)
AALEN_JOHANSEN_1978 = (
    "Aalen and Johansen (1978), An empirical transition matrix for non-homogeneous Markov chains based on censored "
    "observations, Scandinavian Journal of Statistics 5(3), 141-150"
)


def _counts(counts: Any, *, name: str = "counts") -> np.ndarray:
    m = np.asarray(counts, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {m.shape}")
    if not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return m


def cohort(
    counts: Sequence[Sequence[float]],
    withdrawn: Sequence[float] | None = None,
    *,
    treatment: str = "exclude",
    default_state: int | None = -1,
) -> dict[str, Any]:
    """The cohort transition matrix of one period.

    ``counts`` is square, rows the state at the start and columns at the end; ``withdrawn`` the ratings of each start
    state withdrawn by the end. ``treatment="exclude"`` removes them and renormalises each row (the industry
    standard); ``"state"`` keeps them as a last, absorbing column and row. The default row (``default_state``, the
    last by default; ``None`` if there is none) is set to the unit vector: default is absorbing whatever a source
    reports after it (reinstatements). A row with no ratings is returned as NaN and listed in ``empty_rows``.
    """
    n = _counts(counts)
    m = n.shape[0]
    if n.shape != (m, m):
        raise ValueError(f"counts must be square, got shape {n.shape}")
    w = np.zeros(m) if withdrawn is None else np.asarray(withdrawn, dtype=float)
    if w.shape != (m,) or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError(f"withdrawn must be {m} non-negative numbers")
    if treatment not in ("exclude", "state"):
        raise ValueError("treatment must be 'exclude' or 'state'")
    d = None if default_state is None else default_state % m
    base = n.sum(axis=1) + w
    share = np.divide(w, base, out=np.zeros(m), where=base > 0)
    if treatment == "state":
        n = np.hstack([np.vstack([n, np.zeros((1, m))]), np.append(w, 0.0)[:, None]])
        sizes = n.sum(axis=1)
        sizes[m] = 0.0
    else:
        sizes = n.sum(axis=1)
    k = n.shape[0]
    p = np.full((k, k), np.nan)
    empty = []
    for i in range(k):
        if i == d or (treatment == "state" and i == m):
            continue
        if sizes[i] > 0:
            p[i] = n[i] / sizes[i]
        else:
            empty.append(i)
    for i in (d, m if treatment == "state" else None):
        if i is not None:
            p[i] = 0.0
            p[i, i] = 1.0
    return {
        "matrix": p,
        "row_sizes": sizes,
        "counts": n,
        "withdrawn": w,
        "withdrawn_share": share,
        "treatment": treatment,
        "default_state": d,
        "empty_rows": empty,
        "reference": f"{ANDERSON_GOODMAN_1957}, equation (2.9); withdrawals: {SCHUERMANN_HANSON_2004}, footnote 4",
    }


def pooled_cohort(
    counts_by_period: Sequence[Sequence[Sequence[float]]],
    withdrawn_by_period: Sequence[Sequence[float]] | None = None,
    *,
    treatment: str = "exclude",
    default_state: int | None = -1,
) -> dict[str, Any]:
    """Anderson and Goodman's stationary estimator (2.8): the periods' count tables added, then each row normalised."""
    tables = [_counts(c, name="counts_by_period[t]") for c in counts_by_period]
    if not tables:
        raise ValueError("at least one period is needed")
    shape = tables[0].shape
    if any(t.shape != shape for t in tables):
        raise ValueError("every period's table must have the same shape")
    total = np.sum(tables, axis=0)
    w = None
    if withdrawn_by_period is not None:
        ws = [np.asarray(x, dtype=float) for x in withdrawn_by_period]
        if len(ws) != len(tables):
            raise ValueError("one withdrawals vector per period")
        w = np.sum(ws, axis=0)
    out = cohort(total, w, treatment=treatment, default_state=default_state)
    out["periods"] = len(tables)
    out["reference"] = f"{ANDERSON_GOODMAN_1957}, equation (2.8); withdrawals: {SCHUERMANN_HANSON_2004}, footnote 4"
    return out


def _segments(paths: Mapping[str, Any]) -> dict[str, np.ndarray]:
    seg = {k: np.asarray(paths[k]) for k in ("obligor", "start", "stop", "state", "to")}
    n = len(seg["state"])
    if any(len(v) != n for v in seg.values()):
        raise ValueError("every column of the segment table must have the same length")
    if np.any(seg["stop"] < seg["start"]):
        raise ValueError("a segment stops before it starts")
    return seg


def duration_generator(paths: Mapping[str, Any], n_states: int, *, default_state: int | None = -1) -> dict[str, Any]:
    """The maximum likelihood generator of a time-homogeneous chain observed continuously: q_ij = K_ij / S_i.

    A state never occupied keeps a zero row (no information) and is listed in ``unobserved``; default is absorbing.
    """
    seg = _segments(paths)
    m = int(n_states)
    k = np.zeros((m, m))
    s = np.zeros(m)
    np.add.at(s, seg["state"].astype(int), (seg["stop"] - seg["start"]).astype(float))
    moved = seg["to"] >= 0
    np.add.at(k, (seg["state"][moved].astype(int), seg["to"][moved].astype(int)), 1.0)
    np.fill_diagonal(k, 0.0)
    q = np.zeros((m, m))
    occupied = s > 0
    q[occupied] = k[occupied] / s[occupied, None]
    d = None if default_state is None else default_state % m
    if d is not None:
        q[d] = 0.0
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    return {
        "generator": q,
        "jumps": k,
        "holding": s,
        "unobserved": [int(i) for i in np.flatnonzero(~occupied) if i != d],
        "reference": f"{LANDO_SKODEBERG_2002}; likelihood as in Smith and dos Reis (2018), arXiv 1702.08867",
    }


def aalen_johansen(paths: Mapping[str, Any], n_states: int, s: float, t: float) -> dict[str, Any]:
    """The Aalen-Johansen estimate of P(s, t): the product over observed transition times in (s, t] of I + dA(u)."""
    if not t > s:
        raise ValueError("t must be after s")
    seg = _segments(paths)
    m = int(n_states)
    start, stop = seg["start"].astype(float), seg["stop"].astype(float)
    state, to = seg["state"].astype(int), seg["to"].astype(int)
    moved = (to >= 0) & (stop > s) & (stop <= t)
    times = np.unique(stop[moved])
    p = np.eye(m)
    for u in times:
        at = moved & (stop == u)
        # at risk just before u: in the state over an interval that contains u from the left
        risk = np.bincount(state[(start < u) & (stop >= u)], minlength=m).astype(float)
        da = np.zeros((m, m))
        np.add.at(da, (state[at], to[at]), 1.0)
        np.fill_diagonal(da, 0.0)
        nz = risk > 0
        da[nz] = da[nz] / risk[nz, None]
        da[~nz] = 0.0
        np.fill_diagonal(da, -da.sum(axis=1))
        p = p @ (np.eye(m) + da)
    return {"matrix": p, "transition_times": len(times), "reference": AALEN_JOHANSEN_1978}

__all__ = [
    "aalen_johansen",
    "cohort",
    "duration_generator",
    "pooled_cohort",
]
