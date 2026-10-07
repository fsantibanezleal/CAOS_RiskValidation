"""The maximum likelihood generator of a chain observed only at fixed dates (snapshot counts), by EM.

Bladt and Sorensen (2005), Statistical inference for discretely observed Markov jump processes, JRSS B 67(3),
395-410, as presented with closed-form expectations by Smith and dos Reis (2018), Robust and consistent estimation of
generators in credit risk, Quantitative Finance, arXiv 1702.08867 (section 2). The complete-data likelihood of a chain
observed continuously is L(Q) = exp( sum_i [ sum_{j != i} log(q_ij) K_ij - S_i sum_{j != i} q_ij ] ); EM maximises the
likelihood of the snapshots by iterating

    M-step  q_ij <- E_Q[K_ij | y] / E_Q[S_i | y]                                                  (2.3)

with the expectations given in closed form by Van Loan's block exponentials (Proposition 2.4): for a step of length
Delta from state a to state b, with C_gamma^(ij) = [[Q, q_ij e_i e_j'], [0, Q]] and C_phi^(i) = [[Q, e_i e_i'], [0, Q]],

    E[K_ij | a -> b] = (exp(C_gamma^(ij) Delta))_{a, h + b} / (exp(Q Delta))_{a b},
    E[S_i  | a -> b] = (exp(C_phi^(i) Delta))_{a, h + b} / (exp(Q Delta))_{a b},

summed over the steps weighted by the counts n_ab. The observed log-likelihood sum n_ab log P_ab(Delta) does not
decrease from one iteration to the next. An entry that starts at 0 stays at 0, so the start gives every allowed move a
positive rate. Ratings withdrawn within a step leave its counts (the industry treatment, Schuermann and Hanson 2004).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.linalg import expm

SMITH_DOS_REIS_2018 = (
    "Smith and dos Reis (2018), Robust and consistent estimation of generators in credit risk, Quantitative Finance, "
    "doi:10.1080/14697688.2017.1383627 (arXiv 1702.08867); Bladt and Sorensen (2005), JRSS B 67(3), 395-410"
)


def _blocks(q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """exp(Q dt), and for every (i, j) and i the upper-right blocks of the Van Loan exponentials (without the q_ij
    factor): J[i, j] = int_0^dt exp(Qs) e_i e_j' exp(Q(dt - s)) ds, an m x m matrix in (a, b)."""
    m = q.shape[0]
    p = expm(q * dt)
    jb = np.zeros((m, m, m, m))
    for i in range(m):
        for j in range(m):
            c = np.zeros((2 * m, 2 * m))
            c[:m, :m] = q
            c[m:, m:] = q
            c[i, m + j] = 1.0
            jb[i, j] = expm(c * dt)[:m, m:]
    return p, jb, np.stack([jb[i, i] for i in range(m)])


def em_generator(
    steps: Sequence[dict[str, Any]],
    *,
    q0: np.ndarray | None = None,
    tol: float = 1e-9,
    max_iter: int = 2000,
    default_state: int = -1,
) -> dict[str, Any]:
    """EM for the generator of snapshot counts. ``steps`` is a sequence of ``{"counts": m x m, "dt": length}``.

    The start ``q0`` defaults to every move from a non-default state at rate 0.05 / (m - 1) plus the cohort rates of
    the pooled one-step matrix; default is absorbing. Returns the generator, the log-likelihood of each iteration, the
    expected jumps and holding times of the last E-step and whether the relative change fell below ``tol``.
    """
    if not steps:
        raise ValueError("at least one step is needed")
    counts = [np.asarray(s["counts"], dtype=float) for s in steps]
    dts = [float(s.get("dt", 1.0)) for s in steps]
    m = counts[0].shape[0]
    if any(c.shape != (m, m) for c in counts) or any(np.any(c < 0) for c in counts) or any(dt <= 0 for dt in dts):
        raise ValueError("every step needs an m x m non-negative count matrix and a positive dt")
    d = default_state % m
    # the E-step is linear in the counts for a given Q and step length: steps of one length are added first
    by_dt: dict[float, np.ndarray] = {}
    for c, dt in zip(counts, dts, strict=True):
        by_dt[dt] = by_dt.get(dt, np.zeros((m, m))) + c
    if q0 is None:
        pooled = sum(c / dt for c, dt in zip(counts, dts, strict=True))
        rows = pooled.sum(axis=1, keepdims=True)
        rate = np.divide(pooled, rows, out=np.zeros_like(pooled), where=rows > 0)
        q = rate + 0.05 / (m - 1)
    else:
        q = np.array(q0, dtype=float)
    q[d] = 0.0
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))

    loglik: list[float] = []
    converged = False
    ek = es = None
    for _ in range(max_iter):
        ek = np.zeros((m, m))
        es = np.zeros(m)
        ll = 0.0
        for dt, c in by_dt.items():
            p, jb, sb = _blocks(q, dt)
            ratio = np.divide(c, p, out=np.zeros_like(c), where=c > 0)
            ll += float(np.sum(c[c > 0] * np.log(p[c > 0])))
            ek += q * np.einsum("ijab,ab->ij", jb, ratio)
            es += np.einsum("iab,ab->i", sb, ratio)
        loglik.append(ll)
        np.fill_diagonal(ek, 0.0)
        new = np.divide(ek, es[:, None], out=np.zeros_like(ek), where=es[:, None] > 0)
        new[d] = 0.0
        np.fill_diagonal(new, 0.0)
        np.fill_diagonal(new, -new.sum(axis=1))
        q = new
        if len(loglik) > 1 and abs(loglik[-1] - loglik[-2]) <= tol * max(1.0, abs(loglik[-2])):
            converged = True
            break
    return {
        "generator": q,
        "loglik": loglik,
        "expected_jumps": ek,
        "expected_holding": es,
        "iterations": len(loglik),
        "converged": converged,
        "reference": SMITH_DOS_REIS_2018,
    }
