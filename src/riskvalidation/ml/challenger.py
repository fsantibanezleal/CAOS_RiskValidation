"""Champion and challenger: the paired bootstrap of the Brier difference and the share of obligors whose grade differs
by two or more notches, beside the existing ``disc_delong`` (dossier 06 C.7 l.1041-1044; I.1 l.1196; SS1/23 principle
3.3(b)(iii); ECB guide paragraph 60). The existing ``bootstrap_ci`` resamples one score and cannot resample two
forecasts jointly, so the paired form is new. Contract: ``engine-ml-api`` 1.1.0, module
``riskvalidation/ml/challenger.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.ml._checks import binary, closed_unit, index, same_rows, vec

LEVEL = 0.95  # the interval level of the paired bootstrap (a convention, 06 C.7 l.1041-1044 names no value)
CHUNK = 2_000_000  # resampled indices drawn per block, so that the memory of a long bootstrap stays bounded


@dataclass(frozen=True)
class PairedBrier:
    """The Brier difference (challenger minus champion) with its 95 percent interval over ``n_boot`` resamples."""

    diff: float
    lo: float
    hi: float
    n_boot: int

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("lo must not exceed hi")


def paired_brier_bootstrap(
    p_champion: np.ndarray, p_challenger: np.ndarray, y: np.ndarray, n_boot: int, seed: int
) -> PairedBrier:
    """The Brier difference (challenger minus champion) with a 95 percent percentile interval from ``n_boot`` resamples
    of the obligors drawn jointly (the same obligors score both forecasts). Deterministic for a ``seed``; identical
    forecasts give a zero difference and a zero-width interval. The Brier score is ``mean((p - y)^2)``."""
    champ, chall = vec(p_champion, "p_champion"), vec(p_challenger, "p_challenger")
    yy = binary(y, "y", both=False)
    same_rows(p_champion=champ, p_challenger=chall, y=yy)
    closed_unit(champ, "p_champion")
    closed_unit(chall, "p_challenger")
    b = index(n_boot, "n_boot", 10**9, lo=2)
    d = (chall - yy) ** 2 - (champ - yy) ** 2
    n = len(d)
    rng = np.random.default_rng(seed)
    per_block = max(1, CHUNK // n)
    means = np.empty(b)
    done = 0
    while done < b:
        m = min(per_block, b - done)
        means[done : done + m] = d[rng.integers(0, n, size=(m, n))].mean(axis=1)
        done += m
    alpha = (1.0 - LEVEL) / 2.0
    lo, hi = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return PairedBrier(diff=float(d.mean()), lo=float(lo), hi=float(hi), n_boot=b)


def grade_disagreement(pd_a: np.ndarray, pd_b: np.ndarray, edges: np.ndarray, notches: int = 2) -> float:
    """The share of obligors whose grades under the two PD vectors differ by at least ``notches`` (the grade of a PD is
    the number of ``edges`` at or below it). Symmetric in the two vectors."""
    a, b = vec(pd_a, "pd_a"), vec(pd_b, "pd_b")
    same_rows(pd_a=a, pd_b=b)
    closed_unit(a, "pd_a")
    closed_unit(b, "pd_b")
    e = vec(edges, "edges")
    if np.any(np.diff(e) <= 0.0):
        raise ValueError("edges must be strictly increasing")
    k = index(notches, "notches", 10**9, lo=1)
    ga, gb = np.searchsorted(e, a, side="right"), np.searchsorted(e, b, side="right")
    return float(np.mean(np.abs(ga - gb) >= k))
