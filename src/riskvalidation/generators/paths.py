"""Rating paths in continuous time, with a known generator and optional planted defects.

The Markov case is a continuous-time chain with generator Q: an obligor in state i waits an exponential time of rate
q_i = -q_ii and jumps to j with probability q_ij / q_i; default (the last state) is absorbing. Three defects can be
planted, each one a violation a validator should be able to detect, or be fooled by:

- **Momentum** (dos Reis, Pfeuffer and Smith 2020, Quantitative Finance, arXiv 1809.09889, section 4.2): a
  self-exciting downgrade intensity added to the baseline, lambda(t) = q_X(t) + sum_m sum_tau beta_m alpha_m
  exp(-beta_m (t - tau)), m = 1 for downgrades from an investment grade and m = 2 from a speculative grade, tau the
  obligor's earlier downgrade times of that type; the excitation goes to the downgrade destinations with q_jk > 0 in
  equal parts; no momentum at the first rating. The chain is then not Markov: the hazard of leaving a grade depends on
  how the obligor entered it.
- **Regimes**: the generator switches at given calendar times (a recession regime): the chain is Markov but not time
  homogeneous.
- **Withdrawals**: an exit at rate w from every non-default state; informative when the rate is w (1 + k) within the
  last ``window`` years before the obligor's default, so the ratings about to default are withdrawn more often (the
  bias the industry treatment of withdrawals cannot see, dossier 13 section 5 against EBA/GL/2017/16 paragraph 76).

Simulation is exact: Ogata's thinning with, for each obligor, a bound on its total intensity that holds until its
next event (the excitation only decays between events; the regimes' largest rate bounds the baseline), vectorised over
the obligors. Withdrawals are drawn after the path, from the piecewise exponential hazard, and truncate it.

Snapshots follow CEREP's stock model (ESMA, CEREP help file): the state of every obligor at given dates, a rating
withdrawn by a date counted in its start state's withdrawals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.linalg import expm

from riskvalidation.generators._base import Check, GeneratorBase, frozen_array
from riskvalidation.transitions.estimators import CENSORED_END, WITHDRAWN

DOS_REIS_2020 = (
    "dos Reis, Pfeuffer and Smith (2020), Capturing model risk and rating momentum in the estimation of probabilities "
    "of default and credit rating migrations, Quantitative Finance, arXiv 1809.09889"
)


def _check_generator(q: np.ndarray) -> None:
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError(f"a generator must be square, got shape {q.shape}")
    # every comparison below is False on NaN, and an infinite rate can balance an infinite diagonal: a NaN rate made a
    # draw run without end (found by Contraste's C04 review)
    if not np.all(np.isfinite(q)):
        raise ValueError("a generator's rates must be finite")
    off = ~np.eye(q.shape[0], dtype=bool)
    if np.any(q[off] < 0) or np.any(np.abs(q.sum(axis=1)) > 1e-9):
        raise ValueError("a generator has non-negative off-diagonal entries and rows summing to 0")
    if np.any(q[-1] != 0):
        raise ValueError("the last state (default) must be absorbing")


@dataclass(frozen=True)
class RatingPaths(GeneratorBase):
    """Obligors' rating paths from a generator, with momentum, regimes and withdrawals as planted defects.

    ``generator`` is the baseline Q (default last, absorbing); ``obligors`` the number starting in each non-default
    state; ``regimes`` a sequence of (start time, generator) applying from each start time on (the baseline before the
    first); ``momentum`` ((alpha_1, beta_1), (alpha_2, beta_2)) with ``investment_grades`` the number of investment
    grades (downgrades from them are type 1); ``withdrawal_rate`` w, ``informative`` k and ``window``; ``snapshots``
    the dates of the stock-model observations.
    """

    generator: tuple
    obligors: tuple
    horizon: float = 5.0
    regimes: tuple | None = None
    momentum: tuple | None = None
    investment_grades: int | None = None
    withdrawal_rate: float = 0.0
    informative: float = 0.0
    window: float = 1.0
    snapshots: tuple | None = None

    name: ClassVar[str] = "rating_paths"

    def __post_init__(self) -> None:
        q = np.asarray(self.generator, dtype=float)
        _check_generator(q)
        object.__setattr__(self, "generator", frozen_array(q, ndim=2))
        m = q.shape[0]
        ob = np.asarray(self.obligors, dtype=int)
        if ob.shape != (m - 1,) or np.any(ob < 0) or ob.sum() == 0:
            raise ValueError(f"obligors must give {m - 1} non-negative counts, one per non-default state")
        object.__setattr__(self, "obligors", frozen_array(ob, dtype=int, ndim=1))
        if not self.horizon > 0:
            raise ValueError("horizon must be positive")
        if self.regimes is not None:
            regs = []
            last = 0.0
            for start, qr in self.regimes:
                qa = np.asarray(qr, dtype=float)
                _check_generator(qa)
                if qa.shape != q.shape or not float(start) > last:
                    raise ValueError("regimes need increasing positive start times and generators of the same shape")
                last = float(start)
                regs.append((float(start), frozen_array(qa, ndim=2)))
            object.__setattr__(self, "regimes", tuple(regs))
        if self.momentum is not None:
            mom = tuple((float(a), float(b)) for a, b in self.momentum)
            if len(mom) != 2 or any(a < 0 or b <= 0 for a, b in mom):
                raise ValueError("momentum is ((alpha_1, beta_1), (alpha_2, beta_2)) with alpha >= 0 and beta > 0")
            if self.investment_grades is None or not 0 < self.investment_grades < m - 1:
                raise ValueError("momentum needs investment_grades, the number of investment grades")
            object.__setattr__(self, "momentum", mom)
        if self.withdrawal_rate < 0 or self.informative < 0 or self.window <= 0:
            raise ValueError("withdrawal_rate and informative must be non-negative and window positive")
        if self.snapshots is not None:
            snaps = tuple(float(s) for s in self.snapshots)
            if any(b <= a for a, b in zip(snaps, snaps[1:], strict=False)) or snaps[0] < 0 or snaps[-1] > self.horizon:
                raise ValueError("snapshots must increase within [0, horizon]")
            object.__setattr__(self, "snapshots", snaps)

    # ------------------------------------------------------------------------------------------------- structure
    @property
    def n_states(self) -> int:
        return len(self.generator)

    def _generators(self) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
        base = np.asarray(self.generator, dtype=float)
        starts = [0.0] + [s for s, _ in (self.regimes or ())]
        qs = [base] + [np.asarray(qr, dtype=float) for _, qr in (self.regimes or ())]
        return base, starts, qs

    # ------------------------------------------------------------------------------------------------- simulation
    def _simulate(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        base, starts, qs = self._generators()
        m = self.n_states
        d = m - 1
        state0 = np.repeat(np.arange(m - 1), np.asarray(self.obligors, dtype=int))
        n = state0.size
        out_rates = np.stack([-np.diag(q) for q in qs])  # regime x state
        rate_bound = out_rates.max(axis=0)  # the largest baseline exit rate of each state over the regimes
        mom = self.momentum
        ig = self.investment_grades or 0
        # cumulative jump probabilities q_ij / q_i, and the downgrade destinations with q_jk > 0 in equal parts (the
        # excitation's marks), per regime and state
        jump = np.stack([np.where(np.eye(m, dtype=bool), 0.0, q) for q in qs])
        rows = jump.sum(axis=2, keepdims=True)
        cum_jump = np.cumsum(np.divide(jump, rows, out=np.zeros_like(jump), where=rows > 0), axis=2)
        down = np.stack([np.triu(q > 0, k=1).astype(float) for q in qs])
        n_down = down.sum(axis=2, keepdims=True)
        cum_down = np.cumsum(np.divide(down, n_down, out=np.zeros_like(down), where=n_down > 0), axis=2)
        has_down = n_down[:, :, 0] > 0

        seg_ob, seg_start, seg_stop, seg_state, seg_to = [], [], [], [], []
        t = np.zeros(n)
        s = state0.copy()
        entered = np.zeros(n)
        exc = np.zeros((n, 2))  # excitation of each type at time t
        active = s != d
        # obligors already in default (none at start) or done are inactive
        while np.any(active):
            idx = np.flatnonzero(active)
            si = s[idx]
            bound = rate_bound[si] + (exc[idx].sum(axis=1) if mom else 0.0)
            gap = rng.exponential(1.0 / bound)
            cand = t[idx] + gap
            beyond = cand >= self.horizon
            # paths reaching the horizon end there, censored
            if np.any(beyond):
                b = idx[beyond]
                seg_ob.append(b), seg_start.append(entered[b]), seg_stop.append(np.full(b.size, self.horizon))
                seg_state.append(s[b]), seg_to.append(np.full(b.size, CENSORED_END))
                active[b] = False
            keep = ~beyond
            idx, si, cand, bound = idx[keep], si[keep], cand[keep], bound[keep]
            if idx.size == 0:
                continue
            # the intensities at the candidate time
            if mom:
                decay = np.exp(-np.outer(cand - t[idx], [mom[0][1], mom[1][1]]))
                exc_c = exc[idx] * decay
            else:
                exc_c = np.zeros((idx.size, 2))
            reg = np.searchsorted(starts, cand, side="right") - 1
            base_rate = out_rates[reg, si]
            # the excitation acts only where the grade has a downgrade to go to
            exc_c = exc_c * has_down[reg, si][:, None]
            total = base_rate + exc_c.sum(axis=1)
            accept = rng.random(idx.size) * bound < total
            # rejected candidates move the clock (and the excitation) forward with no event
            rej = idx[~accept]
            t[rej] = cand[~accept]
            exc[rej] = exc_c[~accept]
            acc = idx[accept]
            if acc.size == 0:
                continue
            ca, sa, ra = cand[accept], si[accept], reg[accept]
            ea, br = exc_c[accept], base_rate[accept]
            # which event: a baseline jump, or a momentum downgrade
            u = rng.random(acc.size) * (br + ea.sum(axis=1))
            is_momentum = u >= br
            # the destination: by the regime's q_ij / q_i for a baseline jump, evenly over the downgrades for momentum
            v = rng.random(acc.size)[:, None]
            table = np.where(is_momentum[:, None], cum_down[ra, sa], cum_jump[ra, sa])
            new = np.minimum((table < v).sum(axis=1), m - 1)
            seg_ob.append(acc), seg_start.append(entered[acc]), seg_stop.append(ca)
            seg_state.append(sa), seg_to.append(new)
            # momentum: a downgrade raises its type's excitation by beta alpha
            if mom:
                downgrade = new > sa
                typ = np.where(sa < ig, 0, 1)
                jumpsize = np.array([mom[0][1] * mom[0][0], mom[1][1] * mom[1][0]])
                ea[np.flatnonzero(downgrade), typ[downgrade]] += jumpsize[typ[downgrade]]
                exc[acc] = ea
            t[acc] = ca
            entered[acc] = ca
            s[acc] = new
            done = new == d
            if np.any(done):
                # a defaulted obligor stays in default until the horizon (absorbing, no further event)
                dd = acc[done]
                seg_ob.append(dd), seg_start.append(ca[done]), seg_stop.append(np.full(dd.size, self.horizon))
                seg_state.append(np.full(dd.size, d)), seg_to.append(np.full(dd.size, CENSORED_END))
                active[dd] = False
        seg = {
            "obligor": np.concatenate(seg_ob).astype(int),
            "start": np.concatenate(seg_start).astype(float),
            "stop": np.concatenate(seg_stop).astype(float),
            "state": np.concatenate(seg_state).astype(int),
            "to": np.concatenate(seg_to).astype(int),
        }
        order = np.lexsort((seg["start"], seg["obligor"]))
        seg = {k: v[order] for k, v in seg.items()}
        if self.withdrawal_rate > 0:
            seg = self._withdraw(seg, n, rng)
        return seg

    def _withdraw(self, seg: dict[str, np.ndarray], n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw each obligor's withdrawal time from the hazard w (or w (1 + k) within ``window`` before its default),
        running while it is not in default, and truncate its path there."""
        d = self.n_states - 1
        w, k, win = self.withdrawal_rate, self.informative, self.window
        default_time = np.full(n, np.inf)
        hit = seg["to"] == d
        default_time[seg["obligor"][hit]] = seg["stop"][hit]
        end = np.minimum(default_time, self.horizon)
        # piecewise hazard: w on [0, end - window), w (1 + k) on [end - window, end) when a default comes
        e = rng.exponential(1.0, size=n)
        cut = np.where(np.isfinite(default_time), np.maximum(end - win, 0.0), end)
        first = cut * w
        wtime = np.where(e < first, e / w, cut + (e - first) / (w * (1.0 + k)))
        wtime = np.where(np.isfinite(default_time) | (wtime < end), wtime, np.inf)
        wtime[wtime >= end] = np.inf
        ob = seg["obligor"]
        wt = wtime[ob]
        keep = seg["start"] < wt
        cut_here = keep & (seg["stop"] > wt)
        stop = np.where(cut_here, wt, seg["stop"])
        to = np.where(cut_here, WITHDRAWN, seg["to"])
        return {"obligor": ob[keep], "start": seg["start"][keep], "stop": stop[keep], "state": seg["state"][keep],
                "to": to[keep]}

    # ------------------------------------------------------------------------------------------------- snapshots
    def snapshot_states(self, seg: dict[str, np.ndarray], times: tuple[float, ...]) -> np.ndarray:
        """Each obligor's state at each date (-2 once withdrawn), as CEREP's stock model sees it."""
        n = int(seg["obligor"].max()) + 1 if seg["obligor"].size else 0
        out = np.full((n, len(times)), WITHDRAWN, dtype=int)
        for c, tau in enumerate(times):
            ends_here = (seg["stop"] == tau) & (seg["to"] == CENSORED_END)
            inside = (seg["start"] <= tau) & ((seg["stop"] > tau) | ends_here)
            out[seg["obligor"][inside], c] = seg["state"][inside]
        return out

    def snapshot_counts(self, states: np.ndarray) -> list[dict[str, np.ndarray]]:
        """For each step between consecutive dates: the m x m count matrix and each start state's withdrawals."""
        m = self.n_states
        steps = []
        for c in range(states.shape[1] - 1):
            a, b = states[:, c], states[:, c + 1]
            rated = a >= 0
            counts = np.zeros((m, m))
            stay = rated & (b >= 0)
            np.add.at(counts, (a[stay], b[stay]), 1.0)
            withdrawn = np.bincount(a[rated & (b == WITHDRAWN)], minlength=m).astype(float)
            steps.append({"counts": counts, "withdrawn": withdrawn})
        return steps

    def triplet_counts(self, states: np.ndarray) -> np.ndarray:
        """n_ijk over consecutive triples of dates (obligors rated at all three), for the order test."""
        m = self.n_states
        n = np.zeros((m, m, m))
        for c in range(states.shape[1] - 2):
            a, b, z = states[:, c], states[:, c + 1], states[:, c + 2]
            ok = (a >= 0) & (b >= 0) & (z >= 0)
            np.add.at(n, (a[ok], b[ok], z[ok]), 1.0)
        return n

    # ------------------------------------------------------------------------------------------------- contract
    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        seg = self._simulate(rng)
        out: dict[str, Any] = {"segments": seg, "n_states": self.n_states}
        if self.snapshots is not None:
            states = self.snapshot_states(seg, self.snapshots)
            out["snapshot_states"] = states
            out["steps"] = self.snapshot_counts(states)
            out["triplets"] = self.triplet_counts(states)
        return out

    def truth(self) -> dict[str, Any]:
        base, starts, qs = self._generators()
        m = self.n_states
        horizons = [1.0, 2.0, 3.0, 4.0, 5.0]
        markov_homogeneous = self.momentum is None and self.regimes is None
        return {
            "generator": base,
            "regimes": [{"start": s, "generator": q} for s, q in zip(starts, qs, strict=True)],
            "markov": self.momentum is None,
            "time_homogeneous": self.regimes is None,
            "one_year_matrix": expm(base),
            "pd_by_grade": {f"{h:g}": expm(h * base)[: m - 1, m - 1] for h in horizons},
            "step_matrices": (
                [expm((b - a) * base) for a, b in zip(self.snapshots, self.snapshots[1:], strict=False)]
                if self.snapshots is not None and markov_homogeneous else None
            ),
            "withdrawals": {"rate": self.withdrawal_rate, "informative": self.informative, "window": self.window},
            "reference": DOS_REIS_2020 if self.momentum else "continuous-time Markov chain with generator Q",
        }

    def verify(self, n_draws: int = 1, seed: int = 20261006) -> list[Check]:
        """In the homogeneous Markov case without withdrawals, the jumps per unit of time in each state reproduce q_ij:
        K_ij / S_i against q_ij with the Poisson standard error sqrt(K_ij) / S_i (at least 200 expected jumps)."""
        if self.momentum is not None or self.regimes is not None or self.withdrawal_rate > 0:
            raise ValueError("verify checks the homogeneous Markov case")
        rng = np.random.default_rng(seed)
        q = np.asarray(self.generator, dtype=float)
        m = self.n_states
        k = np.zeros((m, m))
        s = np.zeros(m)
        for _ in range(n_draws):
            seg = self._simulate(rng)
            np.add.at(s, seg["state"], seg["stop"] - seg["start"])
            moved = seg["to"] >= 0
            np.add.at(k, (seg["state"][moved], seg["to"][moved]), 1.0)
        checks = []
        for i in range(m - 1):
            for j in range(m):
                if i == j or q[i, j] * s[i] < 200:
                    continue
                checks.append(Check(f"q[{i},{j}]", float(q[i, j]), float(k[i, j] / s[i]),
                                    float(math.sqrt(q[i, j] * s[i]) / s[i])))
        return checks

__all__ = [
    "RatingPaths",
]
