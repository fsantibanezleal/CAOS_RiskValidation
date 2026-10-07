r"""Confidence intervals for a PD by grade: Wald, Agresti-Coull (with the dependence correction), Jeffreys, bootstrap.

Schuermann and Hanson (2004), Estimating probabilities of default, FRBNY Staff Report 190 (read at the page images;
published as Hanson and Schuermann 2006, Journal of Banking and Finance 30(8), 2281-2301):

- Wald (2.2): $CI_W = \widehat{PD}_R \pm \kappa\sqrt{\widehat{PD}_R(1 - \widehat{PD}_R)/N^*_R}$, $N^*_R$ the firm-years
  spent in grade $R$, $\kappa$ the $100(1 - \alpha/2)$th percentile of the standard normal. It "assumes that
  $\widehat{PD}_R$ is estimated from a set of iid draws ... This clearly seems unreasonable ... the Wald confidence
  interval described by (2.2) will likely be too tight". Rule of thumb $\widehat{PD}_R N^*_R \ge 10$.
- Agresti-Coull (3.2), (3.3): the centre $\widetilde{PD}_R = \tilde N_{R,D} / \tilde N_R$ with
  $\tilde N_{R,D} = N_{R,D} + \kappa^2/2$ and $\tilde N_R = N_R + \kappa^2$, and
  $CI_{AC} = \widetilde{PD}_R \pm \kappa\sqrt{\widetilde{PD}_R(1 - \widetilde{PD}_R)/\tilde N_R}$ ("add 2 successes
  and 2 failures", Agresti and Coull 1998, as quoted there).
- Dependence (3.4), after Miao and Gastwirth (2004): with a default correlation $\rho_{ij}$ between obligors,
  $N^\dagger_R = [1/N_R + (2/N_R^2)\sum_{i<j}\sqrt{N_{i,R}N_{j,R}}\,\rho_{ij}]^{-1}$, $N_{i,R}$ the trials (years) of
  obligor $i$ in grade $R$; $N^\dagger_R$ replaces $N_R$ in (2.2) and (3.2), and the Agresti-Coull centre becomes
  $(\widehat{PD}_R N^\dagger_R + \kappa^2/2)/(N^\dagger_R + \kappa^2)$. Both intervals are bounded to $[0, 1]$, as
  their Table 5 prints them (BB, 2002, 531 obligors: reproduced in ``tests/test_reproductions.py``).
- Bootstrap (section 2.3): "By resampling on the rating histories, we create B bootstrap samples ... compute the
  entire migration matrix ... and then focus our attention just on the last vector", B = 10,000 (Efron and Tibshirani:
  1,000 for intervals). The parametric alternative of Christensen, Hansen and Lando (2004), as they describe it:
  "estimate an intensity based migration matrix using all the available data and then generate many, say B*,
  synthetic rating histories from that estimated migration matrix ... compute B* intensity based migration matrices
  ... a simulation-based confidence interval from the default columns".

Jeffreys: the equal-tailed interval of the Beta(D + 1/2, N - D + 1/2) posterior, with the lower bound 0 when D = 0
and the upper bound 1 when D = N (Brown, Cai and DasGupta 2001, doi:10.1214/ss/1009213286, as defined in Zhou, Li and
Yang 2008, doi:10.1098/rsta.2008.0037, Appendix C), the posterior of the ECB's Jeffreys test (``pd.jeffreys``).

Exact properties of the analytical intervals for a binomial count (independent defaults), finite sums over the count
and therefore exact, never simulated: the coverage $C(p) = \sum_{d=0}^{N}\binom{N}{d}p^d(1-p)^{N-d}\,1\{p \in CI(d)\}$
and the expected length (``exact_coverage``), and the probability that the intervals of two grades overlap
(``overlap_probability``), the measure of Schuermann and Hanson's finding that investment-grade PDs "cannot be
distinguished" at the notch level.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats
from scipy.linalg import expm

from riskvalidation.transitions.estimators import duration_generator

SCHUERMANN_HANSON_2004 = (
    "Schuermann and Hanson (2004), Estimating probabilities of default, FRBNY Staff Report 190; Hanson and "
    "Schuermann (2006), Journal of Banking and Finance 30(8), 2281-2301"
)
JEFFREYS = (
    "Brown, Cai and DasGupta (2001), doi:10.1214/ss/1009213286, as defined in Zhou, Li and Yang (2008), "
    "doi:10.1098/rsta.2008.0037, Appendix C"
)
CHL_2004 = (
    "Christensen, Hansen and Lando (2004), Confidence sets for continuous-time rating transition probabilities, "
    "Journal of Banking and Finance 28(11), 2575-2602, doi:10.1016/j.jbankfin.2004.06.003, as described in "
    "Schuermann and Hanson (2004), section 2.3"
)


def _counts(defaults: Any, n: Any) -> tuple[np.ndarray, np.ndarray, bool]:
    d = np.asarray(defaults, dtype=float)
    nn = np.asarray(n, dtype=float)
    scalar = d.ndim == 0 and nn.ndim == 0
    d, nn = np.broadcast_arrays(np.atleast_1d(d), np.atleast_1d(nn))
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(nn))):
        raise ValueError("defaults and n must be finite")
    if np.any(nn <= 0):
        raise ValueError("an empty grade (n = 0) has no PD to bound")
    if np.any(d < 0) or np.any(d > nn):
        raise ValueError("defaults must lie between 0 and n")
    return d.astype(float), nn.astype(float), scalar


def _z(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    return float(stats.norm.ppf(0.5 + level / 2.0))


def _out(method: str, est: np.ndarray, lo: np.ndarray, hi: np.ndarray, n: np.ndarray, n_eff: np.ndarray,
         level: float, scalar: bool, reference: str, **extra: Any) -> dict[str, Any]:
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)

    def pick(a: np.ndarray) -> Any:
        return float(a[0]) if scalar else a

    return {"method": method, "estimate": pick(est), "lower": pick(lo), "upper": pick(hi), "length": pick(hi - lo),
            "n": pick(n), "n_effective": pick(n_eff), "level": level, "reference": reference, **extra}


def effective_n(n: Any, rho: float = 0.0, trials: Any = None) -> Any:
    r"""Schuermann and Hanson's (3.4): $N^\dagger = [1/N + (2/N^2)\sum_{i<j}\sqrt{N_i N_j}\,\rho]^{-1}$ with one
    correlation $\rho$ between every pair. ``trials`` gives each obligor's $N_i$ (``n`` is then their sum); without
    it each of the ``n`` obligors is one trial, and $N^\dagger = N / (1 + (N - 1)\rho)$."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    nn = np.asarray(n, dtype=float)
    if trials is None:
        pairs = nn * (nn - 1.0) / 2.0
    else:
        t = np.asarray(trials, dtype=float)
        if t.ndim != 1 or np.any(t < 0) or nn.ndim != 0 or not np.isclose(t.sum(), nn):
            raise ValueError("trials must be one obligor's count each, summing to n")
        pairs = (np.sqrt(t).sum() ** 2 - t.sum()) / 2.0
    out = 1.0 / (1.0 / nn + 2.0 * pairs * rho / nn**2)
    return float(out) if np.ndim(out) == 0 else out


def _n_eff(nn: np.ndarray, rho: float, trials: Any, scalar: bool) -> np.ndarray:
    if trials is not None and not scalar:
        raise ValueError("trials describe one grade's obligors: pass one grade at a time")
    if rho == 0 and trials is None:
        return nn
    if trials is None:
        return np.asarray(effective_n(nn, rho), dtype=float)
    return np.array([effective_n(float(nn[0]), rho, trials)])


def pd_wald(defaults: Any, n: Any, *, level: float = 0.95, rho: float = 0.0, trials: Any = None) -> dict[str, Any]:
    """The Wald interval (2.2) on D defaults in N firm-years (or obligors), N replaced by the effective N of (3.4)
    when ``rho`` > 0; bounded to [0, 1]. Vectorised over grades."""
    d, nn, scalar = _counts(defaults, n)
    z = _z(level)
    n_eff = _n_eff(nn, rho, trials, scalar)
    p = d / nn
    half = z * np.sqrt(p * (1.0 - p) / n_eff)
    return _out("wald", p, p - half, p + half, nn, n_eff, level, scalar, f"{SCHUERMANN_HANSON_2004}, equation (2.2)",
                rule_of_thumb=(p * nn >= 10.0) if not scalar else bool(p[0] * nn[0] >= 10.0))


def pd_agresti_coull(defaults: Any, n: Any, *, level: float = 0.95, rho: float = 0.0,
                     trials: Any = None) -> dict[str, Any]:
    """The Agresti-Coull interval (3.2)-(3.3), with the effective N of (3.4) when ``rho`` > 0; bounded to [0, 1]."""
    d, nn, scalar = _counts(defaults, n)
    z = _z(level)
    n_eff = _n_eff(nn, rho, trials, scalar)
    p_hat = d / nn
    n_tilde = n_eff + z**2
    centre = (p_hat * n_eff + z**2 / 2.0) / n_tilde
    half = z * np.sqrt(centre * (1.0 - centre) / n_tilde)
    return _out("agresti_coull", centre, centre - half, centre + half, nn, n_eff, level, scalar,
                f"{SCHUERMANN_HANSON_2004}, equations (3.2) to (3.4); Agresti and Coull (1998); Miao and Gastwirth "
                "(2004)", observed_rate=float(p_hat[0]) if scalar else p_hat)


def pd_jeffreys(defaults: Any, n: Any, *, level: float = 0.95) -> dict[str, Any]:
    """The equal-tailed Jeffreys interval, 0 at D = 0 and 1 at D = N; the estimate is the observed rate D / N."""
    d, nn, scalar = _counts(defaults, n)
    _z(level)
    a, b = d + 0.5, nn - d + 0.5
    lo = np.where(d > 0, stats.beta.ppf((1.0 - level) / 2.0, a, b), 0.0)
    hi = np.where(d < nn, stats.beta.ppf(0.5 + level / 2.0, a, b), 1.0)
    return _out("jeffreys", d / nn, lo, hi, nn, nn, level, scalar, JEFFREYS)


def _per_obligor(seg: Mapping[str, np.ndarray], m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each obligor's jump counts (obligors x m*m), holding times (obligors x m), first state and last stop."""
    ob = np.asarray(seg["obligor"])
    ids, idx = np.unique(ob, return_inverse=True)
    state = np.asarray(seg["state"], dtype=int)
    to = np.asarray(seg["to"], dtype=int)
    start = np.asarray(seg["start"], dtype=float)
    stop = np.asarray(seg["stop"], dtype=float)
    k = np.zeros((ids.size, m * m))
    moved = to >= 0
    np.add.at(k, (idx[moved], state[moved] * m + to[moved]), 1.0)
    s = np.zeros((ids.size, m))
    np.add.at(s, (idx, state), stop - start)
    order = np.lexsort((start, idx))  # by obligor, then by start: each obligor's earliest segment first
    _, first_at = np.unique(idx[order], return_index=True)
    return k, s, state[order][first_at], start[order][first_at]


def _pd_from(k: np.ndarray, s: np.ndarray, m: int, d: int, horizon: float) -> np.ndarray:
    """PD by state at the horizon from aggregated jumps (m*m) and holding times (m): exp(horizon Q)[:, d]."""
    q = np.zeros((m, m))
    occupied = s > 0
    q[occupied] = k.reshape(m, m)[occupied] / s[occupied, None]
    q[d] = 0.0
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    return expm(horizon * q)[:, d]


def pd_bootstrap(paths: Mapping[str, Any], n_states: int, *, method: str = "resample", horizon: float = 1.0,
                 n_boot: int = 1000, level: float = 0.95, window: float | None = None, default_state: int = -1,
                 seed: int = 0, keep: bool = False) -> dict[str, Any]:
    """Percentile intervals for the duration-based PD by grade at ``horizon``, exp(horizon Q^)[:, default].

    ``method="resample"`` (Schuermann and Hanson's section 2.3) draws obligors' histories with replacement and
    re-estimates; ``method="parametric"`` (Christensen, Hansen and Lando, as described there) simulates every obligor
    from the estimated generator, starting in its first observed state, over ``window`` (default: the latest stop in
    the data), and re-estimates; default is then the last state, as in the rating-path generator. Replicates are
    drawn with ``seed``. A grade with no time spent in it is refused."""
    if method not in ("resample", "parametric"):
        raise ValueError("method is 'resample' or 'parametric'")
    if n_boot < 2 or not horizon > 0:
        raise ValueError("n_boot must be at least 2 and horizon positive")
    _z(level)
    rng = np.random.default_rng(seed)
    m = int(n_states)
    d = default_state % m
    if method == "parametric" and d != m - 1:
        raise ValueError("the parametric bootstrap simulates with default as the last state")
    fit = duration_generator(paths, m, default_state=d)
    empty = [i for i in range(m) if i != d and fit["holding"][i] <= 0]
    if empty:
        raise ValueError(f"grades {empty} have no time spent in them: no PD to bound")
    estimate = expm(horizon * fit["generator"])[:, d]
    seg = {key: np.asarray(paths[key]) for key in ("obligor", "start", "stop", "state", "to")}
    reps = np.empty((n_boot, m))
    if method == "resample":
        k, s, _, _ = _per_obligor(seg, m)
        n_ob = k.shape[0]
        for b in range(n_boot):
            c = np.bincount(rng.integers(0, n_ob, n_ob), minlength=n_ob).astype(float)
            reps[b] = _pd_from(c @ k, c @ s, m, d, horizon)
        reference = f"{SCHUERMANN_HANSON_2004}, section 2.3"
    else:
        from riskvalidation.generators.paths import RatingPaths

        _, _, first, _ = _per_obligor(seg, m)
        span = float(np.max(seg["stop"])) if window is None else float(window)
        obligors = np.bincount(first[(first >= 0) & (first != d)], minlength=m)[:m]
        gen = RatingPaths(fit["generator"], tuple(int(x) for x in np.delete(obligors, d)), horizon=span)
        for b in range(n_boot):
            sim = gen.draw(rng)["segments"]
            f = duration_generator(sim, m, default_state=d)
            reps[b] = expm(horizon * f["generator"])[:, d]
        reference = CHL_2004
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, a, axis=0), np.quantile(reps, 1.0 - a, axis=0)
    grades = [i for i in range(m) if i != d]
    out = {"method": f"bootstrap_{method}", "estimate": estimate[grades], "lower": lo[grades], "upper": hi[grades],
           "length": (hi - lo)[grades], "se": reps[:, grades].std(axis=0, ddof=1), "n_boot": int(n_boot),
           "horizon": float(horizon), "level": level, "grades": grades, "seed": int(seed), "reference": reference}
    if keep:
        out["replicates"] = reps[:, grades]
    return out


#: the analytical intervals ``exact_coverage`` and ``overlap_probability`` enumerate
INTERVALS = {"wald": pd_wald, "agresti_coull": pd_agresti_coull, "jeffreys": pd_jeffreys}


def _bounds(method: str, n: int, level: float) -> tuple[np.ndarray, np.ndarray]:
    """The interval of every count 0 to n of n obligors (independent defaults: no dependence correction)."""
    if method not in INTERVALS:
        raise ValueError(f"unknown interval {method!r}: one of {sorted(INTERVALS)}")
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive whole number of obligors")
    n = int(n)
    r = INTERVALS[method](np.arange(n + 1, dtype=float), np.full(n + 1, float(n)), level=level)
    return np.asarray(r["lower"], dtype=float), np.asarray(r["upper"], dtype=float)


def _pds(pd: Any) -> tuple[np.ndarray, bool]:
    p = np.asarray(pd, dtype=float)
    scalar = p.ndim == 0
    p = np.atleast_1d(p)
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("a true PD must lie in [0, 1]")
    return p, scalar


def exact_coverage(method: str, n: int, pd: Any, *, level: float = 0.95) -> dict[str, Any]:
    """The exact coverage and expected length of ``method``'s interval for ``n`` obligors whose defaults are
    independent with probability ``pd`` (a scalar or an array of true PDs, one result each): sums over the binomial
    count 0 to n. ``method`` is ``"wald"``, ``"agresti_coull"`` or ``"jeffreys"``."""
    lo, hi = _bounds(method, n, level)
    p, scalar = _pds(pd)
    pmf = stats.binom.pmf(np.arange(lo.size)[None, :], lo.size - 1, p[:, None])  # true PD x count
    inside = (lo[None, :] <= p[:, None]) & (p[:, None] <= hi[None, :])
    coverage = (pmf * inside).sum(axis=1)
    length = pmf @ (hi - lo)

    def pick(a: np.ndarray) -> Any:
        return float(a[0]) if scalar else a

    return {"method": method, "n": int(n), "level": level, "pd": pick(p), "coverage": pick(coverage),
            "expected_length": pick(length), "reference": "exact: the binomial count enumerated, independent defaults"}


def overlap_probability(method: str, n: int, pd_a: float, pd_b: float, *, level: float = 0.95,
                        n_b: int | None = None) -> float:
    """The probability that the intervals of two grades overlap: grade a with ``n`` obligors and true PD ``pd_a``,
    grade b with ``n_b`` (default ``n``) and ``pd_b``, their defaults independent, summed exactly over every pair of
    counts. Near 1 for two grades whose PDs the intervals cannot tell apart at these sizes."""
    lo_a, hi_a = _bounds(method, n, level)
    lo_b, hi_b = (lo_a, hi_a) if n_b is None or n_b == n else _bounds(method, n_b, level)
    pa, _ = _pds(pd_a)
    pb, _ = _pds(pd_b)
    if pa.size != 1 or pb.size != 1:
        raise ValueError("overlap_probability takes one true PD per grade")
    pmf_a = stats.binom.pmf(np.arange(lo_a.size), lo_a.size - 1, pa[0])
    pmf_b = stats.binom.pmf(np.arange(lo_b.size), lo_b.size - 1, pb[0])
    meet = (lo_a[:, None] <= hi_b[None, :]) & (lo_b[None, :] <= hi_a[:, None])
    return float(pmf_a @ meet.astype(float) @ pmf_b)


__all__ = ["INTERVALS", "effective_n", "exact_coverage", "overlap_probability", "pd_agresti_coull", "pd_bootstrap",
           "pd_jeffreys", "pd_wald"]
