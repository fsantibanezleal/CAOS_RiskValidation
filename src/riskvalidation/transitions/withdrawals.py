r"""The PD by grade under informative rating withdrawals: the tipping-point family PD(k), its Manski bounds and the
anchor that pins k down where a second source counts the defaults of the same ratings.

**The problem.** A one-year cohort of a grade holds $N$ ratings at its start; at its end $D$ are in default, $S$ are
still rated and not in default, and $W$ were withdrawn. The industry treatment removes the withdrawals and renormalises
(``cohort``: $D/(D+S)$, Schuermann and Hanson 2004, footnote 4; CEREP's transition page); the alternative keeps them in
the denominator unfollowed ($D/N$); EBA/GL/2017/16 paragraph 76 asks to follow them. Each of these is right only under
one assumption about why a rating is withdrawn, and the data cannot say which: a withdrawn rating that would have
defaulted is not seen to default. When withdrawal is informative (more likely in the year before default) every one of
them understates the PD, and no test of the matrix sees it.

**The model.** Of the $W$ withdrawn ratings, $X$ would have defaulted within the period. Let the odds of being
withdrawn, among the ratings that would default within the period, be $k$ times the odds among those that would not:

$$ k = \frac{X/D}{(W - X)/S} \quad\Longrightarrow\quad X(k) = W\,\frac{kD}{kD + S}, \qquad
   \mathrm{PD}(k) = \frac{D + X(k)}{N}, \quad N = D + S + W. $$

$X$ is the number of would-be defaulters among the withdrawn: the would-be defaulters are $D + X$, of whom $X$ were
withdrawn (odds $X/D$); the others are $S + W - X$, of whom $W - X$ were withdrawn (odds $(W - X)/S$). $\mathrm{PD}(k)$
is increasing in $k$, from $D/N$ at $k = 0$ (nobody withdrawn would have defaulted: the unfollowed treatment) through
$D/(D + S)$ at $k = 1$ (withdrawal independent of default, the usual estimate: the industry treatment) to
$(D + W)/N$ as $k \to \infty$ (every withdrawal was a default in waiting). The last is Manski's (1989,
doi:10.2307/145818) worst-case bound for a missing-not-at-random outcome; the family $\mathrm{PD}(k)$ is the delta-
adjusted tipping-point analysis of clinical trials (Cro et al. 2020, doi:10.1002/sim.8569) with the odds ratio $k$ as
the sensitivity parameter. The **tipping k** of a decision is the $k$ at which $\mathrm{PD}(k)$ crosses a threshold (a
master-scale boundary, an IRB risk-weight step): the informativeness it takes to flip the decision, to be judged
against what is plausible rather than assumed away.

**The anchor.** $k$ is not identified by the transition page alone. Where a second source counts the defaults of the
same grade differently, for example CEREP's default page against its transition page (the default page counts a rating
that defaulted within the period whether or not it was withdrawn before the period's end), that source gives a count
$A$ of defaults among $N_a$ ratings. Setting $A/N_a = \mathrm{PD}(k)$ gives the anchored $k$,
$X = N A/N_a - D$ clipped to $[0, W]$ and $k = X S/(D (W - X))$; the Jeffreys interval of $A/N_a$ (Brown, Cai and
DasGupta 2001) gives $[k_{\rm low}, k_{\rm high}]$ through the same inversion, and the anchored PD interval is the
Jeffreys lower bound of $(D + X_{\rm low})/N$ to the Jeffreys upper bound of $(D + X_{\rm high})/N$. That interval is
conditional on the two sources' definitions of default being the same (they are not exactly: ESMA sets no common
definition), and it counts the sampling noise of both pages, so it is conservative.

**Assumptions** (stated in every result): withdrawal is the only reason a rating leaves the cohort unobserved; $k$ is
one number per grade and period; a withdrawn rating that would default does so within the period; the cohort is one
period, so a rating withdrawn in an earlier period (a selection the single-period model does not see) is outside it;
the anchor's definition of default equals the transition page's. A grade with no observed default ($D = 0$) has
$X(k) = 0$ for every finite $k$: the odds relative to zero defaults are undefined, only the Manski bound $W/N$ is
valid, and the result says so (reason ``no_observed_defaults``).

Units: counts are numbers of ratings, PDs are probabilities per period (a year for a one-year cohort), $k$ is a ratio
of odds (dimensionless). Everything here is closed form: no random numbers are drawn, so the output is deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from riskvalidation.transitions.intervals import pd_jeffreys

MANSKI_1989 = "Manski (1989), Anatomy of the selection problem, Journal of Human Resources 24(3), doi:10.2307/145818"
CRO_2020 = (
    "Cro et al. (2020), Sensitivity analysis for clinical trials with missing continuous outcome data using "
    "controlled multiple imputation: a practical guide, Statistics in Medicine 39(21), doi:10.1002/sim.8569"
)
REFERENCE = (
    f"{MANSKI_1989}; {CRO_2020}; withdrawals: Schuermann and Hanson (2004), FRBNY Staff Report 190, footnote 4, and "
    "EBA/GL/2017/16 paragraph 76; Jeffreys interval: Brown, Cai and DasGupta (2001), doi:10.1214/ss/1009213286"
)
ASSUMPTIONS = (
    "withdrawal is the only reason a rating leaves the cohort unobserved",
    "k, the odds ratio of withdrawal for ratings that default within the period, is one number per grade and period",
    "a withdrawn rating that would default does so within the period",
    "the anchor source's definition of default equals the transition page's",
)

#: tolerance on a count that should lie in [0, W] before it is clipped and reported
_TOL = 1e-9

__all__ = [
    "ASSUMPTIONS",
    "CRO_2020",
    "MANSKI_1989",
    "odds_multiplier",
    "tipping_k",
    "withdrawal_bounds",
    "withdrawal_pd",
    "withdrawn_defaulters",
]


def _dsw(defaults: Any, survivors: Any, withdrawn: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three counts of a grade as float arrays of one shape; finite and non-negative."""
    arrays = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (defaults, survivors, withdrawn))
    d, s, w = np.broadcast_arrays(*arrays)
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
        raise ValueError("defaults, survivors and withdrawn must be finite")
    if np.any(d < 0) or np.any(s < 0) or np.any(w < 0):
        raise ValueError("defaults, survivors and withdrawn must be non-negative")
    return d.copy(), s.copy(), w.copy()


def _k(k: float) -> float:
    kk = float(k)
    if np.isnan(kk) or kk < 0:
        raise ValueError("k, an odds ratio, must be non-negative (np.inf is allowed: Manski's bound)")
    return kk


def withdrawn_defaulters(defaults: Any, survivors: Any, withdrawn: Any, k: float) -> np.ndarray:
    """The expected number $X(k) = W\\,kD/(kD + S)$ of withdrawn ratings that would have defaulted within the period,
    when the odds of withdrawal of a would-be defaulter are ``k`` times those of a rating that would not default.

    ``defaults`` $D$, ``survivors`` $S$ (rated and not in default at the period's end) and ``withdrawn`` $W$ are
    counts of ratings, per grade; ``k`` is a ratio of odds in $[0, \\infty]$. $X(0) = 0$, $X(1) = WD/(D + S)$
    (withdrawal independent of default), $X(\\infty) = W$. A grade with $D = 0$ gives 0 for every finite $k$ (see the
    module text).
    """
    d, s, w = _dsw(defaults, survivors, withdrawn)
    kk = _k(k)
    if np.isinf(kk):
        return w
    den = kk * d + s
    return np.divide(w * kk * d, den, out=np.zeros_like(w), where=den > 0)


def withdrawal_pd(defaults: Any, survivors: Any, withdrawn: Any, k: float) -> np.ndarray:
    """The PD of a grade at the odds ratio ``k`` of withdrawal: $\\mathrm{PD}(k) = (D + X(k))/N$, $N = D + S + W$ the
    cohort at the period's start (a probability per period). $k = 0$ is the unfollowed treatment $D/N$, $k = 1$ the
    industry treatment $D/(D + S)$, $k = \\infty$ Manski's upper bound $(D + W)/N$. A grade with an empty cohort
    ($N = 0$) is NaN. Increasing in $k$.
    """
    d, s, w = _dsw(defaults, survivors, withdrawn)
    n = d + s + w
    x = withdrawn_defaulters(d, s, w, k)
    return np.divide(d + x, n, out=np.full_like(n, np.nan), where=n > 0)


def odds_multiplier(defaults: Any, survivors: Any, withdrawn: Any, extra_defaults: Any) -> np.ndarray:
    """The odds ratio $k = XS/(D(W - X))$ at which $X(k)$ equals ``extra_defaults`` $X$, the number of withdrawn ratings
    that would have defaulted (the inverse of ``withdrawn_defaulters``). $X = 0$ gives $k = 0$, $X = W$ gives
    $\\infty$; NaN where it is not identified ($D = 0$ or $W = 0$). $X$ must lie in $[0, W]$ (a tolerance of
    $10^{-9}$ is clipped), else ``ValueError``.
    """
    d, s, w = _dsw(defaults, survivors, withdrawn)
    x = np.broadcast_to(np.asarray(extra_defaults, dtype=float), d.shape).astype(float)
    if not np.all(np.isfinite(x)) or np.any(x < -_TOL) or np.any(x > w + _TOL):
        raise ValueError("extra_defaults must lie between 0 and the withdrawn count")
    x = np.clip(x, 0.0, w)
    k = np.full(d.shape, np.nan)
    ok = (d > 0) & (w > 0)
    full = ok & (x >= w)
    part = ok & ~full
    k[full] = np.inf
    k[part] = x[part] * s[part] / (d[part] * (w[part] - x[part]))
    return k


def tipping_k(defaults: Any, survivors: Any, withdrawn: Any, threshold: Any) -> np.ndarray:
    """The odds ratio $k^*$ at which $\\mathrm{PD}(k^*)$ equals ``threshold`` (a PD, scalar or per grade), the
    informativeness of withdrawal that flips a decision taken at the threshold (a master-scale boundary).

    $X^* = t N - D$ must lie in $[0, W]$ for the threshold to be reached; then $k^* = X^* S/(D (W - X^*))$ (0 at
    $X^* = 0$, $\\infty$ at $X^* = W$). NaN where $\\mathrm{PD}(k)$ never reaches the threshold for any $k \\in [0,
    \\infty]$ (the decision is robust to withdrawal); for $D = 0$ the answer is $\\infty$ when $W/N \\ge t$ (the
    threshold is reached only at Manski's bound) and NaN otherwise.
    """
    d, s, w = _dsw(defaults, survivors, withdrawn)
    t = np.broadcast_to(np.asarray(threshold, dtype=float), d.shape).astype(float)
    if not np.all(np.isfinite(t)) or np.any(t <= 0) or np.any(t >= 1):
        raise ValueError("threshold must lie in (0, 1)")
    n = d + s + w
    xs = t * n - d
    out = np.full(d.shape, np.nan)
    reach = (n > 0) & (xs >= -_TOL) & (xs <= w + _TOL)
    xs = np.clip(xs, 0.0, w)
    ident = reach & (d > 0) & (w > 0)
    out[ident] = odds_multiplier(d[ident], s[ident], w[ident], xs[ident])
    out[reach & (d == 0) & (xs > 0)] = np.inf
    out[reach & (w == 0) & (d > 0)] = np.nan  # no withdrawal: the PD is D/N for every k
    return out


def _grade_counts(
    counts: Any, withdrawn: Any, default_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    n = np.asarray(counts, dtype=float)
    if n.ndim != 2 or n.shape[0] != n.shape[1]:
        raise ValueError(f"counts must be a square matrix, got shape {n.shape}")
    if not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("counts must be finite and non-negative")
    m = n.shape[0]
    w = np.asarray(withdrawn, dtype=float)
    if w.shape != (m,) or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError(f"withdrawn must be {m} non-negative numbers")
    d = default_state % m
    grades = [i for i in range(m) if i != d]
    dd = n[grades, d]
    ss = n[grades].sum(axis=1) - dd
    return dd, ss, w[grades], grades


def withdrawal_bounds(
    counts: Sequence[Sequence[float]],
    withdrawn: Sequence[float],
    *,
    anchor_defaults: Sequence[float] | None = None,
    anchor_obligors: Sequence[float] | float | None = None,
    thresholds: Sequence[float] | float | None = None,
    k_grid: Sequence[float] = (0.0, 1.0, 3.0, 9.0, np.inf),
    level: float = 0.95,
    default_state: int = -1,
) -> dict[str, Any]:
    """The PD by grade as a function of the informativeness $k$ of withdrawal, its bounds, the anchored $k$ and the
    tipping $k$ of a decision, for one period's cohort.

    ``counts`` is the square count matrix of ``cohort`` (rows the state at the start, columns at the end, the default
    state last by default) and ``withdrawn`` the ratings of each start state withdrawn by the end. For each grade
    below the default state: $D$ the default column, $S$ the rest of its row, $W$ its withdrawals, $N = D + S + W$.

    Returned (arrays have one entry per grade, in order):

    - ``pd_by_k`` ($G \\times$ len(``k_grid``)) and ``k_grid``; ``pd_unfollowed`` ($k = 0$), ``pd_industry`` ($k = 1$,
      the ``cohort`` estimate) and ``pd_manski_upper`` ($k = \\infty$).
    - ``identified``: the Jeffreys sampling band of the whole family, ``lower`` of $D/N$ to ``upper`` of $(D + W)/N$
      at ``level``; this is what the transition page alone supports.
    - ``anchor`` (when ``anchor_defaults`` is given: the defaults $A$ a second source counts for the same grades, over
      ``anchor_obligors`` $N_a$, one number or one per grade, default $N$): ``k`` the anchored odds ratio, ``k_low`` and
      ``k_high`` from the Jeffreys interval of $A/N_a$ at ``level``, ``pd`` the anchored PD $(D + X)/N$, ``lower`` and
      ``upper`` the anchored interval, ``defaults_implied`` $X$ (withdrawn ratings that would have defaulted), and
      ``flag`` per grade: ``ok``, ``anchor_below_observed`` (the anchor counts fewer defaults than the transition page
      already saw: $k$ is clipped to 0) or ``anchor_above_identified`` (more than $D + W$: clipped to $\\infty$).
    - ``tipping`` (when ``thresholds`` is given, a PD or one per grade): ``k`` the tipping $k$ (``tipping_k``),
      ``decision_industry`` (``above`` or ``below`` the threshold at $k = 1$) and ``status`` per grade: ``flips_up``
      (below at $k = 1$, reached at larger $k$), ``flips_down`` (above at $k = 1$, below at smaller $k$) or
      ``robust`` (the threshold is not crossed for any $k \\ge 0$).
    - ``reason`` per grade: ``ok``, ``no_cohort`` ($N = 0$, NaN), ``no_observed_defaults`` ($D = 0$: only Manski's
      bound is valid), ``no_withdrawals`` ($W = 0$: the PD is $D/N$ whatever $k$).
    - ``assumptions`` and ``reference``.

    Raises ``ValueError`` for a malformed matrix, negative counts, ``level`` outside $(0, 1)$, a ``k_grid`` entry that
    is negative or NaN, or an anchor with $A > N_a$ or mismatched length.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    ks = [_k(k) for k in k_grid]
    if not ks:
        raise ValueError("k_grid needs at least one value")
    d, s, w, grades = _grade_counts(counts, withdrawn, default_state)
    n = d + s + w
    g = len(grades)
    pd_by_k = np.column_stack([withdrawal_pd(d, s, w, k) for k in ks])
    reason = []
    for i in range(g):
        if n[i] <= 0:
            reason.append("no_cohort")
        elif d[i] == 0:
            reason.append("no_observed_defaults")
        elif w[i] == 0:
            reason.append("no_withdrawals")
        else:
            reason.append("ok")
    live = n > 0
    safe_n = np.where(live, n, 1.0)

    def jeffreys(count: np.ndarray, side: str) -> np.ndarray:
        r = pd_jeffreys(np.where(live, count, 0.0), safe_n, level=level)
        return np.where(live, r[side], np.nan)

    out: dict[str, Any] = {
        "grades": grades,
        "defaults": d,
        "survivors": s,
        "withdrawn": w,
        "n": n,
        "withdrawn_share": np.divide(w, n, out=np.zeros(g), where=live),
        "k_grid": ks,
        "pd_by_k": pd_by_k,
        "pd_unfollowed": withdrawal_pd(d, s, w, 0.0),
        "pd_industry": withdrawal_pd(d, s, w, 1.0),
        "pd_manski_upper": withdrawal_pd(d, s, w, np.inf),
        "identified": {"lower": jeffreys(d, "lower"), "upper": jeffreys(d + w, "upper")},
        "level": level,
        "anchor": None,
        "tipping": None,
        "reason": reason,
        "assumptions": list(ASSUMPTIONS),
        "reference": REFERENCE,
    }
    if anchor_defaults is not None:
        out["anchor"] = _anchor(d, s, w, n, live, anchor_defaults, anchor_obligors, level, jeffreys)
    if thresholds is not None:
        out["tipping"] = _tipping(d, s, w, n, thresholds)
    return out


def _anchor(d: np.ndarray, s: np.ndarray, w: np.ndarray, n: np.ndarray, live: np.ndarray, anchor_defaults: Any,
            anchor_obligors: Any, level: float, jeffreys: Any) -> dict[str, Any]:
    g = d.shape[0]
    a = np.asarray(anchor_defaults, dtype=float)
    na = n.copy() if anchor_obligors is None else np.broadcast_to(np.asarray(anchor_obligors, dtype=float), (g,)).copy()
    if a.shape != (g,) or na.shape != (g,):
        raise ValueError(f"anchor_defaults and anchor_obligors must give one number for each of the {g} grades")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(na))) or np.any(a < 0):
        raise ValueError("anchor_defaults and anchor_obligors must be finite, defaults non-negative")
    use = live & (na > 0)
    if np.any(live & (na <= 0)) or np.any(a > na):
        raise ValueError("an anchor needs 0 <= anchor_defaults <= anchor_obligors with a positive cohort")
    safe = np.where(use, na, 1.0)
    jr = pd_jeffreys(np.where(use, a, 0.0), safe, level=level)
    rate = np.where(use, a / safe, np.nan)

    def implied(p: np.ndarray) -> np.ndarray:
        return np.where(use, p * n - d, np.nan)

    raw = implied(rate)
    x = np.clip(raw, 0.0, w)
    x_lo = np.clip(implied(np.asarray(jr["lower"])), 0.0, w)
    x_hi = np.clip(implied(np.asarray(jr["upper"])), 0.0, w)
    flag = np.where(raw < -_TOL, "anchor_below_observed", np.where(raw > w + _TOL, "anchor_above_identified", "ok"))
    flag = np.where(use, flag, "no_anchor")

    def kof(xx: np.ndarray) -> np.ndarray:
        ok = use & ~np.isnan(xx)
        res = np.full(d.shape, np.nan)
        res[ok] = odds_multiplier(d[ok], s[ok], w[ok], xx[ok])
        return res

    return {
        "k": kof(x), "k_low": kof(x_lo), "k_high": kof(x_hi),
        "defaults_implied": x,
        "pd": np.where(use, (d + x) / np.where(live, n, 1.0), np.nan),
        "lower": np.where(use, jeffreys(d + x_lo, "lower"), np.nan),
        "upper": np.where(use, jeffreys(d + x_hi, "upper"), np.nan),
        "anchor_rate": rate, "flag": [str(f) for f in flag], "level": level,
    }


def _tipping(d: np.ndarray, s: np.ndarray, w: np.ndarray, n: np.ndarray, thresholds: Any) -> dict[str, Any]:
    g = d.shape[0]
    t = np.broadcast_to(np.asarray(thresholds, dtype=float), (g,)).astype(float)
    k = tipping_k(d, s, w, t)
    pd1 = withdrawal_pd(d, s, w, 1.0)
    lo = withdrawal_pd(d, s, w, 0.0)
    hi = withdrawal_pd(d, s, w, np.inf)
    above = pd1 >= t
    decision, status = [], []
    for i in range(g):
        if not n[i] > 0:
            decision.append("none")
            status.append("robust")
            continue
        decision.append("above" if above[i] else "below")
        if above[i]:
            status.append("flips_down" if lo[i] < t[i] else "robust")
        else:
            status.append("flips_up" if hi[i] >= t[i] else "robust")
    return {"threshold": t, "k": k, "decision_industry": decision, "status": status}
