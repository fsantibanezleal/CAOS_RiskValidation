"""PD curve calibration (Tasche 2013, "The art of probability-of-default curve calibration", Journal of Credit Risk
9(4):63-103, doi:10.21314/jcr.2013.169, arXiv 1212.3716).

A rating model is the joint distribution of a borrower's grade X at the start of a period and its state at the end
(default D or survival N). The unconditional rating profile Pr[X = x] and the PD curve Pr[D | X = x] fix it; so do the
unconditional PD p with the two conditional profiles Pr[X = x | D] and Pr[X = x | N], or p with the likelihood ratio
lambda(x) = Pr[X = x | N] / Pr[X = x | D] (Proposition 3.1, equation 3.8). Calibrating a PD curve to a forecast period
(subscript 1) from an estimation period (subscript 0) means choosing which of these stays invariant, and the choice
depends on what is known at the start of the forecast period (section 4):

- case 1, the forecast rating profile and PD known (4.2): invariant default profile (4.2, implemented as approach ii of
  4.2.1), invariant accuracy ratio (4.3, section 4.2.2), scaled PDs (4.4, 4.7a), scaled likelihood ratio (4.5, 4.8a to
  4.8d);
- case 2, only the forecast PD known (4.3): invariant likelihood ratio (4.10, 4.11), Elkan's change in base rate;
- case 3, only the forecast rating profile known (4.4): invariant PD curve (4.12, 4.13), invariant conditional profiles
  fitted by least squares (4.14b) or least chi-square, then (4.15, 4.16), and invariant likelihood ratio (4.10 with the
  PD of theorem 3.3).

Quasi moment matching (appendix A) smooths a PD curve: the robust logistic curve Pr[D | X = x] = 1 / (1 + exp(alpha +
beta Phi^{-1}(F~_N(x)))) (A.8), F~_N the average of the survival distribution function and its left-continuous version
(A.9), with alpha and beta chosen so that the unconditional PD (3.5a) and the accuracy ratio (A.11b) hit their targets;
the binormal model gives the initial values (A.2c, A.2d, A.6).

The goodness of fit of a calibrated curve is tested with Pearson's chi-square test (Pearson 1900): of the implied
default profile against the observed default numbers, with a Monte Carlo p-value (the registered test
``pd.default_profile``), and of an implied rating profile against the observed grade frequencies (:func:`profile_chi2`).

Grades are passed best first, as the paper's tables list them; internally the paper's numbering is used (grade 1 the
least creditworthy). Every function refuses, with :class:`ImproperModel`, a model that is not a proper joint
distribution of grade and state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import optimize, stats
from scipy.special import ndtri

TASCHE_2013 = (
    "Tasche (2013), The art of probability-of-default curve calibration, Journal of Credit Risk 9(4), "
    "doi:10.21314/jcr.2013.169"
)
_EPS = float(np.finfo(float).eps)


class ImproperModel(ValueError):
    """The approach yields no proper joint distribution of grade and state for these inputs."""


# ---------------------------------------------------------------------------------------------------------------------
# validation and ordering helpers; arrays named *_w are worst first (the paper's numbering)


def _w(a: Sequence[float]) -> np.ndarray:
    return np.asarray(a, dtype=float)[::-1]


def _b(a: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(a)[::-1]]


def _profile(a: Sequence[float], what: str = "a profile") -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or np.any(x < 0) or x.sum() <= 0:
        raise ValueError(f"{what} needs at least two non-negative weights with a positive sum")
    return x / x.sum()


def _curve(a: Sequence[float], *, open_interval: bool = True) -> np.ndarray:
    c = np.asarray(a, dtype=float)
    bad = (c <= 0) | (c >= 1) if open_interval else (c < 0) | (c > 1)
    if c.ndim != 1 or c.size < 2 or not np.all(np.isfinite(c)) or np.any(bad):
        span = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"a PD curve needs one PD in {span} per grade, at least two grades")
    return c


def _rate(p: float, what: str = "the unconditional PD") -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"{what} must lie in (0, 1)")
    return float(p)


def _same_length(*arrays: np.ndarray) -> None:
    if len({a.size for a in arrays}) != 1:
        raise ValueError("every per-grade input needs one value per grade")


def _ar_profiles(dp_w: np.ndarray, sp_w: np.ndarray) -> float:
    """(3.7): sum_{x>=2} P(x | N) P(X <= x - 1 | D) - sum_{x<=k-1} P(x | N) P(X >= x + 1 | D)."""
    below = np.concatenate([[0.0], np.cumsum(dp_w)[:-1]])
    above = np.concatenate([np.cumsum(dp_w[::-1])[::-1][1:], [0.0]])
    return float(np.sum(sp_w * below) - np.sum(sp_w * above))


def _ar_curve(cur_w: np.ndarray, pi_w: np.ndarray) -> float:
    """(A.11b), the accuracy ratio from a PD curve on a finite grade scale."""
    p = float(np.sum(cur_w * pi_w))
    riskier = np.concatenate([[0.0], np.cumsum(cur_w * pi_w)[:-1]])  # sum_{t < x} Pr[D | t] Pr[X = t]
    total = 2.0 * np.sum((1.0 - cur_w) * pi_w * riskier) + np.sum(cur_w * (1.0 - cur_w) * pi_w**2)
    return float(total / (p * (1.0 - p)) - 1.0)


def _f_tilde(sp_w: np.ndarray) -> np.ndarray:
    """(A.9): the average of Pr[X < x | N] and Pr[X <= x | N]."""
    c = np.cumsum(sp_w)
    return (np.concatenate([[0.0], c[:-1]]) + c) / 2.0


# ---------------------------------------------------------------------------------------------------------------------
# section 3: the model and its identities


def model_from_counts(obligors: Sequence[int], defaults: Sequence[int]) -> dict:
    """The empirical model of one period from grade counts (best first): the rating profile, the grade default rates,
    the unconditional default rate, the conditional profiles and the accuracy ratio by (3.7)."""
    n = np.asarray(obligors, dtype=float)
    d = np.asarray(defaults, dtype=float)
    if n.ndim != 1 or n.size < 2 or n.shape != d.shape:
        raise ValueError("obligors and defaults are one count per grade, at least two grades")
    if np.any(n <= 0) or np.any(d < 0) or np.any(d > n) or d.sum() <= 0 or d.sum() >= n.sum():
        raise ValueError("each grade needs obligors > 0 and 0 <= defaults <= obligors, with defaults and survivors")
    dp, sp = d / d.sum(), (n - d) / (n - d).sum()
    return {
        "profile": [float(v) for v in n / n.sum()],
        "default_rate": [float(v) for v in d / n],
        "pd": float(d.sum() / n.sum()),
        "default_profile": [float(v) for v in dp],
        "survival_profile": [float(v) for v in sp],
        "accuracy_ratio": _ar_profiles(dp[::-1], sp[::-1]),
        "obligors": int(n.sum()),
        "defaults": int(d.sum()),
    }


def unconditional_pd(pd_curve: Sequence[float], profile: Sequence[float]) -> float:
    """(3.5a): p = sum_x Pr[D | X = x] Pr[X = x]."""
    c, pi = _curve(pd_curve, open_interval=False), _profile(profile)
    _same_length(c, pi)
    return float(np.sum(c * pi))


def implied_default_profile(pd_curve: Sequence[float], profile: Sequence[float]) -> list[float]:
    """(3.5b): Pr[X = x | D] = Pr[D | X = x] Pr[X = x] / p."""
    c, pi = _curve(pd_curve, open_interval=False), _profile(profile)
    _same_length(c, pi)
    return [float(v) for v in c * pi / np.sum(c * pi)]


def implied_survival_profile(pd_curve: Sequence[float], profile: Sequence[float]) -> list[float]:
    """(3.5c): Pr[X = x | N] = (1 - Pr[D | X = x]) Pr[X = x] / (1 - p)."""
    c, pi = _curve(pd_curve, open_interval=False), _profile(profile)
    _same_length(c, pi)
    return [float(v) for v in (1.0 - c) * pi / np.sum((1.0 - c) * pi)]


def accuracy_ratio(default_profile: Sequence[float], survival_profile: Sequence[float]) -> float:
    """(3.7), the accuracy ratio from the conditional profiles (best first)."""
    dp, sp = _profile(default_profile, "a default profile"), _profile(survival_profile, "a survival profile")
    _same_length(dp, sp)
    return _ar_profiles(dp[::-1], sp[::-1])


def accuracy_ratio_from_curve(pd_curve: Sequence[float], profile: Sequence[float]) -> float:
    """(A.11b), the accuracy ratio from a PD curve and the rating profile (best first)."""
    c, pi = _curve(pd_curve, open_interval=False), _profile(profile)
    _same_length(c, pi)
    p = float(np.sum(c * pi))
    if not 0.0 < p < 1.0:
        raise ValueError("the accuracy ratio needs an unconditional PD in (0, 1)")
    return _ar_curve(c[::-1], pi[::-1])


def likelihood_ratio(pd_curve: Sequence[float], pd: float) -> list[float]:
    """(3.9b): lambda(x) = (1 - Pr[D | X = x]) / Pr[D | X = x] * p / (1 - p)."""
    c, p = _curve(pd_curve), _rate(pd)
    return [float(v) for v in (1.0 - c) / c * p / (1.0 - p)]


def pd_curve_from_likelihood_ratio(likelihood: Sequence[float], pd: float) -> list[float]:
    """(3.9a): Pr[D | X = x] = p / (p + (1 - p) lambda(x))."""
    lam, p = np.asarray(likelihood, dtype=float), _rate(pd)
    if lam.ndim != 1 or np.any(~np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError("a likelihood ratio needs one positive value per grade")
    return [float(v) for v in p / (p + (1.0 - p) * lam)]


def pd_from_likelihood_ratio(profile: Sequence[float], likelihood: Sequence[float]) -> float:
    """Theorem 3.3: the unconditional PD p in [0, 1) solving 1 = sum_x Pr[X = x] / (p + (1 - p) lambda(x)) (3.11a).

    For a non-constant likelihood ratio a solution exists if and only if sum_x Pr[X = x] / lambda(x) >= 1 and
    sum_x Pr[X = x] lambda(x) > 1 (3.11b); it is unique, and it is 0 if and only if the first sum equals 1 (3.11c).
    """
    pi = _profile(profile)
    lam = np.asarray(likelihood, dtype=float)
    _same_length(pi, lam)
    if np.any(~np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError("a likelihood ratio needs one positive value per grade")
    if np.ptp(lam) <= 1e-12 * float(np.max(lam)):
        raise ValueError("a constant likelihood ratio does not determine the unconditional PD (theorem 3.3)")
    inv, fwd = float(np.sum(pi / lam)), float(np.sum(pi * lam))
    if inv < 1.0 - 1e-12 or fwd <= 1.0:
        raise ImproperModel(
            f"theorem 3.3: sum Pr[X = x] / lambda(x) = {inv:.6g} (needs >= 1) and sum Pr[X = x] lambda(x) = {fwd:.6g} "
            "(needs > 1); no unconditional PD makes this profile and likelihood ratio a proper model"
        )
    if inv <= 1.0 + 1e-12:
        return 0.0

    def f(p: float) -> float:
        return float(np.sum(pi / (p + (1.0 - p) * lam))) - 1.0

    # f(0) > 0, f(1) = 0 and f'(1) = sum pi lambda - 1 > 0: the root sits where f turns negative short of 1
    hi = 1.0 - 1e-3
    while f(hi) >= 0.0:
        hi = 1.0 - (1.0 - hi) / 1e3
        if 1.0 - hi < 1e-15:
            raise ImproperModel("theorem 3.3: no unconditional PD below 1 solves (3.11a) at double precision")
    return float(optimize.brentq(f, 0.0, hi, xtol=1e-16, rtol=4 * _EPS))


# ---------------------------------------------------------------------------------------------------------------------
# appendix A: quasi moment matching


def _bracket(f, centre: float, *, width: float = 1.0, limit: int = 60) -> tuple[float, float]:
    """An interval around ``centre`` on which the decreasing ``f`` changes sign, found by doubling its half-width."""
    for _ in range(limit):
        lo, hi = centre - width, centre + width
        if f(lo) >= 0.0 >= f(hi):
            return lo, hi
        width *= 2.0
    raise ImproperModel("no sign change found for a monotone equation; the targets are outside its range")


def qmm(
    profile: Sequence[float],
    pd_target: float,
    ar_target: float,
    survival_profile: Sequence[float] | None = None,
) -> dict:
    """Quasi moment matching (appendix A): the robust logistic PD curve (A.8) on the grades (best first) whose
    unconditional PD (3.5a) is ``pd_target`` and whose accuracy ratio (A.11b) is ``ar_target``.

    ``survival_profile`` (best first) defines F~_N (A.9); in the estimation period it is the empirical one (step 3 of
    A.2: by (3.5c) from the profile, the grade default rates and the unconditional default rate). Without it the
    unconditional profile stands in for it, as section 4.2.2 does in a forecast period where the survival profile
    cannot be observed.
    """
    pi = _profile(profile)[::-1]
    sp = _profile(survival_profile, "a survival profile")[::-1] if survival_profile is not None else pi
    _same_length(pi, sp)
    p = _rate(pd_target, "the target PD")
    if not 0.0 < ar_target < 1.0:
        raise ValueError("the target accuracy ratio must lie in (0, 1)")
    z = ndtri(np.clip(_f_tilde(sp), 1e-15, 1.0 - 1e-15))

    def curve(alpha: float, beta: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(np.clip(alpha + beta * z, -700.0, 700.0)))

    # step 4: initial values from the binormal model (A.2c, A.2d, A.6), mu and tau^2 the mean and variance of
    # Phi^{-1}(F~_N(X)) under the rating profile
    mu = float(np.sum(pi * z))
    tau2 = float(np.sum(pi * (z - mu) ** 2))
    c = math.sqrt(2.0) * float(ndtri((ar_target + 1.0) / 2.0))
    sigma2 = tau2 / (1.0 + p * (1.0 - p) * c**2)
    sigma = math.sqrt(sigma2)
    mu_n, mu_d = mu + p * sigma * c, mu - (1.0 - p) * sigma * c
    start = ((mu_d**2 - mu_n**2) / (2.0 * sigma2) + math.log((1.0 - p) / p), (mu_n - mu_d) / sigma2)

    # step 5: the two equations, solved as nested bracketed roots, which converge whenever a solution exists. For a
    # slope beta the unconditional PD falls strictly as alpha grows (one root alpha(beta)); along alpha(beta) the
    # accuracy ratio rises from 0 at beta = 0 towards the largest one the grade scale and profile allow.
    def alpha_for(beta: float) -> float:
        def f(a: float) -> float:
            return float(np.sum(curve(a, beta) * pi)) - p

        lo, hi = _bracket(f, start[0] + (start[1] - beta) * mu)
        return float(optimize.brentq(f, lo, hi, xtol=1e-14, rtol=4 * _EPS))

    def g(beta: float) -> float:
        return _ar_curve(curve(alpha_for(beta), beta), pi) - ar_target

    hi = max(start[1], 1e-3)
    while g(hi) < 0.0:
        hi *= 2.0
        if hi > 1e4:
            raise ImproperModel(
                f"quasi moment matching: an accuracy ratio of {ar_target:.4g} is not attainable on this grade scale "
                "and profile"
            )
    lo = hi / 2.0
    while lo > 1e-12 and g(lo) > 0.0:
        lo /= 2.0
    beta = float(optimize.brentq(g, lo, hi, xtol=1e-14, rtol=4 * _EPS))
    alpha = alpha_for(beta)
    cur = curve(alpha, beta)
    return {
        "pd_curve": _b(cur),
        "alpha": alpha,
        "beta": beta,
        "scores": _b(z),
        "start": [float(v) for v in start],
        "pd": float(np.sum(cur * pi)),
        "accuracy_ratio": _ar_curve(cur, pi),
        "reference": f"{TASCHE_2013}, appendix A",
    }


# ---------------------------------------------------------------------------------------------------------------------
# section 4.2, case 1: the forecast rating profile and the forecast PD are known


def invariant_default_profile(default_profile_0: Sequence[float], profile_1: Sequence[float], pd_1: float) -> dict:
    """Approach (4.2), Pr1[X = x | D] = Pr0[X = x | D], implemented as approach (ii) of section 4.2.1: the observed
    estimation default profile and the forecast profile give a raw survival profile by (3.5c) with (4.6b), the two
    give a forecast accuracy ratio by (3.7), and QMM fits the forecast curve to it and to ``pd_1``. Proper if and only
    if p1 Pr0[X = x | D] <= Pr1[X = x] and p1 (1 - Pr0[X = x | D]) <= 1 - Pr1[X = x] for every grade (4.6a)."""
    dp0, pi1 = _profile(default_profile_0, "a default profile"), _profile(profile_1)
    _same_length(dp0, pi1)
    p1 = _rate(pd_1, "the forecast PD")
    tol = 1e-12
    bad = np.flatnonzero((p1 * dp0 > pi1 + tol) | (p1 * (1.0 - dp0) > 1.0 - pi1 + tol))
    if bad.size:
        raise ImproperModel(
            f"invariant default profile: condition (4.6a) fails at grade(s) {', '.join(str(int(i) + 1) for i in bad)} "
            "(counted best first); no proper model keeps this default profile at this forecast PD"
        )
    raw = np.clip((pi1 - p1 * dp0) / (1.0 - p1), 0.0, None)
    ar1 = _ar_profiles(dp0[::-1], raw[::-1] / raw.sum())
    if not 0.0 < ar1 < 1.0:
        raise ImproperModel(
            f"invariant default profile: the implied forecast accuracy ratio {ar1:.4g} is not in (0, 1)"
        )
    out = qmm(pi1, p1, ar1, survival_profile=raw)
    out.update(
        {
            "accuracy_ratio_target": ar1,
            "raw_survival_profile": [float(v) for v in raw / raw.sum()],
            "reference": f"{TASCHE_2013}, equations 4.2, 4.6a, 4.6b and section 4.2.1 (ii)",
        }
    )
    return out


def invariant_accuracy_ratio(profile_1: Sequence[float], pd_1: float, ar_0: float) -> dict:
    """Approach (4.3), AR1 = AR0: QMM to the forecast PD and the estimation accuracy ratio, the survival profile
    approximated by the forecast unconditional profile (section 4.2.2)."""
    out = qmm(profile_1, pd_1, ar_0)
    out["reference"] = f"{TASCHE_2013}, equation 4.3 and section 4.2.2"
    return out


def scaled_pds(pd_curve_0: Sequence[float], profile_1: Sequence[float], pd_1: float) -> dict:
    """Approach (4.4), Pr1[D | X = x] = c_PD Pr0[D | X = x] with c_PD = p1 / sum_x Pr0[D | X = x] Pr1[X = x] (4.7a);
    improper when a scaled PD exceeds one (section 4.2.3). The implied default profile is (4.7b)."""
    c0, pi1 = _curve(pd_curve_0), _profile(profile_1)
    _same_length(c0, pi1)
    p1 = _rate(pd_1, "the forecast PD")
    c_pd = p1 / float(np.sum(c0 * pi1))
    cur = c_pd * c0
    if np.any(cur >= 1.0):
        raise ImproperModel(
            f"scaled PDs: c_PD = {c_pd:.4g} lifts a grade's PD to {float(cur.max()):.4g}, which is not a probability"
        )
    return {
        "pd_curve": [float(v) for v in cur],
        "c_pd": c_pd,
        "default_profile": [float(v) for v in cur * pi1 / p1],
        "reference": f"{TASCHE_2013}, equations 4.4, 4.7a and 4.7b",
    }


def scaled_likelihood_ratio(pd_curve_0: Sequence[float], pd_0: float, profile_1: Sequence[float], pd_1: float) -> dict:
    """Approach (4.5), lambda1 = c_LR lambda0, with c_LR the root of 1 = sum_x Pr1[X = x] / (p1 + (1 - p1) c_LR
    lambda0(x)) (4.8a). The right-hand side falls strictly from 1 / p1 to 0 as c_LR grows, so the root always exists
    and is unique, and the model is always proper (section 4.2.4); it lies inside the bracket (4.8b). The curve is
    (4.8d) and the default profile (4.8c)."""
    lam0 = np.asarray(likelihood_ratio(pd_curve_0, pd_0))
    pi1 = _profile(profile_1)
    _same_length(lam0, pi1)
    p1 = _rate(pd_1, "the forecast PD")
    lo, hi = 1.0 / float(np.sum(pi1 * lam0)), float(np.sum(pi1 / lam0))

    def f(c: float) -> float:
        return float(np.sum(pi1 / (p1 + (1.0 - p1) * c * lam0))) - 1.0

    c_lr = float(optimize.brentq(f, lo / 2.0, 2.0 * hi, xtol=1e-300, rtol=4 * _EPS))
    denominator = p1 + (1.0 - p1) * c_lr * lam0
    return {
        "pd_curve": [float(v) for v in p1 / denominator],
        "c_lr": c_lr,
        "bracket": [lo, hi],
        "default_profile": [float(v) for v in pi1 / denominator],
        "reference": f"{TASCHE_2013}, equations 4.5 and 4.8a to 4.8d",
    }


# ---------------------------------------------------------------------------------------------------------------------
# sections 4.3 and 4.4, cases 2 and 3: only the forecast PD, or only the forecast rating profile, is known


def invariant_likelihood_ratio(
    pd_curve_0: Sequence[float],
    pd_0: float,
    *,
    pd_1: float | None = None,
    profile_1: Sequence[float] | None = None,
) -> dict:
    """Assumption (4.10), lambda1 = lambda0. Give exactly one of ``pd_1`` (case 2, section 4.3: the curve is (4.11),
    Elkan's change in base rate) or ``profile_1`` (case 3, section 4.4.3: the forecast PD is the unique solution of
    (3.11a) by theorem 3.3, which exists only under (3.11b), and then the curve is (4.11))."""
    if (pd_1 is None) == (profile_1 is None):
        raise ValueError("give exactly one of pd_1 (case 2) or profile_1 (case 3)")
    lam0 = likelihood_ratio(pd_curve_0, pd_0)
    if pd_1 is not None:
        p1, ref = _rate(pd_1, "the forecast PD"), "equations 4.10 and 4.11 (case 2)"
    else:
        p1, ref = pd_from_likelihood_ratio(profile_1, lam0), "equation 4.10, theorem 3.3 and section 4.4.3 (case 3)"
        if p1 <= 0.0:
            raise ImproperModel("invariant likelihood ratio: theorem 3.3 gives a forecast PD of zero (3.11c)")
    return {"pd_curve": pd_curve_from_likelihood_ratio(lam0, p1), "pd": p1, "reference": f"{TASCHE_2013}, {ref}"}


def invariant_pd_curve(pd_curve_0: Sequence[float], profile_1: Sequence[float]) -> dict:
    """Assumption (4.12), Pr1[D | X = x] = Pr0[D | X = x]; the forecast PD follows from the forecast profile by (4.13).
    Section 4.4.1 warns that this carries the estimation period's unconditional PD into the forecast."""
    c0 = _curve(pd_curve_0)
    return {
        "pd_curve": [float(v) for v in c0],
        "pd": unconditional_pd(c0, profile_1),
        "reference": f"{TASCHE_2013}, equations 4.12 and 4.13",
    }


def invariant_conditional_profiles(
    default_profile_0: Sequence[float],
    survival_profile_0: Sequence[float],
    profile_1: Sequence[float],
    *,
    fit: str = "least_squares",
) -> dict:
    """Assumption (4.9), both conditional profiles invariant (case 3, section 4.4.2). The constraint (3.4a) is
    overdetermined, so the forecast PD is fitted: by least squares, p1* by (4.14b), or by minimising the Pearson
    chi-square distance of the forecast profile from p1 Pr0[X = x | D] + (1 - p1) Pr0[X = x | N] (``fit="least_chi2"``).
    The implied rating profile is (4.15) and the curve (4.16); test the profile with :func:`profile_chi2` before using
    the curve."""
    dp0 = _profile(default_profile_0, "a default profile")
    sp0 = _profile(survival_profile_0, "a survival profile")
    pi1 = _profile(profile_1)
    _same_length(dp0, sp0, pi1)
    diff = dp0 - sp0
    if not np.any(np.abs(diff) > 0):
        raise ValueError("identical conditional profiles do not determine a forecast PD")
    if fit == "least_squares":
        p1 = float(np.clip(np.sum((pi1 - sp0) * diff) / np.sum(diff**2), 0.0, 1.0))
        ref = "equations 4.9, 4.14b, 4.15 and 4.16"
    elif fit == "least_chi2":

        def distance(p: float) -> float:
            e = p * dp0 + (1.0 - p) * sp0
            return float(np.sum((pi1 - e) ** 2 / e))

        res = optimize.minimize_scalar(distance, bounds=(1e-12, 1.0 - 1e-12), method="bounded",
                                       options={"xatol": 1e-14, "maxiter": 2000})
        p1 = float(res.x)
        ref = "equations 4.9, 4.15 and 4.16, the least chi-square variant of 4.14b (section 4.4.2)"
    else:
        raise ValueError('fit is "least_squares" or "least_chi2"')
    implied = p1 * dp0 + (1.0 - p1) * sp0
    if not 0.0 < p1 < 1.0 or np.any(implied <= 0):
        raise ImproperModel(f"invariant conditional profiles: the fitted forecast PD {p1:.4g} gives no proper model")
    return {
        "pd_curve": [float(v) for v in p1 * dp0 / implied],
        "pd": p1,
        "profile": [float(v) for v in implied],
        "fit": fit,
        "reference": f"{TASCHE_2013}, {ref}",
    }


def profile_chi2(counts: Sequence[int], profile: Sequence[float]) -> dict:
    """Pearson's chi-square test (Pearson 1900) of grade counts against a rating profile, with k - 1 degrees of
    freedom: the check section 4.4.2 asks for before a profile implied by (4.15) is used; it gives the p-values of
    Table 8."""
    n = np.asarray(counts, dtype=float)
    pi = _profile(profile)
    _same_length(n, pi)
    if np.any(n < 0) or n.sum() <= 0:
        raise ValueError("counts need non-negative values with a positive sum")
    if np.any(pi <= 0):
        raise ValueError("the profile tested against needs a positive weight in every grade")
    expected = n.sum() * pi
    stat = float(np.sum((n - expected) ** 2 / expected))
    dof = int(n.size - 1)
    return {
        "statistic": stat,
        "dof": dof,
        "p_value": float(stats.chi2.sf(stat, dof)),
        "reference": "Pearson (1900), Philosophical Magazine 50(302):157-175, doi:10.1080/14786440009463897; "
        f"{TASCHE_2013}, section 2.1",
    }


__all__ = [
    "TASCHE_2013",
    "ImproperModel",
    "accuracy_ratio",
    "accuracy_ratio_from_curve",
    "implied_default_profile",
    "implied_survival_profile",
    "invariant_accuracy_ratio",
    "invariant_conditional_profiles",
    "invariant_default_profile",
    "invariant_likelihood_ratio",
    "invariant_pd_curve",
    "likelihood_ratio",
    "model_from_counts",
    "pd_curve_from_likelihood_ratio",
    "pd_from_likelihood_ratio",
    "profile_chi2",
    "qmm",
    "scaled_likelihood_ratio",
    "scaled_pds",
    "unconditional_pd",
]
