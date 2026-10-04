"""PD calibration tests: is the default rate the model predicted the one that occurred?

BCBS Working Paper 14 (2005): "at present no really powerful tests of adequate calibration are currently
available. Due to the correlation effects that have to be respected there even seems to be no way to develop
such tests." The tests below therefore come in families with different assumptions (one grade or all grades,
one period or several, independent or correlated defaults); read them together, never one alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

from riskvalidation.core.build import make_result
from riskvalidation.core.policy import Policy
from riskvalidation.core.registry import register
from riskvalidation.core.result import TestResult
from riskvalidation.validation._common import as_binary, as_float_array, check_counts, check_pd, labels

ECB_2019 = (
    "ECB (2019), Instructions for reporting the validation results of internal models: IRB Pillar I models for "
    "credit risk"
)
WP14 = "BCBS (2005), Studies on the validation of internal rating systems, Working Paper 14 (revised)"

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(256)
_GH_WEIGHTS = _GH_WEIGHTS / math.sqrt(2.0 * math.pi)


def _conditional_pd(pd: float, rho: float, x: np.ndarray) -> np.ndarray:
    """PD conditional on the systematic factor x in the one-factor model (Vasicek)."""
    return stats.norm.cdf((stats.norm.ppf(pd) - math.sqrt(rho) * x) / math.sqrt(1.0 - rho))


def default_count_sf(k: int, n: int, pd: float, rho: float = 0.0) -> float:
    """P(D >= k) for n obligors with PD ``pd`` and asset correlation ``rho`` (binomial when rho = 0)."""
    if k <= 0:
        return 1.0
    if rho == 0.0:
        return float(stats.binom.sf(k - 1, n, pd))
    if not 0.0 < rho < 1.0:
        raise ValueError("asset correlation must lie in [0, 1)")
    cond = _conditional_pd(pd, rho, _GH_NODES)
    return float(np.clip(np.sum(_GH_WEIGHTS * stats.binom.sf(k - 1, n, cond)), 0.0, 1.0))


def critical_default_count(n: int, pd: float, q: float = 0.99, rho: float = 0.0) -> int:
    """Smallest k with P(D >= k) <= 1 - q; the test rejects when D >= k.

    This definition reproduces the critical values tabulated in BCBS WP14 for PD 1%, N 1,000, q 99%
    (19, 35, 49, 63, 77 for rho = 0%, 5%, 10%, 15%, 20%).
    """
    lo, hi = 0, n + 1
    while lo < hi:  # P(D >= k) decreases in k
        mid = (lo + hi) // 2
        if default_count_sf(mid, n, pd, rho) <= 1.0 - q:
            hi = mid
        else:
            lo = mid + 1
    return lo


@register(
    "pd.jeffreys",
    family="pd-calibration",
    name="Jeffreys test",
    h0="the PD applied at the start of the period is greater than or equal to the true PD",
    reference=f"{ECB_2019}, section 2.5.3.1; Brown, Cai and DasGupta (2001), doi:10.1214/ss/1009213286",
    decision="p_value",
)
def pd_jeffreys(
    n: int,
    d: int,
    pd: float,
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """One-sided Jeffreys test: p = F_Beta(D+1/2, N-D+1/2)(PD), small when the PD is underestimated."""
    n_a, d_a = check_counts(n, d)
    pd_v = float(check_pd(pd)[0])
    nn, dd = int(n_a[0]), int(d_a[0])
    if nn == 0:
        raise ValueError("the Jeffreys test needs at least one obligor")
    a, b = dd + 0.5, nn - dd + 0.5
    p = float(stats.beta.cdf(pd_v, a, b))
    return make_result(
        "pd.jeffreys",
        statistic=dd / nn,
        p_value=p,
        policy=policy,
        alternative="the true PD exceeds the PD applied (underestimation)",
        n=nn,
        n_events=dd,
        model_id=model_id,
        segment=segment,
        inputs=(nn, dd, pd_v),
        extras={
            "pd_applied": pd_v,
            "default_rate": dd / nn,
            "posterior_q025": float(stats.beta.ppf(0.025, a, b)),
            "posterior_q975": float(stats.beta.ppf(0.975, a, b)),
        },
    )


def pd_jeffreys_grades(
    n: Sequence[int],
    d: Sequence[int],
    pd: Sequence[float],
    *,
    grades: Sequence[str] | None = None,
    policy: Policy | None = None,
    model_id: str | None = None,
) -> list[TestResult]:
    """K + 1 Jeffreys tests: one per grade and one for the portfolio (ECB 2019, section 2.5.3.1).

    The portfolio PD is the obligor-weighted average of the grade PDs, the average PD across customers.
    """
    n_a, d_a = check_counts(n, d)
    pd_a = check_pd(pd)
    if pd_a.shape != n_a.shape:
        raise ValueError("pd must have one value per grade")
    names = labels(len(n_a), grades)
    out = [
        pd_jeffreys(int(ni), int(di), float(pi), policy=policy, model_id=model_id, segment=g)
        for ni, di, pi, g in zip(n_a, d_a, pd_a, names, strict=True)
        if ni > 0
    ]
    total = int(n_a.sum())
    pd_port = float(np.sum(n_a * pd_a) / total)
    out.append(pd_jeffreys(total, int(d_a.sum()), pd_port, policy=policy, model_id=model_id, segment="portfolio"))
    return out


@register(
    "pd.binomial",
    family="pd-calibration",
    name="Binomial test",
    h0="the true PD is less than or equal to the PD applied, defaults independent",
    reference=f"{WP14}, section on calibration (binomial test; an early warning when defaults are correlated)",
    decision="p_value",
)
def pd_binomial(
    n: int,
    d: int,
    pd: float,
    *,
    q: float = 0.99,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """Exact one-sided binomial test, p = P(Bin(N, PD) >= D)."""
    n_a, d_a = check_counts(n, d)
    pd_v = float(check_pd(pd)[0])
    nn, dd = int(n_a[0]), int(d_a[0])
    p = default_count_sf(dd, nn, pd_v, 0.0)
    k = critical_default_count(nn, pd_v, q, 0.0)
    return make_result(
        "pd.binomial",
        statistic=float(dd),
        p_value=p,
        policy=policy,
        alternative="the true PD exceeds the PD applied",
        n=nn,
        n_events=dd,
        model_id=model_id,
        segment=segment,
        inputs=(nn, dd, pd_v, q),
        notes="assumes independent defaults; WP14: the true type I error can be much larger if defaults are "
        "correlated",
        extras={"critical_count": k, "confidence": q, "expected_defaults": nn * pd_v},
    )


@register(
    "pd.binomial_vasicek",
    family="pd-calibration",
    name="Vasicek-corrected binomial test",
    h0="the true PD is less than or equal to the PD applied, defaults correlated through one systematic factor",
    reference=f"{WP14}, appendix (one-factor model with asset correlation)",
    decision="p_value",
)
def pd_binomial_vasicek(
    n: int,
    d: int,
    pd: float,
    rho: float,
    *,
    q: float = 0.99,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """Binomial test under the Vasicek one-factor mixture; the asset correlation is an assumption to document."""
    n_a, d_a = check_counts(n, d)
    pd_v = float(check_pd(pd)[0])
    if not 0.0 <= rho < 1.0:
        raise ValueError("asset correlation must lie in [0, 1)")
    nn, dd = int(n_a[0]), int(d_a[0])
    p = default_count_sf(dd, nn, pd_v, rho)
    k = critical_default_count(nn, pd_v, q, rho)
    approx = nn * float(
        stats.norm.cdf((stats.norm.ppf(pd_v) + math.sqrt(rho) * stats.norm.ppf(q)) / math.sqrt(1.0 - rho))
    )
    return make_result(
        "pd.binomial_vasicek",
        statistic=float(dd),
        p_value=p,
        policy=policy,
        alternative="the true PD exceeds the PD applied",
        n=nn,
        n_events=dd,
        model_id=model_id,
        segment=segment,
        inputs=(nn, dd, pd_v, rho, q),
        notes=f"asset correlation {rho:g} is a modelling assumption",
        extras={"critical_count": k, "confidence": q, "rho": rho, "large_n_critical_count": approx},
    )


@register(
    "pd.chi2_grades",
    family="pd-calibration",
    name="Chi-square test over grades",
    h0="every grade PD equals the true PD of its grade, defaults independent within and between grades",
    reference=f"{WP14}, calibration (Hosmer-Lemeshow type statistic over rating grades)",
    decision="p_value",
)
def pd_chi2_grades(
    n: Sequence[int],
    d: Sequence[int],
    pd: Sequence[float],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """T = sum (n_i p_i - d_i)^2 / (n_i p_i (1 - p_i)), chi-square with as many degrees of freedom as grades."""
    n_a, d_a = check_counts(n, d)
    pd_a = check_pd(pd)
    keep = n_a > 0
    n_a, d_a, pd_a = n_a[keep], d_a[keep], pd_a[keep]
    t = float(np.sum((n_a * pd_a - d_a) ** 2 / (n_a * pd_a * (1.0 - pd_a))))
    dof = int(n_a.size)
    return make_result(
        "pd.chi2_grades",
        statistic=t,
        p_value=float(stats.chi2.sf(t, dof)),
        policy=policy,
        alternative="at least one grade PD differs from the true PD",
        n=int(n_a.sum()),
        n_events=int(d_a.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(n_a, d_a, pd_a),
        notes="WP14: likely to understate the true type I error under default dependence; slow convergence for "
        "small PDs",
        extras={"dof": dof},
    )


@register(
    "pd.hosmer_lemeshow",
    family="pd-calibration",
    name="Hosmer-Lemeshow test",
    h0="the predicted probabilities equal the true default probabilities in every risk group",
    reference="Hosmer and Lemeshow (1980), Communications in Statistics, Theory and Methods 9(10), "
    "doi:10.1080/03610928008827941",
    decision="p_value",
)
def pd_hosmer_lemeshow(
    defaults: Sequence[int],
    pd: Sequence[float],
    *,
    groups: int = 10,
    dof: int | None = None,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """Groups by quantiles of predicted PD; chi-square with ``groups - 2`` degrees of freedom by default.

    The G - 2 degrees of freedom are those of the original in-sample statistic; on an independent hold-out sample
    some authors use G. Pass ``dof`` to choose, and the choice is recorded.
    """
    y = as_binary(defaults)
    p = check_pd(pd)
    if p.shape != y.shape:
        raise ValueError("defaults and pd must have the same length")
    if groups < 3 or groups > y.size:
        raise ValueError("groups must be at least 3 and at most the sample size")
    order = np.argsort(p, kind="mergesort")
    hl = 0.0
    for idx in np.array_split(order, groups):
        o, e, m = float(y[idx].sum()), float(p[idx].sum()), idx.size
        hl += (o - e) ** 2 / (e * (1.0 - e / m))
    k = groups - 2 if dof is None else int(dof)
    return make_result(
        "pd.hosmer_lemeshow",
        statistic=hl,
        p_value=float(stats.chi2.sf(hl, k)),
        policy=policy,
        alternative="the predicted probabilities are miscalibrated in at least one group",
        n=int(y.size),
        n_events=int(y.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(y, p, groups, k),
        extras={"groups": groups, "dof": k},
    )


@register(
    "pd.spiegelhalter",
    family="pd-calibration",
    name="Spiegelhalter test",
    h0="every obligor's predicted PD equals its true default probability",
    reference="Spiegelhalter (1986), Statistics in Medicine 5(5), doi:10.1002/sim.4780050506; formula as in the "
    "Stata brier manual",
    decision="p_value",
)
def pd_spiegelhalter(
    defaults: Sequence[int],
    pd: Sequence[float],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """Z = (observed Brier - expected Brier) / sd under H0; one-sided upper tail. Obligor level, no binning."""
    y = as_binary(defaults)
    f = check_pd(pd)
    if f.shape != y.shape:
        raise ValueError("defaults and pd must have the same length")
    nn = y.size
    observed = float(np.mean((f - y) ** 2))
    expected = float(np.mean(f * (1.0 - f)))
    var = float(np.sum(f * (1.0 - f) * (1.0 - 2.0 * f) ** 2) / nn**2)
    z = (observed - expected) / math.sqrt(var)
    return make_result(
        "pd.spiegelhalter",
        statistic=z,
        p_value=float(stats.norm.sf(z)),
        policy=policy,
        alternative="the forecasts are miscalibrated",
        n=int(nn),
        n_events=int(y.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(y, f),
        extras={"brier_observed": observed, "brier_expected": expected, "brier_variance": var},
    )


@register(
    "pd.normal_multiperiod",
    family="pd-calibration",
    name="Normal test over several periods",
    h0="none of the true PDs in years 1 to T is greater than its forecast",
    reference=f"{WP14}, calibration (normal test; cross-sectional dependence admissible, independence over time)",
    decision="p_value",
)
def pd_normal_multiperiod(
    default_rates: Sequence[float],
    pd: Sequence[float],
    *,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """sum(d_t - PD_t) / (sqrt(T) sigma_hat), compared with the standard normal upper tail."""
    dr = as_float_array(default_rates, "default_rates")
    pdt = check_pd(pd)
    if dr.shape != pdt.shape:
        raise ValueError("default rates and PDs must have one value per year")
    t = dr.size
    if t < 2:
        raise ValueError("the multi-period normal test needs at least two years")
    e = dr - pdt
    var = (np.sum(e**2) - np.sum(e) ** 2 / t) / (t - 1)
    if var <= 0:
        return make_result(
            "pd.normal_multiperiod",
            statistic=float("nan"),
            policy=policy,
            n=t,
            model_id=model_id,
            segment=segment,
            inputs=(dr, pdt),
            notes="the estimated variance of the yearly differences is zero; the statistic is undefined",
        )
    stat = float(np.sum(e) / (math.sqrt(t) * math.sqrt(var)))
    return make_result(
        "pd.normal_multiperiod",
        statistic=stat,
        p_value=float(stats.norm.sf(stat)),
        policy=policy,
        alternative="at least one yearly PD is underestimated",
        n=t,
        model_id=model_id,
        segment=segment,
        inputs=(dr, pdt),
        extras={"years": t, "sigma_hat": math.sqrt(var)},
    )


WP14_TRAFFIC_PROBS = (0.5, 0.3, 0.15, 0.05)


@register(
    "pd.traffic_lights",
    family="pd-calibration",
    name="Traffic-lights test (Blochwitz, Hohl and Wehn)",
    h0="the yearly PDs are correct; the colour counts are multinomial with the stated probabilities",
    reference=f"{WP14}, traffic lights approach (Blochwitz, Hohl and Wehn); probabilities 0.5, 0.3, 0.15, 0.05",
    decision="p_value",
)
def pd_traffic_lights(
    n: Sequence[int],
    d: Sequence[int],
    pd: Sequence[float],
    *,
    probs: Sequence[float] = WP14_TRAFFIC_PROBS,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """Colour each year by its standardised default rate; p = P(colour profile no better than observed).

    Profiles are ordered as V = 1000 A_g + 100 A_y + 10 A_o + A_r (WP14), which for any T is the lexicographic
    order of (A_g, A_y, A_o, A_r); P(V <= v_obs) is computed by exact multinomial enumeration.
    """
    n_a, d_a = check_counts(n, d)
    pd_a = check_pd(pd)
    if pd_a.shape != n_a.shape:
        raise ValueError("one PD per year is required")
    q = np.asarray(probs, dtype=float)
    if q.shape != (4,) or np.any(q <= 0) or not math.isclose(float(q.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("probs must be four positive numbers summing to 1")
    r = (d_a - n_a * pd_a) / np.sqrt(n_a * pd_a * (1.0 - pd_a))
    cuts = stats.norm.ppf(np.cumsum(q)[:3])
    colour = np.searchsorted(cuts, r, side="left")  # 0 green, 1 yellow, 2 orange, 3 red
    counts = tuple(int(np.sum(colour == c)) for c in range(4))
    t = int(n_a.size)
    p_le = 0.0
    for a_g in range(t + 1):
        for a_y in range(t - a_g + 1):
            for a_o in range(t - a_g - a_y + 1):
                prof = (a_g, a_y, a_o, t - a_g - a_y - a_o)
                if prof <= counts:
                    p_le += float(stats.multinomial.pmf(prof, t, q))
    v = 1000 * counts[0] + 100 * counts[1] + 10 * counts[2] + counts[3]
    return make_result(
        "pd.traffic_lights",
        statistic=float(v),
        p_value=min(1.0, p_le),
        policy=policy,
        alternative="the PDs are underestimated in too many years",
        n=t,
        n_events=int(d_a.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(n_a, d_a, pd_a, q),
        extras={
            "green": counts[0],
            "yellow": counts[1],
            "orange": counts[2],
            "red": counts[3],
            "standardised_rates": r.tolist(),
            "colours": [("green", "yellow", "orange", "red")[c] for c in colour],
        },
    )


@register(
    "pd.brier",
    family="pd-calibration",
    name="Brier score with Murphy decomposition",
    h0="descriptive: mean squared error of the PD forecasts",
    reference="Brier (1950), doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2; Murphy (1973), "
    "doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2",
    decision="metric",
)
def pd_brier(
    defaults: Sequence[int],
    pd: Sequence[float],
    *,
    groups: Sequence[str] | None = None,
    bins: int = 10,
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """BS = reliability - resolution + uncertainty; exact when forecasts are constant within each group.

    Groups are rating grades when given, otherwise ``bins`` quantile bins of the forecast. The Brier skill
    against the trivial forecast (the sample default rate) is reported, since low default rates make any Brier
    score look small (WP14).
    """
    y = as_binary(defaults)
    f = check_pd(pd)
    if f.shape != y.shape:
        raise ValueError("defaults and pd must have the same length")
    nn = y.size
    if groups is not None:
        g = np.asarray(groups)
        if g.shape != y.shape:
            raise ValueError("groups must have one label per obligor")
        _, gid = np.unique(g, return_inverse=True)
    else:
        order = np.argsort(f, kind="mergesort")
        gid = np.empty(nn, dtype=int)
        for k, idx in enumerate(np.array_split(order, min(bins, nn))):
            gid[idx] = k
    bs = float(np.mean((f - y) ** 2))
    dbar = float(y.mean())
    rel = res = 0.0
    exact = True
    for k in np.unique(gid):
        m = gid == k
        nk = int(m.sum())
        fk, dk = float(f[m].mean()), float(y[m].mean())
        rel += nk * (fk - dk) ** 2
        res += nk * (dk - dbar) ** 2
        exact = exact and bool(np.ptp(f[m]) == 0.0)
    rel, res = rel / nn, res / nn
    unc = dbar * (1.0 - dbar)
    return make_result(
        "pd.brier",
        metric=bs,
        policy=policy,
        n=int(nn),
        n_events=int(y.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(y, f, gid),
        extras={
            "reliability": rel,
            "resolution": res,
            "uncertainty": unc,
            "brier_skill": (1.0 - bs / unc) if unc > 0 else None,
            "decomposition_exact": exact,
        },
    )


@register(
    "pd.ece",
    family="pd-calibration",
    name="Expected calibration error",
    h0="descriptive: weighted mean absolute gap between forecast and observed default rate per bin",
    reference="Naeini, Cooper and Hauskrecht (2015), AAAI, doi:10.1609/aaai.v29i1.9602; Guo et al. (2017), "
    "arXiv:1706.04599",
    decision="metric",
)
def pd_ece(
    defaults: Sequence[int],
    pd: Sequence[float],
    *,
    bins: int = 10,
    binning: str = "quantile",
    policy: Policy | None = None,
    model_id: str | None = None,
    segment: str | None = None,
) -> TestResult:
    """ECE = sum_m |B_m|/n |mean(y) - mean(p)|; quantile bins by default, because PDs cluster near zero."""
    y = as_binary(defaults)
    f = check_pd(pd)
    if f.shape != y.shape:
        raise ValueError("defaults and pd must have the same length")
    if binning == "quantile":
        order = np.argsort(f, kind="mergesort")
        parts = np.array_split(order, min(bins, y.size))
    elif binning == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
        idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, bins - 1)
        parts = [np.flatnonzero(idx == b) for b in range(bins)]
    else:
        raise ValueError("binning must be 'quantile' or 'uniform'")
    ece, table = 0.0, []
    for part in parts:
        if part.size == 0:
            continue
        mp, my = float(f[part].mean()), float(y[part].mean())
        ece += part.size / y.size * abs(my - mp)
        table.append({"n": int(part.size), "mean_pd": mp, "default_rate": my})
    return make_result(
        "pd.ece",
        metric=ece,
        policy=policy,
        n=int(y.size),
        n_events=int(y.sum()),
        model_id=model_id,
        segment=segment,
        inputs=(y, f, bins, binning),
        extras={"bins": table, "binning": binning},
    )


__all__ = [
    "critical_default_count",
    "default_count_sf",
    "pd_binomial",
    "pd_binomial_vasicek",
    "pd_brier",
    "pd_chi2_grades",
    "pd_ece",
    "pd_hosmer_lemeshow",
    "pd_jeffreys",
    "pd_jeffreys_grades",
    "pd_normal_multiperiod",
    "pd_spiegelhalter",
    "pd_traffic_lights",
]

