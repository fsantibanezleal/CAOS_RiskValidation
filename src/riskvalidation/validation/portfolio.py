"""The economic capital validation tests (dossier 05 G.5): each returns one `TestResult`.

Acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds. Each function states its
light rule in its docstring and records its parameters in `policy_version`, in `alpha_amber` and `alpha_red` where a
p-value is used, and in `extras["thresholds"]` for the other parameters, so that a view can explain a light. A test
that cannot be evaluated returns the light `not_evaluated` with its reason in `notes` and a null statistic, p-value
and metric once serialised, never a fabricated number. Dossier 05 G.5 asks to check the allocation sums to the total
(Euler full allocation) and to benchmark the model; the tests here are the ones that can be stated as hypotheses on
the outputs of `riskvalidation.portfolio`.

The tests are plain functions that do not read the registry or the default policy: their rules are the contract's
(`engine-portfolio-api`, module validation.portfolio). They state no compliance claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.engines.vasicek import lhp_quantile
from riskvalidation.portfolio.portfolio import check_level, float_vector
from riskvalidation.validation._policy import battery_label

POLICY_LABEL = "contraste-portfolio-battery 1.0.0"
NAN = float("nan")
EPS = 1e-12             # tolerance on a gate comparison, so that a value exactly on a gate is not moved by rounding
MIN_MC = 10             # the fewest Monte Carlo scenarios the Kolmogorov comparison accepts
WILSON_Z = float(stats.norm.ppf(0.975))

G5 = ("dossier 05 G.5 (validation tests of an economic capital model: check the allocation sums to the total, "
      "the sensitivity checks)")
BINOMIAL = ("exceedance counts against the Binomial(n, 1 - q) of the fitted Vasicek quantile (dossier 04 C.4; "
            "Vasicek 2002, Risk 12/2002)")
TASCHE_2008 = ("Tasche (2008), Capital allocation to business units and sub-portfolios: the Euler principle, "
               "arXiv 0708.2542")
HRS_2004 = ("Haaf, Reiss and Schoenmakers (2004), Numerically stable computation of CreditRisk+, "
            "doi:10.21314/jor.2004.097")
KOLMOGOROV = "Kolmogorov distance and its asymptotic law (Kolmogorov 1933; Smirnov 1948)"
WILSON_1927 = ("Wilson (1927), Probable inference, the law of succession, and statistical inference, "
               "JASA 22(158):209-212")


def _not_evaluated(test_id: str, h0: str, alternative: str, reference: str, version: str, reason: str, *,
                   model_id: str | None = None, segment: str | None = None, n: int | None = None,
                   alpha_amber: float | None = None, alpha_red: float | None = None,
                   extras: Mapping[str, Any] | None = None) -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(
        test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=version,
        light=Light.NOT_EVALUATED, statistic=NAN, notes=f"not evaluated: {reason}", model_id=model_id,
        segment=segment, n=n, alpha_amber=alpha_amber, alpha_red=alpha_red, extras=dict(extras or {}),
    )


def _check_alphas(alpha_amber: float, alpha_red: float) -> None:
    if not (0.0 < alpha_red <= alpha_amber < 1.0):
        raise ValueError(f"alpha_red and alpha_amber must satisfy 0 < alpha_red <= alpha_amber < 1, got "
                         f"{alpha_red} and {alpha_amber}")


def _p_light(p: float, alpha_amber: float, alpha_red: float) -> Light:
    if p >= alpha_amber:
        return Light.GREEN
    return Light.AMBER if p >= alpha_red else Light.RED


@register(
    "portfolio.tail_backtest",
    family="portfolio",
    name="Tail backtest",
    h0="the number of evaluation years above the fitted q-quantile is Binomial(n, 1 - q)",
    reference=f"{BINOMIAL}; {G5}",
    decision="p_value",
)
def tail_backtest(realised: np.ndarray, pd: float, rho: float, levels: Sequence[float] = (0.95, 0.99, 0.999),
                  primary_level: float = 0.99, model_id: str | None = None, segment: str | None = None,
                  alpha_amber: float = 0.05, alpha_red: float = 0.01) -> TestResult:
    """The evaluation-window loss rates against the quantiles of the fitted large-portfolio distribution.

    H0: the number of evaluation years above the fitted q-quantile is Binomial(n, 1 - q); alternative: more years
    exceed it than the fit says. Per level: fitted_quantile = lhp_quantile(level, pd, rho) (`engines.vasicek`),
    observed = the years with rate > fitted_quantile (strict), expected = n (1 - level) and the upper tail P(X >=
    observed). statistic = the observed count at `primary_level`, p_value its upper tail, metric the expected count
    there. extras["levels"] lists level, fitted_quantile, expected, observed, years_exceeding (indexes into
    `realised`) and p_value for every level, extras["n_missing"] the NaN years (excluded, never interpolated). No
    usable year: not_evaluated. Light (a Contraste internal gate; a Binomial test on a few years has little power, so
    the p-value is shown beside the counts): green when p_value >= `alpha_amber`, amber when `alpha_red` <= p_value <
    `alpha_amber`, red below `alpha_red`.
    """
    test_id = "portfolio.tail_backtest"
    h0 = "the number of evaluation years above the fitted q-quantile is Binomial(n, 1 - q)"
    alt = "more evaluation years exceed the fitted q-quantile than the fit says"
    reference = f"{BINOMIAL}; {G5}"
    _check_alphas(alpha_amber, alpha_red)
    lv = tuple(check_level(q, "level") for q in levels)
    if not lv:
        raise ValueError("levels is empty")
    primary = check_level(primary_level, "primary_level")
    if not any(math.isclose(primary, q, rel_tol=0.0, abs_tol=1e-12) for q in lv):
        raise ValueError(f"primary_level {primary_level} is not in levels {lv}")
    r = np.array(realised, dtype=float)
    if r.ndim != 1:
        raise ValueError(f"realised must be one-dimensional, got shape {r.shape}")
    miss = np.isnan(r)
    good = r[~miss]
    if np.any(np.isinf(good)) or np.any(good < 0.0) or np.any(good > 1.0):
        raise ValueError("realised loss rates must lie in [0, 1] (NaN marks a missing year)")
    # the engine validates pd and rho (open interval)
    quantiles = {q: float(lhp_quantile(q, pd, rho)) for q in lv}
    version = battery_label(POLICY_LABEL, "portfolio.tail_backtest", alpha_amber=alpha_amber, alpha_red=alpha_red)
    n = int(good.size)
    kw = dict(model_id=model_id, segment=segment, alpha_amber=alpha_amber, alpha_red=alpha_red)
    extras: dict[str, Any] = {"n_missing": int(miss.sum()), "primary_level": primary, "levels": []}
    if n < 1:
        return _not_evaluated(test_id, h0, alt, reference, version, "no usable evaluation year", n=0,
                              extras=extras, **kw)
    chosen: dict[str, Any] | None = None
    index = np.flatnonzero(~miss)
    for q in lv:
        exceed = good > quantiles[q]
        observed = int(exceed.sum())
        row = {"level": q, "fitted_quantile": quantiles[q], "expected": n * (1.0 - q), "observed": observed,
               "years_exceeding": [int(i) for i in index[exceed]],
               "p_value": float(stats.binom.sf(observed - 1, n, 1.0 - q))}
        extras["levels"].append(row)
        if math.isclose(q, primary, rel_tol=0.0, abs_tol=1e-12):
            chosen = row
    assert chosen is not None
    p = min(1.0, max(0.0, chosen["p_value"]))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
        light=_p_light(p, alpha_amber, alpha_red), statistic=float(chosen["observed"]), p_value=p,
        metric=float(chosen["expected"]), n=n, n_events=int(chosen["observed"]),
        inputs_hash=inputs_hash(r, float(pd), float(rho), lv, primary), extras=extras, **kw,
    )


@register(
    "portfolio.mc_convergence",
    family="portfolio",
    name="MC convergence",
    h0="the Monte Carlo estimate equals the analytic anchor",
    reference=f"{G5}; dossier 04 ladder rung C2 (the Monte Carlo quantile converges to C1 within its interval)",
    decision="p_value",
)
def mc_convergence(estimate: float, se: float, anchor: float, quantity: str = "var", model_id: str | None = None,
                   segment: str | None = None, alpha_amber: float = 0.05, alpha_red: float = 0.01) -> TestResult:
    """A Monte Carlo estimate against the analytic anchor (the LHP quantile of rung C1, or the CreditRisk+ recursion).

    H0: the Monte Carlo estimate equals the anchor. statistic = (estimate - anchor) / se, p_value = 2 (1 -
    Phi(|statistic|)), metric = estimate - anchor; extras carry estimate, se, anchor and quantity (one of var, es,
    ec). A non-positive se or a non-finite input: not_evaluated. The anchor is exact only for a large homogeneous
    portfolio: a finite portfolio sits above the LHP by the granularity effect, so compare an LHP-like portfolio and
    report the granularity gap apart (dossier 04 H.5). Light (a Contraste internal gate): green when p_value >=
    `alpha_amber`, amber when `alpha_red` <= p_value < `alpha_amber`, red below `alpha_red`.
    """
    test_id = "portfolio.mc_convergence"
    h0 = "the Monte Carlo estimate equals the analytic anchor"
    alt = "the Monte Carlo estimate differs from the analytic anchor"
    reference = f"{G5}; dossier 04 ladder rung C2 (the Monte Carlo quantile converges to C1 within its interval)"
    _check_alphas(alpha_amber, alpha_red)
    if quantity not in ("var", "es", "ec"):
        raise ValueError(f"quantity must be var, es or ec, got {quantity!r}")
    version = battery_label(POLICY_LABEL, "portfolio.mc_convergence", alpha_amber=alpha_amber, alpha_red=alpha_red)
    kw = dict(model_id=model_id, segment=segment, alpha_amber=alpha_amber, alpha_red=alpha_red)
    extras = {"estimate": estimate, "se": se, "anchor": anchor, "quantity": quantity}
    vals = [float(estimate), float(se), float(anchor)]
    if not all(math.isfinite(v) for v in vals):
        return _not_evaluated(test_id, h0, alt, reference, version, "a non-finite input", extras=extras, **kw)
    est, s, anc = vals
    if not s > 0.0:
        return _not_evaluated(test_id, h0, alt, reference, version, "the standard error is not positive",
                              extras=extras, **kw)
    z = (est - anc) / s
    p = min(1.0, float(2.0 * stats.norm.sf(abs(z))))
    return TestResult(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
                      light=_p_light(p, alpha_amber, alpha_red), statistic=z, p_value=p, metric=est - anc,
                      inputs_hash=inputs_hash(est, s, anc, quantity), extras=extras, **kw)


@register(
    "portfolio.euler_additivity",
    family="portfolio",
    name="Euler additivity",
    h0="the Euler contributions add up to the total",
    reference=f"{TASCHE_2008} (full allocation); {G5}",
    decision="metric",
)
def euler_additivity(contributions: np.ndarray, total: float, tolerance: float = 1e-6, model_id: str | None = None,
                     segment: str | None = None, basis: str | None = None) -> TestResult:
    """The Euler contributions add up to the total (dossier 05 G.4 and G.5: check the allocation sums to the total).

    `basis` names the allocation being tested (``exact-allocation`` for ES and aggregation allocations, ``kernel-var``
    for the kernel VaR contributions); it is written to ``extras["basis"]``, where the packaged policy
    ``contraste-portfolio-battery 1.0.0`` finds its tolerance (1e-06 and 0.01), and is ``None`` when the caller does not
    state it.

    statistic = sum(contributions), metric = |sum - total| / |total|, p_value null; extras carry sum, total and
    thresholds = {"tolerance": tolerance}. total = 0: not_evaluated. The ES contributions of `euler.es_contributions`
    pass at 1e-6; the kernel VaR contributions are tested with the document tolerance (0.01). Light (a Contraste
    internal gate, no amber: an allocation either adds up or it does not): green when metric <= tolerance, red above.
    """
    test_id = "portfolio.euler_additivity"
    h0 = "the Euler contributions add up to the total"
    alt = "the contributions do not add up to the total"
    reference = f"{TASCHE_2008} (full allocation); {G5}"
    c = float_vector(contributions, "contributions")
    tot = float(total)
    tol = float(tolerance)
    if not math.isfinite(tot):
        raise ValueError("total must be finite")
    if not (math.isfinite(tol) and tol > 0.0):
        raise ValueError("tolerance must be positive")
    version = battery_label(POLICY_LABEL, "portfolio.euler_additivity", {"extras": {"basis": basis}}, tolerance=tol)
    total_sum = float(c.sum())
    extras = {"sum": total_sum, "total": tot, "thresholds": {"tolerance": tol}, "basis": basis}
    kw = dict(model_id=model_id, segment=segment, n=int(c.size))
    if tot == 0.0:
        return _not_evaluated(test_id, h0, alt, reference, version, "the total is zero: no relative gap",
                              extras=extras, **kw)
    gap = abs(total_sum - tot) / abs(tot)
    return TestResult(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version,
                      light=Light.GREEN if gap <= tol + EPS else Light.RED, statistic=total_sum, metric=gap,
                      inputs_hash=inputs_hash(c, tot), extras=extras, **kw)


@register(
    "portfolio.crp_recursion_vs_mc",
    family="portfolio",
    name="CreditRisk+ recursion against Monte Carlo",
    h0="the recursion's distribution is the Monte Carlo sample's distribution",
    reference=f"{HRS_2004}; {KOLMOGOROV}; {G5}; dossier 04 ladder rung C4",
    decision="p_value",
)
def crp_recursion_vs_mc(pmf: np.ndarray, mc_units: np.ndarray, model_id: str | None = None, segment: str | None = None,
                        alpha_amber: float = 0.05, alpha_red: float = 0.01) -> TestResult:
    """The CreditRisk+ recursion's distribution against a Monte Carlo sample of the same Poisson-Gamma model (rung C4's
    gate).

    `pmf` (M + 1,) from `creditrisk_plus.pmf_stable`, `mc_units` (S,) integer loss units. statistic = D = max_n
    |F_pmf(n) -
    F_mc(n)| over n = 0..M (the Kolmogorov distance), p_value = 2 sum_{j>=1} (-1)^(j-1) exp(-2 j^2 S D^2) (the
    asymptotic
    Kolmogorov law, 1.0 at D = 0; conservative for a discrete null, the contract's definition), metric = D. extras carry
    negative_entries, the number of negative entries of `pmf` (the Panjer instability check). S < 10 or a sample
    above M:
    not_evaluated with the reason.
    Light (a Contraste internal gate): green when p_value >= `alpha_amber`, amber when `alpha_red` <= p_value <
    `alpha_amber`, red
    below `alpha_red` and red whatever the p-value when any entry of `pmf` is negative.
    """
    test_id = "portfolio.crp_recursion_vs_mc"
    h0 = "the recursion's distribution is the Monte Carlo sample's distribution"
    alt = "the recursion's distribution differs from the Monte Carlo sample's"
    reference = f"{HRS_2004}; {KOLMOGOROV}; {G5}; dossier 04 ladder rung C4"
    _check_alphas(alpha_amber, alpha_red)
    f = float_vector(pmf, "pmf")
    raw = np.array(mc_units)
    if raw.ndim != 1:
        raise ValueError(f"mc_units must be one-dimensional, got shape {raw.shape}")
    if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)) or np.any(np.mod(raw, 1) != 0) or np.any(raw < 0):
        raise ValueError("mc_units must hold nonnegative integer loss units")
    units = raw.astype(np.int64)
    version = battery_label(POLICY_LABEL, "portfolio.crp_recursion_vs_mc", alpha_amber=alpha_amber, alpha_red=alpha_red)
    s = int(units.size)
    negative = int((f < 0.0).sum())
    extras = {"negative_entries": negative, "grid_size": int(f.size)}
    kw = dict(model_id=model_id, segment=segment, n=s, alpha_amber=alpha_amber, alpha_red=alpha_red)
    if s < MIN_MC:
        return _not_evaluated(test_id, h0, alt, reference, version, f"fewer than {MIN_MC} Monte Carlo scenarios",
                              extras=extras, **kw)
    if int(units.max()) >= f.size:
        return _not_evaluated(test_id, h0, alt, reference, version,
                              f"the sample has losses above the grid of {f.size - 1} loss units", extras=extras, **kw)
    cdf_mc = np.cumsum(np.bincount(units, minlength=f.size)) / s
    d = float(np.max(np.abs(np.cumsum(f) - cdf_mc)))
    p = min(1.0, max(0.0, float(stats.kstwobign.sf(math.sqrt(s) * d))))
    light = Light.RED if negative > 0 else _p_light(p, alpha_amber, alpha_red)
    return TestResult(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version, light=light,
                      statistic=d, p_value=p, metric=d, n_events=negative, inputs_hash=inputs_hash(f, units),
                      extras=extras, **kw)


@register(
    "portfolio.correlation_size_power",
    family="portfolio",
    name="Correlation size power",
    h0="at the null correlation the rejection rate equals alpha (the size); elsewhere the rate is the power",
    reference=f"{WILSON_1927}; {G5}; dossier 04 C.4 (the Vasicek correlation) on a simulator with a known truth",
    decision="metric",
)
def correlation_size_power(rejected: np.ndarray, known_rho: float, null_rho: float, alpha: float = 0.05,
                           estimator: str | None = None, model_id: str | None = None, segment: str | None = None,
                           alpha_amber: float = 0.05, alpha_red: float = 0.01, power_green: float = 0.80,
                           power_amber: float = 0.50) -> TestResult:
    """The size and the power of a correlation test on a simulator with a known truth.

    `rejected` (R,) bool, one per valid replication: whether H0: rho = null_rho was rejected at level `alpha`. H0: at
    known_rho = null_rho the rejection rate equals `alpha` (the size); at any other known correlation the rate is the
    power. rate = mean(rejected), se = sqrt(rate (1 - rate) / R), and the Wilson 95% interval. A size cell (known_rho
    equal to null_rho to 1e-12): statistic = (rate - alpha) / sqrt(alpha (1 - alpha) / R), two-sided normal p_value,
    metric = rate. A power cell: statistic and p_value null, metric = rate. extras carry role (size or power),
    known_rho, null_rho, alpha, rate, se, ci_low, ci_high, estimator and thresholds = {"power_green", "power_amber"}.
    R < 1: not_evaluated. Light (Contraste internal gates): size cell, green when p_value >= `alpha_amber`, amber when
    `alpha_red` <= p_value < `alpha_amber`, red below; power cell, green when rate >= `power_green` (0.80, the
    conventional power), amber when rate >= `power_amber` (0.50), red below.
    """
    test_id = "portfolio.correlation_size_power"
    h0 = "at the null correlation the rejection rate equals alpha (the size); elsewhere the rate is the power"
    alt = "the rejection rate differs from alpha at the null correlation"
    reference = f"{WILSON_1927}; {G5}; dossier 04 C.4 (the Vasicek correlation) on a simulator with a known truth"
    _check_alphas(alpha_amber, alpha_red)
    for name, v in (("known_rho", known_rho), ("null_rho", null_rho), ("alpha", alpha)):
        if not (math.isfinite(float(v)) and 0.0 < float(v) < 1.0):
            raise ValueError(f"{name} must lie in (0, 1), got {v}")
    if not 0.0 < power_amber <= power_green <= 1.0:
        raise ValueError("power_green and power_amber must satisfy 0 < power_amber <= power_green <= 1")
    raw = np.array(rejected)
    if raw.ndim != 1:
        raise ValueError(f"rejected must be one-dimensional, got shape {raw.shape}")
    if raw.size and (raw.dtype.kind not in "biu" or not np.all(np.isin(raw, (0, 1)))):
        raise ValueError("rejected must be boolean (or 0 and 1)")
    flags = raw.astype(bool)
    r = int(flags.size)
    version = battery_label(
            POLICY_LABEL, "portfolio.correlation_size_power", alpha_amber=alpha_amber, alpha_red=alpha_red,
            power_green=power_green, power_amber=power_amber)
    size_cell = abs(float(known_rho) - float(null_rho)) <= 1e-12
    extras: dict[str, Any] = {"role": "size" if size_cell else "power", "known_rho": float(known_rho),
                              "null_rho": float(null_rho), "alpha": float(alpha), "estimator": estimator,
                              "thresholds": {"power_green": power_green, "power_amber": power_amber}}
    kw = dict(model_id=model_id, segment=segment, n=r, alpha_amber=alpha_amber, alpha_red=alpha_red)
    if r < 1:
        return _not_evaluated(test_id, h0, alt, reference, version, "no valid replication", extras=extras, **kw)
    k = int(flags.sum())
    rate = k / r
    se = math.sqrt(rate * (1.0 - rate) / r)
    centre = (rate + WILSON_Z**2 / (2 * r)) / (1.0 + WILSON_Z**2 / r)
    half = WILSON_Z * math.sqrt(rate * (1.0 - rate) / r + WILSON_Z**2 / (4 * r * r)) / (1.0 + WILSON_Z**2 / r)
    extras.update(rate=rate, se=se, ci_low=max(0.0, centre - half), ci_high=min(1.0, centre + half), n_rejected=k)
    common = dict(test_id=test_id, h0=h0, alternative=alt, reference=reference, policy_version=version, metric=rate,
                  n_events=k, inputs_hash=inputs_hash(flags, float(known_rho), float(null_rho), float(alpha)),
                  extras=extras, **kw)
    if not size_cell:
        light = Light.GREEN if rate >= power_green - EPS else Light.AMBER if rate >= power_amber - EPS else Light.RED
        return TestResult(light=light, **common)
    z = (rate - alpha) / math.sqrt(alpha * (1.0 - alpha) / r)
    p = min(1.0, float(2.0 * stats.norm.sf(abs(z))))
    return TestResult(light=_p_light(p, alpha_amber, alpha_red), statistic=z, p_value=p, **common)

__all__ = [
    "correlation_size_power",
    "crp_recursion_vs_mc",
    "euler_additivity",
    "mc_convergence",
    "tail_backtest",
]
