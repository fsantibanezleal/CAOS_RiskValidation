"""The CCR validation tests (dossier 05 C and the counterparty ladder at line 476; the U9 research addendum R4, R5 and
R9): each returns one ``TestResult`` that validates against ``common/test-result``, with a ``test_id`` prefixed
``ccr.``.

Acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds. Each function states its light
rule in its docstring and records its parameters in ``policy_version`` and in ``extras["thresholds"]``, so that a view
can explain a light. A test that cannot be evaluated returns the light ``not_evaluated`` with its reason in ``notes``
and a null statistic, metric and p-value once serialised, never a fabricated number. The tests are plain functions that
do not read the registry or the default policy: their rules are the contract's (``engine-ccr-api``, module
validation.ccr). They state no compliance claim.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2, norm

from riskvalidation.ccr import cva as _cva
from riskvalidation.ccr import exposure as _exposure
from riskvalidation.core.result import Light, TestResult
from riskvalidation.lgd_ead._checks import mat, same_length, scalar, vec

POLICY_LABEL = "contraste-ccr-battery-1.0.0"
NAN = float("nan")
EPS = 1e-12             # tolerance on a gate comparison, so that a value exactly on a gate is not moved by rounding
ZERO_ERROR = 1e-12      # a difference at or below this is zero where the standard error is zero

IID_CRITICAL_VALUES: dict[float, float] = {
    0.25: 1.248, 0.15: 1.610, 0.10: 1.933, 0.05: 2.492, 0.01: 3.880, 0.005: 4.500, 0.001: 6.000}
"""The upper-tail probabilities of the Anderson-Darling ``W^2`` for independent uniform PITs and their critical values
(Anderson and Darling; Formenti et al., arXiv 1505.04593, Table 1). Reported beside a simulated rule, never in its
place."""

REF_ANALYTIC = (
    "dossier 05 C.4 and the U9 research addendum R4: the Hull-White one-factor closed forms (QuantLib 1.44 "
    "hullwhite.cpp and jamshidianswaptionengine.cpp); ladder rung 2 against rung 3 (dossier 05, counterparty ladder)"
)
REF_MARTINGALE = ("U9 research addendum R4: the theta fit reprices today's curve, so discounted bond prices are "
                  "martingales (QuantLib 1.44 hullwhite.hpp)")
REF_EEPE = "BCBS CRE53.12 and CRE53.13, Effective EE and Effective EPE (addendum R9); dossier 05 C.1"
REF_CVA = "dossier 05 C.2, CVA = sum PD LGD EPE (ORE xva.tex, the advanced discretisation)"
REF_CONVERGENCE = "dossier 05, counterparty ladder: convergence in the number of paths; ORE xva.tex"
REF_RECONCILE = ("dossier 05, counterparty ladder: the engine's EE profile matches ORE 1.8.17.0 within Monte Carlo "
                 "error at each grid point")
REF_BACKTEST = (
    "BCBS 185 (2010), Sound practices for backtesting counterparty credit risk models, paragraphs 25 to 31; CRE53.33 "
    "(addendum R5)"
)
REF_PIT = (
    "BCBS 185 paragraph 25 (reliability); Formenti, Spadafora, Terraneo and Ramponi, arXiv 1505.04593 (PIT and "
    "Anderson-Darling); critical values simulated under the model (addendum R5; Ruiz 2014 and Anfuso et al. 2014 not "
    "read)"
)
CAVEAT_BCBS_185_27 = (
    "BCBS 185 paragraph 27: criteria based on exception counting often assume independence, and the data may overlap "
    "(dates closer than the horizon); read the count with that in mind"
)


def _version(**params: Any) -> str:
    return "; ".join([POLICY_LABEL, *(f"{k}={v}" for k, v in params.items())])


def _not_evaluated(test_id: str, h0: str, alternative: str, reference: str, version: str, reason: str, *,
                   n: int | None = None, extras: Mapping[str, Any] | None = None) -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=version,
                      light=Light.NOT_EVALUATED, statistic=NAN, notes=f"not evaluated: {reason}", n=n,
                      extras=dict(extras or {}))


def _positive(x: float, name: str) -> float:
    v = scalar(x, name)
    if v <= 0.0:
        raise ValueError(f"{name} must be positive")
    return v


def _nonneg_vec(x: np.ndarray, name: str) -> np.ndarray:
    v = vec(x, name)
    if np.any(v < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return v


def _scaled_gap(diff: np.ndarray, se: np.ndarray) -> np.ndarray:
    """``|diff| / se`` per point; where ``se`` is zero the ratio is 0 for a zero difference and infinite otherwise."""
    gap = np.abs(diff)
    out = np.zeros_like(gap)
    pos = se > 0.0
    out[pos] = gap[pos] / se[pos]
    zero = ~pos
    out[zero] = np.where(gap[zero] <= ZERO_ERROR, 0.0, np.inf)
    return out


def _within_se(test_id: str, h0: str, alternative: str, reference: str, actual: np.ndarray, expected: np.ndarray,
               se: np.ndarray, k_se: float) -> TestResult:
    ratio = _scaled_gap(actual - expected, se)
    stat, metric = float(np.max(ratio)), float(np.max(np.abs(actual - expected)))
    ok = stat <= k_se + EPS
    return TestResult(
        test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=_version(k_se=k_se),
        light=Light.GREEN if ok else Light.RED, statistic=stat, metric=metric, n=int(actual.size),
        extras={"thresholds": {"k_se": k_se}},
        notes="" if ok else (f"the largest standardised difference {stat:.4g} is above k_se = {k_se}"
                             + ("; a point has zero standard error and a nonzero difference"
                                if math.isinf(stat) else "")))


def ee_analytic_vs_simulated(ee_analytic: np.ndarray, ee_simulated: np.ndarray, mc_se: np.ndarray,
                             k_se: float = 3.0) -> TestResult:
    """The analytic EE of a single swap (``exposure.analytic_ee_single_swap``) against the simulated one at the reset
    dates (ladder rung 2 against rung 3), money (K,) each.

    statistic = the largest ``|analytic - simulated| / mc_se`` (a point with no Monte Carlo error
    counts 0 for a zero difference and infinity otherwise); metric = the largest absolute difference. Light (a Contraste
    internal gate): green when every grid point is within ``k_se`` standard errors, red otherwise;
    ``extras["thresholds"] = {"k_se": ...}``. The identity is a risk-neutral one: the caller passes the
    simulated EE on the same measure."""
    k = _positive(k_se, "k_se")
    an, sim, se = vec(ee_analytic, "ee_analytic"), vec(ee_simulated, "ee_simulated"), _nonneg_vec(mc_se, "mc_se")
    same_length(ee_analytic=an, ee_simulated=sim, mc_se=se)
    return _within_se("ccr.ee_analytic_vs_simulated",
                      "the analytic and the simulated EE agree within the Monte Carlo error",
                      "the simulated EE departs from the closed form", REF_ANALYTIC, an, sim, se, k)


def martingale_discounted_bonds(discounted_bond_means: np.ndarray, p0: np.ndarray, mc_se: np.ndarray,
                                k_se: float = 3.0) -> TestResult:
    """The sample mean of ``exp(-integral r) P(t, T; r_t)`` against ``P(0, T)`` at each grid
    point, unitless (K,).

    statistic = the largest ``|mean - P(0, T)| / mc_se``; metric = the largest absolute
    difference. Light (a Contraste internal gate): green within ``k_se`` standard errors, red otherwise. The exact
    version is the repricing of ``P(0, T)`` to 1e-12 by ``hw1f.bond_price`` at time 0."""
    k = _positive(k_se, "k_se")
    mean, p, se = vec(discounted_bond_means, "discounted_bond_means"), vec(p0, "p0"), _nonneg_vec(mc_se, "mc_se")
    same_length(discounted_bond_means=mean, p0=p, mc_se=se)
    return _within_se("ccr.martingale_discounted_bonds",
                      "the discounted bond prices are martingales: their mean is P(0, T)",
                      "the simulated short rate does not reprice today's discount curve", REF_MARTINGALE, mean, p,
                      se, k)


def _recomputed(test_id: str, h0: str, alternative: str, reference: str, recomputed: float, reported: float,
                rel_tol: float, n: int) -> TestResult:
    tol = scalar(rel_tol, "rel_tol")
    if tol < 0.0:
        raise ValueError("rel_tol must be nonnegative")
    rep = scalar(reported, "reported value")
    resid = abs(recomputed - rep)
    rel = resid / abs(recomputed) if recomputed != 0.0 else (0.0 if resid == 0.0 else math.inf)
    ok = rel <= tol
    return TestResult(
        test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=_version(rel_tol=tol),
        light=Light.GREEN if ok else Light.RED, statistic=resid, metric=rel, n=n,
        extras={"thresholds": {"rel_tol": tol}, "recomputed": recomputed, "reported": rep},
        notes="" if ok else f"the reported value departs from the recomputed one by a relative {rel:.3g}")


def eepe_recomputed(grid: np.ndarray, ee: np.ndarray, eepe_reported: float, rel_tol: float = 1e-9) -> TestResult:
    """The effective EPE recomputed from the EE profile (``exposure.effective_ee`` from a zero current exposure, then
    ``exposure.effective_epe`` over the first year) against the reported value, money.

    statistic = the absolute residual; metric = the residual relative to the recomputed value. Light (a Contraste
    internal gate): green when the relative residual is at most ``rel_tol``, red otherwise. The plain EPE is not the
    effective one: reporting it is red."""
    recomputed = _exposure.effective_epe(grid, _exposure.effective_ee(ee))
    return _recomputed("ccr.eepe_recomputed", "the reported EEPE equals the one recomputed from the EE profile",
                       "the reported EEPE is not the effective EPE of the EE profile", REF_EEPE, recomputed,
                       eepe_reported, rel_tol, int(np.asarray(ee).size))



def cva_recomputed(epe_discounted: np.ndarray, pd: np.ndarray, lgd: float, cva_reported: float,
                   rel_tol: float = 1e-9) -> TestResult:
    """The CVA recomputed from the discounted EE (``cva.cva``, ``sum PD LGD EPE``) against the reported
    value, money.

    statistic = the absolute residual; metric = the relative one. Light (a Contraste internal gate): green when the
    relative residual is at most ``rel_tol``, red otherwise."""
    value, _ = _cva.cva(epe_discounted, pd, lgd)
    return _recomputed("ccr.cva_recomputed", "the reported CVA equals the one recomputed from the discounted EE",
                       "the reported CVA departs from sum PD LGD EPE", REF_CVA, value, cva_reported, rel_tol,
                       int(np.asarray(pd).size))


def path_convergence(ee_by_path_count: dict[int, np.ndarray], cva_by_path_count: dict[int, float], *,
                     ee_se_by_path_count: dict[int, np.ndarray], cva_se_by_path_count: dict[int, float],
                     k_se: float = 3.0) -> TestResult:
    """The change of the EE profile and of the CVA between the two largest path counts, money; the keys are path counts.
    The caller passes the Monte Carlo standard errors of each count, ``ee_se_by_path_count`` (K,) and
    ``cva_se_by_path_count``, with the same keys.

    statistic = the largest ``|EE_large - EE_small| / se_small`` over the grid (a point with no standard error counts 0
    for a zero change and infinity otherwise); metric = ``|CVA_large - CVA_small| / se_small`` of the CVA. The band is
    ``k_se`` times the standard error of the smaller count (the unit coefficient-of-variation band of earlier versions
    is gone: the EE of an at-the-money swap has a coefficient of variation above 1, so it lit on noise). Light (a
    Contraste internal gate): green when the larger of the two is within ``k_se``, amber within ``2 k_se``, red
    beyond; ``extras["thresholds"]``. Fewer than two path counts is ``not_evaluated`` with its reason."""
    k = _positive(k_se, "k_se")
    if set(ee_by_path_count) != set(cva_by_path_count):
        raise ValueError("cva_by_path_count must have the same path counts as ee_by_path_count")
    if set(ee_se_by_path_count) != set(ee_by_path_count):
        raise ValueError("ee_se_by_path_count must have the same path counts as ee_by_path_count")
    if set(cva_se_by_path_count) != set(ee_by_path_count):
        raise ValueError("cva_se_by_path_count must have the same path counts as ee_by_path_count")
    counts = sorted(ee_by_path_count)
    if any(int(c) != c or c < 1 for c in counts):
        raise ValueError("ee_by_path_count keys must be positive integers (path counts)")
    test_id = "ccr.path_convergence"
    h0 = "the EE profile and the CVA are converged in the number of paths"
    alt = "the EE profile or the CVA still moves with the number of paths"
    if len(counts) < 2:
        return _not_evaluated(test_id, h0, alt, REF_CONVERGENCE, _version(k_se=k),
                              "fewer than two path counts, so no change can be measured")
    small_n, big_n = counts[-2], counts[-1]
    small, big = vec(ee_by_path_count[small_n], "ee_by_path_count"), vec(ee_by_path_count[big_n], "ee_by_path_count")
    if small.size != big.size:
        raise ValueError("ee_by_path_count profiles must have the same length")
    se_small = _nonneg_vec(ee_se_by_path_count[small_n], "ee_se_by_path_count")
    if se_small.size != small.size:
        raise ValueError("ee_se_by_path_count profiles must have the length of ee_by_path_count")
    _nonneg_vec(ee_se_by_path_count[big_n], "ee_se_by_path_count")
    stat = float(np.max(_scaled_gap(big - small, se_small)))
    c_small = scalar(cva_by_path_count[small_n], "cva_by_path_count")
    c_big = scalar(cva_by_path_count[big_n], "cva_by_path_count")
    cse = scalar(cva_se_by_path_count[small_n], "cva_se_by_path_count")
    if cse < 0.0:
        raise ValueError("cva_se_by_path_count must be nonnegative")
    metric = float(_scaled_gap(np.array([c_big - c_small]), np.array([cse]))[0])
    change = max(stat, metric)
    if change <= k + EPS:
        light = Light.GREEN
    elif change <= 2.0 * k + EPS:
        light = Light.AMBER
    else:
        light = Light.RED
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_CONVERGENCE,
        policy_version=_version(k_se=k, band="k_se times the standard error of the smaller path count"),
        light=light, statistic=stat, metric=metric, n=int(big_n),
        extras={"thresholds": {"k_se": k, "smaller_count": int(small_n), "larger_count": int(big_n)}},
        notes="" if light is Light.GREEN else ("the profile or the CVA moved more than k_se standard errors of the "
                                               "smaller path count"))


def reconcile_ee_profile(ee: np.ndarray, mc_se: np.ndarray, reference_ee: np.ndarray, reference_se: np.ndarray,
                         k_se: float = 2.0) -> TestResult:
    """The engine's EE profile against a reference profile (ORE's, passed in as arrays) within the combined Monte Carlo
    error, money (K,) each.

    statistic = the largest ``|ee - reference| / sqrt(mc_se^2 + reference_se^2)``;
    ``extras["agreement"]`` lists the per-point difference and combined error. Light (a Contraste internal gate): green
    when every point is within ``k_se`` combined standard errors (default 2), red otherwise."""
    k = _positive(k_se, "k_se")
    e, se = vec(ee, "ee"), _nonneg_vec(mc_se, "mc_se")
    ref, rse = vec(reference_ee, "reference_ee"), _nonneg_vec(reference_se, "reference_se")
    same_length(ee=e, mc_se=se, reference_ee=ref, reference_se=rse)
    comb = np.sqrt(se**2 + rse**2)
    ratio = _scaled_gap(e - ref, comb)
    stat = float(np.max(ratio))
    ok = stat <= k + EPS
    agreement = [{"difference": float(d), "combined_se": float(c)} for d, c in zip(e - ref, comb, strict=True)]
    return TestResult(
        test_id="ccr.reconcile_ee_profile",
        h0="the engine's EE profile agrees with the reference within the Monte Carlo error",
        alternative="the engine's EE profile departs from the reference", reference=REF_RECONCILE,
        policy_version=_version(k_se=k), light=Light.GREEN if ok else Light.RED, statistic=stat,
        metric=float(np.max(np.abs(e - ref))), n=int(e.size),
        extras={"thresholds": {"k_se": k}, "agreement": agreement},
        notes="" if ok else f"the largest difference is {stat:.4g} combined standard errors")


def pfe_exceedance_backtest(realised: np.ndarray, pfe_forecast: np.ndarray, quantile: float) -> TestResult:
    """The exceedances of the PFE forecast by the realised mark-to-market of the revalued book, money (N,) each.

    An exceedance is a realised value strictly above its forecast; ``n_events`` is their number. statistic = the Kupiec
    likelihood ratio of unconditional coverage against the expected rate ``1 - quantile``, ``p_value`` its
    chi-square(1) value, metric the exceedance frequency. Light (a Contraste internal gate): green when ``p_value`` is
    at least 0.05, amber below 0.05, red below 0.01; ``extras["thresholds"]``. BCBS 185 paragraph 27 caveat: exception
    counts assume independence and the data may overlap; it is carried in ``extras["caveat"]``."""
    r, f = vec(realised, "realised"), vec(pfe_forecast, "pfe_forecast")
    same_length(realised=r, pfe_forecast=f)
    q = scalar(quantile, "quantile")
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    n = int(r.size)
    x = int(np.sum(r > f))
    ll0 = (n - x) * math.log(q) + x * math.log1p(-q)
    ll1 = float(xlogy(n - x, 1.0 - x / n) + xlogy(x, x / n))
    lr = max(2.0 * (ll1 - ll0), 0.0)
    p = float(chi2.sf(lr, 1))
    light = Light.GREEN if p >= 0.05 else Light.AMBER if p >= 0.01 else Light.RED
    return TestResult(
        test_id="ccr.pfe_exceedance_backtest", h0=f"the PFE forecast at {q} is exceeded with frequency {1.0 - q:.4g}",
        alternative="the exceedance frequency differs from the forecast one", reference=REF_BACKTEST,
        policy_version=_version(quantile=q, alpha_amber=0.05, alpha_red=0.01), light=light, statistic=lr, p_value=p,
        metric=x / n, alpha_amber=0.05, alpha_red=0.01, n=n, n_events=x,
        extras={"thresholds": {"alpha_amber": 0.05, "alpha_red": 0.01}, "quantile": q, "caveat": CAVEAT_BCBS_185_27},
        notes="" if light is Light.GREEN else f"{x} exceedances in {n} dates against {(1.0 - q) * n:.3g} expected")


def pit_from_simulated(realised: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """(N,) the probability integral transform of each realised value in its simulated sample (N, P) by
    ``(i + 1) / (P + 2)``, ``i`` the number of simulated values strictly below the
    realised one: ``1 / (P + 2)`` below the whole sample and
    ``(P + 1) / (P + 2)`` above it (addendum R5, Formenti et al. arXiv 1505.04593)."""
    r = vec(realised, "realised")
    s = mat(simulated, "simulated")
    if s.shape[0] != r.size:
        raise ValueError(f"simulated must have one row per realised value ({r.size}), got {s.shape[0]}")
    below = np.sum(s < r[:, None], axis=1)
    return (below + 1.0) / (s.shape[1] + 2.0)


def _check_pits(pits: np.ndarray) -> np.ndarray:
    p = vec(pits, "pits")
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("pits must lie strictly inside (0, 1)")
    return p


def anderson_darling_w2(pits: np.ndarray) -> float:
    """``W^2 = -n - sum_{k=1}^n (2k - 1) / n [ln F(x_k) + ln(1 - F(x_{n+1-k}))]``
    of the PITs (the ordered values ``x_k``; Anderson and Darling; Formenti et al. arXiv 1505.04593 equation 2); ``F``
    is the uniform CDF, so ``F(x) = x``. The PITs must lie strictly inside (0, 1)."""
    x = np.sort(_check_pits(pits))
    n = x.size
    k = np.arange(1, n + 1)
    return float(-n - np.sum((2 * k - 1) / n * (np.log(x) + np.log1p(-x[::-1]))))


def effective_sample_size(pits: np.ndarray, max_lag: int | None = None) -> float:
    """The number of independent observations the PITs are worth: with ``z_i = Phi^-1(PIT_i)`` and ``rho_k``
    the sample autocorrelation of ``z``,
    ``ESS = n / (1 + 2 sum_{k=1}^L (1 - k / (L + 1)) rho_k)``
    (Bartlett weights, ``L = max_lag``, by default the integer part of ``n ** (1/3)``), clipped to
    ``[1, n]``. Non-overlapping (uncorrelated) PITs give about ``n``; overlapping horizons give less. A convention
    of this contract, UNVERIFIED as a backtesting standard (BCBS 185 paragraph 25 only requires that the size be
    stated)."""
    p = _check_pits(pits)
    n = p.size
    if n < 2:
        raise ValueError("pits needs at least 2 observations")
    lag = max(int(n ** (1.0 / 3.0)), 1) if max_lag is None else max_lag
    if int(lag) != lag or not 1 <= lag <= n - 1:
        raise ValueError(f"max_lag must be an integer in [1, {n - 1}], got {max_lag}")
    z = norm.ppf(p)
    zc = z - z.mean()
    denom = float(zc @ zc)
    if denom <= 0.0:
        raise ValueError("pits have no variance: the autocorrelation is undefined")
    rho = np.array([float(zc[: n - j] @ zc[j:]) / denom for j in range(1, int(lag) + 1)])
    weights = 1.0 - np.arange(1, int(lag) + 1) / (lag + 1.0)
    inflation = 1.0 + 2.0 * float(np.sum(weights * rho))
    ess = n / inflation if inflation > 0.0 else float(n)
    return float(min(max(ess, 1.0), float(n)))


_ESS = effective_sample_size


@dataclass(frozen=True, eq=False)
class CriticalValues:
    p05: float
    p01: float
    effective_sample_size: float
    n_backtests: int
    w2_samples: np.ndarray

    def __post_init__(self) -> None:
        if not self.p05 > 0.0:
            raise ValueError(f"p05 must be positive, got {self.p05}")
        if not self.p01 >= self.p05:
            raise ValueError(f"p01 must be at least p05, got p01={self.p01} and p05={self.p05}")
        if not self.effective_sample_size >= 1.0:
            raise ValueError(f"effective_sample_size must be at least 1, got {self.effective_sample_size}")
        if self.n_backtests < 1 or np.shape(self.w2_samples) != (self.n_backtests,):
            raise ValueError("w2_samples must hold one statistic per backtest")


def simulate_pit_critical_values(pit_sampler: Callable[[np.random.Generator], np.ndarray], n_backtests: int, seed: int,
                                 max_lag: int | None = None) -> CriticalValues:
    """The 95th and 99th percentiles of ``W^2`` when the whole backtest is simulated under the model itself (B5,
    addendum R5): ``pit_sampler(rng)`` returns the PITs (n,) of one backtest at the same dates and with the same
    overlapping horizons as the real one (the caller draws the realised path from the model and applies the
    ``(i + 1) / (P + 2)`` rule). ``p05`` and ``p01`` are the inverse empirical quantiles
    over ``n_backtests`` statistics, the effective sample size is that of the first simulated series, the same ``seed``
    gives the same result; fewer than 100 backtests or PITs outside (0, 1) raise ``ValueError``."""
    nb = int(n_backtests)
    if nb != n_backtests or nb < 100:
        raise ValueError(
            f"n_backtests must be an integer of at least 100 (a percentile of fewer is not stated), got {n_backtests}")
    rng = np.random.default_rng(seed)
    w2 = np.empty(nb)
    first: np.ndarray | None = None
    for i in range(nb):
        pits = np.asarray(pit_sampler(rng), dtype=float)
        if pits.ndim != 1 or pits.size == 0:
            raise ValueError("pit_sampler must return a one-dimensional array of PITs")
        if not np.all(np.isfinite(pits)) or np.any(pits <= 0.0) or np.any(pits >= 1.0):
            raise ValueError("pit_sampler returned a PIT outside (0, 1)")
        if first is None:
            first = pits.copy()
        w2[i] = anderson_darling_w2(pits)
    assert first is not None
    p05 = float(np.quantile(w2, 0.95, method="inverted_cdf"))
    p01 = float(np.quantile(w2, 0.99, method="inverted_cdf"))
    return CriticalValues(p05, p01, _ESS(first, max_lag), nb, w2)


def pit_exposure_backtest(realised: np.ndarray, simulated: np.ndarray,
                          critical_simulated: dict[str, float] | None = None,
                          effective_sample_size: float | None = None) -> TestResult:
    """The PIT-based exposure backtest (BCBS 185 paragraph 25; addendum R5): ``realised`` (N,) the realised values at
    the horizon, one per backtest date, ``simulated`` (N, P) the simulated sample at the same horizon.

    statistic = the Anderson-Darling ``W^2`` of the PITs ``(i + 1) / (P + 2)``; ``extras``
    carry the PITs, the iid critical values (``IID_CRITICAL_VALUES``, beside the simulated rule, never in its place),
    the simulated ones and the effective sample size (given, or ``effective_sample_size`` of the PITs). Light (a
    Contraste internal gate): without ``critical_simulated`` (``{"p05": ..., "p01": ...}`` from
    ``simulate_pit_critical_values``) the result is ``not_evaluated`` with the reason "no simulated critical values" and
    no number; otherwise green at or below the simulated 5 percent value, amber up to the 1 percent value, red above
    it."""
    test_id = "ccr.pit_exposure_backtest"
    h0 = "the PITs of the realised values in the simulated distributions are uniform"
    alt = "the realised values are not draws from the simulated distributions"
    r = vec(realised, "realised")
    if r.size < 2:
        raise ValueError("realised needs at least 2 backtest dates")
    pits = pit_from_simulated(r, simulated)
    if critical_simulated is None:
        return _not_evaluated(test_id, h0, alt, REF_PIT, _version(critical="none"), "no simulated critical values",
                              n=int(r.size), extras={"reason": "no simulated critical values"})
    if effective_sample_size is not None:
        ess = scalar(effective_sample_size, "effective_sample_size")
        if not 1.0 <= ess <= r.size:
            raise ValueError(f"effective_sample_size must lie in [1, {r.size}], got {ess}")
    else:
        ess = 1.0 if float(np.ptp(pits)) == 0.0 else _ESS(pits)     # equal PITs are fully dependent: one observation
    try:
        p05, p01 = float(critical_simulated["p05"]), float(critical_simulated["p01"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"critical_simulated must hold the numeric keys 'p05' and 'p01': {exc}") from exc
    if not (math.isfinite(p05) and math.isfinite(p01) and 0.0 < p05 <= p01):
        raise ValueError(f"critical_simulated must satisfy 0 < p05 <= p01, got p05={p05} and p01={p01}")
    w2 = anderson_darling_w2(pits)
    light = Light.GREEN if w2 <= p05 + EPS else Light.AMBER if w2 <= p01 + EPS else Light.RED
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=REF_PIT,
        policy_version=_version(simulated_p05=p05, simulated_p01=p01, effective_sample_size=ess), light=light,
        statistic=w2, n=int(r.size),
        extras={"pits": [float(x) for x in pits],
                "iid_critical_values": {str(k): v for k, v in IID_CRITICAL_VALUES.items()},
                "simulated_critical_values": {"p05": p05, "p01": p01}, "effective_sample_size": ess,
                "thresholds": {"simulated_p05": p05, "simulated_p01": p01}},
        notes="" if light is Light.GREEN else "W^2 is above the simulated 5 percent critical value")
