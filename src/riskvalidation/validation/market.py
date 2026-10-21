"""The market-risk back-tests (dossier 05 B.1 to B.5, dossier 01 D.4): each returns one `TestResult`.

Acceptance gates are Contraste internal gates, stated as such, not regulatory thresholds. A light rule compares the
p-value (or the stated statistic) to `alpha_amber` and `alpha_red` supplied by the caller: red when `p < alpha_red`,
amber when `p < alpha_amber`, otherwise green; the parameters of a rule that are not alphas travel in
`extras["thresholds"]`. A test that
cannot be evaluated returns the light `not_evaluated` with its reason in `notes` and a null statistic, p-value and
metric once serialised, never a fabricated number. The functions are plain functions that do not read the registry or
the default policy:
their rules are the contract's (`engine-market-api`, module validation.market). They state no compliance claim.

Conventions: `exceedances` and `hits` are flags (`I_t = 1{loss_t > VaR_t}`), losses, VaR and ES of the ES tests are
positive loss numbers (the document convention, `market.conventions.to_loss_numbers`), `dynamic_quantile_test` takes
the VaR in the return convention of the CAViaR it tests (the statistic is invariant to an affine change of the VaR). A
NaN return raises.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from scipy import optimize, special, stats

from riskvalidation.core.hashing import inputs_hash
from riskvalidation.core.registry import register
from riskvalidation.core.result import Light, TestResult
from riskvalidation.market import frtb
from riskvalidation.market.caviar import dq_statistic_unchecked, dynamic_quantile_statistic
from riskvalidation.market.conventions import vec

POLICY_LABEL = "contraste-market-battery 1.0.0"
NAN = float("nan")

KUPIEC_1995 = "Kupiec (1995), J. Derivatives 3(2):73-84, doi:10.3905/jod.1995.407942"
CHRISTOFFERSEN_1998 = "Christoffersen (1998), Int. Economic Review 39(4):841-862, doi:10.2307/2527341"
CHRISTOFFERSEN_PELLETIER_2004 = (
    "Christoffersen and Pelletier (2004), J. Financial Econometrics 2(1):84-108, doi:10.1093/jjfinec/nbh004"
)
BASEL_1996 = (
    "BCBS (1996), Supervisory framework for the use of backtesting, https://www.bis.org/publ/bcbs22.htm; "
    "MAR32.9 Table 1 "
    "(dossier 01 D.4); 12 CFR 217.204 is a reference only"
)
ACERBI_SZEKELY_2014 = (
    "Acerbi and Szekely (2014), Backtesting expected shortfall, Risk, December 2014 (primary text not retrieved); the "
    "reference point -0.70 and -1.8 through Moldenhauer and Pitera, arXiv 1709.01337, section 6"
)
DU_ESCANCIANO_2017 = "Du and Escanciano (2017), Management Science 63(4):940-958, doi:10.1287/mnsc.2015.2342"
MCNEIL_FREY_2000 = "McNeil and Frey (2000), J. Empirical Finance 7(3-4):271-300, doi:10.1016/S0927-5398(00)00012-8"
ENGLE_MANGANELLI_2004 = (
    "Engle and Manganelli (2004), J. Business and Economic Statistics 22(4):367-381, doi:10.1198/073500104000000370"
)
PLA_REFERENCE = "BCBS MAR32.36 to MAR32.42 and MAR32.20 (dossier 01 D.4)"
DESK_REFERENCE = "BCBS MAR32.18 to MAR32.19 (dossier 01 D.4)"


def _not_evaluated(test_id: str, h0: str, alternative: str, reference: str, version: str, reason: str, *,
                   model_id: str | None = None, segment: str | None = None, n: int | None = None,
                   n_events: int | None = None, alpha_amber: float | None = None, alpha_red: float | None = None,
                   extras: Mapping[str, Any] | None = None, inputs: str = "") -> TestResult:
    """No number is carried: the statistic is NaN (null once serialised), the metric and the p-value are None."""
    return TestResult(
        test_id=test_id, h0=h0, alternative=alternative, reference=reference, policy_version=version,
        light=Light.NOT_EVALUATED, statistic=NAN, notes=f"not evaluated: {reason}", model_id=model_id,
        segment=segment, n=n, n_events=n_events, alpha_amber=alpha_amber, alpha_red=alpha_red,
        extras=dict(extras or {}), inputs_hash=inputs,
    )


def _check_alphas(alpha_amber: float, alpha_red: float) -> None:
    if not (0.0 < alpha_red <= alpha_amber < 1.0):
        raise ValueError(f"alpha_red and alpha_amber must satisfy 0 < alpha_red <= alpha_amber < 1, got "
                         f"{alpha_red} and {alpha_amber}")


def _p_light(p: float, alpha_amber: float, alpha_red: float) -> Light:
    if p >= alpha_amber:
        return Light.GREEN
    if p >= alpha_red:
        return Light.AMBER
    return Light.RED


def _level(alpha: float, upper: float = 1.0) -> float:
    a = float(alpha)
    if not (0.0 < a < upper):
        raise ValueError(f"alpha must lie in (0, {upper}), got {alpha}")
    return a


def _segment(alpha: float) -> str:
    return f"alpha={float(alpha):g}"


def _flags(x: Any, name: str) -> np.ndarray:
    """A one-dimensional boolean array from flags (bool, or 0 and 1); a NaN or another value raises."""
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.dtype == bool:
        return a.copy()
    f = a.astype(float)
    if not np.all(np.isfinite(f)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all((f == 0.0) | (f == 1.0)):
        raise ValueError(f"{name} must contain only 0 and 1 (or booleans)")
    return f.astype(bool)


def _xlogy(x: float, y: float) -> float:
    return float(special.xlogy(x, y))


# ----------------------------------------------------------------------------------------------------------------
# coverage tests of the exceedances


def _pof_lr(t: int, x: int, p: float) -> float:
    pi = x / t
    lr = -2.0 * ((t - x) * math.log1p(-p) + _xlogy(x, p) - _xlogy(t - x, 1.0 - pi) - _xlogy(x, pi))
    return max(0.0, lr)


@register(
    "market.kupiec_pof",
    family="market",
    name="Kupiec proportion of failures",
    h0="the exceedance probability equals alpha",
    reference=KUPIEC_1995,
    decision="p_value",
)
def kupiec_pof(exceedances: np.ndarray, alpha: float, model_id: str | None = None, alpha_amber: float = 0.10,
               alpha_red: float = 0.05) -> TestResult:
    """Kupiec proportion-of-failures test (dossier 05 B.1).

    statistic = `LR_POF = -2 ln[(1-p)^(T-x) p^x / ((1-pi)^(T-x) pi^x)]`, `pi = x/T`, `p = alpha` (0 ln 0 = 0),
    asymptotically chi-square(1); p_value the asymptotic p; metric the exact binomial p-value (decision D-1: the total
    probability of the outcomes at most as likely as `x` under `Bin(T, alpha)`). Light (a Contraste internal gate): by
    the asymptotic p_value against `alpha_amber` and `alpha_red`. With `T = 250` and `alpha = 0.01` the expected count
    is 2.5 and the power is low (dossier 05 B.1), which `notes` states; the asymptotic size is then not the nominal
    one (docs).
    """
    test_id = "market.kupiec_pof"
    h0 = "the exceedance probability equals alpha"
    alt = "the exceedance probability differs from alpha"
    _check_alphas(alpha_amber, alpha_red)
    p = _level(alpha)
    ex = _flags(exceedances, "exceedances")
    t, x = int(ex.size), int(ex.sum())
    version = POLICY_LABEL
    kw = dict(model_id=model_id, segment=_segment(alpha), n=t, alpha_amber=alpha_amber, alpha_red=alpha_red)
    if t == 0:
        return _not_evaluated(test_id, h0, alt, KUPIEC_1995, version, "no observation", **kw)
    lr = _pof_lr(t, x, p)
    pv = float(stats.chi2.sf(lr, 1))
    pmf = stats.binom.pmf(np.arange(t + 1), t, p)
    exact = float(min(1.0, pmf[pmf <= pmf[x] * (1.0 + 1e-7)].sum()))
    notes = ""
    if p * t < 10.0:
        notes = (f"expected count {p * t:g} is small: the power is low (dossier 05 B.1) and the asymptotic size is not "
                 "the nominal one; read the exact binomial p-value in the metric")
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=KUPIEC_1995, policy_version=version,
        light=_p_light(pv, alpha_amber, alpha_red), statistic=lr, p_value=pv, metric=exact, n_events=x,
        inputs_hash=inputs_hash(ex, p), notes=notes,
        extras={"p_value_exact_basis": "binomial-minlike", "expected_events": p * t}, **kw)


def _transitions(ex: np.ndarray) -> tuple[int, int, int, int]:
    a, b = ex[:-1], ex[1:]
    return (int(np.sum(~a & ~b)), int(np.sum(~a & b)), int(np.sum(a & ~b)), int(np.sum(a & b)))


def _ind_lr(tr: tuple[int, int, int, int]) -> float:
    n00, n01, n10, n11 = tr
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    null = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    alt = _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    return max(0.0, -2.0 * (null - alt))


@register(
    "market.christoffersen_independence",
    family="market",
    name="Christoffersen independence",
    h0="the exceedances are independent over time",
    reference=CHRISTOFFERSEN_1998,
    decision="p_value",
)
def christoffersen_independence(exceedances: np.ndarray, alpha: float, model_id: str | None = None,
                                alpha_amber: float = 0.10, alpha_red: float = 0.05) -> TestResult:
    """Christoffersen independence test (dossier 05 B.2): statistic `LR_ind` (chi-square(1), `0^0 = 1`),
    extras["transitions"] = [n00, n01, n10, n11]. not_evaluated when there is no exceedance or all days are
    exceedances (decision D-7:
    nothing is left to test for clustering) and with fewer than 2 days. Light: by the p_value."""
    test_id = "market.christoffersen_independence"
    h0 = "the exceedances are independent over time"
    alt = "an exceedance is more (or less) likely after an exceedance"
    _check_alphas(alpha_amber, alpha_red)
    _level(alpha)
    ex = _flags(exceedances, "exceedances")
    t, x = int(ex.size), int(ex.sum())
    version = POLICY_LABEL
    kw = dict(model_id=model_id, segment=_segment(alpha), n=t, n_events=x, alpha_amber=alpha_amber,
              alpha_red=alpha_red, inputs=inputs_hash(ex))
    if t < 2:
        return _not_evaluated(test_id, h0, alt, CHRISTOFFERSEN_1998, version, "fewer than 2 days", **kw)
    if x == 0 or x == t:
        return _not_evaluated(test_id, h0, alt, CHRISTOFFERSEN_1998, version,
                              "no exceedance" if x == 0 else "all days are exceedances (nothing to cluster)",
                              **kw)
    tr = _transitions(ex)
    lr = _ind_lr(tr)
    pv = float(stats.chi2.sf(lr, 1))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=CHRISTOFFERSEN_1998, policy_version=version,
        light=_p_light(pv, alpha_amber, alpha_red), statistic=lr, p_value=pv, model_id=model_id,
        segment=_segment(alpha), n=t, n_events=x, alpha_amber=alpha_amber, alpha_red=alpha_red,
        inputs_hash=inputs_hash(ex), extras={"transitions": list(tr)})


@register(
    "market.christoffersen_conditional_coverage",
    family="market",
    name="Christoffersen conditional coverage",
    h0="the exceedance probability equals alpha and the exceedances are independent",
    reference=CHRISTOFFERSEN_1998,
    decision="p_value",
)
def christoffersen_conditional_coverage(exceedances: np.ndarray, alpha: float, model_id: str | None = None,
                                        alpha_amber: float = 0.10, alpha_red: float = 0.05) -> TestResult:
    """Christoffersen conditional coverage (dossier 05 B.2): `LR_cc = LR_POF + LR_ind`, chi-square(2). not_evaluated
    exactly when `christoffersen_independence` is. Light: by the p_value."""
    test_id = "market.christoffersen_conditional_coverage"
    h0 = "the exceedance probability equals alpha and the exceedances are independent"
    alt = "the coverage is wrong or the exceedances cluster"
    _check_alphas(alpha_amber, alpha_red)
    p = _level(alpha)
    ex = _flags(exceedances, "exceedances")
    ind = christoffersen_independence(ex, alpha, model_id, alpha_amber, alpha_red)
    version = POLICY_LABEL
    t, x = int(ex.size), int(ex.sum())
    kw = dict(model_id=model_id, segment=_segment(alpha), n=t, n_events=x, alpha_amber=alpha_amber,
              alpha_red=alpha_red)
    if ind.light is Light.NOT_EVALUATED:
        return _not_evaluated(test_id, h0, alt, CHRISTOFFERSEN_1998, version, ind.notes.removeprefix("not evaluated: "),
                              inputs=inputs_hash(ex), **kw)
    lr = _pof_lr(t, x, p) + float(ind.statistic)
    pv = float(stats.chi2.sf(lr, 2))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=CHRISTOFFERSEN_1998, policy_version=version,
        light=_p_light(pv, alpha_amber, alpha_red), statistic=lr, p_value=pv, inputs_hash=inputs_hash(ex),
        extras={"lr_pof": _pof_lr(t, x, p), "lr_ind": float(ind.statistic)}, **kw)


# ----------------------------------------------------------------------------------------------------------------
# duration-based test


_B_GRID = np.exp(np.linspace(math.log(0.05), math.log(20.0), 41))


def _spells(ex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Completed no-hit spells between consecutive exceedances and the censored first and last spells."""
    idx = np.flatnonzero(ex)
    completed = np.diff(idx).astype(float)
    # the first spell is censored only when the sample does not start with an exceedance (review m2); a first-day
    # exceedance leaves no first spell
    cens = [float(idx[0] + 1)] if idx[0] > 0 else []
    if idx[-1] < ex.size - 1:
        cens.append(float(ex.size - 1 - idx[-1]))
    return completed, np.array(cens)


def _weibull_lr(completed: np.ndarray, censored: np.ndarray) -> tuple[float, float]:
    """The likelihood-ratio statistic of `b = 1` against the Weibull hazard `lambda(d) = a^b b d^(b-1)` on the spells,
    with the first and last spells censored (survival `exp(-(a d)^b)`), and the maximising `b`. `a^b = d_c / sum(D^b)`
    is profiled out in closed form; `b` is found on a log grid and polished by a bounded search."""
    nd = completed.size
    alld = np.concatenate([completed, censored])
    slog = float(np.sum(np.log(completed)))

    def profile(b: float) -> float:
        return nd * math.log(nd / float(np.sum(alld**b))) + nd * math.log(b) + (b - 1.0) * slog - nd

    values = np.array([profile(float(b)) for b in _B_GRID])
    i = int(np.argmax(values))
    lo, hi = _B_GRID[max(i - 1, 0)], _B_GRID[min(i + 1, _B_GRID.size - 1)]
    res = optimize.minimize_scalar(lambda lb: -profile(math.exp(lb)), bounds=(math.log(lo), math.log(hi)),
                                   method="bounded", options={"xatol": 1e-8})
    b_hat = math.exp(res.x)
    best = max(float(-res.fun), float(values[i]), profile(1.0))
    return max(0.0, 2.0 * (best - profile(1.0))), b_hat


@register(
    "market.duration_test",
    family="market",
    name="Christoffersen-Pelletier duration test",
    h0="the no-hit durations are memoryless (Weibull b = 1)",
    reference=CHRISTOFFERSEN_PELLETIER_2004,
    decision="p_value",
)
def duration_test(exceedances: np.ndarray, alpha: float, model_id: str | None = None, n_mc: int = 10000, seed: int = 0,
                  alpha_amber: float = 0.10, alpha_red: float = 0.05) -> TestResult:
    """Duration-based test of Christoffersen and Pelletier (2004) (dossier 05 B.3).

    The no-hit durations are modelled with the Weibull hazard `lambda(d) = a^b b d^(b-1)`; under correct conditional
    coverage they are memoryless (`b = 1`). statistic = the likelihood-ratio statistic of `b = 1` (asymptotically
    chi-square(1)), the first and last spells censored (survival terms); p_value by Monte Carlo (Dufour 2006) with
    `n_mc` draws of Bernoulli(alpha)
    sequences of the same length, `(1 + #{S_sim >= S}) / (n_mc + 1)`; a simulated sequence with fewer than 2
    exceedances has statistic 0. Same inputs and seed, same p_value. not_evaluated with fewer than 2 exceedances (no
    completed spell).
    The Weibull form is the standard one, not re-checked against the paper's full text (dossier 05 B.3, UNVERIFIED).
    """
    test_id = "market.duration_test"
    h0 = "the no-hit durations are memoryless (Weibull b = 1)"
    alt = "the durations have a Weibull hazard with b != 1 (clustering or too regular exceedances)"
    _check_alphas(alpha_amber, alpha_red)
    p = _level(alpha)
    if int(n_mc) != n_mc or n_mc < 1:
        raise ValueError(f"n_mc must be a positive integer, got {n_mc}")
    ex = _flags(exceedances, "exceedances")
    t, x = int(ex.size), int(ex.sum())
    version = POLICY_LABEL
    kw = dict(model_id=model_id, segment=_segment(alpha), n=t, n_events=x, alpha_amber=alpha_amber,
              alpha_red=alpha_red)
    if x < 2:
        return _not_evaluated(test_id, h0, alt, CHRISTOFFERSEN_PELLETIER_2004, version,
                              "fewer than 2 exceedances: no completed no-hit duration", inputs=inputs_hash(ex), **kw)
    stat, b_hat = _weibull_lr(*_spells(ex))
    # review m3: for perfectly regular spells the MLE of b is unbounded
    at_bound = bool(b_hat >= 19.99 or b_hat <= 0.0501)
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    chunk = max(1, min(int(n_mc), 2_000_000 // max(t, 1)))
    while done < n_mc:
        m = min(chunk, int(n_mc) - done)
        draws = rng.random((m, t)) < p
        for row in draws:
            if int(row.sum()) < 2:
                continue
            s, _ = _weibull_lr(*_spells(row))
            hits += s >= stat - 1e-12
        done += m
    pv = (1.0 + hits) / (n_mc + 1.0)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=CHRISTOFFERSEN_PELLETIER_2004, policy_version=version,
        light=_p_light(pv, alpha_amber, alpha_red), statistic=stat, p_value=pv, inputs_hash=inputs_hash(ex, p),
        notes=("the Weibull shape reached the bound of its search [0.05, 20]: the statistic is capped"
               if at_bound else ""),
        extras={"weibull_b": b_hat, "weibull_b_at_bound": at_bound, "n_mc": int(n_mc), "seed": int(seed),
                "n_completed_spells": x - 1}, **kw)


# ----------------------------------------------------------------------------------------------------------------
# Basel traffic light


@register(
    "market.basel_traffic_light",
    family="market",
    name="Basel traffic light",
    h0="the model's exception count is consistent with its coverage (green zone)",
    reference=BASEL_1996,
    decision="metric",
)
def basel_traffic_light(exceedances: np.ndarray, model_id: str | None = None, coverage: float = 0.99,
                        window: int = 250) -> TestResult:
    """Basel traffic light (MAR32.9 Table 1, dossier 01 D.4 and dossier 05 B.4) on the last `window` days.

    statistic = the number of exceptions `k`, metric = the cumulative probability `P(X <= k)`, `X ~ Bin(window, 1 -
    coverage)`, in percent, p_value null; the light is the zone (amber where the cumulative probability first reaches
    95 percent, red at 99.99 percent); extras carry the MAR32.9 multiplier and the zone. not_evaluated with fewer than
    `window` days.
    """
    test_id = "market.basel_traffic_light"
    h0 = "the model's exception count is consistent with its coverage (green zone)"
    alt = "the exception count is in the amber or red zone"
    ex = _flags(exceedances, "exceedances")
    p = round(1.0 - float(coverage), 12)
    version = POLICY_LABEL
    kw = dict(model_id=model_id, segment=_segment(p), n=int(window))
    if ex.size < window:
        return _not_evaluated(test_id, h0, alt, BASEL_1996, version, f"fewer than {window} days", n_events=None,
                              inputs=inputs_hash(ex), **kw)
    k = int(ex[-window:].sum())
    zone = frtb.traffic_light_zone(k, int(window), float(coverage))
    pct = 100.0 * frtb.binomial_cumulative_probability(k, int(window), p)
    mult = frtb.mar32_multiplier(k)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=BASEL_1996, policy_version=version, light=Light(zone),
        statistic=float(k), metric=pct, n_events=k, inputs_hash=inputs_hash(ex[-window:], coverage),
        extras={"mar32_multiplier": mult, "zone": zone, "coverage": float(coverage), "window": int(window)}, **kw)


# ----------------------------------------------------------------------------------------------------------------
# ES back-tests


def _z_statistic(loss: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float, statistic: str) -> float:
    """Z1 or Z2 of Acerbi and Szekely (dossier 05 B.5): `X_t = -loss_t`, `I_t = 1{loss_t > VaR_t}`.
    `Z1 = (1/N_T) sum X_t I_t / ES_t + 1`, `Z2 = sum X_t I_t / (T alpha ES_t) + 1`; Z1 is NaN when `N_T = 0`."""
    ind = loss > var
    contrib = float(np.sum(-loss[ind] / es[ind]))
    if statistic == "Z1":
        n_t = int(ind.sum())
        return contrib / n_t + 1.0 if n_t else NAN
    return contrib / (loss.size * alpha) + 1.0


@register(
    "market.acerbi_szekely",
    family="market",
    name="Acerbi-Szekely ES back-test",
    h0="the ES forecasts are correct (the statistic has the simulated null distribution)",
    reference=ACERBI_SZEKELY_2014,
    decision="p_value",
)
def acerbi_szekely(loss: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float,
                   simulate_null: Callable[[np.random.Generator], np.ndarray], model_id: str | None = None,
                   statistic: str = "Z2", n_sim: int = 10000, seed: int = 0, alpha_amber: float = 0.05,
                   alpha_red: float = 0.0001) -> TestResult:
    """Acerbi and Szekely ES back-test (dossier 05 B.5) with simulated critical values.

    `loss`, `var`, `es` are positive loss numbers (T,); `simulate_null(rng)` returns one loss path (T,) from the
    model's own predictive distributions. statistic = the realised Z1 or Z2; p_value = the left-tail share of the
    `n_sim` simulated statistics with the +1 correction, `(1 + #{Z_sim <= Z}) / (n_sim + 1)` (Z1 draws with no
    exceedance are redrawn); extras carry the simulated 5 percent and 0.01 percent critical values and the published
    reference point {yellow: -0.70, red: -1.8} (T = 250, alpha = 2.5 percent, verified only through a secondary
    source), never a regulatory threshold. Light: p_value against `alpha_amber` (5 percent) and `alpha_red` (0.01
    percent, which needs at least 10,000
    draws to resolve). not_evaluated when an ES is NaN.
    """
    test_id = "market.acerbi_szekely"
    h0 = "the ES forecasts are correct (the statistic has the simulated null distribution)"
    alt = "risk is underestimated (the statistic is negative)"
    _check_alphas(alpha_amber, alpha_red)
    a = _level(alpha, 0.5)
    if statistic not in ("Z1", "Z2"):
        raise ValueError(f"statistic must be 'Z1' or 'Z2', got {statistic!r}")
    if int(n_sim) != n_sim or n_sim < 1:
        raise ValueError(f"n_sim must be a positive integer, got {n_sim}")
    lo, va = vec(loss, "loss"), vec(var, "var")
    es_a = np.array(es, dtype=float)
    if es_a.shape != lo.shape or va.shape != lo.shape:
        raise ValueError("loss, var and es must have the same length")
    t = int(lo.size)
    version = POLICY_LABEL
    kw = dict(model_id=model_id, segment=_segment(alpha), n=t, alpha_amber=alpha_amber, alpha_red=alpha_red)
    ref = {"yellow": -0.70, "red": -1.8}
    extras0 = {"statistic_name": statistic, "reference": ref, "n_draws": int(n_sim), "seed": int(seed)}
    if np.any(np.isnan(es_a)):
        return _not_evaluated(test_id, h0, alt, ACERBI_SZEKELY_2014, version,
                              "an ES forecast is undefined (NaN) on at least one day", extras=extras0,
                              inputs=inputs_hash(lo, va), **kw)
    if not np.all(np.isfinite(es_a)) or np.any(es_a <= 0.0):
        raise ValueError("es must be positive loss numbers")
    n_t = int((lo > va).sum())
    z_obs = _z_statistic(lo, va, es_a, a, statistic)
    if not math.isfinite(z_obs):
        return _not_evaluated(test_id, h0, alt, ACERBI_SZEKELY_2014, version,
                              "Z1 is undefined without an exceedance", extras=extras0, n_events=0,
                              inputs=inputs_hash(lo, va), **kw)
    rng = np.random.default_rng(seed)
    sims = np.empty(int(n_sim))
    attempts, i = 0, 0
    while i < n_sim:
        attempts += 1
        if attempts > 20 * int(n_sim) + 100:
            raise ValueError("too many simulated paths without an exceedance for Z1: the model rarely exceeds its VaR")
        path = np.asarray(simulate_null(rng), dtype=float)
        if path.shape != lo.shape:
            raise ValueError(f"simulate_null must return a path of shape {lo.shape}, got {path.shape}")
        z = _z_statistic(path, va, es_a, a, statistic)
        if math.isfinite(z):
            sims[i] = z
            i += 1
    scale = max(1.0, abs(z_obs))
    pv = (1.0 + float(np.sum(sims <= z_obs + 1e-12 * scale))) / (n_sim + 1.0)
    extras = dict(extras0, critical_5pct=float(np.quantile(sims, 0.05)),
                  critical_0_01pct=float(np.quantile(sims, 0.0001)))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=ACERBI_SZEKELY_2014, policy_version=version,
        light=_p_light(pv, alpha_amber, alpha_red), statistic=float(z_obs), p_value=float(pv), n_events=n_t,
        inputs_hash=inputs_hash(lo, va, es_a, a), extras=extras,
        notes="the critical values are simulated per model; -0.70 and -1.8 are a published reference point, "
              "not a threshold",
        **kw)


def _pit(pit: Any) -> np.ndarray | None:
    """The PIT array, or None when every value is NaN (no predictive distribution); a partial NaN or a value outside
    [0, 1] raises."""
    u = np.array(pit, dtype=float)
    if u.ndim != 1:
        raise ValueError(f"pit must be one-dimensional, got shape {u.shape}")
    if u.size and np.all(np.isnan(u)):
        return None
    if not np.all(np.isfinite(u)):
        raise ValueError("pit contains non-finite values")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("pit values must lie in [0, 1]")
    return u


def _h(u: np.ndarray, alpha: float) -> np.ndarray:
    """The cumulative violation process `H_t = (1/alpha)(alpha - u_t) 1{u_t <= alpha}` (Du and Escanciano 2017)."""
    return np.where(u <= alpha, (alpha - u) / alpha, 0.0)


@register(
    "market.du_escanciano_unconditional",
    family="market",
    name="Du-Escanciano unconditional ES back-test",
    h0="the PITs are uniform in the tail (the ES forecasts are correct): E[H] = alpha/2",
    reference=DU_ESCANCIANO_2017,
    decision="p_value",
)
def du_escanciano_unconditional(pit: np.ndarray, alpha: float, model_id: str | None = None, alpha_amber: float = 0.10,
                                alpha_red: float = 0.05) -> TestResult:
    """Du and Escanciano unconditional ES test (dossier 05 B.5): `E[H] = alpha/2`, `Var(H) = alpha (1/3 - alpha/4)`,
    statistic `U_ES = sqrt(T)(mean(H) - alpha/2) / sqrt(alpha (1/3 - alpha/4))`, asymptotically N(0, 1); p_value
    one-sided upper tail (decision D-5: a large positive U_ES means underestimated risk). The plain asymptotic version
    ignores parameter estimation uncertainty (stated in `notes`). not_evaluated when the model has no predictive
    distribution (every PIT NaN) or `T < 2`."""
    test_id = "market.du_escanciano_unconditional"
    h0 = "the PITs are uniform in the tail (the ES forecasts are correct): E[H] = alpha/2"
    alt = "E[H] > alpha/2: the tail risk is underestimated"
    _check_alphas(alpha_amber, alpha_red)
    a = _level(alpha)
    u = _pit(pit)
    version = POLICY_LABEL
    n = int(np.size(pit))
    kw = dict(model_id=model_id, segment=_segment(alpha), n=n, alpha_amber=alpha_amber, alpha_red=alpha_red)
    if u is None:
        return _not_evaluated(test_id, h0, alt, DU_ESCANCIANO_2017, version,
                              "the model gives no predictive distribution", **kw)
    if u.size < 2:
        return _not_evaluated(test_id, h0, alt, DU_ESCANCIANO_2017, version, "fewer than 2 observations",
                              inputs=inputs_hash(u), **kw)
    h = _h(u, a)
    stat = math.sqrt(u.size) * (float(h.mean()) - a / 2.0) / math.sqrt(a * (1.0 / 3.0 - a / 4.0))
    pv = float(stats.norm.sf(stat))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=DU_ESCANCIANO_2017, policy_version=version,
        light=_p_light(pv, alpha_amber, alpha_red), statistic=float(stat), p_value=pv, inputs_hash=inputs_hash(u, a),
        notes="asymptotic version: parameter estimation uncertainty is ignored (dossier 05 B.5)", **kw)


@register(
    "market.du_escanciano_conditional",
    family="market",
    name="Du-Escanciano conditional ES back-test",
    h0="the cumulative violations H_t have no autocorrelation up to lag m",
    reference=DU_ESCANCIANO_2017,
    decision="p_value",
)
def du_escanciano_conditional(pit: np.ndarray, alpha: float, lags: int = 5, model_id: str | None = None,
                              alpha_amber: float = 0.10, alpha_red: float = 0.05) -> TestResult:
    """Du and Escanciano conditional ES test (dossier 05 B.5): the Box-Pierce statistic `C_ES(m) = T sum_{j=1}^m
    rho_j^2` of the first `m = lags` autocorrelations of `H_t - alpha/2` (sample autocorrelations), asymptotically
    chi-square(m).
    not_evaluated as the unconditional test, with `T <= lags`, or when `H` has no spread."""
    test_id = "market.du_escanciano_conditional"
    h0 = "the cumulative violations H_t have no autocorrelation up to lag m"
    alt = "the cumulative violations are autocorrelated (clustered tail misses)"
    _check_alphas(alpha_amber, alpha_red)
    a = _level(alpha)
    if int(lags) != lags or lags < 1:
        raise ValueError(f"lags must be a positive integer, got {lags}")
    u = _pit(pit)
    version = POLICY_LABEL
    n = int(np.size(pit))
    kw = dict(model_id=model_id, segment=_segment(alpha), n=n, alpha_amber=alpha_amber, alpha_red=alpha_red,
              extras={"lags": int(lags)})
    if u is None:
        return _not_evaluated(test_id, h0, alt, DU_ESCANCIANO_2017, version,
                              "the model gives no predictive distribution", **kw)
    if u.size <= lags:
        return _not_evaluated(test_id, h0, alt, DU_ESCANCIANO_2017, version, f"T <= lags ({lags})",
                              inputs=inputs_hash(u), **kw)
    c = _h(u, a) - _h(u, a).mean()
    denom = float(c @ c)
    if denom <= 1e-18:
        return _not_evaluated(test_id, h0, alt, DU_ESCANCIANO_2017, version,
                              "H has zero spread (no tail violation structure to test)", inputs=inputs_hash(u), **kw)
    rho = np.array([float(c[j:] @ c[:-j]) / denom for j in range(1, int(lags) + 1)])
    stat = float(u.size * np.sum(rho**2))
    pv = float(stats.chi2.sf(stat, int(lags)))
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=DU_ESCANCIANO_2017, policy_version=version,
        light=_p_light(pv, alpha_amber, alpha_red), statistic=stat, p_value=pv, inputs_hash=inputs_hash(u, a), **kw)


@register(
    "market.mcneil_frey_residual_test",
    family="market",
    name="McNeil-Frey ES residual test",
    h0="the exceedance residuals (L - ES) / sigma have mean zero",
    reference=MCNEIL_FREY_2000,
    decision="p_value",
)
def mcneil_frey_residual_test(loss: np.ndarray, var: np.ndarray, es: np.ndarray, sigma: np.ndarray, alpha: float,
                              model_id: str | None = None, n_boot: int = 10000, seed: int = 0,
                              alpha_amber: float = 0.10, alpha_red: float = 0.05) -> TestResult:
    """McNeil and Frey exceedance-residual test (dossier 05 B.5): on exceedance days `r_t = (L_t - ES_t) / sigma_t`
    should be i.i.d. with mean zero. statistic = the mean of `r_t`, p_value = the one-sided bootstrap p-value of a
    zero mean: the residuals centred at their mean are resampled `n_boot` times and `p = (1 + #{resampled mean >=
    statistic}) / (n_boot + 1)` (a positive mean means the loss exceeds the ES). Same inputs and seed, same p_value.
    not_evaluated with fewer than 2
    exceedances or a NaN ES."""
    test_id = "market.mcneil_frey_residual_test"
    h0 = "the exceedance residuals (L - ES) / sigma have mean zero"
    alt = "the mean residual is positive: the ES is underestimated"
    _check_alphas(alpha_amber, alpha_red)
    _level(alpha)
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot}")
    lo, va, sg = vec(loss, "loss"), vec(var, "var"), vec(sigma, "sigma")
    es_a = np.array(es, dtype=float)
    if not (lo.shape == va.shape == sg.shape == es_a.shape):
        raise ValueError("loss, var, es and sigma must have the same length")
    if np.any(sg <= 0.0):
        raise ValueError("sigma must be positive")
    version = POLICY_LABEL
    exc = lo > va
    n_ex = int(exc.sum())
    kw = dict(model_id=model_id, segment=_segment(alpha), n=int(lo.size), n_events=n_ex, alpha_amber=alpha_amber,
              alpha_red=alpha_red, inputs=inputs_hash(lo, va, sg))
    if np.any(np.isnan(es_a)):
        return _not_evaluated(test_id, h0, alt, MCNEIL_FREY_2000, version, "an ES forecast is undefined (NaN)", **kw)
    if n_ex < 2:
        return _not_evaluated(test_id, h0, alt, MCNEIL_FREY_2000, version,
                              "fewer than 2 exceedances: a bootstrap needs spread", **kw)
    r = (lo[exc] - es_a[exc]) / sg[exc]
    stat = float(r.mean())
    centred = r - stat
    rng = np.random.default_rng(seed)
    ge = 0
    done = 0
    chunk = max(1, min(int(n_boot), 4_000_000 // n_ex))
    while done < n_boot:
        m = min(chunk, int(n_boot) - done)
        means = centred[rng.integers(0, n_ex, size=(m, n_ex))].mean(axis=1)
        ge += int(np.sum(means >= stat - 1e-12 * max(1.0, abs(stat))))
        done += m
    pv = (1.0 + ge) / (n_boot + 1.0)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=MCNEIL_FREY_2000, policy_version=version,
        light=_p_light(pv, alpha_amber, alpha_red), statistic=stat, p_value=pv,
        model_id=model_id, segment=_segment(alpha), n=int(lo.size), n_events=n_ex, alpha_amber=alpha_amber,
        alpha_red=alpha_red, inputs_hash=inputs_hash(lo, va, es_a, sg),
        extras={"n_boot": int(n_boot), "seed": int(seed)})


# ----------------------------------------------------------------------------------------------------------------
# dynamic quantile, PLA and the desk back-test


@register(
    "market.dynamic_quantile_test",
    family="market",
    name="Engle-Manganelli dynamic quantile test",
    h0="the hits are unpredictable from their past and from the VaR (conditional coverage holds)",
    reference=ENGLE_MANGANELLI_2004,
    decision="p_value",
)
def dynamic_quantile_test(hits: np.ndarray, var: np.ndarray, alpha: float, lags: int = 4, model_id: str | None = None,
                          n_mc: int = 10000, seed: int = 0, alpha_amber: float = 0.10,
                          alpha_red: float = 0.05) -> TestResult:
    """The dynamic quantile (DQ) test of Engle and Manganelli (2004): the statistic is that of
    `caviar.dynamic_quantile_statistic` (the regression of `hits - alpha` on a constant, `lags` lagged hits and the
    contemporaneous VaR).

    The asymptotic chi-square law is poor with sparse hits (review M1: the plain test rejected correct forecasts 9.9
    percent of the time at alpha 0.01 and T 250, 14.3 percent at T 750), so the **light follows a Monte Carlo p-value**
    (Dufour form): `n_mc` Bernoulli(alpha) hit sequences of the same length with the same VaR column, `p = (1 +
    #{S_sim >= S}) / (n_mc + 1)`, reproducible for the same `seed`; the asymptotic p-value stays in
    # `extras["p_value_asymptotic"]`.
    not_evaluated when `T <= lags + 2`."""
    test_id = "market.dynamic_quantile_test"
    h0 = "the hits are unpredictable from their past and from the VaR (conditional coverage holds)"
    alt = "the hits are predictable: the quantile forecasts are conditionally biased"
    _check_alphas(alpha_amber, alpha_red)
    a = _level(alpha, 0.5)
    hit = _flags(hits, "hits")
    v = vec(var, "var", allow_empty=True)
    if hit.shape != v.shape:
        raise ValueError(f"hits and var must have the same length, got {hit.size} and {v.size}")
    if int(lags) != lags or lags < 0:
        raise ValueError(f"lags must be a non-negative integer, got {lags}")
    if int(n_mc) != n_mc or n_mc < 1:
        raise ValueError(f"n_mc must be a positive integer, got {n_mc}")
    version = POLICY_LABEL
    t, x = int(hit.size), int(hit.sum())
    kw = dict(model_id=model_id, segment=_segment(alpha), n=t, n_events=x, alpha_amber=alpha_amber,
              alpha_red=alpha_red, extras={"lags": int(lags)})
    if t <= lags + 2:
        return _not_evaluated(test_id, h0, alt, ENGLE_MANGANELLI_2004, version, f"T <= lags + 2 ({lags + 2})",
                              inputs=inputs_hash(hit, v), **kw)
    stat, pv_asym = dynamic_quantile_statistic(hit.astype(float), v, a, lags=int(lags))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_mc)):
        sim = (rng.random(t) < a).astype(float)
        s, _rank = dq_statistic_unchecked(sim, v, a, int(lags))
        ge += s >= stat - 1e-12 * max(1.0, abs(stat))
    pv = (1.0 + ge) / (n_mc + 1.0)
    extras = {"lags": int(lags), "p_value_basis": "monte-carlo", "p_value_asymptotic": pv_asym, "n_mc": int(n_mc),
              "seed": int(seed)}
    kw = dict(kw, extras=extras)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=ENGLE_MANGANELLI_2004, policy_version=version,
        light=_p_light(pv, alpha_amber, alpha_red), statistic=stat, p_value=pv,
        inputs_hash=inputs_hash(hit, v, a), **kw)


@register(
    "market.pla_test",
    family="market",
    name="Profit and loss attribution test",
    h0="the risk-theoretical P&L is a good attribution of the hypothetical P&L",
    reference=PLA_REFERENCE,
    decision="metric",
)
def pla_test(hpl: np.ndarray, rtpl: np.ndarray, model_id: str | None = None) -> TestResult:
    """The profit-and-loss attribution test (MAR32.36 to MAR32.42, dossier 01 D.4): statistic = the Spearman
    correlation of `hpl` and `rtpl`, metric = the KS statistic (an exact step count over 250), p_value null; the light
    is the zone of `frtb.pla_zone` (red: the desk is ineligible for the IMA; amber: capital surcharge, MAR32.43 and
    MAR32.44).
    extras["thresholds"] carries the boundaries. not_evaluated unless both series have exactly 250 finite observations
    and a nonzero spread. `alpha_amber` and `alpha_red` are null."""
    test_id = "market.pla_test"
    h0 = "the risk-theoretical P&L is a good attribution of the hypothetical P&L"
    alt = "the attribution is poor (Spearman low or KS high)"
    version = POLICY_LABEL
    thresholds = {"spearman_green": 0.80, "spearman_red": 0.70, "ks_green": 0.09, "ks_red": 0.12}
    h = np.asarray(hpl, dtype=float)
    r = np.asarray(rtpl, dtype=float)
    kw = dict(model_id=model_id, extras={"thresholds": thresholds})
    if h.ndim != 1 or r.ndim != 1:
        raise ValueError("hpl and rtpl must be one-dimensional")
    n = int(min(h.size, r.size))
    if h.size != frtb.PLA_N or r.size != frtb.PLA_N:
        return _not_evaluated(test_id, h0, alt, PLA_REFERENCE, version,
                              f"both series need exactly {frtb.PLA_N} observations (MAR32.20)", n=n, **kw)
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(r))):
        return _not_evaluated(test_id, h0, alt, PLA_REFERENCE, version, "a series has a non-finite observation",
                              n=n, inputs=inputs_hash(h, r), **kw)
    if np.ptp(h) == 0.0 or np.ptp(r) == 0.0:
        return _not_evaluated(test_id, h0, alt, PLA_REFERENCE, version, "a series has zero spread", n=n,
                              inputs=inputs_hash(h, r), **kw)
    rho = frtb.pla_spearman(h, r)
    ks = frtb.pla_ks(h, r)
    zone = frtb.pla_zone_exact(h, r)   # integer ranks and step gap: no float quotient on a boundary (review m4)
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=PLA_REFERENCE, policy_version=version, light=Light(zone),
        statistic=rho, metric=ks, n=n, inputs_hash=inputs_hash(h, r), model_id=model_id,
        extras={"zone": zone, "thresholds": thresholds})


@register(
    "market.desk_backtest",
    family="market",
    name="Desk backtest",
    h0="the desk's exception count in the last 250 days does not exceed the limit",
    reference=DESK_REFERENCE,
    decision="metric",
)
def desk_backtest(loss: np.ndarray, var: np.ndarray, alpha: float, model_id: str | None = None,
                  basis: str = "HPL") -> TestResult:
    """The desk-level VaR back-test (MAR32.18 to MAR32.19): the exceptions of the last 250 days against the desk VaR
    at 99 or 97.5 percent (positive loss numbers; `basis` in HPL, APL). statistic = the exception count
    (`frtb.desk_exceptions`, a missing value counts as an exception), metric = the limit (`frtb.desk_exception_limit`:
    12 at 0.01, 30 at 0.025), p_value null. Light: red when the count exceeds the limit, otherwise green (the dossier
    states one limit; there is no amber). not_evaluated with fewer than 250 days."""
    test_id = "market.desk_backtest"
    h0 = "the desk's exception count in the last 250 days does not exceed the limit"
    alt = "the desk has more exceptions than the limit (it moves to the standardised approach)"
    if basis not in ("HPL", "APL"):
        raise ValueError(f"basis must be 'HPL' or 'APL', got {basis!r}")
    limit = frtb.desk_exception_limit(alpha)
    lo = np.asarray(loss, dtype=float)
    va = np.asarray(var, dtype=float)
    if lo.ndim != 1 or lo.shape != va.shape:
        raise ValueError("loss and var must be one-dimensional of the same length")
    version = POLICY_LABEL
    kw = dict(model_id=model_id, segment=_segment(alpha), n=frtb.PLA_N, extras={"basis": basis})
    if lo.size < 250:
        return _not_evaluated(test_id, h0, alt, DESK_REFERENCE, version, "fewer than 250 days",
                              inputs=inputs_hash(lo, va), **kw)
    count = frtb.desk_exceptions(lo[-250:], va[-250:])
    return TestResult(
        test_id=test_id, h0=h0, alternative=alt, reference=DESK_REFERENCE, policy_version=version,
        light=Light.RED if count > limit else Light.GREEN, statistic=float(count), metric=float(limit), n_events=count,
        inputs_hash=inputs_hash(lo[-250:], va[-250:], alpha), **kw)

__all__ = [
    "acerbi_szekely",
    "basel_traffic_light",
    "christoffersen_conditional_coverage",
    "christoffersen_independence",
    "desk_backtest",
    "du_escanciano_conditional",
    "du_escanciano_unconditional",
    "duration_test",
    "dynamic_quantile_test",
    "kupiec_pof",
    "mcneil_frey_residual_test",
    "pla_test",
]
