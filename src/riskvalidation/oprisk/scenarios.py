"""Scenario analysis and its combination with loss data (dossier 05 F.5).

Scenarios are expert estimates of the frequency and severity of rare large events. The combination methods are (i)
scenario points as pseudo-observations appended to the tail sample, (ii) the body fitted to internal data and the tail
to scenario-implied quantiles, (iii) Bayesian credibility with the scenarios as the prior for the tail parameters
(Shevchenko and Wuthrich 2006, J. Operational Risk 1(3):3-26: a Gamma prior on the Pareto shape, section 3.3).
Validation must test the sensitivity of capital to each scenario point (the scenario influence function). Contract:
``engine-oprisk-api`` 1.0.0, module scenarios.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from riskvalidation.oprisk._checks import positive, scalar, vec
from riskvalidation.oprisk.distributions import SeverityFit

KINDS = ("percentile", "individual")
PRIOR_KIND = "unrestricted-gamma-on-pareto-shape"
_NAN = math.nan


@dataclass(frozen=True)
class ScenarioPoint:
    """One scenario: ``kind`` in ``{"percentile", "individual"}`` (the template of the 2008 LDCE scenarios, dossier 05
    F.5, restricted to the two the C17 document carries); ``loss`` in EUR; ``exceedance_frequency`` the expected number
    of
    losses of at least ``loss`` per year (one in ten years is ``0.1``)."""

    scenario_id: str
    kind: str
    loss: float
    exceedance_frequency: float

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        positive(self.loss, "loss")
        positive(self.exceedance_frequency, "exceedance_frequency")


def scenario_pseudo_observations(points: Sequence[ScenarioPoint], n_years: float) -> tuple[np.ndarray, np.ndarray]:
    """Method (i): ``(losses, weights)``, each ``(P,)``: the loss of each point in EUR and its pseudo-count
    ``exceedance_frequency * n_years``."""
    n_years = positive(n_years, "n_years")
    if len(points) == 0:
        raise ValueError("no scenario points")
    losses = np.array([p.loss for p in points], dtype=float)
    weights = np.array([p.exceedance_frequency * n_years for p in points], dtype=float)
    return losses, weights


# ----------------------------------------------------------------------------------------------------------------
# method (ii): the tail fitted to scenario-implied quantiles


def _log_excess_residuals(xi: float, beta: float, cond: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``ln(q_model(p) - threshold) - ln(loss - threshold)`` with ``q_model(p) - threshold = beta / xi (cond^(-xi) -
    1)``
    and ``cond = 1 - p`` the conditional exceedance probability of each point (the objective of dossier 05 F.6)."""
    return np.log(beta / xi * np.expm1(-xi * np.log(cond))) - np.log(u)


_STARTS = (0.05, 0.2, 0.5, 1.0, 2.0)
XI_BOUNDS = (1e-3, 50.0)


def _fit_scenario_gpd(cond: np.ndarray, u: np.ndarray, starts: Sequence[float] = _STARTS,
                      warm: tuple[float, float] | None = None) -> tuple[float, float, optimize.OptimizeResult]:
    """Least squares on the log excess quantiles from several deterministic starts (and, when given, a warm start): the
    best optimum. The search is bounded in log space (``xi`` in [1e-3, 50], ``beta`` within twelve decades of the
    excesses) so that no start wanders into an overflow."""
    best: optimize.OptimizeResult | None = None
    i = int(np.argmax(u))
    lo = np.array([math.log(XI_BOUNDS[0]), math.log(float(u.min()) * 1e-6)])
    hi = np.array([math.log(XI_BOUNDS[1]), math.log(float(u.max()) * 1e6)])
    initial = [(xi0, xi0 * u[i] / math.expm1(-xi0 * math.log(cond[i]))) for xi0 in starts]
    if warm is not None:
        initial.insert(0, warm)
    for xi0, beta0 in initial:
        start = np.clip(np.array([math.log(xi0), math.log(beta0)]), lo + 1e-9, hi - 1e-9)
        res = optimize.least_squares(lambda z: _log_excess_residuals(math.exp(z[0]), math.exp(z[1]), cond, u), start,
                                     bounds=(lo, hi), method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
        if best is None or res.cost < best.cost - 1e-300:
            best = res
    assert best is not None
    return math.exp(best.x[0]), math.exp(best.x[1]), best


def _on_search_bound(xi: float, beta: float, u: np.ndarray) -> bool:
    """True when the optimum sits on the box of ``_fit_scenario_gpd``: such a value is the box, not a statistic."""
    tol = 1e-6
    lo_b, hi_b = float(u.min()) * 1e-6, float(u.max()) * 1e6
    return bool(xi <= XI_BOUNDS[0] * (1 + tol) or xi >= XI_BOUNDS[1] * (1 - tol)
                or beta <= lo_b * (1 + tol) or beta >= hi_b * (1 - tol))


def _scenario_inputs(points: Sequence[ScenarioPoint], threshold: float,
                     frequency_above_threshold: float) -> tuple[float, float, list[ScenarioPoint]]:
    h = scalar(threshold, "threshold")
    fa = positive(frequency_above_threshold, "frequency_above_threshold")
    pts = list(points)
    if len(pts) < 2:
        raise ValueError("at least two scenario points are needed")
    if len({p.loss for p in pts}) < 2:
        raise ValueError("at least two scenario points with distinct losses are needed")
    if any(p.loss <= h for p in pts):
        raise ValueError("every scenario loss must exceed the threshold")
    return h, fa, pts


def _conditional(freq: np.ndarray, fa: float) -> np.ndarray:
    cond = freq / fa
    if np.any(cond <= 0.0) or np.any(cond >= 1.0):
        raise ValueError("a scenario point implies a conditional exceedance probability outside (0, 1)")
    return cond


def fit_tail_to_scenarios(points: Sequence[ScenarioPoint], threshold: float,
                          frequency_above_threshold: float) -> SeverityFit:
    """Method (ii): a ``"gpd"`` ``SeverityFit`` with location ``threshold`` fitted by least squares on the log excess
    quantiles. Each point implies ``P(X > loss | X > threshold) = exceedance_frequency / frequency_above_threshold``,
    i.e.
    the conditional quantile level ``p = 1 - exceedance_frequency / frequency_above_threshold``; the fit minimises
    ``sum_s (ln(q_model(p_s) - threshold) - ln(loss_s - threshold))^2``, the objective with which dossier 05 F.6 fits
    the LDCE percentiles. At least two points with distinct losses, every loss above the threshold and every conditional
    probability in (0, 1). Exact when the points lie on a GPD (zero residual). The fit has no likelihood (``loglik`` is
    ``nan``); its standard errors come from the residual variance and are ``nan`` with two points; ``converged`` is the
    least squares solver's and is False when the optimum sits on the search bound of ``xi`` or ``beta`` (a bound is
    not a statistic).
    """
    h, fa, pts = _scenario_inputs(points, threshold, frequency_above_threshold)
    cond = _conditional(np.array([p.exceedance_frequency for p in pts]), fa)
    u = np.array([p.loss - h for p in pts])
    xi, beta, best = _fit_scenario_gpd(cond, u)
    n = len(pts)
    se = (_NAN, _NAN, 0.0)
    if n > 2:
        centre = np.array([xi, beta])
        eps = 1e-6 * centre

        def resid(p: np.ndarray) -> np.ndarray:
            return _log_excess_residuals(p[0], p[1], cond, u)

        jac = np.column_stack([
            (resid(centre + np.eye(2)[k] * eps[k]) - resid(centre - np.eye(2)[k] * eps[k])) / (2.0 * eps[k])
            for k in range(2)
        ])
        s2 = 2.0 * best.cost / (n - 2)
        try:
            diag = np.diag(s2 * np.linalg.inv(jac.T @ jac))
            if np.all(diag >= 0.0):
                se = (math.sqrt(float(diag[0])), math.sqrt(float(diag[1])), 0.0)
        except np.linalg.LinAlgError:
            pass
    return SeverityFit("gpd", (xi, beta, h), se, h, True, n, _NAN,
                       bool(best.success and not _on_search_bound(xi, beta, u)))


# ----------------------------------------------------------------------------------------------------------------
# method (iii): Bayesian credibility (Shevchenko and Wuthrich 2006)


@dataclass(frozen=True)
class GammaPrior:
    """A Gamma prior on the Pareto shape ``alpha = 1 / xi`` (dimensionless; ``xi`` the tail index): ``shape`` and
    ``rate`` (``mean = shape / rate``), ``interval`` the expert interval of ``alpha`` and ``interval_level`` its
    probability mass.
    ``kind`` is always ``"unrestricted-gamma-on-pareto-shape"``, repeated on every C17 document (decision D4: Shevchenko
    and Wuthrich restrict the prior to ``alpha`` above 1, a finite mean, which would exclude the true value of the tail
    index 1.1 variant, ``alpha = 0.909``; this prior is unrestricted and says so)."""

    shape: float
    rate: float
    mean: float
    interval: tuple[float, float]
    interval_level: float
    kind: str = PRIOR_KIND

    def __post_init__(self) -> None:
        positive(self.shape, "shape")
        positive(self.rate, "rate")
        if abs(self.mean - self.shape / self.rate) > 1e-12 * max(1.0, abs(self.mean)):
            raise ValueError("mean must equal shape / rate")
        if self.kind != PRIOR_KIND:
            raise ValueError(f"kind must be {PRIOR_KIND!r}")


def gamma_prior_from_mean_interval(mean: float, lower: float, upper: float, interval_level: float = 0.9) -> GammaPrior:
    """The Gamma ``(shape, rate = shape / mean)`` whose probability of ``[lower, upper]`` is ``interval_level``
    (Shevchenko and Wuthrich 2006, section 4.3: the prior from an expert's mean and interval probability), found by
    bracketed root
    finding on the shape in ``[1e-3, 1e6]``. Needs ``0 < lower < mean < upper``."""
    mean, lower, upper = positive(mean, "mean"), positive(lower, "lower"), positive(upper, "upper")
    level = scalar(interval_level, "interval_level")
    if not 0.0 < level < 1.0:
        raise ValueError(f"interval_level must lie in (0, 1), got {level}")
    if not lower < mean < upper:
        raise ValueError("the mean must lie strictly inside the interval")

    def excess(k: float) -> float:
        scale = mean / k
        return float(stats.gamma.cdf(upper, k, scale=scale) - stats.gamma.cdf(lower, k, scale=scale)) - level

    grid = np.logspace(-3.0, 6.0, 181)
    values = np.array([excess(float(k)) for k in grid])
    flips = np.nonzero((values[:-1] < 0.0) & (values[1:] >= 0.0))[0]
    if flips.size == 0:
        raise ValueError("no Gamma shape in [1e-3, 1e6] gives the interval its probability mass")
    k = float(optimize.brentq(excess, float(grid[flips[0]]), float(grid[flips[0] + 1]), xtol=1e-14, rtol=1e-14))
    return GammaPrior(shape=k, rate=k / mean, mean=mean, interval=(lower, upper), interval_level=level)


def gamma_prior_from_scenarios(points: Sequence[ScenarioPoint], threshold: float, frequency_above_threshold: float,
                               basis_years: float, n_bootstrap: int = 1000, seed: int = 0,
                               interval_level: float = 0.9) -> GammaPrior:
    """The mapping of the C17 scenario points to the expert input of Shevchenko and Wuthrich (decision D4).

    (1) ``fit_tail_to_scenarios`` gives a GPD with tail index ``xi_s``; ``alpha_s = 1 / xi_s`` is the prior mean. (2) A
    parametric bootstrap gives the expert interval: each exceedance frequency is redrawn as ``k / basis_years`` with
    ``k`` Poisson with mean ``exceedance_frequency * basis_years`` (a count of 0 is replaced by 0.5), the GPD is
    refitted (from the point estimate as the start), and
    the 5th and 95th percentiles of ``alpha`` over ``n_bootstrap`` replications
    (``numpy.random.Generator(PCG64(seed))``) form the interval of mass ``interval_level``. (3)
    ``gamma_prior_from_mean_interval`` solves the Gamma shape and rate.
    ``basis_years`` is the number of years of experience each scenario frequency is worth (a policy parameter of the
    caller). Deterministic for the same ``seed``; the prior is never truncated to ``alpha`` above 1. ``ValueError`` when
    a point is worth less than one expected event in the basis years (``exceedance_frequency * basis_years < 1``: most
    redraws are then 0 and the refits collapse onto the search bound), when the point estimate or any bootstrap
    replicate sits on the search bound of the least squares fit (the share is named; a bound is never put into the
    percentiles), and when a replicate's redrawn frequency reaches ``frequency_above_threshold``.
    """
    h, fa, pts = _scenario_inputs(points, threshold, frequency_above_threshold)
    basis = positive(basis_years, "basis_years")
    level = scalar(interval_level, "interval_level")
    if not 0.0 < level < 1.0:
        raise ValueError(f"interval_level must lie in (0, 1), got {level}")
    if int(n_bootstrap) != n_bootstrap or n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be an integer of at least 2, got {n_bootstrap}")
    if int(seed) != seed or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    freq = np.array([p.exceedance_frequency for p in pts])
    if np.any(freq * basis < 1.0):
        raise ValueError("every scenario point must be worth at least one expected event in basis_years "
                         f"(exceedance_frequency * basis_years >= 1); the smallest is "
                         f"{float(np.min(freq * basis)):.3g}")
    base = fit_tail_to_scenarios(pts, h, fa)
    if not base.converged:
        raise ValueError("the scenario fit sits on the search bound: the points do not define a tail")
    alpha_s = 1.0 / base.params[0]
    u = np.array([p.loss - h for p in pts])
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    alphas = np.empty(int(n_bootstrap))
    on_bound = 0
    for b in range(int(n_bootstrap)):
        counts = rng.poisson(freq * basis)
        redrawn = np.where(counts == 0, 0.5, counts) / basis
        xi_b, beta_b, _ = _fit_scenario_gpd(_conditional(redrawn, fa), u, starts=(), warm=tuple(base.params[:2]))
        on_bound += _on_search_bound(xi_b, beta_b, u)
        alphas[b] = 1.0 / xi_b
    if on_bound:
        raise ValueError(f"{on_bound} of {int(n_bootstrap)} bootstrap refits sit on the search bound: the interval "
                         "would end at the bound, not at a statistic; raise basis_years or use more frequent points")
    lo, hi = (float(v) for v in np.percentile(alphas, [50.0 * (1.0 - level), 50.0 * (1.0 + level)]))
    return gamma_prior_from_mean_interval(alpha_s, lo, hi, level)


@dataclass(frozen=True)
class CredibilityFit:
    """The posterior of the Pareto shape given the tail data: ``posterior_shape`` and ``posterior_rate`` (the Gamma
    posterior), ``alpha_mean`` its mean, ``alpha_mle = n / sum ln(x_i / L)`` (None when ``n = 0``),
    ``credibility_weight`` ``w = sum ln(x_i / L) / (sum ln(x_i / L) + rate)`` (Shevchenko and Wuthrich eq. 25) with
    ``alpha_mean = w alpha_mle + (1 - w) prior.mean``; ``xi = 1 / alpha_mean``, ``beta = xi L`` and ``location = L``
    (EUR) the GPD that is the one-parameter Pareto with shape ``alpha_mean`` above ``L``; ``infinite_mean`` is
    ``alpha_mean <= 1`` (a tail index of
    1 or more); ``n`` the number of excesses."""

    prior: GammaPrior
    posterior_shape: float
    posterior_rate: float
    alpha_mean: float
    alpha_mle: float | None
    credibility_weight: float
    xi: float
    beta: float
    location: float
    infinite_mean: bool
    n: int


def credibility_pareto(prior: GammaPrior, losses: np.ndarray, threshold: float) -> CredibilityFit:
    """Bayesian credibility of the one-parameter Pareto ``f(x | alpha) = (alpha / L)(x / L)^(-alpha - 1)`` for ``x >=
    L`` with the Gamma(``shape``, ``rate``) prior on ``alpha``: the Gamma posterior has shape ``shape + N`` and rate
    ``rate + sum ln(x_i / L)`` (Shevchenko and Wuthrich 2006, section 3.3, eqs. 21 to 25), whose mean is the credibility
    weighting of the maximum likelihood estimate and the prior mean. ``N = 0`` returns the prior. The prior is
    unrestricted, so ``alpha_mean`` may be 1 or below and then ``infinite_mean`` is true and no mean-based number
    follows. Raises when a loss is below ``threshold`` or when ``sum ln(x_i / L) = 0`` with ``N > 0``.
    """
    if not isinstance(prior, GammaPrior):
        raise ValueError("prior must be a GammaPrior")
    big_l = positive(threshold, "threshold")
    x = vec(losses, "losses", allow_empty=True)
    if np.any(x < big_l):
        raise ValueError("a loss lies below the threshold")
    n = int(x.size)
    s = float(np.sum(np.log(x / big_l))) if n else 0.0
    if n and not s > 0.0:
        raise ValueError("every loss equals the threshold: sum ln(x / L) = 0, the shape cannot be updated")
    shape, rate = prior.shape + n, prior.rate + s
    alpha_mean = shape / rate
    xi = 1.0 / alpha_mean
    return CredibilityFit(prior=prior, posterior_shape=shape, posterior_rate=rate, alpha_mean=alpha_mean,
                          alpha_mle=(n / s if n else None), credibility_weight=s / (s + prior.rate), xi=xi,
                          beta=xi * big_l, location=big_l, infinite_mean=bool(alpha_mean <= 1.0), n=n)


# ----------------------------------------------------------------------------------------------------------------
# the scenario influence function


def scenario_influence(points: Sequence[ScenarioPoint],
                       capital_fn: Callable[[Sequence[ScenarioPoint]], float]) -> np.ndarray:
    """``influence_i = (capital(all) - capital(all without i)) / capital(all)``, dimensionless, ``(P,)``. ``capital_fn``
    maps a set of scenario points to the 99.9 percent capital in EUR (the caller closes over the losses, the threshold
    and the
    prior weight); it is called ``P + 1`` times. Raises when ``capital(all) <= 0``."""
    pts = list(points)
    if not pts:
        raise ValueError("no scenario points")
    base = scalar(capital_fn(list(pts)), "capital(all)")
    if base <= 0.0:
        raise ValueError(f"capital(all) must be positive, got {base}")
    out = np.empty(len(pts))
    for i in range(len(pts)):
        without = scalar(capital_fn(pts[:i] + pts[i + 1:]), f"capital(without {i})")
        out[i] = (base - without) / base
    return out

__all__ = [
    "CredibilityFit",
    "GammaPrior",
    "ScenarioPoint",
    "credibility_pareto",
    "fit_tail_to_scenarios",
    "gamma_prior_from_mean_interval",
    "gamma_prior_from_scenarios",
    "scenario_influence",
    "scenario_pseudo_observations",
]
