"""The default correlation from a panel of yearly default counts by rating grade (BL-058; contract
`engine-portfolio-api`).

The data are two T x G matrices: the number of defaults ``D[t, g]`` of grade g in year t and the cohort ``N[t, g]``, the
number of obligors in the grade at the start of the year. Two estimators, both read on the primary texts (the research
`wip/contraste/u18-engine-research-2026-11-11.md` of CAOS_MANAGE):

- **Gordy and Heitfield (2002), the maximum likelihood with one factor and one common loading (their MLE3).** An obligor
  of grade g defaults when its asset value falls below the threshold gamma_g, the asset value being w X + sqrt(1 -
  w^2) e with X the systematic factor of the year, X ~ N(0, 1), serially independent. Given X the grades are
  independent and the defaults binomial with p_g(X) = Phi((gamma_g - w X) / sqrt(1 - w^2)). The likelihood of the
  panel is the product over years of the one-dimensional integral over X of the product over grades of the binomial
  probabilities. The estimate is (gamma_1, ..., gamma_G, w); the PD of grade g is Phi(gamma_g), the asset correlation
  w^2, and the default correlation of two obligors of grade g their joint default probability less PD^2 over PD (1 -
  PD), with the joint probability the bivariate normal Phi_2(gamma_g, gamma_g; w^2) (their equation 5). The integral
  is a Gauss-Hermite rule with rule of ``n_nodes`` nodes on a grid of the factor; the paper names none, so the engine
  chooses it and reports how far the log-likelihood moves when the nodes are doubled. The default is the trapezoid
  rule on [-10, 10] with ``N_NODES`` nodes (spacing 0.05): the integrand of a year is a product of binomial
  probabilities, peaked, and the rule is spectrally accurate for it, where Gauss-Hermite with up to 256 nodes moved
  the log-likelihood of a 40-year panel by 1e-5 (``quadrature="gauss-hermite"`` selects it; see the method page).
- **Frey and McNeil (2003), equation (20) with k = 2, per grade.** The joint default probability of two obligors of a
  grade is estimated without bias by the mean over years of D (D - 1) / (N (N - 1)); the default correlation follows
  and the asset correlation by inverting the bivariate normal. It is the cross-check and the fallback where a likelihood
  is not identified.

The default correlation, not the asset correlation, is what the Schuermann and Hanson correction of
`transitions.intervals.effective_n` (their equation 3.4) takes.

A cell that is NaN in either matrix is a grade-year the table lacks: excluded and counted in ``n_missing``, never
filled. A cell with fewer than two obligors is excluded and counted in ``n_small_cohort`` (the moment term is
undefined at N < 2). Counts that are not integers, are negative or exceed their cohort raise ``PanelFitError``. A
grade without a default in any usable year has no identified threshold: it is dropped and reported. A boundary or an
unidentified estimate is a flag or a typed error, never a filled value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, log_ndtr, ndtr, ndtri

from riskvalidation.portfolio.asrf_fit import BOUND_HIGH, BOUND_LOW, _endpoint, _joint_default
from riskvalidation.portfolio.portfolio import check_level, positive_int

N_NODES = 401              # trapezoid nodes on [-FACTOR_RANGE, FACTOR_RANGE]; the tests double them (change < 1e-10)
FACTOR_RANGE = 10.0
QUADRATURES = ("trapezoid", "gauss-hermite")
MAX_REFINEMENTS = 4        # doublings of the default rule: 401, 801, 1601, 3201, 6401 nodes
MAX_GAUSS_HERMITE = 256    # numpy hermegauss overflows from about 300 nodes; a fit allows half, for its doubling check
QUADRATURE_TOL = 1e-6      # |log-likelihood(n nodes) - log-likelihood(2 n nodes)| above this flags the fit
MIN_YEARS = 3              # usable years (rows with at least one usable cell)
SHORT_PANEL = 20           # Gordy and Heitfield studied T = 20 to 160; below it the downward bias is not corrected
GAMMA_BOUNDS = (-10.0, 5.0)
RHO_STARTS = (0.02, 0.10, 0.30)
RHO_BOUNDS = (BOUND_LOW, BOUND_HIGH)


class PanelFitError(ValueError):
    """A panel that cannot be fitted: malformed counts, too few usable years, no grade with a default."""


@dataclass(frozen=True)
class PanelFit:
    """The one-factor common-loading fit of a default panel (Gordy and Heitfield's MLE3).

    ``grades`` are the columns used, ``dropped`` the ``(column, reason)`` pairs that were not; ``pd``, ``gamma`` and
    ``default_correlation`` have one entry per used grade. ``asset_correlation`` is w^2 and ``loading`` is w.
    ``rho_interval`` is the profile-likelihood interval of the asset correlation at ``level`` (None when not asked).
    ``flags`` name every boundary or caveat: ``short_panel``, ``asset_correlation_at_lower_bound`` (no excess
    dispersion), ``asset_correlation_at_upper_bound``, ``quadrature_not_converged``, ``optimiser_not_converged``,
    ``grade_few_defaults`` (a grade with a single default). ``converged`` is True only when no flag but
    ``short_panel`` and ``grade_few_defaults`` is raised."""

    grades: tuple[int, ...]
    dropped: tuple[tuple[int, str], ...]
    pd: np.ndarray
    gamma: np.ndarray
    asset_correlation: float
    loading: float
    default_correlation: np.ndarray
    log_likelihood: float
    rho_interval: tuple[float, float] | None
    level: float
    converged: bool
    flags: tuple[str, ...]
    n_nodes: int
    quadrature: str
    quadrature_diff: float
    n_years: int
    n_cells: int
    n_missing: int
    n_small_cohort: int


@dataclass(frozen=True)
class PanelMoments:
    """The Frey and McNeil moment estimates of every grade (arrays of length G, NaN where ``status`` says not
    identified).

    ``pd`` is the mean of D / N over the usable years, ``joint_default`` the mean of D (D - 1) / (N (N - 1)),
    ``default_correlation`` (joint - pd^2) / (pd - pd^2) and ``asset_correlation`` the root of Phi_2(G(pd), G(pd);
    rho) = joint. ``status`` per grade: ``ok``; ``no_excess_dispersion`` (joint <= pd^2: both correlations are 0.0, the
    boundary, flagged); ``fewer_than_two_defaults`` and ``too_few_years`` (not identified: NaN);
    ``second_moment_above_bound`` (the asset correlation has no root below 1 - 1e-9: NaN)."""

    pd: np.ndarray
    joint_default: np.ndarray
    default_correlation: np.ndarray
    asset_correlation: np.ndarray
    status: tuple[str, ...]
    n_years: tuple[int, ...]
    n_cells: int
    n_missing: int
    n_small_cohort: int


def _counts(defaults: Any, cohort: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """(D, N, usable mask, n_missing, n_small) from the raw inputs; raises PanelFitError for a malformed panel."""
    d = np.array(defaults, dtype=float)
    n = np.array(cohort, dtype=float)
    if d.ndim != 2 or n.ndim != 2:
        raise PanelFitError(
            f"defaults and cohort must be two-dimensional (years x grades), got {d.shape} and {n.shape}")
    if d.shape != n.shape:
        raise PanelFitError(f"defaults and cohort must have the same shape, got {d.shape} and {n.shape}")
    if d.size == 0:
        raise PanelFitError("the panel is empty")
    if np.any(np.isinf(d)) or np.any(np.isinf(n)):
        raise PanelFitError("counts must be finite (NaN marks a missing grade-year)")
    missing = np.isnan(d) | np.isnan(n)
    dv, nv = np.where(missing, 0.0, d), np.where(missing, 0.0, n)
    if np.any(dv < 0) or np.any(nv < 0):
        raise PanelFitError("counts must not be negative")
    if np.any(np.abs(dv - np.rint(dv)) > 1e-9) or np.any(np.abs(nv - np.rint(nv)) > 1e-9):
        raise PanelFitError("counts must be integers")
    dv, nv = np.rint(dv), np.rint(nv)
    if np.any(dv > nv):
        raise PanelFitError("a grade-year has more defaults than obligors")
    small = ~missing & (nv < 2)
    usable = ~missing & (nv >= 2)
    return dv, nv, usable, int(missing.sum()), int(small.sum())


def _check_rule(quadrature: str, n_nodes: int, fit: bool = False) -> int:
    if quadrature not in QUADRATURES:
        raise ValueError(f"quadrature must be one of {QUADRATURES}, got {quadrature!r}")
    k = positive_int(n_nodes, "n_nodes")
    if quadrature == "trapezoid" and k < 3:
        raise ValueError("a trapezoid rule needs at least 3 nodes")
    limit = MAX_GAUSS_HERMITE // 2 if fit else MAX_GAUSS_HERMITE
    if quadrature == "gauss-hermite" and k > limit:
        raise ValueError(f"gauss-hermite is limited to {limit} nodes here (a fit doubles them to check the rule)")
    return k


def _doubled(quadrature: str, n_nodes: int) -> int:
    """The refined rule of the doubling check: half the spacing for the trapezoid, twice the nodes for Gauss-Hermite."""
    return 2 * n_nodes - 1 if quadrature == "trapezoid" else 2 * n_nodes


def _nodes(quadrature: str, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and log-weights of the standard normal integral (the weights sum to 1)."""
    if quadrature == "gauss-hermite":  # probabilists' form
        x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
        return x, np.log(w / math.sqrt(2.0 * math.pi))
    x = np.linspace(-FACTOR_RANGE, FACTOR_RANGE, n_nodes)
    h = x[1] - x[0]
    w = np.full(n_nodes, h)
    w[0] = w[-1] = h / 2.0
    return x, -0.5 * x * x - 0.5 * math.log(2.0 * math.pi) + np.log(w)


class _Panel:
    """The usable cells of the used grades, with what the likelihood needs precomputed."""

    def __init__(self, d: np.ndarray, n: np.ndarray, usable: np.ndarray, n_nodes: int,
                 quadrature: str = "trapezoid") -> None:
        self.d = np.where(usable, d, 0.0)
        self.f = np.where(usable, n - d, 0.0)
        self.g = d.shape[1]
        self.t = d.shape[0]
        self.const = float(np.sum(np.where(usable, gammaln(n + 1.0) - gammaln(d + 1.0) - gammaln(n - d + 1.0), 0.0)))
        self.z, self.logw = _nodes(quadrature, n_nodes)
        self.n_nodes = n_nodes

    def nll(self, gamma: np.ndarray, rho: float, want_grad: bool = True) -> tuple[float, np.ndarray | None]:
        """(-log-likelihood, gradient in (gamma, rho)) at the given parameters."""
        w, s = math.sqrt(rho), math.sqrt(1.0 - rho)
        a = (gamma[:, None] - w * self.z[None, :]) / s                      # (G, K)
        lp, lq = log_ndtr(a), log_ndtr(-a)
        lik = self.d @ lp + self.f @ lq + self.logw[None, :]               # (T, K)
        m = lik.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lik - m).sum(axis=1))
        total = float(lse.sum()) + self.const
        if not want_grad:
            return -total, None
        post = np.exp(lik - lse[:, None])                                   # (T, K) posterior of the factor per year
        logphi = -0.5 * a * a - 0.5 * math.log(2.0 * math.pi)
        ra, rb = np.exp(logphi - lp), np.exp(logphi - lq)                  # phi / Phi(a), phi / Phi(-a)
        dterm = ra * (self.d.T @ post) - rb * (self.f.T @ post)             # (G, K): d log-lik / d a, summed over years
        g_gamma = dterm.sum(axis=1) / s
        da_drho = -self.z[None, :] / (2.0 * w * s) + (gamma[:, None] - w * self.z[None, :]) / (2.0 * s**3)
        g_rho = float((dterm * da_drho).sum())
        return -total, -np.concatenate([g_gamma, [g_rho]])


def _optimise(panel: _Panel, start_gamma: np.ndarray, rho0: float, fixed_rho: float | None = None):
    """L-BFGS-B from (start_gamma, rho0); with ``fixed_rho`` only the thresholds move (the profile)."""
    g = panel.g
    if fixed_rho is None:
        x0 = np.concatenate([start_gamma, [rho0]])
        bounds = [GAMMA_BOUNDS] * g + [RHO_BOUNDS]

        def fun(x):
            v, gr = panel.nll(x[:g], float(x[g]))
            return v, gr
    else:
        x0, bounds = start_gamma.copy(), [GAMMA_BOUNDS] * g

        def fun(x):
            v, gr = panel.nll(x, fixed_rho)
            return v, gr[:g]
    return minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-9, "maxcor": 20})


def default_correlation(pd: Any, asset_correlation: float) -> Any:
    """The default correlation of two obligors with the PD ``pd`` and an asset correlation (Gordy and Heitfield's
    equation 5): (Phi_2(G(pd), G(pd); rho) - pd^2) / (pd (1 - pd)), the joint default probability by the integral of the
    conditional PD squared. ``pd`` in (0, 1), the correlation in (0, 1); a scalar or an array."""
    rho = float(asset_correlation)
    if not 0.0 < rho < 1.0:
        raise ValueError("asset_correlation must lie in (0, 1)")
    a = np.asarray(pd, dtype=float)
    if np.any(a <= 0.0) or np.any(a >= 1.0) or not np.all(np.isfinite(a)):
        raise ValueError("pd must lie strictly between 0 and 1")
    out = np.array([(_joint_default(float(p), rho) - p * p) / (p * (1.0 - p)) for p in a.ravel()]).reshape(a.shape)
    return float(out) if out.ndim == 0 else out


def panel_log_likelihood(defaults: Any, cohort: Any, gamma: Any, asset_correlation: float,
                         n_nodes: int = N_NODES, quadrature: str = "trapezoid") -> float:
    """The log-likelihood of the panel at the thresholds ``gamma`` (one per column) and the asset correlation: the sum
    over years of the log of the integral over the factor of the product over grades of the binomial probabilities
    (Gordy and Heitfield's equation 10), by the ``quadrature`` rule with ``n_nodes`` nodes. Missing and small cells are
    excluded as in the fit."""
    d, n, usable, _, _ = _counts(defaults, cohort)
    gm = np.array(gamma, dtype=float)
    if gm.shape != (d.shape[1],) or not np.all(np.isfinite(gm)):
        raise ValueError(f"gamma must be one finite threshold per grade, got shape {gm.shape}")
    rho = float(asset_correlation)
    if not 0.0 < rho < 1.0:
        raise ValueError("asset_correlation must lie in (0, 1)")
    k = _check_rule(quadrature, n_nodes)
    v, _ = _Panel(d, n, usable, k, quadrature).nll(gm, rho, want_grad=False)
    return -v


def fit_panel_asrf(defaults: Any, cohort: Any, *, level: float = 0.95, interval: bool = True,
                   n_nodes: int | None = None, quadrature: str = "trapezoid") -> PanelFit:
    """Gordy and Heitfield's MLE3 on a T x G panel of defaults and cohorts (see the module docstring).

    Raises ``PanelFitError`` for malformed counts, for fewer than ``MIN_YEARS`` usable years and when no grade has a
    default. Deterministic: fixed starts, a fixed quadrature. ``n_nodes`` None (the default) starts at ``N_NODES`` and
    refines the rule (the doubling of ``_doubled``) up to ``MAX_REFINEMENTS`` times until the log-likelihood at the
    estimate moves by at most ``QUADRATURE_TOL`` when the nodes are doubled, so that cohorts of thousands of obligors
    are integrated accurately; an explicit ``n_nodes`` is used as given and the check is reported
    (``quadrature_diff``, the flag ``quadrature_not_converged``). ``interval=False`` skips the profile-likelihood
    interval of the asset correlation (the likelihood-ratio interval, the values whose profile log-likelihood is at
    least max - chi2_1(level) / 2; an endpoint that reaches the parameter space is the bound, 1e-4 or 0.9999)."""
    lv = check_level(level, "level")
    adaptive = n_nodes is None
    k = _check_rule(quadrature, N_NODES if adaptive and quadrature == "trapezoid" else 64 if adaptive else n_nodes,
                    fit=True)
    d, n, usable, n_missing, n_small = _counts(defaults, cohort)
    n_cells = int(usable.sum())
    dropped: list[tuple[int, str]] = []
    keep: list[int] = []
    for j in range(d.shape[1]):
        if not usable[:, j].any():
            dropped.append((j, "no_usable_cells"))
        elif d[usable[:, j], j].sum() == 0:
            dropped.append((j, "no_defaults"))
        else:
            keep.append(j)
    if not keep:
        raise PanelFitError("no grade has a default in a usable year: no threshold is identified")
    d, n, usable = d[:, keep], n[:, keep], usable[:, keep]
    n_years = int(usable.any(axis=1).sum())
    if n_years < MIN_YEARS:
        raise PanelFitError(f"at least {MIN_YEARS} usable years are needed, got {n_years}")
    pooled = np.array([d[usable[:, j], j].sum() / n[usable[:, j], j].sum() for j in range(len(keep))])
    gamma0 = ndtri(np.clip(pooled, 1e-12, 1.0 - 1e-12))
    for refinement in range(MAX_REFINEMENTS + 1):
        panel = _Panel(d, n, usable, k, quadrature)
        best = None
        for rho0 in RHO_STARTS:
            res = _optimise(panel, gamma0, rho0)
            if best is None or res.fun < best.fun:
                best = res
        gamma, rho = best.x[:-1], float(best.x[-1])
        ll = -float(best.fun)
        k2 = _doubled(quadrature, k)
        if not adaptive or refinement == MAX_REFINEMENTS or (
                quadrature == "gauss-hermite" and _doubled(quadrature, k2) > MAX_GAUSS_HERMITE):
            break
        if abs(ll + _Panel(d, n, usable, k2, quadrature).nll(gamma, rho, want_grad=False)[0]) <= QUADRATURE_TOL:
            break
        k = k2
    flags: list[str] = []
    if n_years < SHORT_PANEL:
        flags.append("short_panel")
    if rho <= RHO_BOUNDS[0] * (1.0 + 1e-9):
        flags.append("asset_correlation_at_lower_bound")
    if rho >= RHO_BOUNDS[1] * (1.0 - 1e-9):
        flags.append("asset_correlation_at_upper_bound")
    if not best.success and best.status != 2:
        flags.append("optimiser_not_converged")
    ll2 = -_Panel(d, n, usable, k2, quadrature).nll(gamma, rho, want_grad=False)[0]
    qdiff = abs(ll - ll2)
    if qdiff > QUADRATURE_TOL:
        flags.append("quadrature_not_converged")
    few = [j for jj, j in enumerate(keep) if d[usable[:, jj], jj].sum() < 2]
    if few:
        flags.append("grade_few_defaults")
    pd = ndtr(gamma)
    rho_clip = min(max(rho, BOUND_LOW), BOUND_HIGH)
    dcorr = np.asarray(default_correlation(pd, rho_clip), dtype=float)
    ci = None
    if interval:
        ci = _profile_interval(panel, gamma, rho_clip, ll, lv)
    ok = not any(f not in ("short_panel", "grade_few_defaults") for f in flags)
    return PanelFit(grades=tuple(keep), dropped=tuple(dropped), pd=pd, gamma=gamma, asset_correlation=rho,
                    loading=math.sqrt(rho), default_correlation=dcorr, log_likelihood=ll, rho_interval=ci, level=lv,
                    converged=ok, flags=tuple(flags), n_nodes=k, quadrature=quadrature,
                    quadrature_diff=qdiff, n_years=n_years,
                    n_cells=n_cells, n_missing=n_missing, n_small_cohort=n_small)


def _profile_interval(panel: _Panel, gamma_hat: np.ndarray, rho_hat: float, ll_hat: float,
                      level: float) -> tuple[float, float]:
    """The likelihood-ratio interval of the asset correlation: the profile log-likelihood (the thresholds maximised at
    each correlation, from the estimate's thresholds) at least max - chi2_1(level) / 2."""
    cutoff = ll_hat - float(stats.chi2.ppf(level, 1)) / 2.0

    def profile(rho: float) -> float:
        res = _optimise(panel, gamma_hat, rho, fixed_rho=rho)
        return -float(res.fun)

    return (_endpoint(profile, rho_hat, BOUND_LOW, cutoff), _endpoint(profile, rho_hat, BOUND_HIGH, cutoff))


def fit_panel_moments(defaults: Any, cohort: Any) -> PanelMoments:
    """Frey and McNeil's moment estimator (2003, equation 20 with k = 2) of every grade (see the module docstring).

    Per grade: pi = mean(D / N), pi2 = mean(D (D - 1) / (N (N - 1))) over the usable years, the default correlation
    (pi2 - pi^2) / (pi - pi^2) and the asset correlation solving Phi_2(G(pi), G(pi); rho) = pi2. A grade with fewer than
    two defaults or fewer than ``MIN_YEARS`` usable years is not identified (NaN and a status); a grade without excess
    dispersion (pi2 <= pi^2) has both correlations 0.0 with the status ``no_excess_dispersion``. Never a filled
    value."""
    d, n, usable, n_missing, n_small = _counts(defaults, cohort)
    g = d.shape[1]
    pd, joint = np.full(g, np.nan), np.full(g, np.nan)
    dcorr, acorr = np.full(g, np.nan), np.full(g, np.nan)
    status: list[str] = []
    years: list[int] = []
    for j in range(g):
        u = usable[:, j]
        years.append(int(u.sum()))
        if u.sum() < MIN_YEARS:
            status.append("too_few_years")
            continue
        dj, nj = d[u, j], n[u, j]
        pd[j] = float(np.mean(dj / nj))
        joint[j] = float(np.mean(dj * (dj - 1.0) / (nj * (nj - 1.0))))
        if dj.sum() < 2:
            status.append("fewer_than_two_defaults")
            continue
        if pd[j] >= 1.0:
            status.append("fewer_than_two_defaults")
            continue
        if joint[j] <= pd[j] ** 2 * (1.0 + 1e-12):
            dcorr[j], acorr[j] = 0.0, 0.0
            status.append("no_excess_dispersion")
            continue
        dcorr[j] = (joint[j] - pd[j] ** 2) / (pd[j] - pd[j] ** 2)
        lo, hi = 1e-9, 1.0 - 1e-9
        if _joint_default(pd[j], hi) < joint[j]:
            status.append("second_moment_above_bound")
            continue
        acorr[j] = float(brentq(lambda r, pj=pd[j], jj=joint[j]: _joint_default(pj, r) - jj, lo, hi, xtol=1e-14,
                                rtol=1e-13))
        status.append("ok")
    return PanelMoments(pd=pd, joint_default=joint, default_correlation=dcorr, asset_correlation=acorr,
                        status=tuple(status), n_years=tuple(years), n_cells=int(usable.sum()), n_missing=n_missing,
                        n_small_cohort=n_small)

__all__ = [
    "N_NODES",
    "QUADRATURES",
    "PanelFit",
    "PanelFitError",
    "PanelMoments",
    "default_correlation",
    "fit_panel_asrf",
    "fit_panel_moments",
    "panel_log_likelihood",
]
