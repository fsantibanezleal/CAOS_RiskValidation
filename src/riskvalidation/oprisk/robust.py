"""Robust severity estimators and the mean-adjusted single loss approximation (state of the art, tier sota).

The maximum likelihood fit of a heavy-tailed severity is not robust: a few gross losses move the capital by a multiple
(``docs/oprisk/06_robust-severity.md``). Two alternatives, both on the left-truncated law of dossier 05 F.2 (internal
data are collected above the threshold ``H``, so the density is ``f(x) / (1 - F(H))``):

- ``fit_severity_cvm``: the minimum distance estimator of the Cramer-von Mises statistic
  ``W2 = 1 / (12 n) + sum_i (u_(i) - (2 i - 1) / (2 n))^2`` with ``u = (F(x) - F(H)) / (1 - F(H))`` (Cramer 1928;
  von Mises 1931; the statistic in this form is Anderson and Darling's 1952 computing formula; as an estimator,
  Wolfowitz 1957, minimum distance method).
- ``fit_severity_obre``: the optimally bias-robust estimator (OBRE) of Hampel, Ronchetti, Rousseeuw and Stahel
  (1986, *Robust Statistics: the approach based on influence functions*, section 4.2): the solution of ``sum_i A
  (s(x_i; theta) - a) W_i = 0`` with ``s`` the score of the truncated density, ``W_i = min(1, c / ||A (s_i -
  a)||)``, and ``(a, A)`` fixed by ``E[A (s - a) W] = 0`` and ``E[A (s - a) W s'] = I`` under the model (the
  expectations are taken by a quantile midpoint rule of 2,000 nodes and the conditions solved by Levenberg-Marquardt
  on the whitened scores). The bound ``c`` on the gross-error sensitivity is chosen so that the asymptotic
  efficiency ``tr(I^-1) / tr(E[psi psi'])`` at the Cramer-von Mises estimate equals the requested ``efficiency``.

Opdyke and Cavallo (2012, *Estimating operational risk capital: the challenges of truncation, the hazards of maximum
likelihood estimation, and the promise of robust statistics*, Journal of Operational Risk 7(3)) recommend these two
estimators, with the mean-adjusted single loss approximation (``sla_var_mean_adjusted``), for the truncated severity of
an operational risk loss distribution. The primary text was not read here (it is behind a paywall); the estimators are
implemented from the textbook definitions above and the SLA form from the secondary source named in its docstring.

Laws: ``"lognormal"`` ``(mu, sigma)``, ``"weibull"`` ``(shape, scale)`` and ``"gpd"`` ``(xi, beta, location)`` with the
location fixed at the threshold, as ``distributions.fit_severity``. Money is a float in the caller's currency (the
engine never converts); the parameters of a log-scale are dimensionless. Every function is pure and deterministic (no
random draw). Contract: ``engine-oprisk-api`` 1.1.0, module robust.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from riskvalidation.oprisk._checks import level_in_open_unit, positive, scalar, vec
from riskvalidation.oprisk.distributions import (
    FITTED_SEVERITY_LAWS,
    _log_sf_scalar,
    _logpdf,
    _nelder_mead,
    _sev_params,
    fit_severity,
    severity_moment,
    severity_ppf,
    truncated_severity_cdf,
)
from riskvalidation.oprisk.sla import sla_var

_NODES = 2000  # quantile midpoints of the model expectations
_FD = 1e-5  # step of the central difference of the score on the unconstrained parameters
_MIN_OBS = 10  # an OBRE needs this many losses


# ----------------------------------------------------------------------------------------------------------------
# the mean-adjusted SLA


@dataclass(frozen=True)
class MeanAdjustedSla:
    """The mean-adjusted single loss approximation in the caller's currency:
    ``var = var_first_order + mean_adjustment`` with ``mean_adjustment = (E[N] - 1) E[X]``. ``status`` is
    ``"SECONDARY_SOURCE"``: the form is the one a published text cites (arXiv 1607.02319, Basel Committee SMA analysis,
    eq. for the SLA with the mean), and the Opdyke and Cavallo 2012 primary text was not read."""

    level: float
    frequency_mean: float
    var_first_order: float
    mean_adjustment: float
    var: float
    status: str


def sla_var_mean_adjusted(frequency_mean: float, sev_law: str, sev_params: Sequence[float],
                          level: float) -> MeanAdjustedSla:
    """``F^{-1}(1 - (1 - level) / E[N]) + (E[N] - 1) E[X]``: the single loss approximation with the mean adjustment.

    Compared with ``sla.sla_var_mean_corrected`` (``+ E[N] E[X]``, UNVERIFIED) the adjustment is one mean smaller: the
    exact aggregate quantile of a subexponential compound sum is, to the second order, the severity quantile at
    ``1 - (1 - level) / E[N]`` plus ``(E[N] - 1) E[X]`` (the other ``E[N] - 1`` losses are at their means; Degen 2010,
    J. Operational Risk 5(4):3-17, doi:10.21314/JOP.2010.084, as the SLA is stated in arXiv 1607.02319).

    Raises ``ValueError("infinite mean")`` for a severity without a mean (a tail index of 1 or more), for ``E[N] < 1``
    and for the arguments ``sla.sla_var`` refuses. Units: ``frequency_mean`` a count per year, the result the
    severity's currency per year.
    """
    en = positive(frequency_mean, "frequency_mean")
    level = level_in_open_unit(level)
    if en < 1.0:
        raise ValueError(f"E[N] = {en} is below 1: the mean adjustment (E[N] - 1) E[X] would be negative")
    first = sla_var(en, sev_law, sev_params, level)
    mean = severity_moment(sev_law, sev_params, 1)
    if math.isinf(mean):
        raise ValueError("infinite mean: the severity has no mean (tail index of 1 or more), so the mean adjustment "
                         "does not exist")
    adj = (en - 1.0) * mean
    return MeanAdjustedSla(level=level, frequency_mean=en, var_first_order=first, mean_adjustment=adj,
                           var=first + adj, status="SECONDARY_SOURCE")


# ----------------------------------------------------------------------------------------------------------------
# the fitted record


@dataclass(frozen=True)
class RobustSeverityFit:
    """A robust severity fit: ``params`` as in ``distributions`` (the GPD location is the threshold), ``method`` is
    ``"cvm"`` or ``"obre"``, ``statistic`` the Cramer-von Mises distance at the fit (dimensionless), ``converged``
    whether the optimiser met its tolerance. For ``"obre"``: ``bound`` is the tuning constant ``c``, ``efficiency`` the
    asymptotic efficiency at the fit (versus maximum likelihood) and ``weights`` the final weights ``W_i`` in (0, 1]
    in the order of the input (a weight well below 1 marks a loss the estimator discounts). For ``"cvm"``: ``bound``
    and ``efficiency`` are ``nan`` and ``weights`` is empty."""

    law: str
    method: str
    params: tuple[float, ...]
    threshold: float
    n: int
    statistic: float
    converged: bool
    bound: float
    efficiency: float
    weights: np.ndarray


# ----------------------------------------------------------------------------------------------------------------
# shared input handling


def _inputs(law: str, losses: np.ndarray, threshold: float, min_n: int) -> tuple[np.ndarray, float]:
    if law not in FITTED_SEVERITY_LAWS:
        raise ValueError(f"the robust fits support {FITTED_SEVERITY_LAWS}, got {law!r}")
    x = vec(losses, "losses")
    h = scalar(threshold, "collection_threshold")
    if h < 0.0:
        raise ValueError(f"collection_threshold must not be negative, got {h}")
    if x.size < min_n:
        raise ValueError(f"at least {min_n} losses are needed, got {x.size}")
    if np.any(x < h):
        raise ValueError("a loss lies below the collection threshold: below-threshold rows are rejected")
    if law != "gpd" and np.any(x <= 0.0):
        raise ValueError(f"the {law} law needs positive losses")
    return x, h


def _full(law: str, eta: np.ndarray, h: float) -> tuple[float, ...]:
    """Parameters from the unconstrained vector: the log of every positive parameter, the GPD location at ``h``."""
    if law == "lognormal":
        return (float(eta[0]), math.exp(float(eta[1])))
    if law == "weibull":
        return (math.exp(float(eta[0])), math.exp(float(eta[1])))
    return (math.exp(float(eta[0])), math.exp(float(eta[1])), h)


def _eta(law: str, params: Sequence[float]) -> np.ndarray:
    if law == "lognormal":
        return np.array([params[0], math.log(params[1])])
    return np.array([math.log(params[0]), math.log(params[1])])


def cramer_von_mises_distance(law: str, params: Sequence[float], losses: np.ndarray,
                              collection_threshold: float = 0.0) -> float:
    """``W2 = 1 / (12 n) + sum_i (u_(i) - (2 i - 1) / (2 n))^2`` (dimensionless) of the losses against the severity law
    truncated at ``collection_threshold`` (``u = (F(x) - F(H)) / (1 - F(H))``; with a zero threshold, ``u = F(x)``).
    Zero cannot be reached: the minimum for ``n`` observations is ``1 / (12 n)``."""
    x = vec(losses, "losses")
    p = _sev_params(law, params)
    h = scalar(collection_threshold, "collection_threshold")
    u = np.sort(truncated_severity_cdf(law, p, x, h))
    n = x.size
    i = np.arange(1, n + 1)
    return float(1.0 / (12.0 * n) + np.sum((u - (2.0 * i - 1.0) / (2.0 * n)) ** 2))


# ----------------------------------------------------------------------------------------------------------------
# Cramer-von Mises estimator


def _start(law: str, x: np.ndarray, h: float) -> np.ndarray:
    """A starting vector from the truncated maximum likelihood fit, else from the moments of the logarithm."""
    try:
        f = fit_severity(law, x, h, truncated=True)
        if all(math.isfinite(v) and v > 0.0 for v in f.params[(1 if law == "lognormal" else 0):2]):
            return _eta(law, f.params)
    except ValueError:
        pass
    lx = np.log(np.maximum(x, 1e-300) if law != "gpd" else np.maximum(x - h, 1e-300) + 1.0)
    if law == "lognormal":
        return np.array([float(lx.mean()), math.log(max(float(lx.std()), 1e-6))])
    return np.array([math.log(1.0), math.log(float(np.mean(x - h) if law == "gpd" else np.mean(x)))])


def _cvm_solve(law: str, xs: np.ndarray, h: float, x0: np.ndarray) -> tuple[np.ndarray, float, bool]:
    n = xs.size
    i = np.arange(1, n + 1)
    target = (2.0 * i - 1.0) / (2.0 * n)

    def obj(eta: np.ndarray) -> float:
        try:
            p = _full(law, eta, h)
            u = truncated_severity_cdf(law, p, xs, h)
        except (ValueError, OverflowError, FloatingPointError):
            return math.inf
        v = float(1.0 / (12.0 * n) + np.sum((u - target) ** 2))
        return v if math.isfinite(v) else math.inf

    res = _nelder_mead(obj, x0)
    return res.x, float(res.fun), bool(res.success and math.isfinite(res.fun))


def fit_severity_cvm(law: str, losses: np.ndarray, collection_threshold: float = 0.0) -> RobustSeverityFit:
    """The minimum Cramer-von Mises distance fit of a severity law to losses recorded at or above
    ``collection_threshold`` (money units). Derivative free (Nelder-Mead from the truncated maximum likelihood fit),
    deterministic. Raises ``ValueError`` for an unsupported law, fewer than 3 losses, a non-finite loss or a loss below
    the threshold."""
    x, h = _inputs(law, losses, collection_threshold, 3)
    xs = np.sort(x)
    eta, w2, ok = _cvm_solve(law, xs, h, _start(law, x, h))
    return RobustSeverityFit(law, "cvm", _full(law, eta, h), h, int(x.size), w2, ok, math.nan, math.nan,
                             np.empty(0))


# ----------------------------------------------------------------------------------------------------------------
# optimally bias-robust estimator


class _Model:
    """The truncated density of one law on the unconstrained parameters, its score and its quantile nodes."""

    def __init__(self, law: str, h: float):
        self.law = law
        self.h = h
        self.k = 2
        self.u = (np.arange(_NODES) + 0.5) / _NODES

    def logf(self, eta: np.ndarray, x: np.ndarray) -> np.ndarray:
        p = _sev_params(self.law, _full(self.law, eta, self.h))
        return _logpdf(self.law, p, x) - _log_sf_scalar(self.law, p, self.h)

    def score(self, eta: np.ndarray, x: np.ndarray) -> np.ndarray:
        s = np.empty((x.size, self.k))
        for j in range(self.k):
            e = np.zeros(self.k)
            e[j] = _FD
            s[:, j] = (self.logf(eta + e, x) - self.logf(eta - e, x)) / (2.0 * _FD)
        return s

    def nodes(self, eta: np.ndarray) -> np.ndarray:
        """The scores at the model's quantile midpoints."""
        p = _full(self.law, eta, self.h)
        if self.h > 0.0 and self.law != "gpd":
            f_h = float(truncated_severity_cdf(self.law, p, np.array([self.h]), 0.0)[0])
        else:
            f_h = 0.0
        xs = severity_ppf(self.law, p, f_h + self.u * (1.0 - f_h))
        return self.score(eta, xs)

    def whitening(self, s: np.ndarray) -> np.ndarray:
        """``T`` with ``T' F T = I`` for the Fisher information ``F`` of the node scores: the conditions and the weights
        are invariant to this change of basis, and the solver is far better conditioned in it."""
        lc = np.linalg.cholesky(s.T @ s / s.shape[0])
        return np.linalg.inv(lc).T


def _solve_aA(s: np.ndarray, c: float, start: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """``(a, A, w, z, ok)``: the centre ``a`` and the symmetric matrix ``A`` of the OBRE conditions on the model scores
    ``s`` (``G x 2``) at the bound ``c``, the weights at the nodes, the solution vector (a warm start for the next
    call) and whether the conditions ``E[(s - a) W] = 0`` and ``E[A (s - a) W s'] = I`` hold to 1e-9. Solved by
    Levenberg-Marquardt (the plain fixed point on ``A`` diverges: a larger ``A`` gives smaller weights, a smaller
    ``M1`` and a larger ``A``)."""
    g = s.shape[0]

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return z[:2], np.array([[z[2], z[3]], [z[3], z[4]]])

    def weights(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, amat = unpack(z)
        d = s - a
        r = np.sqrt(np.sum((d @ amat) ** 2, axis=1))
        return d, np.minimum(1.0, c / np.maximum(r, 1e-300)), amat

    def resid(z: np.ndarray) -> np.ndarray:
        d, w, amat = weights(z)
        dw = d * w[:, None]
        m = amat @ (dw.T @ s) / g
        return np.concatenate([dw.mean(axis=0), (m - np.eye(2)).ravel()])

    if start is None:
        ainv = np.linalg.inv(s.T @ s / g)
        start = np.array([0.0, 0.0, ainv[0, 0], ainv[0, 1], ainv[1, 1]])
    def solved(r) -> bool:
        return bool(np.all(np.isfinite(r.x)) and float(np.max(np.abs(r.fun))) < 1e-9)

    sol = optimize.least_squares(resid, start, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
    if not solved(sol):
        # MINPACK's step tests can stop short of the root on some SciPy releases (1.15 stalls at a residual near 0.16
        # where 1.18 reaches 1e-16): the trust-region solver on the Jacobian's scale, then Levenberg-Marquardt from it
        alt = optimize.least_squares(resid, sol.x if np.all(np.isfinite(sol.x)) else start, method="trf", x_scale="jac",
                                     xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        alt = optimize.least_squares(resid, alt.x, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
        if solved(alt) or float(np.max(np.abs(alt.fun))) < float(np.max(np.abs(sol.fun))):
            sol = alt
    d, w, _ = weights(sol.x)
    ok = solved(sol)
    a, amat = unpack(sol.x)
    return a, amat, w, sol.x, ok


def _efficiency(s: np.ndarray, c: float, start: np.ndarray | None = None) -> tuple[float, np.ndarray, bool]:
    g = s.shape[0]
    a, amat, w, z, ok = _solve_aA(s, c, start)
    psi = ((s - a) * w[:, None]) @ amat
    v = psi.T @ psi / g
    fisher = s.T @ s / g
    return float(np.trace(np.linalg.inv(fisher)) / np.trace(v)), z, ok


def _calibrate(s: np.ndarray, efficiency: float) -> float:
    """The smallest bound ``c`` whose asymptotic efficiency reaches the target, by bisection on ``log c`` with a warm
    start; a bound below the existence limit (conditions unsolvable) counts as too small."""
    hi = 60.0
    e_hi, z, _ = _efficiency(s, hi)
    if e_hi < efficiency:
        return hi
    lo = math.sqrt(s.shape[1]) * 1.0005
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        e, z_mid, ok = _efficiency(s, mid, z)
        if ok and e >= efficiency:
            hi, z = mid, z_mid
        else:
            lo = mid
        if hi / lo - 1.0 < 1e-5:
            break
    return hi


def fit_severity_obre(law: str, losses: np.ndarray, collection_threshold: float = 0.0,
                      efficiency: float = 0.95) -> RobustSeverityFit:
    """The optimally bias-robust (Hampel) fit of a severity law to losses at or above ``collection_threshold`` (money
    units), at the asymptotic ``efficiency`` (versus maximum likelihood, strictly between 0.5 and 1; 0.95 is the usual
    choice, and an efficiency near 1 approaches the maximum likelihood fit). Starts from the Cramer-von Mises estimate;
    deterministic. Raises ``ValueError`` for an unsupported law, an efficiency outside (0.5, 1), fewer than 10 losses,
    a non-finite loss or a loss below the threshold. ``converged`` is False when the estimating equation is not solved
    to tolerance; the fit then holds the Cramer-von Mises start."""
    x, h = _inputs(law, losses, collection_threshold, _MIN_OBS)
    eff = scalar(efficiency, "efficiency")
    if not 0.5 < eff < 1.0:
        raise ValueError(f"efficiency must lie in (0.5, 1), got {eff}")
    xs = np.sort(x)
    eta0, _, _ = _cvm_solve(law, xs, h, _start(law, x, h))
    model = _Model(law, h)
    s0 = model.nodes(eta0)
    c = _calibrate(s0 @ model.whitening(s0), eff)

    def weights_and_psi(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sn = model.nodes(eta)
        tw = model.whitening(sn)
        a, amat, _, _, _ = _solve_aA(sn @ tw, c)
        si = model.score(eta, x) @ tw
        d = si - a
        w = np.minimum(1.0, c / np.maximum(np.linalg.norm(d @ amat, axis=1), 1e-300))
        return a, amat, w, d * w[:, None]

    def equation(eta: np.ndarray) -> np.ndarray:
        try:
            return weights_and_psi(eta)[3].mean(axis=0)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return np.full(2, 1e6)

    sol = optimize.root(equation, eta0, method="hybr", tol=1e-11)
    ok = bool(sol.success and np.all(np.isfinite(sol.x)) and float(np.max(np.abs(equation(sol.x)))) < 1e-6)
    eta = sol.x if ok else eta0
    _, _, w, _ = weights_and_psi(eta)
    p = _full(law, eta, h)
    sn = model.nodes(eta)
    achieved = _efficiency(sn @ model.whitening(sn), c)[0]
    w2 = cramer_von_mises_distance(law, p, x, h)
    return RobustSeverityFit(law, "obre", p, h, int(x.size), w2, ok, c, achieved, w)


__all__ = [
    "MeanAdjustedSla",
    "RobustSeverityFit",
    "cramer_von_mises_distance",
    "fit_severity_cvm",
    "fit_severity_obre",
    "sla_var_mean_adjusted",
]
