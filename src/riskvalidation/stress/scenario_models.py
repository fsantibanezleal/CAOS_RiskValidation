"""Rungs 2 and 3 of the macro stress ladder (dossier 05 H.2, l.388 and l.390): the VAR conditional forecast and growth-
at-risk.

Engine functions with their own acceptance tests that no C18 variant bakes (decision Q13). The Minnesota BVAR is here
(`fit_bvar_minnesota`, BL-078); the panel quantile satellite is `stress.panel_quantile`. The NumPy and SciPy core needs
nothing else;
`gar_quantile_regressions` reads `statsmodels` (the optional extra `stress`), imported inside the function.

- VAR: `y_t = c + sum_l A_l y_(t-l) + u_t`, fitted by OLS equation by equation. A stress scenario is a conditional
forecast given
  a path for a subset of variables (05 l.388): the conditioned variables follow the path exactly.
- Minnesota BVAR (Banbura, Giannone and Reichlin 2008, ECB WP 966, equations (2) and (5) to (9); Karlsson 2012,
equations
  (13) to (22)): the Normal-inverted-Wishart prior imposed by dummy observations, whose posterior mean is the OLS fit
  of the data stacked on the dummies and whose posterior is iW(Sigma_tilde, T + T_d + 2 - k) in SciPy's `invwishart`
  convention; the point fit is a `VarFit` that `var_conditional_forecast` consumes unchanged.
- Growth-at-Risk (Adrian, Boyarchenko and Giannone 2019, AER 109(4):1263-1289, doi:10.1257/aer.20161923): step 1
quantile
  regressions of average annualised growth on a constant, current growth and the Chicago Fed NFCI; step 2 the Azzalini-
  Capitanio skewed t density fitted to the estimated 5, 25, 75 and 95 percent quantiles by squared distance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln, stdtr, stdtrit

DEFAULT_TAUS = (0.05, 0.25, 0.75, 0.95)


@dataclass(frozen=True)
class VarFit:
    """A fitted VAR(p): the intercept, matrices `A_1 .. A_p` and the residual covariance (units of the variables)."""

    intercept: np.ndarray  # (n,)
    coefficients: tuple[np.ndarray, ...]  # A_1 .. A_p, each (n, n)
    sigma: np.ndarray  # (n, n) residual covariance


def fit_var(y: np.ndarray, lags: int) -> VarFit:
    """OLS equation by equation of `y_t = c + sum_l A_l y_(t-l) + u_t` (05 H.2) on `y` (T, n); `sigma = U'U / (T - p -
    (n p + 1))`.
    `ValueError` for `T <= lags * n + 1` and when no residual degree of freedom is left."""
    a = np.array(y, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"y must be a finite (T, n) array, got shape {a.shape}")
    if int(lags) != lags or lags < 1:
        raise ValueError(f"lags must be an integer of at least 1, got {lags!r}")
    p, (t, n) = int(lags), a.shape
    if t <= p * n + 1:
        raise ValueError(f"T = {t} is not above lags * n + 1 = {p * n + 1}")
    dof = (t - p) - (p * n + 1)
    if dof < 1:
        raise ValueError(f"T = {t} leaves no residual degree of freedom for {p * n + 1} coefficients per equation")
    x = np.column_stack([np.ones(t - p)] + [a[p - k : t - k] for k in range(1, p + 1)])
    target = a[p:]
    beta = np.linalg.lstsq(x, target, rcond=None)[0]  # (1 + n p, n)
    resid = target - x @ beta
    sigma = resid.T @ resid / dof
    mats = tuple(beta[1 + (k - 1) * n : 1 + k * n].T.copy() for k in range(1, p + 1))
    return VarFit(intercept=beta[0].copy(), coefficients=mats, sigma=0.5 * (sigma + sigma.T))


def var_conditional_forecast(
    fit: VarFit, y_history: np.ndarray, path: np.ndarray, conditioned: Sequence[int]
) -> np.ndarray:
    """A stress scenario as a conditional forecast given a path for a subset of variables (05 l.388). `y_history` is (p,
    n) the
    last observations, oldest first; `path` (H, m) the given path of the `m` conditioned variables; returns (H, n).

    Period by period, the conditioned variables are set to the path (the shock that makes them follow it is the path
    less their unconditional one-step prediction, `u_c`); the free variables take their own equation with the realised
    lags plus the conditional mean of their shock, `Sigma_fc Sigma_cc^-1 u_c` (the regression of the free shocks on the
    conditioned ones; zero for a diagonal `sigma`). This is the period-by-period form; the full-horizon Waggoner and Zha
    form is UNVERIFIED against the
    paper (the dossier names the method only)."""
    mats = tuple(np.array(m, dtype=float) for m in fit.coefficients)
    n = mats[0].shape[0]
    p = len(mats)
    hist = np.array(y_history, dtype=float)
    if hist.shape != (p, n) or not np.all(np.isfinite(hist)):
        raise ValueError(f"y_history must be a finite ({p}, {n}) array (lags, n), got shape {hist.shape}")
    cond = [int(i) for i in conditioned]
    if len(set(cond)) != len(cond) or any(i < 0 or i >= n for i in cond):
        raise ValueError(f"conditioned must be distinct variable indices in [0, {n - 1}], got {list(conditioned)}")
    pth = np.array(path, dtype=float)
    if pth.ndim != 2 or pth.shape[1] != len(cond) or not np.all(np.isfinite(pth)):
        raise ValueError(f"path must be a finite (H, {len(cond)}) array, got shape {pth.shape}")
    horizon = pth.shape[0]
    free = [i for i in range(n) if i not in cond]
    sigma = np.array(fit.sigma, dtype=float)
    gain = None
    if cond and free:
        s_fc = sigma[np.ix_(free, cond)]
        if np.any(s_fc != 0.0):
            s_cc = sigma[np.ix_(cond, cond)]
            try:
                gain = np.linalg.solve(s_cc.T, s_fc.T).T  # Sigma_fc Sigma_cc^-1
            except np.linalg.LinAlgError as exc:
                raise ValueError("the covariance of the conditioned shocks is singular") from exc
    intercept = np.array(fit.intercept, dtype=float)
    lagged = [hist[p - 1 - k].copy() for k in range(p)]  # lagged[0] = y_(t-1), lagged[1] = y_(t-2), ...
    out = np.empty((horizon, n))
    for h in range(horizon):
        step = intercept + sum(m @ lagged[k] for k, m in enumerate(mats))
        row = step.copy()
        if cond:
            row[cond] = pth[h]
            if gain is not None:
                row[free] = step[free] + gain @ (pth[h] - step[cond])
        out[h] = row
        lagged = [row.copy()] + lagged[:-1]
    return out


# ----------------------------------------------------------------------------------------------------------------
# The skewed t density


def _check_scale(sigma: float, nu: float) -> None:
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    if not (np.isfinite(nu) and nu > 0.0):
        raise ValueError(f"nu must be positive, got {nu!r}")


def _std_pdf(w: np.ndarray, alpha: float, nu: float) -> np.ndarray:
    """The standard (mu 0, sigma 1) Azzalini-Capitanio density (the t density in log form, `special.stdtr`)."""
    log_c = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * math.log(nu * math.pi)
    t_pdf = np.exp(log_c - 0.5 * (nu + 1.0) * np.log1p(w * w / nu))
    return 2.0 * t_pdf * stdtr(nu + 1.0, alpha * w * np.sqrt((nu + 1.0) / (nu + w * w)))


def skewed_t_pdf(y: np.ndarray, mu: float, sigma: float, alpha: float, nu: float) -> np.ndarray:
    """The Azzalini-Capitanio skewed t density
    `f(y) = (2 / sigma) t((y - mu) / sigma; nu) T(alpha (y - mu) / sigma sqrt((nu + 1) / (nu + ((y - mu) / sigma)^2));
    nu + 1)`
    (05 l.390; Adrian, Boyarchenko and Giannone 2019). `alpha = 0` is the Student t; `sigma > 0`, `nu > 0`."""
    _check_scale(float(sigma), float(nu))
    a = np.array(y, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError("y contains non-finite values")
    for name, v in (("mu", mu), ("alpha", alpha)):
        if not np.isfinite(float(v)):
            raise ValueError(f"{name} is non-finite")
    return _std_pdf((a - mu) / sigma, float(alpha), float(nu)) / sigma


@dataclass(frozen=True)
class SkewedTFit:
    """A skewed t fitted to quantiles; `distance` is the squared distance to the target quantiles."""

    mu: float
    sigma: float
    alpha: float
    nu: float
    distance: float  # the squared distance to the four target quantiles


# 12-point Gauss-Legendre rule on panels of width 1: the density is smooth on that scale for nu of about 0.5 and more
_GL_X, _GL_W = np.polynomial.legendre.leggauss(12)


def _std_cdf_from_zero(z: float, alpha: float, nu: float) -> float:
    """`F(z) - F(0)` of the standard skewed t, by Gauss-Legendre panels from 0 to z."""
    if z == 0.0:
        return 0.0
    n_panels = max(1, int(math.ceil(abs(z))))
    edges = np.linspace(0.0, z, n_panels + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    pts = mid[:, None] + half[:, None] * _GL_X[None, :]
    return float(np.sum(half[:, None] * _GL_W[None, :] * _std_pdf(pts, alpha, nu)))


def _std_quantile(tau: float, alpha: float, nu: float) -> float:
    """The `tau` quantile of the standard skewed t. `F(0) = 1/2 - arctan(alpha) / pi` (the sign of Y is that of the
    skew-normal
    numerator, so it does not depend on nu; checked against quadrature in the tests), then a root find of `F(z) -
    tau`."""
    f0 = 0.5 - math.atan(alpha) / math.pi
    if tau == f0:
        return 0.0
    guess = float(stdtrit(nu, tau))
    lo, hi = (0.0, max(1.0, abs(guess)) * 2.0) if tau > f0 else (-max(1.0, abs(guess)) * 2.0, 0.0)

    def gap(z: float) -> float:
        return f0 + _std_cdf_from_zero(z, alpha, nu) - tau

    for _ in range(60):  # widen the bracket until it holds the root (heavy tails, strong skew)
        if gap(lo) <= 0.0 <= gap(hi):
            break
        lo, hi = (lo, hi * 2.0) if tau > f0 else (lo * 2.0, hi)
    return float(optimize.brentq(gap, lo, hi, xtol=1e-13, rtol=1e-13))


def _project(alpha: float, nu: float, taus: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """For a shape (alpha, nu) the location and scale that minimise the squared distance are a linear LS."""
    z = np.array([_std_quantile(float(t), alpha, nu) for t in taus])
    design = np.column_stack([np.ones_like(z), z])
    mu, sigma = np.linalg.lstsq(design, q, rcond=None)[0]
    if sigma <= 1e-12:
        sigma = 1e-12
        mu = float(np.mean(q - sigma * z))
    resid = q - (mu + sigma * z)
    return float(mu), float(sigma), float(resid @ resid)


# the degrees of freedom the fit searches, as their logarithm (the search variable)
_LOG_NU_BOUNDS = (math.log(0.5), math.log(1000.0))


def _nu_of(log_nu: float) -> float:
    """nu from the search variable, clipped to [0.5, 1000] before exponentiating: on quantiles close to a normal's the
    distance is flat above the upper bound and Nelder-Mead walks log nu far up, where `exp` overflowed (BL-106)."""
    return math.exp(min(max(log_nu, _LOG_NU_BOUNDS[0]), _LOG_NU_BOUNDS[1]))


def fit_skewed_t_to_quantiles(quantiles: np.ndarray, taus: Sequence[float] = DEFAULT_TAUS) -> SkewedTFit:
    """Step 2 of growth-at-risk: fit the skewed t by minimising the squared distance to the estimated 5, 25, 75 and 95
    percent quantiles (05 l.390). The location and scale are the exact least-squares solution for each shape, so only
    (alpha, nu) are searched, by Nelder-Mead from several fixed starts; `distance` is the minimum found over the starts.
    `ValueError` for decreasing
    quantiles."""
    q = np.array(quantiles, dtype=float)
    tau = np.array(taus, dtype=float)
    if q.ndim != 1 or tau.ndim != 1 or q.size != tau.size or q.size < 4:
        raise ValueError(
            f"quantiles and taus must be one-dimensional of equal length, at least 4, got {q.shape} and {tau.shape}"
        )
    if not np.all(np.isfinite(q)) or not np.all((tau > 0.0) & (tau < 1.0)) or np.any(np.diff(tau) <= 0.0):
        raise ValueError("taus must be increasing in (0, 1) and quantiles finite")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("quantiles must be increasing in tau: the quantiles given decrease or repeat")
    best: SkewedTFit | None = None
    for alpha0, nu0 in ((0.0, 5.0), (2.0, 5.0), (-2.0, 5.0), (0.0, 20.0)):

        def objective(v: np.ndarray) -> float:
            return _project(float(v[0]), _nu_of(float(v[1])), tau, q)[2]

        res = optimize.minimize(
            objective,
            np.array([alpha0, math.log(nu0)]),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-20, "maxiter": 800, "maxfev": 1600},
        )
        alpha, nu = float(res.x[0]), _nu_of(float(res.x[1]))
        mu, sigma, dist = _project(alpha, nu, tau, q)
        if best is None or dist < best.distance:
            best = SkewedTFit(mu=mu, sigma=sigma, alpha=alpha, nu=nu, distance=dist)
    assert best is not None
    return best


# ----------------------------------------------------------------------------------------------------------------
# Growth-at-risk, step 1


def gar_quantile_regressions(y: np.ndarray, x: np.ndarray, taus: Sequence[float] = DEFAULT_TAUS) -> np.ndarray:
    """Step 1 of growth-at-risk: the quantile regression of the average annualised growth `y` (T,) on `x` (T, k, with a
    constant column) at each tau (05 l.390), minimising the check loss; returns (len(taus), k). Uses
    `statsmodels.QuantReg` (the optional
    extra `stress`). The NFCI is in no dossier 08 row: a bake that needs it must register it first."""
    t = list(taus)
    if not t or any(not (0.0 < float(v) < 1.0) for v in t):
        raise ValueError(f"each tau must lie in (0, 1), got {t}")
    yy = np.array(y, dtype=float)
    xx = np.array(x, dtype=float)
    if yy.ndim != 1 or xx.ndim != 2 or xx.shape[0] != yy.size:
        raise ValueError(f"y (T,) and x (T, k) must agree, got {yy.shape} and {xx.shape}")
    if not (np.all(np.isfinite(yy)) and np.all(np.isfinite(xx))):
        raise ValueError("y or x contains non-finite values")
    if yy.size <= xx.shape[1]:
        raise ValueError(f"T = {yy.size} is not above the {xx.shape[1]} regressors")
    try:
        from statsmodels.regression.quantile_regression import QuantReg
    except ImportError as exc:
        raise ImportError(
            "gar_quantile_regressions needs statsmodels, the optional extra `stress` "
            "(pip install riskvalidation[stress])"
        ) from exc
    out = np.empty((len(t), xx.shape[1]))
    for i, tau in enumerate(t):
        out[i] = np.asarray(QuantReg(yy, xx).fit(q=float(tau), max_iter=5000, p_tol=1e-10).params, dtype=float)
    return out


# ----------------------------------------------------------------------------------------------------------------
# The Minnesota BVAR by dummy observations


def _bvar_inputs(y: object, lags: int, lam: float) -> tuple[np.ndarray, int, float]:
    a = np.array(y, dtype=float)
    if a.ndim != 2 or a.shape[1] < 1 or not np.all(np.isfinite(a)):
        raise ValueError(f"y must be a finite (T, n) array, got shape {a.shape}")
    if int(lags) != lags or lags < 1:
        raise ValueError(f"lags must be an integer of at least 1, got {lags!r}")
    p = int(lags)
    if a.shape[0] < p + 2:
        raise ValueError(f"y has T = {a.shape[0]} observations, fewer than lags + 2 = {p + 2}")
    lv = float(lam)
    if not (math.isfinite(lv) and lv > 0.0):
        raise ValueError(f"lam must be a finite number above 0, got {lam!r}")
    return a, p, lv


def _bvar_vector(x: object, n: int, name: str, positive: bool) -> np.ndarray:
    v = np.array(x, dtype=float)
    if v.shape != (n,) or not np.all(np.isfinite(v)) or (positive and np.any(v <= 0.0)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}vector of length {n}, got {x!r}")
    return v


def _lag_regression(a: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """X (T - p, n p + 1) with rows (y_(t-1), ..., y_(t-p), 1) and the target Y = y[p:]."""
    t = a.shape[0]
    x = np.column_stack([a[p - k : t - k] for k in range(1, p + 1)] + [np.ones(t - p)])
    return x, a[p:]


def _ar_scales(a: np.ndarray, p: int) -> np.ndarray:
    """The residual standard deviation of a univariate AR(p) with intercept per variable (Banbura et al., after
    Litterman and Sims and Zha): sqrt(RSS / (T - p - (p + 1))), or RSS / (T - p) when no degree of freedom is left."""
    t, n = a.shape
    out = np.empty(n)
    for i in range(n):
        x, target = _lag_regression(a[:, [i]], p)
        beta = np.linalg.lstsq(x, target, rcond=None)[0]
        resid = target[:, 0] - x @ beta[:, 0]
        dof = (t - p) - (p + 1)
        out[i] = math.sqrt(float(resid @ resid) / (dof if dof > 0 else (t - p)))
    if np.any(out <= 1e-12 * max(1.0, float(np.max(np.abs(a))))):
        raise ValueError(
            "sigma_scale: the univariate AR(p) residual scale of a variable is zero (too few observations or a "
            "constant variable); pass sigma_scale"
        )
    return out


def bvar_dummies(
    sigma_scale: np.ndarray,
    delta: np.ndarray,
    lam: float,
    lags: int,
    eps: float = 1e-6,
    sum_coef_tau: float | None = None,
    mu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The dummy observations (Y_d, X_d) of Banbura, Giannone and Reichlin (2008), equation (5), with J_p = diag(1, ...,
    p): Y_d = [diag(delta_i sigma_i) / lam; 0_(n (p - 1) x n); diag(sigma_i); 0_(1 x n)], X_d = [J_p (x) diag(sigma_i) /
    lam, 0; 0, 0; 0, eps] (the prior on the autoregressive coefficients, on the covariance, on the intercept); T_d = n p
    + n + 1 rows. With `sum_coef_tau` and the means `mu`, the n sum-of-coefficients rows of (9) are appended, Y_d =
    diag(delta_i mu_i) / tau and X_d = [1_p' (x) diag(delta_i mu_i) / tau, 0]: a row of ones over the lags, as Karlsson
    (2012) writes them and as the identity sum_l A_l = I of the limit tau -> 0 needs. Columns of X_d: the lags in order,
    then the intercept."""
    sig = np.asarray(sigma_scale, dtype=float)
    dl = np.asarray(delta, dtype=float)
    n, p = sig.size, int(lags)
    k = n * p + 1
    yd = np.zeros((n * p + n + 1, n))
    xd = np.zeros((n * p + n + 1, k))
    yd[:n] = np.diag(dl * sig) / lam
    for lag in range(1, p + 1):
        xd[(lag - 1) * n : lag * n, (lag - 1) * n : lag * n] = lag * np.diag(sig) / lam
    yd[n * p : n * p + n] = np.diag(sig)
    xd[-1, -1] = eps
    if sum_coef_tau is not None:
        if mu is None:
            raise ValueError("mu is required with sum_coef_tau")
        scale = dl * np.asarray(mu, dtype=float) / float(sum_coef_tau)
        yd = np.vstack([yd, np.diag(scale)])
        xd = np.vstack([xd, np.column_stack([np.tile(np.diag(scale), (1, p)), np.zeros((n, 1))])])
    return yd, xd


@dataclass(frozen=True, eq=False)
class BvarPosterior:
    """The posterior of the dummy-observation Normal-inverted-Wishart prior: B | Psi ~ MN(`b_mean`, Psi, `xtx_inv`),
    Psi ~ iW(`scale`, `dof`) in SciPy's `invwishart` convention (density proportional to |Psi|^(-(v + n + 1)/2)
    exp(-tr(S Psi^-1) / 2), mean S / (v - n - 1)); B is (k, n) with rows (A_1', ..., A_p', c'). `n_dummy` is T_d,
    `n_rows` the T - p regression rows. The hyperparameters used are kept."""

    lags: int
    n_vars: int
    n_rows: int
    n_dummy: int
    lam: float
    eps: float
    sum_coef_tau: float | None
    sigma_scale: np.ndarray
    delta: np.ndarray
    mu: np.ndarray | None
    b_mean: np.ndarray
    xtx_inv: np.ndarray
    scale: np.ndarray
    dof: float

    def var_fit(self) -> VarFit:
        """The point fit: the coefficients the posterior mean, `sigma` the mean Sigma_tilde / (dof - n - 1) of Psi."""
        n, p = self.n_vars, self.lags
        mats = tuple(self.b_mean[(k - 1) * n : k * n].T.copy() for k in range(1, p + 1))
        sigma = self.scale / (self.dof - n - 1.0)
        return VarFit(intercept=self.b_mean[-1].copy(), coefficients=mats, sigma=0.5 * (sigma + sigma.T))


def bvar_posterior(
    y: np.ndarray,
    lags: int,
    lam: float,
    delta: Sequence[float] | None = None,
    sum_coef_tau: float | None = None,
    eps: float = 1e-6,
    sigma_scale: Sequence[float] | None = None,
) -> BvarPosterior:
    """The posterior of the Minnesota prior in its dummy-observation Normal-inverted-Wishart form (Banbura, Giannone and
    Reichlin 2008, equations (5) to (7) and (9)) on `y` (T, n): the dummies of `bvar_dummies` are stacked under the
    data, B_tilde = (X_*'X_*)^-1 X_*'Y_* and Sigma_tilde = (Y_* - X_* B_tilde)'(Y_* - X_* B_tilde), Psi | Y ~
    iW(Sigma_tilde, T + T_d + 2 - k) with k = n p + 1 (the improper prior |Psi|^(-(n + 3)/2) added for the mean to
    exist). The cross-variable tightness theta is fixed at 1, which this form needs: a single tightness for all
    coefficients, so the Litterman equation-by-equation variant and theta other than 1 are not available.

    Parameters: `lam` the overall tightness (0 gives the prior, infinity the OLS fit); `delta` the prior mean of the
    first own lag per variable (ones, a random walk, by default; 0 for a mean-reverting variable); `sum_coef_tau` adds
    the sum-of-coefficients dummies (9) (Banbura et al. use 10 lam), with the sample means of `y`; `eps` the
    near-uninformative intercept prior; `sigma_scale` the scales sigma_i, by default the residual standard deviation of
    a univariate AR(p) per variable. `ValueError` for fewer than lags + 2 observations, a non-positive `lam`, a
    non-finite input, or a `delta` or `sigma_scale` of the wrong length."""
    a, p, lv = _bvar_inputs(y, lags, lam)
    n = a.shape[1]
    dl = np.ones(n) if delta is None else _bvar_vector(delta, n, "delta", False)
    tau: float | None = None
    if sum_coef_tau is not None:
        tau = float(sum_coef_tau)
        if not (math.isfinite(tau) and tau > 0.0):
            raise ValueError(f"sum_coef_tau must be a finite number above 0, got {sum_coef_tau!r}")
    e = float(eps)
    if not (math.isfinite(e) and e > 0.0):
        raise ValueError(f"eps must be a finite number above 0, got {eps!r}")
    sig = _ar_scales(a, p) if sigma_scale is None else _bvar_vector(sigma_scale, n, "sigma_scale", True)
    mu = a.mean(axis=0) if tau is not None else None
    x, target = _lag_regression(a, p)
    yd, xd = bvar_dummies(sig, dl, lv, p, e, tau, mu)
    xs, ys = np.vstack([x, xd]), np.vstack([target, yd])
    xtx = xs.T @ xs
    xtx_inv = np.linalg.inv(xtx)
    b = np.linalg.solve(xtx, xs.T @ ys)
    resid = ys - xs @ b
    k = n * p + 1
    n_rows, n_dummy = x.shape[0], xd.shape[0]
    return BvarPosterior(
        lags=p, n_vars=n, n_rows=n_rows, n_dummy=n_dummy, lam=lv, eps=e, sum_coef_tau=tau, sigma_scale=sig, delta=dl,
        mu=mu, b_mean=b, xtx_inv=0.5 * (xtx_inv + xtx_inv.T), scale=0.5 * (resid.T @ resid + (resid.T @ resid).T),
        dof=float(n_rows + n_dummy + 2 - k),
    )


def fit_bvar_minnesota(
    y: np.ndarray,
    lags: int,
    lam: float,
    delta: Sequence[float] | None = None,
    sum_coef_tau: float | None = None,
    eps: float = 1e-6,
    sigma_scale: Sequence[float] | None = None,
) -> VarFit:
    """The posterior mean of the Minnesota BVAR as a `VarFit` (the intercept, the matrices A_1 .. A_p and `sigma`, the
    posterior mean of Psi), which `var_conditional_forecast` consumes unchanged. See `bvar_posterior` for the prior, the
    parameters and the refusals.

    Degrees of freedom: the posterior is iW(Sigma_tilde, T + T_d + 2 - k) under the convention of SciPy's `invwishart`
    (density proportional to |Psi|^(-(v + n + 1)/2) exp(-tr(S Psi^-1) / 2), mean S / (v - n - 1)), so `sigma` =
    Sigma_tilde / (T + T_d + 2 - k - n - 1). Karlsson's (2012) proper-prior form, v_bar = T + v_0 with v_0 = T_d - k,
    has 2 degrees of freedom fewer (Banbura et al. add the improper prior |Psi|^(-(n + 3)/2)); the coefficients are the
    same in both."""
    return bvar_posterior(y, lags, lam, delta, sum_coef_tau, eps, sigma_scale).var_fit()


@dataclass(frozen=True, eq=False)
class BvarDraws:
    """Posterior draws: `sigma` (N, n, n) the draws of Psi, `coefficients` (N, k, n) the draws of B with rows (A_1',
    ..., A_p', c'), `lags` and `n_vars`."""

    lags: int
    n_vars: int
    sigma: np.ndarray
    coefficients: np.ndarray

    def var_fit(self, i: int) -> VarFit:
        """Draw `i` as a `VarFit`."""
        n, b = self.n_vars, self.coefficients[i]
        mats = tuple(b[(k - 1) * n : k * n].T.copy() for k in range(1, self.lags + 1))
        return VarFit(intercept=b[-1].copy(), coefficients=mats, sigma=self.sigma[i].copy())


def bvar_posterior_draws(posterior: BvarPosterior, n_draws: int, seed: int) -> BvarDraws:
    """`n_draws` draws of the posterior (Karlsson 2012, Algorithm 1 with (18) and (19)): Psi from
    `scipy.stats.invwishart` with the posterior degrees of freedom and scale, then B | Psi matricvariate normal with
    mean `b_mean`, row covariance `xtx_inv` and column covariance Psi, B = B_tilde + L_Omega Z L_Psi' with Z standard
    normal. The mean of the Psi draws is `posterior.var_fit().sigma`. Randomness
    `numpy.random.Generator(PCG64(seed))`."""
    if int(n_draws) != n_draws or n_draws < 1:
        raise ValueError(f"n_draws must be a positive integer, got {n_draws!r}")
    count = int(n_draws)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    n, k = posterior.n_vars, posterior.b_mean.shape[0]
    psi = np.asarray(stats.invwishart.rvs(df=posterior.dof, scale=posterior.scale, size=count, random_state=rng))
    psi = psi.reshape(count, n, n)
    psi = 0.5 * (psi + np.swapaxes(psi, 1, 2))
    l_psi = np.linalg.cholesky(psi)
    l_om = np.linalg.cholesky(posterior.xtx_inv)
    z = rng.standard_normal((count, k, n))
    b = posterior.b_mean[None, :, :] + l_om[None, :, :] @ z @ np.swapaxes(l_psi, 1, 2)
    return BvarDraws(lags=posterior.lags, n_vars=n, sigma=psi, coefficients=b)


__all__ = [
    "BvarDraws",
    "BvarPosterior",
    "SkewedTFit",
    "VarFit",
    "bvar_dummies",
    "bvar_posterior",
    "bvar_posterior_draws",
    "fit_bvar_minnesota",
    "fit_skewed_t_to_quantiles",
    "fit_var",
    "gar_quantile_regressions",
    "skewed_t_pdf",
    "var_conditional_forecast",
]
