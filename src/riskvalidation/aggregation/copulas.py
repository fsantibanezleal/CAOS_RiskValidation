"""Rungs 3 and 4 of the aggregation ladder: Gaussian and Student t copula aggregation (dossier 05 G.1 item 3).

The copula draw of the uniforms U is restated here from the sources the engine's own `portfolio.copula` docstring cites,
because `portfolio.copula` keeps its sampler private:

- Gaussian copula (Li 2000, doi:10.3905/jfi.2000.319253): Z = L e with L a factor of the correlation matrix R and e
  independent standard normals, U = Phi(Z).
- Student t copula (Frey and McNeil 2003, doi:10.21314/jor.2003.089; Demarta and McNeil 2005,
  doi:10.1111/j.1751-5823.2005.tb00254.x): W = nu / chi2_nu independent of Z, X = sqrt(W) Z, U = t_nu(X). The t copula
  has
  the symmetric tail dependence of `t_tail_dependence`.
- Clayton copula (Clayton 1978; exchangeable, one parameter theta > 0): the Marshall and Olkin (1988) frailty
  algorithm as stated in Hofert (2008, Algorithm 1 and Table 1), V ~ Gamma(1 / theta, 1), U_i = (1 - ln X_i / V)^(-1 /
  theta) with X_i independent uniforms, for any d of 2 or more, and for d = 2 the conditional inverse of the h-function
  of Aas, Czado, Frigessi and Bakken (2009, Appendix B.3). The Clayton copula is lower tail dependent; a loss aggregate
  needs the dependence in the large-loss tail, so the draws are the survival copula 1 - U by default.

The total of the K risks is the row sum of L_k = F_k^-1(U_k) (Sklar). The VaR, ES, EC = VaR - EL and the Monte Carlo
errors
of an aggregate are those of the reused `portfolio.portfolio.measures_from_sample`. Vine copulas are in
`aggregation.vines` (optional extra `vine`).

Randomness: `numpy.random.Generator(numpy.random.PCG64(seed))`; the Gaussian and t result is reproducible
for the same (seed, n, chunk_size) and another chunk size gives a different but equally valid sample; the Clayton draws
are produced in fixed blocks of `CLAYTON_BLOCK` rows, so they depend on (seed, n) only and never on the chunk size.
Nothing reads global state, the clock or
the environment; inputs are never mutated.

Contract: `engine-aggregation-api` 1.0.0 (the Clayton additions of 1.3.0 are recorded when Contraste consumes them),
module aggregation.copulas.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import special

from riskvalidation.aggregation.simple import _matrix, check_correlation
from riskvalidation.portfolio.portfolio import check_level, measures_from_sample, positive_int

DEFAULT_CHUNK = 100_000
UNIFORM_FLOOR = 1e-16       # keeps a draw strictly inside (0, 1) where Phi or t_nu round to 0 or 1
CLAYTON_BLOCK = 16_384      # rows per random block of the Clayton draw: a seed's rows do not depend on the chunk
PSD_FLOOR = 1e-12           # eigenvalues of a singular correlation below this multiple of K are treated as zero

Ppf = Callable[[np.ndarray], np.ndarray]


def _vector(x: object, name: str) -> np.ndarray:
    a = np.array(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def marginal_ppf_sample(sorted_losses: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """The quantile function of a loss sample: `ppf(u) = numpy.interp(u, (i - 0.5) / n, sorted_losses)`, linear between
    the Hazen positions and clamped at the ends. Golden: [0, 10] gives 5.0 at 0.5, 0.0 at 0.1 and 10.0 at 0.9."""
    x = _vector(sorted_losses, "sorted_losses")
    if x.size < 2:
        raise ValueError("sorted_losses needs at least 2 values")
    if np.any(np.diff(x) < 0.0):
        raise ValueError("sorted_losses must be sorted ascending")
    positions = (np.arange(1, x.size + 1) - 0.5) / x.size

    def ppf(u: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(u, dtype=float), positions, x)

    return ppf


def marginal_ppf_pmf(grid: np.ndarray, pmf: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """The quantile function of a probability mass function on an ascending grid: the smallest grid point whose
    cumulative probability is >= u - 1e-12 (the rule of `oprisk.aggregate.quantile_from_pmf`). Golden: grid [0, 1, 2],
    pmf [0.5, 0.3, 0.2] gives 0, 1, 1 and 2 at 0.5, 0.51, 0.8 and 0.81."""
    g = _vector(grid, "grid")
    p = _vector(pmf, "pmf")
    if g.size != p.size or g.size == 0:
        raise ValueError("grid and pmf must be non-empty and of the same length")
    if np.any(np.diff(g) < 0.0):
        raise ValueError("grid must be sorted ascending")
    if np.any(p < 0.0):
        raise ValueError("pmf must not contain a negative entry")
    if abs(float(p.sum()) - 1.0) > 1e-9:
        raise ValueError(f"pmf must sum to 1 (to 1e-9), got {float(p.sum())}")
    cum = np.cumsum(p)

    def ppf(u: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(cum, np.asarray(u, dtype=float) - 1e-12, side="left")
        return g[np.minimum(idx, g.size - 1)]

    return ppf


def _factor(corr: np.ndarray) -> np.ndarray:
    """A matrix A with A A' = corr: the Cholesky factor, or for a singular positive semi-definite correlation (the
    all-ones matrix of the comonotonic case) the eigen factor with the null eigenvalues set to zero and the rows
    rescaled to unit length, so that every Z_k is exactly standard normal and a unit correlation is exactly
    comonotonic."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, vec = np.linalg.eigh(0.5 * (corr + corr.T))
        w = np.where(w > PSD_FLOOR * corr.shape[0], w, 0.0)
        a = vec * np.sqrt(w)
        return a / np.linalg.norm(a, axis=1, keepdims=True)


def _check_family(family: str, nu: float | None) -> float | None:
    if family == "gaussian":
        if nu is not None:
            raise ValueError("nu must be None for the gaussian family")
        return None
    if family == "t":
        if nu is None:
            raise ValueError("nu is required for the t family and must be above 2")
        v = float(nu)
        if not (math.isfinite(v) and v > 2.0):
            raise ValueError(f"nu must be a finite number above 2, got {nu}")
        return v
    raise ValueError(f"family must be 'gaussian' or 't', got {family!r}")


def _draw(rng: np.random.Generator, factor: np.ndarray, family: str, nu: float | None, n: int) -> np.ndarray:
    z = rng.standard_normal((n, factor.shape[0])) @ factor.T
    if family == "gaussian":
        u = special.ndtr(z)
    else:
        assert nu is not None
        w = nu / rng.chisquare(nu, size=n)
        u = special.stdtr(nu, np.sqrt(w)[:, None] * z)
    return np.clip(u, UNIFORM_FLOOR, 1.0 - UNIFORM_FLOOR)


def simulate_copula_uniforms(corr: np.ndarray, family: str, nu: float | None, n: int, seed: int) -> np.ndarray:
    """(n, K) uniforms in (0, 1) with the Gaussian or Student t copula of the correlation `corr` (module docstring).

    The Kendall rank correlation of a pair is (2 / pi) arcsin(rho) for both families, 0.3333 for rho = 0.5."""
    r = _matrix(corr, "corr")
    check_correlation(r)
    nu_v = _check_family(family, nu)
    count = positive_int(n, "n")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return _draw(rng, _factor(r), family, nu_v, count)


def simulate_component_losses(ppfs: Sequence[Ppf], corr: np.ndarray, family: str, nu: float | None, n_draws: int,
                              seed: int, chunk_size: int = DEFAULT_CHUNK) -> np.ndarray:
    """(n_draws, K) money, L_k = ppf_k(U_k) (dossier 05 G.1 item 3, Sklar), drawn in chunks of `chunk_size` rows so the
    working memory of the draw is of the order chunk_size * K * 8 bytes; the returned array is n_draws * K * 8 bytes
    (320 MB at 1e7 draws of 4 risks, the known maximum of the sizing)."""
    r = _matrix(corr, "corr")
    check_correlation(r)
    k = r.shape[0]
    if len(ppfs) != k:
        raise ValueError(f"ppfs must hold one quantile function per risk: got {len(ppfs)} for a {k} x {k} corr")
    nu_v = _check_family(family, nu)
    total = positive_int(n_draws, "n_draws")
    chunk = positive_int(chunk_size, "chunk_size")
    factor = _factor(r)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    out = np.empty((total, k), dtype=float)
    start = 0
    while start < total:
        m = min(chunk, total - start)
        u = _draw(rng, factor, family, nu_v, m)
        for j in range(k):
            col = np.asarray(ppfs[j](u[:, j]), dtype=float)
            if col.shape != (m,) or not np.all(np.isfinite(col)):
                raise ValueError(f"ppfs[{j}] must return {m} finite losses")
            out[start:start + m, j] = col
        start += m
    return out


def t_tail_dependence(rho: float, nu: float) -> float:
    """The symmetric tail dependence coefficient of the bivariate t copula, 2 t_(nu+1)(-sqrt((nu + 1)(1 - rho) / (1 +
    rho))) (dossier 05 G.1 item 3; McNeil, Frey and Embrechts, the chapter number is UNVERIFIED there). Golden: (0.5, 4)
    0.2532, (0, 4) 0.0756, (0.5, 8) 0.1173, (0.5, 30) 0.0030."""
    r, v = float(rho), float(nu)
    if not (math.isfinite(r) and -1.0 < r < 1.0):
        raise ValueError(f"rho must lie in (-1, 1), got {rho}")
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"nu must be positive, got {nu}")
    return float(2.0 * special.stdtr(v + 1.0, -math.sqrt((v + 1.0) * (1.0 - r) / (1.0 + r))))


def clayton_copula(u: float, v: float, theta: float) -> float:
    """C_theta(u, v) = (u^-theta + v^-theta - 1)^(-1/theta) (dossier 05 G.1 item 3), bivariate; the draws are
    `simulate_clayton_uniforms`. Golden: (0.5, 0.5, 1) gives 1/3 and C(u, 1) = u."""
    a, b, th = float(u), float(v), float(theta)
    if not (math.isfinite(a) and 0.0 < a <= 1.0):
        raise ValueError(f"u must lie in (0, 1], got {u}")
    if not (math.isfinite(b) and 0.0 < b <= 1.0):
        raise ValueError(f"v must lie in (0, 1], got {v}")
    if not (math.isfinite(th) and th > 0.0):
        raise ValueError(f"theta must be positive, got {theta}")
    return float((a ** -th + b ** -th - 1.0) ** (-1.0 / th))


def clayton_lower_tail_dependence(theta: float) -> float:
    """The lower tail dependence of the Clayton copula, 2^(-1 / theta) (dossier 05 G.1 item 3). Goldens: 0.5 at theta 1,
    0.7071 at theta 2."""
    th = float(theta)
    if not (math.isfinite(th) and th > 0.0):
        raise ValueError(f"theta must be positive, got {theta}")
    return float(2.0 ** (-1.0 / th))


def _theta(theta: float) -> float:
    th = float(theta)
    if not (math.isfinite(th) and th > 0.0):
        raise ValueError(f"theta must be a finite number above 0, got {theta}")
    return th


def clayton_kendall_tau(theta: float) -> float:
    """Kendall's tau of the Clayton copula, theta / (theta + 2) (Joe 2014, p. 168; the general Archimedean integral
    1 + 4 int phi / phi' of Genest and MacKay 1986 gives the same). Goldens: 1/3, 1/2 and 2/3 at theta 1, 2 and 4."""
    th = _theta(theta)
    return float(th / (th + 2.0))


def clayton_theta_from_tau(tau: float) -> float:
    """The inverse of `clayton_kendall_tau`, theta = 2 tau / (1 - tau), for tau in (0, 1). Goldens: 1, 2 and 4 at
    1/3, 1/2 and 2/3."""
    t = float(tau)
    if not (math.isfinite(t) and 0.0 < t < 1.0):
        raise ValueError(f"tau must lie in (0, 1), got {tau}")
    return float(2.0 * t / (1.0 - t))


def clayton_h_function(u1: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
    """The h-function of the Clayton copula, h(u1 | u2) = dC(u1, u2)/du2 = u2^(-theta-1) (u1^-theta + u2^-theta -
    1)^(-1-1/theta) (Aas et al. 2009, eq. 7 and Appendix B.3). The copula is exchangeable, so h(u2 | u1) is the same
    function with the arguments swapped. `u1`, `u2` in (0, 1)."""
    th = _theta(theta)
    a = np.asarray(u1, dtype=float)
    b = np.asarray(u2, dtype=float)
    if not (np.all((a > 0.0) & (a < 1.0)) and np.all((b > 0.0) & (b < 1.0))):
        raise ValueError("u1 and u2 must lie in (0, 1)")
    s = a ** -th + b ** -th - 1.0
    return np.asarray(np.exp(-(th + 1.0) * np.log(b) - (1.0 + 1.0 / th) * np.log(s)))


def clayton_h_inverse(w: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
    """The inverse of `clayton_h_function` in its first argument, h^-1(w | u2) = [(w u2^(theta+1))^(-theta/(theta+1)) +
    1 - u2^-theta]^(-1/theta) (Aas et al. 2009, Appendix B.3), so that h^-1(h(u1 | u2) | u2) = u1. The map is flat where
    the copula density is small (dh/du1 is the density: 5e-8 at theta 4 in the corner (0.99, 0.01)), so there the
    rounding of w limits the round trip in u1 to about 64 eps w / density, above 1e-12."""
    th = _theta(theta)
    ww = np.asarray(w, dtype=float)
    b = np.asarray(u2, dtype=float)
    if not (np.all((ww > 0.0) & (ww < 1.0)) and np.all((b > 0.0) & (b < 1.0))):
        raise ValueError("w and u2 must lie in (0, 1)")
    inner = np.exp(-(th / (th + 1.0)) * (np.log(ww) + (th + 1.0) * np.log(b))) + 1.0 - b ** -th
    return np.asarray(np.maximum(inner, np.finfo(float).tiny) ** (-1.0 / th))


def _check_clayton_args(theta: float, d: int, n: int, method: str, survival: bool) -> tuple[float, int, int]:
    th = _theta(theta)
    if int(d) != d or d < 2:
        raise ValueError(f"d must be an integer of 2 or more, got {d}")
    count = positive_int(n, "n")
    if method not in ("marshall_olkin", "conditional"):
        raise ValueError(f"method must be 'marshall_olkin' or 'conditional', got {method!r}")
    if method == "conditional" and d != 2:
        raise ValueError(f"the conditional method is the bivariate one (d = 2), got d = {d}")
    if not isinstance(survival, (bool, np.bool_)):
        raise ValueError(f"survival must be a bool, got {survival!r}")
    return th, int(d), count


def _clayton_block(rng: np.random.Generator, theta: float, d: int, m: int, method: str) -> np.ndarray:
    """m rows of the lower-tail Clayton copula, by the Marshall-Olkin frailty algorithm or the conditional inverse."""
    if method == "conditional":
        w = np.clip(rng.random((m, 2)), UNIFORM_FLOOR, 1.0 - UNIFORM_FLOOR)
        u2 = w[:, 0]
        u1 = clayton_h_inverse(w[:, 1], u2, theta)
        return np.column_stack([u1, u2])
    shape = 1.0 / theta
    # V ~ Gamma(1 / theta, 1) in the log scale: a shape below 1 is Gamma(shape + 1) U^(1 / shape) (Stuart), so that the
    # frailty of a very large theta (a shape of 0.005 underflows to 0 in 2.5% of the draws) never divides by zero
    if shape >= 1.0:
        log_v = np.log(rng.gamma(shape, size=m))
    else:
        log_v = np.log(rng.gamma(shape + 1.0, size=m)) + np.log(rng.random(m)) / shape
    x = np.clip(rng.random((m, d)), UNIFORM_FLOOR, 1.0 - UNIFORM_FLOOR)
    t = np.log(-np.log(x)) - log_v[:, None]            # ln((-ln X) / V)
    return np.asarray(np.exp(-np.logaddexp(0.0, t) / theta))   # (1 + (-ln X) / V)^(-1 / theta)


def _clayton_blocks(theta: float, d: int, n: int, seed: int, method: str, survival: bool):
    """The draw in blocks of at most `CLAYTON_BLOCK` rows from one generator: the sequence of rows depends on (seed, n)
    only."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    done = 0
    while done < n:
        m = min(CLAYTON_BLOCK, n - done)
        u = np.clip(_clayton_block(rng, theta, d, m, method), UNIFORM_FLOOR, 1.0 - UNIFORM_FLOOR)
        if survival:
            u = np.clip(1.0 - u, UNIFORM_FLOOR, 1.0 - UNIFORM_FLOOR)
        yield u
        done += m


def simulate_clayton_uniforms(theta: float, d: int, n: int, seed: int, method: str = "marshall_olkin",
                              survival: bool = True) -> np.ndarray:
    """(n, d) uniforms strictly inside (UNIFORM_FLOOR, 1 - UNIFORM_FLOOR) with the Clayton copula of parameter `theta`.

    `method="marshall_olkin"` (any d of 2 or more): V ~ Gamma(1 / theta, 1), X_i independent uniforms, U_i = (1 - ln X_i
    / V)^(-1 / theta) (Marshall and Olkin 1988 as Hofert 2008, Algorithm 1 and Table 1). `method="conditional"` (d = 2
    only): U2 uniform and U1 = h^-1(W | U2) with W uniform (Aas et al. 2009, Appendix B.3). `survival=True`, the
    default, returns 1 - U, the survival copula, whose dependence sits in the joint large-loss tail (the Clayton copula
    itself is lower tail dependent). The Kendall tau of a pair is `clayton_kendall_tau(theta)` either way. The draws
    depend on (theta, d, n, seed, method) only."""
    th, dim, count = _check_clayton_args(theta, d, n, method, survival)
    return np.concatenate(list(_clayton_blocks(th, dim, count, seed, method, bool(survival))), axis=0)


def simulate_clayton_losses(ppfs: Sequence[Ppf], theta: float, n_draws: int, seed: int, method: str = "marshall_olkin",
                            survival: bool = True, chunk_size: int = DEFAULT_CHUNK) -> np.ndarray:
    """(n_draws, K) money, L_k = ppf_k(U_k) with U the Clayton draw of `simulate_clayton_uniforms` (K = len(ppfs)
    risks), the quantile functions applied to about `chunk_size` rows at a time (the working memory of the draw). The
    values do not depend on the chunk size."""
    k = len(ppfs)
    if k < 2:
        raise ValueError(f"ppfs must hold at least 2 quantile functions (one per risk), got {k}")
    th, dim, total = _check_clayton_args(theta, k, n_draws, method, survival)
    chunk = positive_int(chunk_size, "chunk_size")
    out = np.empty((total, k), dtype=float)
    pending: list[np.ndarray] = []
    state = {"rows": 0, "start": 0}

    def flush() -> None:
        if not pending:
            return
        u = np.concatenate(pending, axis=0)
        m = u.shape[0]
        start = state["start"]
        for j in range(k):
            col = np.asarray(ppfs[j](u[:, j]), dtype=float)
            if col.shape != (m,) or not np.all(np.isfinite(col)):
                raise ValueError(f"ppfs[{j}] must return {m} finite losses")
            out[start:start + m, j] = col
        state["start"] = start + m
        state["rows"] = 0
        pending.clear()

    for block in _clayton_blocks(th, dim, total, seed, method, bool(survival)):
        pending.append(block)
        state["rows"] += block.shape[0]
        if state["rows"] >= chunk:
            flush()
    flush()
    return out


@dataclass(frozen=True)
class CopulaAggregate:
    """An aggregate at one level, money in the caller's currency: `el` is the caller's sum of the marginal expected
    losses, `ec = var - el`; `var_se` and `es_se` (batch means, `portfolio.copula.mc_standard_errors`) are both None or
    both set, None below 200 draws."""

    family: str
    nu: float | None
    n_draws: int
    seed: int
    chunk_size: int
    level: float
    el: float
    var: float
    es: float
    ec: float
    var_se: float | None
    es_se: float | None


def copula_aggregate(ppfs: Sequence[Ppf], corr: np.ndarray, family: str, nu: float | None, el: float,
                     level: float = 0.999, n_draws: int = 1_000_000, seed: int = 0,
                     chunk_size: int = DEFAULT_CHUNK) -> CopulaAggregate:
    """The VaR, ES and EC = VaR - EL of the row sums of `simulate_component_losses`, computed by the reused
    `measures_from_sample`. `el` money (the sum of the marginal expected losses)."""
    lv = check_level(level, "level")
    e = float(el)
    if not math.isfinite(e):
        raise ValueError("el must be finite")
    losses = simulate_component_losses(ppfs, corr, family, nu, n_draws, seed, chunk_size)
    m = measures_from_sample(losses.sum(axis=1), [lv], e)[0]
    return CopulaAggregate(family=family, nu=None if nu is None else float(nu), n_draws=int(n_draws), seed=int(seed),
                           chunk_size=int(chunk_size), level=lv, el=e, var=m.var, es=float(m.es), ec=m.ec,
                           var_se=m.var_se, es_se=m.es_se)


@dataclass(frozen=True)
class ClaytonAggregate:
    """An aggregate of Clayton-linked losses at one level, money in the caller's currency: `el` is the caller's sum of
    the marginal expected losses, `ec = var - el`; `var_se` and `es_se` as in `CopulaAggregate`."""

    theta: float
    method: str
    survival: bool
    n_draws: int
    seed: int
    chunk_size: int
    level: float
    el: float
    var: float
    es: float
    ec: float
    var_se: float | None
    es_se: float | None


def clayton_aggregate(ppfs: Sequence[Ppf], theta: float, el: float, level: float = 0.999, n_draws: int = 1_000_000,
                      seed: int = 0, chunk_size: int = DEFAULT_CHUNK, method: str = "marshall_olkin",
                      survival: bool = True) -> ClaytonAggregate:
    """The VaR, ES and EC = VaR - EL of the row sums of `simulate_clayton_losses`, computed by the reused
    `measures_from_sample`, as `copula_aggregate` does for the Gaussian and t families. The Clayton copula is
    exchangeable with one parameter, so there is no correlation matrix. As theta grows the aggregate approaches the
    comonotonic sum (`bounds.comonotonic_var`); as theta falls it approaches the independent aggregate. `el` money."""
    lv = check_level(level, "level")
    e = float(el)
    if not math.isfinite(e):
        raise ValueError("el must be finite")
    losses = simulate_clayton_losses(ppfs, theta, n_draws, seed, method, survival, chunk_size)
    m = measures_from_sample(losses.sum(axis=1), [lv], e)[0]
    return ClaytonAggregate(theta=float(theta), method=method, survival=bool(survival), n_draws=int(n_draws),
                            seed=int(seed), chunk_size=int(chunk_size), level=lv, el=e, var=m.var, es=float(m.es),
                            ec=m.ec, var_se=m.var_se, es_se=m.es_se)


__all__ = [
    "ClaytonAggregate",
    "CopulaAggregate",
    "clayton_aggregate",
    "clayton_copula",
    "clayton_h_function",
    "clayton_h_inverse",
    "clayton_kendall_tau",
    "clayton_lower_tail_dependence",
    "clayton_theta_from_tau",
    "copula_aggregate",
    "marginal_ppf_pmf",
    "marginal_ppf_sample",
    "simulate_clayton_losses",
    "simulate_clayton_uniforms",
    "simulate_component_losses",
    "simulate_copula_uniforms",
    "t_tail_dependence",
]
