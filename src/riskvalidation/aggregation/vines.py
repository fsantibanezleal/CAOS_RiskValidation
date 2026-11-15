"""Vine copula aggregation (BL-078): the selection, the simulation and the aggregate of a regular vine, behind the
optional extra `vine`.

A vine decomposes a d-dimensional copula density into d (d - 1) / 2 pair-copula densities (Aas, Czado, Frigessi and
Bakken 2009, doi:10.1016/j.insmatheco.2007.02.001, equations (8) and (9)); a regular vine admits every tree structure
that satisfies the proximity condition (Bedford and Cooke 2002, Annals of Statistics 30(4):1031-1068). The structure and
the pair copulas are selected sequentially by Dissmann, Brechmann, Czado and Kurowicka (2013, Algorithm 3.1,
doi:10.1016/j.csda.2012.08.010): tree by tree the maximum spanning tree of the absolute empirical Kendall tau, then per
edge the pair-copula family of smallest AIC. The selection and the simulation are those of pyvinecopulib 1.0.1 (Nagler
and Vatter, MIT licence, Python 3.11 or later), which the module imports inside its functions: the core of the engine
stays importable with NumPy and SciPy only (as under Pyodide) and a call that needs the package without it raises
ImportError naming the extra (`pip install riskvalidation[vine]`). Fixed seeds and one thread make a fit and a draw
reproducible.

The inputs of `fit_vine` are pseudo-observations in (0, 1) (the ranks over n + 1 of the caller's data); the margins stay
with the caller, as the quantile functions of `vine_aggregate` are the caller's. Alongside, `gaussian_h_function`,
`gaussian_log_density` and `clayton_log_density` (with `copulas.clayton_h_function`) are the pair-copula building blocks
in NumPy and SciPy alone: the log-likelihood of a vine is the sum of the log-densities of its pair copulas evaluated on
the h-function transforms of the data (Aas et al. 2009, equations (15) and (17)), and the tests reconcile
pyvinecopulib's `loglik` with that sum.

Randomness: pyvinecopulib's generator under `seeds=[seed]` and `num_threads=1`; reproducible for a seed. Contract:
`engine-aggregation-api` (the vine additions of 1.3.0 are recorded when Contraste consumes them), module
aggregation.vines.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special

from riskvalidation.portfolio.portfolio import check_level, measures_from_sample, positive_int

Ppf = Callable[[np.ndarray], np.ndarray]

DEFAULT_FAMILIES = ("indep", "gaussian", "student", "clayton", "gumbel", "frank", "joe", "bb1", "bb6", "bb7", "bb8")
TREE_CRITERIA = ("tau", "rho", "hoeffd", "mcor")
SELECTION_CRITERIA = ("loglik", "aic", "bic")


def _pv() -> Any:
    try:
        import pyvinecopulib
    except ImportError as exc:
        raise ImportError(
            "the vine functions need pyvinecopulib, the optional extra `vine` "
            "(pip install riskvalidation[vine]; Python 3.11 or later)"
        ) from exc
    return pyvinecopulib


# ----------------------------------------------------------------------------------------------------------------
# Pair-copula building blocks (NumPy and SciPy only)


def _unit(x: object, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all((a > 0.0) & (a < 1.0))):
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return a


def _rho(rho: float) -> float:
    r = float(rho)
    if not (math.isfinite(r) and -1.0 < r < 1.0):
        raise ValueError(f"rho must lie in (-1, 1), got {rho}")
    return r


def gaussian_h_function(u1: np.ndarray, u2: np.ndarray, rho: float) -> np.ndarray:
    """The h-function of the Gaussian pair copula, h(u1 | u2) = Phi((Phi^-1(u1) - rho Phi^-1(u2)) / sqrt(1 - rho^2))
    (Aas et al. 2009, Appendix B.1). Golden: 0.5 at u1 = u2 = 0.5 for every rho."""
    r = _rho(rho)
    z1 = special.ndtri(_unit(u1, "u1"))
    z2 = special.ndtri(_unit(u2, "u2"))
    return np.asarray(special.ndtr((z1 - r * z2) / math.sqrt(1.0 - r * r)))


def gaussian_log_density(u1: np.ndarray, u2: np.ndarray, rho: float) -> np.ndarray:
    """log c(u1, u2) of the Gaussian pair copula, -1/2 ln(1 - rho^2) - (rho^2 (z1^2 + z2^2) - 2 rho z1 z2) / (2 (1 -
    rho^2)) with z = Phi^-1(u) (Aas et al. 2009, Appendix B.1)."""
    r = _rho(rho)
    z1 = special.ndtri(_unit(u1, "u1"))
    z2 = special.ndtri(_unit(u2, "u2"))
    quad = (r * r * (z1 * z1 + z2 * z2) - 2.0 * r * z1 * z2) / (2.0 * (1.0 - r * r))
    return np.asarray(-0.5 * math.log1p(-r * r) - quad)


def clayton_log_density(u1: np.ndarray, u2: np.ndarray, theta: float) -> np.ndarray:
    """log c(u1, u2) of the Clayton pair copula, ln(1 + theta) - (1 + theta) ln(u1 u2) - (2 + 1 / theta) ln(u1^-theta +
    u2^-theta - 1) (Aas et al. 2009, Appendix B.3)."""
    th = float(theta)
    if not (math.isfinite(th) and th > 0.0):
        raise ValueError(f"theta must be a finite number above 0, got {theta}")
    a = _unit(u1, "u1")
    b = _unit(u2, "u2")
    s = a ** -th + b ** -th - 1.0
    return np.asarray(math.log1p(th) - (1.0 + th) * (np.log(a) + np.log(b)) - (2.0 + 1.0 / th) * np.log(s))


# ----------------------------------------------------------------------------------------------------------------
# Fit, simulation, aggregate


@dataclass(frozen=True)
class VineFit:
    """A fitted (or given) regular vine: `model` is the pyvinecopulib `Vinecop`; `loglik`, `aic` and `n_obs` are those
    of the data it was fitted on (None for a vine wrapped without data); `n_params` the effective number of parameters;
    `dim` the dimension; `description` a JSON-able dict: the structure kind, the variable order and, per tree, per edge,
    the conditioned pair, the conditioning set, the family, the rotation, the parameters and Kendall's tau of the pair
    copula."""

    model: Any = field(repr=False, compare=False)
    dim: int
    n_obs: int | None
    loglik: float | None
    aic: float | None
    n_params: float
    description: dict[str, Any]


def _describe(model: Any, structure: str) -> dict[str, Any]:
    d = int(model.dim)
    m = np.asarray(model.matrix)
    trees: list[list[dict[str, Any]]] = []
    for t in range(int(model.trunc_lvl)):
        edges: list[dict[str, Any]] = []
        for e in range(d - 1 - t):
            pc = model.get_pair_copula(t, e)
            edges.append({
                "conditioned": [int(m[d - 1 - e, e]), int(m[t, e])],
                "conditioning": [int(m[k, e]) for k in range(t)],
                "family": str(pc.family.name),
                "rotation": int(pc.rotation),
                "parameters": [float(x) for x in np.asarray(pc.parameters).ravel()],
                "tau": float(pc.tau),
            })
        trees.append(edges)
    return {"dim": d, "structure": structure, "order": [int(x) for x in model.order], "trunc_lvl": int(model.trunc_lvl),
            "trees": trees}


def vine_fit_from_model(model: Any, u: np.ndarray | None = None, structure: str = "given") -> VineFit:
    """Wraps a pyvinecopulib `Vinecop` in a `VineFit`: with the pseudo-observations `u` the record carries the
    log-likelihood, AIC and n of that data; without, those three are None. For a vine whose parameters are known (the
    tests of `simulate_vine`) or fitted elsewhere."""
    _pv()
    d = int(model.dim)
    ll: float | None = None
    aic: float | None = None
    n: int | None = None
    if u is not None:
        x = _pseudo(u, "u")
        if x.shape[1] != d:
            raise ValueError(f"u must have the vine's {d} columns, got {x.shape[1]}")
        ll, aic, n = float(model.loglik(x)), float(model.aic(x)), int(x.shape[0])
    return VineFit(model=model, dim=d, n_obs=n, loglik=ll, aic=aic, n_params=float(model.npars),
                   description=_describe(model, structure))


def _pseudo(u: object, name: str) -> np.ndarray:
    a = np.array(u, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2 or a.shape[0] < 2:
        raise ValueError(f"{name} must be an (n, d) array with n and d of at least 2, got shape {a.shape}")
    if not (np.all(np.isfinite(a)) and np.all((a > 0.0) & (a < 1.0))):
        raise ValueError(f"{name} must hold pseudo-observations strictly inside (0, 1) (the ranks over n + 1)")
    return np.asfortranarray(a)


def fit_vine(u: np.ndarray, family_set: Sequence[str] | None = None, trunc_lvl: int | None = None,
             tree_criterion: str = "tau", selection_criterion: str = "aic", structure: str | None = None,
             seed: int = 0) -> VineFit:
    """Fits a regular vine to the pseudo-observations `u` (n, d), by the sequential selection of Dissmann et al. (2013,
    Algorithm 3.1) as pyvinecopulib implements it: per tree the maximum spanning tree of the absolute `tree_criterion`
    (Kendall's "tau" by default) and per edge the family of smallest `selection_criterion` ("aic") among `family_set`
    (default `DEFAULT_FAMILIES`: independence, Gaussian, Student t, Clayton, Gumbel, Frank, Joe, BB1, BB6, BB7, BB8 with
    the rotations), parameters by maximum likelihood. `trunc_lvl` truncates the vine (independence above it).
    `structure` `"d_vine"` or `"c_vine"` fixes the variable order 1..d of that vine and fits only the pair copulas;
    None selects the structure. One thread and `seeds=[seed]` make the fit reproducible. The record is frozen; see
    `VineFit`."""
    pv = _pv()
    x = _pseudo(u, "u")
    d = x.shape[1]
    names = tuple(DEFAULT_FAMILIES if family_set is None else family_set)
    if not names:
        raise ValueError("family_set must name at least one family")
    families = []
    for name in names:
        fam = getattr(pv.BicopFamily, str(name), None)
        if fam is None or str(name).startswith("_"):
            raise ValueError(f"family_set holds an unknown family {name!r}")
        families.append(fam)
    if tree_criterion not in TREE_CRITERIA:
        raise ValueError(f"tree_criterion must be one of {TREE_CRITERIA}, got {tree_criterion!r}")
    if selection_criterion not in SELECTION_CRITERIA:
        raise ValueError(f"selection_criterion must be one of {SELECTION_CRITERIA}, got {selection_criterion!r}")
    if trunc_lvl is not None and (int(trunc_lvl) != trunc_lvl or trunc_lvl < 1):
        raise ValueError(f"trunc_lvl must be an integer of at least 1 or None, got {trunc_lvl!r}")
    kwargs: dict[str, Any] = {}
    if structure is not None:
        if structure not in ("d_vine", "c_vine"):
            raise ValueError(f"structure must be None, 'd_vine' or 'c_vine', got {structure!r}")
        order = list(range(1, d + 1))
        kwargs["structure"] = pv.DVineStructure(order) if structure == "d_vine" else pv.CVineStructure(order)
    controls = pv.FitControlsVinecop(
        family_set=families, tree_criterion=tree_criterion, selection_criterion=selection_criterion,
        trunc_lvl=(2 ** 64 - 1 if trunc_lvl is None else int(trunc_lvl)), num_threads=1, seeds=[int(seed)])
    model = pv.Vinecop.from_data(x, controls, **kwargs)
    return vine_fit_from_model(model, x, structure="selected" if structure is None else structure)


def simulate_vine(fit: VineFit, n: int, seed: int) -> np.ndarray:
    """(n, d) uniforms from the vine (Aas et al. 2009, Algorithms 1 and 2; Dissmann et al. 2013, Algorithm 2.2: the
    inverse Rosenblatt transform of independent uniforms), by pyvinecopulib with `seeds=[seed]` and one thread:
    reproducible for a seed."""
    _pv()
    count = positive_int(n, "n")
    return np.asarray(fit.model.sample(count, num_threads=1, seeds=[int(seed)]), dtype=float)


@dataclass(frozen=True)
class VineAggregate:
    """An aggregate of vine-linked losses at one level, money in the caller's currency: `el` is the caller's sum of the
    marginal expected losses, `ec = var - el`; `var_se` and `es_se` as in `copulas.CopulaAggregate`."""

    dim: int
    n_draws: int
    seed: int
    level: float
    el: float
    var: float
    es: float
    ec: float
    var_se: float | None
    es_se: float | None


def vine_aggregate(ppfs: Sequence[Ppf], fit: VineFit, el: float, level: float = 0.999, n_draws: int = 1_000_000,
                   seed: int = 0) -> VineAggregate:
    """The VaR, ES and EC = VaR - EL of the row sums of L_k = ppf_k(U_k), U drawn by `simulate_vine`, through the reused
    `measures_from_sample` (as `copulas.copula_aggregate`). One quantile function per variable of the vine. The draw is
    one (n_draws, d) array of n_draws * d * 8 bytes. `el` money."""
    lv = check_level(level, "level")
    e = float(el)
    if not math.isfinite(e):
        raise ValueError("el must be finite")
    if len(ppfs) != fit.dim:
        raise ValueError(
            f"ppfs must hold one quantile function per variable: got {len(ppfs)} for a {fit.dim}-dimensional vine")
    u = simulate_vine(fit, n_draws, seed)
    total = np.zeros(u.shape[0])
    for j in range(fit.dim):
        col = np.asarray(ppfs[j](u[:, j]), dtype=float)
        if col.shape != (u.shape[0],) or not np.all(np.isfinite(col)):
            raise ValueError(f"ppfs[{j}] must return {u.shape[0]} finite losses")
        total += col
    m = measures_from_sample(total, [lv], e)[0]
    return VineAggregate(dim=fit.dim, n_draws=int(n_draws), seed=int(seed), level=lv, el=e, var=m.var, es=float(m.es),
                         ec=m.ec, var_se=m.var_se, es_se=m.es_se)


__all__ = [
    "DEFAULT_FAMILIES",
    "VineAggregate",
    "VineFit",
    "clayton_log_density",
    "fit_vine",
    "gaussian_h_function",
    "gaussian_log_density",
    "simulate_vine",
    "vine_aggregate",
    "vine_fit_from_model",
]
