"""The dynamic NII simulation and the stochastic EaR and EVaR: Hull-White one-factor and dynamic Nelson-Siegel
scenarios (dossier 05 C.4, D.5 and D.6).

Contract: ``engine-irrbb-api`` 1.1.0, module simulation. Every function with randomness takes a seed and uses
``numpy.random.default_rng(seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskvalidation.irrbb.curves import dns_loadings, forecast_dns_factors
from riskvalidation.lgd_ead._checks import same_length, scalar, vec


@dataclass(frozen=True)
class NiiPath:
    monthly_nii: np.ndarray      # (M,) money per month
    total: float                 # money over the horizon


def simulate_nii(balances: np.ndarray, sides: np.ndarray, rate0: np.ndarray, spread: np.ndarray, beta: np.ndarray,
                 reprice_months: np.ndarray, short_rate_path: np.ndarray, growth_annual: float = 0.0) -> NiiPath:
    """The monthly net interest income of a book over 12 to 36 months (dossier 05 D.6).

    Per item: ``balances >= 0``, ``sides`` +1 (asset) or -1 (liability), the current rate ``rate0``, and a reset rule
    ``spread + beta r`` (an asset indexed to the market has ``beta = 1``; an NMD deposit has the deposit beta). Months
    have the 0-based index ``m = 0..M-1`` and use ``short_rate_path[m]``. ``reprice_months = 0`` resets every month, so
    the rate of month ``m`` is ``spread + beta path[m]``; ``k >= 1`` keeps ``rate0`` for the months ``m < k`` and, from
    ``m = k`` on, uses ``spread + beta path[(m // k) k]``, so the item resets at ``m = k, 2k, ...`` and the rate set
    at a reset applies until the next one. The whole balance of an item resets together; a staggered ladder is a set of
    items. ``growth_annual`` is the balance-sheet assumption: 0 is the constant balance sheet (run-off replaced by
    identical new business, DIS70.5, dossier 01 H.1). Monthly NII is
    ``sum_i side_i balance_i (1 + g)^(m / 12) rate_im / 12`` with ``m = 0`` for the first month."""
    bal, side = vec(balances, "balances"), vec(sides, "sides")
    r0, sp, bt = vec(rate0, "rate0"), vec(spread, "spread"), vec(beta, "beta")
    rm = np.asarray(reprice_months)
    if rm.ndim != 1 or rm.size == 0 or rm.dtype.kind not in "iu":
        raise ValueError("reprice_months must be a non-empty one-dimensional integer array")
    path = vec(short_rate_path, "short_rate_path")
    same_length(balances=bal, sides=side, rate0=r0, spread=sp, beta=bt, reprice_months=rm.astype(float))
    g = scalar(growth_annual, "growth_annual")
    if np.any(bal < 0.0):
        raise ValueError("balances must be nonnegative")
    if not np.all(np.isin(side, (-1.0, 1.0))):
        raise ValueError("sides must be +1 (asset) or -1 (liability)")
    if np.any(rm < 0):
        raise ValueError("reprice_months must be nonnegative")
    if not 12 <= path.size <= 36:
        raise ValueError(f"short_rate_path must cover 12 to 36 months, got {path.size}")
    if g <= -1.0:
        raise ValueError("growth_annual must be above -1")
    months = np.arange(path.size)                                           # (M,) the 0-based month index
    step = np.where(rm == 0, 1, rm)                                         # (I,)
    last_reset = step[None, :] * (months[:, None] // step[None, :])         # (M, I): the month of the latest reset
    reset_rate = sp[None, :] + bt[None, :] * path[last_reset]
    rate = np.where(months[:, None] >= rm[None, :], reset_rate, r0[None, :])    # (M, I)
    growth = (1.0 + g) ** (months / 12.0)                                   # (M,)
    monthly = (rate * (side * bal)[None, :]).sum(axis=1) * growth / 12.0
    return NiiPath(monthly, float(monthly.sum()))


def _forward_curve(zero_rates: np.ndarray, tenors: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The instantaneous forward ``f(0, t) = R(t) + t R'(t)`` of the continuously compounded zero curve, linear in
    tenor with flat extrapolation (the slope of the segment to the right of a pillar)."""
    r = np.interp(t, tenors, zero_rates)
    seg = np.clip(np.searchsorted(tenors, t, side="right") - 1, 0, tenors.size - 1)
    slope = np.zeros_like(t)
    inside = (seg < tenors.size - 1) & (t >= tenors[0])
    nxt = np.minimum(seg + 1, tenors.size - 1)
    slope[inside] = ((zero_rates[nxt] - zero_rates[seg]) / (tenors[nxt] - tenors[seg]))[inside]
    return r + t * slope


def hw1f_scenarios(zero_rates: np.ndarray, tenors: np.ndarray, a: float, sigma: float, horizon_months: int,
                   n_paths: int, seed: int) -> np.ndarray:
    """(P, horizon_months + 1) short-rate paths of the Hull-White one-factor model fitted to today's curve (dossier 05
    C.4 and D.5): ``r(t) = x(t) + alpha(t)`` with ``dx = -a x dt + sigma dW``, ``x(0) = 0`` simulated exactly on the
    monthly grid, and ``alpha(t) = f(0, t) + sigma^2 / (2 a^2) (1 - e^(-a t))^2`` (Brigo and Mercurio 2006, eq. 3.34),
    ``f(0, t)`` the instantaneous forward of the initial curve. Antithetic: the second half of the paths mirrors the
    first (an odd ``n_paths``
    drops the last mirror). With ``sigma = 0`` every path is the initial instantaneous forward curve."""
    z0, ten = vec(zero_rates, "zero_rates"), vec(tenors, "tenors")
    same_length(zero_rates=z0, tenors=ten)
    if np.any(np.diff(ten) <= 0.0) or ten[0] <= 0.0:
        raise ValueError("tenors must be strictly increasing and positive")
    a_, s_ = scalar(a, "a"), scalar(sigma, "sigma")
    if a_ <= 0.0:
        raise ValueError("a (mean reversion) must be positive")
    if s_ < 0.0:
        raise ValueError("sigma must be nonnegative")
    h, p = int(horizon_months), int(n_paths)
    if h < 1 or p < 1:
        raise ValueError("horizon_months and n_paths must be at least 1")
    t = np.arange(h + 1) / 12.0
    alpha = _forward_curve(z0, ten, t) + s_**2 / (2.0 * a_**2) * (1.0 - np.exp(-a_ * t)) ** 2
    if s_ == 0.0:
        return np.tile(alpha, (p, 1))
    half = (p + 1) // 2
    z = np.random.default_rng(seed).standard_normal((half, h))
    z = np.vstack([z, -z])[:p]
    dt = 1.0 / 12.0
    decay = np.exp(-a_ * dt)
    vol = s_ * np.sqrt((1.0 - decay**2) / (2.0 * a_))
    x = np.zeros((p, h + 1))
    for k in range(h):
        x[:, k + 1] = x[:, k] * decay + vol * z[:, k]
    return x + alpha[None, :]


def dns_scenarios(factor_history: np.ndarray, maturities_months: np.ndarray, horizon_months: int, n_paths: int,
                  seed: int) -> np.ndarray:
    """(P, horizon_months, N) yields at the maturities: ``forecast_dns_factors`` through the Diebold-Li loadings. Only
    the history given is used (no look-ahead: the C14 bake passes the factors up to the valuation date)."""
    m = vec(maturities_months, "maturities_months")
    load = dns_loadings(m)
    factors = forecast_dns_factors(factor_history, horizon_months, n_paths, seed)
    return factors @ load.T


@dataclass(frozen=True)
class EarEvar:
    ear: float
    evar: float


def ear_evar(nii_paths: np.ndarray, eve_paths: np.ndarray, base_nii: float, base_eve: float,
             level: float = 0.99) -> EarEvar:
    """Loss-positive earnings at risk and economic value at risk: ``ear = base_nii - quantile(nii_paths, 1 - level)``
    and ``evar = base_eve - quantile(eve_paths, 1 - level)`` with linearly interpolated quantiles
    (``numpy.quantile``)."""
    nii, ev = vec(nii_paths, "nii_paths", allow_empty=True), vec(eve_paths, "eve_paths", allow_empty=True)
    same_length(nii_paths=nii, eve_paths=ev)
    if nii.size < 2:
        raise ValueError("at least 2 paths are needed")
    lv = scalar(level, "level")
    if not 0.0 < lv < 1.0:
        raise ValueError("level must lie in (0, 1)")
    q = 1.0 - lv
    return EarEvar(scalar(base_nii, "base_nii") - float(np.quantile(nii, q)),
                   scalar(base_eve, "base_eve") - float(np.quantile(ev, q)))



@dataclass(frozen=True)
class EarEvarMcError:
    level: float
    nii_quantile_se: float
    eve_quantile_se: float
    ear_se: float
    evar_se: float


def _batch_indices(n_paths: int, n_batches: int, antithetic: bool) -> np.ndarray:
    """(n_batches, m) path indices of the batches of ``ear_evar_mc_error``. Contiguous: the first
    ``n_batches * (P // n_batches)`` paths cut in order. Antithetic: path ``i + P // 2`` mirrors path ``i``; the first
    ``n_batches * (half // n_batches)`` pair indices are cut into ``n_batches`` contiguous groups and batch ``j`` holds
    the paths ``idx`` and ``idx + half`` of its pairs, so no pair is split and no pair is shared."""
    nb = n_batches
    if not antithetic:
        size = n_paths // nb
        return np.arange(nb * size).reshape(nb, size)
    if n_paths % 2:
        raise ValueError(f"antithetic batches need an even number of paths, got {n_paths}")
    half = n_paths // 2
    if half < nb:
        raise ValueError(f"antithetic batches need at least {2 * nb} paths ({nb} pairs) for {nb} batches, "
                         f"got {n_paths}")
    pairs = np.arange(nb * (half // nb)).reshape(nb, half // nb)
    return np.hstack([pairs, pairs + half])


def ear_evar_mc_error(nii_paths: np.ndarray, eve_paths: np.ndarray, level: float = 0.99,
                      n_batches: int = 20, antithetic: bool = False) -> EarEvarMcError:
    """The Monte Carlo standard error, in money, of the ``1 - level`` quantile of the NII paths and of the EVE paths,
    which is the Monte Carlo error of the EaR and of the EVaR reported by ``ear_evar`` at the same ``level``.

    Batch means: the first ``n_batches * (P // n_batches)`` paths, in the order they were simulated, are cut into
    ``n_batches`` contiguous batches, the ``1 - level`` quantile (linear interpolation, as ``ear_evar``) is taken in
    each batch, and the standard error is ``sd(batch quantiles, ddof=1) / sqrt(n_batches)``. The base NII and EVE are
    constants, so ``ear_se = nii_quantile_se`` and ``evar_se = eve_quantile_se``. Deterministic, no randomness; it
    needs ``n_batches >= 2`` and ``P >= 2 n_batches``. The relative noise of the estimate itself is about ``1 / sqrt(2
    (n_batches - 1))``."""
    nii, ev = vec(nii_paths, "nii_paths", allow_empty=True), vec(eve_paths, "eve_paths", allow_empty=True)
    same_length(nii_paths=nii, eve_paths=ev)
    lv = scalar(level, "level")
    if not 0.0 < lv < 1.0:
        raise ValueError("level must lie in (0, 1)")
    nb = int(n_batches)
    if nb != n_batches or nb < 2:
        raise ValueError("n_batches must be an integer of at least 2")
    if nii.size < 2 * nb:
        raise ValueError(f"at least {2 * nb} paths are needed for {nb} batches, got {nii.size}")
    idx = _batch_indices(nii.size, nb, bool(antithetic))

    def error(paths: np.ndarray) -> float:
        quantiles = np.quantile(paths[idx], 1.0 - lv, axis=1)
        return float(np.std(quantiles, ddof=1) / np.sqrt(nb))

    nii_se, eve_se = error(nii), error(ev)
    return EarEvarMcError(lv, nii_se, eve_se, nii_se, eve_se)

__all__ = [
    "EarEvar",
    "EarEvarMcError",
    "NiiPath",
    "dns_scenarios",
    "ear_evar",
    "ear_evar_mc_error",
    "hw1f_scenarios",
    "simulate_nii",
]
