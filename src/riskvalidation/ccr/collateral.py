"""Collateral, the margin period of risk and the dynamic initial margin: ORE's credit support amount and delivery amount
equations with the close-out grid approach and a sticky-date close-out (addendum R3, ORE ``xva.tex`` at tag 1.8.17.0),
the CRE53.24 to CRE53.26 floors, and ORE's regression dynamic initial margin (addendum R2).

The Andersen, Pykhtin and Sokol (2017) MPoR timelines are not modelled. The dynamic initial margin is a regression
proxy, never SIMM: the SIMM methodology is published under a notice that restricts its use to a market participant's own
transactions, so its risk weights are not reimplemented here. Contract: ``engine-ccr-api`` 1.0.0, module collateral.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from riskvalidation.lgd_ead._checks import mat, scalar, vec

MIN_QUANTILE = 0.5
"""The lowest confidence level of the dynamic initial margin: below it the factor ``Phi^-1(q)`` is negative and the DIM
could not be non-negative."""


def _nonnegative(x: float, name: str, *, allow_inf: bool = False) -> float:
    v = float(x)
    if math.isnan(v) or (math.isinf(v) and not allow_inf) or v < 0.0:
        raise ValueError(f"{name} must be nonnegative" + (" (infinity allowed)" if allow_inf else ""))
    return v


def credit_support_amount(npv: np.ndarray, independent_amount: float, threshold_receive: float,
                          threshold_pay: float) -> np.ndarray:
    """With ``x = NPV + IA``: ``max(0, x - Th_rec)`` if ``x >= 0``,
    ``min(0, x + Th_pay)`` if ``x < 0`` (ORE ``eq:CSA``). With ``NPV = 0``,
    ``Th_rec = 0`` and ``IA > 0`` the amount equals ``IA``: the independent amount is an offset to
    the threshold. A threshold may be infinite (no margin is called on that side)."""
    n = np.array(npv, dtype=float)
    if not np.all(np.isfinite(n)):
        raise ValueError("npv contains non-finite values")
    ia = scalar(independent_amount, "independent_amount")
    th_r = _nonnegative(threshold_receive, "threshold_receive", allow_inf=True)
    th_p = _nonnegative(threshold_pay, "threshold_pay", allow_inf=True)
    x = n + ia
    return np.where(x >= 0.0, np.maximum(0.0, x - th_r), np.minimum(0.0, x + th_p))


def delivery_amount(csa: np.ndarray, collateral: np.ndarray, outstanding_calls: np.ndarray, mta: float) -> np.ndarray:
    """``Delta = CSA - C - M`` and ``D = Delta`` if
    ``|Delta| >= MTA``, else 0 (ORE ``eq:DA``); with ``MTA = 0`` it is ``Delta``."""
    c, k, m = np.array(csa, dtype=float), np.array(collateral, dtype=float), np.array(outstanding_calls, dtype=float)
    if not (c.shape == k.shape == m.shape):
        raise ValueError(
            f"csa, collateral and outstanding_calls must have the same shape, got {c.shape}, {k.shape}, {m.shape}")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(k)) and np.all(np.isfinite(m))):
        raise ValueError("csa, collateral and outstanding_calls must be finite")
    t = _nonnegative(mta, "mta")
    delta = c - k - m
    return np.where(np.abs(delta) >= t, delta, 0.0)


def mpor_floor(repo_only: bool = False, n_trades: int = 0, illiquid: bool = False,
               disputes_over_two: bool = False) -> int:
    """The supervisory floor in business days: 5 for repo-style netting sets and 10 otherwise (CRE53.24); 20 when the
    number of trades exceeds 5000 or the collateral is illiquid or an OTC derivative cannot be easily replaced
    (``illiquid`` covers both); more than two disputes longer than the MPoR over two quarters double the floor that
    applies (CRE53.25)."""
    if int(n_trades) != n_trades or n_trades < 0:
        raise ValueError("n_trades must be a nonnegative integer")
    floor = 5 if repo_only else 10
    if n_trades > 5000 or illiquid:
        floor = 20
    return 2 * floor if disputes_over_two else floor


def mpor_with_remargining(floor: int, remargining_days: int) -> int:
    """CRE53.26: ``F + N - 1`` for re-margining every ``N`` days; ``N = 1`` gives ``F``."""
    if int(floor) != floor or floor < 1:
        raise ValueError("floor must be a positive integer")
    if int(remargining_days) != remargining_days or remargining_days < 1:
        raise ValueError("remargining_days must be a positive integer")
    return int(floor) + int(remargining_days) - 1


def collateralised_exposure(value_closeout: np.ndarray, collateral: np.ndarray) -> np.ndarray:
    """``max(V(t) - C(t), 0)`` (ORE; Hull and White 2012 equation 4): ``V`` the netting-set value on the
    close-out grid and ``C`` the balance determined at the default date minus the MPoR. Positive even with zero
    thresholds and minimum transfer amounts because of the lag, and tending to zero with the MPoR."""
    v, c = np.array(value_closeout, dtype=float), np.array(collateral, dtype=float)
    if v.shape != c.shape:
        raise ValueError(f"value_closeout and collateral must have the same shape, got {v.shape} and {c.shape}")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(c))):
        raise ValueError("value_closeout and collateral must be finite")
    return np.maximum(v - c, 0.0)


def variation_margin_exposure(values_main: np.ndarray, values_closeout: np.ndarray, independent_amount: float,
                              threshold_receive: float, threshold_pay: float,
                              mta: float) -> tuple[np.ndarray, np.ndarray]:
    """``(exposure, collateral_balance)`` each (P, K). The balance follows ``credit_support_amount`` and
    ``delivery_amount`` on the main grid, date by date from a zero balance with no outstanding calls (each call settles
    at its date, the lag being the close-out grid); the exposure is
    ``collateralised_exposure(values_closeout, balance)``. The close-out values are those of the sticky-date
    approach (a swap with no flow in the window needs no spike).

    A netting set with no CSA (both thresholds infinite) is uncollateralised: ORE takes the default value on the main
    grid, so the exposure is ``max(values_main, 0)`` and the balance 0."""
    vm, vc = mat(values_main, "values_main"), mat(values_closeout, "values_closeout")
    if vm.shape != vc.shape:
        raise ValueError(f"values_closeout must have the shape of values_main {vm.shape}, got {vc.shape}")
    ia = scalar(independent_amount, "independent_amount")
    th_r = _nonnegative(threshold_receive, "threshold_receive", allow_inf=True)
    th_p = _nonnegative(threshold_pay, "threshold_pay", allow_inf=True)
    t = _nonnegative(mta, "mta")
    if math.isinf(th_r) and math.isinf(th_p):
        return np.maximum(vm, 0.0), np.zeros_like(vm)
    balance = np.zeros(vm.shape[0])
    held = np.empty_like(vm)
    zero = np.zeros_like(balance)
    for k in range(vm.shape[1]):
        csa = credit_support_amount(vm[:, k], ia, th_r, th_p)
        balance = balance + delivery_amount(csa, balance, zero, t)
        held[:, k] = balance
    return collateralised_exposure(vc, held), held


def dynamic_initial_margin(values_now: np.ndarray, values_mpor: np.ndarray, flows: np.ndarray, quantile: float,
                           horizon_calendar_days: int, mpor_calendar_days: int, order: int = 2,
                           regressor: np.ndarray | None = None) -> np.ndarray:
    """(P,) the dynamic initial margin per path (ORE ``xva.tex`` and ``dimregressioncalculator.cpp``, addendum R2), a
    regression proxy and never SIMM. ``X = V(t + MPoR) + F - V(t)``; ``E[X^2]``
    is the regression of ``X^2`` on a polynomial of the regressor of the given order (order 0 is the sample variance;
    the default regressor is ``values_now``); ``S = sqrt(max(E[X^2] - E[X]^2, 0))`` with
    ``E[X]`` the sample mean; and
    ``DIM = Phi^-1(quantile) S sqrt(horizon_calendar_days / mpor_calendar_days)``, both in
    CALENDAR days as ORE computes ``horizonCalendarDays / mporCalendarDays`` (10 business days are 14 calendar
    days, 20 are 28). Zero where ``S`` is zero; non-negative, so the quantile must be at least 0.5."""
    now, nxt, f = vec(values_now, "values_now"), vec(values_mpor, "values_mpor"), vec(flows, "flows")
    for name, arr in (("values_mpor", nxt), ("flows", f)):
        if arr.size != now.size:
            raise ValueError(f"{name} must have the length of values_now ({now.size}), got {arr.size}")
    q = scalar(quantile, "quantile")
    if not MIN_QUANTILE <= q < 1.0:
        raise ValueError(f"quantile must lie in [{MIN_QUANTILE}, 1), got {q}")
    h, m = scalar(horizon_calendar_days, "horizon_calendar_days"), scalar(mpor_calendar_days, "mpor_calendar_days")
    if h <= 0.0:
        raise ValueError("horizon_calendar_days must be positive")
    if m <= 0.0:
        raise ValueError("mpor_calendar_days must be positive")
    if int(order) != order or order < 0:
        raise ValueError("order must be a nonnegative integer")
    reg = now if regressor is None else vec(regressor, "regressor")
    if reg.size != now.size:
        raise ValueError(f"regressor must have the length of values_now ({now.size}), got {reg.size}")
    x = nxt + f - now
    sd = float(np.std(reg))
    z = (reg - float(np.mean(reg))) / sd if sd > 0.0 else np.zeros_like(reg)
    design = np.vander(z, int(order) + 1, increasing=True)
    coef = np.linalg.lstsq(design, x**2, rcond=None)[0]
    second = design @ coef
    s = np.sqrt(np.maximum(second - float(np.mean(x)) ** 2, 0.0))
    return float(norm.ppf(q)) * s * math.sqrt(h / m)


def apply_initial_margin(exposure_before_im: np.ndarray, dim: np.ndarray) -> np.ndarray:
    """``max(exposure - DIM, 0)`` (ORE ``nettedexposurecalculator.cpp``); at most the exposure, unchanged
    for ``DIM = 0``. IM applies only with variation margin ("don't apply initial margin without VM"): the
    caller passes the VM exposure."""
    e, d = np.array(exposure_before_im, dtype=float), np.array(dim, dtype=float)
    if e.shape != d.shape:
        raise ValueError(f"exposure_before_im and dim must have the same shape, got {e.shape} and {d.shape}")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(d))):
        raise ValueError("exposure_before_im and dim must be finite")
    if np.any(d < 0.0):
        raise ValueError("dim must be nonnegative")
    return np.maximum(e - d, 0.0)
