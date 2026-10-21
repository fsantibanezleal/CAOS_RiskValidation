"""The single loss approximation (SLA) of dossier 05 F.4.

For a subexponential severity ``F`` and a frequency with ``sum_n (1 + eps)^n P(N = n) < infinity``,
``1 - H(z) ~ E[N] (1 - F(z))`` as ``z -> infinity`` (Embrechts, Kluppelberg and Mikosch 1997, Theorem 1.3.9, as cited by
Shevchenko eq. 69), hence ``VaR_alpha[Z] ~ F^{-1}(1 - (1 - alpha) / E[N])`` (Shevchenko eq. 70; Bocker and Kluppelberg
2005, Risk 18(12):90-93). Shevchenko shows that it converges for heavy tails and is poor for LN(0, 1) at practical
quantiles.

The mean-corrected form ``F^{-1}(1 - (1 - alpha) / E[N]) + E[N] E[X]`` is the "commonly quoted corrected form" of
dossier 05 F.4, which was **not checked** against Bocker and Sprittulla, "Operational VaR: meaningful means" (Risk 2006)
or Degen (J. Operational Risk 5(4):3-17, 2010): it is UNVERIFIED, and its result says so. A severity with an infinite
mean (a GPD tail index of 1 or more) has no correction: the function raises. Contract: ``engine-oprisk-api`` 1.0.0,
module sla.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from riskvalidation.oprisk._checks import level_in_open_unit, positive
from riskvalidation.oprisk.distributions import severity_isf, severity_moment


def sla_var(frequency_mean: float, sev_law: str, sev_params: Sequence[float], level: float) -> float:
    """``F^{-1}(1 - (1 - level) / E[N])`` in EUR (Shevchenko eq. 70). Raises when ``E[N] <= 1 - level`` (the argument of
    ``F^{-1}`` would not be in (0, 1)); nondecreasing in ``E[N]`` and in ``level``."""
    en = positive(frequency_mean, "frequency_mean")
    level = level_in_open_unit(level)
    tail = (1.0 - level) / en
    if not tail < 1.0:
        raise ValueError(f"E[N] = {en} is not above 1 - level = {1.0 - level}: the single loss approximation is "
                         "undefined")
    return float(severity_isf(sev_law, sev_params, np.array([tail]))[0])


def sla_var_poisson_gpd(lam: float, xi: float, beta: float, level: float, location: float = 0.0) -> float:
    """``location + beta / xi ((lam / (1 - level))^xi - 1)``: the closed form for a Poisson count and a GPD severity
    (dossier 05 F.4); it scales like ``lam^xi``. Equal to ``sla_var(lam, "gpd", (xi, beta, location), level)``."""
    lam = positive(lam, "lam")
    xi = positive(xi, "xi")
    beta = positive(beta, "beta")
    level = level_in_open_unit(level)
    ratio = lam / (1.0 - level)
    if not ratio > 1.0:
        raise ValueError(f"lam = {lam} is not above 1 - level = {1.0 - level}: the approximation is undefined")
    return float(location) + beta / xi * math.expm1(xi * math.log(ratio))


@dataclass(frozen=True)
class SlaResult:
    """The mean-corrected single loss approximation, in EUR: ``var = var_first_order + mean_correction``. ``status`` is
    always ``"UNVERIFIED"``: the corrected form was not checked against its primary text (dossier 05 F.4), and the
    result says so."""

    level: float
    var_first_order: float
    mean_correction: float
    var: float
    status: str


def sla_var_mean_corrected(frequency_mean: float, sev_law: str, sev_params: Sequence[float], level: float) -> SlaResult:
    """The first-order SLA plus ``E[N] E[X]`` (UNVERIFIED: dossier 05 F.4 did not check this form against Bocker and
    Sprittulla 2006 or Degen 2010; a changed form is a new minor version of the contract).

    Raises ``ValueError("infinite mean")`` when the severity mean is infinite (a GPD tail index of 1 or more): the
    correction does not exist, and the C17 bake then writes the rung as null with that reason.
    """
    first = sla_var(frequency_mean, sev_law, sev_params, level)
    mean = severity_moment(sev_law, sev_params, 1)
    if math.isinf(mean):
        raise ValueError("infinite mean: the severity has no mean (tail index of 1 or more), so the mean correction "
                         "does not exist")
    correction = float(frequency_mean) * mean
    return SlaResult(level=float(level), var_first_order=first, mean_correction=correction, var=first + correction,
                     status="UNVERIFIED")

__all__ = [
    "SlaResult",
    "sla_var",
    "sla_var_mean_corrected",
    "sla_var_poisson_gpd",
]
