"""A compact known-truth desk P&L generator for the PLA complement tests (the C13 design of CAOS_Contraste, restated
here so the engine tests do not depend on a product repository): a two-factor desk, a log-AR(1) volatility known at
the start of the day, pricing noise in the HPL, and a risk model that may use a volatility lagged by `lag` days."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

LOADINGS = np.array([0.006, -0.004])
RHO = 0.6
PHI = 0.98
VOL_SD = 0.35
NOISE_SHARE = 0.10
BURN = 300
N = 250


def pair(seed: int, lag: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One (hpl, rtpl) pair of 250 days; `lag` > 0 feeds the risk model a volatility that is `lag` days old."""
    rng = np.random.default_rng(seed)
    n = BURN + N
    e, w, u = rng.standard_normal(n), rng.standard_normal((n, 2)), rng.standard_normal(n)
    eta = VOL_SD * np.sqrt(1.0 - PHI ** 2)
    g = lfilter([eta], [1.0, -PHI], e)
    sigma = np.exp(-VOL_SD ** 2 + np.concatenate(([0.0], g[:-1])))
    sigma_m = sigma if lag == 0 else np.concatenate((np.full(lag, sigma[0]), sigma[:-lag]))
    chol = np.array([[1.0, 0.0], [RHO, np.sqrt(1.0 - RHO ** 2)]])
    x = (w @ chol.T) @ LOADINGS
    sd = float(np.sqrt(LOADINGS @ np.array([[1.0, RHO], [RHO, 1.0]]) @ LOADINGS))
    hpl = sigma * x + NOISE_SHARE * sd * u
    rtpl = sigma_m * x
    return hpl[BURN:], rtpl[BURN:]
