"""Simulators shared by the market tests (imported from the test directory)."""

from __future__ import annotations

import numpy as np


def simulate_garch(n, omega=2e-6, alpha=0.08, beta=0.90, nu=None, mu=0.0003, seed=11, gamma=0.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 500) if nu is None else rng.standard_t(nu, n + 500) * np.sqrt((nu - 2) / nu)
    s2 = omega / (1 - alpha - beta - gamma / 2)
    out = np.empty(n + 500)
    for t in range(n + 500):
        e = np.sqrt(s2) * z[t]
        out[t] = mu + e
        s2 = omega + (alpha + gamma * (e < 0)) * e**2 + beta * s2
    return out[500:]
