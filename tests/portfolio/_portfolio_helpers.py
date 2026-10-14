"""Shared builders of the portfolio tests (not collected: the name starts with an underscore)."""

from __future__ import annotations

import numpy as np

from riskvalidation.portfolio.portfolio import Portfolio

FIXTURE_GROUPS = (
    {"id": "large", "n_names": 10, "ead_each": 4000000.0, "pd": 0.0015, "lgd": 0.45, "loading": 0.4, "sector": "all"},
    {"id": "mid", "n_names": 50, "ead_each": 600000.0, "pd": 0.0030, "lgd": 0.45, "loading": 0.4, "sector": "all"},
    {"id": "small", "n_names": 1000, "ead_each": 30000.0, "pd": 0.0050, "lgd": 0.45, "loading": 0.4, "sector": "all"},
)


def identical(n: int, pd: float = 0.02, rho: float = 0.15, lgd: float = 1.0, ead: float = 1.0, gamma: float = 0.25,
              n_groups: int = 1) -> Portfolio:
    """n identical names loading on one factor with the loading sqrt(rho), split evenly in `n_groups` groups."""
    group = (np.arange(n) % n_groups).astype(np.int64)
    return Portfolio(
        ead=np.full(n, ead), pd=np.full(n, pd), lgd=np.full(n, lgd), loadings=np.full((n, 1), np.sqrt(rho)),
        factor_corr=np.eye(1), group=group, group_ids=tuple(f"g{i}" for i in range(n_groups)), lgd_gamma=gamma,
        currency="USD", label="illustrative",
    )
