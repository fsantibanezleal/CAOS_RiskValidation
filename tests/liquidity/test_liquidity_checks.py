"""RV-1101: the input checks of `riskvalidation.liquidity._checks` (contract `engine-liquidity-api`)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.liquidity import _checks as c


def test_check_nonnegative_stock():
    out = c.check_nonnegative_stock([0.0, 5.0], "balance")
    assert out.dtype == np.float64 and out.tolist() == [0.0, 5.0]
    assert c.check_nonnegative_stock(3.0, "balance").tolist() == [3.0]  # a scalar becomes a one-element array
    with pytest.raises(ValueError, match="^balance: must be non-negative"):
        c.check_nonnegative_stock([-1.0], "balance")
    for bad in ([np.nan], [np.inf], [1.0, -np.inf]):
        with pytest.raises(ValueError, match="^stock: "):
            c.check_nonnegative_stock(bad, "stock")
    with pytest.raises(ValueError, match="^stock: .*empty"):
        c.check_nonnegative_stock([], "stock")
    # the input is never mutated and the result is a new array
    x = np.array([1.0, 2.0])
    y = c.check_nonnegative_stock(x, "x")
    y[0] = 99.0
    assert x[0] == 1.0
    # a matrix of money is allowed (any shape)
    assert c.check_nonnegative_stock(np.ones((2, 3)), "m").shape == (2, 3)


def test_check_unit_share():
    assert c.check_unit_share([0.0, 1.0], "q").tolist() == [0.0, 1.0]  # a rate of exactly 0 or 1 is legitimate
    with pytest.raises(ValueError, match="^q: "):
        c.check_unit_share(1.0, "q", closed=False)
    with pytest.raises(ValueError, match="^q: "):
        c.check_unit_share(0.0, "q", closed=False)
    with pytest.raises(ValueError, match="^q: "):
        c.check_unit_share(1.2, "q")
    with pytest.raises(ValueError, match="^q: "):
        c.check_unit_share([0.5, -0.01], "q")
    with pytest.raises(ValueError, match="^q: "):
        c.check_unit_share(np.nan, "q")
    assert c.check_unit_share(0.5, "q", closed=False).tolist() == [0.5]


def test_check_weights_sum_to_one():
    assert c.check_weights_sum_to_one(np.array([0.057, 0.864, 0.079]), "w").shape == (3,)
    with pytest.raises(ValueError, match="^w: .*sum"):
        c.check_weights_sum_to_one(np.array([0.5, 0.4]), "w")
    with pytest.raises(ValueError, match="^w: "):
        c.check_weights_sum_to_one(np.array([1.5, -0.5]), "w")
    # the tolerance is the caller's
    c.check_weights_sum_to_one(np.array([0.5, 0.4999]), "w", tol=1e-3)
    with pytest.raises(ValueError):
        c.check_weights_sum_to_one(np.array([0.5, 0.4999]), "w", tol=1e-9)


def test_check_increasing_grid():
    assert c.check_increasing_grid(np.array([1, 2, 5]), "g").tolist() == [1.0, 2.0, 5.0]
    with pytest.raises(ValueError, match=r"^g: .*index 1"):
        c.check_increasing_grid(np.array([1, 1, 2]), "g")
    with pytest.raises(ValueError, match=r"^g: .*index 2"):
        c.check_increasing_grid(np.array([1, 3, 2]), "g")
    with pytest.raises(ValueError, match="^g: "):
        c.check_increasing_grid(np.array([0.5, 1.0]), "g", minimum=1.0)
    c.check_increasing_grid(np.array([1.0, 1.5]), "g", minimum=1.0)
    with pytest.raises(ValueError, match="^g: "):
        c.check_increasing_grid(np.array([]), "g")
