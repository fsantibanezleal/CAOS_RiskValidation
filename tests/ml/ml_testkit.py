"""Shared helpers of the U14-E1 tests: the row check against `common/test-result` and small data builders."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from riskvalidation.core.result import Light, TestResult

# The fields and types of docs/contracts/schemas/common/test-result.schema.json (Contraste, engine-ml-api 1.0.0).
ROW_FIELDS = ("test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
              "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras")
LIGHTS = ("green", "amber", "red", "not_evaluated", "error")


def _num_or_null(v: Any) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))


def assert_row(result: TestResult, test_id: str) -> dict[str, Any]:
    """The serialised row has exactly the schema's fields and types, the id prefix `ml.`, and filled text fields."""
    row = result.to_dict()
    assert set(row) == set(ROW_FIELDS), sorted(set(row) ^ set(ROW_FIELDS))
    assert row["test_id"] == test_id and row["test_id"].startswith("ml.")
    assert row["light"] in LIGHTS
    for f in ("statistic", "p_value", "metric", "alpha_amber", "alpha_red"):
        assert _num_or_null(row[f]), (f, row[f])
    if row["p_value"] is not None:
        assert 0.0 <= row["p_value"] <= 1.0
    for f in ("n", "n_events"):
        assert row[f] is None or (isinstance(row[f], int) and not isinstance(row[f], bool) and row[f] >= 0), (f, row[f])
    for f in ("h0", "alternative", "reference", "policy_version"):
        assert isinstance(row[f], str) and row[f].strip(), f
    assert isinstance(row["extras"], dict)
    return row


def assert_not_evaluated(result: TestResult, test_id: str) -> None:
    """A not_evaluated row carries its reason and no number once serialised."""
    row = assert_row(result, test_id)
    assert result.light is Light.NOT_EVALUATED
    assert row["statistic"] is None and row["p_value"] is None and row["metric"] is None
    assert row["notes"].startswith("not evaluated:") and len(row["notes"]) > len("not evaluated:") + 3


def logistic_xy(n: int = 2000, p: int = 3, seed: int = 0, signal: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Features and a 0/1 label drawn from a logistic model whose first column carries the signal."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    eta = -0.5 + signal * x[:, 0] + 0.5 * signal * x[:, 1 % p]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.int64)
    return x, y


def sigmoid_predict(w: np.ndarray, b: float = -0.5):
    """A predict callable on arrays: expit(b + X @ w)."""
    w = np.asarray(w, dtype=float)

    def predict(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(b + np.asarray(x, dtype=float) @ w)))

    return predict
