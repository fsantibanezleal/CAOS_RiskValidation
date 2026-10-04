"""RV-005: the inputs hash is deterministic and sensitive to every input."""

from __future__ import annotations

import numpy as np

from riskvalidation.core.hashing import inputs_hash


def test_hash_deterministic_and_sensitive():
    a = np.array([0.1, 0.2, 0.3])
    assert inputs_hash(a, 5, "x") == inputs_hash(a.copy(), 5, "x")
    assert inputs_hash(a) != inputs_hash(np.array([0.1, 0.2, 0.30000001]))
    assert inputs_hash(a) != inputs_hash(a.astype(np.float32))  # dtype matters
    assert inputs_hash(a) != inputs_hash(a.reshape(3, 1))  # shape matters
    assert inputs_hash(a, 5) != inputs_hash(5, a)  # order matters
    assert inputs_hash({"b": 1, "a": 2}) == inputs_hash({"a": 2, "b": 1})  # mapping order does not
    big_endian = a.astype(">f8")
    assert inputs_hash(big_endian) == inputs_hash(a)  # canonical byte order
    assert len(inputs_hash()) == 64
