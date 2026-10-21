"""Deterministic fingerprints of test inputs.

A validation result is valid only for the exact inputs it tested. ``inputs_hash`` makes that checkable: equal
inputs give the same SHA-256 on every platform, and changing any value, shape or dtype changes it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _feed(h: Any, obj: Any) -> None:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        if arr.dtype.byteorder == ">":  # canonical little-endian bytes
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        h.update(b"ndarray")
        h.update(arr.dtype.str.encode())
        h.update(repr(arr.shape).encode())
        h.update(arr.tobytes())
    elif isinstance(obj, Mapping):
        h.update(b"mapping")
        for key in sorted(obj, key=str):
            h.update(str(key).encode())
            _feed(h, obj[key])
    elif isinstance(obj, (list, tuple)) or (isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))):
        h.update(b"sequence")
        h.update(str(len(obj)).encode())
        for item in obj:
            _feed(h, item)
    elif isinstance(obj, np.generic):
        _feed(h, np.asarray(obj))
    else:
        h.update(b"scalar")
        h.update(json.dumps(obj, sort_keys=True, default=repr).encode())


def inputs_hash(*objs: Any) -> str:
    """SHA-256 hex digest over the given inputs, in order."""
    h = hashlib.sha256()
    for obj in objs:
        _feed(h, obj)
    return h.hexdigest()

__all__ = [
    "inputs_hash",
]
