"""The test catalogue: every validation test registers its identity, hypothesis and reference once.

The registry is what lets a suite, a report or a web page list the tests that exist, with the primary source
that defines each one, without importing anything but this package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DECISIONS = ("p_value", "metric")


@dataclass(frozen=True)
class TestSpec:
    """Identity of one registered test."""

    __test__ = False  # not a pytest test class

    test_id: str
    family: str
    name: str
    h0: str
    reference: str
    decision: str
    func: Callable[..., Any]

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {self.decision!r}")


_CATALOGUE: dict[str, TestSpec] = {}


def register(
    test_id: str, *, family: str, name: str, h0: str, reference: str, decision: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a test function under ``test_id``."""

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        existing = _CATALOGUE.get(test_id)
        if existing is not None and existing.func.__qualname__ != func.__qualname__:
            raise ValueError(f"test id {test_id!r} is already registered by {existing.func.__qualname__}")
        spec = TestSpec(test_id, family, name, h0, reference, decision, func)
        _CATALOGUE[test_id] = spec
        func.__rv_spec__ = spec  # type: ignore[attr-defined]
        return func

    return wrap


def spec(test_id: str) -> TestSpec:
    try:
        return _CATALOGUE[test_id]
    except KeyError as exc:
        raise KeyError(f"no registered test {test_id!r}") from exc


def catalogue(family: str | None = None) -> list[TestSpec]:
    """Registered tests, optionally of one family, sorted by id."""
    items = (s for s in _CATALOGUE.values() if family is None or s.family == family)
    return sorted(items, key=lambda s: s.test_id)
