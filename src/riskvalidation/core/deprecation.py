"""The deprecation helper of the API policy (``docs/api-policy.md``, decision D16-5).

From 1.0.0 the engine follows semantic versioning: a removal or a signature change of a public name is a major
release. A public name is first deprecated in a minor release, where it keeps working and emits a
``DeprecationWarning`` naming the version that deprecated it, the major that removes it and the replacement; a
major removes it no earlier than the next major after that minor. Nothing is deprecated in 1.0.0; the helper exists
so that a 1.x minor can.

>>> @deprecated(since="1.2.0", removal="2.0.0", use="new_name")
... def old_name():
...     return 1
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _parse(value: str, field: str) -> tuple[int, int, int]:
    m = _VERSION.match(value) if isinstance(value, str) else None
    if not m:
        raise ValueError(f"{field}: must be a version X.Y.Z, got {value!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def deprecated(since: str, removal: str, use: str) -> Callable[[T], T]:
    """Mark a function or a class as deprecated.

    ``since`` is the version that deprecates it, ``removal`` the major version that removes it (``X.0.0``, later than
    ``since``) and ``use`` the replacement. Calling the function, or constructing the class, emits a
    ``DeprecationWarning`` attributed to the caller; the behaviour is unchanged. A ``removal`` that is not a later
    major, or an empty ``use``, is refused with the field named, because it would break the policy."""
    s_major = _parse(since, "since")[0]
    r_major, r_minor, r_patch = _parse(removal, "removal")
    if r_minor != 0 or r_patch != 0 or r_major <= s_major:
        raise ValueError(f"removal: must be a major version X.0.0 later than since={since}, got {removal!r}")
    if not isinstance(use, str) or not use.strip():
        raise ValueError("use: must name the replacement, got an empty value")

    def message(name: str) -> str:
        return (f"{name} is deprecated since riskvalidation {since} and will be removed in {removal}; "
                f"use {use} instead")

    def decorate(obj: Any) -> Any:
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
        note = f"\n\n.. deprecated:: {since}\n   Removed in {removal}; use {use} instead."
        if isinstance(obj, type):
            original = obj.__init__

            @functools.wraps(original)
            def init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(message(name), DeprecationWarning, stacklevel=2)
                original(self, *args, **kwargs)

            obj.__init__ = init  # type: ignore[misc,assignment]
            obj.__doc__ = (obj.__doc__ or "") + note
            return obj

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message(name), DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        wrapper.__doc__ = (obj.__doc__ or "") + note
        return wrapper

    return decorate


__all__ = ["deprecated"]
