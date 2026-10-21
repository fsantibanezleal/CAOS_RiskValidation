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
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")

_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _parse(value: str, field: str) -> tuple[int, int, int]:
    m = _VERSION.match(value) if isinstance(value, str) else None
    if not m:
        raise ValueError(f"{field}: must be a version X.Y.Z, got {value!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _check_versions(since: str, removal: str) -> None:
    s_major = _parse(since, "since")[0]
    r_major, r_minor, r_patch = _parse(removal, "removal")
    if r_minor != 0 or r_patch != 0 or r_major <= s_major:
        raise ValueError(f"removal: must be a major version X.0.0 later than since={since}, got {removal!r}")


def deprecated(since: str, removal: str, use: str) -> Callable[[T], T]:
    """Mark a function or a class as deprecated.

    ``since`` is the version that deprecates it, ``removal`` the major version that removes it (``X.0.0``, later than
    ``since``) and ``use`` the replacement. Calling the function, or constructing the class, emits a
    ``DeprecationWarning`` attributed to the caller; the behaviour is unchanged. A ``removal`` that is not a later
    major, or an empty ``use``, is refused with the field named, because it would break the policy."""
    _check_versions(since, removal)
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


def deprecated_parameter(
    name: str, since: str, removal: str, use: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for a function whose keyword argument ``name`` is renamed ``use``.

    The wrapped function no longer lists ``name`` in its signature. A call that still passes ``name`` emits a
    ``DeprecationWarning`` (the version, the removing major and ``use``) and the value is passed as ``use``; passing
    both is a ``TypeError``. Versions are checked as for :func:`deprecated`."""
    _check_versions(since, removal)
    if not use.strip():
        raise ValueError("use: must name the replacement, got an empty value")

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if name in kwargs:
                if use in kwargs:
                    raise TypeError(f"{func.__qualname__}() got both {name!r} and {use!r}; {name!r} is deprecated")
                warnings.warn(
                    f"the argument {name!r} of {func.__qualname__} is deprecated since riskvalidation {since} and "
                    f"will be removed in {removal}; use {use!r} instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[use] = kwargs.pop(name)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def module_getattr(module_name: str, aliases: Mapping[str, tuple[Any, str, str, str]]) -> Callable[[str], Any]:
    """The ``__getattr__`` of a module that keeps deprecated module-level names working.

    ``aliases`` maps a name to ``(value, since, removal, use)``. Reading the name emits a ``DeprecationWarning`` and
    returns the value; any other name raises ``AttributeError`` as usual. Use it as
    ``__getattr__ = module_getattr(__name__, {...})`` in the module; versions are checked when the table is built."""
    for _, since, removal, _use in aliases.values():
        _check_versions(since, removal)

    def __getattr__(attr: str) -> Any:
        if attr in aliases:
            value, since, removal, use = aliases[attr]
            warnings.warn(
                f"{module_name}.{attr} is deprecated since riskvalidation {since} and will be removed in {removal}; "
                f"use {use} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return value
        raise AttributeError(f"module {module_name!r} has no attribute {attr!r}")

    return __getattr__


__all__ = ["deprecated", "deprecated_parameter", "module_getattr"]
