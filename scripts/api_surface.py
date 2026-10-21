#!/usr/bin/env python3
"""Dump the public surface of an installed ``riskvalidation`` for the semantic-versioning gate.

- `names`: print `{module: [names of its __all__]}` of the installed package (the 1.0 surface, from the working tree).
- ``snapshot --only names.json``: print, for each module and name of ``names.json`` that exists in the installed
  package, its kind (``function``, ``class``, ``module`` or ``constant``) and, for a function or a class, its
  parameters as ``[name, kind, has_default]``. Run it in a venv with the previous release installed (the baseline),
  and commit the output as ``tests/api_surface_<version>.json``; ``tests/test_api_surface.py`` then fails when a name of
  the baseline disappears or its signature stops accepting what it accepted. Regenerate only at a major release.

Usage: python scripts/api_surface.py names > names.json
       python scripts/api_surface.py snapshot --only names.json > tests/api_surface_0_9_0.json
"""

from __future__ import annotations

import enum
import importlib
import inspect
import json
import pkgutil
import sys

import riskvalidation


def public_modules() -> list[str]:
    names = ["riskvalidation"]
    for m in pkgutil.walk_packages(riskvalidation.__path__, "riskvalidation."):
        if not any(part.startswith("_") for part in m.name.split(".")[1:]):
            names.append(m.name)
    return sorted(names)


def _get(mod, name):
    if hasattr(mod, name):
        return getattr(mod, name)
    return importlib.import_module(f"{mod.__name__}.{name}")


def kind_of(obj) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "constant"


def params_of(obj) -> list[list] | None:
    if kind_of(obj) not in ("function", "class"):
        return None
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        return None  # the signature of an enumeration differs between Python versions
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [[p.name, p.kind.name, p.default is not inspect.Parameter.empty] for p in sig.parameters.values()]


def names() -> dict[str, list[str]]:
    out = {}
    for modname in public_modules():
        mod = importlib.import_module(modname)
        out[modname] = list(getattr(mod, "__all__", []))
    return out


def snapshot(only: dict[str, list[str]]) -> dict:
    out: dict[str, dict] = {}
    for modname, wanted in only.items():
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        entry = {}
        for name in wanted:
            try:
                obj = _get(mod, name)
            except (ImportError, AttributeError):
                continue
            entry[name] = {"kind": kind_of(obj), "params": params_of(obj)}
        if entry:
            out[modname] = entry
    return {"baseline": getattr(riskvalidation, "__version__", "?"), "modules": out}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("names", "snapshot"):
        print(__doc__)
        return 2
    if sys.argv[1] == "names":
        json.dump(names(), sys.stdout, indent=1, sort_keys=True)
    else:
        path = sys.argv[sys.argv.index("--only") + 1]
        with open(path, encoding="utf-8") as fh:
            json.dump(snapshot(json.load(fh)), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
