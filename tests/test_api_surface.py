"""U16-E1 (review M3): the semantic-versioning promise, enforced against the previous release.

``tests/api_surface_0_9_0.json`` is the 0.9.0 surface restricted to the names the 1.0 ``__all__`` lists carry (generated
by ``scripts/api_surface.py snapshot`` in a venv with 0.9.0 installed). Every name in it must still exist in the same
kind, and a function or a class must still accept every call the baseline accepted: its positional parameters keep
their names and order, none is removed, a new parameter has a default. The BL-082 move is the one allowed break. A major
release regenerates the snapshot."""

from __future__ import annotations

import enum
import importlib
import inspect
import json
from pathlib import Path

import pytest

import riskvalidation


def kind_of(obj) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    return "function" if callable(obj) else "constant"


_POS = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")


def compatible(old: list | None, obj) -> str | None:
    """Why ``obj`` no longer accepts every call the baseline parameters ``old`` accepted, or ``None``."""
    if old is None or (inspect.isclass(obj) and issubclass(obj, enum.Enum)):
        return None  # an enumeration's signature differs between Python versions
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "the signature can no longer be read"
    new = {p.name: p for p in sig.parameters.values()}
    new_pos = [p.name for p in sig.parameters.values() if p.kind.name in _POS]
    old_pos = [n for n, k, _ in old if k in _POS]
    if new_pos[: len(old_pos)] != old_pos:
        return f"positional parameters {old_pos} are not the start of {new_pos}"
    for name, kind, has_default in old:
        if kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            if not any(p.kind.name == kind for p in new.values()):
                return f"the {kind.lower()} parameter was removed"
            continue
        if name not in new:
            return f"parameter {name} was removed"
        if kind == "POSITIONAL_OR_KEYWORD" and new[name].kind.name not in _POS:
            return f"parameter {name} is no longer accepted by position"
        if has_default and new[name].default is inspect.Parameter.empty:
            return f"parameter {name} lost its default"
    old_names = {n for n, _, _ in old}
    for name, p in new.items():
        if name not in old_names and p.default is inspect.Parameter.empty and p.kind.name not in (
            "VAR_POSITIONAL", "VAR_KEYWORD"
        ):
            return f"new parameter {name} has no default"
    return None


SNAPSHOTS = {
    p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(__file__).parent.glob("api_surface_*.json"))
}
#: names that changed on purpose before the freeze: BL-082, the stress-chain probabilities move to the configuration
ALLOWED = {
    "api_surface_0_9_0.json": {
        "riskvalidation.liquidity.deposits.DepositorConfig",
        "riskvalidation.liquidity.deposits.DepositorSegment",
    },
}
ITEMS = [
    (name, mod, n) for name, snap in SNAPSHOTS.items() for mod, names in snap["modules"].items() for n in names
]
CASES = [(name, mod) for name, snap in SNAPSHOTS.items() for mod in sorted(snap["modules"])]


def _major() -> int:
    return int(riskvalidation.__version__.split(".")[0])


def _current_names() -> set[tuple[str, str]]:
    out = set()
    import pkgutil

    for m in [riskvalidation, *[importlib.import_module(x.name) for x in pkgutil.walk_packages(
            riskvalidation.__path__, "riskvalidation.") if not any(p.startswith("_") for p in x.name.split(".")[1:])]]:
        out |= {(m.__name__, n) for n in getattr(m, "__all__", [])}
    return out


@pytest.mark.skipif(_major() >= 2, reason="a major release may break the previous surface: regenerate the snapshot")
@pytest.mark.parametrize("snap,modname", CASES, ids=[f"{s}:{m}" for s, m in CASES])
def test_no_name_of_the_baseline_disappears_or_loses_a_call(snap, modname):
    mod = importlib.import_module(modname)
    problems = []
    for name, old in SNAPSHOTS[snap]["modules"][modname].items():
        try:
            obj = getattr(mod, name) if hasattr(mod, name) else importlib.import_module(f"{modname}.{name}")
        except ImportError:
            problems.append(f"{name}: removed")
            continue
        if name not in getattr(mod, "__all__", []):
            problems.append(f"{name}: no longer in {modname}.__all__ (it is out of the public surface)")
        if kind_of(obj) != old["kind"]:
            problems.append(f"{name}: was a {old['kind']}, is a {kind_of(obj)}")
            continue
        if f"{modname}.{name}" in ALLOWED.get(snap, set()):
            continue
        why = compatible(old["params"], obj)
        if why:
            problems.append(f"{name}: {why}")
    assert not problems, problems


def test_the_snapshots_are_the_baselines_and_not_empty():
    assert {s["baseline"] for s in SNAPSHOTS.values()} == {"0.9.0", "1.0.0"}
    assert len(ITEMS) > 2500
    assert "riskvalidation.liquidity.deposits" in SNAPSHOTS["api_surface_0_9_0.json"]["modules"]


def test_the_newest_snapshot_covers_every_public_name_of_the_1_0_line():
    """No public name escapes the gate: while the version is 1.0.x (or earlier) every name of every ``__all__`` is in
    the 1.0.0 snapshot, and from 1.1 a new name may be added (the removal direction stays). The gate cannot stay on
    0.9.0: the newest baseline's major is at least the current major."""
    snap = SNAPSHOTS["api_surface_1_0_0.json"]
    in_snapshot = {(m, n) for m, names in snap["modules"].items() for n in names}
    missing = sorted(_current_names() - in_snapshot)
    parts = riskvalidation.__version__.split(".")
    if _major() < 1 or (parts[0] == "1" and parts[1] == "0"):
        assert not missing, f"public names outside the 1.0.0 snapshot (regenerate it): {missing[:20]}"
    assert max(int(s["baseline"].split(".")[0]) for s in SNAPSHOTS.values()) >= _major()


def test_the_gate_catches_a_removal_a_reorder_and_a_new_required_parameter():
    def base(a, b=1, *, c=2):  # the baseline: a, b, c
        return a, b, c

    old = [["a", "POSITIONAL_OR_KEYWORD", False], ["b", "POSITIONAL_OR_KEYWORD", True], ["c", "KEYWORD_ONLY", True]]
    assert compatible(old, base) is None

    def removed(a, *, c=2):
        return a, c

    def reordered(b, a, *, c=2):
        return a, b, c

    def lost_default(a, b, *, c=2):
        return a, b, c

    def new_required(a, b=1, *, c=2, d):
        return a, b, c, d

    def new_optional(a, b=1, *, c=2, d=3):
        return a, b, c, d

    assert compatible(old, removed)  # b is gone
    assert "not the start" in compatible(old, reordered)
    assert "lost its default" in compatible(old, lost_default)
    assert "no default" in compatible(old, new_required)
    assert compatible(old, new_optional) is None
