"""U16-E1 (decision D16-5): the public API of 1.0.0. Every subpackage states its ``__all__``; each name is importable,
has a docstring where it can carry one and is documented in ``docs/api-policy.md`` under its subpackage; ``import *``
imports exactly ``__all__``; the wheel carries ``py.typed``."""

from __future__ import annotations

import importlib
import importlib.resources
import inspect
import pkgutil
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import riskvalidation

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs" / "api-policy.md"
SUBPACKAGES = sorted(m.name for m in pkgutil.iter_modules(riskvalidation.__path__) if m.ispkg)


def _documented() -> dict[str, set[str]]:
    """The names listed under each ``## `riskvalidation.<sub>` `` heading of the policy (the top level is
    ``## `riskvalidation` ``)."""
    sections: dict[str, set[str]] = {}
    current = None
    for line in POLICY.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## `(riskvalidation(?:\.\w+)?)`", line)
        if m:
            current = m.group(1)
            sections[current] = set()
        elif line.startswith("## "):
            current = None
        elif current:
            sections[current] |= set(re.findall(r"`([A-Za-z_]\w*)`", line))
    return sections


def test_the_expected_subpackages_exist():
    assert len(SUBPACKAGES) == 21, SUBPACKAGES


@pytest.mark.parametrize("sub", SUBPACKAGES)
def test_every_subpackage_states_its_all(sub):
    mod = importlib.import_module(f"riskvalidation.{sub}")
    assert hasattr(mod, "__all__"), f"riskvalidation.{sub} defines no __all__"
    names = list(mod.__all__)
    assert len(names) == len(set(names)), f"duplicates in riskvalidation.{sub}.__all__"
    assert all(isinstance(n, str) and not n.startswith("_") for n in names)
    assert mod.__doc__ and mod.__doc__.strip(), f"riskvalidation.{sub} has no docstring"


@pytest.mark.parametrize("sub", ["", *SUBPACKAGES])
def test_every_public_name_is_importable_and_documented(sub):
    modname = f"riskvalidation.{sub}" if sub else "riskvalidation"
    mod = importlib.import_module(modname)
    listed = _documented().get(modname, set())
    undocumented, missing, no_doc = [], [], []
    for name in mod.__all__:
        if not hasattr(mod, name):  # a submodule named in __all__ is imported by `import *`, not by the package
            try:
                importlib.import_module(f"{modname}.{name}")
            except ImportError:
                missing.append(name)
                continue
        obj = getattr(mod, name)
        if name not in listed:
            undocumented.append(name)
        if (inspect.isroutine(obj) or inspect.isclass(obj) or inspect.ismodule(obj)) and not inspect.getdoc(obj):
            no_doc.append(name)
    assert not missing, f"{modname}.__all__ names that cannot be imported: {missing}"
    assert not no_doc, f"{modname} public names without a docstring: {no_doc}"
    assert not undocumented, f"{modname} public names missing from docs/api-policy.md: {undocumented}"


@pytest.mark.parametrize("sub", ["", *SUBPACKAGES])
def test_the_policy_lists_no_name_the_code_does_not_export(sub):
    modname = f"riskvalidation.{sub}" if sub else "riskvalidation"
    mod = importlib.import_module(modname)
    assert modname in _documented(), f"docs/api-policy.md has no section for {modname}"
    stale = _documented()[modname] - set(mod.__all__)
    assert not stale, f"docs/api-policy.md lists names {modname} does not export: {sorted(stale)}"


@pytest.mark.parametrize("sub", ["", *SUBPACKAGES])
def test_star_import_is_exactly_all(sub):
    modname = f"riskvalidation.{sub}" if sub else "riskvalidation"
    ns: dict = {}
    exec(f"from {modname} import *", ns)  # noqa: S102
    got = {k for k in ns if k != "__builtins__"}
    assert got == set(importlib.import_module(modname).__all__)


def test_py_typed_ships_with_the_package():
    assert (importlib.resources.files("riskvalidation") / "py.typed").is_file()


def test_the_built_wheel_carries_py_typed(tmp_path):
    """The wheel's own file list, from a build of the source tree. Run against an installed wheel outside the tree,
    the installed file list is read instead."""
    if not (ROOT / "pyproject.toml").is_file():
        from importlib.metadata import files

        assert any(str(f).replace("\\", "/").endswith("riskvalidation/py.typed") for f in files("riskvalidation"))
        return
    out = subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path),
                          str(ROOT)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    wheel = next(tmp_path.glob("riskvalidation-*.whl"))
    with zipfile.ZipFile(wheel) as z:
        assert "riskvalidation/py.typed" in z.namelist()
