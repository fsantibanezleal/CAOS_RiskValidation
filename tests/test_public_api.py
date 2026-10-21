"""U16-E1 (decision D16-5, Q2 option c): the public API of 1.0.0 is the union of the ``__all__`` lists. Every public
module carries its own ``__all__``; each name is importable, has a docstring where it can carry one and is listed on
``docs/api-policy.md`` under its module, and the page lists no other name and no other module; ``import *`` imports
exactly ``__all__``; the wheel carries ``py.typed``."""

from __future__ import annotations

import importlib
import importlib.resources
import inspect
import os
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


def _public_modules() -> list[str]:
    names = ["riskvalidation"]
    for m in pkgutil.walk_packages(riskvalidation.__path__, "riskvalidation."):
        if not any(part.startswith("_") for part in m.name.split(".")[1:]):
            names.append(m.name)
    return sorted(names)


MODULES = _public_modules()
SUBPACKAGES = sorted(m.name for m in pkgutil.iter_modules(riskvalidation.__path__) if m.ispkg)
_HEADING = re.compile(r"^#{2,3} `(riskvalidation(?:\.\w+)*)`\s*$")


def _documented() -> dict[str, set[str]]:
    """The names listed under each ``## `module` `` or ``### `module` `` heading of the page."""
    sections: dict[str, set[str]] = {}
    current = None
    for line in POLICY.read_text(encoding="utf-8").splitlines():
        m = _HEADING.match(line)
        if m:
            current = m.group(1)
            sections[current] = set()
        elif line.startswith("#"):
            current = None
        elif current:
            sections[current] |= set(re.findall(r"`([A-Za-z_]\w*)`", line))
    return sections


def _obj(mod, name):
    if hasattr(mod, name):
        return getattr(mod, name)
    return importlib.import_module(f"{mod.__name__}.{name}")  # a submodule named in __all__


def test_the_expected_packages_exist():
    assert len(SUBPACKAGES) == 21, SUBPACKAGES
    assert len(MODULES) > 160


@pytest.mark.parametrize("modname", MODULES)
def test_every_public_module_states_its_own_all(modname):
    mod = importlib.import_module(modname)
    assert hasattr(mod, "__all__"), f"{modname} defines no __all__"
    names = list(mod.__all__)
    assert len(names) == len(set(names)), f"duplicates in {modname}.__all__"
    assert all(isinstance(n, str) and (not n.startswith("_") or n == "__version__") for n in names)
    assert mod.__doc__ and mod.__doc__.strip(), f"{modname} has no docstring"


@pytest.mark.parametrize("modname", MODULES)
def test_every_public_name_is_importable_documented_and_listed_on_the_page(modname):
    mod = importlib.import_module(modname)
    listed = _documented().get(modname)
    assert listed is not None, f"docs/api-policy.md has no section for {modname}"
    missing, no_doc = [], []
    for name in mod.__all__:
        try:
            obj = _obj(mod, name)
        except ImportError:
            missing.append(name)
            continue
        if (inspect.isroutine(obj) or inspect.isclass(obj) or inspect.ismodule(obj)) and not inspect.getdoc(obj):
            no_doc.append(name)
    assert not missing, f"{modname}.__all__ names that cannot be imported: {missing}"
    assert not no_doc, f"{modname} public names without a docstring: {no_doc}"
    not_listed = set(mod.__all__) - listed
    stale = listed - set(mod.__all__)
    assert not not_listed, f"{modname} public names missing from docs/api-policy.md: {sorted(not_listed)}"
    assert not stale, f"docs/api-policy.md lists names {modname} does not export: {sorted(stale)}"


def test_the_page_lists_exactly_the_public_modules():
    documented = set(_documented())
    assert documented == set(MODULES), (sorted(documented - set(MODULES)), sorted(set(MODULES) - documented))


def test_the_page_is_what_the_generator_writes():
    script = ROOT / "scripts" / "make_api_policy.py"
    if not script.is_file():
        pytest.skip("the generator is not in the tree this test runs from")
    out = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize("modname", MODULES)
def test_star_import_is_exactly_all(modname):
    ns: dict = {}
    exec(f"from {modname} import *", ns)  # noqa: S102
    got = {k for k in ns if k != "__builtins__"}
    assert got == set(importlib.import_module(modname).__all__)


def test_every_package_name_that_is_a_module_is_a_public_module():
    """A module named in a package's ``__all__`` is itself a public module with a section on the page."""
    for pkg in ["riskvalidation", *(f"riskvalidation.{s}" for s in SUBPACKAGES)]:
        mod = importlib.import_module(pkg)
        for name in mod.__all__:
            obj = _obj(mod, name)
            if inspect.ismodule(obj):
                assert obj.__name__ in MODULES, f"{pkg}.__all__ names the module {obj.__name__}, which has no section"


def test_py_typed_ships_with_the_package():
    assert (importlib.resources.files("riskvalidation") / "py.typed").is_file()
    pyproject = ROOT / "pyproject.toml"
    if pyproject.is_file():
        assert "py.typed" in pyproject.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not os.environ.get("RISKVALIDATION_BUILD_WHEEL"),
    reason="builds a wheel (build backend, network): set RISKVALIDATION_BUILD_WHEEL=1 (the release gate)",
)
def test_the_built_wheel_carries_py_typed(tmp_path):
    """The wheel's own file list, from a build of the source tree."""
    out = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path), str(ROOT)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    wheel = next(tmp_path.glob("riskvalidation-*.whl"))
    with zipfile.ZipFile(wheel) as z:
        assert "riskvalidation/py.typed" in z.namelist()


def test_the_installed_distribution_lists_py_typed():
    """Run against an installed wheel (outside the tree), the installed file list names py.typed."""
    if (ROOT / "pyproject.toml").is_file():
        pytest.skip("run from the source tree: the wheel test above covers the build")
    from importlib.metadata import files

    assert any(str(f).replace("\\", "/").endswith("riskvalidation/py.typed") for f in files("riskvalidation"))
