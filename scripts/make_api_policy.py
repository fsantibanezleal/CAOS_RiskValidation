#!/usr/bin/env python3
"""Regenerate the public-surface listing of ``docs/api-policy.md`` from the ``__all__`` lists of the code.

The page keeps its prose by hand; the block between ``<!-- generated:begin -->`` and ``<!-- generated:end -->`` lists,
for the package and for each subpackage, the names of its ``__all__`` and, for each public module, the names of the
module's own ``__all__``. ``tests/test_public_api.py`` fails when the page and the code disagree in either direction.

Usage: python scripts/make_api_policy.py [--check]   (--check exits 1 when the page is out of date)
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

import riskvalidation

PAGE = Path(__file__).resolve().parents[1] / "docs" / "api-policy.md"
BEGIN, END = "<!-- generated:begin -->", "<!-- generated:end -->"

DESC = {
    "": "The top-level exports: the result record, the policy, the registry and the suite runner, the test catalogue and "
    "the version.",
    "aggregation": "Risk aggregation: the ladder from the simple sum to the worst-case bounds, the dependence band and "
    "model risk.",
    "ccr": "Counterparty credit risk: Hull-White paths, swap exposure, collateral, CVA, DVA and wrong-way risk.",
    "core": "The core contracts: the result record, the policy, the registry, hashing, the suite runner and the "
    "deprecation helpers.",
    "engines": "The reference engines: the one-factor model, most prudent estimation for low-default portfolios and PD "
    "curve calibration.",
    "fairness": "Fairness: group metrics, inference on selection rates, conditional disparity, replication and "
    "situation testing.",
    "generators": "The known-truth generators and the helpers that feed their data to the validation tests.",
    "governance": "Model governance: the scope test, the inventory, tiering, lifecycle, findings, review schedule, "
    "opinion, risk appetite metrics, the validation report and the reason codes. Every tier count, band and deadline is "
    "a stated convention.",
    "harness": "The size and power harness: simulation, rejection probabilities, exact critical values and the "
    "agreement rules.",
    "irrbb": "Interest-rate risk in the banking book: curves, shocks, slotting, EVE and NII, the behavioural models and "
    "the simulation.",
    "lgd_ead": "LGD and EAD: workout LGD, the LGD rungs, downturn LGD, CCF and the IRB floors.",
    "liquidity": "Liquidity: run-off, the counterbalancing capacity, ladders, scenarios, reverse stress and the "
    "depositor generator.",
    "market": "Market risk: VaR and ES estimators, the scoring and comparison of forecasts and the FRTB statistics.",
    "ml": "ML model validation: robustness, stability, explanations, effects, generalisation, conformal prediction, "
    "leakage and data quality.",
    "oprisk": "Operational risk: the loss distribution approach (distributions, aggregation, the single loss "
    "approximation, scenarios) and the standardised approach (SMA).",
    "policies": "The packaged policy data and the rules a policy entry names. The policy files are data; the module "
    "and the rule names are public.",
    "portfolio": "Portfolio credit capital: the loss distribution of a credit portfolio by the rungs of dossier 04 and "
    "its Euler contributions.",
    "provisions": "IFRS 9 expected credit loss and the local statutory provision regimes (CMF B-1 of Chile, Banco de "
    "Espana Circular 4/2017).",
    "regulatory": "The regulatory calculators: published formulas that turn parameters into capital, provisions or "
    "liquidity metrics. The IRB names are exported by name; the other calculators are modules.",
    "stress": "Macro stress testing: satellites, scenario paths, the capital engine, reverse stress and the scenario "
    "models.",
    "transitions": "Rating transitions: estimators, the embedding problem and the EM generator, intervals, mobility and "
    "the through-the-cycle portfolio.",
    "validation": "The validation tests, one TestResult function each, grouped by family modules; the family modules "
    "and the test functions are public.",
}


def _obj(mod, name):
    try:
        return getattr(mod, name)
    except AttributeError:
        return importlib.import_module(f"{mod.__name__}.{name}")


def _public_modules(package: str) -> list[str]:
    pkg = importlib.import_module(package)
    return sorted(
        f"{package}.{m.name}"
        for m in pkgutil.iter_modules(pkg.__path__)
        if not m.name.startswith("_") and not m.ispkg
    )


def _names(mod) -> list[str]:
    return list(getattr(mod, "__all__", []))


def _nested_packages(package: str) -> list[str]:
    """The public subpackages one level below ``package`` (``riskvalidation.validation.sequential``, 1.1.0)."""
    pkg = importlib.import_module(package)
    return sorted(f"{package}.{m.name}" for m in pkgutil.iter_modules(pkg.__path__) if m.ispkg and not m.name.startswith("_"))


def _module_sections(package: str, out: list[str]) -> None:
    """A ``###`` section for each public module of ``package``, then for each nested subpackage (its ``__all__``) and,
    recursively, its own modules."""
    for sub in _public_modules(package):
        names = _names(importlib.import_module(sub))
        out.append(f"### `{sub}`\n")
        out.append("Names: " + ", ".join(f"`{n}`" for n in names) + ".\n" if names else "The list is empty.\n")
    for nested in _nested_packages(package):
        names = _names(importlib.import_module(nested))
        out.append(f"### `{nested}`\n")
        out.append("Names, imported from the subpackage: " + ", ".join(f"`{n}`" for n in names) + ".\n" if names else "The list is empty.\n")
        _module_sections(nested, out)


def build() -> str:
    out: list[str] = []
    subs = sorted(m.name for m in pkgutil.iter_modules(riskvalidation.__path__) if m.ispkg)
    for s in ["", *subs]:
        name = f"riskvalidation.{s}" if s else "riskvalidation"
        mod = importlib.import_module(name)
        out.append(f"## `{name}`\n")
        out.append(DESC[s] + "\n")
        mods, other = [], []
        for n in _names(mod):
            (mods if inspect.ismodule(_obj(mod, n)) else other).append(n)
        if mods:
            out.append(f"Modules, imported by path (`from {name} import <module>`): "
                       + ", ".join(f"`{n}`" for n in mods) + ".\n")
        if other:
            out.append(f"Names, imported from the package (`from {name} import <name>`): "
                       + ", ".join(f"`{n}`" for n in other) + ".\n")
        if not (mods or other):
            out.append("The list is empty.\n")
        if s:
            _module_sections(name, out)
    return "\n".join(out)


def main() -> int:
    text = PAGE.read_text(encoding="utf-8")
    i, j = text.index(BEGIN), text.index(END)
    new = text[: i + len(BEGIN)] + "\n\n" + build() + "\n" + text[j:]
    if "--check" in sys.argv:
        if new != text:
            print("docs/api-policy.md is out of date: run python scripts/make_api_policy.py")
            return 1
        print("docs/api-policy.md is current")
        return 0
    PAGE.write_text(new, encoding="utf-8", newline="\n")
    print("wrote", PAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
