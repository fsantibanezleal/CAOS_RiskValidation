"""RV-519: the ECL documentation pages carry their equations, sources, golden values and the limits of the claims."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "provisions" / "ecl"
NAMES = ("01_term-structures.md", "02_staging.md", "03_measurement.md", "04_overlays-and-reconciliation.md",
         "05_validation-battery.md")


def test_ecl_pages_carry_equations_sources_and_limits():
    index = (PAGES / "README.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"({name})" in index, name
        text = (PAGES / name).read_text(encoding="utf-8")
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv:\d{4}\.\d{4,5}|https://", text), f"{name} cites no source"
        assert "Symbols" in text or "symbols" in text, f"{name} defines no symbols"
    # the worked example of dossier 02 B.4 is stated with its printed values
    t1 = (PAGES / "01_term-structures.md").read_text(encoding="utf-8")
    assert "0.526" in t1 and "1.429" in t1 and "7.342" in t1
    t3 = (PAGES / "03_measurement.md").read_text(encoding="utf-8")
    assert "2.681" in t3 and "1.800" in t3
    # the limits of section I.5 and the status of the threefold backstop
    limits = (PAGES / "05_validation-battery.md").read_text(encoding="utf-8") + index
    assert "supervisory expectation" in limits and "not IFRS 9" in limits
    assert "30 days past due" in limits and "90 days past due" in limits
    assert "not an IFRS 9 compliant" in limits or "no compliance claim" in limits
    assert "Contraste internal gate" in (PAGES / "05_validation-battery.md").read_text(encoding="utf-8")
