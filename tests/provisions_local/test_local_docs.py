"""RV-567: the documentation pages of the local statutory provisions carry every table, their sources and their limits."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "provisions" / "local"
FIX = ROOT / "tests" / "fixtures" / "provisions_local"


def _pct(x) -> str:
    """A value as the page prints it, in the dossier's decimal comma."""
    return str(x).replace(".", ",") if not isinstance(x, int) else str(x)


def test_local_docs_pages():
    b1 = (DOCS / "01_cmf-b1.md").read_text(encoding="utf-8")
    es = (DOCS / "02_bde-annex9.md").read_text(encoding="utf-8")
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    assert "(01_cmf-b1.md)" in index and "(02_bde-annex9.md)" in index
    for text in (b1, es):
        assert "$$" in text
        assert "no compliance claim" in text.lower()
    assert "not in force" in b1 and "August 2026" in b1
    assert "IFRS 9" in b1 and "5.5" in b1  # the regimes kept apart: Chapter 5.5 is not applied to loans
    assert "UNVERIFIED" in es and "Circular 1/2025" in es
    # every transcribed value of the individual table and of the mortgage table appears on the page
    b1_fix = json.loads((FIX / "b1_tables.json").read_text(encoding="utf-8"))
    for cat, (pi, pdi, pe) in b1_fix["individual"]["rows"].items():
        row = next(line for line in b1.splitlines() if line.startswith(f"| {cat} |"))
        assert all(f"{v:.5f}".replace(".", ",") in row or _pct(v) in row for v in (pi, pdi, pe)), (cat, row)
    for key, row in b1_fix["mortgage"]["rows"].items():
        for param in ("PI", "PDI", "PE"):
            for v in row[param]:
                assert f"{v:.4f}".replace(".", ",") in b1 or _pct(v) in b1, (key, param, v)
    # every cell of Annex 9 appears
    es_fix = json.loads((FIX / "annex9_tables.json").read_text(encoding="utf-8"))
    for seg, cells in es_fix["point_144"].items():
        row = next(line for line in es.splitlines() if line.startswith(f"| {seg} |") and "," in line)
        assert all(f"{v:.1f}".replace(".", ",") in row for v in cells), (seg, row)
    for seg, cells in es_fix["point_140"].items():
        rows = [line for line in es.splitlines() if re.match(rf"\| {re.escape(seg)} \|", line) and "|" in line]
        assert any(all(str(v) in line.split("|") or f" {v} " in line for v in cells) for line in rows), seg
