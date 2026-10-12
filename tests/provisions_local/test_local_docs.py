"""RV-567: the local statutory provisions pages carry every table, their sources and their limits."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "provisions" / "local"
FIX = ROOT / "tests" / "fixtures" / "provisions_local"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _row(text: str, *labels: str) -> list[float]:
    """The trailing numeric cells of the first table row whose leading cells are ``labels``."""
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[: len(labels)] == list(labels):
            return [float(c) for c in cells[len(labels):]]
    raise AssertionError(f"no table row starting with {labels}")


def _has_row(text: str, label: str, wanted: list[float]) -> bool:
    """Whether some table row starts with ``label`` and ends with the numeric cells ``wanted``."""
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == label and len(cells) > len(wanted):
            try:
                if [float(c) for c in cells[len(cells) - len(wanted):]] == wanted:
                    return True
            except ValueError:
                continue
    return False


def test_local_docs_pages():
    b1 = (DOCS / "01_cmf-b1.md").read_text(encoding="utf-8")
    es = (DOCS / "02_bde-annex9.md").read_text(encoding="utf-8")
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    assert "(01_cmf-b1.md)" in index and "(02_bde-annex9.md)" in index
    for text in (b1, es):
        assert "$$" in text and "no compliance claim" in text
        assert "dossier 03" in text
    # the regimes are kept apart and the pending reform is not claimed
    assert "not IFRS 9 impairment" in b1 and "Chapter 5.5" in b1
    assert "not in force" in b1 and "August 2026" in b1
    assert "UNVERIFIED" in es and "Circular 1/2025" in es

    t = _load("b1_tables.json")
    # B-1 individual table and the default PP column
    for cat, cells in t["individual"]["rows"].items():
        assert _row(b1, cat) == cells, cat
    for cat, r in t["default"]["rows"].items():
        assert _has_row(b1, cat, [r["pp_pct"]]), cat
    # mortgage: 4 PVG buckets by PI, PDI, PE
    labels = {"<=40": "PVG <= 40%", "<=80": "40% < PVG <= 80%", "<=90": "80% < PVG <= 90%", ">90": "PVG > 90%"}
    for key, r in t["mortgage"]["rows"].items():
        for param in ("PI", "PDI", "PE"):
            assert _row(b1, labels[key], param) == r[param], (key, param)
    # leasing, student, commercial, consumer
    lease, com, stu = t["leasing"], t["commercial"], t["student"]
    days = ("0", "1-29", "30-59", "60-89", "Default")
    for k, bucket in enumerate(days):
        assert _has_row(b1, bucket, [lease["PI_real_estate"][k], lease["PI_non_real_estate"][k]]), bucket
        assert _has_row(b1, bucket, [com["PI_guarantee_ptvg_le_100"][k], com["PI_guarantee_ptvg_gt_100"][k],
                                     com["PI_no_guarantee"][k]]), bucket
    for label, re_, nre in zip(("PVB <= 40%", "40% < PVB <= 50%", "50% < PVB <= 80%", "80% < PVB <= 90%", "PVB > 90%"),
                               lease["PDI_real_estate"], lease["PDI_non_real_estate"], strict=True):
        assert _row(b1, label) == [re_, nre], label
    assert _row(b1, "CAE PI (%)") == stu["PI_due"]["CAE"] + [stu["PDI_due"]]
    assert _row(b1, "CORFO PI (%)") == stu["PI_due"]["CORFO"] + [stu["PDI_due"]]
    assert "PI CAE 41.6 and CORFO 16.5; PDI CAE 50.3 and CORFO 45.8" in b1
    for i, label in enumerate(("PTVG <= 60%", "60% < PTVG <= 75%", "75% < PTVG <= 90%", "90% < PTVG")):
        assert any(c.startswith(f"| With | {label} |") for c in b1.splitlines())
        assert _has_row(b1, "With", [com["PDI_with_guarantee"][i], com["PDI_with_guarantee_recourse"][i]])
    assert _has_row(b1, "Without", [com["PDI_no_guarantee"], com["PDI_no_guarantee_recourse"]])
    cons = t["consumer"]
    for label, row in zip(("0 to 7", "8 to 30", "31 to 60", "61 to 89"), cons["PI"], strict=True):
        assert _row(b1, label) == row, label
    assert _row(b1, "With residential mortgage in system") == cons["PDI_with_mortgage"]
    assert _row(b1, "Without residential mortgage in system") == cons["PDI_without_mortgage"]

    a = _load("annex9_tables.json")
    p144, p140 = es.split("## Point 140")[0], es.split("## Point 140")[1]
    for seg, cells in a["point_144"].items():
        assert _row(p144, seg) == cells, seg
    for seg, cells in a["point_140"].items():
        assert _row(p140, seg) == cells, seg
    for category, stage in a["stage_map"].items():
        assert f"`{stage}`" in es, category
