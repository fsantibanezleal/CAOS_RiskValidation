"""RV-414: the documentation is part of the package. Every relative link resolves, every figure is used and exists,
every registered test is documented on a user-facing page, and the rating-transitions section is complete: an index
linked from the README and the docs map, and pages that carry their equations and their primary sources."""

from __future__ import annotations

import re
from pathlib import Path

from riskvalidation.core.registry import catalogue

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md")) + [ROOT / "README.md"]
LINK = re.compile(r"\]\(([^)#\s]+)(#[^)\s]*)?\)")
TRANSITION_PAGES = ("01_estimators.md", "02_embedding-and-em.md", "03_markov-tests.md",
                    "04_intervals-and-mobility.md", "05_ttc-portfolio.md")


def _text(page: Path) -> str:
    return page.read_text(encoding="utf-8")


def test_every_relative_link_resolves():
    broken = []
    for page in PAGES:
        for m in LINK.finditer(_text(page)):
            target = m.group(1)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # https:, mailto:
                continue
            if not (page.parent / target).resolve().exists():
                broken.append(f"{page.relative_to(ROOT).as_posix()} -> {target}")
    assert not broken, broken


def test_every_figure_exists_and_is_used():
    used = set()
    for page in PAGES:
        used |= set(re.findall(r"assets/([\w.-]+\.svg)", _text(page)))
    on_disk = {p.name for p in (DOCS / "assets").glob("*.svg")}
    assert used <= on_disk, sorted(used - on_disk)
    assert on_disk <= used, sorted(on_disk - used)


def test_every_registered_test_is_on_a_user_facing_page():
    """The design documents name every test too; a test counts as documented only on a page a user reads."""
    text = "\n".join(_text(p) for p in PAGES if "design" not in p.relative_to(DOCS.parent).parts)
    missing = [s.test_id for s in catalogue() if s.family != "selftest" and f"`{s.test_id}`" not in text]
    assert not missing, missing


def test_transitions_section():
    index = _text(DOCS / "transitions.md")
    for name in TRANSITION_PAGES:
        assert f"(transitions/{name})" in index, name
        text = _text(DOCS / "transitions" / name)
        assert "$$" in text, f"{name} has no display equation"
        assert re.search(r"doi:10\.\d{4,}|arXiv \d{4}\.\d{4,5}", text), f"{name} cites no primary source"
    assert "(transitions.md)" in _text(DOCS / "README.md")
    assert "(docs/transitions.md)" in _text(ROOT / "README.md")
    assert "(generators/05_rating-paths.md)" in _text(DOCS / "generators.md")
    # the three published results the section reproduces are stated with their printed values
    assert "0.116900" in _text(DOCS / "transitions" / "02_embedding-and-em.md")
    assert "1,332.56" in _text(DOCS / "transitions" / "04_intervals-and-mobility.md")
    assert "0.722%" in _text(DOCS / "transitions" / "05_ttc-portfolio.md")
