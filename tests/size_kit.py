"""The runner shared by the nine size and power suites of U16-E2 (BL-088): one study of `riskvalidation.generators.
size_studies` run with `harness.simulate`, the size judged by `within_size`, the power recorded, and the docs page held
to the numbers of the run.

Set SIZE_STUDIES_OUT to a path to write the measured rates as JSON (the figures of the docs pages)."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from riskvalidation.generators.size_studies import Study
from riskvalidation.harness import Rate, simulate, size_bound, within_size

ROOT = Path(__file__).resolve().parents[1]
OUT = os.environ.get("SIZE_STUDIES_OUT")

# tests whose measured size exceeds the bound in their null scenario: the page states the measured size. An entry here
# is a finding, not a pass: the suite asserts the excess is still there (so the entry leaves when the test is fixed).
DOCUMENTED_EXCESS = {
    "portfolio.mc_convergence": ROOT / "docs" / "portfolio" / "07_validation-tests.md",
}


@dataclass(frozen=True)
class Measured:
    """What one study measured: sizes and bounds at 5% and 1%, the power at 5%, the wall time."""

    study: Study
    size5: float
    size1: float
    bound5: float
    bound1: float
    power: float
    undefined: int
    seconds: float
    holds: bool
    rate5: Rate
    rate1: Rate

    def pct(self, value: float) -> str:
        return f"{100.0 * value:.2f}%"

    def numbers(self) -> list[str]:
        """The figures a docs page must state, as the page prints them."""
        return [self.pct(self.size5), self.pct(self.bound5), self.pct(self.size1), self.pct(self.bound1),
                self.pct(self.power), str(self.study.seed), str(self.study.power_seed), str(self.study.n_rep)]


def measure(study: Study) -> Measured:
    """Run the null and the alternative of a study (each its own seeded 4,000-repetition simulation)."""
    t0 = time.perf_counter()
    null = simulate(study.test_id, study.null, n_rep=study.n_rep, seed=study.seed, test_kwargs=study.test_kwargs)
    alt = simulate(study.test_id, study.alternative, n_rep=study.n_rep, seed=study.power_seed,
                   test_kwargs=study.test_kwargs)
    r5, r1 = null.rate(0.05), null.rate(0.01)
    out = Measured(
        study=study, size5=r5.rate, size1=r1.rate, bound5=size_bound(0.05, r5.n), bound1=size_bound(0.01, r1.n),
        power=alt.rate(0.05).rate, undefined=max(r5.undefined, alt.rate(0.05).undefined),
        seconds=time.perf_counter() - t0, holds=within_size(r5, 0.05) and within_size(r1, 0.01), rate5=r5, rate1=r1)
    _record(out, alt.rate(0.05).to_dict(), r5.to_dict(), r1.to_dict())
    return out


def _record(m: Measured, power: dict, size5: dict, size1: dict) -> None:
    if not OUT:
        return
    path = Path(OUT)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[f"{m.study.test_id} | {m.study.label}"] = {
        "size5": m.size5, "size1": m.size1, "bound5": m.bound5, "bound1": m.bound1, "power": m.power,
        "seconds": round(m.seconds, 1), "seed": m.study.seed, "power_seed": m.study.power_seed,
        "n_rep": m.study.n_rep, "within_size": m.holds, "undefined": m.undefined,
        "size5_detail": size5, "size1_detail": size1, "power_detail": power}
    path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")


def page_paragraph(page: Path, study: Study) -> str:
    """The paragraph of a docs page that states the study: it names the test id and the study's label."""
    text = page.read_text(encoding="utf-8")
    for para in re.split(r"\n\s*\n", text):
        if f"Size and power of `{study.test_id}`" in para and study.label in para:
            return para
    raise AssertionError(f"{page.name} has no 'Size and power of `{study.test_id}`' paragraph for '{study.label}'")


def check(study: Study, page: Path) -> Measured:
    """Measure a study and hold it to the contract of U16-E2: every repetition evaluated, the size within its bound
    at 5% and at 1% (or a documented excess that still holds), the power a number in [0, 1], and the page's
    paragraph stating exactly the figures of this run."""
    m = measure(study)
    assert m.undefined == 0, (study.test_id, "repetitions without a p-value", m.undefined)
    if study.test_id in DOCUMENTED_EXCESS:
        assert not m.holds, f"{study.test_id} now holds its size: leave DOCUMENTED_EXCESS and update its page"
        assert DOCUMENTED_EXCESS[study.test_id] == page
    else:
        assert m.holds, (study.test_id, study.label, m.size5, m.bound5, m.size1, m.bound1)
    assert 0.0 <= m.power <= 1.0
    para = page_paragraph(page, study)
    for figure in m.numbers():
        assert figure in para, (study.test_id, study.label, "the page must state", figure)
    assert study.alternative_text in para, (study.test_id, study.label, "the page must state the alternative")
    if study.test_id in DOCUMENTED_EXCESS:
        assert "measured size" in para
    return m
