"""RV-U16-E2 (follow-up): the power of the eight market back-tests whose size suites had no alternative, each
against one stated departure from its named null (`riskvalidation.generators.size_studies`), run with
`harness.simulate`. There is no pass threshold for power: the number is recorded on the docs page and the suite holds
the page to the run."""

from __future__ import annotations

import pytest

from riskvalidation.generators.size_studies import STUDIES, rejection_label, simulate_study
from size_kit import ROOT, page_paragraph

# a local release gate; the engine CI deselects it with -m "not sizepower"
pytestmark = pytest.mark.sizepower
PAGE = ROOT / "docs" / "market" / "07_backtests.md"
MARKET_IDS = [tid for tid in STUDIES if tid.startswith("market.") and tid != "market.mcneil_frey_residual_test"]
CASES = [(tid, i) for tid in MARKET_IDS for i in range(len(STUDIES[tid]))]


@pytest.mark.parametrize(("test_id", "index"), CASES, ids=[f"{t}[{i}]" for t, i in CASES])
def test_power_of_the_market_backtests_against_their_stated_alternative(test_id, index):
    """market back-tests: power at 5% on the alternative of the named study, in [0, 1], stated by the page."""
    st = STUDIES[test_id][index]
    sim = simulate_study(st, "alternative")
    rate = sim.rate(rejection_label(st))
    assert rate.undefined == 0 and sim.n_rep == (st.power_n_rep or st.n_rep)
    assert 0.0 <= rate.rate <= 1.0
    para = page_paragraph(PAGE, st)
    for figure in (f"{100 * rate.rate:.2f}%", f"seed {st.power_seed}", f"{sim.n_rep} repetitions", st.alternative_text):
        assert figure in para, (test_id, st.label, "the page must state", figure)
