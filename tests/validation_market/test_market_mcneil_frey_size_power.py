"""RV-U16-E2: the size and power of `market.mcneil_frey_residual_test`, the bootstrap test that the exceedance residuals
have mean zero, measured on normal losses with their true VaR and ES (size) and an ES forecast understated by 5%
(power)."""

from __future__ import annotations

import pytest

from riskvalidation.generators.size_studies import study
from size_kit import ROOT, check

# U16-E2 (BL-088): a 4,000-repetition local release gate; the engine CI deselects it with -m "not sizepower"
pytestmark = pytest.mark.sizepower
PAGE = ROOT / "docs" / "market" / "07_backtests.md"


def test_size_and_power_of_the_mcneil_frey_residual_test():
    """market.mcneil_frey_residual_test: size at 5% and 1% on its null and power against its alternative."""
    check(study("market.mcneil_frey_residual_test"), PAGE)
