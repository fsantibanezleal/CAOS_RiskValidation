"""RV-U16-E2: the size and power of `ccr.pfe_exceedance_backtest`, the Kupiec likelihood ratio of the PFE exceedances,
measured on a PFE series whose forecast is the true quantile (size) and a forecast understated by 10% (power)."""

from __future__ import annotations

import pytest

from riskvalidation.generators.size_studies import study
from size_kit import ROOT, check

# U16-E2 (BL-088): a 4,000-repetition local release gate; the engine CI deselects it with -m "not sizepower"
pytestmark = pytest.mark.sizepower
PAGE = ROOT / "docs" / "ccr" / "06_validation-tests.md"


def test_size_and_power_of_the_pfe_exceedance_backtest():
    """ccr.pfe_exceedance_backtest: size at 5% and 1% on its null and power against its alternative."""
    check(study("ccr.pfe_exceedance_backtest"), PAGE)
