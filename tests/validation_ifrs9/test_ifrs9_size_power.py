"""RV-U16-E2: the size and power of `ifrs9.stage_migration_backtest`, the Anderson-Goodman chi-square on the stage
transitions, measured on multinomial transitions drawn from the predicted matrix (size) and from a shifted one
(power)."""

from __future__ import annotations

import pytest

from riskvalidation.generators.size_studies import study
from size_kit import ROOT, check

# U16-E2 (BL-088): a 4,000-repetition local release gate; the engine CI deselects it with -m "not sizepower"
pytestmark = pytest.mark.sizepower
PAGE = ROOT / "docs" / "provisions" / "ecl" / "05_validation-battery.md"


def test_size_and_power_of_the_stage_migration_backtest():
    """ifrs9.stage_migration_backtest: size at 5% and 1% on its null and power against its alternative."""
    check(study("ifrs9.stage_migration_backtest"), PAGE)
