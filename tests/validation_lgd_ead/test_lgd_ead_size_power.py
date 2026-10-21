"""RV-U16-E2: the size and power of the three LGD p-value tests (the paired t of the segment mean, the Bonferroni
per-period t-tests and the one-sided downturn t-test), each on Beta-distributed workout LGDs whose mean gap is
exact."""

from __future__ import annotations

import pytest

from riskvalidation.generators.size_studies import study
from size_kit import ROOT, check

# U16-E2 (BL-088): a 4,000-repetition local release gate; the engine CI deselects it with -m "not sizepower"
pytestmark = pytest.mark.sizepower
PAGE = ROOT / "docs" / "lgd_ead" / "06_validation-tests.md"


def test_size_and_power_of_the_lgd_calibration_by_segment():
    """lgd_ead.lgd_calibration_by_segment: size at 5% and 1% on its null and power against its alternative."""
    check(study("lgd_ead.lgd_calibration_by_segment"), PAGE)


def test_size_and_power_of_the_lgd_calibration_over_time():
    """lgd_ead.lgd_calibration_over_time: size at 5% and 1% on its null and power against its alternative."""
    check(study("lgd_ead.lgd_calibration_over_time"), PAGE)


def test_size_and_power_of_the_downturn_coverage():
    """lgd_ead.downturn_coverage: size at 5% and 1% on its null and power against its alternative."""
    check(study("lgd_ead.downturn_coverage"), PAGE)
