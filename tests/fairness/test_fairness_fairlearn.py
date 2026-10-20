"""RV-1456: the metrics reconcile with fairlearn 0.15.0, the optional reference (never imported by the engine)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.fairness.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    group_selection_rates,
)

fairlearn_metrics = pytest.importorskip("fairlearn.metrics", reason="fairlearn is the optional reference")


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_metrics_reconcile_with_fairlearn(seed):
    rng = np.random.default_rng(seed)
    n = 800
    g = np.where(rng.random(n) < 0.45, "a", "b")
    y = (rng.random(n) < 0.5).astype(int)
    d = (rng.random(n) < np.where(g == "a", 0.55, 0.7) * (0.6 + 0.4 * y)).astype(int)
    rates = group_selection_rates(d, g)
    dpd = max(abs(v) for v in demographic_parity_difference(rates, "a").values())
    assert dpd == pytest.approx(
        fairlearn_metrics.demographic_parity_difference(y, d, sensitive_features=g), abs=1e-12
    )
    ratio = demographic_parity_ratio(rates, "a")["b"]
    flr = fairlearn_metrics.demographic_parity_ratio(y, d, sensitive_features=g)
    assert min(ratio, 1.0 / ratio) == pytest.approx(flr, abs=1e-12)  # fairlearn reports the min over the two ratios
    assert equalized_odds_difference(y, d, g) == pytest.approx(
        fairlearn_metrics.equalized_odds_difference(y, d, sensitive_features=g), abs=1e-12
    )
    tpr = fairlearn_metrics.MetricFrame(
        metrics=fairlearn_metrics.true_positive_rate, y_true=y, y_pred=d, sensitive_features=g
    ).difference(method="between_groups")
    assert equal_opportunity_difference(y, d, g) == pytest.approx(tpr, abs=1e-12)
