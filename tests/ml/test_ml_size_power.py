"""RV-1437: the size under the null and the power against the planted defect of each `ml.` validation test, by
`harness.run.simulate` over no fewer than 4,000 repetitions (contract engine-ml-api 1.0.0, SDD l.335). The measured rates
are reported in `docs/ml/08_size-and-power.md`; the bounds below are design bounds of the study, stated before the run."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ml_sims import N_REP, SEED, conditions, study

from riskvalidation.ml.cbpe import cbpe_auc
from riskvalidation.validation.discrimination import auc_with_variance

COND = conditions()
SIZE_MAX = {"ml.conformal_coverage": 0.10}  # the binomial test on an estimated quantile is slightly oversized
SIZE_DEFAULT, POWER_MIN = 0.05, 0.90


def test_at_least_4000_repetitions():
    assert N_REP >= 4000


@pytest.mark.parametrize("test_id", sorted(COND))
def test_size_under_the_null_and_power_against_the_planted_defect(test_id):
    run_null, gen_null, gen_def, run_def, _ = COND[test_id]
    null = study(run_null, gen_null)
    defect = study(run_def or run_null, gen_def, seed=SEED + 1)
    size, power = null.rate("alarm"), defect.rate("alarm")
    assert null.n_rep >= 4000 and defect.n_rep >= 4000
    assert null.rates["alarm"].undefined == 0 and defect.rates["alarm"].undefined == 0  # never an unevaluated light
    bound = SIZE_MAX.get(test_id, SIZE_DEFAULT)
    assert size.rate <= bound, f"size {size.rate:.4f} above {bound} (se {size.se:.4f})"
    assert power.rate >= POWER_MIN, f"power {power.rate:.4f} below {POWER_MIN} (se {power.se:.4f})"


def test_cbpe_estimate_tracks_the_labelled_auc_when_calibrated_and_not_when_it_is_not():
    """No light, so no size and power: the estimate is read against the labelled AUC over 4,000 repetitions."""
    reps, n = N_REP, 600
    seq = np.random.SeedSequence(SEED).spawn(reps)
    diff_cal, diff_mis = [], []
    for child in seq:
        rng = np.random.default_rng(child)
        p = rng.beta(0.8, 2.5, n)
        y_cal = (rng.random(n) < p).astype(int)  # calibrated
        y_mis = (rng.random(n) < np.clip(p**2.0 * 1.0, 0.0, 1.0)).astype(int)  # the probabilities are overstated
        est = cbpe_auc(p).estimated_auc
        for y, out in ((y_cal, diff_cal), (y_mis, diff_mis)):
            if 0 < y.sum() < n:
                out.append(est - auc_with_variance(p, y)[0])
    cal, mis = np.array(diff_cal), np.array(diff_mis)
    se = cal.std(ddof=1) / math.sqrt(len(cal))
    assert abs(cal.mean()) <= 4.0 * se + 1e-3  # unbiased when the assumptions hold (the estimate is not the truth)
    assert abs(cal.mean()) < 0.01
    assert abs(mis.mean()) > 0.03  # a calibration failure moves the estimate away from the labelled AUC
