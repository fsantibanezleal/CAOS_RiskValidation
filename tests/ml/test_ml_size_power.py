"""RV-1437: the size under the null and the power against the planted defect of each `ml.` validation test, by
`harness.run.simulate` over no fewer than 4,000 repetitions (contract engine-ml-api 2.0.0, SDD l.335). The measured
rates are reported in `docs/ml/08_size-and-power.md`.

History of the bounds (the review of unit U14-E1, majors M-2 and M-3). The first design bounded the size of
`ml.conformal_coverage` at 0.10 and, in the last commit of that design, of `ml.shift_resilience` at 0.15: the 0.15 bound
was set AFTER the run that measured a size of 0.1055 (finding F4), so it was not a bound stated before the run. The
permutation reference replaced that design (Beta-Binomial null for the conformal test, exact permutation reference for
the shift test) and both sizes now answer to the default bound: at most 0.05 plus two Monte Carlo standard errors
(`harness.rates.size_bound`). These runs take minutes to hours: the module carries the `sizepower` marker (the engine CI
deselects it with `-m "not sizepower"`; the local release gate runs it, sharded over RISKVALIDATION_SIZE_WORKERS processes for the registered studies)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ml_sims import N_REP, SEED, conditions, study, study_for

from riskvalidation.harness.rates import size_bound
from riskvalidation.ml.cbpe import cbpe_auc
from riskvalidation.validation.discrimination import auc_with_variance

pytestmark = pytest.mark.sizepower

COND = conditions()
POWER_MIN = 0.90


def test_at_least_4000_repetitions():
    assert N_REP >= 4000


@pytest.mark.parametrize("test_id", sorted(COND))
def test_size_under_the_null_and_power_against_the_planted_defect(test_id):
    """The size under the null of every `ml.` light, among them the two p-value tests the catalogue's size rule names
    here: `ml.shift_resilience` (`shift_resilience`) and `ml.conformal_coverage` (`conformal_coverage`)."""
    run_null, gen_null, gen_def, run_def, _ = COND[test_id]
    null = study_for(test_id, run_null, gen_null, which="null")  # RISKVALIDATION_SIZE_WORKERS > 1: sharded (BL-112)
    defect = study_for(test_id, run_def or run_null, gen_def, which="alternative", seed=SEED + 1)
    size, power = null.rate("alarm"), defect.rate("alarm")
    assert null.n_rep >= 4000 and defect.n_rep >= 4000
    print(  # the measured rates, copied into docs/ml/08 (run with -s)
        f"SIZEPOWER {test_id}: size {size.rate:.4f} se {size.se:.4f} wilson {size.interval}; "
        f"power {power.rate:.4f} se {power.se:.4f} wilson {power.interval}"
    )
    assert null.rates["alarm"].undefined == 0 and defect.rates["alarm"].undefined == 0  # never an unevaluated light
    bound = size_bound(0.05, null.n_rep, z=2.0)  # 0.05 plus two Monte Carlo standard errors, for every test
    assert size.rate <= bound, f"size {size.rate:.4f} above {bound:.4f} (se {size.se:.4f})"
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
