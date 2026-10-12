"""RV-523 to RV-525: the fixes of the engine review (F3 events above the cohort, F4 reconciliation that cannot be
evaluated, F5 the Vasicek band at a value)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from riskvalidation.core.result import Light
from riskvalidation.provisions import reconciliation as rc
from riskvalidation.validation import ifrs9

FIXTURE = Path(__file__).resolve().parents[1] / "provisions" / "fixtures" / "c06_reporting_date_reconciliation.json"


def load_fixture():
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    make = {"allowance": rc.allowance_rollforward, "gross_carrying_amount": rc.gca_rollforward}
    rows = {k: [make[k](r["stage"], r["opening"], {m["line_id"]: m["amount"] for m in r["movements"]}, r["closing"])
                for r in d["reconciliation"][k]] for k in make}
    ecl_by_stage = {r["stage"]: r["ecl"] for r in d["stage_summary"]["rows"]}
    return d, rows["allowance"], rows["gross_carrying_amount"], d["waterfall"]["lines"], ecl_by_stage


def a(*v, dtype=float):
    return np.array(v, dtype=dtype)


def test_lifetime_pd_backtest_refuses_more_events_than_obligors_at_risk():
    m = a(12, 24, dtype=np.int64)
    ok = ifrs9.lifetime_pd_backtest(m, a(0.01, 0.02), a(0.01, 0.02), a(100, 100), a(1, 100))  # equal is allowed
    assert ok.light is Light.GREEN
    with pytest.raises(ValueError, match="n_events"):
        ifrs9.lifetime_pd_backtest(m, a(0.01, 0.02), a(0.01, 0.02), a(100, 100), a(1, 101))
    # a horizon not yet observed is not judged
    part = ifrs9.lifetime_pd_backtest(m, a(0.01, 0.02), a(0.01, np.nan), a(100, 100), a(1, 101))
    assert part.metric == 0.0


def test_vasicek_band_values_against_an_independent_quadrature():
    # central 95% and 99% default-count intervals, PD 2% n 1000 rho 0.1 and PD 5% n 800 rho 0.1, recomputed by
    # numerical integration over the factor (scipy.integrate.quad), not by the engine
    r = ifrs9.lifetime_pd_backtest(a(12, 24, dtype=np.int64), a(0.02, 0.05), a(0.02, 0.05), a(1000, 800), a(20, 40),
                                   rho=0.1)
    assert r.extras["band_low"] == pytest.approx([0.002, 0.0075], abs=1e-12)
    assert r.extras["band_high"] == pytest.approx([0.067, 0.14125], abs=1e-12)
    assert r.extras["band_low_red"] == pytest.approx([0.0, 0.00375], abs=1e-12)
    assert r.extras["band_high_red"] == pytest.approx([0.098, 0.19375], abs=1e-12)


def test_reconciliation_without_opening_or_closing_is_not_evaluated():
    d, al, gca, lines, e = load_fixture()
    cut = [ln for ln in lines if ln["kind"] not in ("opening", "closing")]
    assert len(cut) < len(lines)
    r = ifrs9.reconciliation_identities(al, gca, cut, e)
    assert r.light is Light.NOT_EVALUATED and r.notes and "opening" in r.notes
    assert r.metric is None or math.isnan(r.metric)
    json.dumps(r.to_dict(), allow_nan=False)  # no inf or NaN reaches the serialised result
    # a stage without a stage ECL cannot be judged either
    r2 = ifrs9.reconciliation_identities(al, gca, lines, {})
    assert r2.light is Light.NOT_EVALUATED and r2.notes
    json.dumps(r2.to_dict(), allow_nan=False)
    # a movement that has no waterfall line is a failing identity with a finite metric
    no_line = [ln for ln in lines if ln["kind"] in ("opening", "closing", "originations")]
    r3 = ifrs9.reconciliation_identities(al, gca, no_line, e)
    assert r3.metric is not None and math.isfinite(r3.metric)
    assert r3.extras["stage_ecl_gap_max"] is None or math.isfinite(r3.extras["stage_ecl_gap_max"])
    json.dumps(r3.to_dict(), allow_nan=False)
