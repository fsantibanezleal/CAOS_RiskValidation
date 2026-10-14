"""RV-826: every market back-test returns a serialisable `common/test-result` row, never a number when it cannot run,
and raises on a NaN return (contract `engine-market-api`, module validation.market)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.core.result import Light, TestResult
from riskvalidation.validation import market as m

CONTRACT_KEYS = {
    "test_id", "model_id", "segment", "statistic", "p_value", "metric", "h0", "alternative", "policy_version",
    "alpha_amber", "alpha_red", "light", "n", "n_events", "inputs_hash", "reference", "notes", "extras",
}


def calls():
    rng = np.random.default_rng(1)
    hits = rng.random(300) < 0.05
    hits[100:103] = True
    var = -0.02 + 0.003 * rng.standard_normal(300)
    pit = rng.random(300)
    loss = np.abs(rng.normal(0.01, 0.01, 300))
    h = np.arange(250, dtype=float)
    pla = rng.normal(size=250)
    return {
        "kupiec_pof": lambda: m.kupiec_pof(hits, 0.05),
        "christoffersen_independence": lambda: m.christoffersen_independence(hits, 0.05),
        "christoffersen_conditional_coverage": lambda: m.christoffersen_conditional_coverage(hits, 0.05),
        "duration_test": lambda: m.duration_test(hits, 0.05, n_mc=99),
        "basel_traffic_light": lambda: m.basel_traffic_light(hits),
        "acerbi_szekely": lambda: m.acerbi_szekely(loss, np.full(300, 0.03), np.full(300, 0.04), 0.025,
                                                   lambda g: np.abs(g.normal(0.01, 0.01, 300)), n_sim=200),
        "du_escanciano_unconditional": lambda: m.du_escanciano_unconditional(pit, 0.05),
        "du_escanciano_conditional": lambda: m.du_escanciano_conditional(pit, 0.05),
        "mcneil_frey_residual_test": lambda: m.mcneil_frey_residual_test(loss, np.full(300, 0.02), np.full(300, 0.03),
                                                                         np.full(300, 0.01), 0.05, n_boot=200),
        "dynamic_quantile_test": lambda: m.dynamic_quantile_test(hits, var, 0.05),
        "pla_test": lambda: m.pla_test(h, h + pla[:250] * 3),
        "desk_backtest": lambda: m.desk_backtest(np.full(250, 0.03), np.full(250, 0.02), 0.01),
    }


def test_every_result_serialises_with_its_reference():
    for name, fn in calls().items():
        r = fn()
        assert isinstance(r, TestResult)
        row = r.to_dict()
        assert set(row) == CONTRACT_KEYS, name
        assert row["test_id"] == f"market.{name}"
        assert row["reference"].strip() and row["h0"].strip() and row["policy_version"].strip(), name
        assert row["light"] in {"green", "amber", "red", "not_evaluated"}
        if name == "pla_test":
            assert row["segment"] is None
        elif name == "basel_traffic_light":
            assert row["segment"] == "alpha=0.01"
        else:
            assert row["segment"] is not None and row["segment"].startswith("alpha="), name
        assert row["inputs_hash"], name
        if row["light"] == "not_evaluated":
            assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None
    # the parameters of a light rule travel in extras (the lights with a threshold beyond the alphas)
    assert "thresholds" in calls()["pla_test"]().extras


def test_degenerate_inputs_are_not_evaluated():
    empty = np.zeros(0, dtype=bool)
    none = np.zeros(120, dtype=bool)
    cases = [
        m.kupiec_pof(empty, 0.05), m.christoffersen_independence(none, 0.05),
        m.christoffersen_conditional_coverage(none, 0.05), m.duration_test(none, 0.05, n_mc=50),
        m.basel_traffic_light(none[:100]),
        m.acerbi_szekely(np.zeros(3), np.full(3, 0.02), np.array([0.03, np.nan, 0.03]), 0.025,
                         lambda g: np.zeros(3), n_sim=50),
        m.du_escanciano_unconditional(np.full(20, np.nan), 0.05), m.du_escanciano_conditional(np.full(20, 0.5), 0.05),
        m.mcneil_frey_residual_test(np.zeros(10), np.full(10, 1.0), np.full(10, 2.0), np.ones(10), 0.05, n_boot=50),
        m.dynamic_quantile_test(none[:5], np.full(5, -0.02), 0.05),
        m.pla_test(np.arange(10, dtype=float), np.arange(10, dtype=float)),
        m.desk_backtest(np.zeros(100), np.full(100, 0.02), 0.01),
    ]
    for r in cases:
        assert r.light is Light.NOT_EVALUATED, r.test_id
        assert r.notes.startswith("not evaluated:"), r.test_id
        row = r.to_dict()
        assert row["statistic"] is None and row["metric"] is None and row["p_value"] is None, r.test_id
        assert r.p_value is None and r.metric is None


def test_nan_inputs_raise():
    hits = np.zeros(100, dtype=bool)
    hits[3] = True
    x = np.full(100, 0.01)
    x[5] = np.nan
    with pytest.raises(ValueError):
        m.kupiec_pof(np.array([0.0, np.nan, 1.0]), 0.05)  # a flag array with a NaN is not a flag array
    with pytest.raises(ValueError):
        m.acerbi_szekely(x, np.full(100, 0.02), np.full(100, 0.03), 0.025, lambda g: np.zeros(100), n_sim=10)
    with pytest.raises(ValueError):
        m.mcneil_frey_residual_test(x, np.full(100, 0.02), np.full(100, 0.03), np.ones(100), 0.05)
    with pytest.raises(ValueError):
        m.dynamic_quantile_test(hits, x, 0.05)
    with pytest.raises(ValueError):
        m.duration_test(np.array([0, 1, np.nan]), 0.05)
    with pytest.raises(ValueError):
        m.du_escanciano_unconditional(np.array([0.2, np.nan, 0.4]), 0.05)
    with pytest.raises(ValueError):
        m.christoffersen_independence(hits[:50].astype(float) * np.nan, 0.05)
