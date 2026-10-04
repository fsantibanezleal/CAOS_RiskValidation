"""RV-003: a suite records a failing test and finishes the rest."""

from __future__ import annotations

from conftest import selftest_p, selftest_raises
from riskvalidation import Light, run_suite


def test_runner_never_aborts(policy):
    run = run_suite(
        [
            (selftest_p, {"p": 0.5}),
            (selftest_raises, {}),
            (selftest_p, {"p": 0.001}),
            (selftest_p, {"p": 0.03}),
        ],
        policy=policy,
        model_id="m-1",
        segment="all",
    )
    lights = [r.light for r in run.results]
    assert lights == [Light.GREEN, Light.ERROR, Light.RED, Light.AMBER]
    err = run.results[1]
    assert err.test_id == "selftest.raises"
    assert "ZeroDivisionError" in err.notes and "planted failure" in err.notes
    assert all(r.model_id == "m-1" and r.segment == "all" for r in run.results)
    assert run.counts()["error"] == 1 and run.counts()["green"] == 1
    assert run.worst() is Light.ERROR
    assert len(run.table()) == 4
