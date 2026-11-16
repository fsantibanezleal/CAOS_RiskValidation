"""The sharded size-study runner (``riskvalidation.generators.size_sharded``; the CLI is
``scripts/size_study_sharded.py``; Contraste backlog BL-093, engine 1.2.0, moved into the package by BL-112,
engine 1.4.0).

Each repetition of ``harness.simulate`` runs from its own spawned seed, so a study cut into slices on a process pool
gives the counts of the unsharded run exactly, whatever the number of workers. The studies used here are small (10 to 30
repetitions); the 4,000-repetition studies are the ``sizepower`` suites and are never run by these tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from riskvalidation.generators import size_sharded as sharded
from riskvalidation.generators.size_studies import STUDIES, simulate_study

SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
ML_TESTS = str(Path(__file__).resolve().parent / "ml")
if ML_TESTS not in sys.path:
    sys.path.insert(0, ML_TESTS)

import size_study_sharded as cli  # noqa: E402

STUDY = "ccr.pfe_exceedance_backtest"


def _same(a, b) -> None:
    assert a.n_rep == b.n_rep and a.seed == b.seed and a.test_id == b.test_id
    assert {k: (v.rejections, v.n, v.undefined) for k, v in a.rates.items()} == {
        k: (v.rejections, v.n, v.undefined) for k, v in b.rates.items()
    }
    for name in ("p_values", "metrics", "statistics"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize(("n_rep", "workers"), [(10, 2), (10, 3), (10, 4), (7, 1), (3, 8), (1, 4)])
def test_sharded_counts_equal_simulate_exactly(n_rep, workers):
    """10 repetitions on 2 workers (the brief's case), an uneven cut, one worker, and fewer repetitions than workers."""
    study = STUDIES[STUDY][0]
    _same(sharded.run_sharded(STUDY, n_rep, workers), simulate_study(study, "null", n_rep=n_rep))


def test_sharded_counts_equal_simulate_on_the_alternative_and_another_seed():
    study = STUDIES[STUDY][0]
    got = sharded.run_sharded(STUDY, 12, 3, which="alternative", seed=777)
    _same(got, simulate_study(study, "alternative", n_rep=12, seed=777))
    assert got.seed == 777


def test_the_counts_depend_on_the_seed_so_the_equality_is_not_vacuous():
    study = STUDIES[STUDY][0]
    a = simulate_study(study, "null", n_rep=30, seed=1)
    b = simulate_study(study, "null", n_rep=30, seed=2)
    assert not np.array_equal(a.statistics, b.statistics)
    assert not np.array_equal(sharded.run_sharded(STUDY, 30, 2, seed=1).statistics, b.statistics)


@pytest.mark.parametrize(
    ("test_id", "n_rep", "workers"),
    [
        ("market.acerbi_szekely", 9, 4),  # a test with its own randomness: the seed is drawn after the data
        ("market.dynamic_quantile_test", 6, 3),  # a seeded test of a family of studies
        ("ml.conformal_coverage", 5, 2),  # a study that runs a pipeline on each draw
    ],
)
def test_seeded_tests_and_pipelines_match_simulate(test_id, n_rep, workers):
    study = STUDIES[test_id][0]
    _same(sharded.run_sharded(test_id, n_rep, workers), simulate_study(study, "null", n_rep=n_rep))


@pytest.mark.parametrize(
    ("n_rep", "workers", "want"),
    [
        (10, 2, [(0, 5), (5, 10)]),
        (10, 3, [(0, 4), (4, 7), (7, 10)]),
        (3, 8, [(0, 1), (1, 2), (2, 3)]),
        (1, 1, [(0, 1)]),
    ],
)
def test_slices_cover_the_repetitions_in_order_without_an_empty_one(n_rep, workers, want):
    assert sharded.slices(n_rep, workers) == want


@pytest.mark.parametrize(("n_rep", "workers"), [(0, 2), (5, 0), (-1, 1)])
def test_slices_refuse_a_non_positive_count(n_rep, workers):
    with pytest.raises(ValueError):
        sharded.slices(n_rep, workers)


def test_an_unregistered_study_is_a_keyerror_naming_the_studies():
    with pytest.raises(KeyError, match="ccr.pfe_exceedance_backtest"):
        sharded.run_sharded("no.such_test", 4, 2)


def test_harness_numpy_is_restored_after_a_slice():
    import riskvalidation.harness.run as run

    before = run.np
    sharded.run_sharded(STUDY, 2, 1)
    assert run.np is before


def test_the_cli_writes_the_summed_counts(tmp_path, capsys):
    out = tmp_path / "result.json"
    assert cli.main(["--study", STUDY, "--reps", "10", "--workers", "2", "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    want = simulate_study(STUDIES[STUDY][0], "null", n_rep=10).to_dict()
    assert doc["n_rep"] == 10 and doc["rates"] == want["rates"] and doc["study"]["workers"] == 2
    assert capsys.readouterr().out.strip().startswith("{")


def test_the_cli_is_a_thin_layer_over_the_package_function():
    assert cli.run_sharded is sharded.run_sharded
    assert not hasattr(cli, "_run_slice") and not hasattr(cli, "merge")


# ------------------------------------------------------------------ BL-112: the release gate runs sharded (RV-318)


def test_the_runner_is_public_by_path_with_its_own_all():
    assert {"run_sharded", "slices", "merge", "size_workers", "simulate_registered", "SIZE_WORKERS_ENV"} <= set(
        sharded.__all__
    )
    assert sharded.SIZE_WORKERS_ENV == "RISKVALIDATION_SIZE_WORKERS"


@pytest.mark.parametrize(("value", "want"), [(None, 1), ("1", 1), ("6", 6), (" 12 ", 12)])
def test_the_workers_variable_is_one_when_unset_and_an_integer_otherwise(value, want):
    env = {} if value is None else {sharded.SIZE_WORKERS_ENV: value}
    assert sharded.size_workers(env) == want


@pytest.mark.parametrize("value", ["0", "-2", "abc", "2.5", ""])
def test_the_workers_variable_refuses_zero_and_junk_naming_itself(value):
    with pytest.raises(ValueError, match="RISKVALIDATION_SIZE_WORKERS"):
        sharded.size_workers({sharded.SIZE_WORKERS_ENV: value})


def test_the_workers_variable_is_read_from_the_process_environment(monkeypatch):
    monkeypatch.delenv(sharded.SIZE_WORKERS_ENV, raising=False)
    assert sharded.size_workers() == 1
    monkeypatch.setenv(sharded.SIZE_WORKERS_ENV, "3")
    assert sharded.size_workers() == 3


def test_simulate_registered_with_one_worker_is_simulate_study_itself(monkeypatch):
    study = STUDIES[STUDY][0]
    monkeypatch.setenv(sharded.SIZE_WORKERS_ENV, "1")
    got = sharded.simulate_registered(study, "null", n_rep=6)
    _same(got, simulate_study(study, "null", n_rep=6))
    assert "slices" not in got.extras
    monkeypatch.delenv(sharded.SIZE_WORKERS_ENV)
    _same(sharded.simulate_registered(study, "alternative", n_rep=6), simulate_study(study, "alternative", n_rep=6))


def test_simulate_registered_above_one_worker_shards_and_equals_the_unsharded_run(monkeypatch):
    study = STUDIES[STUDY][0]
    monkeypatch.setenv(sharded.SIZE_WORKERS_ENV, "3")
    got = sharded.simulate_registered(study, "alternative", n_rep=9, seed=5)
    assert got.extras["slices"] == 3
    _same(got, simulate_study(study, "alternative", n_rep=9, seed=5))


def test_simulate_registered_refuses_a_study_that_is_not_registered():
    from dataclasses import replace

    other = replace(STUDIES[STUDY][0], seed=1)
    with pytest.raises(KeyError, match="not a registered"):
        sharded.simulate_registered(other, "null", n_rep=2, workers=2)


@pytest.mark.parametrize("which", ["null", "alternative"])
def test_ml_shift_resilience_gate_path_sharded_equals_unsharded_element_by_element(which):
    """The path of `tests/ml/test_ml_size_power.py` (`ml_sims.study_for`) on `ml.shift_resilience`, 12 repetitions on 3
    workers against the unsharded `harness.simulate` the gate runs by default: equal counts, undefined counts and
    element-wise equal p-values, metrics and statistics (NaN where NaN)."""
    import ml_sims

    test_id = "ml.shift_resilience"
    run_null, gen_null, gen_def, run_def, _ = ml_sims.conditions()[test_id]
    if which == "null":
        run, gen, seed = run_null, gen_null, ml_sims.SEED
    else:
        run, gen, seed = run_def or run_null, gen_def, ml_sims.SEED + 1
    plain = ml_sims.study(run, gen, seed=seed, n_rep=12)
    got = ml_sims.study_for(test_id, run, gen, which=which, seed=seed, n_rep=12, workers=3)
    assert got.extras["slices"] == 3 and "slices" not in plain.extras
    assert got.n_rep == plain.n_rep == 12 and got.seed == plain.seed == seed
    assert {k: (v.rejections, v.n, v.undefined) for k, v in got.rates.items()} == {
        k: (v.rejections, v.n, v.undefined) for k, v in plain.rates.items()
    }
    for name in ("p_values", "metrics", "statistics"):
        np.testing.assert_array_equal(getattr(got, name), getattr(plain, name))  # NaN equals NaN here


def test_ml_gate_seeds_and_repetitions_are_the_registered_studys():
    """The tests' seeds and 4,000 repetitions are the registered studies' own, so a sharded gate run is the same run."""
    import ml_sims

    for test_id in ("ml.shift_resilience", "ml.conformal_coverage"):
        st = STUDIES[test_id][0]
        assert (ml_sims.SEED, ml_sims.SEED + 1, ml_sims.N_REP) == (st.seed, st.power_seed, st.n_rep)
        assert (st.power_n_rep or st.n_rep) == ml_sims.N_REP
        assert test_id in ml_sims.REGISTERED
    assert "ml.robustness" not in ml_sims.REGISTERED
