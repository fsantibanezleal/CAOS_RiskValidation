"""The sharded size-study runner (``scripts/size_study_sharded.py``, Contraste backlog BL-093, engine 1.2.0).

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

from riskvalidation.generators.size_studies import STUDIES, simulate_study

SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import size_study_sharded as sharded  # noqa: E402

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
    assert sharded.main(["--study", STUDY, "--reps", "10", "--workers", "2", "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    want = simulate_study(STUDIES[STUDY][0], "null", n_rep=10).to_dict()
    assert doc["n_rep"] == 10 and doc["rates"] == want["rates"] and doc["study"]["workers"] == 2
    assert capsys.readouterr().out.strip().startswith("{")
