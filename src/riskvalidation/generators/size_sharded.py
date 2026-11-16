"""Run a registered size study in slices on a process pool and sum the counts.

``harness.simulate`` gives repetition ``i`` its own stream, ``SeedSequence(seed).spawn(n_rep)[i]``, so the repetitions
are independent and a slice ``[lo, hi)`` of them can run on its own and give the same data. This module cuts the
repetitions of a study of :mod:`riskvalidation.generators.size_studies` into contiguous slices, runs each slice in a
worker process through ``simulate_study`` (the one entry point of the suites: its generator, feed, arguments and
rejection rule) with the streams ``spawn(n_rep)[lo:hi]`` of the whole study, and sums the counts. The rejection counts,
the undefined counts and the p-values, metrics and statistics of every repetition equal the unsharded run exactly,
whatever the number of workers: each repetition runs from the same spawned seed. A study with fewer repetitions than
workers uses as many workers as it has repetitions.

The release gate (the ``sizepower`` suites, ADR-0074) reads the environment variable ``RISKVALIDATION_SIZE_WORKERS``
through :func:`size_workers`: unset or ``1`` runs each registered study as before, on one worker; a larger value runs it
through :func:`run_sharded` on that many processes (:func:`simulate_registered`). The command line is
``scripts/size_study_sharded.py``:

    python scripts/size_study_sharded.py --study ccr.pfe_exceedance_backtest --reps 4000 --workers 4 --out result.json

The size studies stay out of CI (ADR-0074). Public by path, like :mod:`riskvalidation.generators.size_studies`, which it
imports (and which imports the fairness and ML pipelines, so :mod:`riskvalidation.generators` does not import this).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np

from riskvalidation.generators.size_studies import STUDIES, Study, simulate_study
from riskvalidation.harness import run as harness_run
from riskvalidation.harness.rates import Rate
from riskvalidation.harness.run import Simulation


def slices(n_rep: int, workers: int) -> list[tuple[int, int]]:
    """Contiguous ``(lo, hi)`` slices of ``range(n_rep)``, as many as ``min(workers, n_rep)``, sizes differing by at
    most one (the longer first); none is empty.
    """
    if n_rep < 1:
        raise ValueError("n_rep must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    k = min(workers, n_rep)
    base, extra = divmod(n_rep, k)
    out, lo = [], 0
    for i in range(k):
        hi = lo + base + (1 if i < extra else 0)
        out.append((lo, hi))
        lo = hi
    return out


def _numpy_with_slice(total: int, lo: int, hi: int) -> Any:
    """``numpy`` as ``harness.run`` sees it, with one change: ``np.random.SeedSequence(seed).spawn(n)`` gives the
    streams ``spawn(total)[lo:hi]`` of the whole study. Nothing else of ``numpy`` is touched, and no other module sees
    the change, so ``simulate`` itself runs unmodified on its own slice of the repetitions.
    """

    class _Seed:
        def __init__(self, seed: Any) -> None:
            self._seed = seed

        def spawn(self, n: int) -> list:
            if n != hi - lo:
                raise ValueError(f"the slice has {hi - lo} repetitions, simulate asked for {n}")
            return np.random.SeedSequence(self._seed).spawn(total)[lo:hi]

    class _Random:
        SeedSequence = _Seed

        def __getattr__(self, name: str) -> Any:
            return getattr(np.random, name)

    class _Numpy:
        random = _Random()

        def __getattr__(self, name: str) -> Any:
            return getattr(np, name)

    return _Numpy()


def _find(test_id: str, index: int) -> Study:
    if test_id not in STUDIES:
        raise KeyError(f"no registered size study for {test_id!r}; the studies are {sorted(STUDIES)}")
    return STUDIES[test_id][index]


def _run_slice(job: tuple[str, int, str, int, int, int, int]) -> Simulation:
    test_id, index, which, seed, total, lo, hi = job
    real = harness_run.np
    harness_run.np = _numpy_with_slice(total, lo, hi)
    try:
        return simulate_study(_find(test_id, index), which, n_rep=hi - lo, seed=seed)
    finally:
        harness_run.np = real


def merge(parts: list[Simulation], n_rep: int, seed: int) -> Simulation:
    """The study over all its repetitions from the slices in order: the counts summed, the per-repetition arrays
    joined."""
    first = parts[0]
    rates = {
        label: Rate(
            sum(p.rates[label].rejections for p in parts),
            sum(p.rates[label].n for p in parts),
            sum(p.rates[label].undefined for p in parts),
        )
        for label in first.rates
    }
    return replace(
        first,
        n_rep=n_rep,
        seed=seed,
        rates=rates,
        p_values=np.concatenate([p.p_values for p in parts]),
        metrics=np.concatenate([p.metrics for p in parts]),
        statistics=np.concatenate([p.statistics for p in parts]),
        extras={"cached_inputs": sum(p.extras.get("cached_inputs", 0) for p in parts), "slices": len(parts)},
    )


def run_sharded(
    test_id: str,
    n_rep: int | None = None,
    workers: int = 2,
    *,
    which: str = "null",
    index: int = 0,
    seed: int | None = None,
) -> Simulation:
    """The study ``STUDIES[test_id][index]`` (its null or its alternative) over ``n_rep`` repetitions in slices on
    ``workers`` processes, equal to ``simulate_study(study, which, n_rep=n_rep, seed=seed)``.
    """
    study = _find(test_id, index)
    if which not in ("null", "alternative"):
        raise ValueError("which is 'null' or 'alternative'")
    null = which == "null"
    total = n_rep or (study.n_rep if null else (study.power_n_rep or study.n_rep))
    seed = seed if seed is not None else (study.seed if null else study.power_seed)
    jobs = [(test_id, index, which, seed, total, lo, hi) for lo, hi in slices(total, workers)]
    if len(jobs) == 1:
        parts = [_run_slice(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            parts = list(pool.map(_run_slice, jobs))
    return merge(parts, total, seed)


SIZE_WORKERS_ENV = "RISKVALIDATION_SIZE_WORKERS"


def size_workers(environ: Mapping[str, str] | None = None) -> int:
    """The workers the release gate runs its registered studies on: ``RISKVALIDATION_SIZE_WORKERS`` as a positive
    integer, ``1`` when it is unset. Zero, a negative number, a fraction, an empty value or text raise a ``ValueError``
    that names the variable."""
    env = os.environ if environ is None else environ
    raw = env.get(SIZE_WORKERS_ENV)
    if raw is None:
        return 1
    try:
        n = int(raw.strip())
    except ValueError:
        n = 0
    if n < 1:
        raise ValueError(f"{SIZE_WORKERS_ENV} must be a positive integer (unset or 1 runs on one worker), got {raw!r}")
    return n


def simulate_registered(
    study: Study, which: str = "null", *, n_rep: int | None = None, seed: int | None = None, workers: int | None = None
) -> Simulation:
    """``simulate_study(study, which, n_rep=n_rep, seed=seed)``, sharded over ``workers`` processes when that is above 1
    (default :func:`size_workers`: the environment variable). With one worker it is ``simulate_study`` itself. A worker
    rebuilds the study by its test id and position in ``STUDIES``, so a study that is not registered raises ``KeyError``
    when sharded."""
    workers = size_workers() if workers is None else workers
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return simulate_study(study, which, n_rep=n_rep, seed=seed)
    registered = STUDIES.get(study.test_id, ())
    index = next((i for i, st in enumerate(registered) if st == study), None)
    if index is None:
        raise KeyError(
            f"{study.test_id!r} ({study.label!r}) is not a registered size study; a sharded run rebuilds it by id"
        )
    return run_sharded(study.test_id, n_rep, workers, which=which, index=index, seed=seed)


__all__ = ["SIZE_WORKERS_ENV", "merge", "run_sharded", "simulate_registered", "size_workers", "slices"]
