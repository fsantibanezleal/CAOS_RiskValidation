"""Run a registered size study in slices on a process pool and sum the counts (the command line of
``riskvalidation.generators.size_sharded``, where the slicing, the merge and ``run_sharded`` live since engine 1.4.0).

    python scripts/size_study_sharded.py --study ccr.pfe_exceedance_backtest --reps 4000 --workers 4 --out result.json

The counts, p-values, metrics and statistics equal the unsharded ``simulate_study`` run exactly, whatever the workers.
The size studies stay out of CI (ADR-0074): this is a local tool to bring the hours of a ``sizepower`` study down by the
number of cores; the release gate itself shards through the environment variable ``RISKVALIDATION_SIZE_WORKERS``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from riskvalidation.generators.size_sharded import run_sharded


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study", required=True, help="the test id of a registered study (ccr.pfe_exceedance_backtest)")
    ap.add_argument("--reps", type=int, default=None, help="repetitions (default: the study's own)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--which", choices=("null", "alternative"), default="null")
    ap.add_argument("--index", type=int, default=0, help="which study of the test, where it has several")
    ap.add_argument("--seed", type=int, default=None, help="the study's seed (default: the study's own)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sim = run_sharded(a.study, a.reps, a.workers, which=a.which, index=a.index, seed=a.seed)
    doc = sim.to_dict()
    doc["study"] = {"test_id": a.study, "index": a.index, "which": a.which, "workers": a.workers}
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    a.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
