"""RV-518: the provisions modules and the IFRS 9 battery are pure NumPy and SciPy and never mutate their inputs."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np

from riskvalidation.provisions import ecl, overlays, reconciliation, staging, term_structure

SRC = Path(__file__).resolve().parents[2] / "src" / "riskvalidation"
FILES = [SRC / "provisions" / f"{m}.py" for m in ("term_structure", "staging", "ecl", "overlays", "reconciliation")]
FILES.append(SRC / "validation" / "ifrs9.py")
FORBIDDEN = {"pandas", "lifelines", "statsmodels", "sklearn", "polars"}


def test_modules_import_no_dataframe_library():
    for f in FILES:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
        assert not (roots & FORBIDDEN), (f.name, roots & FORBIDDEN)
    code = (
        "import sys, riskvalidation.provisions.term_structure, riskvalidation.provisions.staging, "
        "riskvalidation.provisions.ecl, riskvalidation.provisions.overlays, riskvalidation.provisions.reconciliation, "
        "riskvalidation.validation.ifrs9; "
        "print(sorted(m for m in ('pandas','lifelines','statsmodels') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out


def frozen(*a):
    x = np.array(a)
    x.setflags(write=False)
    return x


def test_inputs_are_not_mutated():
    """Read-only arrays raise on any write: every function below must work on them and return new arrays."""
    h = frozen(0.01, 0.02, 0.03)
    m = frozen(3, 6, 9)
    s = term_structure.from_hazard(h, m)
    term_structure.from_marginal(frozen(*s.marginal), m)
    term_structure.from_cumulative(frozen(*s.cumulative), m)
    term_structure.competing_risks(h, frozen(0.0, 0.0, 0.0), m)
    term_structure.forward_pd(s, 3, 9)
    term_structure.weighted_pit_pd(0.02, 0.12, frozen(1.0, 0.0), frozen(0.5, 0.5))
    pol = staging.StagingPolicy(multiplier_schedule=((0.5, 2.0),), absolute_floor=0.0, probation_months=3)
    po, pc = frozen(0.02, 0.03), frozen(0.05, 0.03)
    no = frozen(False, False)
    dpd = frozen(0, 40).astype(np.int64)
    dpd.setflags(write=False)
    res = staging.allocate_stages(po, pc, dpd, no, no, no, pol, collective=no)
    staging.apply_cure_probation(res.stage, frozen(2, 2).astype(np.int8), frozen(0, 0).astype(np.int64), pol)
    staging.collective_transfer(frozen(1.0, 2.0), frozen("a", "b"), {"a": 0.1, "b": 0.2})
    e = ecl.ecl_scenario(frozen(0.01, 0.01).reshape(1, 2), frozen(0.4), frozen(100.0), frozen(0.05), frozen(6, 12), 12)
    ecl.probability_weighted_ecl(frozen(1.0, 2.0), frozen(0.5, 0.5))
    ecl.stage3_ecl(frozen(10.0), frozen(1.0).reshape(1, 1), frozen(6).reshape(1, 1), frozen(0.05))
    ecl.portfolio_allowance(frozen(1, 2), frozen(1.0, 2.0), frozen(3.0, 4.0), frozen(0.0, 0.0))
    ecl.weight_sensitivity(frozen(1.0, 2.0), 0)
    assert e.flags.writeable
    o = overlays.Overlay("A", "r", "r", "expert", "pd", None, None, 1.0, "none", "2020-12-31", "USD")
    overlays.overlay_lines([o])
    reconciliation.check_identities([], [], [], {}, [o])
