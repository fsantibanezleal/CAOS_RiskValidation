"""RV-006: the core imports with only NumPy and SciPy (the Pyodide-safe contract)."""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("pandas", "sklearn", "statsmodels", "torch", "lightgbm", "xgboost", "matplotlib")


def test_core_imports_without_optional_deps():
    code = (
        "import sys, riskvalidation, riskvalidation.core.build, riskvalidation.core.suite;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported by the core: {out.stdout.strip()} {out.stderr}"
