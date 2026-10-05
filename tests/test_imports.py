"""RV-006 and RV-211: the core, the regulatory calculators and the engines import with only NumPy and SciPy (the
Pyodide-safe contract), and the regulatory functions return their references."""

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


def test_core_dependencies():
    """RV-211: the U2 modules import without optional dependencies and every result names its source."""
    code = (
        "import sys, riskvalidation.regulatory, riskvalidation.engines.vasicek, riskvalidation.engines.low_default,"
        " riskvalidation.engines.pd_curve;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"heavy modules imported by U2: {out.stdout.strip()} {out.stderr}"

    from riskvalidation.engines import low_default, pd_curve
    from riskvalidation.regulatory import REGIMES, capital_requirement, el_versus_provisions

    for rid, reg in REGIMES.items():
        assert reg.in_force and reg.references["risk_weight"] and reg.references["scaling"], rid
        cls = "corporate" if "corporate" in reg.classes else reg.classes[0]
        res = capital_requirement(cls, 0.01, None if cls in reg.fixed_lgd else 0.45, regime=rid)
        assert res.references["risk_weight"] and res.references["correlation"], rid
    assert "CAP30.13" in el_versus_provisions(1.0, 0.5, 100.0)["references"]
    assert "Pluto and Tasche" in low_default.most_prudent_pd((10, 10), (0, 0), 0.9)["reference"]
    assert "Tasche (2013)" in pd_curve.qmm([0.5, 0.3, 0.2], 0.05, 0.4)["reference"]
