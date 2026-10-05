"""The README quickstart is a real, runnable program."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_runs():
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "quickstart.py")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "portfolio" in out.stdout and "worst:" in out.stdout


def test_credit_math_runs():
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "credit_math.py")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "basel2" in out.stdout and "c_LR" in out.stdout and "most prudent" in out.stdout
