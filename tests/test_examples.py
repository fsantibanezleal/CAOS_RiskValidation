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


def test_size_power_example_runs():
    """RV-316: the U3 example exercises the generators, the harness and the exact rejection probabilities."""
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "size_power.py")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "true size 11.13%" in out.stdout and "ties=upper" in out.stdout and "Hosmer-Lemeshow" in out.stdout


def test_rating_transitions_example_runs():
    """RV-414: the U4 example runs every part of the transitions module, and with momentum planted the order and
    momentum tests turn red."""
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "rating_transitions.py")], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    assert "Theorem 3" in out.stdout and "TTC portfolio" in out.stdout and "bootstrap" in out.stdout
    planted = out.stdout.split("momentum planted")[1]
    assert "rating.markov_order" in planted and planted.count("red") >= 2
