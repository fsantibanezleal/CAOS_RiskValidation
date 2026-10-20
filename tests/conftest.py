"""Shared fixtures: a small policy and two self-test registrations used only by the core tests."""

from __future__ import annotations

import sys
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

import riskvalidation
from riskvalidation.core.build import make_result
from riskvalidation.core.policy import Policy, Threshold
from riskvalidation.core.registry import register
from riskvalidation.core.result import TestResult

# Every TestResult that a function of the engine builds during the session: (test_id, h0, reference) -> count.
# The session-wide check in tests/test_registry_complete.py reads it after every other test has run. Recording lives
# in the tests only: it wraps the constructor's validation hook and never touches src/. An engine frame is one inside
# the imported package's folder, so the check holds on the source tree and on an installed wheel alike (the publish
# workflow tests the built wheel).
PRODUCED: Counter = Counter()
_ENGINE_DIR = Path(riskvalidation.__file__).resolve().parent.as_posix() + "/"
_RESULT_FILE = _ENGINE_DIR + "core/result.py"


@cache
def _is_engine_caller(filename: str) -> bool:
    path = Path(filename).resolve().as_posix()
    return path.startswith(_ENGINE_DIR) and path != _RESULT_FILE


_original_post_init = TestResult.__post_init__


def _recording_post_init(self) -> None:
    _original_post_init(self)
    # frames: this hook, the generated __init__, the constructing caller
    if _is_engine_caller(sys._getframe(2).f_code.co_filename):
        PRODUCED[(self.test_id, self.h0, self.reference)] += 1


TestResult.__post_init__ = _recording_post_init  # type: ignore[method-assign]


def pytest_collection_modifyitems(items):
    """The session-wide registry check runs after every other test, so it sees every result they produced."""
    last = [i for i in items if i.name.endswith("_session_wide")]
    rest = [i for i in items if not i.name.endswith("_session_wide")]
    items[:] = rest + last


@register(
    "selftest.p",
    family="selftest",
    name="p-value self-test",
    h0="the self-test null holds",
    reference="riskvalidation core test fixture",
    decision="p_value",
)
def selftest_p(p: float, *, policy=None, model_id=None, segment=None):
    return make_result("selftest.p", p_value=p, policy=policy, model_id=model_id, segment=segment)


@register(
    "selftest.raises",
    family="selftest",
    name="always raises",
    h0="never evaluated",
    reference="riskvalidation core test fixture",
    decision="metric",
)
def selftest_raises(*, policy=None, model_id=None, segment=None):
    raise ZeroDivisionError("planted failure")


@pytest.fixture
def policy() -> Policy:
    return Policy(
        policy_id="test-policy",
        version="1.2.3",
        description="fixture",
        thresholds={"selftest.p": Threshold(kind="p_value", amber=0.05, red=0.01, source="fixture")},
    )
