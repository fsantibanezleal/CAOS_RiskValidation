"""U16-E1 (decision D16-5): the semantic-versioning and deprecation policy of 1.0.0, and the helper that applies it."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from riskvalidation.core.deprecation import deprecated

ROOT = Path(__file__).resolve().parents[1]


def test_deprecated_function_warns_with_the_three_facts():
    @deprecated(since="1.2.0", removal="2.0.0", use="new_name")
    def old(x):
        """Old docstring."""
        return x + 1

    with pytest.warns(DeprecationWarning) as rec:
        assert old(1) == 2
    msg = str(rec[0].message)
    assert "1.2.0" in msg and "2.0.0" in msg and "new_name" in msg and "old" in msg
    assert old.__name__ == "old" and "Old docstring." in old.__doc__
    assert rec[0].filename == __file__  # attributed to the caller, not to the helper


def test_deprecated_class_warns_on_construction_and_keeps_behaviour():
    @deprecated(since="1.1.0", removal="2.0.0", use="Better")
    class Old:
        """An old class."""

        def __init__(self, v):
            self.v = v

    with pytest.warns(DeprecationWarning, match=r"1\.1\.0.*2\.0\.0.*Better"):
        obj = Old(3)
    assert obj.v == 3 and isinstance(obj, Old) and Old.__name__ == "Old"


def test_a_name_that_is_not_deprecated_does_not_warn():
    def fine():
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fine() == 1


@pytest.mark.parametrize("since,removal", [("1.2", "2.0.0"), ("1.2.0", "2.1.0"), ("2.0.0", "2.0.0"),
                                           ("1.5.0", "1.9.0"), ("3.0.0", "2.0.0"), ("x", "2.0.0")])
def test_deprecated_refuses_a_removal_that_is_not_a_later_major(since, removal):
    with pytest.raises(ValueError, match="since|removal"):
        deprecated(since=since, removal=removal, use="x")


def test_deprecated_refuses_an_empty_replacement():
    with pytest.raises(ValueError, match="use"):
        deprecated(since="1.0.0", removal="2.0.0", use="")


def test_the_classifier_is_production_stable():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "Development Status :: 5 - Production/Stable" in text
    assert not re.search(r"Development Status :: [1-4]", text)


def test_the_policy_states_versioning_and_deprecation():
    text = (ROOT / "docs" / "api-policy.md").read_text(encoding="utf-8")
    for needle in ("Semantic versioning", "DeprecationWarning", "at least one minor release", "py.typed",
                   "riskvalidation.core.deprecation", "What is not public"):
        assert needle in text, needle


def test_deprecated_parameter_warns_and_maps_the_old_name_to_the_new():
    from riskvalidation.core.deprecation import deprecated_parameter

    @deprecated_parameter("alpha", since="1.3.0", removal="2.0.0", use="level")
    def f(x, level=0.05):
        return x, level

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f(1, level=0.1) == (1, 0.1)
    with pytest.warns(DeprecationWarning, match=r"'alpha'.*1\.3\.0.*2\.0\.0.*'level'") as rec:
        assert f(1, alpha=0.2) == (1, 0.2)
    assert rec[0].filename == __file__
    with pytest.raises(TypeError, match="both"):
        f(1, alpha=0.2, level=0.3)
    with pytest.raises(ValueError, match="removal"):
        deprecated_parameter("a", since="1.0.0", removal="1.5.0", use="b")


def test_module_getattr_keeps_a_moved_name_working_with_a_warning():
    from riskvalidation.core.deprecation import module_getattr

    getter = module_getattr("riskvalidation.fake", {"OLD": (42, "1.1.0", "2.0.0", "riskvalidation.fake.NEW")})
    with pytest.warns(DeprecationWarning, match=r"riskvalidation\.fake\.OLD.*1\.1\.0.*2\.0\.0.*fake\.NEW"):
        assert getter("OLD") == 42
    with pytest.raises(AttributeError, match="no attribute 'other'"):
        getter("other")
    with pytest.raises(ValueError, match="removal"):
        module_getattr("m", {"X": (1, "1.0.0", "1.1.0", "Y")})
