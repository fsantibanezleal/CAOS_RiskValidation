"""RV-001: every result carries its identity, hypothesis, reference and policy version."""

from __future__ import annotations

import json
import math

import pytest

from riskvalidation import Light, TestResult


def _ok(**kw):
    base = dict(test_id="x.y", h0="h", reference="ref", policy_version="p 1", light=Light.GREEN, p_value=0.3)
    base.update(kw)
    return TestResult(**base)


def test_required_fields():
    for field in ("test_id", "h0", "reference", "policy_version"):
        with pytest.raises(ValueError):
            _ok(**{field: ""})
        with pytest.raises(ValueError):
            _ok(**{field: "   "})
    with pytest.raises(ValueError):
        TestResult(test_id="x", h0="h", reference="r", policy_version="p", light=Light.GREEN)
    with pytest.raises(TypeError):
        _ok(light="green")
    with pytest.raises(ValueError):
        _ok(p_value=1.5)
    # an error result needs no statistic
    err = TestResult(test_id="x", h0="h", reference="r", policy_version="p", light=Light.ERROR, notes="boom")
    assert err.light is Light.ERROR


def test_to_dict_is_json_safe():
    r = _ok(statistic=float("nan"), metric=math.inf, extras={"ci": (0.1, float("nan")), "k": 3})
    d = r.to_dict()
    text = json.dumps(d, allow_nan=False)
    assert '"light": "green"' in text
    assert d["statistic"] is None and d["metric"] is None
    assert d["extras"]["ci"] == [0.1, None]
