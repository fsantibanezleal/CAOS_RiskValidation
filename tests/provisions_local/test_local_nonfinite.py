"""Non-finite inputs are refused by the local regimes (review F1): no inf or NaN reaches a result."""

from __future__ import annotations

import math

import pytest

from riskvalidation.provisions import local_cl as cl
from riskvalidation.provisions import local_es as es

INF = math.inf
NAN = math.nan


@pytest.mark.parametrize("bad", [INF, -INF, NAN])
def test_cl_money_and_rates_refuse_non_finite(bad):
    with pytest.raises(ValueError):
        cl.b1_individual_provision(bad, 0.0, "A4")
    with pytest.raises(ValueError):
        cl.b1_group_provision(bad, 0.02, 0.5, 0.0004)
    with pytest.raises(ValueError):
        cl.b1_group_provision(1000.0, 0.02, bad, 0.0004)
    with pytest.raises(ValueError):
        cl.b1_group_provision(1000.0, bad, 0.5, 0.0004)
    with pytest.raises(ValueError):
        cl.b1_default_provision(bad, 0.0)
    with pytest.raises(ValueError):
        cl.b1_default_provision(1000.0, bad)
    with pytest.raises(ValueError):
        cl.b1_consumer_pe(bad, True, False, "instalment")
    with pytest.raises(ValueError):
        cl.b1_individual_provision(1000.0, bad, "A4")
    with pytest.raises(ValueError):
        cl.b1_additional_provisions(bad, 0.0, INF)


def test_cl_bucket_refuses_infinite_value():
    with pytest.raises(ValueError):
        cl._bucket(INF, (0.1, None), "x", "rule")


@pytest.mark.parametrize("bad", [INF, -INF, NAN])
def test_es_refuses_non_finite(bad):
    with pytest.raises(ValueError):
        es.annex9_uncovered_amount(bad, [1.0])
    with pytest.raises(ValueError):
        es.annex9_uncovered_amount(100.0, [bad])
    with pytest.raises(ValueError):
        es.annex9_doubtful_coverage("B.1.1", bad)
    with pytest.raises(ValueError):
        es.annex9_coverage("B.1.1", "normal", bad, [1.0])
    with pytest.raises(ValueError):
        es.annex9_coverage("B.1.1", "normal", 100.0, [bad])
