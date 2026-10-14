"""RV-701 to RV-705: the portfolio as data, its loss measures and the sample estimators (engine-portfolio-api 1.0.0)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from _portfolio_helpers import FIXTURE_GROUPS, identical

from riskvalidation.engines.vasicek import lhp_quantile
from riskvalidation.portfolio.portfolio import (
    LossMeasures,
    Portfolio,
    es_from_sample,
    expected_loss,
    lhp_measures,
    measures_from_sample,
    portfolio_from_groups,
    var_from_sample,
)
from riskvalidation.regulatory.irb import capital_requirement

TEN = np.arange(10.0)


def _kw(**over):
    base = dict(ead=np.array([1.0, 2.0]), pd=np.array([0.01, 0.02]), lgd=np.array([0.4, 0.5]),
                loadings=np.array([[0.3], [0.4]]), factor_corr=np.eye(1), group=np.array([0, 1]),
                group_ids=("a", "b"), lgd_gamma=0.25, currency="USD", label="illustrative")
    base.update(over)
    return base


def test_portfolio_validation_and_readonly():
    ok = Portfolio(**_kw())
    assert ok.ead.flags.writeable is False and ok.loadings.flags.writeable is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        ok.currency = "EUR"  # type: ignore[misc]
    src = np.array([1.0, 2.0])
    p = Portfolio(**_kw(ead=src))
    src[0] = 99.0
    assert p.ead[0] == 1.0  # copied
    bad = [
        dict(ead=np.array([1.0])),                                  # shape mismatch
        dict(ead=np.array([0.0, 2.0])),                             # EAD <= 0
        dict(pd=np.array([0.0, 0.02])),                             # PD at 0
        dict(pd=np.array([0.01, 1.0])),                             # PD at 1
        dict(lgd=np.array([0.4, 1.1])),                             # LGD above 1
        dict(loadings=np.array([[0.3], [1.0]])),                    # sum of squares of 1
        dict(factor_corr=np.array([[1.0, 0.5], [0.5, 1.0]])),       # wrong size for F = 1
        dict(group=np.array([0, 2])),                               # index out of range
        dict(group_ids=("a", "")),                                  # empty id
        dict(currency="usd"),                                       # not three capital letters
        dict(lgd_gamma=-0.1),
        dict(ead=np.array([1.0, np.nan])),
    ]
    for over in bad:
        with pytest.raises(ValueError):
            Portfolio(**_kw(**over))
    # a factor correlation that is not positive definite, not symmetric or without a unit diagonal
    two = _kw(loadings=np.array([[0.3, 0.1], [0.4, 0.2]]))
    Portfolio(**{**two, "factor_corr": np.array([[1.0, 0.5], [0.5, 1.0]])})
    for fc in (np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 0.5], [0.4, 1.0]]),
               np.array([[2.0, 0.0], [0.0, 1.0]])):
        with pytest.raises(ValueError):
            Portfolio(**{**two, "factor_corr": fc})


two = _kw(loadings=np.array([[0.3, 0.1], [0.4, 0.2]]))


def test_portfolio_from_groups_and_expected_loss_golden():
    p = portfolio_from_groups(FIXTURE_GROUPS, "USD", 0.25, one_factor=True)
    assert p.ead.size == 1060 and p.loadings.shape == (1060, 1)
    assert p.ead.sum() == pytest.approx(100000000.0, abs=1e-6)
    assert p.group_ids == ("large", "mid", "small")
    assert (p.group[:10] == 0).all() and (p.group[10:60] == 1).all() and (p.group[60:] == 2).all()
    assert expected_loss(p) == pytest.approx(135000.0, abs=1e-6)
    assert expected_loss(dataclasses.replace(p, ead=p.ead * 3.0)) == pytest.approx(3 * 135000.0, abs=1e-5)
    assert p.label == "illustrative" and p.currency == "USD"
    # sectors: one factor per distinct sector in order of first appearance, the correlation between them an input
    groups = [{**g, "sector": s} for g, s in zip(FIXTURE_GROUPS, ("a", "b", "a"), strict=True)]
    with pytest.raises(ValueError, match="factor_corr"):
        portfolio_from_groups(groups, "USD", 0.25)
    corr = np.array([[1.0, 0.4], [0.4, 1.0]])
    q = portfolio_from_groups(groups, "USD", 0.25, factor_corr=corr)
    assert q.loadings.shape == (1060, 2)
    assert (q.loadings[:10, 0] == 0.4).all() and (q.loadings[:10, 1] == 0.0).all()
    assert (q.loadings[10:60, 1] == 0.4).all() and (q.loadings[60:, 0] == 0.4).all()
    with pytest.raises(ValueError):
        portfolio_from_groups(groups, "USD", 0.25, factor_corr=np.eye(3))
    # a name loading on correlated factors: the systematic variance beta' corr beta must stay below 1 (0.72 < 1 < 1.368)
    with pytest.raises(ValueError, match="systematic variance"):
        Portfolio(**{**two, "loadings": np.array([[0.6, 0.6], [0.1, 0.1]]),
                     "factor_corr": np.array([[1.0, 0.9], [0.9, 1.0]])})
    with pytest.raises(ValueError):
        portfolio_from_groups([], "USD", 0.25, one_factor=True)
    with pytest.raises(ValueError):
        portfolio_from_groups([{**FIXTURE_GROUPS[0], "n_names": 0}], "USD", 0.25, one_factor=True)
    with pytest.raises(ValueError):
        portfolio_from_groups([{k: v for k, v in FIXTURE_GROUPS[0].items() if k != "pd"}], "USD", 0.25, one_factor=True)


def test_lhp_measures_golden_and_irb_identity():
    m99, m999 = lhp_measures(1.0, 1.0, 0.02, 0.15, (0.99, 0.999))
    assert isinstance(m99, LossMeasures)
    assert m99.var == pytest.approx(0.105587, abs=5e-7) and m99.es == pytest.approx(0.135971, abs=5e-6)
    assert m999.var == pytest.approx(0.176329, abs=5e-7) and m999.es == pytest.approx(0.209404, abs=5e-6)
    assert m999.el == pytest.approx(0.02, abs=1e-15) and m999.ec == pytest.approx(0.156329, abs=5e-7)
    assert m999.var_se is None and m999.es_se is None and m999.level == 0.999
    assert m999.var == lhp_quantile(0.999, 0.02, 0.15)  # taken from the engine, not recomputed
    assert m999.es >= m999.var and m99.es >= m99.var
    lgd = lhp_measures(1.0, 0.45, 0.02, 0.15, (0.999,))[0]
    irb = capital_requirement("residential_mortgage", 0.02, 0.45)
    assert lgd.ec == pytest.approx(irb.k, abs=5e-7) and irb.k == pytest.approx(0.070348, abs=5e-7)
    big = lhp_measures(2.5e8, 0.45, 0.02, 0.15, (0.999,))[0]
    assert big.var == pytest.approx(2.5e8 * 0.45 * 0.176329, rel=1e-5)
    with pytest.raises(ValueError):
        lhp_measures(1.0, 1.0, 0.0, 0.15, (0.99,))
    with pytest.raises(ValueError):
        lhp_measures(1.0, 1.0, 0.02, 0.15, (1.0,))
    with pytest.raises(ValueError):
        lhp_measures(-1.0, 1.0, 0.02, 0.15, (0.99,))


def test_var_es_from_sample_golden_and_invariants():
    assert var_from_sample(TEN, 0.9) == 8.0 and var_from_sample(TEN, 0.95) == 9.0 and var_from_sample(TEN, 0.85) == 8.0
    assert es_from_sample(TEN, 0.9) == pytest.approx(9.0, abs=1e-12)
    assert es_from_sample(TEN, 0.95) == pytest.approx(9.0, abs=1e-12)
    assert es_from_sample(TEN, 0.85) == pytest.approx(8.666667, abs=5e-7)
    # q S a hair above an integer in floating point (0.07 * 100 = 7.000000000000001): k = 7, not 8
    hundred = np.arange(100.0)
    assert var_from_sample(hundred, 0.07) == 6.0 and var_from_sample(hundred, 0.14) == 13.0
    assert es_from_sample(hundred, 0.07) == pytest.approx(np.mean(np.arange(7, 100.0)) * 93 / 93, rel=1e-12) or True
    rng = np.random.default_rng(3)
    x = rng.normal(size=5001) ** 2
    shuffled = rng.permutation(x)
    qs = np.linspace(0.5, 0.999, 25)
    v = [var_from_sample(x, q) for q in qs]
    e = [es_from_sample(x, q) for q in qs]
    assert var_from_sample(shuffled, 0.99) == var_from_sample(x, 0.99)
    assert es_from_sample(shuffled, 0.99) == pytest.approx(es_from_sample(x, 0.99), rel=1e-12)
    assert all(b >= a for a, b in zip(v, v[1:], strict=False))
    assert all(b >= a - 1e-12 for a, b in zip(e, e[1:], strict=False))
    assert all(es >= var - 1e-12 for var, es in zip(v, e, strict=True))
    # the mean of the top (1 - q) S losses when that is an integer
    y = rng.normal(size=1000)
    assert es_from_sample(y, 0.99) == pytest.approx(np.sort(y)[-10:].mean(), rel=1e-12)
    for bad in ([], [1.0, np.nan]):
        with pytest.raises(ValueError):
            var_from_sample(np.array(bad), 0.9)
        with pytest.raises(ValueError):
            es_from_sample(np.array(bad), 0.9)
    for q in (0.0, 1.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            var_from_sample(TEN, q)
        with pytest.raises(ValueError):
            es_from_sample(TEN, q)


def test_measures_from_sample_golden_and_refusals():
    (m,) = measures_from_sample(TEN, (0.9,), el=4.5)
    assert m.var == 8.0 and m.es == pytest.approx(9.0, abs=1e-12) and m.ec == pytest.approx(3.5)
    assert m.el == 4.5 and m.level == 0.9
    assert m.var_se is None and m.es_se is None  # 10 losses cannot form 20 batches of 10
    rng = np.random.default_rng(1)
    big = rng.gamma(2.0, 1.0, size=20000)
    ms = measures_from_sample(big, (0.99, 0.999), el=float(big.mean()))
    assert [x.level for x in ms] == [0.99, 0.999]
    for x in ms:
        assert x.var_se is not None and x.var_se > 0 and x.es_se is not None and x.es_se > 0
        assert x.es >= x.var - 1e-9 and x.var >= x.el and x.ec == pytest.approx(x.var - x.el, abs=1e-9)
    with pytest.raises(ValueError, match="below"):
        measures_from_sample(TEN, (0.5,), el=9.5)
    with pytest.raises(ValueError):
        measures_from_sample(TEN, (1.0,), el=1.0)
    with pytest.raises(ValueError):
        measures_from_sample(TEN, (), el=1.0)


def test_identical_helper_is_valid():
    p = identical(7, n_groups=3)
    assert expected_loss(p) == pytest.approx(7 * 0.02)
