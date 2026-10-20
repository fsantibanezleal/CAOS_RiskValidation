"""RV-910 to RV-915: SA-CCR (CRE52) function by function (contract `engine-regulatory-api`, module sa_ccr). The
goldens are the contract's; the formulas are recomputed in the tests by a second route (plain `math`) so that no test
only types the answer.
The SA-CCR block of the C12 documents is reproduced field by field from its trades."""

from __future__ import annotations

import math

import numpy as np
import pytest
from reg_helpers import load_c12_blocks

from riskvalidation.regulatory import sa_ccr as s


def _sd(start: float, end: float) -> float:
    return (math.exp(-0.05 * start) - math.exp(-0.05 * end)) / 0.05


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_supervisory_duration_goldens_and_additivity():
    assert s.supervisory_duration(0, 5) == pytest.approx(4.423984, abs=5e-7)
    assert s.supervisory_duration(0, 1) == pytest.approx(0.975412, abs=5e-7)
    assert s.supervisory_duration(1, 5) == pytest.approx(3.448573, abs=5e-7)
    for start, end in ((0, 0.5), (0, 10), (1, 11), (2.5, 7.25), (0, 30)):
        assert s.supervisory_duration(start, end) == pytest.approx(_sd(start, end), rel=1e-13)
    assert abs(s.supervisory_duration(0, 1) + s.supervisory_duration(1, 5) - s.supervisory_duration(0, 5)) < 1e-12
    assert (
        abs(s.supervisory_duration(0.3, 2.2) + s.supervisory_duration(2.2, 9) - s.supervisory_duration(0.3, 9)) < 1e-12
    )
    for start, end, name in (
        (-0.1, 1, "start"),
        (1, 1, "end"),
        (2, 1, "end"),
        (float("nan"), 1, "start"),
        (0, float("inf"), "end"),
    ):
        with pytest.raises(ValueError, match=name):
            s.supervisory_duration(start, end)


def test_supervisory_delta_option():
    price, strike, vol, expiry = 0.06, 0.05, 0.5, 1.0
    d1 = (math.log(price / strike) + 0.5 * vol**2 * expiry) / (vol * math.sqrt(expiry))
    assert s.supervisory_delta_option(price, strike, vol, expiry, is_call=True, is_long=True) == pytest.approx(
        _phi(d1), rel=1e-12
    )
    assert s.supervisory_delta_option(price, strike, vol, expiry, is_call=True, is_long=False) == pytest.approx(
        -_phi(d1), rel=1e-12
    )
    assert s.supervisory_delta_option(price, strike, vol, expiry, is_call=False, is_long=True) == pytest.approx(
        -_phi(-d1), rel=1e-12
    )
    assert s.supervisory_delta_option(price, strike, vol, expiry, is_call=False, is_long=False) == pytest.approx(
        _phi(-d1), rel=1e-12
    )
    # CRE99 netting set 1 swaption: a bought put, price 6 percent, strike 5 percent, vol 50 percent, expiry 1 year
    assert s.supervisory_delta_option(0.06, 0.05, 0.5, 1.0, is_call=False, is_long=True) == pytest.approx(
        -0.2693952, abs=5e-8
    )
    for p, k, v, t in ((0.06, 0.05, 0.5, 1.0), (100.0, 120.0, 0.2, 0.5), (3.0, 3.0, 1.2, 4.0), (50.0, 10.0, 0.7, 2.0)):
        bc = s.supervisory_delta_option(p, k, v, t, True, True)
        sc = s.supervisory_delta_option(p, k, v, t, True, False)
        bp = s.supervisory_delta_option(p, k, v, t, False, True)
        sp = s.supervisory_delta_option(p, k, v, t, False, False)
        assert 0 < bc < 1 and -1 < sc < 0 and -1 < bp < 0 and 0 < sp < 1
        assert bc - bp == pytest.approx(1.0, abs=1e-12)  # put-call parity of the deltas
        assert sc == -bc and sp == -bp
    for kwargs, name in (
        ({"price": 0.0}, "price"),
        ({"strike": -1.0}, "strike"),
        ({"vol": 0.0}, "vol"),
        ({"expiry": 0.0}, "expiry"),
        ({"price": float("nan")}, "price"),
    ):
        args = {"price": 0.06, "strike": 0.05, "vol": 0.5, "expiry": 1.0, "is_call": True, "is_long": True}
        args.update(kwargs)
        with pytest.raises(ValueError, match=name):
            s.supervisory_delta_option(**args)


def test_supervisory_delta_tranche():
    assert s.supervisory_delta_tranche(0.0, 1.0, True) == pytest.approx(1.0, rel=1e-12)
    assert s.supervisory_delta_tranche(0.0, 0.1, True) == pytest.approx(15 / (1 * 2.4), rel=1e-12)
    assert s.supervisory_delta_tranche(0.0, 0.1, True) == pytest.approx(6.25, rel=1e-12)
    assert s.supervisory_delta_tranche(0.03, 0.07, False) == pytest.approx(
        -15 / ((1 + 14 * 0.03) * (1 + 14 * 0.07)), rel=1e-12
    )
    assert s.supervisory_delta_tranche(0.2, 0.5, False) == -s.supervisory_delta_tranche(0.2, 0.5, True)
    for a, d, name in (
        (-0.1, 0.5, "attachment"),
        (0.5, 0.5, "detachment"),
        (0.6, 0.5, "detachment"),
        (0.0, 1.1, "detachment"),
    ):
        with pytest.raises(ValueError, match=name):
            s.supervisory_delta_tranche(a, d, True)


def test_maturity_factor():
    assert s.maturity_factor(0.25) == pytest.approx(0.5, abs=1e-15)
    assert s.maturity_factor(5.0) == 1.0
    assert s.maturity_factor(1.0) == 1.0
    assert s.maturity_factor(0.5) == pytest.approx(math.sqrt(0.5), rel=1e-15)
    assert s.maturity_factor(1.0, mpor_business_days=14) == pytest.approx(0.354965, abs=5e-7)
    assert s.maturity_factor(1.0, 14) == pytest.approx(1.5 * math.sqrt(14 / 250), rel=1e-14)
    assert s.maturity_factor(7.0, mpor_business_days=10) == pytest.approx(0.3, rel=1e-14)
    unmargined = [s.maturity_factor(m / 10) for m in range(1, 30)]
    assert unmargined == sorted(unmargined) and max(unmargined) == 1.0
    margined = [s.maturity_factor(1.0, d) for d in range(1, 60)]
    assert margined == sorted(margined)
    # the 10 business day floor of CRE52.50 belongs to the caller: a shorter MPoR is not raised
    assert s.maturity_factor(1.0, 5) == pytest.approx(1.5 * math.sqrt(5 / 250), rel=1e-14)
    for kwargs, name in (
        ({"maturity_years": -1.0}, "maturity_years"),
        ({"maturity_years": float("nan")}, "maturity_years"),
        ({"maturity_years": 1.0, "mpor_business_days": 0}, "mpor_business_days"),
        ({"maturity_years": 1.0, "mpor_business_days": -3}, "mpor_business_days"),
    ):
        with pytest.raises(ValueError, match=name):
            s.maturity_factor(**kwargs)


def test_maturity_bucket():
    assert s.maturity_bucket(0.5) == "B1"
    assert s.maturity_bucket(0.0) == "B1"
    assert s.maturity_bucket(0.999999) == "B1"
    assert s.maturity_bucket(1.0) == "B2"
    assert s.maturity_bucket(3.0) == "B2"
    assert s.maturity_bucket(5.0) == "B2"
    assert s.maturity_bucket(5.01) == "B3"
    assert s.maturity_bucket(30.0) == "B3"
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="end_years"):
            s.maturity_bucket(bad)


def test_ir_effective_notional():
    assert s.ir_effective_notional(100, 0, 0) == pytest.approx(100.0)
    assert s.ir_effective_notional(100, -100, 0) == pytest.approx(math.sqrt(10000 + 10000 - 14000), rel=1e-12)
    assert s.ir_effective_notional(100, -100, 0) == pytest.approx(77.459667, abs=5e-7)
    assert s.ir_effective_notional(100, 100, 0, offset=False) == pytest.approx(200.0)
    assert s.ir_effective_notional(-5, 7, -3, offset=False) == pytest.approx(15.0)
    d1, d2, d3 = 3.0, -2.0, 5.0
    want = math.sqrt(d1**2 + d2**2 + d3**2 + 1.4 * d1 * d2 + 1.4 * d2 * d3 + 0.6 * d1 * d3)
    assert s.ir_effective_notional(d1, d2, d3) == pytest.approx(want, rel=1e-13)
    rng = np.random.default_rng(20261019)
    for d in rng.normal(0.0, 100.0, size=(200, 3)):
        off = s.ir_effective_notional(*map(float, d))
        no = s.ir_effective_notional(*map(float, d), offset=False)
        assert 0.0 <= off <= no + 1e-9
    # the CRE99 netting set 1 effective notionals give the add-ons at the 0.5 percent factor
    assert 0.005 * 59270 == pytest.approx(296.35, abs=1e-9) and 0.005 * 10083 == pytest.approx(50.415, abs=1e-9)
    with pytest.raises(ValueError, match="d_b2"):
        s.ir_effective_notional(1.0, float("nan"), 0.0)


def _blocks():
    return load_c12_blocks()


def _trades_from_block(block):
    return [
        s.IrTrade(
            t["hedging_set"],
            float(t["start_years"]),
            float(t["end_years"]),
            float(t["notional"]["amount"]),
            float(t["delta"]),
        )
        for t in block["trades"]
    ]


def test_ir_addon_invariants():
    swap = s.IrTrade("USD", 0.0, 10.0, 1_000_000.0, 1.0)
    one = s.ir_addon([swap])
    assert one.addon == pytest.approx(0.005 * 1_000_000.0 * _sd(0, 10), rel=1e-13)
    assert one.trades[0].bucket == "B3" and one.trades[0].maturity_factor == 1.0
    # the bucket follows the END date: a forward-starting trade (start 2, end 7) is B3, never B2 (CRE52.57)
    fwd = s.ir_addon([s.IrTrade("USD", 2.0, 7.0, 1_000_000.0, 1.0)])
    assert fwd.trades[0].bucket == "B3" and fwd.hedging_sets[0].d_b2 == 0.0
    assert one.trades[0].supervisory_duration == pytest.approx(_sd(0, 10), rel=1e-14)
    assert one.trades[0].adjusted_notional == pytest.approx(1_000_000.0 * _sd(0, 10), rel=1e-14)
    # a receiver has delta -1: its effective notional is negative but the add-on is the same size
    receiver = s.IrTrade("USD", 0.0, 10.0, 1_000_000.0, -1.0)
    rec = s.ir_addon([receiver])
    assert rec.trades[0].effective_notional == pytest.approx(-one.trades[0].effective_notional)
    assert rec.addon == pytest.approx(one.addon, rel=1e-13)
    # two opposite swaps of equal terms net to zero
    assert s.ir_addon([swap, receiver]).addon == pytest.approx(0.0, abs=1e-9)
    # two currencies sum their hedging-set add-ons (CRE52.57)
    eur = s.IrTrade("EUR", 0.0, 4.0, 500_000.0, 1.0)
    both = s.ir_addon([swap, eur])
    assert [h.hedging_set for h in both.hedging_sets] == ["USD", "EUR"]
    assert both.addon == pytest.approx(s.ir_addon([swap]).addon + s.ir_addon([eur]).addon, rel=1e-13)
    for h in both.hedging_sets:
        assert h.addon == pytest.approx(0.005 * h.effective_notional, rel=1e-13) and h.supervisory_factor == 0.005
    # no offset: the absolute bucket sum, never below the offset value
    mixed = [
        s.IrTrade("USD", 0.0, 0.5, 1e6, 1.0),
        s.IrTrade("USD", 0.0, 3.0, 1e6, -1.0),
        s.IrTrade("USD", 0.0, 9.0, 1e6, 1.0),
    ]
    assert s.ir_addon(mixed, offset=False).addon >= s.ir_addon(mixed, offset=True).addon
    assert s.ir_addon(mixed, offset=False).hedging_sets[0].offset is False
    # margined: every trade takes the margined maturity factor
    margined = s.ir_addon([swap], mpor_business_days=10)
    assert margined.trades[0].maturity_factor == pytest.approx(0.3, rel=1e-14)
    assert margined.addon == pytest.approx(0.3 * one.addon, rel=1e-13)
    assert s.ir_addon([]).addon == 0.0
    with pytest.raises(ValueError, match="notional"):
        s.ir_addon([s.IrTrade("USD", 0.0, 1.0, float("nan"), 1.0)])
    with pytest.raises(ValueError, match="end"):
        s.ir_addon([s.IrTrade("USD", 2.0, 1.0, 1.0, 1.0)])


def test_ir_addon_reproduces_the_c12_blocks():
    checked = 0
    for variant, doc in _blocks().items():
        block = doc["sa_ccr"]
        mpor = block["mpor_business_days"]
        result = s.ir_addon(_trades_from_block(block), mpor_business_days=mpor, offset=True)
        assert len(result.trades) == len(block["trades"])
        for got, want in zip(result.trades, block["trades"], strict=True):
            assert got.hedging_set == want["hedging_set"]
            assert got.supervisory_duration == pytest.approx(want["supervisory_duration"], rel=1e-8)
            assert got.adjusted_notional == pytest.approx(want["adjusted_notional"]["amount"], rel=1e-9)
            assert got.maturity_factor == pytest.approx(want["maturity_factor"], rel=1e-8)
            assert got.bucket == want["bucket"]
            assert got.effective_notional == pytest.approx(want["effective_notional"], rel=1e-8)
            checked += 1
        for got, want in zip(result.hedging_sets, block["hedging_sets"], strict=True):
            assert got.hedging_set == want["hedging_set"] and got.offset is want["offset"]
            assert got.d_b1 == pytest.approx(want["d_b1"], rel=1e-8)
            assert got.d_b2 == pytest.approx(want["d_b2"], rel=1e-8)
            assert got.d_b3 == pytest.approx(want["d_b3"], rel=1e-8)
            assert got.effective_notional == pytest.approx(want["effective_notional"], rel=1e-8)
            assert got.supervisory_factor == want["supervisory_factor"]
            assert got.addon == pytest.approx(want["addon"], rel=1e-8)
        assert result.addon == pytest.approx(block["addon_by_class"]["interest_rate"], rel=1e-8), variant
        # the whole block: RC, aggregate add-on, multiplier, PFE and EAD from the same trades
        v, c = block["v"]["amount"], block["c"]["amount"]
        res = s.netting_set_ead(
            v,
            c,
            s.AddOnByClass(interest_rate=result.addon),
            margined=block["margined"],
            th=block["th"]["amount"],
            mta=block["mta"]["amount"],
            nica=block["nica"]["amount"],
            alpha=block["alpha"],
        )
        assert res.rc == pytest.approx(block["rc"]["amount"], rel=1e-9, abs=1e-6)
        assert res.addon_aggregate == pytest.approx(block["addon_aggregate"]["amount"], rel=1e-8)
        assert res.multiplier == pytest.approx(block["multiplier"], rel=1e-9)
        assert res.pfe == pytest.approx(block["pfe"]["amount"], rel=1e-8)
        assert res.ead == pytest.approx(block["ead"]["amount"], rel=1e-8)
    assert len(_blocks()) == 4 and checked == sum(len(d["sa_ccr"]["trades"]) for d in _blocks().values())


def test_fx_addon():
    assert s.fx_addon({"EURUSD": 1_000.0}) == pytest.approx(40.0)
    assert s.fx_addon({"EURUSD": 1_000.0, "USDJPY": -500.0}) == pytest.approx(0.04 * 1_500.0)
    assert s.fx_addon({"EURUSD": 0.0}) == 0.0 and s.fx_addon({}) == 0.0
    base = {"EURUSD": 750.0, "USDCLP": -250.0, "GBPUSD": 10.0}
    for k in (0.0, 0.5, 3.0, 12.5):
        scaled = {key: k * v for key, v in base.items()}
        assert s.fx_addon(scaled) == pytest.approx(k * s.fx_addon(base), rel=1e-13)
    with pytest.raises(ValueError, match="hedging_sets"):
        s.fx_addon({"EURUSD": float("nan")})


def test_entity_aggregate():
    assert s.entity_aggregate([42.0], [0.3]) == pytest.approx(42.0)
    assert s.entity_aggregate([-42.0], [0.9]) == pytest.approx(42.0)  # one entity: its own add-on in absolute value
    a = [105.86194, -279.91632, 168.11140]
    rho = [0.5, 0.5, 0.8]
    sys_term = sum(r * x for r, x in zip(rho, a, strict=True)) ** 2
    idio = sum((1 - r**2) * x**2 for r, x in zip(rho, a, strict=True))
    assert sys_term == pytest.approx(2252.635, abs=2e-3) and idio == pytest.approx(77344.043, abs=2e-3)
    assert s.entity_aggregate(a, rho) == pytest.approx(282.128832, abs=2e-5)
    assert s.entity_aggregate(a, rho) == pytest.approx(math.sqrt(sys_term + idio), rel=1e-13)
    assert s.entity_aggregate([3.0, -4.0, 5.0], [1.0, 1.0, 1.0]) == pytest.approx(4.0)  # |sum A|
    assert s.entity_aggregate([3.0, 4.0], [0.0, 0.0]) == pytest.approx(5.0)  # sqrt(sum A^2)
    assert s.entity_aggregate([], []) == 0.0
    with pytest.raises(ValueError, match="rhos"):
        s.entity_aggregate([1.0, 2.0], [0.5])
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="rhos"):
            s.entity_aggregate([1.0], [bad])
    with pytest.raises(ValueError, match="addons"):
        s.entity_aggregate([float("inf")], [0.5])


def test_credit_equity_commodity_addons():
    sf = {"AAA": 0.0038, "AA": 0.0038, "A": 0.0042, "BBB": 0.0054, "BB": 0.0106, "B": 0.016, "CCC": 0.06}
    for rating, factor in sf.items():
        one = s.credit_addon([s.CreditEntity(rating, False, 1000.0)])
        assert one == pytest.approx(factor * 1000.0, rel=1e-13), rating
    assert s.credit_addon([s.CreditEntity("IG", True, 1000.0)]) == pytest.approx(0.0038 * 1000.0, rel=1e-13)
    assert s.credit_addon([s.CreditEntity("SG", True, -1000.0)]) == pytest.approx(0.0106 * 1000.0, rel=1e-13)
    entities = [
        s.CreditEntity("AA", False, 27858.4),
        s.CreditEntity("BBB", False, -51837.0),
        s.CreditEntity("IG", True, 44238.8),
    ]
    addons = [0.0038 * 27858.4, 0.0054 * -51837.0, 0.0038 * 44238.8]
    assert s.credit_addon(entities) == pytest.approx(s.entity_aggregate(addons, [0.5, 0.5, 0.8]), rel=1e-13)
    assert s.credit_addon([]) == 0.0
    with pytest.raises(ValueError, match="rating"):
        s.credit_addon([s.CreditEntity("D", False, 1.0)])
    with pytest.raises(ValueError, match="rating"):
        s.credit_addon([s.CreditEntity("AAA", True, 1.0)])
    # equity: 32 percent and 50 percent for a single name, 20 percent and 80 percent for an index
    ents = [s.EquityEntity(False, 100.0), s.EquityEntity(False, -60.0), s.EquityEntity(True, 250.0)]
    want = s.entity_aggregate([0.32 * 100.0, 0.32 * -60.0, 0.20 * 250.0], [0.5, 0.5, 0.8])
    assert s.equity_addon(ents) == pytest.approx(want, rel=1e-13)
    assert s.equity_addon([s.EquityEntity(False, 10.0)]) == pytest.approx(3.2)
    assert s.equity_addon([]) == 0.0
    # commodity: the sum over hedging sets of the entity aggregate (the CRE99 netting set 3 shape)
    energy = [s.CommodityEntity(0.18, 0.4, -11339.745962155614)]
    metals = [s.CommodityEntity(0.18, 0.4, 10000.0)]
    assert s.commodity_addon({"energy": energy}) == pytest.approx(2041.1542731880104, rel=1e-12)
    assert s.commodity_addon({"metals": metals}) == pytest.approx(1800.0, rel=1e-12)
    assert s.commodity_addon({"energy": energy, "metals": metals}) == pytest.approx(3841.1542731880104, rel=1e-12)
    two = [s.CommodityEntity(0.18, 0.4, 100.0), s.CommodityEntity(0.4, 0.4, -50.0)]
    assert s.commodity_addon({"energy": two}) == pytest.approx(s.entity_aggregate([18.0, -20.0], [0.4, 0.4]), rel=1e-13)
    assert s.commodity_addon({}) == 0.0
    with pytest.raises(ValueError, match="rho"):
        s.commodity_addon({"energy": [s.CommodityEntity(0.18, 1.5, 1.0)]})
    with pytest.raises(ValueError, match="sf"):
        s.commodity_addon({"energy": [s.CommodityEntity(-0.18, 0.4, 1.0)]})


def test_replacement_cost():
    assert s.replacement_cost(30 - 20 + 50, 0.0) == 60.0  # CRE99 netting set 1
    assert s.replacement_cost(80, 200, th=0, mta=5, nica=150, margined=True) == 0.0  # netting set 5
    assert s.replacement_cost(20, 0.0) == 20.0 and s.replacement_cost(40, 0.0) == 40.0
    assert s.replacement_cost(-20, 0.0) == 0.0
    # margined: max(V - C, TH + MTA - NICA, 0)
    assert s.replacement_cost(10, 0, th=50, mta=10, nica=20, margined=True) == 40.0
    assert s.replacement_cost(100, 0, th=50, mta=10, nica=20, margined=True) == 100.0
    assert s.replacement_cost(-5, 0, th=0, mta=0, nica=100, margined=True) == 0.0
    # unmargined ignores the margin terms
    assert s.replacement_cost(10, 0, th=50, mta=10, nica=20, margined=False) == 10.0
    with pytest.raises(ValueError, match="th"):
        s.replacement_cost(1.0, 0.0, th=float("nan"))


def test_multiplier():
    assert s.multiplier(-20.0, 282.128832) == pytest.approx(0.965208, abs=5e-7)
    assert s.multiplier(-120.0, 1400.96238) == pytest.approx(0.958123, abs=5e-7)
    want = min(1.0, 0.05 + 0.95 * math.exp(-20.0 / (2 * 0.95 * 282.1288318596666)))
    assert s.multiplier(-20.0, 282.1288318596666) == pytest.approx(want, rel=1e-13)
    assert s.multiplier(0.0, 100.0) == 1.0 and s.multiplier(50.0, 100.0) == 1.0
    assert s.multiplier(-10.0, 0.0) == 1.0  # a zero add-on returns 1 (contract)
    xs = [-1000.0 + 50.0 * i for i in range(0, 41)]
    values = [s.multiplier(x, 100.0) for x in xs]
    assert values == sorted(values) and all(0.05 <= v <= 1.0 for v in values)
    assert s.multiplier(-1e9, 100.0) == pytest.approx(0.05, abs=1e-12)  # tends to the floor
    assert s.multiplier(-20.0, 282.128832, floor=0.1) == pytest.approx(
        0.1 + 0.9 * math.exp(-20.0 / (2 * 0.9 * 282.128832))
    )
    with pytest.raises(ValueError, match="floor"):
        s.multiplier(-1.0, 1.0, floor=1.0)
    with pytest.raises(ValueError, match="addon_aggregate"):
        s.multiplier(-1.0, -1.0)


def test_netting_set_ead_invariants():
    addons = s.AddOnByClass(interest_rate=100.0, fx=20.0, credit=30.0, equity=0.0, commodity=50.0)
    res = s.netting_set_ead(10.0, 0.0, addons)
    assert res.addon_aggregate == 200.0 and res.rc == 10.0 and res.multiplier == 1.0
    assert res.pfe == pytest.approx(200.0) and res.ead == pytest.approx(1.4 * (10.0 + 200.0))
    # ead = alpha (rc + pfe), alpha 1.4 by default and a parameter
    neg = s.netting_set_ead(-50.0, 0.0, addons)
    assert neg.multiplier == pytest.approx(s.multiplier(-50.0, 200.0), rel=1e-14)
    assert neg.ead == pytest.approx(1.4 * (neg.rc + neg.pfe), rel=1e-14) and neg.pfe == pytest.approx(
        neg.multiplier * 200.0
    )
    assert s.netting_set_ead(10.0, 0.0, addons, alpha=1.0).ead == pytest.approx(210.0)
    # margined path uses the margined replacement cost
    m = s.netting_set_ead(80.0, 200.0, addons, margined=True, th=0.0, mta=5.0, nica=150.0)
    assert m.rc == 0.0 and m.multiplier == pytest.approx(s.multiplier(-120.0, 200.0), rel=1e-14)
    # an empty netting set has zero exposure
    zero = s.netting_set_ead(0.0, 0.0, s.AddOnByClass())
    assert (zero.rc, zero.addon_aggregate, zero.pfe, zero.ead) == (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        s.netting_set_ead(1.0, 0.0, addons, alpha=0.0)
    with pytest.raises(ValueError, match="interest_rate"):
        s.netting_set_ead(1.0, 0.0, s.AddOnByClass(interest_rate=-1.0))
    with pytest.raises(ValueError, match="v"):
        s.netting_set_ead(float("nan"), 0.0, addons)
