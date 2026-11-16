"""RV-2006 to RV-2009: the default risk capital requirement for non-securitisations (MAR22.1 to MAR22.26). The Basel
text holds one numeric example of the gross JTD (the FAQ under MAR22.15 and MAR22.16) and none of the bucket formula;
the fixtures marked DERIVED are computed here by loops and by a linear program, written apart from the engine, and never
typed as answers."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

from riskvalidation.regulatory import drc

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "regulatory" / "u9" / "07_drc.md"

#: MAR22.24 Table 2 as printed on the Basel Framework page (bis.org, read 2026-11-15)
TABLE_2 = """\
ROW: Default risk weights for non-securitisations by credit quality category | Table 2
ROW: Credit quality category | Default risk weight
ROW: AAA | 0.5%
ROW: AA | 2%
ROW: A | 3%
ROW: BBB | 6%
ROW: BB | 15%
ROW: B | 30%
ROW: CCC | 50%
ROW: Unrated | 15%
ROW: Defaulted | 100%
ROW:
"""

RW = {
    r: w
    for r, w in zip(
        ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "unrated", "defaulted"],
        [0.005, 0.02, 0.03, 0.06, 0.15, 0.30, 0.50, 0.15, 1.00],
        strict=True,
    )
}


def test_parameters_pinned_to_mar22():
    rows = [[c.strip() for c in line[4:].split("|")] for line in TABLE_2.strip("\n").split("\n")[2:-1]]
    assert len(rows) == 9
    assert tuple(drc.RATINGS) == tuple(r[0].lower() if r[0] == "Unrated" or r[0] == "Defaulted" else r[0] for r in rows)
    for name, weight in rows:
        key = name.lower() if name in ("Unrated", "Defaulted") else name
        rp = drc.DEFAULT_RISK_WEIGHTS[key]
        assert rp.value == float(weight.rstrip("%")) / 100.0, key
        assert rp.paragraph == "MAR22.24" and "Table 2" in rp.table and rp.in_force == "2023-01-01"
    assert len(drc.DEFAULT_RISK_WEIGHTS) == 9
    # LGD (MAR22.12), the horizon and the maturity floor (MAR22.15, MAR22.18), the buckets (MAR22.22)
    assert {k: (v.value, v.paragraph) for k, v in drc.LGD.items()} == {
        "equity_or_non_senior": (1.0, "MAR22.12"),
        "senior": (0.75, "MAR22.12"),
        "covered_bond": (0.25, "MAR22.12"),
    }
    assert (drc.HORIZON_YEARS.value, drc.HORIZON_YEARS.paragraph) == (1.0, "MAR22.15")
    assert (drc.MATURITY_FLOOR_YEARS.value, drc.MATURITY_FLOOR_YEARS.paragraph) == (0.25, "MAR22.18")
    assert tuple(drc.BUCKETS) == ("corporates", "sovereigns", "local_governments")
    assert all(rp.paragraph == "MAR22.22" for rp in drc.BUCKETS.values())
    assert [rp.value for rp in drc.BUCKETS.values()] == [1.0, 2.0, 3.0]
    # the seniority ladder behind MAR22.19 follows the four classes of MAR22.12, the most junior first
    assert list(drc.SENIORITY) == ["equity", "non_senior_debt", "senior_debt", "covered_bond"]
    assert sorted(drc.SENIORITY.values()) == [0, 1, 2, 3]
    assert drc.SENIORITY["equity"] < drc.SENIORITY["non_senior_debt"] < drc.SENIORITY["senior_debt"]


def _ref_gross(direction, notional, pnl, lgd, maturity):
    raw = lgd * notional + pnl
    base = max(raw, 0.0) if direction == "long" else min(raw, 0.0)
    return base * min(1.0, max(0.25, maturity))


def test_gross_jtd_formula_and_the_basel_faq():
    g = drc.gross_jtd
    # DRC-B: a long senior bond of notional 100 with no P&L: 0.75 x 100; a six month maturity halves it
    assert g("long", 100.0, 0.0, "senior", 2.0) == 75.0
    assert g("long", 100.0, 0.0, "senior", 1.0) == 75.0
    assert g("long", 100.0, 0.0, "senior", 0.5) == 37.5
    # the maturity weight is floored at one quarter: a one month position counts as three months (MAR22.18)
    assert g("long", 100.0, 0.0, "senior", 1.0 / 12.0) == 18.75
    assert g("long", 100.0, 0.0, "senior", 0.0) == 18.75
    # the formula of MAR22.11 with a P&L already taken: a loss is recorded negative (MAR22.13)
    assert g("long", 100.0, -20.0, 1.0, 5.0) == 80.0
    assert g("long", 100.0, -150.0, 1.0, 5.0) == 0.0  # the max with zero
    assert g("long", 100.0, 10.0, 0.75, 5.0) == 85.0
    assert g("short", -100.0, 10.0, 0.75, 5.0) == -65.0
    assert g("short", -100.0, 80.0, 0.75, 5.0) == 0.0  # the min with zero
    assert g("short", -100.0, -5.0, 0.75, 5.0) == -80.0
    # an LGD as a number, the three classes by name, a bought call option (notional 0) takes its P&L only
    assert g("long", 100.0, 0.0, 0.25, 5.0) == g("long", 100.0, 0.0, "covered_bond", 5.0) == 25.0
    assert g("long", 100.0, 0.0, "equity_or_non_senior", 5.0) == 100.0
    assert g("long", 0.0, 4.0, "senior", 5.0) == 4.0
    # MAR22.12 last bullet: a price not linked to the recovery rate takes no LGD
    assert g("long", 100.0, 0.0, "senior", 5.0, recovery_linked=False) == 100.0
    # MAR22.13: terms that allow unwinding with no exposure to default risk give zero
    assert g("long", 100.0, 0.0, "senior", 0.5, unwind_allowed=True) == 0.0
    assert g("short", -100.0, 0.0, 1.0, 0.5, unwind_allowed=True) == 0.0
    # against the formula, over a grid
    for direction, notional, pnl, lgd, maturity in itertools.product(
        ("long", "short"), (0.0, 50.0), (-80.0, -10.0, 0.0, 25.0, 90.0), (0.25, 0.75, 1.0), (0.05, 0.25, 0.6, 1.0, 3.0)
    ):
        signed = notional if direction == "long" else -notional
        assert g(direction, signed, pnl, lgd, maturity) == pytest.approx(
            _ref_gross(direction, signed, pnl, lgd, maturity), abs=1e-12
        )
    # the Basel FAQ under MAR22.15 and MAR22.16: index futures of -10 and the underlying of +10, both on a three-month
    # maturity (the one month future is floored): 1/4 x 10 - 1/4 x 10 = 0
    longs = g("long", 10.0, 0.0, 1.0, 0.25)
    shorts = g("short", -10.0, 0.0, 1.0, 1.0 / 12.0)
    assert (longs, shorts) == (2.5, -2.5)
    ex = [
        drc.DrcExposure("EQ-INDEX", "corporates", "BBB", "equity", shorts),
        drc.DrcExposure("EQ-INDEX", "corporates", "BBB", "equity", longs),
    ]
    (net,) = drc.net_jtd(ex)
    assert net.net_long == 0.0 and net.net_short == 0.0
    assert longs + shorts == 0.0
    # refusals naming the field
    with pytest.raises(ValueError, match="notional"):
        g("long", float("nan"), 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="pnl"):
        g("long", 1.0, float("inf"), 1.0, 1.0)
    with pytest.raises(ValueError, match="maturity_years"):
        g("long", 1.0, 0.0, 1.0, float("nan"))
    with pytest.raises(ValueError, match="maturity_years"):
        g("long", 1.0, 0.0, 1.0, -0.5)
    with pytest.raises(ValueError, match="direction"):
        g("flat", 1.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="'junior'"):
        g("long", 1.0, 0.0, "junior", 1.0)
    with pytest.raises(ValueError, match="lgd"):
        g("long", 1.0, 0.0, 1.5, 1.0)
    with pytest.raises(ValueError, match="notional"):
        g("long", -1.0, 0.0, 1.0, 1.0)  # a long carries a positive notional (MAR22.13)
    with pytest.raises(ValueError, match="notional"):
        g("short", 1.0, 0.0, 1.0, 1.0)  # a short carries a negative one


def _ref_offset(longs, shorts):
    """The largest total offset by a linear program: x_ij >= 0 on the pairs where the short is not more senior than
    the long (MAR22.19), sum_j x_ij <= L_i and sum_i x_ij <= S_j, maximise sum x."""
    pairs = [(i, j) for i, (_, sl) in enumerate(longs) for j, (_, ss) in enumerate(shorts) if ss <= sl]
    if not pairs:
        return 0.0
    c = -np.ones(len(pairs))
    a, b = [], []
    for i, (_, _) in enumerate(longs):
        a.append([1.0 if p[0] == i else 0.0 for p in pairs])
        b.append(longs[i][0])
    for j, (_, _) in enumerate(shorts):
        a.append([1.0 if p[1] == j else 0.0 for p in pairs])
        b.append(shorts[j][0])
    res = linprog(c, A_ub=a, b_ub=b, bounds=(0, None), method="highs")
    assert res.status == 0
    return float(-res.fun)


def test_offsetting_by_seniority():
    E = drc.DrcExposure

    def one(exposures):
        (n,) = drc.net_jtd(exposures)
        return n.net_long, n.net_short

    # MAR22.19: "a short exposure in an equity may offset a long exposure in a bond, but a short exposure in a bond
    # cannot offset a long exposure in the equity"
    assert one([E("A", "corporates", "BBB", "senior_debt", 75.0), E("A", "corporates", "BBB", "equity", -60.0)]) == (
        15.0,
        0.0,
    )
    assert one([E("A", "corporates", "BBB", "equity", 100.0), E("A", "corporates", "BBB", "senior_debt", -75.0)]) == (
        100.0,
        -75.0,
    )
    # the same seniority offsets in full; a short larger than the long leaves a net short
    assert one(
        [E("A", "corporates", "BBB", "senior_debt", 40.0), E("A", "corporates", "BBB", "senior_debt", -70.0)]
    ) == (
        0.0,
        -30.0,
    )
    # a short of the lowest rank offsets any long of the same obligor, a short of a covered bond only a covered bond
    assert one(
        [E("A", "sovereigns", "AA", "covered_bond", 25.0), E("A", "sovereigns", "AA", "senior_debt", -25.0)]
    ) == (
        0.0,
        0.0,
    )
    assert one(
        [E("A", "sovereigns", "AA", "senior_debt", 25.0), E("A", "sovereigns", "AA", "covered_bond", -25.0)]
    ) == (
        25.0,
        -25.0,
    )
    # a nested case that a naive order gets wrong: the senior short must take the senior long, the equity short the
    # equity long; the total offset is 80, not 40
    nested = [
        E("A", "corporates", "A", "equity", 40.0),
        E("A", "corporates", "A", "senior_debt", 40.0),
        E("A", "corporates", "A", "equity", -40.0),
        E("A", "corporates", "A", "senior_debt", -40.0),
    ]
    assert one(nested) == (0.0, 0.0)
    assert one(nested[::-1]) == (0.0, 0.0)
    # positions under one year are weighted by their maturity over the one-year horizon (MAR22.19): a three-month short
    # of 75 against a long of 75 of more than one year offsets only a quarter of its size
    long_bond = drc.gross_jtd("long", 100.0, 0.0, "senior", 2.0)
    short_equity = drc.gross_jtd("short", -75.0, 0.0, "equity_or_non_senior", 0.25)
    assert (long_bond, short_equity) == (75.0, -18.75)
    assert one(
        [E("A", "corporates", "BBB", "senior_debt", long_bond), E("A", "corporates", "BBB", "equity", short_equity)]
    ) == (
        56.25,
        0.0,
    )
    # both legs under one year: the scaling applies to both (MAR22.20): 75 x 1/2 against -75 x 1/4
    lg = drc.gross_jtd("long", 100.0, 0.0, "senior", 0.5)
    sh = drc.gross_jtd("short", -100.0, 0.0, "senior", 0.25)
    assert (lg, sh) == (37.5, -18.75)
    assert one([E("A", "corporates", "BBB", "senior_debt", lg), E("A", "corporates", "BBB", "senior_debt", sh)]) == (
        18.75,
        0.0,
    )
    # different obligors never offset; the result keeps the obligor, the bucket and the rating category
    out = drc.net_jtd(
        [
            E("A", "corporates", "BBB", "senior_debt", 10.0),
            E("B", "corporates", "BB", "senior_debt", -10.0),
            E("C", "sovereigns", "AAA", "senior_debt", 5.0),
        ]
    )
    assert {(n.obligor, n.bucket, n.rating, n.net_long, n.net_short) for n in out} == {
        ("A", "corporates", "BBB", 10.0, 0.0),
        ("B", "corporates", "BB", 0.0, -10.0),
        ("C", "sovereigns", "AAA", 5.0, 0.0),
    }
    # a number as a seniority rank (higher is more senior) works as the names do
    assert one([E("A", "corporates", "BBB", 2, 50.0), E("A", "corporates", "BBB", 1, -80.0)]) == (0.0, -30.0)
    # the totals equal the linear program on random books, whatever the order of the exposures
    rng = np.random.default_rng(15112026)
    for _ in range(300):
        n = int(rng.integers(1, 8))
        rows = [
            (float(rng.integers(1, 100)) * (1 if rng.random() < 0.5 else -1), int(rng.integers(0, 4))) for _ in range(n)
        ]
        names = list(drc.SENIORITY)
        book = [E("A", "corporates", "BB", names[s], v) for v, s in rows]
        rng.shuffle(book)
        longs = [(v, s) for v, s in rows if v > 0]
        shorts = [(-v, s) for v, s in rows if v < 0]
        offset = _ref_offset(longs, shorts)
        total_long, total_short = sum(v for v, _ in longs), sum(v for v, _ in shorts)
        (net,) = drc.net_jtd(book)
        assert net.net_long == pytest.approx(total_long - offset, abs=1e-9)
        assert net.net_short == pytest.approx(-(total_short - offset), abs=1e-9)
    # refusals
    with pytest.raises(ValueError, match="rating"):
        drc.net_jtd([E("A", "corporates", "BBBB", "senior_debt", 1.0)])
    with pytest.raises(ValueError, match="bucket"):
        drc.net_jtd([E("A", "banks", "BBB", "senior_debt", 1.0)])
    with pytest.raises(ValueError, match="'preferred'"):
        drc.net_jtd([E("A", "corporates", "BBB", "preferred", 1.0)])
    with pytest.raises(ValueError, match="jtd"):
        drc.net_jtd([E("A", "corporates", "BBB", "equity", float("nan"))])
    with pytest.raises(ValueError, match="obligor 'A'"):
        drc.net_jtd([E("A", "corporates", "BBB", "equity", 1.0), E("A", "sovereigns", "BBB", "equity", 1.0)])
    with pytest.raises(ValueError, match="obligor 'A'"):
        drc.net_jtd([E("A", "corporates", "BBB", "equity", 1.0), E("A", "corporates", "BB", "equity", 1.0)])
    assert drc.net_jtd([]) == []


def _ref_bucket(long_by_rating, short_by_rating):
    long_sum = sum(long_by_rating.values())
    short_abs = sum(abs(v) for v in short_by_rating.values())
    hbr = long_sum / (long_sum + short_abs) if long_sum + short_abs > 0 else 0.0
    weighted_long = sum(RW[r] * v for r, v in long_by_rating.items())
    weighted_short = sum(RW[r] * abs(v) for r, v in short_by_rating.items())
    return max(weighted_long - hbr * weighted_short, 0.0)


def test_hbr_bucket_and_total():
    # MAR22.23: the net long sum over the net long plus the absolute net short sum
    assert drc.hedge_benefit_ratio(100.0, 50.0) == pytest.approx(2.0 / 3.0, rel=1e-15)
    assert drc.hedge_benefit_ratio(10.0, 0.0) == 1.0
    assert drc.hedge_benefit_ratio(0.0, 10.0) == 0.0
    # the text leaves 0 / 0 undefined: the engine defines it as 0, and the bucket's charge is then 0
    assert drc.hedge_benefit_ratio(0.0, 0.0) == 0.0
    assert drc.bucket_drc({}, {}) == 0.0
    with pytest.raises(ValueError, match="net_long_sum"):
        drc.hedge_benefit_ratio(-1.0, 0.0)
    with pytest.raises(ValueError, match="net_short_abs_sum"):
        drc.hedge_benefit_ratio(0.0, float("nan"))

    # DRC-C: net long 100 in BBB (6%), net short -50 in BB (15%): HBR = 2/3, DRC = 6 - (2/3) x 7.5 = 1
    assert drc.bucket_drc({"BBB": 100.0}, {"BB": -50.0}) == pytest.approx(1.0, rel=1e-14)
    assert drc.bucket_drc({"BBB": 100.0}, {"BB": -50.0}) == pytest.approx(
        _ref_bucket({"BBB": 100}, {"BB": -50}), rel=1e-14
    )
    # DRC-D: the floor at zero: long 10 in AAA, short -100 in CCC
    assert drc.bucket_drc({"AAA": 10.0}, {"CCC": -100.0}) == 0.0
    assert _ref_bucket({"AAA": 10.0}, {"CCC": -100.0}) == 0.0
    # no short: the risk-weighted long; no long: zero (HBR 0)
    assert drc.bucket_drc({"BBB": 100.0, "unrated": 10.0}, {}) == pytest.approx(6.0 + 1.5)
    assert drc.bucket_drc({}, {"BBB": -100.0}) == 0.0
    # several rating categories on both sides against the loops
    rng = np.random.default_rng(22112026)
    names = list(RW)
    for _ in range(200):
        long_r = {
            r: float(rng.integers(0, 200)) for r in rng.choice(names, size=int(rng.integers(0, 5)), replace=False)
        }
        short_r = {
            r: -float(rng.integers(0, 200)) for r in rng.choice(names, size=int(rng.integers(0, 5)), replace=False)
        }
        got = drc.bucket_drc(long_r, short_r)
        assert got >= 0.0
        assert got == pytest.approx(_ref_bucket(long_r, short_r), rel=1e-12, abs=1e-12)
    with pytest.raises(ValueError, match="'BBBB'"):
        drc.bucket_drc({"BBBB": 1.0}, {})
    with pytest.raises(ValueError, match="long_by_rating"):
        drc.bucket_drc({"BBB": -1.0}, {})
    with pytest.raises(ValueError, match="short_by_rating"):
        drc.bucket_drc({}, {"BBB": 1.0})
    with pytest.raises(ValueError, match="long_by_rating"):
        drc.bucket_drc({"BBB": float("inf")}, {})

    # MAR22.26 and DRC-E: the plain sum of the buckets, no hedging between them
    one = ({"BBB": 100.0}, {"BB": -50.0})
    res = drc.drc_non_securitisation({"corporates": one, "sovereigns": one, "local_governments": one})
    assert res.charge == pytest.approx(3.0, rel=1e-14)
    assert set(res.bucket_charges) == {"corporates", "sovereigns", "local_governments"}
    assert all(v == pytest.approx(1.0, rel=1e-14) for v in res.bucket_charges.values())
    # a long in one bucket and the same short in another do not hedge each other: 6, not 0
    res = drc.drc_non_securitisation({"corporates": ({"BBB": 100.0}, {}), "sovereigns": ({}, {"BBB": -100.0})})
    assert res.charge == pytest.approx(6.0) and res.bucket_charges["sovereigns"] == 0.0
    assert res.bucket_charges["local_governments"] == 0.0  # a bucket not passed has no charge
    assert drc.drc_non_securitisation({}).charge == 0.0
    assert "MAR22.26" in res.paragraphs["charge"] and "MAR22.4" in res.paragraphs["securitisations"]
    with pytest.raises(ValueError, match="'banks'"):
        drc.drc_non_securitisation({"banks": one})

    # from exposures to the charge: gross JTD, offsetting, buckets, HBR, the sum
    book = [
        drc.DrcExposure("A", "corporates", "BBB", "senior_debt", drc.gross_jtd("long", 100.0, 0.0, "senior", 3.0)),
        drc.DrcExposure("A", "corporates", "BBB", "equity", drc.gross_jtd("short", -20.0, 0.0, 1.0, 3.0)),
        drc.DrcExposure("B", "corporates", "BB", "senior_debt", drc.gross_jtd("short", -100.0, 0.0, "senior", 3.0)),
        drc.DrcExposure("S", "sovereigns", "AA", "senior_debt", drc.gross_jtd("long", 200.0, 0.0, "senior", 3.0)),
    ]
    got = drc.drc_from_exposures(book)
    # A: long 75, short equity -20 offsets: net long 55 (BBB); B: net short -75 (BB); S: net long 150 (AA)
    corp = max(0.06 * 55.0 - (55.0 / (55.0 + 75.0)) * 0.15 * 75.0, 0.0)
    sov = 0.02 * 150.0
    assert got.bucket_charges["corporates"] == pytest.approx(corp, rel=1e-14)
    assert got.bucket_charges["sovereigns"] == pytest.approx(sov, rel=1e-14)
    assert got.charge == pytest.approx(corp + sov, rel=1e-14)
    assert math.isfinite(got.charge)


def test_stated_limits_and_page():
    doc = (drc.__doc__ or "").lower()
    for phrase in (
        "securitisation",
        "no diversification",
        "mar22.4",
        "correlation trading",
        "no compliance claim",
        "hedge benefit",
    ):
        assert phrase in doc, phrase
    page = PAGE.read_text(encoding="utf-8").lower()
    for phrase in (
        "https://www.bis.org/basel_framework/chapter/mar/22.htm?inforce=20230101&published=20200327",
        "mar22.11",
        "mar22.19",
        "mar22.25",
        "mar22.26",
        "hedge benefit ratio",
        "defined as 0",
        "securitisation",
        "same or lower seniority",
    ):
        assert phrase in page, phrase
    assert "—" not in page and "—" not in drc.__doc__
    readme = (ROOT / "docs" / "regulatory" / "u9" / "README.md").read_text(encoding="utf-8")
    assert readme.count("07_drc.md") == 1
