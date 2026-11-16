"""RV-2001 to RV-2005: SA-CVA (MAR50.27 to MAR50.77). The tables are pinned cell by cell against the verbatim rows of
the Basel Framework page as read on bis.org on 2026-11-15 (`TABLE_<n>` below are its `ROW:` lines as carried by the
research report of the Contraste initiative, an en-dash typed as a hyphen); the Basel text holds no numeric example of
the SA-CVA, so the DERIVED fixtures are computed here by loops written apart from the engine's matrix code and never
typed as answers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from riskvalidation.regulatory import sa_cva as sa

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "regulatory" / "u9" / "06_sa-cva.md"


TABLE_3 = """\
ROW: Risk weight for interest rate risk (specified currencies) | Table 3
ROW: Risk factor | 1 year | 2 years | 5 years | 10 years | 30 years | Inflation
ROW: Risk weight | 1.11% | 0.93% | 0.74% | 0.74% | 0.74% | 1.11%
ROW: | | | | | |
"""  # noqa: E501
TABLE_4 = """\
ROW: Correlations for interest rate risk factors (specified currencies) | Table 4
ROW: | 1 year | 2 years | 5 years | 10 years | 30 years | Inflation
ROW: 1 year | 100% | 91% | 72% | 55% | 31% | 40%
ROW: 2 years | | 100% | 87% | 72% | 45% | 40%
ROW: 5 years | | | 100% | 91% | 68% | 40%
ROW: 10 years | | | | 100% | 83% | 40%
ROW: 30 years | | | | | 100% | 40%
ROW: Inflation | | | | | | 100%
ROW: | | | | | |
"""  # noqa: E501
TABLE_5 = """\
ROW: Buckets for counterparty credit spread delta risk | Table 5
ROW: Bucket number | Sector
ROW: 1 | a) Sovereigns including central banks, multilateral development banks
ROW: b) Local government, government-backed non-financials, education and public administration
ROW: 2 | Financials including government-backed financials
ROW: 3 | Basic materials, energy, industrials, agriculture, manufacturing, mining and quarrying
ROW: 4 | Consumer goods and services, transportation and storage, administrative and support service activities
ROW: 5 | Technology, telecommunications
ROW: 6 | Health care, utilities, professional and technical activities
ROW: 7 | Other sector
ROW: 8 | Qualified Indices
ROW: | |
"""  # noqa: E501
TABLE_6 = """\
ROW: Cross-bucket correlations for counterparty credit spread delta risk | | Table 6
ROW: Bucket | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
ROW: 1 | 100% | 10% | 20% | 25% | 20% | 15% | 0% | 45%
ROW: 2 | | 100% | 5% | 15% | 20% | 5% | 0% | 45%
ROW: 3 | | | 100% | 20% | 25% | 5% | 0% | 45%
ROW: 4 | | | | 100% | 25% | 5% | 0% | 45%
ROW: 5 | | | | | 100% | 5% | 0% | 45%
ROW: 6 | | | | | | 100% | 0% | 45%
ROW: 7 | | | | | | | 100% | 0%
ROW: 8 | | | | | | | | 100%
ROW: | | | | | | | |
"""  # noqa: E501
TABLE_7 = """\
ROW: Risk weights for counterparty credit spread delta risk | Table 7
ROW: Bucket | 1 a) | 1 b) | 2 | 3 | 4 | 5 | 6 | 7 | 8
ROW: IG names | 0.5% | 1.0% | 5.0% | 3.0% | 3.0% | 2.0% | 1.5% | 5.0% | 1.5%
ROW: HY and NR names | 2.0% | 4.0% | 12.0% | 7.0% | 8.5% | 5.5% | 5.0% | 12.0% | 5.0%
ROW: | | | | | | | | |
"""  # noqa: E501
TABLE_8 = """\
ROW: Buckets for reference credit spread risk | Table 8
ROW: Bucket number | Credit quality | Sector
ROW: 1 | IG | Sovereigns including central banks, multilateral development banks
ROW: 2 | Local government, government-backed non-financials, education and public administration
ROW: 3 | Financials including government-backed financials
ROW: 4 | Basic materials, energy, industrials, agriculture, manufacturing, mining and quarrying
ROW: 5 | Consumer goods and services, transportation and storage, administrative and support service activities
ROW: 6 | Technology, telecommunications
ROW: 7 | Health care, utilities, professional and technical activities
ROW: 8 | HY and NR | Sovereigns including central banks, multilateral development banks
ROW: 9 | Local government, government-backed non-financials, education and public administration
ROW: 10 | Financials including government-backed financials
ROW: 11 | Basic materials, energy, industrials, agriculture, manufacturing, mining and quarrying
ROW: 12 | Consumer goods and services, transportation and storage, administrative and support service activities
ROW: 13 | Technology, telecommunications
ROW: 14 | Health care, utilities, professional and technical activities
ROW: 15 | (Not applicable) | Other sector
ROW: 16 | IG | Qualified Indices
ROW: 17 | HY | Qualified Indices
ROW: | | |
"""  # noqa: E501
TABLE_9 = """\
ROW: Cross-bucket correlations for reference credit spread risk | | | Table 9
ROW: Bucket | 1/8 | 2/9 | 3/10 | 4/11 | 5/12 | 6/13 | 7/14 | 15 | 16 | 17
ROW: 1/8 | 100% | 75% | 10% | 20% | 25% | 20% | 15% | 0% | 45% | 45%
ROW: 2/9 | | 100% | 5% | 15% | 20% | 15% | 10% | 0% | 45% | 45%
ROW: 3/10 | | | 100% | 5% | 15% | 20% | 5% | 0% | 45% | 45%
ROW: 4/11 | | | | 100% | 20% | 25% | 5% | 0% | 45% | 45%
ROW: 5/12 | | | | | 100% | 25% | 5% | 0% | 45% | 45%
ROW: 6/13 | | | | | | 100% | 5% | 0% | 45% | 45%
ROW: 7/14 | | | | | | | 100% | 0% | 45% | 45%
ROW: 15 | | | | | | | | 100% | 0% | 0%
ROW: 16 | | | | | | | | | 100% | 75%
ROW: 17 | | | | | | | | | | 100%
ROW: | | | | | | | | | |
"""  # noqa: E501
TABLE_10 = """\
ROW: Risk weights for reference credit spread delta risk | Table 10
ROW: IG bucket | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
ROW: Risk weight | 0.5% | 1.0% | 5.0% | 3.0% | 3.0% | 2.0% | 1.5% | 2.0% | 4.0%
ROW:
ROW: HY/NR bucket | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
ROW: Risk weight | 12.0% | 7.0% | 8.5% | 5.5% | 5.0% | 12.0% | 1.5% | 5.0% |
"""  # noqa: E501
TABLE_11 = """\
ROW: Buckets for equity risk | Table 11
ROW: Bucket number | Size | Region | Sector
ROW: 1 | Large | Emerging market economies | Consumer goods and services, transportation and storage, administrative and support service activities, healthcare, utilities
ROW: 2 | Telecommunications, industrials
ROW: 3 | Basic materials, energy, agriculture, manufacturing, mining and quarrying
ROW: 4 | Financials including government-backed financials, real estate activities, technology
ROW: 5 | Advanced economies | Consumer goods and services, transportation and storage, administrative and support service activities, healthcare, utilities
ROW: 6 | Telecommunications, industrials
ROW: 7 | Basic materials, energy, agriculture, manufacturing, mining and quarrying
ROW: 8 | Financials including government-backed financials, real estate activities, technology
ROW: 9 | Small | Emerging market economies | All sectors described under bucket numbers 1, 2, 3, and 4
ROW: 10 | Advanced economies | All sectors described under bucket numbers 5, 6, 7, and 8
ROW: 11 | (Not applicable) | Other sector
ROW: 12 | Large cap, advanced economies | Qualified Indices
ROW: 13 | Other | Qualified Indices
ROW: | | | |
"""  # noqa: E501
TABLE_12 = """\
ROW: Risk weights for equity delta risk | Table 12
ROW: Bucket number | Risk weight
ROW: 1 | 55%
ROW: 2 | 60%
ROW: 3 | 45%
ROW: 4 | 55%
ROW: 5 | 30%
ROW: 6 | 35%
ROW: 7 | 40%
ROW: 8 | 50%
ROW: 9 | 70%
ROW: 10 | 50%
ROW: 11 | 70%
ROW: 12 | 15%
ROW: 13 | 25%
ROW: | |
"""  # noqa: E501
TABLE_13 = """\
ROW: Buckets for commodity risk | Table 13
ROW: Bucket number | Commodity group | Examples
ROW: 1 | Energy - Solid combustibles | coal, charcoal, wood pellets, nuclear fuel (such as uranium)
ROW: 2 | Energy - Liquid combustibles | crude oil (such as Light-sweet, heavy, West Texas Intermediate and Brent); biofuels (such as bioethanol and biodiesel); petrochemicals (such as propane, ethane, gasoline, methanol and butane); refined fuels (such as jet fuel, kerosene, gasoil, fuel oil, naphtha, heating oil and diesel)
ROW: 3 | Energy - Electricity and carbon trading | electricity (such as spot, day-ahead, peak and off-peak); carbon emissions trading (such as certified emissions reductions, in-delivery month EU allowance, Regional Greenhouse Gas Initiative CO2 allowance and renewable energy certificates)
ROW: 4 | Freight | dry-bulk route (such as Capesize, Panamax, Handysize and Supramax); liquid-bulk/gas shipping route (such as Suezmax, Aframax and very large crude carriers)
ROW: 5 | Metals - non-precious | base metal (such as aluminium, copper, lead, nickel, tin and zinc); steel raw materials (such as steel billet, steel wire, steel coil, steel scrap and steel rebar, iron ore, tungsten, vanadium, titanium and tantalum); minor metals (such as cobalt, manganese, molybdenum)
ROW: 6 | Gaseous combustibles | natural gas; liquefied natural gas
ROW: 7 | Precious metals (including gold) | gold; silver; platinum; palladium
ROW: 8 | Grains & oilseed | corn; wheat; soybean (such as soybean seed, soybean oil and soybean meal); oats; palm oil; canola; barley; rapeseed (such as rapeseed seed, rapeseed oil, and rapeseed meal); red bean, sorghum; coconut oil; olive oil; peanut oil; sunflower oil; rice
ROW: 9 | Livestock & dairy | cattle (such live and feeder); hog; poultry; lamb; fish; shrimp; dairy (such as milk, whey, eggs, butter and cheese)
ROW: 10 | Softs and other agriculturals | cocoa; coffee (such as arabica and robusta); tea; citrus and orange juice; potatoes; sugar; cotton; wool; lumber and pulp; rubber
ROW: 11 | Other commodity | industrial minerals (such as potash, fertiliser and phosphate rocks), rare earths; terephthalic acid; flat glass
ROW: | | |
"""  # noqa: E501
TABLE_14 = """\
ROW: Risk weights for commodity delta risk | Table 14
ROW: Bucket number | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11
ROW: RW | 30% | 35% | 60% | 80% | 40% | 45% | 20% | 35% | 25% | 35% | 50%
ROW: | | | | | | | | | | |
"""  # noqa: E501


def _rows(table: str) -> list[list[str]]:
    """The `ROW:` lines after the title row as lists of stripped cells."""
    out = []
    for line in table.strip("\n").split("\n")[1:]:
        assert line.startswith("ROW:")
        out.append([c.strip() for c in line[4:].split("|")])
    return out


def _pct(cell: str) -> float:
    return float(cell.rstrip("%")) / 100.0


def _upper(table: str, first_row: int, n: int) -> list[list[float | None]]:
    """The printed upper triangle of a correlation table: the data rows from `first_row`, the label column dropped."""
    rows = _rows(table)[first_row : first_row + n]
    return [[_pct(c) if c else None for c in r[1:]] for r in rows]


def _check_upper(matrix: np.ndarray, upper: list[list[float | None]]) -> None:
    n = len(upper)
    assert matrix.shape == (n, n)
    assert not matrix.flags.writeable
    for i in range(n):
        for j in range(n):
            if j >= i:
                assert upper[i][j] is not None, f"printed cell ({i}, {j}) is blank"
                assert matrix[i, j] == upper[i][j], (i, j)
            else:
                assert upper[i][j] is None, f"lower cell ({i}, {j}) is printed"
            assert matrix[i, j] == matrix[j, i], f"asymmetric at ({i}, {j})"


def test_parameters_pinned_to_mar50():
    # the scalars
    assert (sa.R.value, sa.R.paragraph) == (0.01, "MAR50.53")
    assert (sa.M_CVA.value, sa.M_CVA.paragraph) == (1.0, "MAR50.41")
    assert (sa.RWA_FACTOR.value, sa.RWA_FACTOR.paragraph) == (12.5, "MAR50.1")
    assert sa.R.in_force == sa.M_CVA.in_force == sa.RWA_FACTOR.in_force == "2023-01-01"
    assert sa.IR_GAMMA.value == 0.5 and sa.IR_GAMMA.paragraph == "MAR50.55"
    assert sa.FX_GAMMA.value == 0.6 and sa.FX_GAMMA.paragraph == "MAR50.60"
    assert sa.FX_DELTA_RISK_WEIGHT.value == 0.11 and sa.FX_DELTA_RISK_WEIGHT.paragraph == "MAR50.61"
    assert sa.FX_VEGA_RISK_WEIGHT.value == 1.0 and sa.FX_VEGA_RISK_WEIGHT.paragraph == "MAR50.62"
    assert sa.IR_VEGA_RISK_WEIGHT.value == 1.0 and sa.IR_VEGA_RISK_WEIGHT.paragraph == "MAR50.58"
    assert sa.IR_VEGA_RHO.value == 0.4 and sa.IR_VEGA_RHO.paragraph == "MAR50.58"
    assert sa.RCS_VEGA_RISK_WEIGHT.value == 1.0 and sa.RCS_VEGA_RISK_WEIGHT.paragraph == "MAR50.69"
    assert sa.COM_VEGA_RISK_WEIGHT.value == 1.0 and sa.COM_VEGA_RISK_WEIGHT.paragraph == "MAR50.77"
    assert sa.COM_GAMMA_WITHIN.value == 0.2 and sa.COM_GAMMA_WITHIN.paragraph == "MAR50.75"
    assert sa.SPECIFIED_CURRENCIES == ("USD", "EUR", "GBP", "AUD", "CAD", "SEK", "JPY")  # MAR50.56
    assert sa.IR_DELTA_RHO_OTHER.value == 0.4 and sa.IR_DELTA_RHO_OTHER.paragraph == "MAR50.57"
    assert sa.IR_DELTA_RISK_WEIGHT_OTHER.value == 0.0158 and sa.IR_DELTA_RISK_WEIGHT_OTHER.paragraph == "MAR50.57"

    # Table 3 and Table 4 (specified currencies)
    t3 = _rows(TABLE_3)
    labels = ["1y", "2y", "5y", "10y", "30y", "inflation"]
    assert t3[0][1:] == ["1 year", "2 years", "5 years", "10 years", "30 years", "Inflation"]
    assert tuple(sa.IR_FACTORS_SPECIFIED) == tuple(labels)
    for label, cell in zip(labels, t3[1][1:], strict=True):
        rp = sa.IR_DELTA_RISK_WEIGHTS_SPECIFIED[label]
        assert rp.value == _pct(cell) and rp.paragraph == "MAR50.56" and "Table 3" in rp.table
    assert set(sa.IR_DELTA_RISK_WEIGHTS_SPECIFIED) == set(labels)
    _check_upper(sa.IR_DELTA_CORRELATION_SPECIFIED, _upper(TABLE_4, 1, 6))

    # Table 5: eight buckets, bucket 1 in two sub-buckets
    t5 = _rows(TABLE_5)
    expected_sectors = {"1a": t5[1][1].removeprefix("a) "), "1b": t5[2][0].removeprefix("b) ")}
    for r in t5[3:-1]:
        expected_sectors[r[0]] = r[1]
    assert len(expected_sectors) == 9 and {k[0] for k in expected_sectors} == set("12345678")
    assert dict(sa.CCS_BUCKET_SECTORS) == expected_sectors
    assert len({k[0] for k in sa.CCS_BUCKET_SECTORS}) == 8

    # Table 6 (symmetric, 8 x 8) and Table 7 (weights by bucket label and credit quality)
    _check_upper(sa.CCS_GAMMA, _upper(TABLE_6, 1, 8))
    t7 = _rows(TABLE_7)
    cols = [h.replace(" ", "").replace(")", "") for h in t7[0][1:]]
    assert cols == list(sa.CCS_BUCKET_SECTORS)
    for quality, row in (("IG", t7[1]), ("HY_NR", t7[2])):
        for label, cell in zip(cols, row[1:], strict=True):
            rp = sa.CCS_RISK_WEIGHTS[(label, quality)]
            assert rp.value == _pct(cell) and rp.paragraph == "MAR50.65" and "Table 7" in rp.table
    assert len(sa.CCS_RISK_WEIGHTS) == 18
    assert sa.CCS_TENORS == (0.5, 1.0, 3.0, 5.0, 10.0)

    # Table 8: 17 reference credit spread buckets with their credit quality and sector
    quality_map = {"IG": "IG", "HY and NR": "HY_NR", "(Not applicable)": "n/a", "HY": "HY"}
    expected_rcs, quality = [], ""
    for r in _rows(TABLE_8)[1:-1]:
        if len(r) == 3:
            quality, sector = quality_map[r[1]], r[2]
        else:
            sector = r[1]
        expected_rcs.append((int(r[0]), quality, sector))
    assert len(expected_rcs) == 17
    assert tuple(sa.RCS_BUCKETS) == tuple(expected_rcs)

    # Table 9 (10 x 10 symmetric) and its expansion to 17 buckets with the halving rule of MAR50.67 (2)
    _check_upper(sa.RCS_GAMMA_TABLE_9, _upper(TABLE_9, 1, 10))
    group = {b: (b - 1) % 7 for b in range(1, 15)} | {15: 7, 16: 8, 17: 9}
    assert sa.RCS_GAMMA.shape == (17, 17) and not sa.RCS_GAMMA.flags.writeable
    for b in range(1, 18):
        for c in range(1, 18):
            if b == c:
                continue
            base = sa.RCS_GAMMA_TABLE_9[min(group[b], group[c]), max(group[b], group[c])]
            different_quality = b <= 14 and c <= 14 and (b <= 7) != (c <= 7)
            want = base / 2 if different_quality else base
            assert sa.RCS_GAMMA[b - 1, c - 1] == want, (b, c)

    # Table 10: two stacked blocks, the investment grade buckets 1 to 9 and the high yield and not rated 10 to 17
    t10 = _rows(TABLE_10)
    cells = [(i + 1, c) for i, c in enumerate(t10[1][1:])] + [(10 + i, c) for i, c in enumerate(t10[4][1:]) if c]
    assert [n for n, _ in cells] == list(range(1, 18))
    for n, cell in cells:
        rp = sa.RCS_DELTA_RISK_WEIGHTS[n]
        assert rp.value == _pct(cell) and rp.paragraph == "MAR50.68" and "Table 10" in rp.table
    assert len(sa.RCS_DELTA_RISK_WEIGHTS) == 17

    # Table 11: 13 equity buckets; Table 12 weights; the cross-bucket correlations of MAR50.71
    t11 = _rows(TABLE_11)[1:-1]
    assert [int(r[0]) for r in t11] == list(range(1, 14))
    assert [b[0] for b in sa.EQ_BUCKETS] == list(range(1, 14))
    assert [b[2] for b in sa.EQ_BUCKETS] == [r[-1] for r in t11]
    assert [b[1] for b in sa.EQ_BUCKETS] == [
        "Large | Emerging market economies",
        "Large | Emerging market economies",
        "Large | Emerging market economies",
        "Large | Emerging market economies",
        "Large | Advanced economies",
        "Large | Advanced economies",
        "Large | Advanced economies",
        "Large | Advanced economies",
        "Small | Emerging market economies",
        "Small | Advanced economies",
        "(Not applicable)",
        "Large cap, advanced economies",
        "Other",
    ]
    for r in _rows(TABLE_12)[1:-1]:
        rp = sa.EQ_DELTA_RISK_WEIGHTS[int(r[0])]
        assert rp.value == _pct(r[1]) and rp.paragraph == "MAR50.72" and "Table 12" in rp.table
    assert len(sa.EQ_DELTA_RISK_WEIGHTS) == 13
    for b in range(1, 14):
        for c in range(1, 14):
            if b == c:
                continue
            if 11 in (b, c):
                want = 0.0
            elif {b, c} == {12, 13}:
                want = 0.75
            elif b >= 12 or c >= 12:
                want = 0.45
            else:
                want = 0.15
            assert sa.EQ_GAMMA[b - 1, c - 1] == want, (b, c)
    assert not sa.EQ_GAMMA.flags.writeable

    # Table 13: 11 commodity buckets; Table 14 weights; gamma of MAR50.75
    t13 = _rows(TABLE_13)[1:-1]
    assert [int(r[0]) for r in t13] == list(range(1, 12))
    assert tuple(sa.COM_BUCKETS) == tuple((int(r[0]), r[1]) for r in t13)
    for n, cell in zip(range(1, 12), _rows(TABLE_14)[1][1:], strict=True):
        rp = sa.COM_DELTA_RISK_WEIGHTS[n]
        assert rp.value == _pct(cell) and rp.paragraph == "MAR50.76" and "Table 14" in rp.table
    assert len(sa.COM_DELTA_RISK_WEIGHTS) == 11
    for b in range(1, 12):
        for c in range(1, 12):
            if b != c:
                assert sa.COM_GAMMA[b - 1, c - 1] == (0.0 if 11 in (b, c) else 0.2)

    # equity vega: 78 percent for the large market capitalisation buckets (1 to 8 and 12), 100 percent for the others
    for b in range(1, 14):
        want = 0.78 if b in (1, 2, 3, 4, 5, 6, 7, 8, 12) else 1.0
        assert sa.EQ_VEGA_RISK_WEIGHTS[b].value == want and sa.EQ_VEGA_RISK_WEIGHTS[b].paragraph == "MAR50.73"

    # the parameter records carry their paragraphs and are read-only
    p = sa.sa_cva_parameters("ir", "delta", specified_currency=True)
    assert p.paragraphs["risk_weights"].startswith("MAR50.56") and p.factors == tuple(labels)
    assert p.risk_weight("EUR", "5y") == 0.0074 and not p.correlation.flags.writeable
    p = sa.sa_cva_parameters("ir", "delta", specified_currency=False)
    assert p.factors == ("parallel", "inflation") and p.risk_weight("CLP", "parallel") == 0.0158
    assert p.correlation[0, 1] == 0.4 and p.gamma_between("CLP", "BRL") == 0.5
    assert sa.sa_cva_parameters("fx", "delta").gamma_between("USD", "GBP") == 0.6
    assert sa.sa_cva_parameters("rcs", "vega").risk_weight("3") == 1.0
    assert sa.sa_cva_parameters("rcs", "delta").gamma_between("1", "8") == 0.5  # 100% of Table 9 halved
    assert sa.sa_cva_parameters("ccs", "delta").risk_weight("1b", quality="HY_NR") == 0.04
    assert sa.sa_cva_parameters("ccs", "delta").gamma_between("1a", "8") == 0.45
    assert sa.sa_cva_parameters("eq", "vega").risk_weight("12") == 0.78
    assert sa.sa_cva_parameters("com", "delta").risk_weight("4") == 0.8
    with pytest.raises(ValueError, match="no vega"):
        sa.sa_cva_parameters("ccs", "vega")
    with pytest.raises(ValueError, match="'xx'"):
        sa.sa_cva_parameters("xx", "delta")


# ---------------------------------------------------------------------------------------------------------------------
# The bucket formula (RV-2002): independent loops over MAR50.53 (1) to (3)


def _ref_k(ws_cva, ws_hdg, rho, r=0.01):
    """K_b by loops over MAR50.53 (1): sqrt(max(0, sum WS^2 + sum k != l rho WS WS + R sum WSh^2))."""
    ws = [a - b for a, b in zip(ws_cva, ws_hdg, strict=True)]
    total = 0.0
    for k in range(len(ws)):
        total += ws[k] ** 2
        for ell in range(len(ws)):
            if ell != k:
                total += rho[k][ell] * ws[k] * ws[ell]
    total += r * sum(h * h for h in ws_hdg)
    return math.sqrt(max(0.0, total))


def _ref_s(ws, k_b):
    return max(-k_b, min(sum(ws), k_b))


def _ref_cross(k, s, gamma, m_cva=1.0):
    """K by loops over MAR50.53 (2)."""
    total = 0.0
    for b in range(len(k)):
        total += k[b] ** 2
        for c in range(len(k)):
            if c != b:
                total += gamma[b][c] * s[b] * s[c]
    return m_cva * math.sqrt(max(0.0, total))


def _t4_matrix():
    up = _upper(TABLE_4, 1, 6)
    return [[up[min(i, j)][max(i, j)] for j in range(6)] for i in range(6)]


def test_bucket_formula_derived_fixtures():
    # MAR50.51, MAR50.52: WS^CVA = RW s^CVA, WS^Hdg = RW s^Hdg, WS = WS^CVA - WS^Hdg
    assert sa.weighted_sensitivity(0.0111, 2e6, 5e5) == pytest.approx((22200.0, 5550.0, 16650.0), rel=1e-12)
    arr = sa.weighted_sensitivity(np.array([0.5, 0.25]), np.array([4.0, 8.0]), np.array([1.0, 2.0]))
    assert [list(a) for a in arr] == [[2.0, 2.0], [0.5, 0.5], [1.5, 1.5]]

    one = [[1.0]]
    # SA-A: one factor, no hedge: K_b = |WS|; a single risk factor takes no correlation term
    assert sa.bucket_k([100.0], [0.0], one) == 100.0 == _ref_k([100.0], [0.0], one)
    assert sa.bucket_k([100.0], [0.0], None) == 100.0
    assert sa.bucket_k([-100.0], [0.0], None) == 100.0
    # SA-B: a perfect hedge leaves sqrt(R) x hedge: sqrt(0.01 x 100^2) = 10
    assert sa.bucket_k([100.0], [100.0], one) == pytest.approx(10.0, rel=1e-14)
    assert sa.bucket_k([100.0], [100.0], one) == pytest.approx(_ref_k([100.0], [100.0], one), rel=1e-14)
    # SA-C: two factors at rho 0.91 (Table 4, 1 year and 2 years)
    t4 = _t4_matrix()
    rho = [[1.0, t4[0][1]], [t4[1][0], 1.0]]
    assert rho[0][1] == 0.91
    k_c1 = sa.bucket_k([100.0, 50.0], [0.0, 0.0], rho)
    assert k_c1 == pytest.approx(math.sqrt(100.0**2 + 50.0**2 + 2 * 0.91 * 5000.0), rel=1e-14)
    assert k_c1 == pytest.approx(146.9694, abs=5e-5) and k_c1 == pytest.approx(_ref_k([100, 50], [0, 0], rho))
    k_c2 = sa.bucket_k([100.0, 50.0], [0.0, 30.0], rho)
    assert k_c2 == pytest.approx(118.5285, abs=5e-5)
    assert k_c2 == pytest.approx(_ref_k([100, 50], [0, 30], rho), rel=1e-14)
    # SA-D: from sensitivities in EUR, s^CVA = (2e6, 1e6) at 1 year and 2 years: WS = (22,200; 9,300)
    rw = [_pct(c) for c in _rows(TABLE_3)[1][1:3]]
    ws = [rw[0] * 2e6, rw[1] * 1e6]
    assert ws == pytest.approx([22200.0, 9300.0])
    k_d = sa.bucket_k(ws, [0.0, 0.0], rho)
    assert k_d == pytest.approx(_ref_k(ws, [0.0, 0.0], rho), rel=1e-14)
    assert k_d == pytest.approx(30904.485, abs=5e-4)

    # the hedging disallowance is INSIDE the root: K_b^2 grows by exactly R sum (WS^Hdg)^2
    base = sa.bucket_k([100.0, 50.0], [0.0, 30.0], rho, r=0.0)
    assert sa.bucket_k([100.0, 50.0], [0.0, 30.0], rho, r=0.01) ** 2 == pytest.approx(base**2 + 0.01 * 900.0, rel=1e-13)
    # a negative radicand is floored at 0 (an indefinite correlation matrix: three factors at rho -1)
    neg = [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    assert _ref_k([1, 1, 1], [0, 0, 0], neg, r=0.0) == 0.0
    assert sa.bucket_k([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], neg, r=0.0) == 0.0
    # an empty bucket has charge 0 and S_b 0
    assert sa.bucket_k([], [], np.zeros((0, 0))) == 0.0
    assert sa.bucket_s([], 0.0) == 0.0

    # S_b = max(-K_b, min(sum WS, K_b)) always, not only on a negative radicand
    for ws_list, k_b, want in (
        ([100.0], 100.0, 100.0),  # the sum equals K_b
        ([60.0, 70.0], 100.0, 100.0),  # the sum 130 is capped at K_b
        ([-60.0, -70.0], 100.0, -100.0),  # the sum -130 is floored at -K_b
        ([20.0, -5.0], 100.0, 15.0),  # inside the band
        ([5.0], 0.0, 0.0),  # a zero K_b closes the band
    ):
        assert sa.bucket_s(ws_list, k_b) == want == _ref_s(ws_list, k_b)
    # two offsetting factors at rho 0: K_b = sqrt(2) x 100, S_b = 0
    k_b = sa.bucket_k([100.0, -100.0], [0.0, 0.0], np.eye(2))
    assert k_b == pytest.approx(100.0 * math.sqrt(2.0)) and sa.bucket_s([100.0, -100.0], k_b) == 0.0

    # refusals naming the field
    with pytest.raises(ValueError, match="ws_cva"):
        sa.bucket_k([float("nan")], [0.0], one)
    with pytest.raises(ValueError, match="ws_hdg"):
        sa.bucket_k([1.0], [float("inf")], one)
    with pytest.raises(ValueError, match="rho"):
        sa.bucket_k([1.0, 2.0], [0.0, 0.0], None)
    with pytest.raises(ValueError, match="rho"):
        sa.bucket_k([1.0, 2.0], [0.0, 0.0], [[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(ValueError, match="rho"):
        sa.bucket_k([1.0, 2.0], [0.0, 0.0], [[1.0, 1.5], [1.5, 1.0]])
    with pytest.raises(ValueError, match="ws_hdg"):
        sa.bucket_k([1.0, 2.0], [0.0], [[1.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="at least 0"):
        sa.bucket_k([1.0], [0.0], one, r=-0.1)
    with pytest.raises(ValueError, match="k_b"):
        sa.bucket_s([1.0], -1.0)
    with pytest.raises(ValueError, match="s_cva"):
        sa.weighted_sensitivity(0.01, float("nan"), 0.0)
    with pytest.raises(ValueError, match="rw"):
        sa.weighted_sensitivity(-0.01, 1.0, 0.0)


def test_cross_bucket_and_invariants():
    # SA-E: K = (100, 60), S = (100, -60), gamma 0.5 (interest rate, MAR50.55): sqrt(100^2 + 60^2 - 2 x 0.5 x 6000)
    gamma = [[1.0, 0.5], [0.5, 1.0]]
    k = sa.risk_class_k([100.0, 60.0], [100.0, -60.0], gamma)
    assert k == pytest.approx(math.sqrt(100.0**2 + 60.0**2 + 2 * 0.5 * 100.0 * -60.0), rel=1e-14)
    assert k == pytest.approx(87.178, abs=5e-4)
    assert k == pytest.approx(_ref_cross([100, 60], [100, -60], gamma), rel=1e-14)
    # m_CVA scales the whole root (MAR50.53 (2), MAR50.41)
    assert sa.risk_class_k([100.0, 60.0], [100.0, -60.0], gamma, m_cva=1.5) == pytest.approx(1.5 * k, rel=1e-14)
    # two buckets at gamma 1 and same-sign S give the sum of their K_b
    assert sa.risk_class_k([100.0, 60.0], [100.0, 60.0], [[1.0, 1.0], [1.0, 1.0]]) == pytest.approx(160.0, rel=1e-14)
    # gamma 0: the root of the sum of squares
    assert sa.risk_class_k([3.0, 4.0], [3.0, 4.0], np.eye(2)) == pytest.approx(5.0, rel=1e-14)
    # one bucket: its K_b; none: zero
    assert sa.risk_class_k([7.0], [-7.0], [[1.0]]) == pytest.approx(7.0, rel=1e-14)
    assert sa.risk_class_k([], [], np.zeros((0, 0))) == 0.0
    # K is non negative: a negative radicand is floored at zero (gamma -1 and two equal buckets of the same sign)
    assert sa.risk_class_k([5.0, 5.0], [5.0, 5.0], [[1.0, -1.0], [-1.0, 1.0]]) == pytest.approx(0.0, abs=1e-12)
    assert sa.risk_class_k([5.0, 5.0], [5.0, -5.0], [[1.0, -1.0], [-1.0, 1.0]]) == pytest.approx(math.sqrt(100.0))
    rng = np.random.default_rng(20261115)
    for _ in range(200):
        n = int(rng.integers(1, 6))
        k_b = rng.uniform(0, 10, n)
        s_b = rng.uniform(-1, 1, n) * k_b
        g = rng.uniform(-1, 1, (n, n))
        g = (g + g.T) / 2
        np.fill_diagonal(g, 1.0)
        value = sa.risk_class_k(k_b, s_b, g)
        assert value >= 0.0
        assert value == pytest.approx(_ref_cross(list(k_b), list(s_b), g.tolist()), rel=1e-12, abs=1e-12)
    # with no hedge R plays no role
    ws = [4.0, -3.0, 2.0]
    rho = [[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]]
    assert sa.bucket_k(ws, [0, 0, 0], rho, r=0.01) == sa.bucket_k(ws, [0, 0, 0], rho, r=0.5)
    # a perfect hedge of a single factor leaves exactly sqrt(R) |WS^Hdg|
    assert sa.bucket_k([50.0], [50.0], None, r=0.01) == pytest.approx(math.sqrt(0.01) * 50.0, rel=1e-14)
    # refusals
    with pytest.raises(ValueError, match="m_cva"):
        sa.risk_class_k([1.0], [1.0], [[1.0]], m_cva=0.5)
    with pytest.raises(ValueError, match="k_b"):
        sa.risk_class_k([-1.0], [1.0], [[1.0]])
    with pytest.raises(ValueError, match="s_b"):
        sa.risk_class_k([1.0], [float("nan")], [[1.0]])
    with pytest.raises(ValueError, match="gamma"):
        sa.risk_class_k([1.0, 1.0], [1.0, 1.0], [[1.0]])


# ---------------------------------------------------------------------------------------------------------------------
# The capital (RV-2004): a reference written from the parsed tables, apart from the engine's parameter records

S = sa.Sensitivity


def _w_table(table: str, row: int) -> list[float]:
    return [_pct(c) for c in _rows(table)[row][1:] if c]


def _gamma_ref_rcs(b: int, c: int) -> float:
    group = {n: (n - 1) % 7 for n in range(1, 15)} | {15: 7, 16: 8, 17: 9}
    up = _upper(TABLE_9, 1, 10)
    i, j = sorted((group[b], group[c]))
    base = up[i][j]
    return base / 2 if (b <= 14 and c <= 14 and (b <= 7) != (c <= 7)) else base


def _gamma_ref_eq(b: int, c: int) -> float:
    if 11 in (b, c):
        return 0.0
    if {b, c} == {12, 13}:
        return 0.75
    return 0.45 if max(b, c) >= 12 else 0.15


def _gamma_ref_com(b: int, c: int) -> float:
    return 0.0 if 11 in (b, c) else 0.2


def _gamma_ref_ccs(b: int, c: int) -> float:
    up = _upper(TABLE_6, 1, 8)
    i, j = sorted((b - 1, c - 1))
    return up[i][j]


def _rho_ccs(f, g, index: bool) -> float:
    """MAR50.65: rho_tenor x rho_name x rho_quality; f and g are (name, series, tenor, ig, group)."""
    tenor = 1.0 if f[2] == g[2] else 0.9
    if index:
        name = 1.0 if (f[0], f[1]) == (g[0], g[1]) else (0.9 if f[0] == g[0] else 0.8)
    else:
        name = 1.0 if f[0] == g[0] else (0.9 if f[4] is not None and f[4] == g[4] else 0.5)
    return tenor * name * (1.0 if f[3] == g[3] else 0.8)


def _reference_capital(portfolio_by_class, m_cva=1.0):
    """K by risk class from the independent loops: `portfolio_by_class[(cls, kind)]` is a dict bucket -> list of
    (rw, s_cva, s_hdg, rho_row_key) and the helpers above."""
    out = {}
    for (cls, kind), buckets in portfolio_by_class.items():
        names = sorted(buckets, key=str)
        k_list, s_list = [], []
        for name in names:
            factors, rho = buckets[name]
            ws_cva = [rw * sc for rw, sc, _ in factors]
            ws_hdg = [rw * sh for rw, _, sh in factors]
            k_b = _ref_k(ws_cva, ws_hdg, rho)
            k_list.append(k_b)
            s_list.append(_ref_s([a - b for a, b in zip(ws_cva, ws_hdg, strict=True)], k_b))
        gamma = [[GAMMA_REF[cls](a, b) if a != b else 1.0 for b in names] for a in names]
        out[(cls, kind)] = _ref_cross(k_list, s_list, gamma, m_cva)
    return out


GAMMA_REF = {
    "ir": lambda a, b: 0.5,
    "fx": lambda a, b: 0.6,
    "ccs": lambda a, b: _gamma_ref_ccs(int(a.rstrip("ab")), int(b.rstrip("ab"))),
    "rcs": lambda a, b: _gamma_ref_rcs(int(a), int(b)),
    "eq": lambda a, b: _gamma_ref_eq(int(a), int(b)),
    "com": lambda a, b: _gamma_ref_com(int(a), int(b)),
}
ONE = [[1.0]]


def _build():
    """The sensitivities of the test portfolio and the reference inputs built from the parsed tables."""
    t3 = dict(zip(["1y", "2y", "5y", "10y", "30y", "inflation"], _w_table(TABLE_3, 1), strict=True))
    t4 = _t4_matrix()
    idx = {"1y": 0, "2y": 1, "5y": 2, "10y": 3, "30y": 4, "inflation": 5}
    inputs, ref = [], {}

    def put(cls, kind, bucket, rw, s_cva, s_hdg, rho_key):
        ref.setdefault((cls, kind), {}).setdefault(bucket, ([], rho_key))[0].append((rw, s_cva, s_hdg))

    # interest rate delta: EUR (the reporting currency: five tenors), USD (a specified currency), CLP (another)
    ir_eur = [("1y", 2e6, 5e5), ("2y", 1e6, 0.0), ("inflation", 0.0, 1e5)]
    for f, c, h in ir_eur:
        inputs.append(S("ir", "delta", "EUR", f, c, h))
    rho = [[t4[idx[a[0]]][idx[b[0]]] for b in ir_eur] for a in ir_eur]
    for f, c, h in ir_eur:
        put("ir", "delta", "EUR", t3[f], c, h, rho)
    inputs.append(S("ir", "delta", "USD", "5y", 0.0, 3e5))
    put("ir", "delta", "USD", t3["5y"], 0.0, 3e5, ONE)
    clp = [("parallel", 1e6, 0.0), ("inflation", 2e5, 0.0)]
    for f, c, h in clp:
        inputs.append(S("ir", "delta", "CLP", f, c, h))
        put("ir", "delta", "CLP", 0.0158, c, h, [[1.0, 0.4], [0.4, 1.0]])
    # interest rate vega: flat 100 percent, rho 0.4 between the rate and the inflation volatilities
    inputs += [S("ir", "vega", "EUR", "ir_vol", 1e5, 2e4), S("ir", "vega", "EUR", "inflation_vol", 3e4, 0.0)]
    for _f, c, h in (("ir_vol", 1e5, 2e4), ("inflation_vol", 3e4, 0.0)):
        put("ir", "vega", "EUR", 1.0, c, h, [[1.0, 0.4], [0.4, 1.0]])
    # FX
    inputs += [S("fx", "delta", "USD", None, 3e4, 0.0), S("fx", "delta", "GBP", None, -1e4, -5e3)]
    put("fx", "delta", "USD", 0.11, 3e4, 0.0, ONE)
    put("fx", "delta", "GBP", 0.11, -1e4, -5e3, ONE)
    inputs.append(S("fx", "vega", "USD", None, 1e4, 0.0))
    put("fx", "vega", "USD", 1.0, 1e4, 0.0, ONE)
    # counterparty credit spread delta: bucket 2 (A at 1 and 5 years, B related to C, C high yield), 1a, index bucket 8
    w7 = {
        q: dict(zip(["1a", "1b", "2", "3", "4", "5", "6", "7", "8"], _w_table(TABLE_7, r), strict=True))
        for q, r in (("IG", 1), ("HY_NR", 2))
    }
    b2 = [
        ("A", None, 1.0, True, None, 4e3, 1e3),
        ("A", None, 5.0, True, None, 6e3, 0.0),
        ("B", None, 1.0, True, "G", 2e3, 0.0),
        ("C", None, 3.0, False, "G", 1e3, 5e2),
    ]
    for name, _series, tenor, ig, group, sc, sh in b2:
        inputs.append(S("ccs", "delta", 2, tenor, sc, sh, name=name, investment_grade=ig, legal_group=group))
    rho = [[_rho_ccs(f[:5], g[:5], False) if f is not g else 1.0 for g in b2] for f in b2]
    for _name, _series, _tenor, ig, _group, sc, sh in b2:
        put("ccs", "delta", "2", w7["IG" if ig else "HY_NR"]["2"], sc, sh, rho)
    inputs.append(S("ccs", "delta", "1a", 3.0, 7e3, 0.0, name="D", investment_grade=True))
    put("ccs", "delta", "1a", w7["IG"]["1a"], 7e3, 0.0, ONE)
    x1, x2 = ("X", "s1", 5.0, True, None), ("X", "s2", 5.0, True, None)
    idxs = [x1 + (9e3, 0.0), x2 + (-2e3, 0.0)]
    for name, series, tenor, ig, _group, sc, sh in idxs:
        inputs.append(S("ccs", "delta", 8, tenor, sc, sh, name=name, series=series, investment_grade=ig))
    rho = [[_rho_ccs(f[:5], g[:5], True) if f is not g else 1.0 for g in idxs] for f in idxs]
    for _name, _series, _tenor, _ig, _group, sc, sh in idxs:
        put("ccs", "delta", "8", w7["IG"]["8"], sc, sh, rho)
    # reference credit spread delta and vega
    w10 = _w_table(TABLE_10, 1) + [_pct(c) for c in _rows(TABLE_10)[4][1:] if c]
    for b, sc, sh in ((3, 5e4, 1e4), (10, -2e4, 0.0), (16, 1e4, 0.0)):
        inputs.append(S("rcs", "delta", b, None, sc, sh))
        put("rcs", "delta", str(b), w10[b - 1], sc, sh, ONE)
    inputs.append(S("rcs", "vega", 3, None, 4e4, 0.0))
    put("rcs", "vega", "3", 1.0, 4e4, 0.0, ONE)
    # equity delta and vega
    w12 = [_pct(r[1]) for r in _rows(TABLE_12)[1:-1]]
    for b, sc, sh in ((1, 2e4, 0.0), (5, -1e4, 0.0), (12, 3e4, 1e4)):
        inputs.append(S("eq", "delta", b, None, sc, sh))
        put("eq", "delta", str(b), w12[b - 1], sc, sh, ONE)
    for b, sc, sh, rw in ((1, 5e3, 0.0, 0.78), (9, 5e3, 0.0, 1.0), (12, 1e3, 5e2, 0.78)):
        inputs.append(S("eq", "vega", b, None, sc, sh))
        put("eq", "vega", str(b), rw, sc, sh, ONE)
    # commodity
    w14 = _w_table(TABLE_14, 1)
    for b, sc, sh in ((4, 1e4, 0.0), (11, 2e4, 5e3), (7, -3e3, 0.0)):
        inputs.append(S("com", "delta", b, None, sc, sh))
        put("com", "delta", str(b), w14[b - 1], sc, sh, ONE)
    inputs.append(S("com", "vega", 4, None, 1e3, 0.0))
    put("com", "vega", "4", 1.0, 1e3, 0.0, ONE)
    return inputs, {key: {b: v for b, v in buckets.items()} for key, buckets in ref.items()}


def test_capital_composition_and_refusals():
    inputs, ref_in = _build()
    ref = _reference_capital(ref_in)
    res = sa.sa_cva_capital(inputs, "EUR")
    got = {(c.risk_class, c.kind): c.k for c in res.classes}
    assert set(got) == set(ref)
    for key, value in ref.items():
        assert got[key] == pytest.approx(value, rel=1e-12), key
    # capital = K_delta + K_vega, each the simple sum over the risk classes (MAR50.42 to 50.45); RWA = 12.5 x capital
    k_delta = sum(v for (cls, kind), v in ref.items() if kind == "delta")
    k_vega = sum(v for (cls, kind), v in ref.items() if kind == "vega")
    assert res.k_delta == pytest.approx(k_delta, rel=1e-12) and res.k_vega == pytest.approx(k_vega, rel=1e-12)
    assert res.capital == pytest.approx(k_delta + k_vega, rel=1e-12) and res.rwa == pytest.approx(12.5 * res.capital)
    assert ("ccs", "vega") not in got and {c for c, k in got if k == "vega"} == {"ir", "fx", "rcs", "eq", "com"}
    assert {c for c, k in got if k == "delta"} == {"ir", "fx", "ccs", "rcs", "eq", "com"}
    assert res.m_cva == 1.0 and "MAR50.42" in res.paragraphs["capital"]
    # the order of the sensitivities does not matter, and a repeated key is netted before it is weighted
    assert sa.sa_cva_capital(list(reversed(inputs)), "EUR").capital == pytest.approx(res.capital, rel=1e-13)
    split = [S("com", "delta", 4, None, 4e3, 0.0), S("com", "delta", 4, None, 6e3, 0.0)]
    whole = [S("com", "delta", 4, None, 1e4, 0.0)]
    netted = sa.sa_cva_capital(split, "EUR").capital
    assert netted == sa.sa_cva_capital(whole, "EUR").capital == pytest.approx(0.8 * 1e4)
    # m_CVA scales every risk class
    assert sa.sa_cva_capital(inputs, "EUR", m_cva=1.25).capital == pytest.approx(1.25 * res.capital, rel=1e-13)
    # the reporting currency decides the interest rate class: for CLP the CLP curve takes five tenors, EUR still does
    clp = [S("ir", "delta", "CLP", "1y", 1e6, 0.0)]
    assert sa.sa_cva_capital(clp, "CLP").capital == pytest.approx(0.0111 * 1e6)
    assert sa.sa_cva_capital([S("ir", "delta", "CLP", "parallel", 1e6, 0.0)], "EUR").capital == pytest.approx(0.0158e6)
    with pytest.raises(ValueError, match="'parallel'"):
        sa.sa_cva_capital([S("ir", "delta", "CLP", "parallel", 1e6, 0.0)], "CLP")
    with pytest.raises(ValueError, match="'1y'"):
        sa.sa_cva_capital([S("ir", "delta", "CLP", "1y", 1e6, 0.0)], "EUR")
    # an empty portfolio and a hedge-only portfolio
    empty = sa.sa_cva_capital([], "EUR")
    assert (empty.k_delta, empty.k_vega, empty.capital, empty.rwa) == (0.0, 0.0, 0.0, 0.0)
    hedge_only = sa.sa_cva_capital([S("eq", "delta", 1, None, 0.0, 1e4)], "EUR")
    # WS = -5,500 and WS^Hdg = 5,500: K_b = sqrt(5,500^2 + R 5,500^2)
    assert hedge_only.capital == pytest.approx(math.sqrt((0.55e4) ** 2 + 0.01 * (0.55e4) ** 2), rel=1e-12)
    # a perfect hedge leaves sqrt(R) of the weighted sensitivity: 10 percent
    perfect = sa.sa_cva_capital([S("eq", "delta", 1, None, 1e4, 1e4)], "EUR")
    assert perfect.capital == pytest.approx(0.1 * 0.55 * 1e4, rel=1e-12)
    # equity buckets 1 to 8 and 12 take the 78 percent vega weight, the others 100 percent
    assert sa.sa_cva_capital([S("eq", "vega", 12, None, 100.0, 0.0)], "EUR").capital == pytest.approx(78.0)
    assert sa.sa_cva_capital([S("eq", "vega", 13, None, 100.0, 0.0)], "EUR").capital == pytest.approx(100.0)

    # refusals, each naming what is wrong
    with pytest.raises(ValueError, match="counterparty credit spread.*vega"):
        sa.sa_cva_capital([S("ccs", "vega", 2, 1.0, 1.0, 0.0, name="A", investment_grade=True)], "EUR")
    with pytest.raises(ValueError, match="'credit'"):
        sa.sa_cva_capital([S("credit", "delta", 1, None, 1.0, 0.0)], "EUR")
    with pytest.raises(ValueError, match="'gamma'"):
        sa.sa_cva_capital([S("eq", "gamma", 1, None, 1.0, 0.0)], "EUR")
    for field, kwargs in (("s_cva", {"s_cva": float("nan")}), ("s_hdg", {"s_hdg": float("inf")})):
        args = {"s_cva": 1.0, "s_hdg": 0.0} | kwargs
        with pytest.raises(ValueError, match=field):
            sa.sa_cva_capital([S("eq", "delta", 1, None, **args)], "EUR")
    with pytest.raises(ValueError, match="'EUR'"):
        sa.sa_cva_capital([S("fx", "delta", "EUR", None, 1.0, 0.0)], "EUR")  # no FX bucket for the reporting currency
    with pytest.raises(ValueError, match="bucket 14"):
        sa.sa_cva_capital([S("eq", "delta", 14, None, 1.0, 0.0)], "EUR")
    with pytest.raises(ValueError, match="bucket 12"):
        sa.sa_cva_capital([S("com", "delta", 12, None, 1.0, 0.0)], "EUR")
    with pytest.raises(ValueError, match="'1a' or '1b'"):
        sa.sa_cva_capital([S("ccs", "delta", 1, 1.0, 1.0, 0.0, name="A", investment_grade=True)], "EUR")
    with pytest.raises(ValueError, match="investment_grade"):
        sa.sa_cva_capital([S("ccs", "delta", 2, 1.0, 1.0, 0.0, name="A")], "EUR")
    with pytest.raises(ValueError, match="name"):
        sa.sa_cva_capital([S("ccs", "delta", 2, 1.0, 1.0, 0.0, investment_grade=True)], "EUR")
    with pytest.raises(ValueError, match="tenor"):
        sa.sa_cva_capital([S("ccs", "delta", 2, 2.0, 1.0, 0.0, name="A", investment_grade=True)], "EUR")
    with pytest.raises(ValueError, match="factor"):
        sa.sa_cva_capital([S("eq", "delta", 1, "spot", 1.0, 0.0)], "EUR")
    with pytest.raises(ValueError, match="m_cva"):
        sa.sa_cva_capital(inputs, "EUR", m_cva=0.9)
    with pytest.raises(ValueError, match="reporting_currency"):
        sa.sa_cva_capital(inputs, "")
    with pytest.raises(ValueError, match="Sensitivity"):
        sa.sa_cva_capital([("eq", "delta", 1)], "EUR")
    # the counterparty credit spread factor of one name keeps one credit quality
    clash = [
        S("ccs", "delta", 2, 1.0, 1.0, 0.0, name="A", investment_grade=True),
        S("ccs", "delta", 2, 1.0, 1.0, 0.0, name="A", investment_grade=False),
    ]
    with pytest.raises(ValueError, match="investment_grade"):
        sa.sa_cva_capital(clash, "EUR")

    # the correlation of MAR50.65 as a function, name by name
    assert sa.ccs_rho(True, "same", True) == 1.0
    assert sa.ccs_rho(False, "same", True) == 0.9
    assert sa.ccs_rho(True, "related", True) == 0.9
    assert sa.ccs_rho(True, "other", True) == 0.5
    assert sa.ccs_rho(True, "other", False) == pytest.approx(0.4)
    assert sa.ccs_rho(False, "other", False) == pytest.approx(0.9 * 0.5 * 0.8)
    assert sa.ccs_rho(True, "other", True, index=True) == 0.8
    assert sa.ccs_rho(True, "related", True, index=True) == 0.9
    with pytest.raises(ValueError, match="name_relation"):
        sa.ccs_rho(True, "alike", True)


def test_stated_limits():
    doc = (sa.__doc__ or "").lower()
    for phrase in (
        "the cva sensitivities are the caller's",
        "index look-through",
        "is the caller's",
        "single risk factor",
        "no correlation term",
        "mar50.31 to mar50.36",
        "mar50.49",
        "no compliance claim",
        "no vega",
        "no curvature",
    ):
        assert phrase in doc, phrase
    assert "credit" in doc and "the hedging disallowance" in doc
    page = PAGE.read_text(encoding="utf-8").lower()
    for phrase in (
        "the cva sensitivities are the caller's",
        "index look-through",
        "single risk factor",
        "no correlation term",
        "https://www.bis.org/basel_framework/chapter/mar/50.htm?inforce=20230101&published=20200708",
        "mar50.53",
        "mar50.41",
        "mar50.1",
        "securitisation",
    ):
        assert phrase in page, phrase
    assert "\u2014" not in page and "\u2014" not in sa.__doc__
    assert (
        "cva risk" in page
        and (ROOT / "docs" / "regulatory" / "u9" / "README.md").read_text(encoding="utf-8").count("06_sa-cva.md") == 1
    )
