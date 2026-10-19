"""RV-919 to RV-923: the sensitivities-based method with GIRR and the liquidity-horizon cascade (contract
`engine-regulatory-api`, module frtb). The goldens are the contract's and the fixture's invariant cases; the
aggregation is also recomputed by a second route (explicit loops) on seeded random data. The Table 2 matrix below is
the table of the research
addendum R7 (MAR21.46), typed in percent as printed."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from reg_helpers import SQRT2, load_c12_blocks, load_calculators, named

from riskvalidation.regulatory import frtb as f

DOC = load_calculators()
SCENARIOS = ("low", "medium", "high")
TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)
RW_PERCENT = (1.7, 1.7, 1.6, 1.3, 1.2, 1.1, 1.1, 1.1, 1.1, 1.1)
T2_PERCENT = (
    (100.0, 97.0, 91.4, 81.1, 71.9, 56.6, 40.0, 40.0, 40.0, 40.0),
    (97.0, 100.0, 97.0, 91.4, 86.1, 76.3, 56.6, 41.9, 40.0, 40.0),
    (91.4, 97.0, 100.0, 97.0, 94.2, 88.7, 76.3, 65.7, 56.6, 41.9),
    (81.1, 91.4, 97.0, 100.0, 98.5, 95.6, 88.7, 82.3, 76.3, 65.7),
    (71.9, 86.1, 94.2, 98.5, 100.0, 98.0, 93.2, 88.7, 84.4, 76.3),
    (56.6, 76.3, 88.7, 95.6, 98.0, 100.0, 97.0, 94.2, 91.4, 86.1),
    (40.0, 56.6, 76.3, 88.7, 93.2, 97.0, 100.0, 98.5, 97.0, 94.2),
    (40.0, 41.9, 65.7, 82.3, 88.7, 94.2, 98.5, 100.0, 99.0, 97.0),
    (40.0, 40.0, 56.6, 76.3, 84.4, 91.4, 97.0, 99.0, 100.0, 98.5),
    (40.0, 40.0, 41.9, 65.7, 76.3, 86.1, 94.2, 97.0, 98.5, 100.0),
)


def test_girr_parameters_goldens():
    p = f.girr_parameters()  # specified_currency=True by default
    assert tuple(p.tenors) == TENORS
    assert p.sqrt2_applied is True
    np.testing.assert_allclose(p.risk_weights, np.array(RW_PERCENT) / 100 / SQRT2, rtol=1e-14)
    np.testing.assert_allclose(p.risk_weights_table, np.array(RW_PERCENT) / 100, rtol=1e-14)
    assert p.risk_weights[6] == pytest.approx(0.011 / SQRT2, rel=1e-14)
    plain = f.girr_parameters(specified_currency=False)
    assert plain.sqrt2_applied is False
    np.testing.assert_allclose(plain.risk_weights, np.array(RW_PERCENT) / 100, rtol=1e-14)
    assert p.curvature_shock_undivided == 0.017 and plain.curvature_shock == 0.017
    assert p.curvature_shock == pytest.approx(0.017 / SQRT2, rel=1e-14) and p.curvature_shock == pytest.approx(
        0.012021, abs=5e-7
    )
    assert p.same_tenor_other_curve == 0.999 and p.gamma == 0.5
    assert p.correlation[1, 5] == pytest.approx(0.763, abs=5e-4)  # 0.5 to 5 years
    assert p.correlation[3, 4] == pytest.approx(0.985, abs=5e-4)  # 2 to 3 years
    assert p.correlation[0, 9] == pytest.approx(0.40, abs=5e-4)  # 0.25 to 30 years
    for name in ("risk_weights", "correlation", "tenors"):
        with pytest.raises(ValueError):
            getattr(p, name)[0] = 9.0  # the arrays are read-only
    assert dataclasses.is_dataclass(p)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.gamma = 0.1  # type: ignore[misc]
    assert "MAR21.42" in p.paragraphs["risk_weights"] and "MAR21.46" in p.paragraphs["correlation"]
    assert p.in_force == "2023-01-01"


def test_girr_correlation_matrix_and_formula():
    rho = f.girr_parameters().correlation
    assert rho.shape == (10, 10)
    np.testing.assert_allclose(rho, np.array(T2_PERCENT) / 100.0, rtol=0, atol=1e-12)
    assert np.array_equal(rho, rho.T) and np.all(np.diag(rho) == 1.0)
    off = rho[~np.eye(10, dtype=bool)]
    assert off.min() >= 0.40 and off.max() < 1.0
    # the reconstructed footnote 13 formula max(exp(-0.03 |Tk - Tl| / min(Tk, Tl)), 0.40) to the printed 0.1 percent
    t = np.array(TENORS)
    for k in range(10):
        for m in range(10):
            if k != m:
                formula = max(math.exp(-0.03 * abs(t[k] - t[m]) / min(t[k], t[m])), 0.40)
                assert abs(rho[k, m] - formula) <= 5e-4 + 1e-12, (k, m)
    # the same independent data in the other call
    assert np.array_equal(f.girr_parameters(False).correlation, rho)


def test_scenario_correlation_goldens():
    ex = {e["id"]: e for e in DOC["invariant_cases"]}["sbm-low-and-high-scenarios"]
    out = named(ex["outputs"])
    rho = named(ex["inputs"])["rho"]
    assert float(f.scenario_correlation(np.array(rho), "low")) == pytest.approx(out["rho_low"], rel=1e-12)
    assert float(f.scenario_correlation(np.array(rho), "medium")) == pytest.approx(out["rho_medium"], rel=1e-12)
    assert float(f.scenario_correlation(np.array(rho), "high")) == pytest.approx(out["rho_high"], rel=1e-12)
    r = np.array([0.5, 0.8, 1.0])
    np.testing.assert_allclose(f.scenario_correlation(r, "low"), [0.375, 0.6, 1.0], rtol=1e-14)
    np.testing.assert_allclose(f.scenario_correlation(r, "medium"), r)
    np.testing.assert_allclose(f.scenario_correlation(r, "high"), [0.625, 1.0, 1.0], rtol=1e-14)
    # the formulas elementwise over a grid, on a matrix, without touching the input
    grid = np.linspace(-1.0, 1.0, 21)
    m = np.outer(grid, np.ones(3))
    before = m.copy()
    low = f.scenario_correlation(m, "low")
    high = f.scenario_correlation(m, "high")
    assert low.shape == m.shape and np.array_equal(m, before)
    for i, g in enumerate(grid):
        assert low[i, 0] == pytest.approx(max(2 * g - 1, 0.75 * g), abs=1e-14)
        assert high[i, 0] == pytest.approx(min(1.25 * g, 1.0), abs=1e-14)
    assert np.all(f.scenario_correlation(np.array([1.0]), "medium") == 1.0)
    with pytest.raises(ValueError, match="scenario"):
        f.scenario_correlation(r, "stress")
    with pytest.raises(ValueError, match="rho"):
        f.scenario_correlation(np.array([0.5, np.nan]), "high")


def _ref_bucket(ws, rho):
    n = len(ws)
    total = sum(ws[k] ** 2 for k in range(n))
    for k in range(n):
        for m in range(n):
            if k != m:
                total += rho[k][m] * ws[k] * ws[m]
    return math.sqrt(max(0.0, total))


def test_bucket_charge_invariants():
    assert f.bucket_charge(np.array([-7.0]), np.array([[1.0]])) == pytest.approx(7.0)
    ones = np.ones((3, 3))
    assert f.bucket_charge(np.array([1.0, -2.0, 4.0]), ones) == pytest.approx(3.0)  # |sum WS| for rho = 1
    assert f.bucket_charge(np.array([3.0, 4.0]), np.eye(2)) == pytest.approx(5.0)
    for scenario in SCENARIOS:  # perfectly correlated stays perfectly correlated in every scenario
        assert f.bucket_charge(np.array([1.0, -2.0, 4.0]), ones, scenario) == pytest.approx(3.0)
    # a negative radicand is floored at zero: the result is never negative
    anti = np.array([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    assert f.bucket_charge(np.array([1.0, 1.0, 1.0]), anti) == 0.0
    rng = np.random.default_rng(11)
    for _ in range(60):
        n = int(rng.integers(1, 7))
        ws = rng.normal(0, 10, n)
        a = rng.uniform(-1, 1, (n, n))
        rho = (a + a.T) / 2
        np.fill_diagonal(rho, 1.0)
        for scenario in SCENARIOS:
            rho_s = f.scenario_correlation(rho, scenario)
            want = _ref_bucket(list(ws), rho_s.tolist())
            got = f.bucket_charge(ws, rho, scenario)
            assert got >= 0.0 and got == pytest.approx(want, rel=1e-11, abs=1e-12)
    for bad_ws, bad_rho, name in (
        (np.array([1.0, 2.0]), np.eye(3), "rho"),
        (np.array([np.nan]), np.eye(1), "ws"),
        (np.array([1.0, 2.0]), np.array([[1.0, 2.0], [2.0, 1.0]]), "rho"),
    ):
        with pytest.raises(ValueError, match=name):
            f.bucket_charge(bad_ws, bad_rho)
    with pytest.raises(ValueError, match="scenario"):
        f.bucket_charge(np.array([1.0]), np.eye(1), "stress")


def _gamma(n, g):
    m = np.full((n, n), float(g))
    np.fill_diagonal(m, 1.0)
    return m


def _ref_cross(k, s, gamma):
    n = len(k)
    rad = sum(k[b] ** 2 for b in range(n)) + sum(
        gamma[b][c] * s[b] * s[c] for b in range(n) for c in range(n) if b != c
    )
    fallback = rad < 0
    if fallback:
        s = [max(min(s[b], k[b]), -k[b]) for b in range(n)]
        rad = sum(k[b] ** 2 for b in range(n)) + sum(
            gamma[b][c] * s[b] * s[c] for b in range(n) for c in range(n) if b != c
        )
    return math.sqrt(max(0.0, rad)), fallback


def _run_case(case_id):
    ex = {e["id"]: e for e in DOC["invariant_cases"]}[case_id]
    inp = named(ex["inputs"])
    want = named(ex["outputs"])
    buckets = [
        (np.array(inp["ws_1"]), np.array(inp["rho_1"]).reshape(len(inp["ws_1"]), -1)),
        (np.array(inp["ws_2"]), np.array(inp["rho_2"]).reshape(len(inp["ws_2"]), -1)),
    ]
    gamma = _gamma(2, inp["gamma"])
    cross = {}
    for scenario in SCENARIOS:
        k = np.array([f.bucket_charge(ws, rho, scenario) for ws, rho in buckets])
        s = np.array([ws.sum() for ws, _ in buckets])
        assert k[0] == pytest.approx(want[f"k_b1_{scenario}"], rel=1e-9)
        assert k[1] == pytest.approx(want[f"k_b2_{scenario}"], rel=1e-9)
        charge = f.cross_bucket_charge(k, s, gamma, scenario)
        assert charge == pytest.approx(want[f"cross_bucket_{scenario}"], rel=1e-9, abs=1e-9), (case_id, scenario)
        # the fallback flag of the fixture is the sign of the plain radicand, recomputed here
        gs = f.scenario_correlation(gamma, scenario)
        plain = float(k @ k + sum(gs[b, c] * s[b] * s[c] for b in range(2) for c in range(2) if b != c))
        assert bool(plain < 0) is bool(want[f"fallback_{scenario}"]), (case_id, scenario)
        ref, fb = _ref_cross(list(k), list(s), gs.tolist())
        assert bool(fb) is bool(want[f"fallback_{scenario}"]) and charge == pytest.approx(ref, rel=1e-11, abs=1e-12)
        cross[scenario] = charge
    capital = f.sbm_capital(cross, {s: 0.0 for s in SCENARIOS}, {s: 0.0 for s in SCENARIOS})
    assert capital.capital == pytest.approx(want["sbm_charge"], rel=1e-9)
    return cross


def test_cross_bucket_charge_and_fixture_cases():
    first = _run_case("sbm-bucket-and-cross-bucket-aggregation")
    assert first["high"] > first["medium"] > first["low"]  # the maximum over the scenarios is the high one
    second = _run_case("sbm-cross-bucket-fallback")
    # the contract's text: K_b = 17.3205, S_b = +-20, the fallback S_b = +-17.3205 and the charge 10.9545 (medium)
    k = np.array([17.320508075688775] * 2)
    assert f.cross_bucket_charge(k, np.array([20.0, -20.0]), _gamma(2, 0.8)) == pytest.approx(10.9545, abs=5e-5)
    assert second["medium"] == pytest.approx(10.954451150103322, rel=1e-12)
    # invariants: one bucket gives its K_b; never negative; the fallback keeps S_b inside [-K_b, K_b]
    assert f.cross_bucket_charge(np.array([6.0]), np.array([100.0]), np.array([[1.0]])) == pytest.approx(6.0)
    assert f.cross_bucket_charge(np.array([3.0, 4.0]), np.array([0.0, 0.0]), _gamma(2, 0.5)) == pytest.approx(5.0)
    rng = np.random.default_rng(5)
    fell_back = 0
    for _ in range(200):
        n = int(rng.integers(1, 6))
        k = rng.uniform(0, 10, n)
        s = rng.uniform(-1, 1, n) * k * 2
        a = rng.uniform(0, 1, (n, n))
        gamma = (a + a.T) / 2
        np.fill_diagonal(gamma, 1.0)
        for scenario in SCENARIOS:
            gs = f.scenario_correlation(gamma, scenario)
            ref, fb = _ref_cross(list(k), list(s), gs.tolist())
            fell_back += fb
            got = f.cross_bucket_charge(k, s, gamma, scenario)
            assert got >= 0.0 and got == pytest.approx(ref, rel=1e-10, abs=1e-12)
    assert fell_back > 0  # the random draws exercise the MAR21.4(5)(b) path
    for args, name in (
        ((np.array([1.0, 2.0]), np.array([1.0]), _gamma(2, 0.5)), "s_b"),
        ((np.array([1.0, 2.0]), np.array([1.0, 1.0]), _gamma(3, 0.5)), "gamma"),
        ((np.array([-1.0]), np.array([1.0]), np.eye(1)), "k_b"),
    ):
        with pytest.raises(ValueError, match=name):
            f.cross_bucket_charge(*args)


def test_girr_delta_charge_golden_and_scaling():
    p = f.girr_parameters()
    s = np.zeros(10)
    s[6] = 8.0e7  # a 10 million USD 10-year payer swap: s_k = 8.0e7 USD per unit of rate at the 10 year tenor
    res = f.girr_delta_charge(s, np.array(TENORS), p)
    assert isinstance(res, f.SbmResult)
    assert res.weighted[6] == pytest.approx(0.011 * 8.0e7 / SQRT2, rel=1e-13) and res.weighted[6] == pytest.approx(
        622253.97, abs=5e-3
    )
    assert np.count_nonzero(res.weighted) == 1
    assert len(res.k_by_scenario) == 3
    assert all(
        k == pytest.approx(622253.97, abs=5e-3) for k in res.k_by_scenario
    )  # one factor gives |WS| in every scenario
    assert res.charge == pytest.approx(622253.97, abs=5e-3)
    # the sqrt(2)-divided charge is 1 / sqrt(2) times the undivided one (the weights scale uniformly)
    rng = np.random.default_rng(3)
    sens = rng.normal(0, 1e7, 10)
    div = f.girr_delta_charge(sens, np.array(TENORS), f.girr_parameters(True))
    und = f.girr_delta_charge(sens, np.array(TENORS), f.girr_parameters(False))
    assert div.charge == pytest.approx(und.charge / SQRT2, rel=1e-13)
    for a, b in zip(div.k_by_scenario, und.k_by_scenario, strict=True):
        assert a == pytest.approx(b / SQRT2, rel=1e-13)
    # the charge is the maximum over the scenarios; each scenario is the bucket charge of the weighted sensitivities
    for scenario, k in zip(SCENARIOS, div.k_by_scenario, strict=True):
        assert k == pytest.approx(f.bucket_charge(div.weighted, f.girr_parameters().correlation, scenario), rel=1e-13)
    assert div.charge == max(div.k_by_scenario)
    zero = f.girr_delta_charge(np.zeros(10), np.array(TENORS), p)
    assert zero.charge == 0.0 and zero.k_by_scenario == (0.0, 0.0, 0.0)
    # inputs are not mutated, a wrong shape or tenor grid is refused
    before = sens.copy()
    f.girr_delta_charge(sens, np.array(TENORS), p)
    assert np.array_equal(sens, before)
    with pytest.raises(ValueError, match="sensitivities"):
        f.girr_delta_charge(np.zeros(9), np.array(TENORS[:9]), p)
    with pytest.raises(ValueError, match="tenors"):
        f.girr_delta_charge(np.zeros(10), np.array(TENORS) * 2, p)
    with pytest.raises(ValueError, match="sensitivities"):
        f.girr_delta_charge(np.full(10, np.nan), np.array(TENORS), p)


def test_girr_delta_charge_reproduces_the_c12_blocks():
    checked = 0
    for variant, doc in load_c12_blocks().items():
        block = doc["girr"]
        sens = np.array(block["sensitivities"])
        res = f.girr_delta_charge(sens, np.array(TENORS), f.girr_parameters(block["sqrt2_division"]))
        np.testing.assert_allclose(res.weighted, block["weighted_sensitivities"], rtol=1e-8, atol=1e-5)
        for got, want in zip(res.k_by_scenario, block["scenarios"], strict=True):
            assert got == pytest.approx(want["k_b"], rel=1e-9), variant
        assert [s["scenario"] for s in block["scenarios"]] == list(SCENARIOS)
        assert res.charge == pytest.approx(block["delta_charge"]["amount"], rel=1e-9)
        und = f.girr_delta_charge(sens, np.array(TENORS), f.girr_parameters(False))
        assert und.charge == pytest.approx(block["delta_charge_undivided"]["amount"], rel=1e-9)
        checked += 1
    assert checked == 4


def _ref_curvature(plus, minus, rho_delta, gamma, scenario):
    rho = f.scenario_correlation(np.array(rho_delta) ** 2, scenario).tolist()
    gam = f.scenario_correlation(np.array(gamma), scenario).tolist()
    psi = lambda x, y: 0.0 if (x < 0 and y < 0) else 1.0  # noqa: E731
    kb, sb = [], []
    for b in range(len(plus)):
        roots = []
        for cvr in (plus[b], minus[b]):
            n = len(cvr)
            tot = sum(max(c, 0.0) ** 2 for c in cvr)
            tot += sum(rho[i][j] * cvr[i] * cvr[j] * psi(cvr[i], cvr[j]) for i in range(n) for j in range(n) if i != j)
            roots.append(math.sqrt(max(0.0, tot)))
        up = roots[0] > roots[1]
        kb.append(max(roots))
        sb.append(sum(plus[b]) if up else sum(minus[b]))
    tot = sum(k * k for k in kb)
    tot += sum(gam[b][c] * sb[b] * sb[c] * psi(sb[b], sb[c]) for b in range(len(kb)) for c in range(len(kb)) if b != c)
    return math.sqrt(max(0.0, tot))


def test_curvature_charge_invariants():
    one = np.array([[1.0]])
    assert f.curvature_charge(np.array([[5.0]]), np.array([[-3.0]]), one, one) == pytest.approx(5.0)
    assert f.curvature_charge(np.array([[-2.0]]), np.array([[7.0]]), one, one) == pytest.approx(7.0)
    # all-negative CVR give 0
    neg = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    assert f.curvature_charge(neg, neg, np.eye(2), _gamma(2, 0.5)) == 0.0
    # psi: two factors, one negative; the correlation is the square of the delta one (0.5 -> 0.25)
    plus = np.array([[3.0, -2.0]])
    minus = np.array([[-1.0, -1.0]])
    rho = np.array([[1.0, 0.5], [0.5, 1.0]])
    k_plus = math.sqrt(9.0 + 2 * 0.25 * 3.0 * -2.0)
    assert f.curvature_charge(plus, minus, rho, np.array([[1.0]])) == pytest.approx(k_plus, rel=1e-13)
    rng = np.random.default_rng(13)
    for _ in range(80):
        b, k = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        plus = rng.normal(0, 5, (b, k))
        minus = rng.normal(0, 5, (b, k))
        a = rng.uniform(-1, 1, (k, k))
        rho = (a + a.T) / 2
        np.fill_diagonal(rho, 1.0)
        g = rng.uniform(0, 1, (b, b))
        gamma = (g + g.T) / 2
        np.fill_diagonal(gamma, 1.0)
        for scenario in SCENARIOS:
            got = f.curvature_charge(plus, minus, rho, gamma, scenario)
            want = _ref_curvature(plus.tolist(), minus.tolist(), rho.tolist(), gamma.tolist(), scenario)
            assert got >= 0.0 and got == pytest.approx(want, rel=1e-10, abs=1e-12)
    before = plus.copy()
    f.curvature_charge(plus, minus, rho, gamma)
    assert np.array_equal(plus, before)
    with pytest.raises(ValueError, match="cvr_minus"):
        f.curvature_charge(np.zeros((2, 3)), np.zeros((2, 2)), np.eye(3), np.eye(2))
    with pytest.raises(ValueError, match="rho"):
        f.curvature_charge(np.zeros((2, 3)), np.zeros((2, 3)), np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="gamma"):
        f.curvature_charge(np.zeros((2, 3)), np.zeros((2, 3)), np.eye(3), np.eye(3))
    with pytest.raises(ValueError, match="cvr_plus"):
        f.curvature_charge(np.array([[np.nan]]), np.array([[0.0]]), np.eye(1), np.eye(1))


def test_sbm_capital():
    delta = {"low": 10.0, "medium": 12.0, "high": 15.0}
    vega = {"low": 1.0, "medium": 2.0, "high": 0.5}
    curv = {"low": 0.0, "medium": 4.0, "high": 1.0}
    res = f.sbm_capital(delta, vega, curv)
    assert isinstance(res, f.SbmCapital)
    assert res.capital == pytest.approx(18.0) and res.scenario == "medium"
    assert res.by_scenario == (11.0, 18.0, 16.5)
    for sc in SCENARIOS:
        assert res.capital >= delta[sc] + vega[sc] + curv[sc]
    only = f.sbm_capital(
        {"low": 0.0, "medium": 5.0, "high": 0.0},
        {"low": 0.0, "medium": 0.0, "high": 0.0},
        {"low": 0.0, "medium": 0.0, "high": 0.0},
    )
    assert only.capital == 5.0 and only.scenario == "medium"
    zero = {s: 0.0 for s in SCENARIOS}
    assert f.sbm_capital(zero, zero, zero).capital == 0.0
    with pytest.raises(ValueError, match="vega"):
        f.sbm_capital(delta, {"low": 1.0, "medium": 2.0}, curv)
    with pytest.raises(ValueError, match="delta"):
        f.sbm_capital({"low": -1.0, "medium": 0.0, "high": 0.0}, zero, zero)
    with pytest.raises(ValueError, match="curvature"):
        f.sbm_capital(zero, zero, {"low": float("nan"), "medium": 0.0, "high": 0.0})


def test_liquidity_cascade_goldens():
    cases = {e["id"]: e for e in DOC["invariant_cases"]}
    ten = cases["cascade-all-lh-ten"]
    es_p = named(ten["inputs"])["es_t_p"]
    assert f.liquidity_cascade_es(es_p, {}) == pytest.approx(named(ten["outputs"])["es"], rel=1e-14)
    assert f.liquidity_cascade_es(es_p, {10: es_p}) == pytest.approx(es_p, rel=1e-14)
    two = cases["cascade-two-horizons"]
    inp = named(two["inputs"])
    got = f.liquidity_cascade_es(inp["es_t_p"], {20: inp["es_t_p_2"]}, base_horizon=int(inp["t"]))
    assert got == pytest.approx(named(two["outputs"])["es"], rel=1e-12)
    assert got == pytest.approx(math.sqrt(100 + 36 * 1), rel=1e-14) and got == pytest.approx(11.661904, abs=5e-7)
    # the full cascade recomputed: sqrt(ES^2 + sum_j (ES_j sqrt((LH_j - LH_{j-1}) / T))^2)
    es = {20: 8.0, 40: 6.0, 60: 3.0, 120: 1.5}
    lh = (10, 20, 40, 60, 120)
    total = 10.0**2 + sum((es[lh[j]] * math.sqrt((lh[j] - lh[j - 1]) / 10)) ** 2 for j in range(1, 5))
    assert f.liquidity_cascade_es(10.0, es) == pytest.approx(math.sqrt(total), rel=1e-13)
    # the base horizon T enters the weights
    assert f.liquidity_cascade_es(10.0, {20: 6.0}, base_horizon=20) == pytest.approx(
        math.sqrt(100 + 36 * 0.5), rel=1e-13
    )


def test_liquidity_cascade_invariants_and_refusals():
    base = {20: 4.0, 40: 3.0, 60: 2.0, 120: 1.0}
    ref = f.liquidity_cascade_es(10.0, base)
    assert ref >= 10.0
    for lh in base:
        bigger = dict(base)
        bigger[lh] = base[lh] + 1.0
        assert f.liquidity_cascade_es(10.0, bigger) > ref  # monotone in each ES_T(P, j)
        assert f.liquidity_cascade_es(10.0, {**base, lh: 0.0}) <= ref
    assert f.liquidity_cascade_es(10.0, {}) == 10.0
    assert f.liquidity_cascade_es(0.0, base) == pytest.approx(
        math.sqrt(sum((base[h] * math.sqrt((h - p) / 10)) ** 2 for p, h in ((10, 20), (20, 40), (40, 60), (60, 120)))),
        rel=1e-13,
    )
    before = dict(base)
    f.liquidity_cascade_es(10.0, base)
    assert base == before
    with pytest.raises(ValueError, match="es_t_by_horizon"):
        f.liquidity_cascade_es(10.0, {30: 1.0})
    with pytest.raises(ValueError, match="es_t_by_horizon"):
        f.liquidity_cascade_es(10.0, {20: -1.0})
    with pytest.raises(ValueError, match="es_t_by_horizon"):
        f.liquidity_cascade_es(10.0, {10: 9.0})  # the horizon-10 entry is ES_T(P) itself
    with pytest.raises(ValueError, match="es_t_full"):
        f.liquidity_cascade_es(-1.0, {})
    with pytest.raises(ValueError, match="es_t_full"):
        f.liquidity_cascade_es(float("nan"), {})
    with pytest.raises(ValueError, match="base_horizon"):
        f.liquidity_cascade_es(10.0, {}, base_horizon=0)
