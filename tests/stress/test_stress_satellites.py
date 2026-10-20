"""RV-1301 to RV-1304: the satellites (contract `engine-stress-api`, module stress.satellites)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from riskvalidation.engines.vasicek import conditional_pd, implied_factor
from riskvalidation.stress import satellites as s


def arr(*x):
    return np.array(x, dtype=float)


def quarters(start_year: int, n: int) -> list[str]:
    """ISO quarter ends from the first quarter of `start_year`."""
    ends = ("03-31", "06-30", "09-30", "12-31")
    return [f"{start_year + k // 4}-{ends[k % 4]}" for k in range(n)]


# ---------------------------------------------------------------------------------------------------- RV-1301


def test_link_goldens_and_invariants():
    assert float(s.link("probit", 0.5)) == 0.0
    assert float(s.link("logit", 0.2)) == pytest.approx(-1.3863, abs=5e-5)
    assert float(s.link("logit", 0.2)) == pytest.approx(math.log(0.25), abs=1e-15)
    assert float(s.link("identity", -0.0003)) == -0.0003
    p = np.linspace(0.001, 0.999, 200)
    for name in ("probit", "logit", "identity"):
        assert np.all(np.diff(s.link(name, p)) > 0.0), name

    # the probit against the normal quantile computed by bisection on the error function
    def phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if phi(mid) < 0.02 else (lo, mid)
    assert float(s.link("probit", 0.02)) == pytest.approx(0.5 * (lo + hi), abs=1e-9)
    assert float(s.link("identity", 5.0)) == 5.0 and float(s.link("identity", 0.0)) == 0.0
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="identity"):
            s.link("probit", bad)
        with pytest.raises(ValueError):
            s.link("logit", bad)
    with pytest.raises(ValueError, match="unknown"):
        s.link("cloglog", 0.5)
    with pytest.raises(ValueError):
        s.link("identity", float("nan"))


def test_inverse_link_goldens_and_roundtrip():
    assert float(s.inverse_link("probit", 1.6449)) == pytest.approx(0.95, abs=5e-5)
    assert float(s.inverse_link("logit", -1.3863)) == pytest.approx(0.2, abs=5e-5)
    assert float(s.inverse_link("identity", -0.0003)) == -0.0003
    p = np.linspace(0.001, 0.999, 99)
    for name in ("probit", "logit", "identity"):
        assert np.max(np.abs(s.inverse_link(name, s.link(name, p)) - p)) <= 1e-12, name
    x = np.linspace(-4, 4, 33)
    for name in ("probit", "logit"):
        r = s.inverse_link(name, x)
        assert np.all((r > 0.0) & (r < 1.0)) and np.all(np.diff(r) > 0.0)
    with pytest.raises(ValueError, match="unknown"):
        s.inverse_link("nope", 0.0)


# ---------------------------------------------------------------------------------------------------- RV-1302


def test_fit_satellite_goldens():
    q = quarters(2000, 4)
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    # the identity golden: net recoveries (a negative and a zero rate) fit linearly
    rates = arr(-0.002, 0.0, 0.002, 0.004)
    fit = s.fit_satellite(rates, x, ["real_gdp_growth"], [0], 0, "identity", q, q[0], q[-1])
    assert fit.estimates == pytest.approx((-0.002, 0.002), abs=1e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.coefficient_names == ("intercept", "real_gdp_growth_lag0")
    assert (fit.link, fit.window_start, fit.window_end, fit.n_obs, fit.ar_order) == ("identity", q[0], q[-1], 4, 0)
    assert fit.regressor_names == ("real_gdp_growth",) and fit.lags == (0,)
    with pytest.raises(
        ValueError,
        match="a probit or logit fit on a non-positive rate: the link needs a rate in \\(0, 1\\); "
        "use the identity link",
    ):
        s.fit_satellite(rates, x, ["real_gdp_growth"], [0], 0, "probit", q, q[0], q[-1])
    # the exact line link(rate) = 1 + 2 x, on the identity link (any value allowed) and through the probit
    z = 1.0 + 2.0 * x[:, 0]
    exact = s.fit_satellite(z, x, ["unemployment_rate"], [0], 0, "identity", q, q[0], q[-1])
    assert exact.estimates == pytest.approx((1.0, 2.0), abs=1e-10)
    assert exact.std_errors == pytest.approx((0.0, 0.0), abs=1e-9)
    assert exact.r_squared == pytest.approx(1.0, abs=1e-12)
    ref = np.linalg.lstsq(np.column_stack([np.ones(4), x[:, 0]]), z, rcond=None)[0]
    assert np.max(np.abs(np.array(exact.estimates) - ref)) <= 1e-10
    x2 = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    rate = s.inverse_link("probit", 0.3 + 0.4 * x2[:, 0])
    q5 = quarters(2000, 5)
    pf = s.fit_satellite(rate, x2, ["unemployment_rate"], [0], 0, "probit", q5, q5[0], q5[-1])
    assert pf.estimates == pytest.approx((0.3, 0.4), abs=1e-9) and pf.link == "probit"
    lf = s.fit_satellite(
        s.inverse_link("logit", 0.3 + 0.4 * x2[:, 0]), x2, ["unemployment_rate"], [0], 0, "logit", q5, q5[0], q5[-1]
    )
    assert lf.estimates == pytest.approx((0.3, 0.4), abs=1e-9)


def _design(g, x, ar, lags, lo, hi):
    """The regression the contract describes, built by explicit loops: rows t = lo + L .. hi."""
    big = max([ar, *lags])
    rows, y = [], []
    for t in range(lo + big, hi + 1):
        row = [1.0] + [g[t - k] for k in range(1, ar + 1)]
        for j in range(x.shape[1]):
            row += [x[t - lag, j] for lag in lags]
        rows.append(row)
        y.append(g[t])
    return np.array(rows), np.array(y)


def test_fit_satellite_matches_lstsq_with_ar_and_lags():
    rng = np.random.default_rng(11)
    n = 60
    x = np.column_stack([rng.normal(2.0, 1.0, n), rng.normal(5.0, 2.0, n)])
    g = np.zeros(n)
    for t in range(2, n):
        g[t] = -1.0 + 0.5 * g[t - 1] + 0.1 * x[t, 0] + 0.05 * x[t - 2, 0] - 0.02 * x[t - 1, 1] + rng.normal(0, 0.05)
    rate = s.inverse_link("probit", g)
    q = quarters(1985, n)
    lo, hi = 3, 52
    fit = s.fit_satellite(rate, x, ["real_gdp_growth", "treasury_3m"], [0, 2], 1, "probit", q, q[lo], q[hi])
    gx, yy = _design(g, x, 1, [0, 2], lo, hi)
    # the rows of the window only: t from lo, so the first used row is lo + max lag
    beta = np.linalg.lstsq(gx, yy, rcond=None)[0]
    assert np.max(np.abs(np.array(fit.estimates) - beta)) <= 1e-10
    assert fit.coefficient_names == (
        "intercept",
        "ar_1",
        "real_gdp_growth_lag0",
        "real_gdp_growth_lag2",
        "treasury_3m_lag0",
        "treasury_3m_lag2",
    )
    resid = yy - gx @ beta
    dof = len(yy) - len(beta)
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(gx.T @ gx)))
    assert np.max(np.abs(np.array(fit.std_errors) - se)) <= 1e-9
    assert fit.residual_sd == pytest.approx(math.sqrt(s2), rel=1e-10)
    r2 = 1.0 - float(resid @ resid) / float(((yy - yy.mean()) ** 2).sum())
    assert fit.r_squared == pytest.approx(r2, abs=1e-12) and 0.0 <= fit.r_squared <= 1.0
    assert fit.n_obs == len(yy) == (hi - lo + 1) - 2 and fit.ar_order == 1 and fit.lags == (0, 2)


def test_fit_satellite_window_and_n_obs():
    n = 40
    x = np.linspace(0, 4, n)[:, None] ** 1.5
    g = 0.2 + 0.3 * x[:, 0] + 0.01 * np.sin(np.arange(n))
    rate = s.inverse_link("identity", g)
    q = quarters(1990, n)
    a = s.fit_satellite(rate, x, ["prime_rate"], [0, 1], 0, "identity", q, q[5], q[30])
    assert a.n_obs == 26 - 1
    # quarters outside the window never enter: perturbing them leaves the fit unchanged
    rate2 = rate.copy()
    rate2[:5] += 1.0
    rate2[31:] -= 1.0
    b = s.fit_satellite(rate2, x, ["prime_rate"], [0, 1], 0, "identity", q, q[5], q[30])
    assert a.estimates == b.estimates and a.std_errors == b.std_errors
    # a gap outside the window is out of the window, a gap inside is refused
    q_gap_out = q[:2] + q[3:]
    ok = s.fit_satellite(
        np.delete(rate, 2), np.delete(x, 2, axis=0), ["prime_rate"], [0], 0, "identity", q_gap_out, q[5], q[30]
    )
    assert ok.n_obs == 26
    # an AR term takes its largest lag out of the sample
    c = s.fit_satellite(rate, x, ["prime_rate"], [0], 3, "identity", q, q[5], q[30])
    assert c.n_obs == 26 - 3


def test_fit_satellite_refusals():
    n = 30
    q = quarters(2000, n)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 2))
    rate = np.full(n, 0.02) + 0.005 * rng.normal(size=n) ** 2

    def call(**kw):
        args = dict(
            rates=rate,
            regressors=x,
            regressor_names=["real_gdp_growth", "unemployment_rate"],
            lags=[0],
            ar_order=0,
            link_name="probit",
            quarters=q,
            window_start=q[0],
            window_end=q[-1],
        )
        args.update(kw)
        return s.fit_satellite(**args)

    assert call().n_obs == n
    for bad_rate, word in ((0.0, "non-positive"), (-0.001, "non-positive")):
        r = rate.copy()
        r[7] = bad_rate
        for link_name in ("probit", "logit"):
            with pytest.raises(ValueError, match=word):
                call(rates=r, link_name=link_name)
    r = rate.copy()
    r[7] = 1.0
    with pytest.raises(ValueError, match="1 or more"):
        call(rates=r)
    r[7] = 0.0
    # a non-positive rate outside the window is out of the window
    assert call(rates=r, window_start=q[8]).n_obs == n - 8
    with pytest.raises(ValueError, match="window_end"):
        call(window_end="2030-12-31")
    with pytest.raises(ValueError, match="window_start"):
        call(window_start="1999-12-31")
    with pytest.raises(ValueError, match="gap"):
        call(quarters=q[:10] + q[11:], rates=rate[:-1], regressors=x[:-1])
    with pytest.raises(ValueError, match="bbb_corporate_yield"):
        call(regressor_names=["real_gdp_growth", "bbb_corporate_yield"])
    with pytest.raises(ValueError, match="observations"):
        call(window_end=q[2])
    with pytest.raises(ValueError, match="constant"):
        call(rates=np.full(n, 0.02))
    with pytest.raises(ValueError, match="rank"):
        call(regressors=np.column_stack([x[:, 0], x[:, 0]]))
    with pytest.raises(ValueError, match="link_name"):
        call(link_name="cloglog")
    with pytest.raises(ValueError, match="quarter end"):
        call(quarters=[f"2000-01-{k + 1:02d}" for k in range(n)])
    with pytest.raises(ValueError, match="regressor_names"):
        call(regressor_names=["real_gdp_growth"])
    with pytest.raises(ValueError, match="lags"):
        call(lags=[-1])
    # no input is mutated
    snapshot = (copy.deepcopy(rate), copy.deepcopy(x), list(q))
    call()
    assert np.array_equal(snapshot[0], rate) and np.array_equal(snapshot[1], x) and snapshot[2] == q


# ---------------------------------------------------------------------------------------------------- RV-1304


def _fit(alpha=0.0, phi=(0.5,), beta=0.1, link_name="probit", lags=(0,), names=("unemployment_rate",)):
    coef_names = ["intercept"] + [f"ar_{k + 1}" for k in range(len(phi))]
    est = [alpha, *phi]
    for r in names:
        for lag in lags:
            coef_names.append(f"{r}_lag{lag}")
            est.append(beta)
    return s.SatelliteFit(
        link=link_name,
        window_start="1985-03-31",
        window_end="2019-12-31",
        n_obs=100,
        ar_order=len(phi),
        regressor_names=tuple(names),
        lags=tuple(lags),
        coefficient_names=tuple(coef_names),
        estimates=tuple(est),
        std_errors=tuple(0.0 for _ in est),
        r_squared=0.5,
        residual_sd=0.1,
    )


def test_project_scenario_golden():
    pr = s.project_scenario(_fit(), arr(1.0), np.zeros((0, 1)), np.array([[2.0], [2.0]]))
    assert pr.link_values == pytest.approx((0.7, 0.55), abs=1e-12)
    assert pr.rates == pytest.approx((0.758036, 0.708840), abs=5e-7)
    assert np.max(np.abs(pr.rates - s.inverse_link("probit", pr.link_values))) == 0.0
    # the same recursion by an explicit loop with two lags, an AR(2) and the history rows
    fit = _fit(alpha=-0.3, phi=(0.4, 0.2), beta=0.05, lags=(0, 1))
    hist = np.array([[1.0]])
    path = np.array([[2.0], [3.0], [1.0], [0.0]])
    out = s.project_scenario(fit, arr(0.5, 0.8), hist, path)  # last link values: g_{-1} = 0.5, g_0 = 0.8
    g = {-1: 0.5, 0: 0.8}
    xs = {0: 1.0, **{k + 1: path[k, 0] for k in range(4)}}
    for t in range(1, 5):
        g[t] = -0.3 + 0.4 * g[t - 1] + 0.2 * g[t - 2] + 0.05 * xs[t] + 0.05 * xs[t - 1]
    assert out.link_values == pytest.approx([g[t] for t in range(1, 5)], abs=1e-12)
    # identity link: the rate is the link value and may be a net recovery
    idf = s.project_scenario(
        _fit(alpha=-0.002, phi=(0.0,), beta=0.001, link_name="identity"),
        arr(0.0),
        np.zeros((0, 1)),
        np.array([[0.0], [1.0]]),
    )
    assert idf.rates == pytest.approx((-0.002, -0.001), abs=1e-15) and np.array_equal(idf.rates, idf.link_values)


def test_project_scenario_monotone_and_shapes():
    fit = _fit(beta=0.2, lags=(0, 1))
    hist = np.array([[1.0]])
    base = np.array([[1.0], [2.0], [3.0]])
    lo = s.project_scenario(fit, arr(0.0), hist, base)
    hi = s.project_scenario(fit, arr(0.0), hist, base + np.array([[0.0], [0.5], [1.0]]))
    assert np.all(hi.link_values >= lo.link_values) and np.all(hi.rates >= lo.rates)
    with pytest.raises(ValueError, match="last_link_values"):
        s.project_scenario(fit, arr(0.0, 0.0), hist, base)
    with pytest.raises(ValueError, match="regressor_history"):
        s.project_scenario(fit, arr(0.0), np.zeros((2, 1)), base)
    with pytest.raises(ValueError, match="regressor_path"):
        s.project_scenario(fit, arr(0.0), hist, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="regressor_path"):
        s.project_scenario(fit, arr(0.0), hist, np.zeros((0, 1)))
    inputs = (arr(0.0), hist.copy(), base.copy())
    s.project_scenario(fit, *inputs)
    assert np.array_equal(inputs[1], hist) and np.array_equal(inputs[2], base)


def test_probit_round_trip_through_the_reused_vasicek_functions():
    """DR to Z to DR (dossier 04 l.791-793): the satellite's probit rate maps to the implied factor and back."""
    assert float(conditional_pd(0.02, 0.12, 0.0)) == pytest.approx(0.014287, abs=5e-7)
    assert float(conditional_pd(0.02, 0.12, -2.0)) == pytest.approx(0.073424, abs=5e-7)
    pr = s.project_scenario(_fit(alpha=-2.0, phi=(0.0,), beta=0.0), arr(0.0), np.zeros((0, 1)), np.zeros((3, 1)))
    for rate in pr.rates:
        z = float(implied_factor(float(rate), 0.02, 0.12))
        assert float(conditional_pd(0.02, 0.12, z)) == pytest.approx(float(rate), abs=1e-12)
    z = float(implied_factor(0.073424, 0.02, 0.12))
    assert z == pytest.approx(-2.0, abs=1e-4)
