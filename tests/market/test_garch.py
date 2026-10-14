"""RV-807: GARCH, GJR and EGARCH by quasi-maximum likelihood through `arch`, and the VaR and ES forecast (contract
`engine-market-api`, module market.garch). The tests that call `arch` skip when the optional extra `market` is absent."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from scipy import stats

from market_sim import simulate_garch
from riskvalidation.market import garch as g
from riskvalidation.market import parametric as p


def test_garch_forecast_normal_golden():
    fit = g.GarchFit(kind="garch", innovation="normal",
                     params={"mu": 0.0, "omega": 1e-4, "alpha[1]": 0.0, "beta[1]": 0.0}, log_likelihood=0.0,
                     converged=True, n_obs=300)
    y = np.random.default_rng(1).normal(0, 0.01, size=300)
    pytest.importorskip("arch")
    v, e = g.garch_var_es_forecast(fit, y, 0.01)
    assert v == pytest.approx(-0.023263, abs=5e-7) and e == pytest.approx(-0.026652, abs=5e-7)
    assert (v, e) == p.delta_normal_var_es(0.0, 0.01, 0.01)
    assert e <= v


def test_garch_forecast_student_t_closed_form():
    arch = pytest.importorskip("arch")
    y = simulate_garch(600, nu=6)
    params = {"mu": 0.0003, "omega": 2e-6, "alpha[1]": 0.08, "beta[1]": 0.9, "nu": 6.0}
    fit = g.GarchFit(kind="garch", innovation="student-t", params=params, log_likelihood=0.0, converged=True,
                     n_obs=600)
    v, e = g.garch_var_es_forecast(fit, y, 0.025)
    # the one-step variance from arch's own filter, with the fixed parameters in percent units
    am = arch.arch_model(y * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    names = am.parameter_names()
    scale = {"mu": 100.0, "omega": 1e4}
    vec = [params[n] * scale.get(n, 1.0) for n in names]
    s2 = float(am.fix(vec).forecast(horizon=1, reindex=False).variance.values[-1, 0]) / 1e4
    sigma = s2**0.5
    nu = 6.0
    q = stats.t.ppf(0.025, nu) * np.sqrt((nu - 2) / nu)
    es_q = -(stats.t.pdf(stats.t.ppf(0.025, nu), nu) / 0.025) * ((nu + stats.t.ppf(0.025, nu) ** 2) / (nu - 1)) * np.sqrt(
        (nu - 2) / nu)
    assert v == pytest.approx(0.0003 + sigma * q, rel=1e-9)
    assert e == pytest.approx(0.0003 + sigma * es_q, rel=1e-9)
    assert e <= v
    # the closed form agrees with the quadrature of the definition
    from riskvalidation.market.conventions import var_es_from_distribution
    vq, eq = var_es_from_distribution(lambda u: stats.t.ppf(u, nu) * np.sqrt((nu - 2) / nu), 0.025)
    assert eq == pytest.approx(es_q, abs=1e-4) and vq == pytest.approx(q, abs=1e-9)
    # no read beyond the last element: the same series with a later day appended gives the forecast of that later day,
    # and a function given the first 600 days is unaffected by what the caller holds after them
    long = np.concatenate([y, [0.5]])
    assert g.garch_var_es_forecast(fit, long[:600], 0.025) == (v, e)


def test_fit_garch_recovers_known_parameters():
    pytest.importorskip("arch")
    y = simulate_garch(3000, nu=6, seed=21)
    f = g.fit_garch(y, "garch", "student-t", "constant")
    assert isinstance(f, g.GarchFit) and f.converged and f.n_obs == 3000
    assert f.kind == "garch" and f.innovation == "student-t"
    assert f.params["omega"] > 0 and f.params["alpha[1]"] >= 0 and f.params["beta[1]"] >= 0
    assert f.params["alpha[1]"] + f.params["beta[1]"] < 1
    assert f.params["alpha[1]"] == pytest.approx(0.08, abs=0.04)
    assert f.params["beta[1]"] == pytest.approx(0.90, abs=0.05)
    assert f.params["omega"] == pytest.approx(2e-6, rel=1.0)  # decimals squared, not percent squared
    assert f.params["mu"] == pytest.approx(0.0003, abs=0.0005)
    assert 3.0 < f.params["nu"] < 15.0 and f.log_likelihood > 0
    # normal innovations carry no degrees of freedom
    fn = g.fit_garch(y, "garch", "normal", "zero")
    assert "nu" not in fn.params and "mu" not in fn.params


def test_fit_garch_kinds_deterministic_and_refusals():
    pytest.importorskip("arch")
    y = simulate_garch(1200, nu=7, seed=31, gamma=0.08)
    for kind in ("garch", "gjr", "egarch"):
        f1 = g.fit_garch(y, kind, "student-t")
        f2 = g.fit_garch(y, kind, "student-t")
        assert f1 == f2 and f1.kind == kind
        assert f1.params["nu"] > 2.0
        v, e = g.garch_var_es_forecast(f1, y, 0.01)
        assert e <= v < 0.0
    gjr = g.fit_garch(y, "gjr", "student-t")
    assert gjr.params["gamma[1]"] > 0.0  # the leverage term of the data
    sk = g.fit_garch(y, "garch", "skew-t")
    assert sk.innovation == "skew-t" and "lambda" in sk.params
    v, e = g.garch_var_es_forecast(sk, y, 0.025)
    assert e <= v
    with pytest.raises(ValueError):
        g.fit_garch(y[:249])
    with pytest.raises(ValueError):
        g.fit_garch(np.append(y, np.nan))
    for kw in ({"kind": "tgarch"}, {"innovation": "cauchy"}, {"mean": "ar"}):
        with pytest.raises(ValueError):
            g.fit_garch(y, **kw)


def test_arch_is_imported_lazily():
    code = (
        "import sys, riskvalidation.market.garch, riskvalidation.market.evt, riskvalidation.market.scoring,"
        " riskvalidation.validation.market;"
        "sys.exit(1 if 'arch' in sys.modules else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    # a missing arch gives a clear error naming the extra
    code2 = (
        "import sys; sys.modules['arch'] = None\n"
        "import numpy as np\n"
        "from riskvalidation.market import garch\n"
        "try:\n"
        "    garch.fit_garch(np.zeros(300))\n"
        "except ImportError as e:\n"
        "    assert 'market' in str(e); sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    out = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
