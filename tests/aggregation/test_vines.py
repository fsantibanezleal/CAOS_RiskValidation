"""RV-1906 to RV-1910: the vine copula functions of `aggregation.vines` (optional extra `vine`, pyvinecopulib 1.0.1)."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from riskvalidation.aggregation import copulas as c
from riskvalidation.aggregation import vines as v

ROOT = Path(__file__).resolve().parents[2]
MEASURED_ON = "1.0.1"


def _pv():
    return pytest.importorskip("pyvinecopulib")


def _gauss(pv, rho):
    return pv.Bicop(pv.BicopFamily.gaussian, 0, np.array([[rho]]))


def _clayton(pv, theta, rotation=0):
    return pv.Bicop(pv.BicopFamily.clayton, rotation, np.array([[theta]]))


def test_core_imports_without_the_extra():
    # the module imports with pyvinecopulib blocked, and a call that needs it raises ImportError naming the extra
    code = (
        "import sys; sys.modules['pyvinecopulib'] = None\n"
        "import riskvalidation.aggregation.vines as v, numpy as np\n"
        "try:\n"
        "    v.fit_vine(np.random.default_rng(0).uniform(size=(50, 3)))\n"
        "except ImportError as e:\n"
        "    print('ImportError', 'vine' in str(e), 'pyvinecopulib' in str(e)); sys.exit(0)\n"
        "sys.exit(3)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ImportError True True" in out.stdout
    # the core modules never import it
    code2 = (
        "import sys, riskvalidation.aggregation.copulas, riskvalidation.aggregation.vines,"
        " riskvalidation.stress.scenario_models;"
        "sys.exit(1 if 'pyvinecopulib' in sys.modules else 0)"
    )
    out2 = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True)
    assert out2.returncode == 0, out2.stdout + out2.stderr


def test_pinned_version():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'vine = ["pyvinecopulib>=1.0.1,<2; python_version >= \'3.11\'"]' in text
    pv = _pv()
    assert pv.__version__ == MEASURED_ON, f"the reconciliation was measured on pyvinecopulib {MEASURED_ON}"


def test_fit_vine_record_and_structure_recovery():
    pv = _pv()
    # a C-vine on 4 variables: variable 4 is the centre of the strong star in tree 1, trees 2 and 3 are weak
    truth = pv.Vinecop.from_structure(
        structure=pv.CVineStructure([1, 2, 3, 4]),
        pair_copulas=[[_gauss(pv, 0.85)] * 3, [_gauss(pv, 0.1)] * 2, [_gauss(pv, 0.1)]],
    )
    m = np.asarray(truth.matrix)
    centre = int(m[0, 0])
    assert all(int(m[0, j]) == centre for j in range(3))
    u = truth.sample(3000, seeds=[3])
    fit = v.fit_vine(u, family_set=("indep", "gaussian", "clayton"), seed=1)
    assert isinstance(fit, v.VineFit)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.n_params = 3  # frozen
    assert fit.dim == 4 and fit.n_obs == 3000
    assert fit.loglik == pytest.approx(fit.model.loglik(u), rel=1e-12)
    assert fit.aic == pytest.approx(fit.model.aic(u), rel=1e-12)
    assert fit.aic == pytest.approx(-2.0 * fit.loglik + 2.0 * fit.n_params, rel=1e-9)
    assert fit.n_params == pytest.approx(fit.model.npars)
    desc = json.loads(json.dumps(fit.description))
    assert desc["dim"] == 4 and len(desc["trees"]) == 3 and len(desc["trees"][0]) == 3
    edge = desc["trees"][0][0]
    assert set(edge) >= {"conditioned", "conditioning", "family", "rotation", "parameters", "tau"}
    # tree 1 is the star on the true centre
    for e in desc["trees"][0]:
        assert centre in e["conditioned"]
        assert e["conditioning"] == []
        assert e["family"] == "gaussian"
        assert abs(e["parameters"][0]) > 0.8
    # the fit is reproducible and the seed and one thread are fixed
    again = v.fit_vine(u, family_set=("indep", "gaussian", "clayton"), seed=1)
    assert again.description == fit.description and again.loglik == fit.loglik
    # a fixed structure is kept: only the pair-copulas are fitted
    fixed = v.fit_vine(u, family_set=("gaussian",), structure="d_vine", seed=1)
    assert fixed.description["structure"] == "d_vine"
    # refusals
    with pytest.raises(ValueError, match="u"):
        v.fit_vine(np.array([[0.2, 1.2], [0.1, 0.3], [0.5, 0.5]]))
    with pytest.raises(ValueError, match="u"):
        v.fit_vine(np.full((10, 1), 0.5))
    with pytest.raises(ValueError, match="family_set"):
        v.fit_vine(u, family_set=("nope",))


def test_simulate_known_vines():
    pv = _pv()
    # d = 2 with a single Clayton pair copula: the Kendall tau is theta / (theta + 2)
    theta = 2.0
    v2 = v.vine_fit_from_model(pv.Vinecop.from_structure(structure=pv.DVineStructure([1, 2]),
                                                       pair_copulas=[[_clayton(pv, theta)]]))
    s = v.simulate_vine(v2, 400_000, seed=5)
    assert s.shape == (400_000, 2)
    assert stats.kendalltau(s[:, 0], s[:, 1]).statistic == pytest.approx(c.clayton_kendall_tau(theta), abs=0.003)
    assert np.array_equal(v.simulate_vine(v2, 1000, seed=5), v.simulate_vine(v2, 1000, seed=5))
    assert not np.array_equal(v.simulate_vine(v2, 1000, seed=5), v.simulate_vine(v2, 1000, seed=6))
    # d = 3 D-vine, Gaussian pair copulas 0.5, 0.5 and 0.3 (conditional): rho_13 = 0.3 sqrt(0.75 0.75) + 0.25 = 0.475
    v3 = v.vine_fit_from_model(pv.Vinecop.from_structure(
        structure=pv.DVineStructure([1, 2, 3]), pair_copulas=[[_gauss(pv, 0.5), _gauss(pv, 0.5)], [_gauss(pv, 0.3)]]))
    x = v.simulate_vine(v3, 200_000, seed=7)
    z = stats.norm.ppf(x)
    rho = np.corrcoef(z.T)
    assert rho[0, 2] == pytest.approx(0.3 * 0.75 + 0.25, abs=0.01)
    assert rho[0, 1] == pytest.approx(0.5, abs=0.01) and rho[1, 2] == pytest.approx(0.5, abs=0.01)
    # the Rosenblatt transform of the vine's own sample is independent uniform
    r = np.asarray(v3.model.rosenblatt(x))
    for j in range(3):
        assert stats.kstest(r[:, j], "uniform").pvalue > 1e-3
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(stats.kendalltau(r[:, i], r[:, j]).statistic) < 0.01
    with pytest.raises(ValueError, match="n"):
        v.simulate_vine(v3, 0, seed=0)


def _dvine3_loglik_by_engine(u, family_12, family_23, family_132):
    """Aas et al. (2009) eq. (17) for the 3-variable D-vine: log c12(u1, u2) + log c23(u2, u3) + log c13|2(h(u1 | u2),
    h(u3 | u2)), the h-functions and log-densities being the engine's own."""
    h12 = family_12["h"](u[:, 0], u[:, 1])        # F(u1 | u2) from the pair (1, 2)
    h32 = family_23["h"](u[:, 2], u[:, 1])        # F(u3 | u2) from the pair (2, 3)
    return float(np.sum(family_12["logc"](u[:, 0], u[:, 1]) + family_23["logc"](u[:, 1], u[:, 2])
                        + family_132["logc"](h12, h32)))


@pytest.mark.parametrize("variant", ["clayton_gauss_clayton", "gauss_gauss_gauss", "survival_clayton_gauss_clayton"])
def test_loglik_reconciles_with_the_engine_h_functions(variant):
    pv = _pv()
    th12, th23, th132 = 2.0, 0.6, 1.5

    def clayton_family(theta):
        return {"h": lambda a, b: c.clayton_h_function(a, b, theta),
                "logc": lambda a, b: v.clayton_log_density(a, b, theta)}

    def survival_clayton_family(theta):
        return {"h": lambda a, b: 1.0 - c.clayton_h_function(1.0 - a, 1.0 - b, theta),
                "logc": lambda a, b: v.clayton_log_density(1.0 - a, 1.0 - b, theta)}

    def gauss_family(rho):
        return {"h": lambda a, b: v.gaussian_h_function(a, b, rho),
                "logc": lambda a, b: v.gaussian_log_density(a, b, rho)}

    if variant == "clayton_gauss_clayton":
        pcs = [[_clayton(pv, th12), _gauss(pv, th23)], [_clayton(pv, th132)]]
        f12, f23, f132 = clayton_family(th12), gauss_family(th23), clayton_family(th132)
    elif variant == "gauss_gauss_gauss":
        pcs = [[_gauss(pv, 0.5), _gauss(pv, -0.4)], [_gauss(pv, 0.3)]]
        f12, f23, f132 = gauss_family(0.5), gauss_family(-0.4), gauss_family(0.3)
    else:
        pcs = [[_clayton(pv, th12, 180), _gauss(pv, th23)], [_clayton(pv, th132)]]
        f12, f23, f132 = survival_clayton_family(th12), gauss_family(th23), clayton_family(th132)
    model = pv.Vinecop.from_structure(structure=pv.DVineStructure([1, 2, 3]), pair_copulas=pcs)
    rng = np.random.default_rng(11)
    u = np.clip(rng.uniform(size=(400, 3)), 0.02, 0.98)       # a fixed dataset, not the vine's own sample
    u[:200] = model.sample(200, seeds=[4])                    # half of it from the vine, half far from it
    ours = _dvine3_loglik_by_engine(u, f12, f23, f132)
    assert ours == pytest.approx(model.loglik(u), abs=1e-9)
    # and the record of `vine_fit_from_model` carries the same number
    assert v.vine_fit_from_model(model, u).loglik == pytest.approx(ours, abs=1e-9)


def test_pair_copula_helpers():
    # the Gaussian h-function is Phi((z1 - rho z2) / sqrt(1 - rho^2)) and its log-density the bivariate normal ratio
    u1, u2 = np.array([0.2, 0.6, 0.9]), np.array([0.3, 0.5, 0.1])
    rho = 0.4
    z1, z2 = stats.norm.ppf(u1), stats.norm.ppf(u2)
    assert v.gaussian_h_function(u1, u2, rho) == pytest.approx(stats.norm.cdf((z1 - rho * z2) / np.sqrt(1 - rho**2)))
    mv = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    ref = mv.logpdf(np.column_stack([z1, z2])) - stats.norm.logpdf(z1) - stats.norm.logpdf(z2)
    assert v.gaussian_log_density(u1, u2, rho) == pytest.approx(ref, abs=1e-12)
    # the Clayton log-density is the log of the derivative of the h-function in its first argument (second route)
    th = 2.0
    eps = 1e-6
    d = (c.clayton_h_function(u1 + eps, u2, th) - c.clayton_h_function(u1 - eps, u2, th)) / (2 * eps)
    assert np.exp(v.clayton_log_density(u1, u2, th)) == pytest.approx(d, rel=1e-6)
    with pytest.raises(ValueError, match="rho"):
        v.gaussian_h_function(u1, u2, 1.0)
    with pytest.raises(ValueError, match="theta"):
        v.clayton_log_density(u1, u2, 0.0)


def test_vine_aggregate():
    pv = _pv()
    model = pv.Vinecop.from_structure(
        structure=pv.DVineStructure([1, 2, 3]), pair_copulas=[[_clayton(pv, 2.0), _gauss(pv, 0.5)], [_gauss(pv, 0.3)]])
    fit = v.vine_fit_from_model(model)

    def pareto(u):
        return (1.0 - np.asarray(u, dtype=float)) ** -0.5 - 1.0

    ppfs = [pareto] * 3
    a = v.vine_aggregate(ppfs, fit, el=0.0, level=0.99, n_draws=200_000, seed=3)
    b = v.vine_aggregate(ppfs, fit, el=0.0, level=0.99, n_draws=200_000, seed=3)
    assert a == b
    assert a.es >= a.var > 0.0 and a.ec == pytest.approx(a.var - a.el)
    assert a.var_se is not None and a.es_se is not None
    # the measures are those of measures_from_sample on the row sums of the vine's own draws
    from riskvalidation.portfolio.portfolio import measures_from_sample

    u = v.simulate_vine(fit, 200_000, seed=3)
    ref = measures_from_sample(pareto(u).sum(axis=1), [0.99], 0.0)[0]
    assert a.var == pytest.approx(ref.var, rel=1e-12) and a.es == pytest.approx(float(ref.es), rel=1e-12)
    with pytest.raises(ValueError, match="ppfs"):
        v.vine_aggregate([pareto, pareto], fit, el=0.0)
    with pytest.raises(ValueError, match="el"):
        v.vine_aggregate(ppfs, fit, el=float("nan"))
