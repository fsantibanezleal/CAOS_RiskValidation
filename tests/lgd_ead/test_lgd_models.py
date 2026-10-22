"""RV-605 to RV-610: the LGD rungs of dossier 04 F.3 (OLS, fractional logit, beta, hurdle, Tobit, cure
decomposition) and
`predict_lgd` (contract `engine-lgd-ead-api`, module lgd_models)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import optimize, stats

from riskvalidation.lgd_ead import lgd_models as m


def a(*v):
    return np.array(v, dtype=float)


def A(rows):
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------- RV-605


def test_endpoint_transform():
    np.testing.assert_allclose(m.endpoint_transform(a(0.0, 0.5, 1.0), 0.001), [0.001, 0.5, 0.999], atol=1e-15)
    y = a(0.002, 0.3, 0.998)
    np.testing.assert_array_equal(m.endpoint_transform(y, 0.001), y)  # inside values unchanged
    with pytest.raises(ValueError, match="y"):  # review F7: a value outside [0, 1] is refused, never mapped silently
        m.endpoint_transform(a(-0.2, 0.5), 0.01)
    with pytest.raises(ValueError, match="y"):
        m.endpoint_transform(a(0.5, 1.4), 0.01)
    for bad in (0.0, 0.5, -0.1, 0.7, float("nan")):
        with pytest.raises(ValueError, match="epsilon"):
            m.endpoint_transform(a(0.1), bad)
    with pytest.raises(ValueError):
        m.endpoint_transform(a(np.nan), 0.01)


def test_fit_ols_golden_and_clipping():
    fit = m.fit_ols(A([[1, 0], [1, 1]]), a(0, 1))
    assert isinstance(fit, m.LgdFit) and dataclasses.is_dataclass(fit)
    assert fit.kind == "ols"
    np.testing.assert_allclose(fit.coefficients, [0.0, 1.0], atol=1e-12)
    assert fit.names == ("x0", "x1")
    # the clipping happens in predict_lgd, the only place a model clips
    assert m.predict_lgd(fit, A([[1, 2]]))[0] == 1.0
    assert m.predict_lgd(fit, A([[1, -3]]))[0] == 0.0
    assert m.predict_lgd(fit, A([[1, 0.25]]))[0] == pytest.approx(0.25)
    named = m.fit_ols(A([[1, 0], [1, 1]]), a(0, 1), names=("const", "z"))
    assert named.names == ("const", "z")
    with pytest.raises(ValueError, match="rank"):
        m.fit_ols(A([[1, 2], [1, 2], [1, 2]]), a(0.1, 0.2, 0.3))
    with pytest.raises(ValueError, match="names"):
        m.fit_ols(A([[1, 0], [1, 1]]), a(0, 1), names=("only",))
    with pytest.raises(ValueError):
        m.fit_ols(A([[1, 0], [1, 1]]), a(0, 1, 0))
    with pytest.raises(ValueError, match="non-finite"):
        m.fit_ols(A([[1, 0], [1, np.nan]]), a(0, 1))


# ---------------------------------------------------------------- RV-606


def test_fractional_logit_golden_and_score():
    X = A([[1, 0], [1, 0], [1, 1], [1, 1]])
    y = a(0, 0.5, 0.5, 1.0)
    fit = m.fit_fractional_logit(X, y)
    assert fit.kind == "fractional-logit"
    np.testing.assert_allclose(fit.coefficients, [-1.098612, 2.197225], atol=5e-7)
    np.testing.assert_allclose(fit.coefficients, [np.log(1 / 3), 2 * np.log(3)], atol=1e-8)
    np.testing.assert_allclose(m.predict_lgd(fit, A([[1, 0], [1, 1]])), [0.25, 0.75], atol=1e-8)
    # the quasi-score equation X'(y - mu) = 0 and sum(mu) = sum(y) hold on data without a closed form
    rng = np.random.default_rng(7)
    n = 400
    Xr = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    p = 1 / (1 + np.exp(-(0.3 + 0.8 * Xr[:, 1] - 0.5 * Xr[:, 2])))
    yr = np.clip(p + rng.normal(scale=0.15, size=n), 0.0, 1.0)
    yr[:30] = 0.0
    yr[30:60] = 1.0
    fr = m.fit_fractional_logit(Xr, yr)
    mu = m.predict_lgd(fr, Xr)
    assert np.max(np.abs(Xr.T @ (yr - mu))) < 1e-8
    assert mu.sum() == pytest.approx(yr.sum(), abs=1e-8)
    assert np.all((mu > 0) & (mu < 1))
    # the coefficients maximise the Bernoulli quasi-likelihood: no direction improves it
    def nll(b):
        z = Xr @ b
        return float(np.sum(np.logaddexp(0, z) - yr * z))
    ref = optimize.minimize(nll, np.zeros(3), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14,
                                                                           "maxiter": 5000}).x
    np.testing.assert_allclose(fr.coefficients, ref, atol=1e-4)
    # a response outside [0, 1] is refused, never clipped
    with pytest.raises(ValueError, match="y"):
        m.fit_fractional_logit(X, a(0, 0.5, 0.5, 1.2))
    with pytest.raises(ValueError, match="y"):
        m.fit_fractional_logit(X, a(-0.1, 0.5, 0.5, 1.0))
    with pytest.raises(ValueError, match="rank"):
        m.fit_fractional_logit(A([[1, 1], [1, 1], [1, 1], [1, 1]]), y)
    # perfect separation has no finite maximiser: refused, not silently returned
    with pytest.raises(ValueError, match="converge|separation"):
        m.fit_fractional_logit(A([[1, 0], [1, 1], [1, 2], [1, 3]]), a(0, 0, 1, 1))


def test_fractional_logit_is_scale_invariant_in_the_covariate_unit():
    """Review F2: a legitimate fit with a coefficient above 40 (a rate in decimals) is not refused, and the same data in
    percent gives the same predictions."""
    rate = np.linspace(0.02, 0.10, 17)
    y = 1 / (1 + np.exp(-(-3.0 + 50.0 * rate)))
    dec = m.fit_fractional_logit(np.column_stack([np.ones(17), rate]), y)
    pct = m.fit_fractional_logit(np.column_stack([np.ones(17), rate * 100]), y)
    np.testing.assert_allclose(dec.coefficients, [-3.0, 50.0], atol=1e-5)
    np.testing.assert_allclose(pct.coefficients, [-3.0, 0.5], atol=1e-7)
    np.testing.assert_allclose(m.predict_lgd(dec, np.column_stack([np.ones(17), rate])),
                               m.predict_lgd(pct, np.column_stack([np.ones(17), rate * 100])), atol=1e-8)
    # the direct EAD model and the hurdle stage 2 use the same solver
    from riskvalidation.lgd_ead import ccf
    ead = ccf.fit_direct_ead(np.column_stack([np.ones(17), rate]), 1000 * y, np.full(17, 1000.0))
    np.testing.assert_allclose(ead.coefficients, [-3.0, 50.0], atol=1e-5)
    mixed = np.where(np.arange(17) % 4 == 0, 0.0, y)
    h = m.fit_hurdle(np.column_stack([np.ones(17), rate]), mixed, min_positive=5)
    assert np.all(np.isfinite(h.coefficients)) and abs(h.coefficients[1]) > 40
    # genuine separation is still refused
    with pytest.raises(ValueError, match="converge"):
        m.fit_fractional_logit(np.column_stack([np.ones(4), [0.0, 0.01, 0.02, 0.03]]), a(0, 0, 1, 1))


# the 1.0.0 solver's coefficients of the well-conditioned fit below (recorded before the BL-094 change)
GOLDEN_WELL_CONDITIONED = [0.2147058663304752, 0.5523970735768755, -0.39250684846553585]


def _large_fractional(seed: int, n: int = 70_000, p: int = 17):
    """A design like the Contraste C08 cohorts: standardized covariates, a fractional response with some endpoints,
    tens of thousands of rows (seeds 7 and 24 stalled the 1.0.0 solver at 4 BLAS threads)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    mu = 1 / (1 + np.exp(-(X @ rng.normal(0, 0.3, p))))
    return X, np.clip(mu + rng.normal(0, 0.25, n), 0.0, 1.0)


@pytest.mark.parametrize("seed", [7, 24])
def test_fractional_logit_converges_when_the_gain_is_below_the_likelihood_rounding(seed):
    """BL-094: near the optimum the predicted gain of a Newton step (about 1e-12) is below the rounding of a
    log-likelihood summed over tens of thousands of rows (about 1e-8), so a step-halving test on that sum cannot tell a
    better point from a worse one. The 1.0.0 solver halved such steps away and stalled with a score near 1e-8 ("did
    not converge within 200 iterations", the C08 L2 and L4 rungs). The full step is taken when the decrement is below
    the rounding bound; the fit is the optimum."""
    X, y = _large_fractional(seed)
    fit = m.fit_fractional_logit(X, y)
    mu = m.predict_lgd(fit, X)
    assert np.max(np.abs(X.T @ (y - mu))) < 1e-9
    def nll(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z))
    def grad(b):
        return X.T @ (1 / (1 + np.exp(-(X @ b))) - y)
    ref = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B",
                            options={"maxiter": 10_000, "gtol": 1e-10, "ftol": 1e-16}).x
    np.testing.assert_allclose(fit.coefficients, ref, atol=1e-5)
    # the hurdle's two stages use the same solver
    yz = np.where(np.arange(len(y)) % 7 == 0, 0.0, y)
    h = m.fit_hurdle(X, yz)
    assert np.all(np.isfinite(h.coefficients))


def test_fractional_logit_keeps_its_path_on_a_well_conditioned_fit():
    """The decrement rule changes nothing where the log-likelihood can arbitrate: the 1.0.0 coefficients of a small
    fit stand."""
    rng = np.random.default_rng(11)
    n = 2_000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = np.clip(1 / (1 + np.exp(-(0.2 + 0.6 * X[:, 1] - 0.4 * X[:, 2]))) + rng.normal(scale=0.2, size=n), 0.0, 1.0)
    np.testing.assert_allclose(m.fit_fractional_logit(X, y).coefficients, GOLDEN_WELL_CONDITIONED, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- RV-607


def _beta_data(seed=3, n=300):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    mu = 1 / (1 + np.exp(-(-0.2 + 0.7 * X[:, 1])))
    y = rng.beta(mu * 6.0, (1 - mu) * 6.0)
    y[:25] = 0.0
    y[25:40] = 1.0
    return X, y


def test_beta_invariants_and_endpoint_sensitivity():
    X, y = _beta_data()
    f1 = m.fit_beta(X, y, 0.01)
    f2 = m.fit_beta(X, y, 0.0001)
    for f, eps in ((f1, 0.01), (f2, 0.0001)):
        assert f.kind == "beta"
        assert f.extras["epsilon"] == eps and f.extras["phi"] > 0
        mu = m.predict_lgd(f, X)
        assert np.all((mu > 0) & (mu < 1))
    # both are returned and the fit is a function of epsilon (the endpoint sensitivity is reported, not hidden)
    assert not np.allclose(f1.coefficients, f2.coefficients, atol=1e-6) or f1.extras["phi"] != f2.extras["phi"]
    # deterministic
    again = m.fit_beta(X, y, 0.01)
    np.testing.assert_array_equal(again.coefficients, f1.coefficients)
    # the optimum of the beta log-likelihood on the transformed response (independent evaluation with scipy.stats)
    yt = m.endpoint_transform(y, 0.01)

    def nll(theta):
        mu_ = 1 / (1 + np.exp(-(X @ theta[:2])))
        phi = np.exp(theta[2])
        return -float(stats.beta.logpdf(yt, mu_ * phi, (1 - mu_) * phi).sum())

    theta = np.append(f1.coefficients, np.log(f1.extras["phi"]))
    best = optimize.minimize(nll, theta + 0.3, method="Nelder-Mead",
                             options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 8000})
    assert nll(theta) <= best.fun + 1e-6
    # on an exact-moment problem the mean is recovered: constant mean 0.3 gives mu = 0.3
    flat = m.fit_beta(A([[1]] * 6), a(0.2, 0.3, 0.4, 0.25, 0.35, 0.3), 0.01)
    assert m.predict_lgd(flat, A([[1]]))[0] == pytest.approx(0.3, abs=1e-3)
    for bad in (0.0, 0.5, 0.6, -1.0):
        with pytest.raises(ValueError, match="epsilon"):
            m.fit_beta(X, y, bad)
    with pytest.raises(ValueError, match="y"):
        m.fit_beta(X, y + 0.5, 0.01)


# ---------------------------------------------------------------- RV-608


def test_hurdle_golden_and_refusals():
    X = A([[1], [1], [1], [1]])
    y = a(0, 0.4, 0.4, 0.4)
    fit = m.fit_hurdle(X, y, min_positive=3)
    assert fit.kind == "hurdle"
    assert fit.coefficients[0] == pytest.approx(np.log(0.4 / 0.6), abs=1e-8)
    assert fit.extras["stage1_coefficients"][0] == pytest.approx(np.log(3.0), abs=1e-8)
    assert m.predict_lgd(fit, A([[1]]))[0] == pytest.approx(0.3, abs=1e-9)
    with pytest.raises(ValueError, match="min_positive"):
        m.fit_hurdle(X, y)  # 3 positive losses < the default 5
    with pytest.raises(ValueError, match="y"):
        m.fit_hurdle(X, a(0, 0.4, 0.4, 1.4), min_positive=3)
    # a separate severity design, a column subset of X, is carried to the prediction
    rng = np.random.default_rng(11)
    n = 300
    Xf = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    pos = rng.random(n) < 1 / (1 + np.exp(-(0.5 + Xf[:, 1])))
    sev = 1 / (1 + np.exp(-(-0.4 + 0.6 * Xf[:, 2])))
    yy = np.where(pos, np.clip(sev + rng.normal(scale=0.1, size=n), 0.01, 1.0), 0.0)
    f2 = m.fit_hurdle(Xf, yy, X_severity=Xf[:, [0, 2]])
    p = m.predict_lgd(f2, Xf)
    assert p.shape == (n,) and np.all((p >= 0) & (p <= 1))
    s1 = 1 / (1 + np.exp(-(Xf @ f2.extras["stage1_coefficients"])))
    s2 = 1 / (1 + np.exp(-(Xf[:, [0, 2]] @ f2.coefficients)))
    np.testing.assert_allclose(p, s1 * s2, atol=1e-12)
    with pytest.raises(ValueError, match="column"):
        m.fit_hurdle(Xf, yy, X_severity=np.column_stack([np.ones(n), rng.normal(size=n)]))


# ---------------------------------------------------------------- RV-609


def _tobit_negloglik(b, s, X, y):
    xb = X @ b
    ll = np.where(y <= 0, stats.norm.logcdf((0 - xb) / s),
                  np.where(y >= 1, stats.norm.logsf((1 - xb) / s), stats.norm.logpdf((y - xb) / s) - np.log(s)))
    return -float(ll.sum())


def test_tobit_golden_and_censored_optimum():
    fit = m.fit_tobit(A([[1, 0], [1, 1], [1, 2]]), a(0.2, 0.3, 0.6))
    assert fit.kind == "tobit"
    np.testing.assert_allclose(fit.coefficients, [0.166667, 0.2], atol=1e-6)
    assert fit.extras["sigma"] == pytest.approx(0.0471405, abs=1e-6)
    assert fit.extras["sigma"] == pytest.approx(np.sqrt(0.0066666667 / 3), abs=1e-6)
    # with no censoring the coefficients equal OLS on a larger sample
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = 0.5 + 0.05 * X[:, 1] + rng.normal(scale=0.02, size=200)
    assert y.min() > 0 and y.max() < 1
    np.testing.assert_allclose(m.fit_tobit(X, y).coefficients, m.fit_ols(X, y).coefficients, atol=1e-6)
    # with censoring: the likelihood optimum, reached from another start
    ystar = 0.45 + 0.3 * X[:, 1] + rng.normal(scale=0.25, size=200)
    yc = np.clip(ystar, 0.0, 1.0)
    assert (yc == 0).sum() > 5 and (yc == 1).sum() > 5
    ft = m.fit_tobit(X, yc)
    s = ft.extras["sigma"]
    assert s > 0
    best = optimize.minimize(lambda t: _tobit_negloglik(t[:2], np.exp(t[2]), X, yc), np.array([0.2, 0.1, -1.0]),
                             method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 10000})
    assert _tobit_negloglik(ft.coefficients, s, X, yc) <= best.fun + 1e-7
    np.testing.assert_allclose(ft.coefficients, best.x[:2], atol=2e-3)
    # the prediction is the expectation of the censored variable, inside [0, 1], and matches numerical integration
    pred = m.predict_lgd(ft, X)
    assert np.all((pred >= 0) & (pred <= 1))
    mu0 = float(X[0] @ ft.coefficients)
    num = stats.norm.expect(lambda z: min(1.0, max(0.0, mu0 + s * z)), lb=-12, ub=12)
    assert pred[0] == pytest.approx(num, abs=1e-6)
    with pytest.raises(ValueError, match="y"):
        m.fit_tobit(X, y + 1.0)


# ---------------------------------------------------------------- RV-610


def test_cure_decomposition():
    assert m.cure_decomposition(a(0.3), a(0.05), a(0.6))[0] == pytest.approx(0.435, abs=1e-15)
    out = m.cure_decomposition(a(0.0, 0.5, 1.0), a(0.1, 0.1, 0.1), a(0.9, 0.9, 0.9))
    np.testing.assert_allclose(out, [0.9, 0.5, 0.1], atol=1e-15)
    assert out[2] == 0.1  # p_cure = 1 returns lgd_cure exactly
    for bad in ((a(1.1), a(0.1), a(0.5)), (a(0.5), a(-0.1), a(0.5)), (a(0.5), a(0.1), a(1.5))):
        with pytest.raises(ValueError):
            m.cure_decomposition(*bad)
    with pytest.raises(ValueError, match="same"):
        m.cure_decomposition(a(0.5, 0.5), a(0.1), a(0.5))


def test_predict_lgd_range_and_shape():
    X, y = _beta_data(n=120)
    y = np.clip(y, 0, 1)
    fits = [m.fit_ols(X, y), m.fit_fractional_logit(X, y), m.fit_beta(X, y, 0.01), m.fit_hurdle(X, y),
            m.fit_tobit(X, y)]
    assert [f.kind for f in fits] == ["ols", "fractional-logit", "beta", "hurdle", "tobit"]
    far = A([[1, 50], [1, -50], [1, 0]])
    for f in fits:
        out = m.predict_lgd(f, far)
        assert out.shape == (3,) and np.all((out >= 0) & (out <= 1)), f.kind
        assert not np.any(np.isnan(out))
        with pytest.raises(ValueError, match="columns"):
            m.predict_lgd(f, A([[1, 2, 3]]))
    with pytest.raises(ValueError, match="non-finite"):
        m.predict_lgd(fits[1], A([[1, np.inf]]))
