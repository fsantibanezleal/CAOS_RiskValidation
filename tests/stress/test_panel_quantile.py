"""RV-1915 to RV-1919: the panel quantile satellite (Canay's two-step and Koenker's penalized estimators, the bootstrap
intervals and the projection of a quantile grid), in `stress.panel_quantile` (contract `engine-stress-api`)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import stats

from riskvalidation.stress import panel_quantile as pq
from riskvalidation.stress import scenario_models as sm


def _panel(n=30, t=12, seed=1, noise="normal"):
    """A panel with a constant, one covariate and individual effects correlated with the covariate."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(size=(n, t))
    eta = rng.normal(size=n)
    alpha = 2.0 * (x1.sum(axis=1) + eta)
    alpha -= alpha.mean()
    eps = rng.normal(2.0, 1.0, size=(n, t)) if noise == "normal" else rng.standard_t(5, size=(n, t))
    y = (eps - 1.0) + eps * x1 + alpha[:, None]
    ent = np.repeat(np.arange(n), t)
    x = np.column_stack([np.ones(n * t), x1.ravel()])
    return y.ravel(), x, ent


def test_fit_record_and_regime():
    y, x, ent = _panel(n=30, t=12)
    taus = [0.75, 0.25, 0.5]
    fit = pq.fit_panel_quantile(y, x, ent, taus)
    assert isinstance(fit, pq.PanelQuantileFit) and fit.method == "two_step"
    assert fit.taus.tolist() == [0.25, 0.5, 0.75]                      # sorted ascending
    assert fit.coef.shape == (3, 2) and fit.alpha.shape == (30,)
    assert fit.n == 30 and fit.T == 12 and fit.n_obs == 360 and fit.converged
    assert fit.n_over_t == pytest.approx(30 / 12) and fit.n2_over_t == pytest.approx(900 / 12)
    assert fit.wide_panel is True                                      # n / T >= 1
    assert pq.fit_panel_quantile(*_panel(n=10, t=20), [0.5]).wide_panel is False
    assert pq.fit_panel_quantile(*_panel(n=20, t=20), [0.5]).wide_panel is True   # n / T = 1 is wide
    assert abs(float(np.mean(fit.alpha))) < 1e-12
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.method = "penalized"
    # step 1 is the within estimator of the slope, the constant column has no within estimate
    xd = x[:, 1] - np.repeat(x[:, 1].reshape(30, 12).mean(axis=1), 12)
    yd = y - np.repeat(y.reshape(30, 12).mean(axis=1), 12)
    assert fit.gamma[1] == pytest.approx((xd @ yd) / (xd @ xd), rel=1e-10) and np.isnan(fit.gamma[0])
    # step 2 is the pooled quantile regression of Y - alpha_hat on X
    corrected = y - np.repeat(fit.alpha, 12)
    for k, tau in enumerate(fit.taus):
        beta, _ = pq._pooled_quantile(corrected, x, float(tau))
        assert fit.coef[k] == pytest.approx(beta, abs=1e-9)
    # the penalized method: the record carries the settings
    pen = pq.fit_panel_quantile(y, x, ent, [0.25, 0.75], method="penalized", lam=1.0)
    assert pen.method == "penalized" and pen.lam == 1.0 and pen.gamma is None and pen.converged
    assert pen.weights.tolist() == [0.5, 0.5] and pen.coef.shape == (2, 2)
    assert np.all(np.diff(pen.coef[:, 1]) > -0.5)
    # the input is not mutated, string entity labels work, refusals name the field
    y0, x0 = y.copy(), x.copy()
    labels = np.array([f"bank{e:02d}" for e in ent])
    again = pq.fit_panel_quantile(y, x, labels, taus)
    assert np.array_equal(y0, y) and np.array_equal(x0, x) and again.coef == pytest.approx(fit.coef)
    with pytest.raises(ValueError, match="taus"):
        pq.fit_panel_quantile(y, x, ent, [0.0, 0.5])
    with pytest.raises(ValueError, match="taus"):
        pq.fit_panel_quantile(y, x, ent, [0.5, 0.5])
    with pytest.raises(ValueError, match="method"):
        pq.fit_panel_quantile(y, x, ent, [0.5], method="nope")
    with pytest.raises(ValueError, match="lam"):
        pq.fit_panel_quantile(y, x, ent, [0.5], method="penalized", lam=-1.0)
    with pytest.raises(ValueError, match="agree"):
        pq.fit_panel_quantile(y[:-1], x, ent, [0.5])
    with pytest.raises(ValueError, match="non-finite"):
        bad = y.copy()
        bad[3] = np.nan
        pq.fit_panel_quantile(bad, x, ent, [0.5])
    with pytest.raises(ValueError, match="at least 2 periods"):
        pq.fit_panel_quantile(y[:-11], x[:-11], ent[:-11], [0.5])
    with pytest.raises(ValueError, match="full column rank"):
        pq.fit_panel_quantile(y, np.column_stack([x, x[:, 1]]), ent, [0.5])


def _canay_model_1(n, t, rng):
    """Canay's (2011) section 5 design, Model 1: Y = (eps - 1) + eps X + alpha, alpha = gamma (sum_t X + eta) -
    E(alpha), X ~ Beta(1, 1), eta ~ N(0, 1), gamma = 2, eps ~ N(2, 1): the slope at tau is Phi^-1(tau) + 2."""
    x1 = rng.uniform(size=(n, t))
    eta = rng.normal(size=n)
    alpha = 2.0 * (x1.sum(axis=1) + eta) - 2.0 * (0.5 * t)
    eps = rng.normal(2.0, 1.0, size=(n, t))
    y = (eps - 1.0) + eps * x1 + alpha[:, None]
    return y.ravel(), np.column_stack([np.ones(n * t), x1.ravel()]), np.repeat(np.arange(n), t)


@pytest.mark.sizepower
@pytest.mark.parametrize(
    ("t", "bias_pub", "mse_pub", "chen_huo"),
    [(20, 0.0377, 0.0264, 0.036), (10, 0.0793, 0.0605, 0.077), (5, 0.1494, 0.1473, None)],
)
def test_canay_table_1(t, bias_pub, mse_pub, chen_huo):
    n, tau, reps = 100, 0.25, 1000
    truth = float(stats.norm.ppf(tau)) + 2.0
    assert truth == pytest.approx(1.3255, abs=1e-4)
    rng = np.random.default_rng(20261115 + t)
    est = np.empty(reps)
    for r in range(reps):
        y, x, ent = _canay_model_1(n, t, rng)
        est[r] = pq.fit_panel_quantile(y, x, ent, [tau]).coef[0, 1]
    rel_bias = float(np.mean(est) - truth) / truth
    mse = float(np.mean((est - truth) ** 2))
    assert abs(rel_bias - bias_pub) < 0.02, (rel_bias, bias_pub)
    assert abs(mse - mse_pub) <= 0.25 * mse_pub, (mse, mse_pub)
    if chen_huo is not None:
        assert abs(rel_bias - chen_huo) < 0.02, (rel_bias, chen_huo)


def test_limits_and_invariance():
    qr = pytest.importorskip("statsmodels.regression.quantile_regression")
    y, x, ent = _panel(n=12, t=15, seed=4)
    n, t = 12, 15
    tau = 0.3
    # lam = 0: the fixed-effects estimator, quantile regression on the covariates and one dummy per entity (no constant)
    x_slope = x[:, [1]]
    fe = pq.fit_panel_quantile(y, x_slope, ent, [tau], method="penalized", lam=0.0)
    dummies = np.kron(np.eye(n), np.ones((t, 1)))
    ref = qr.QuantReg(y, np.column_stack([x_slope, dummies])).fit(q=tau, max_iter=5000, p_tol=1e-12).params
    assert fe.coef[0, 0] == pytest.approx(ref[0], abs=1e-6)
    assert fe.alpha == pytest.approx(ref[1:], abs=1e-5)
    # a very large lam: pooled quantile regression with zero effects
    pooled = pq.fit_panel_quantile(y, x, ent, [tau], method="penalized", lam=1e6)
    ref2 = qr.QuantReg(y, x).fit(q=tau, max_iter=5000, p_tol=1e-12).params
    assert np.all(pooled.alpha == 0.0)
    assert pooled.coef[0] == pytest.approx(ref2, abs=1e-6)
    # a single entity: the plain quantile regression of growth-at-risk
    rng = np.random.default_rng(2)
    xs = np.column_stack([np.ones(80), rng.normal(size=80)])
    ys = 1.0 + 0.5 * xs[:, 1] + rng.standard_t(4, size=80)
    one = pq.fit_panel_quantile(ys, xs, np.zeros(80, dtype=int), [0.1, 0.5, 0.9], method="penalized", lam=1.0)
    assert one.n == 1
    assert one.coef == pytest.approx(sm.gar_quantile_regressions(ys, xs, [0.1, 0.5, 0.9]), abs=1e-6)
    # adding a constant per entity changes the effects and not the slopes
    shift = np.repeat(np.linspace(-3.0, 5.0, n), t)
    for method, kw, design in (("two_step", {}, x), ("penalized", {"lam": 0.0}, x_slope)):
        a = pq.fit_panel_quantile(y, design, ent, [0.25, 0.75], method=method, **kw)
        b = pq.fit_panel_quantile(y + shift, design, ent, [0.25, 0.75], method=method, **kw)
        assert b.coef[:, -1] == pytest.approx(a.coef[:, -1], abs=1e-6)
        c_i = np.linspace(-3.0, 5.0, n)
        expected = c_i - (c_i.mean() if method == "two_step" else 0.0)
        assert b.alpha - a.alpha == pytest.approx(expected, abs=1e-6)
    # the median with symmetric errors: the two-step slope is near the within-OLS slope (Canay's gamma)
    big = _panel(n=60, t=20, seed=9)
    med = pq.fit_panel_quantile(*big, [0.5])
    assert med.coef[0, 1] == pytest.approx(med.gamma[1], abs=0.35)
    assert pq.fit_panel_quantile(*big, [0.5], method="penalized", lam=1.0).converged


def test_no_analytic_interval_and_bootstrap():
    y, x, ent = _panel(n=20, t=10, seed=5)
    fit = pq.fit_panel_quantile(y, x, ent, [0.25, 0.75])
    banned = ("se", "std", "stderr", "bse", "cov", "conf", "ci", "interval", "var", "lower", "upper", "p_value")
    for f in dataclasses.fields(fit):
        assert not any(f.name == b or f.name.startswith(b + "_") or f.name.endswith("_" + b) for b in banned), f.name
    assert not [a for a in dir(pq) if "standard_error" in a or "analytic" in a or "asymptotic" in a]
    doc = (pq.fit_panel_quantile.__doc__ or "") + (pq.__doc__ or "")
    for text in ("no analytic standard error", "Besstremyannaya", "Chen and Huo", "bootstrap"):
        assert text in doc
    # the entity bootstrap
    bs = pq.panel_quantile_bootstrap(y, x, ent, [0.25, 0.75], n_boot=40, seed=3)
    assert bs.coef_draws.shape == (40, 2, 2) and bs.lower.shape == (2, 2) and bs.alpha_draws is None
    assert np.all(bs.lower <= bs.upper) and np.all(bs.upper - bs.lower > 0.0)
    assert np.all((bs.lower <= fit.coef + 1.0) & (fit.coef - 1.0 <= bs.upper))
    again = pq.panel_quantile_bootstrap(y, x, ent, [0.25, 0.75], n_boot=40, seed=3)
    assert np.array_equal(bs.coef_draws, again.coef_draws)
    other_seed = pq.panel_quantile_bootstrap(y, x, ent, [0.25, 0.75], n_boot=40, seed=4)
    assert not np.array_equal(bs.coef_draws, other_seed.coef_draws)
    assert bs.heuristic is True and "heuristic" in bs.label and "conjecture" in bs.label     # n / T = 2: a wide panel
    narrow = pq.panel_quantile_bootstrap(*_panel(n=6, t=20, seed=6), [0.5], n_boot=10, seed=1)
    assert narrow.heuristic is False and "heuristic" not in narrow.label and "conjecture" in narrow.label
    # the stationary bootstrap over time keeps the entities and resamples the periods
    st = pq.panel_quantile_bootstrap(y, x, ent, [0.25, 0.75], scheme="stationary", n_boot=30, mean_block=4.0, seed=2)
    assert st.alpha_draws.shape == (30, 20) and st.coef_draws.shape == (30, 2, 2) and st.scheme == "stationary"
    assert abs(np.mean(st.coef_draws[:, 0, 1]) - fit.coef[0, 1]) < 1.0
    # the block lengths are geometric with mean 4: the share of continued indices is 1 - 1/4
    rng = np.random.default_rng(0)
    seq = pq._stationary_indices(rng, 100_000, 4.0)
    cont = np.mean(seq[1:] == (seq[:-1] + 1) % 100_000)
    assert cont == pytest.approx(0.75, abs=0.01)
    assert bool(np.all((seq >= 0) & (seq < 100_000)))
    with pytest.raises(ValueError, match="scheme"):
        pq.panel_quantile_bootstrap(y, x, ent, [0.5], scheme="nope")
    with pytest.raises(ValueError, match="n_boot"):
        pq.panel_quantile_bootstrap(y, x, ent, [0.5], n_boot=0)
    with pytest.raises(ValueError, match="mean_block"):
        pq.panel_quantile_bootstrap(y, x, ent, [0.5], mean_block=0.5)
    with pytest.raises(ValueError, match="balanced"):
        pq.panel_quantile_bootstrap(y[:-2], x[:-2], ent[:-2], [0.5], scheme="stationary", n_boot=3)


def test_projection_rearranges_and_refuses():
    y, x, ent = _panel(n=8, t=15, seed=7)
    fit = pq.fit_panel_quantile(y, x, ent, [0.1, 0.5, 0.9])
    future = np.column_stack([np.ones(4), np.linspace(0.0, 1.0, 4)])
    out = pq.project_panel_quantiles(fit, future)
    assert out.quantiles.shape == (8, 4, 3) and out.taus.tolist() == [0.1, 0.5, 0.9]
    assert np.all(np.diff(out.quantiles, axis=2) >= 0.0)
    direct = fit.alpha[:, None, None] + np.einsum("hp,qp->hq", future, fit.coef)[None]
    if out.crossings == 0:
        assert out.quantiles == pytest.approx(direct, abs=1e-12)
    # crossing quantiles are rearranged (a hand-made fit whose slopes cross inside the covariate range)
    crossing = dataclasses.replace(fit, coef=np.array([[0.0, 2.0], [0.5, 0.5], [1.0, -1.0]]))
    cx = pq.project_panel_quantiles(crossing, future)
    assert cx.crossings > 0
    assert np.all(np.diff(cx.quantiles, axis=2) >= 0.0)
    raw = crossing.alpha[:, None, None] + np.einsum("hp,qp->hq", future, crossing.coef)[None]
    assert np.sort(raw, axis=2) == pytest.approx(cx.quantiles, abs=1e-12)
    assert cx.quantiles.sum() == pytest.approx(raw.sum())            # a rearrangement: same values, other order
    # per-entity paths, and the limits only from the bootstrap
    per_entity = np.broadcast_to(future, (8, 4, 2)).copy()
    assert pq.project_panel_quantiles(fit, per_entity).quantiles == pytest.approx(out.quantiles)
    assert out.lower is None and out.upper is None and out.label is None
    bs = pq.panel_quantile_bootstrap(y, x, ent, [0.1, 0.5, 0.9], n_boot=25, seed=1)
    with_bs = pq.project_panel_quantiles(fit, future, bootstrap=bs)
    assert with_bs.lower.shape == (8, 4, 3) and np.all(with_bs.lower <= with_bs.upper) and with_bs.label == bs.label
    assert np.all(np.diff(with_bs.lower, axis=2) >= -1e-9)
    # the refusals
    with pytest.raises(ValueError, match="columns"):
        pq.project_panel_quantiles(fit, future[:, :1])
    with pytest.raises(ValueError, match="columns"):
        pq.project_panel_quantiles(fit, np.column_stack([future, future[:, 1]]))
    with pytest.raises(ValueError, match="one path per entity"):
        pq.project_panel_quantiles(fit, per_entity[:3])
    with pytest.raises(ValueError, match="non-finite"):
        pq.project_panel_quantiles(fit, np.array([[1.0, np.nan]]))
    other = pq.fit_panel_quantile(y, x, ent, [0.2, 0.8])
    with pytest.raises(ValueError, match="bootstrap"):
        pq.project_panel_quantiles(other, future, bootstrap=bs)
