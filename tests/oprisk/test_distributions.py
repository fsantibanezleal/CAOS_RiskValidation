"""RV-1201 to RV-1208: the frequency and severity laws, their truncation, the fits, the splice and the tail profile.

The goldens are the contract's (`engine-oprisk-api`, module distributions); every other reference is recomputed
here from SciPy or from the definition, never copied from the code under test.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.oprisk import distributions as d

E = math.e


def arr(*x):
    return np.array(x, dtype=float)


# ----------------------------------------------------------------------------------------------------------------
# frequency laws


def test_frequency_pmf_goldens_and_recursion():
    assert d.frequency_pmf("poisson", (2.0,), 2) == pytest.approx([0.135335, 0.270671, 0.270671], abs=5e-7)
    assert d.frequency_pmf("negative-binomial", (2, 0.5), 2) == pytest.approx([0.25, 0.25, 0.1875], abs=1e-15)
    assert d.frequency_pmf("binomial", (2, 0.5), 2) == pytest.approx([0.25, 0.5, 0.25], abs=1e-15)
    # the (a, b, 0) recursion p_n = (a + b / n) p_{n-1} with the coefficients of Shevchenko Table 1
    cases = {"poisson": ((7.5,), (0.0, 7.5)), "negative-binomial": ((3.0, 0.4), (0.6, 0.6 * 2.0)),
             "binomial": ((5, 0.3), (-0.3 / 0.7, 0.3 * 6 / 0.7))}
    for law, (params, (a, b)) in cases.items():
        p = d.frequency_pmf(law, params, 12)
        n = np.arange(1, 13)
        assert p[1:] == pytest.approx((a + b / n) * p[:-1], rel=1e-12, abs=1e-15), law
        assert np.all(p >= 0.0) and p.sum() <= 1.0 + 1e-15
    # the three laws against SciPy
    assert d.frequency_pmf("poisson", (3.2,), 9) == pytest.approx(stats.poisson.pmf(np.arange(10), 3.2), rel=1e-13)
    assert d.frequency_pmf("negative-binomial", (2.5, 0.3), 9) == pytest.approx(
        stats.nbinom.pmf(np.arange(10), 2.5, 0.3), rel=1e-13)


def test_frequency_pmf_refusals_and_tail_not_added():
    for law, params in (("poisson", (0.0,)), ("poisson", (-1.0,)), ("negative-binomial", (0.0, 0.5)),
                        ("negative-binomial", (2.0, 0.0)), ("negative-binomial", (2.0, 1.0)), ("binomial", (2.5, 0.5)),
                        ("binomial", (0, 0.5)), ("binomial", (3, 1.0)), ("geometric", (0.5,))):
        with pytest.raises(ValueError):
            d.frequency_pmf(law, params, 5)
    with pytest.raises(ValueError):
        d.frequency_pmf("poisson", (2.0,), -1)
    # the mass beyond n_max is not added
    assert d.frequency_pmf("poisson", (50.0,), 5).sum() < 1e-10


# ----------------------------------------------------------------------------------------------------------------
# severity cdf, ppf, moments


def test_severity_cdf_goldens():
    assert d.severity_cdf("lognormal", (0.0, 2.0), arr(1.0))[0] == pytest.approx(0.5, abs=1e-15)
    assert d.severity_cdf("gpd", (0.5, 100.0, 0.0), arr(100.0))[0] == pytest.approx(1 - 1.5 ** -2, abs=1e-15)
    assert d.severity_cdf("gpd", (0.5, 100.0, 0.0), arr(100.0))[0] == pytest.approx(0.555556, abs=5e-7)
    assert d.severity_cdf("weibull", (1.0, 1.0), arr(1.0))[0] == pytest.approx(0.632121, abs=5e-7)
    # against SciPy
    x = np.linspace(0.01, 50.0, 40)
    assert d.severity_cdf("lognormal", (0.3, 0.8), x) == pytest.approx(stats.lognorm.cdf(x, 0.8, scale=math.exp(0.3)),
                                                                      rel=1e-12)
    assert d.severity_cdf("weibull", (1.7, 4.0), x) == pytest.approx(stats.weibull_min.cdf(x, 1.7, scale=4.0),
                                                                      rel=1e-12)
    assert d.severity_cdf("gpd", (0.4, 3.0, 2.0), x) == pytest.approx(stats.genpareto.cdf(x, 0.4, loc=2.0, scale=3.0),
                                                                      rel=1e-12)


def test_severity_cdf_origin_monotone_and_limits():
    laws = (("lognormal", (0.0, 1.0), 0.0), ("weibull", (2.0, 3.0), 0.0), ("gpd", (0.5, 10.0, 7.0), 7.0))
    for law, params, origin in laws:
        x = np.concatenate([[-5.0, 0.0, origin, origin - 1e-9], np.linspace(origin, origin + 1e6, 200)])
        f = d.severity_cdf(law, params, x)
        assert f[0] == 0.0 and f[1] == 0.0 and f[2] == 0.0 and f[3] == 0.0, law
        assert np.all(np.diff(f[4:]) >= 0.0) and f[-1] > 0.99, law
        assert d.severity_cdf(law, params, arr(1e300))[0] == pytest.approx(1.0)


def test_splice_cdf_is_continuous_and_has_weight_w_at_u():
    params = (1.0, 0.7, 5.0, 0.4, 3.0, 0.9)  # mu, sigma, u, xi, beta, w
    at_u = d.severity_cdf("lognormal-gpd-splice", params, arr(5.0))[0]
    assert at_u == pytest.approx(0.9, abs=1e-15)
    eps = 1e-9
    around = d.severity_cdf("lognormal-gpd-splice", params, arr(5.0 - eps, 5.0 + eps))
    assert around == pytest.approx([0.9, 0.9], abs=1e-8)
    x = np.linspace(0.01, 200.0, 300)
    f = d.severity_cdf("lognormal-gpd-splice", params, x)
    assert np.all(np.diff(f) >= 0.0) and f[-1] < 1.0
    # above u it is w + (1 - w) G(x - u)
    xa = arr(8.0, 40.0)
    assert d.severity_cdf("lognormal-gpd-splice", params, xa) == pytest.approx(
        0.9 + 0.1 * stats.genpareto.cdf(xa - 5.0, 0.4, scale=3.0), rel=1e-12)
    # below u it is w F_b(x) / F_b(u)
    xb = arr(1.0, 3.0)
    fb = stats.lognorm(0.7, scale=math.exp(1.0))
    assert d.severity_cdf("lognormal-gpd-splice", params, xb) == pytest.approx(0.9 * fb.cdf(xb) / fb.cdf(5.0),
                                                                              rel=1e-12)


def test_severity_cdf_refusals():
    x = arr(1.0)
    for law, params in (("lognormal", (0.0, 0.0)), ("lognormal", (0.0, -1.0)), ("weibull", (0.0, 1.0)),
                        ("weibull", (1.0, -1.0)), ("gpd", (0.0, 1.0, 0.0)), ("gpd", (-0.2, 1.0, 0.0)),
                        ("gpd", (0.5, 0.0, 0.0)), ("lognormal-gpd-splice", (0.0, 1.0, 5.0, 0.4, 3.0, 1.0)),
                        ("lognormal-gpd-splice", (0.0, 1.0, 5.0, 0.4, 3.0, 0.0)), ("pareto", (1.0,)),
                        ("lognormal", (0.0,))):
        with pytest.raises(ValueError):
            d.severity_cdf(law, params, x)


def test_severity_ppf_goldens_and_inversion():
    assert d.severity_ppf("gpd", (0.5, 100.0, 0.0), arr(0.99))[0] == pytest.approx(1800.0, rel=1e-12)
    assert d.severity_ppf("lognormal", (0.0, 2.0), arr(0.5))[0] == pytest.approx(1.0, abs=1e-12)
    q = arr(0.001, 0.1, 0.5, 0.9, 0.999, 0.999999)
    for law, params in (("lognormal", (0.5, 1.2)), ("weibull", (0.8, 5.0)), ("gpd", (0.7, 20.0, 5.0)),
                        ("lognormal-gpd-splice", (1.0, 0.7, 5.0, 0.4, 3.0, 0.9))):
        x = d.severity_ppf(law, params, q)
        assert d.severity_cdf(law, params, x) == pytest.approx(q, abs=1e-12), law
    # the GPD closed form and the ends
    q2 = arr(0.2, 0.7)
    closed = 4.0 + 10.0 / 0.3 * ((1 - q2) ** -0.3 - 1)
    assert d.severity_ppf("gpd", (0.3, 10.0, 4.0), q2) == pytest.approx(closed, rel=1e-13)
    ends = d.severity_ppf("gpd", (0.3, 10.0, 4.0), arr(0.0, 1.0))
    assert ends[0] == 4.0 and math.isinf(ends[1])
    ends = d.severity_ppf("lognormal", (0.0, 1.0), arr(0.0, 1.0))
    assert ends[0] == 0.0 and math.isinf(ends[1])
    with pytest.raises(ValueError):
        d.severity_ppf("lognormal", (0.0, 1.0), arr(1.5))
    with pytest.raises(ValueError):
        d.severity_ppf("lognormal", (0.0, 1.0), arr(-0.1))


def test_severity_moment_goldens_and_infinite_mean():
    assert d.severity_moment("lognormal", (0.0, 2.0), 1) == pytest.approx(7.389056, abs=5e-7)
    assert d.severity_moment("gpd", (0.25, 100.0, 0.0), 1) == pytest.approx(133.333333, abs=5e-7)
    assert d.severity_moment("gpd", (0.25, 100.0, 0.0), 2) == pytest.approx(53333.333333, abs=5e-7)
    assert d.severity_moment("weibull", (2.0, 1.0), 1) == pytest.approx(0.886227, abs=5e-7)
    # a tail index of 1 or more has no mean; k xi >= 1 has no k-th moment
    assert d.severity_moment("gpd", (1.0, 100.0, 0.0), 1) == math.inf
    assert d.severity_moment("gpd", (1.1, 100.0, 0.0), 1) == math.inf
    assert d.severity_moment("gpd", (0.5, 100.0, 0.0), 2) == math.inf
    assert d.severity_moment("gpd", (0.5, 100.0, 0.0), 1) == pytest.approx(200.0, rel=1e-13)
    # the location enters through the binomial expansion
    assert d.severity_moment("gpd", (0.25, 100.0, 50.0), 2) == pytest.approx(
        50.0 ** 2 + 2 * 50.0 * 133.333333333 + 53333.3333333, rel=1e-10)
    # raw moments of the lognormal and the Weibull against SciPy
    assert d.severity_moment("lognormal", (0.3, 0.5), 3) == pytest.approx(
        stats.lognorm(0.5, scale=math.exp(0.3)).moment(3), rel=1e-12)
    assert d.severity_moment("weibull", (1.7, 4.0), 2) == pytest.approx(stats.weibull_min(1.7, scale=4.0).moment(2),
                                                                         rel=1e-12)
    for law, params in (("lognormal", (0.0, 1.0)), ("weibull", (1.0, 1.0))):
        with pytest.raises(ValueError):
            d.severity_moment(law, params, 0)


def test_splice_moment_sums_body_and_tail_and_is_infinite_with_the_tail():
    # compare with numerical integration of the body and the closed form of the tail
    params = (1.0, 0.7, 5.0, 0.2, 3.0, 0.9)
    m1 = d.severity_moment("lognormal-gpd-splice", params, 1)
    xs = np.linspace(1e-9, 5.0, 2_000_001)
    fb = stats.lognorm(0.7, scale=math.exp(1.0))
    dens = 0.9 * fb.pdf(xs) / fb.cdf(5.0)
    body = float(np.trapezoid(xs * dens, xs))
    tail = 0.1 * (5.0 + 3.0 / (1 - 0.2))
    assert m1 == pytest.approx(body + tail, rel=1e-5)
    assert d.severity_moment("lognormal-gpd-splice", (1.0, 0.7, 5.0, 1.2, 3.0, 0.9), 1) == math.inf


# ----------------------------------------------------------------------------------------------------------------
# truncation


def test_truncated_cdf_golden_and_threshold_stability():
    gpd = (0.5, 100.0, 0.0)
    got = d.truncated_severity_cdf("gpd", gpd, arr(250.0), 100.0)[0]
    assert got == pytest.approx(1 - 1.5 ** -2, abs=1e-12)
    f = lambda x: d.severity_cdf("gpd", gpd, arr(x))[0]  # noqa: E731
    assert got == pytest.approx((f(250.0) - f(100.0)) / (1 - f(100.0)), abs=1e-12)
    assert f(250.0) == pytest.approx(0.802469, abs=5e-7)
    # beta' = beta + xi (H - location), location H
    x = arr(150.0, 400.0, 5000.0)
    shifted = d.severity_cdf("gpd", (0.5, 150.0, 100.0), x)
    assert d.truncated_severity_cdf("gpd", gpd, x, 100.0) == pytest.approx(shifted, rel=1e-12)
    # zero below H, zero at H, nondecreasing, tends to 1
    y = np.concatenate([[0.0, 50.0, 100.0], np.linspace(100.0, 1e7, 300)])
    t = d.truncated_severity_cdf("gpd", gpd, y, 100.0)
    assert t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0 and np.all(np.diff(t[2:]) >= 0.0) and t[-1] > 0.99
    # the lognormal at a threshold of zero is the plain cdf
    z = np.linspace(0.1, 30.0, 20)
    assert d.truncated_severity_cdf("lognormal", (0.5, 1.0), z, 0.0) == pytest.approx(
        d.severity_cdf("lognormal", (0.5, 1.0), z), rel=1e-13)


def test_truncated_cdf_refuses_a_threshold_with_no_mass_above():
    with pytest.raises(ValueError):
        d.truncated_severity_cdf("weibull", (1.0, 1.0), arr(1.0), 1e6)


def test_rescale_frequency_golden_and_invariants():
    assert d.rescale_frequency(10.0, "gpd", (0.5, 100.0, 0.0), 100.0) == pytest.approx(22.5, rel=1e-12)
    # equal to the observed rate when F(H) = 0 (here H below the origin), at least the observed rate otherwise
    assert d.rescale_frequency(7.0, "gpd", (0.5, 100.0, 50.0), 20.0) == 7.0
    assert d.rescale_frequency(7.0, "lognormal", (0.0, 1.0), 2.0) > 7.0
    with pytest.raises(ValueError):
        d.rescale_frequency(-1.0, "lognormal", (0.0, 1.0), 1.0)
    with pytest.raises(ValueError):
        d.rescale_frequency(1.0, "weibull", (1.0, 1.0), 1e6)


# ----------------------------------------------------------------------------------------------------------------
# frequency fits


def test_fit_frequency_golden_and_negative_binomial():
    fit = d.fit_frequency(np.array([3, 5, 4]))
    assert fit.law == "poisson" and fit.params == (4.0,) and fit.n_years == 3
    assert fit.std_errors[0] == pytest.approx(1.154701, abs=5e-7)
    assert fit.loglik == pytest.approx(float(stats.poisson.logpmf([3, 5, 4], 4.0).sum()), rel=1e-12)
    # an overdispersed sample: the negative binomial maximum likelihood estimate
    rng = np.random.default_rng(11)
    counts = rng.negative_binomial(4.0, 0.25, size=4000)
    nb = d.fit_frequency(counts, "negative-binomial")
    assert nb.law == "negative-binomial" and len(nb.params) == 2 and len(nb.std_errors) == 2
    r, q = nb.params
    assert r == pytest.approx(4.0, rel=0.15) and q == pytest.approx(0.25, rel=0.1)
    assert all(s >= 0.0 for s in nb.std_errors)
    assert nb.loglik == pytest.approx(float(stats.nbinom.logpmf(counts, r, q).sum()), rel=1e-9)
    # it is the optimum: perturbing either parameter lowers the log-likelihood
    for dr, dq in ((0.05, 0.0), (-0.05, 0.0), (0.0, 0.005), (0.0, -0.005)):
        assert stats.nbinom.logpmf(counts, r + dr, q + dq).sum() <= nb.loglik + 1e-8
    assert d.fit_frequency(counts, "negative-binomial") == nb  # deterministic


def test_fit_frequency_refusals():
    with pytest.raises(ValueError, match="underdispersed"):
        d.fit_frequency(np.array([4, 4, 4, 4]), "negative-binomial")
    with pytest.raises(ValueError, match="underdispersed"):
        d.fit_frequency(np.array([3, 5, 4]), "negative-binomial")  # variance 1 < mean 4
    for bad in (np.array([3]), np.array([], dtype=int), np.array([3, -1, 4]), np.array([0, 0, 0]),
                np.array([1.5, 2.0, 3.0])):
        with pytest.raises(ValueError):
            d.fit_frequency(bad)
    with pytest.raises(ValueError):
        d.fit_frequency(np.array([3, 5, 4]), "binomial")
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.fit_frequency(np.array([3, 5, 4])).law = "x"  # type: ignore[misc]


# ----------------------------------------------------------------------------------------------------------------
# severity fits


def test_fit_severity_untruncated_lognormal_closed_form_golden():
    fit = d.fit_severity("lognormal", arr(1.0, E, E ** 2), 0.0, truncated=False)
    assert fit.params[0] == pytest.approx(1.0, abs=1e-12) and fit.params[1] == pytest.approx(0.816497, abs=5e-7)
    assert fit.std_errors[0] == pytest.approx(0.471405, abs=5e-7)
    assert fit.law == "lognormal" and fit.n == 3 and fit.truncated is False and fit.converged is True
    lx = np.log(arr(1.0, E, E ** 2))
    dist = stats.lognorm(fit.params[1], scale=math.exp(fit.params[0]))
    assert fit.loglik == pytest.approx(float(dist.logpdf(np.exp(lx)).sum()), rel=1e-12)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.n = 1  # type: ignore[misc]


def _lognormal_truncated_sample(n, mu, sigma, h, seed):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(mu, sigma, size=int(n * 20))
    return x[x >= h][:n]


def test_fit_severity_truncated_lognormal_recovers_the_parameters_where_the_naive_fit_is_biased():
    mu, sigma, h = 9.0, 1.5, 12_000.0
    x = _lognormal_truncated_sample(4000, mu, sigma, h, 3)
    naive = d.fit_severity("lognormal", x, h, truncated=False)
    trunc = d.fit_severity("lognormal", x, h, truncated=True)
    assert trunc.truncated and trunc.threshold == h and trunc.converged
    assert trunc.params[0] == pytest.approx(mu, abs=0.35) and trunc.params[1] == pytest.approx(sigma, abs=0.2)
    assert abs(naive.params[0] - mu) > 3 * abs(trunc.params[0] - mu)  # the biased baseline (rung 1)
    # the log-likelihood is sum ln f - N ln (1 - F(H)) and the fit is its maximiser
    def ll(p):
        dist = stats.lognorm(p[1], scale=math.exp(p[0]))
        return float(dist.logpdf(x).sum() - x.size * dist.logsf(h))
    assert trunc.loglik == pytest.approx(ll(trunc.params), rel=1e-9)
    for dm, ds in ((0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)):
        assert ll((trunc.params[0] + dm, trunc.params[1] + ds)) <= trunc.loglik + 1e-7
    assert all(s > 0 and math.isfinite(s) for s in trunc.std_errors)
    assert d.fit_severity("lognormal", x, h) == trunc  # deterministic


def test_fit_severity_truncated_gpd_recovers_the_excess_parameters():
    rng = np.random.default_rng(5)
    h = 20_000.0
    xi, beta = 0.6, 25_000.0
    y = stats.genpareto.rvs(xi, scale=beta, size=5000, random_state=rng)
    x = h + y
    fit = d.fit_severity("gpd", x, h, truncated=True)
    assert fit.law == "gpd" and fit.params[2] == h and fit.converged
    assert fit.params[0] == pytest.approx(xi, abs=0.06) and fit.params[1] == pytest.approx(beta, rel=0.1)
    # it is the maximum of the GPD likelihood of the excesses
    def ll(p):
        return float(stats.genpareto.logpdf(y, p[0], scale=p[1]).sum())
    assert fit.loglik == pytest.approx(ll(fit.params), rel=1e-9)
    for dx, db in ((0.01, 0), (-0.01, 0), (0, 500.0), (0, -500.0)):
        assert ll((fit.params[0] + dx, fit.params[1] + db)) <= fit.loglik + 1e-6
    assert fit.std_errors[0] > 0 and fit.std_errors[1] > 0 and math.isnan(fit.std_errors[2]) is False
    # the threshold stability: a fit of the excesses over a higher threshold has beta' = beta + xi (u - h)
    u = h + 30_000.0
    hi = d.fit_severity("gpd", x[x >= u], u, truncated=True)
    assert hi.params[0] == pytest.approx(xi, abs=0.15)
    assert hi.params[1] == pytest.approx(beta + xi * (u - h), rel=0.25)


def test_fit_severity_weibull_truncated_and_untruncated():
    rng = np.random.default_rng(9)
    h = 2.0
    x = 5.0 * rng.weibull(1.4, size=40_000)
    x = x[x >= h][:3000]
    t = d.fit_severity("weibull", x, h, truncated=True)
    u = d.fit_severity("weibull", x, h, truncated=False)
    assert t.params[0] == pytest.approx(1.4, abs=0.12) and t.params[1] == pytest.approx(5.0, rel=0.1)
    assert abs(u.params[0] - 1.4) > abs(t.params[0] - 1.4)
    dist = stats.weibull_min(t.params[0], scale=t.params[1])
    assert t.loglik == pytest.approx(float(dist.logpdf(x).sum() - x.size * dist.logsf(h)), rel=1e-9)


def test_fit_severity_refuses_below_threshold_and_bad_inputs():
    with pytest.raises(ValueError):
        d.fit_severity("lognormal", arr(5.0, 15.0, 30.0), 10.0)
    with pytest.raises(ValueError):
        d.fit_severity("lognormal", arr(), 0.0)
    with pytest.raises(ValueError):
        d.fit_severity("lognormal", arr(1.0, np.nan), 0.0)
    with pytest.raises(ValueError):
        d.fit_severity("lognormal", arr(1.0), 0.0)  # one loss: no spread
    with pytest.raises(ValueError):
        d.fit_severity("pareto", arr(1.0, 2.0, 3.0), 0.0)
    with pytest.raises(ValueError):
        d.fit_severity("lognormal", arr(1.0, 2.0, 3.0), -1.0)


def test_fit_severity_boundary_gives_nan_standard_errors_and_not_converged():
    # exponential-like excesses push the GPD tail index to its boundary xi -> 0: never a silent number
    rng = np.random.default_rng(1)
    h = 10.0
    x = h + rng.exponential(5.0, size=3000)
    fit = d.fit_severity("gpd", x, h)
    if not fit.converged:
        assert all(math.isnan(s) for s in fit.std_errors[:2])
    else:
        assert fit.params[0] < 0.1  # a converged fit sits next to the exponential
    # a constant sample has a singular information matrix
    flat = d.fit_severity("gpd", np.full(50, 20.0) + np.arange(50) * 0.0 + 1.0, 20.0)
    assert flat.converged is False and all(math.isnan(s) for s in flat.std_errors[:2])


# ----------------------------------------------------------------------------------------------------------------
# splice


def test_fit_splice_weight_golden_and_recovery():
    # t = 0.2 and c = 0.5 give w = 0.8 / 0.9 (checked on the closed form of the contract and the inversion)
    w = (1 - 0.2) / (1 - 0.2 * 0.5)
    assert w == pytest.approx(0.888889, abs=5e-7)
    assert (1 - w) / (1 - w * 0.5) == pytest.approx(0.2, abs=1e-15)

    rng = np.random.default_rng(21)
    h, u = 1_000.0, 20_000.0
    mu, sigma, xi, beta, w = 8.0, 1.2, 0.5, 15_000.0, 0.85
    n = 40_000
    fb = stats.lognorm(sigma, scale=math.exp(mu))
    uni = rng.random(n)
    body = fb.ppf(uni / w * fb.cdf(u))          # the body law below u (the lognormal rescaled to [0, u])
    tail = u + stats.genpareto.ppf((uni - w) / (1 - w), xi, scale=beta)
    x = np.where(uni <= w, body, tail)
    x = x[x >= h]                                # the bank records from H up
    fit = d.fit_splice(x, h, u, min_tail=10)
    assert fit.params[2] == u == fit.splice_threshold and fit.params[5] == fit.weight
    assert fit.collection_threshold == h and fit.n_body + fit.n_tail == x.size
    assert fit.n_tail == int((x > u).sum()) and fit.converged
    # w follows from the observed tail share and the body's c
    t = fit.n_tail / x.size
    body_dist = stats.lognorm(fit.params[1], scale=math.exp(fit.params[0]))
    c = body_dist.cdf(h) / body_dist.cdf(u)
    assert fit.weight == pytest.approx((1 - t) / (1 - t * c), rel=1e-12)
    assert (1 - fit.weight) / (1 - fit.weight * c) == pytest.approx(t, rel=1e-12)
    assert fit.params[3] == pytest.approx(xi, abs=0.15) and fit.params[4] > 0
    assert len(fit.std_errors) == 6
    assert d.fit_splice(x, h, u, min_tail=10) == fit  # deterministic


def test_fit_splice_refusals():
    x = np.linspace(1_000.0, 50_000.0, 200)
    with pytest.raises(ValueError):
        d.fit_splice(x, 1_000.0, 49_000.0, min_tail=10)        # fewer than min_tail above u
    with pytest.raises(ValueError):
        d.fit_splice(x, 1_000.0, 500.0)                         # u not above H
    with pytest.raises(ValueError):
        d.fit_splice(x[x > 40_000.0], 1_000.0, 1_001.0, min_tail=5)  # no loss at or below u
    with pytest.raises(ValueError):
        d.fit_splice(arr(500.0, 2_000.0, 3_000.0), 1_000.0, 2_500.0)  # below the collection threshold


# ----------------------------------------------------------------------------------------------------------------
# tail index profile


def test_tail_index_profile_on_exact_gpd_data_is_stable_in_the_modified_scale():
    rng = np.random.default_rng(8)
    xi, beta = 0.5, 100.0
    x = stats.genpareto.rvs(xi, scale=beta, size=60_000, random_state=rng)
    th = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
    p = d.tail_index_profile(x, th, min_exceedances=50)
    assert p.thresholds.shape == p.xi.shape == p.xi_se.shape == p.beta.shape == p.n_exceedances.shape == (5,)
    assert np.all(np.diff(p.n_exceedances) <= 0) and p.n_exceedances.dtype.kind == "i"
    assert p.n_exceedances[0] == int((x > 0).sum())
    assert p.converged.all()
    assert p.xi == pytest.approx(xi, abs=0.06)
    modified = p.beta - p.xi * p.thresholds
    assert modified == pytest.approx(beta, rel=0.12)
    assert np.all(p.xi_se > 0)
    q = d.tail_index_profile(x, th, min_exceedances=50)
    assert np.array_equal(p.xi, q.xi) and np.array_equal(p.beta, q.beta)


def test_tail_index_profile_lists_a_thin_threshold_with_nan():
    x = np.linspace(1.0, 100.0, 60) ** 1.5
    th = np.array([1.0, 50.0, 500.0, 5000.0])
    p = d.tail_index_profile(x, th, min_exceedances=10)
    thin = p.n_exceedances < 10
    assert thin.any() and not thin.all()
    assert np.all(np.isnan(p.xi[thin])) and np.all(np.isnan(p.xi_se[thin])) and np.all(np.isnan(p.beta[thin]))
    assert not p.converged[thin].any()
    assert p.thresholds.size == 4  # listed, never dropped
    for bad in (np.array([2.0, 1.0]), np.array([1.0, 1.0]), np.array([])):
        with pytest.raises(ValueError):
            d.tail_index_profile(x, bad)
