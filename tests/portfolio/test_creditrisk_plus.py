"""RV-710 to RV-712: CreditRisk+ with Gamma sectors, the stable recursion and the Panjer comparison (dossier 04 H.4)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from riskvalidation.portfolio.creditrisk_plus import (
    CreditRiskPlusInput,
    measures_from_pmf,
    pmf_panjer,
    pmf_stable,
)

POISSON_GOLDEN = (0.548812, 0.054881, 0.112506, 0.175711, 0.027992, 0.034868, 0.028804, 0.006003)
NEGBIN_GOLDEN = (0.826446, 0.150263, 0.020490, 0.002484)


def poisson_input() -> CreditRiskPlusInput:
    return CreditRiskPlusInput(
        nu=np.array([1, 2, 3]), pd=np.array([0.1, 0.2, 0.3]), w0=np.ones(3), weights=np.zeros((3, 0)),
        sigma2=np.zeros(0), loss_unit=1.0, currency="USD",
    )


def negbin_input() -> CreditRiskPlusInput:
    return CreditRiskPlusInput(
        nu=np.array([1]), pd=np.array([0.2]), w0=np.array([0.0]), weights=np.array([[1.0]]), sigma2=np.array([0.5]),
        loss_unit=1.0, currency="USD",
    )


def two_sector_input(n: int = 40, seed: int = 4, max_nu: int = 6) -> CreditRiskPlusInput:
    rng = np.random.default_rng(seed)
    w = rng.dirichlet([2.0, 2.0, 1.0], size=n)
    return CreditRiskPlusInput(
        nu=rng.integers(1, max_nu + 1, size=n), pd=rng.uniform(0.005, 0.08, size=n), w0=w[:, 0], weights=w[:, 1:],
        sigma2=np.array([0.6, 1.2]), loss_unit=1000.0, currency="EUR",
    )


def pgf_fft_pmf(inp: CreditRiskPlusInput, m: int) -> np.ndarray:
    """An independent computation: invert the PGF of dossier 04 H.4 on the unit circle (discrete Fourier transform)."""
    z = np.exp(2j * np.pi * np.arange(m) / m)
    expo = np.zeros(m, dtype=complex)
    for i in range(inp.nu.size):
        d = z ** int(inp.nu[i]) - 1.0
        expo += inp.w0[i] * inp.pd[i] * d
    for k in range(inp.sigma2.size):
        inner = np.zeros(m, dtype=complex)
        for i in range(inp.nu.size):
            inner += inp.weights[i, k] * inp.pd[i] * (z ** int(inp.nu[i]) - 1.0)
        expo -= np.log(1.0 - inp.sigma2[k] * inner) / inp.sigma2[k]
    return np.fft.fft(np.exp(expo)).real / m  # coefficients of z^n, n = 0 .. m - 1


def test_input_validation():
    ok = poisson_input()
    assert ok.nu.flags.writeable is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        ok.loss_unit = 2.0  # type: ignore[misc]

    def build(**over):
        kw = dict(nu=ok.nu, pd=ok.pd, w0=ok.w0, weights=ok.weights, sigma2=ok.sigma2, loss_unit=1.0, currency="USD")
        kw.update(over)
        return CreditRiskPlusInput(**kw)

    bad = [
        dict(w0=np.array([1.0, 1.0, 0.9])),                    # weights do not add up to 1
        dict(nu=np.array([1, 0, 3])),                          # exposure below one unit
        dict(nu=np.array([1.0, 2.5, 3.0])),                    # not integers
        dict(pd=np.array([0.1, 0.2, 1.0])),                    # PD at 1
        dict(w0=np.array([1.0, 1.0, -0.0001])),                # negative weight
        dict(weights=np.zeros((3, 1)), sigma2=np.array([0.5])),  # w0 + weights = 1 holds, but... see below
        dict(loss_unit=0.0),
        dict(currency="usd"),
        dict(sigma2=np.array([0.5])),                          # K = 1 declared by sigma2 but weights have 0 columns
    ]
    # a sector with zero weight everywhere is allowed; the first and last cases must raise
    for over in (bad[0], bad[1], bad[2], bad[3], bad[4], bad[6], bad[7], bad[8]):
        with pytest.raises(ValueError):
            build(**over)
    build(**bad[5])
    with pytest.raises(ValueError):
        build(weights=np.ones((3, 1)), w0=np.zeros(3), sigma2=np.array([-0.5]))  # variance must be positive
    assert negbin_input().weights.shape == (1, 1)


def test_pmf_stable_poisson_and_negative_binomial_goldens():
    p = pmf_stable(poisson_input(), 7)
    assert p.shape == (8,)
    assert np.allclose(p, POISSON_GOLDEN, atol=5e-7)
    assert p[0] == pytest.approx(np.exp(-0.6), abs=1e-15) and p[1] == pytest.approx(np.exp(-0.6) * 0.1, abs=1e-15)
    nb = pmf_stable(negbin_input(), 3)
    assert np.allclose(nb, NEGBIN_GOLDEN, atol=5e-7)
    # the closed form of the negative binomial, computed independently of the engine
    from scipy import stats

    r, s, pdv = 2.0, 0.5, 0.2
    closed = stats.nbinom.pmf(np.arange(4), r, 1.0 / (1.0 + s * pdv))
    assert np.allclose(nb, closed, atol=1e-13)
    # K = 0 equals the direct convolution of the Poisson laws of nu_i Y_i to 1e-12
    inp = poisson_input()
    n_max = 60
    conv = np.array([1.0])
    for nu_i, p_i in zip(inp.nu, inp.pd, strict=True):
        law = np.zeros(n_max + 1)
        k = np.arange(0, n_max // nu_i + 1)
        law[k * nu_i] = stats.poisson.pmf(k, p_i)
        conv = np.convolve(conv, law)[: n_max + 1] if conv.size > 1 else law
    assert np.allclose(pmf_stable(inp, n_max), conv, atol=1e-12, rtol=0)


def test_pmf_stable_mean_tail_and_two_sectors():
    inp = two_sector_input()
    long = pmf_stable(inp, 2000)
    assert np.isclose(long.sum(), 1.0, atol=1e-9)
    mean = float((np.arange(long.size) * long).sum())
    assert mean == pytest.approx(float((inp.nu * inp.pd).sum()), rel=1e-6)  # mean-one factors: mean = sum(nu p)
    # an independent inversion of the probability generating function
    ref = pgf_fft_pmf(inp, 4096)
    assert np.allclose(long[:300], ref[:300], atol=1e-11, rtol=0)
    # the tail mass is left out, never added to the last entry
    short = pmf_stable(inp, 40)
    assert np.array_equal(short, long[:41]) or np.allclose(short, long[:41], atol=1e-15, rtol=0)
    assert short.sum() < 1.0 - 1e-6
    # the variance is larger than the Poisson one when a sector is present (over-dispersion)
    var = float(((np.arange(long.size) - mean) ** 2 * long).sum())
    assert var > mean
    # the stable result and the Monte Carlo of the same Poisson-Gamma model agree
    rng = np.random.default_rng(12)
    s_n = 400_000
    sectors = np.column_stack([rng.gamma(1.0 / s2, s2, size=s_n) for s2 in inp.sigma2])  # mean 1, variance sigma2
    lam = inp.pd[None, :] * (inp.w0[None, :] + sectors @ inp.weights.T)
    counts = rng.poisson(lam)
    x = counts @ inp.nu
    emp = np.bincount(x, minlength=long.size)[: long.size] / s_n
    assert np.abs(np.cumsum(emp) - np.cumsum(long)).max() < 0.004


def test_pmf_stable_is_never_negative_on_a_large_portfolio():
    inp = two_sector_input(n=600, seed=9, max_nu=60)
    mean = float((inp.nu * inp.pd).sum())
    n_max = int(mean * 6)
    p = pmf_stable(inp, n_max)
    assert p.min() >= 0.0 and p.sum() <= 1.0 + 1e-12 and p.size == n_max + 1
    # a long grid with many exposure bands and large variances
    rng = np.random.default_rng(1)
    nu = rng.integers(1, 120, size=1500)
    w = rng.dirichlet([1.0, 1.0, 1.0, 1.0], size=1500)
    big = CreditRiskPlusInput(nu=nu, pd=rng.uniform(0.001, 0.05, size=1500), w0=w[:, 0], weights=w[:, 1:],
                              sigma2=np.array([0.4, 0.9, 2.0]), loss_unit=1.0, currency="USD")
    q = pmf_stable(big, 4000)
    assert q.min() >= 0.0 and q.sum() <= 1.0 + 1e-12
    assert np.isfinite(q).all()


def test_pmf_panjer_agrees_on_goldens_and_is_not_clipped():
    for inp, n in ((poisson_input(), 7), (negbin_input(), 3)):
        assert np.abs(pmf_panjer(inp, n) - pmf_stable(inp, n)).max() < 1e-10
    two = two_sector_input()
    assert np.abs(pmf_panjer(two, 150) - pmf_stable(two, 150)).max() < 1e-8
    # on a long grid with many exposure bands the two recursions still agree to round-off in the bulk, and the Panjer
    # entries are returned as computed (the module never clips or renormalises them); the tail relative error is the
    # instability the page `docs/portfolio/03_creditrisk-plus.md` reports
    rng = np.random.default_rng(5)
    nu = rng.integers(1, 40, size=300)
    w = rng.dirichlet([1.0, 1.0, 1.0], size=300)
    hard = CreditRiskPlusInput(nu=nu, pd=rng.uniform(0.01, 0.08, size=300), w0=w[:, 0], weights=w[:, 1:],
                               sigma2=np.array([0.5, 1.0]), loss_unit=1.0, currency="USD")
    stable = pmf_stable(hard, 3000)
    panjer = pmf_panjer(hard, 3000)
    assert stable.min() >= 0.0 and panjer.shape == stable.shape
    assert np.abs(panjer - stable).max() < 1e-12
    tail = stable > 1e-250
    assert (np.abs(panjer[tail] / stable[tail] - 1.0)).max() < 1e-6
    assert panjer.sum() == pytest.approx(stable.sum(), abs=1e-12)


def test_measures_from_pmf_golden_and_refusal():
    pmf = np.array([0.5, 0.3, 0.2])
    hi, lo = measures_from_pmf(pmf, 1.0, (0.8, 0.7))
    assert hi.var == 1.0 and hi.es == pytest.approx(2.0, abs=1e-12) and hi.el == pytest.approx(0.7)
    assert lo.var == 1.0 and lo.es == pytest.approx(1.666667, abs=5e-7)
    assert hi.ec == pytest.approx(0.3) and hi.var_se is None and hi.es_se is None
    scaled = measures_from_pmf(pmf, 1000.0, (0.8,))[0]
    assert scaled.var == 1000.0 and scaled.es == pytest.approx(2000.0) and scaled.el == pytest.approx(700.0)
    given = measures_from_pmf(pmf, 1.0, (0.8,), el=0.5)[0]
    assert given.el == 0.5 and given.ec == pytest.approx(0.5)
    with pytest.raises(ValueError, match="mass"):
        measures_from_pmf(np.array([0.5, 0.3]), 1.0, (0.95,))
    with pytest.raises(ValueError):
        measures_from_pmf(pmf, 1.0, (1.0,))
    with pytest.raises(ValueError):
        measures_from_pmf(pmf, 0.0, (0.8,))
    with pytest.raises(ValueError):
        measures_from_pmf(np.array([0.5, np.nan, 0.5]), 1.0, (0.8,))
    # on a real distribution: ES at least VaR and the measures are on the unit grid
    inp = two_sector_input()
    p = pmf_stable(inp, 1500)
    m = measures_from_pmf(p, inp.loss_unit, (0.99, 0.999))
    for x in m:
        assert x.es >= x.var and x.var >= 0 and (x.var / inp.loss_unit) == int(x.var / inp.loss_unit)
    assert m[1].var >= m[0].var
