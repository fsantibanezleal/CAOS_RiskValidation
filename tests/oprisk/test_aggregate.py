"""RV-1209 to RV-1215: the aggregate annual loss by Panjer recursion, FFT and Monte Carlo, and what is read from it.

The benchmark is Shevchenko (2010) Example 9.4 and Table 2 (arXiv 1008.1108): Poisson(100) with LN(0, 2). Its
references are recomputed here by a direct implementation of the recursion on a plain Python loop with SciPy's
lognormal, so the engine is not tested against itself.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.oprisk import aggregate as a
from riskvalidation.oprisk import distributions as d

LN02 = ("lognormal", (0.0, 2.0))
POI100 = ("poisson", (100.0,))
G = 6000


def arr(*x):
    return np.array(x, dtype=float)


@pytest.fixture(scope="module")
def severity():
    return a.discretise_severity(*LN02, 1.0, G)


@pytest.fixture(scope="module")
def panjer_h(severity):
    return a.panjer_recursion(*POI100, severity)


@pytest.fixture(scope="module")
def fft_h(severity):
    return a.fft_aggregate(*POI100, severity)


# ----------------------------------------------------------------------------------------------------------------
# Panjer coefficients and h0


def test_panjer_coefficients_goldens():
    assert a.panjer_coefficients("poisson", (100.0,)) == (0.0, 100.0)
    assert a.panjer_coefficients("negative-binomial", (2, 0.5)) == pytest.approx((0.5, 0.5))
    assert a.panjer_coefficients("binomial", (2, 0.5)) == pytest.approx((-1.0, 3.0))
    with pytest.raises(ValueError):
        a.panjer_coefficients("lognormal", (0.0, 1.0))
    with pytest.raises(ValueError):
        a.panjer_coefficients("poisson", (-1.0,))


def test_panjer_h0_goldens_and_the_generating_function():
    assert a.panjer_h0("poisson", (2.0,), 0.5) == pytest.approx(0.367879, abs=5e-7)
    assert a.panjer_h0("negative-binomial", (2, 0.5), 0.5) == pytest.approx(0.444444, abs=5e-7)
    assert a.panjer_h0("binomial", (2, 0.5), 0.5) == pytest.approx(0.5625, abs=5e-7)
    f0 = float(stats.lognorm.cdf(0.5, 2.0))
    assert f0 == pytest.approx(0.364456, abs=5e-7)
    assert a.panjer_h0(*POI100, f0) == pytest.approx(2.504194e-28, rel=1e-6)
    # h0 = sum_k f0^k p_k
    for law, params in (("poisson", (2.0,)), ("negative-binomial", (2.0, 0.5)), ("binomial", (4, 0.3))):
        p = d.frequency_pmf(law, params, 400)
        assert a.panjer_h0(law, params, 0.37) == pytest.approx(float((0.37 ** np.arange(401) * p).sum()), rel=1e-12)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            a.panjer_h0("poisson", (2.0,), bad)


# ----------------------------------------------------------------------------------------------------------------
# discretisation


def test_discretise_severity_golden_bounds_and_missing_mass():
    f = a.discretise_severity(*LN02, 1.0, 10)
    assert f[:3] == pytest.approx([0.364456, 0.215872, 0.096248], abs=5e-7)
    assert f.shape == (10,) and np.all(f >= 0.0) and f.sum() < 1.0  # the missing mass is the tail beyond the grid
    cdf = stats.lognorm(2.0).cdf
    j = np.arange(10)
    fwd = a.discretise_severity(*LN02, 1.0, 10, "forward")
    assert fwd == pytest.approx(cdf(j + 1.0) - cdf(j.astype(float)), abs=1e-15)
    back = a.discretise_severity(*LN02, 1.0, 10, "backward")
    assert back[0] == 0.0 and back[1:] == pytest.approx(cdf(j[1:].astype(float)) - cdf(j[1:] - 1.0), abs=1e-15)
    up = a.discretise_severity(*LN02, 1.0, 10, "forward").cumsum()
    lo = back.cumsum()
    mid = f.cumsum()
    assert np.all(lo <= mid + 1e-15) and np.all(mid <= up + 1e-15)
    # a step scales the grid, not the masses
    assert a.discretise_severity("lognormal", (np.log(10.0), 2.0), 10.0, 10) == pytest.approx(f, abs=1e-14)
    for kwargs in ({"n_points": 1}, {"step": 0.0}, {"method": "trapezoid"}):
        args = {"step": 1.0, "n_points": 10, "method": "central", **kwargs}
        with pytest.raises(ValueError):
            a.discretise_severity(*LN02, **args)


# ----------------------------------------------------------------------------------------------------------------
# Panjer recursion


def test_panjer_thinned_poisson_is_poisson_one():
    f = np.zeros(30)
    f[:2] = 0.5
    h = a.panjer_recursion("poisson", (2.0,), f)
    assert h[:3] == pytest.approx([0.367879, 0.367879, 0.183940], abs=5e-7)
    assert h == pytest.approx(stats.poisson.pmf(np.arange(30), 1.0), abs=1e-14)
    assert h.sum() <= 1.0 + 1e-15


def test_panjer_recursion_other_laws_match_the_definition():
    # negative binomial and binomial against the explicit convolution sum_n p_n f^{*n}
    f = np.array([0.2, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for law, params, nmax in (("negative-binomial", (2.0, 0.5), 80), ("binomial", (3, 0.4), 3)):
        h = a.panjer_recursion(law, params, f)
        p = d.frequency_pmf(law, params, nmax)
        want = np.zeros(10)
        conv = np.zeros(10)
        conv[0] = 1.0
        for n in range(nmax + 1):
            want += p[n] * conv
            conv = np.convolve(conv, f)[:10]
        assert h == pytest.approx(want, abs=1e-12), law


def test_panjer_recursion_refusals():
    f = np.array([0.5, 0.5, 0.0])
    with pytest.raises(ValueError):
        a.panjer_recursion("lognormal", (0.0, 1.0), f)
    with pytest.raises(ValueError):
        a.panjer_recursion("poisson", (2.0,), np.array([1.0, 0.0]))  # f0 outside [0, 1)
    with pytest.raises(ValueError):
        a.panjer_recursion("poisson", (2000.0,), np.array([0.0, 1.0]))  # h0 underflows: no silent zeros


def test_benchmark_panjer_cumulative_sums_and_quantile(panjer_h):
    cs = np.cumsum(panjer_h)
    assert cs[5847] == pytest.approx(0.998999329180, abs=5e-10)
    assert cs[5848] == pytest.approx(0.998999773, abs=5e-10)
    assert cs[5849] == pytest.approx(0.999000217026, abs=5e-10)
    assert a.quantile_from_pmf(panjer_h, 1.0, 0.999) == 5849.0
    assert 5849.0 <= a.quantile_from_pmf(panjer_h, 1.0, 0.999) <= 5850.0
    assert panjer_h[0] == pytest.approx(2.504194e-28, rel=1e-6)


def test_benchmark_panjer_independent_recursion_and_moments(severity, panjer_h):
    """A plain Python loop of eq. 36 (Poisson: a = 0, b = lambda), written here for the test."""
    h = np.zeros(400)
    f = severity[:400]
    h[0] = math.exp(100.0 * (f[0] - 1.0))
    for n in range(1, 400):
        h[n] = sum(100.0 * j / n * f[j] * h[n - j] for j in range(1, n + 1))
    assert panjer_h[:400] == pytest.approx(h, rel=1e-12, abs=1e-300)
    # the discrete mean on the grid is close to lambda E[X] (the grid cuts the tail)
    x = np.arange(G, dtype=float)
    assert float((x * panjer_h).sum()) < 738.9056


# ----------------------------------------------------------------------------------------------------------------
# FFT


def test_fft_thinned_poisson_golden():
    f = np.zeros(30)
    f[:2] = 0.5
    h = a.fft_aggregate("poisson", (2.0,), f)
    assert h.shape == (30,)
    assert h[:3] == pytest.approx([0.367879, 0.367879, 0.183940], abs=5e-7)
    assert h == pytest.approx(stats.poisson.pmf(np.arange(30), 1.0), abs=1e-12)


def test_fft_agrees_with_panjer_on_the_benchmark(panjer_h, fft_h):
    assert np.max(np.abs(fft_h - panjer_h)) < 1e-9
    cs = np.cumsum(fft_h)
    assert cs[5847] == pytest.approx(0.998999329180, abs=5e-10)
    assert cs[5849] == pytest.approx(0.999000217026, abs=5e-10)
    assert a.quantile_from_pmf(fft_h, 1.0, 0.999) == 5849.0
    assert np.all(fft_h >= 0.0)


def test_fft_other_laws_agree_with_panjer():
    f = a.discretise_severity("lognormal", (0.0, 1.0), 0.5, 512)
    for law, params in (("negative-binomial", (2.0, 0.3)), ("binomial", (20, 0.2)), ("poisson", (3.0,))):
        hp = a.panjer_recursion(law, params, f)
        hf = a.fft_aggregate(law, params, f)
        assert np.max(np.abs(hf - hp)) < 1e-10, law


def test_fft_zero_padding_matters_and_an_unstable_transform_is_refused(severity):
    ref = a.panjer_recursion(*POI100, severity[:1000])
    wrapped = a.fft_aggregate(*POI100, severity[:1000], padding_factor=1)   # the heavy tail wraps around
    padded = a.fft_aggregate(*POI100, severity[:1000], padding_factor=8)
    assert np.max(np.abs(padded - ref)) < 1e-9
    assert np.max(np.abs(wrapped - ref)) > 1e-6
    # a severity that carries more than unit mass makes the negative binomial transform diverge: negative values
    # beyond the roundoff tolerance are aliasing and are refused, never clipped
    with pytest.raises(ValueError, match="alias"):
        a.fft_aggregate("negative-binomial", (1.0, 0.5), np.array([0.0, 3.0, 0.0, 0.0]))
    for bad in ({"padding_factor": 0}, {"tilt": -0.1}):
        with pytest.raises(ValueError):
            a.fft_aggregate(*POI100, severity, **bad)
    with pytest.raises(ValueError):
        a.fft_aggregate(*POI100, np.array([0.5, -0.1, 0.2]))


def test_fft_exponential_tilting_recovers_the_untilted_pmf():
    f = a.discretise_severity("lognormal", (0.0, 1.5), 1.0, 2048)
    base = a.fft_aggregate("poisson", (20.0,), f)
    tilted = a.fft_aggregate("poisson", (20.0,), f, tilt=0.002)
    assert np.max(np.abs(tilted - base)) < 1e-10
    ref = a.panjer_recursion("poisson", (20.0,), f)
    assert np.max(np.abs(tilted - ref)) < 1e-10


# ----------------------------------------------------------------------------------------------------------------
# Monte Carlo


def test_simulate_annual_losses_is_deterministic_and_independent_of_the_chunk():
    args = ("poisson", (5.0,), "lognormal", (1.0, 0.8))
    s1 = a.simulate_annual_losses(*args, n_years=5_000, seed=17)
    s2 = a.simulate_annual_losses(*args, n_years=5_000, seed=17, chunk_years=777)
    s3 = a.simulate_annual_losses(*args, n_years=5_000, seed=17, chunk_years=1)
    s4 = a.simulate_annual_losses(*args, n_years=5_000, seed=18)
    assert np.array_equal(s1.losses, s2.losses) and np.array_equal(s1.losses, s3.losses)
    assert not np.array_equal(s1.losses, s4.losses)
    assert s1.n_years == 5_000 and s1.seed == 17 and s1.losses.shape == (5_000,) and np.all(s1.losses >= 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s1.seed = 3  # type: ignore[misc]
    for bad in ({"n_years": 0}, {"n_years": 10, "chunk_years": 0}):
        with pytest.raises(ValueError):
            a.simulate_annual_losses(*args, **{"n_years": 10, "seed": 1, **bad})


def test_simulate_annual_losses_moments_for_each_frequency_law():
    for freq in (("poisson", (6.0,)), ("negative-binomial", (3.0, 0.4)), ("binomial", (12, 0.5))):
        s = a.simulate_annual_losses(*freq, "lognormal", (0.0, 0.5), n_years=200_000, seed=4)
        p = d.frequency_pmf(*freq, 400)
        n = np.arange(401)
        en, vn = float((n * p).sum()), float((n * n * p).sum()) - float((n * p).sum()) ** 2
        m1, m2 = d.severity_moment("lognormal", (0.0, 0.5), 1), d.severity_moment("lognormal", (0.0, 0.5), 2)
        mean, var = en * m1, en * (m2 - m1 * m1) + vn * m1 * m1  # compound mean and variance
        se = math.sqrt(var / 200_000)
        assert abs(s.losses.mean() - mean) < 4 * se, freq
        assert s.losses.var() == pytest.approx(var, rel=0.05), freq
    s0 = a.simulate_annual_losses("poisson", (0.2,), "lognormal", (0.0, 1.0), n_years=2_000, seed=2)
    assert (s0.losses == 0.0).mean() == pytest.approx(math.exp(-0.2), abs=0.03)  # the count follows the law


def test_benchmark_monte_carlo_mean_within_four_standard_errors():
    s = a.simulate_annual_losses(*POI100, *LN02, n_years=1_000_000, seed=2026, chunk_years=20_000)
    assert abs(s.losses.mean() - 738.9056) <= 2.184
    q, se = a.mc_quantile(s.losses, 0.999)
    assert 5000.0 < q < 7000.0 and se > 0.0


def test_mc_quantile_goldens():
    x = np.arange(1.0, 1001.0)
    assert a.mc_quantile(x, 0.5) == (500.0, 16.0)
    assert a.mc_quantile(x, 0.999) == (999.0, 1.0)
    shuffled = np.random.default_rng(0).permutation(x)
    assert a.mc_quantile(shuffled, 0.5) == (500.0, 16.0)
    assert shuffled.tolist() != x.tolist()  # the input is not sorted in place
    assert a.mc_quantile(arr(3.0, 1.0, 2.0), 0.7)[0] == 3.0  # ceil(2.1) = 3, never moved by 3 * 0.7 float fuzz
    assert a.mc_quantile(np.arange(1.0, 101.0), 0.07)[0] == 7.0   # 100 * 0.07 = 7.000000000000001 in floats
    assert a.mc_quantile(np.arange(1.0, 101.0), 0.55)[0] == 55.0  # 100 * 0.55 = 55.00000000000001 in floats
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError):
            a.mc_quantile(x, bad)
    with pytest.raises(ValueError):
        a.mc_quantile(arr(), 0.5)


# ----------------------------------------------------------------------------------------------------------------
# quantile, exceedance and moments


def test_quantile_from_pmf_goldens_and_refusal():
    h = arr(0.5, 0.3, 0.2)
    assert a.quantile_from_pmf(h, 10.0, 0.75) == 10.0
    assert a.quantile_from_pmf(h, 10.0, 0.9) == 20.0
    assert a.quantile_from_pmf(h, 10.0, 0.5) == 0.0
    with pytest.raises(ValueError):
        a.quantile_from_pmf(arr(0.5, 0.3), 10.0, 0.9)  # mass beyond the grid: never extrapolated
    with pytest.raises(ValueError):
        a.quantile_from_pmf(h, 10.0, 0.0)


def test_exceedance_from_pmf_golden():
    x, sf = a.exceedance_from_pmf(arr(0.5, 0.3, 0.2), 10.0)
    assert x.tolist() == [0.0, 10.0, 20.0]
    assert sf == pytest.approx([0.5, 0.2, 0.0], abs=1e-15)
    assert np.all(np.diff(sf) <= 0.0) and sf.min() >= 0.0 and sf.max() <= 1.0
    _, sf2 = a.exceedance_from_pmf(arr(0.5, 0.3), 1.0)  # the grid misses mass: the curve does not reach 0
    assert sf2[-1] == pytest.approx(0.2)


def test_compound_poisson_moments_benchmark_and_infinite_cases():
    m = a.compound_poisson_moments(100.0, *LN02)
    assert m.mean == pytest.approx(738.9056, abs=5e-5)
    assert m.variance == pytest.approx(298095.7987, abs=5e-5)
    assert m.skewness == pytest.approx(40.3428, abs=1e-4)
    assert m.mean == pytest.approx(100 * math.e ** 2, rel=1e-14)
    assert m.variance == pytest.approx(100 * math.e ** 8, rel=1e-14)
    assert m.skewness == pytest.approx(math.e ** 6 / 10, rel=1e-13)
    inf_mean = a.compound_poisson_moments(10.0, "gpd", (1.1, 100.0, 0.0))
    assert inf_mean.mean == math.inf and inf_mean.variance == math.inf and math.isnan(inf_mean.skewness)
    inf_var = a.compound_poisson_moments(10.0, "gpd", (0.7, 100.0, 0.0))
    assert math.isfinite(inf_var.mean) and inf_var.variance == math.inf and math.isnan(inf_var.skewness)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.mean = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        a.compound_poisson_moments(0.0, *LN02)


# ----------------------------------------------------------------------------------------------------------------
# expected shortfall (the grid must carry the tail)


def test_es_from_pmf_golden_point_mass_and_the_var_bound():
    h = arr(0.5, 0.3, 0.2)
    assert a.es_from_pmf(h, 10.0, 0.75) == pytest.approx(18.0, abs=1e-12)        # q = 10, E[Z; Z > q] = 4
    assert a.es_from_pmf(arr(0.0, 0.0, 1.0), 10.0, 0.5) == pytest.approx(20.0)   # a point mass: ES = VaR
    for level in (0.1, 0.5, 0.75, 0.9, 0.99):
        assert a.es_from_pmf(h, 10.0, level) >= a.quantile_from_pmf(h, 10.0, level) - 1e-12
    with pytest.raises(ValueError):
        a.es_from_pmf(arr(0.5, 0.3), 10.0, 0.9)
    with pytest.raises(ValueError):
        a.es_from_pmf(h, 10.0, 0.75, max_missing_share=0.0)


def test_es_from_pmf_matches_a_monte_carlo_expected_shortfall_on_a_light_tail():
    f = a.discretise_severity("lognormal", (0.0, 1.0), 0.25, 8192)
    h = a.fft_aggregate("poisson", (10.0,), f)
    es = a.es_from_pmf(h, 0.25, 0.99)
    var = a.quantile_from_pmf(h, 0.25, 0.99)
    s = a.simulate_annual_losses("poisson", (10.0,), "lognormal", (0.0, 1.0), n_years=400_000, seed=11)
    q, _ = a.mc_quantile(s.losses, 0.99)
    mc_es = float(s.losses[s.losses > q].mean())
    assert es >= var and es == pytest.approx(mc_es, rel=0.03)


def test_es_from_pmf_refuses_a_grid_that_misses_the_tail_and_never_computes_an_infinite_mean(panjer_h):
    # the benchmark on 6000 points misses 0.000936 of the mass: a grid-only reading would be 377.8 below the VaR 5849
    assert 1.0 - panjer_h.sum() == pytest.approx(0.000936, abs=2e-6)
    with pytest.raises(ValueError, match="grid"):
        a.es_from_pmf(panjer_h, 1.0, 0.999)
    # a tolerance loose enough to accept it would give a number below the VaR: that is why the guard exists
    loose = a.es_from_pmf(panjer_h, 1.0, 0.999, max_missing_share=10.0)
    assert loose < a.quantile_from_pmf(panjer_h, 1.0, 0.999)
    # a tail index of 1.1 has an infinite mean: the grid-only ES grows with the grid and is refused at any size
    for g in (1000, 3000):
        f = a.discretise_severity("gpd", (1.1, 1.0, 0.0), 1.0, g)
        h = a.panjer_recursion("poisson", (2.0,), f)
        with pytest.raises(ValueError, match="grid"):
            a.es_from_pmf(h, 1.0, 0.99)


# ----------------------------------------------------------------------------------------------------------------
# the other benchmark values of Shevchenko's tables


def test_benchmark_table_2_row_5848_and_table_4_forward_bound():
    f = a.discretise_severity(*LN02, 1.0, G)
    h = a.panjer_recursion(*POI100, f)
    assert np.cumsum(h)[5848] == pytest.approx(0.998999773241, abs=5e-10)       # Table 2, index 5848
    fwd = a.panjer_recursion(*POI100, a.discretise_severity(*LN02, 1.0, G, "forward"))
    cs = np.cumsum(fwd)
    assert cs[5811] == pytest.approx(0.998999719, abs=5e-9) and cs[5812] == pytest.approx(0.999000163, abs=5e-9)
    # the table rows rule (the paper text says 5811)
    assert a.quantile_from_pmf(fwd, 1.0, 0.999) == 5812.0
    back = a.panjer_recursion(*POI100, a.discretise_severity(*LN02, 1.0, G, "backward"))
    assert np.cumsum(back)[5849] == pytest.approx(0.998970962, abs=5e-9)           # Table 4 lower bound
