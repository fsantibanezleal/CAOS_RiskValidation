"""RV-1216: the single loss approximation (Shevchenko eq. 70) and the UNVERIFIED mean-corrected form."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

from riskvalidation.oprisk import sla

GPD = ("gpd", (0.5, 100.0, 0.0))


def test_sla_var_golden_and_closed_form():
    v = sla.sla_var(10.0, *GPD, 0.99)
    assert v == pytest.approx(6124.555320, abs=5e-7)
    assert v == pytest.approx(float(stats.genpareto.ppf(1 - 0.01 / 10, 0.5, scale=100.0)), rel=1e-13)
    assert v == sla.sla_var_poisson_gpd(10.0, 0.5, 100.0, 0.99)
    assert sla.sla_var_poisson_gpd(10.0, 0.5, 100.0, 0.99) == pytest.approx(200 * (math.sqrt(1000.0) - 1), rel=1e-13)
    # the lognormal quantile F^{-1}(1 - (1 - level) / E[N])
    ln = sla.sla_var(50.0, "lognormal", (1.0, 0.8), 0.999)
    assert ln == pytest.approx(float(stats.lognorm.ppf(1 - 0.001 / 50, 0.8, scale=math.e)), rel=1e-12)


def test_sla_var_location_and_scaling_like_lambda_to_xi():
    base = sla.sla_var_poisson_gpd(10.0, 0.7, 50.0, 0.999, location=1_000.0)
    assert base == pytest.approx(1_000.0 + 50.0 / 0.7 * ((10.0 / 0.001) ** 0.7 - 1), rel=1e-13)
    assert sla.sla_var(10.0, "gpd", (0.7, 50.0, 1_000.0), 0.999) == pytest.approx(base, rel=1e-12)
    # excess over the location scales like lambda ** xi
    x1 = sla.sla_var_poisson_gpd(10.0, 0.7, 50.0, 0.999) + 50.0 / 0.7
    x2 = sla.sla_var_poisson_gpd(40.0, 0.7, 50.0, 0.999) + 50.0 / 0.7
    assert x2 / x1 == pytest.approx(4.0 ** 0.7, rel=1e-12)


def test_sla_var_monotone_and_refusals():
    ns = np.array([2.0, 5.0, 20.0, 100.0])
    vals = [sla.sla_var(n, *GPD, 0.999) for n in ns]
    assert vals == sorted(vals)
    lv = [sla.sla_var(10.0, *GPD, q) for q in (0.9, 0.99, 0.999, 0.9999)]
    assert lv == sorted(lv)
    with pytest.raises(ValueError):
        sla.sla_var(0.001, *GPD, 0.999)             # E[N] <= 1 - level
    with pytest.raises(ValueError):
        sla.sla_var(10.0, *GPD, 1.0)
    with pytest.raises(ValueError):
        sla.sla_var(10.0, *GPD, 0.0)
    for xi, beta in ((0.0, 100.0), (-0.1, 100.0), (0.5, 0.0)):
        with pytest.raises(ValueError):
            sla.sla_var_poisson_gpd(10.0, xi, beta, 0.99)


def test_sla_dossier_05_f6_figure_is_reproduced_and_rounding_is_what_moves_it():
    """Dossier 05 F.6 (erratum: the figure reproduces): xi 0.96378 and beta 22,349.4 give EUR 647.1 million, and the
    parameters rounded to 0.96 and 22,350 give 624.1 million."""
    fitted = sla.sla_var_poisson_gpd(41.0, 0.96378, 22_349.4, 0.999, location=20_000.0)
    assert fitted == pytest.approx(647.12e6, rel=1e-3)
    rounded = sla.sla_var_poisson_gpd(41.0, 0.96, 22_350.0, 0.999, location=20_000.0)
    assert rounded == pytest.approx(624.1e6, rel=1e-3)


def test_mean_corrected_form_is_unverified_and_adds_lambda_times_the_mean():
    r = sla.sla_var_mean_corrected(10.0, *GPD, 0.99)
    assert r.var_first_order == pytest.approx(6124.555320, abs=5e-7)
    assert r.mean_correction == pytest.approx(2000.0, rel=1e-13)
    assert r.var == pytest.approx(8124.555320, abs=5e-7)
    assert r.var == r.var_first_order + r.mean_correction and r.var >= r.var_first_order
    assert r.status == "UNVERIFIED" and r.level == 0.99
    assert "UNVERIFIED" in (sla.sla_var_mean_corrected.__doc__ or "")
    assert "UNVERIFIED" in (sla.SlaResult.__doc__ or "")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.var = 0.0  # type: ignore[misc]
    ln = sla.sla_var_mean_corrected(20.0, "lognormal", (0.0, 1.0), 0.999)
    assert ln.mean_correction == pytest.approx(20.0 * math.exp(0.5), rel=1e-13) and ln.status == "UNVERIFIED"


def test_mean_corrected_form_raises_where_the_mean_is_infinite():
    for xi in (1.0, 1.1, 2.0):
        with pytest.raises(ValueError, match="infinite mean"):
            sla.sla_var_mean_corrected(10.0, "gpd", (xi, 100.0, 0.0), 0.99)
    # the first-order form still exists there
    assert math.isfinite(sla.sla_var(10.0, "gpd", (1.1, 100.0, 0.0), 0.99))
