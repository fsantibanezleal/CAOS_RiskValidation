"""U2: the one-factor model and its large homogeneous portfolio (RV-205)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtri

from riskvalidation.engines.vasicek import conditional_pd, implied_factor, lhp_cdf, lhp_pdf, lhp_quantile
from riskvalidation.regulatory import capital_requirement


def test_lhp_identities():
    """RV-205: the quantile inverts the distribution, equals the conditional PD at the stressed factor and the IRB
    retail formula's conditional PD; the implied factor inverts the conditional PD; the distribution's mean is the
    PD."""
    pd, rho = 0.02, 0.12
    for q in (0.5, 0.9, 0.99, 0.999):
        x = lhp_quantile(q, pd, rho)
        assert lhp_cdf(x, pd, rho) == pytest.approx(q, abs=1e-12)
        assert conditional_pd(pd, rho, -float(ndtri(q))) == pytest.approx(x, rel=1e-12)
    z = np.array([-3.0, -1.0, 0.0, 1.5])
    assert implied_factor(conditional_pd(pd, rho, z), pd, rho) == pytest.approx(z, abs=1e-10)
    mean, _ = integrate.quad(lambda x: x * lhp_pdf(x, pd, rho), 0.0, 1.0, limit=200)
    assert mean == pytest.approx(pd, rel=1e-6)
    # the IRB formula for a retail class: K / LGD + PD is the 99.9% quantile at the class's correlation
    r = capital_requirement("residential_mortgage", pd, 0.45)
    assert r.k / 0.45 + pd == pytest.approx(lhp_quantile(0.999, pd, 0.15), rel=1e-12)
    with pytest.raises(ValueError):
        implied_factor(0.0, pd, rho)
