"""Risk aggregation: the ladder from the simple sum to the worst-case bounds, the dependence band, model risk and the
validation tests.

Modules are imported by path (for example ``from riskvalidation.aggregation import simple``); nothing is re-exported
here.
Capital is economic capital on the unexpected-loss basis, ``EC = VaR - EL``. The simple sum is the headline capital of
the
case; every other figure of the band is information, not capital. The engine computes and states no compliance claim:
thresholds and light parameters are supplied by the caller or are Contraste internal gates, not regulatory requirements.
The validation tests are in ``riskvalidation.validation.aggregation``. Contract: ``engine-aggregation-api`` 1.0.0.
"""

__all__ = [
    "band",
    "bounds",
    "copulas",
    "model_risk",
    "simple",
]
