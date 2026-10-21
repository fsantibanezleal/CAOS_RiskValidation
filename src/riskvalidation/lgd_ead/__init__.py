"""LGD and EAD: the workout LGD, the LGD rungs, downturn LGD, CCF and EAD, and the IRB floors.

Modules are imported by path (for example ``from riskvalidation.lgd_ead import workout``); nothing is re-exported here.
The engine computes and states no compliance claim: discount rates, downturn periods, thresholds and light
parameters are policy parameters the caller supplies. LGD and CCF values outside [0, 1] are kept and flagged, never
silently clipped. The CCF is in the EU convention (the share of the undrawn amount drawn by default, the LEQ in US
terms). The validation tests are in ``riskvalidation.validation.lgd_ead``. Contract: ``engine-lgd-ead-api`` 1.0.0.
"""

__all__ = [
    "ccf",
    "downturn",
    "floors",
    "lgd_models",
    "workout",
]
