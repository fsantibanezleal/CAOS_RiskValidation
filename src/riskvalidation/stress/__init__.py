"""Macro stress testing: the satellites of a default or loss rate, the scenario paths, the capital engine of the IAPE
projection, the reverse stress and the scenario models (the VAR conditional forecast and growth-at-risk).

Modules are imported by path (for example ``from riskvalidation.stress import capital``); nothing is re-exported here.
Rates, ratios and probabilities are decimals; money is a float in one currency and is never converted; a path keeps the
unit of its source. The engine computes and states no compliance claim: the reverse stress statement and the P2G proxy
are Contraste's, not regulatory text. The supervisory replication (C19) is not part of this package. The validation
tests are in ``riskvalidation.validation.stress``. Contract: ``engine-stress-api`` 1.0.0.
"""

__all__ = [
    "capital",
    "paths",
    "reverse",
    "satellites",
    "scenario_models",
]
