"""Liquidity: run-off, the counterbalancing capacity, ladders, scenarios, reverse stress and the depositor generator.

Modules are imported by path (for example ``from riskvalidation.liquidity import runoff``); nothing is re-exported here.
Rates are decimals per relevant day, cumulative outflow is ``1 - (1 - q)^n``, and a survival period is "survives the
horizon" with the horizon when the net liquidity position never turns negative, never a number. The engine computes and
states no compliance claim; a missing input is refused with the field named, never filled. The LCR and NSFR calculators
are in ``riskvalidation.regulatory.lcr_nsfr`` and the validation tests in ``riskvalidation.validation.liquidity``.
Contract: ``engine-liquidity-api`` 1.0.0.
"""
