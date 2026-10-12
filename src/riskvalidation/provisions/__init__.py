"""IFRS 9 expected credit loss: term structures, staging, measurement, overlays and reconciliation.

Modules are imported by path (``from riskvalidation.provisions import ecl``); nothing is re-exported here. The engine
computes and states no compliance claim: multipliers, floors, probation lengths and horizons are policy parameters
the caller supplies, and only the 30 and 90 days past due presumptions are in IFRS 9.
"""
