"""IFRS 9 expected credit loss and the local statutory provision regimes.

The IFRS 9 half (term structures, staging, ECL, overlays, reconciliation) and the local half (``local_cl``: CMF B-1 of
Chile; ``local_es``: Banco de Espana Circular 4/2017 Annex 9) are separate modules, imported by path (for example
``from riskvalidation.provisions import ecl``); nothing is re-exported here. The engine computes and states no
compliance claim: multipliers, floors, probation lengths and horizons are policy parameters the caller supplies, and
only the 30 and 90 days past due presumptions are in IFRS 9. A local regime is kept apart from IFRS 9.
"""
