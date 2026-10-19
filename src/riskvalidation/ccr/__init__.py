"""Counterparty credit risk: the Hull-White one-factor model on any time grid, fixed-float swaps along paths, the
exposure measures (EE, ENE, effective EE and EPE, PFE, netting, effective maturity), collateral with the margin period
of risk and the dynamic initial margin, and CVA, DVA and wrong-way risk (dossier 05 C, the U9 research addendum R1 to R6
and R9).

Modules are imported by path (for example ``from riskvalidation.ccr import hw1f``); nothing is
re-exported here. Money is a float in the currency of the netting set, rates are decimals compounded continuously, times
are years from the valuation date and the margin period of risk is in business days with 250 to the year. EE is the
positive exposure to the counterparty, ENE the bank's own reported positive, CVA a positive cost and DVA a positive
benefit. The engine computes and states no compliance claim. It is pure NumPy and SciPy: ORE and QuantLib never enter
it, and the reconciliation against them is the caller's. The validation tests are in ``riskvalidation.validation.ccr``.
Contract: ``engine-ccr-api`` 1.0.0.
"""
