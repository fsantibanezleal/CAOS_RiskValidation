"""Interest-rate risk in the banking book: curves, shocks, slotting, EVE and NII under SRP31 and RAN 21-13 Annex 1,
the behavioural models and the simulation.

Modules are imported by path (for example ``from riskvalidation.irrbb import eve_nii``); nothing is re-exported here.
A positive delta EVE or delta NII is a loss; discounting is continuously compounded. The engine computes and states no
compliance claim: thresholds, floors a supervisor may choose and light parameters are supplied by the caller or stated
by the function. A shock size outside the transcribed tables is refused unless the caller supplies it with its label.
The validation tests are in ``riskvalidation.validation.irrbb``. Contract: ``engine-irrbb-api`` 1.1.0.
"""
