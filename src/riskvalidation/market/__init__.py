"""Market risk: VaR and ES estimators (historical, parametric, GARCH, EVT, CAViaR, GAS-FZ), the FZ0 scoring and the
model comparison, and the FRTB statistics (contract `engine-market-api` 1.0.0).

The estimators work in the return convention (VaR and ES negative decimals, dossier 05 A.0); positive loss numbers exist
only at the reporting layer, through `conventions.to_loss_numbers`. The engine core is NumPy and SciPy; `arch` (the
optional extra `market`) is imported inside the functions that need it. The back-tests are in
`riskvalidation.validation.market`.
"""

__all__ = [
    "caviar",
    "conformal",
    "conventions",
    "evt",
    "frtb",
    "garch",
    "gas_fz",
    "historical",
    "parametric",
    "pla_complement",
    "scoring",
]
