# Reference engines

The published models a validator benchmarks a bank's model against. Each page states the model from its primary
source, the implementation choices, and the published tables the tests reproduce.

| Module | What | Page |
|---|---|---|
| `riskvalidation.engines.vasicek` | the one-factor model: conditional PD, the large homogeneous portfolio's distribution, quantile and density, the implied systematic factor | [Vasicek and the LHP](engines/01_vasicek-lhp.md) |
| `riskvalidation.engines.low_default` | most prudent estimation for low-default portfolios, independent and correlated (Pluto and Tasche) | [Low-default portfolios](engines/02_low-default.md) |
| `riskvalidation.engines.pd_curve` | the rating model's identities, quasi moment matching, and the PD curve calibration approaches of Tasche (2013) for each information case | [PD curve calibration](engines/03_pd-curve-calibration.md) |

Golden values: the Pluto and Tasche tables (one misprint found and documented) and Tasche (2013) Tables 2 to 9,
recomputed from the raw S&P counts.
