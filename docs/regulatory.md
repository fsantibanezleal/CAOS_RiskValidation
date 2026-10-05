# Regulatory calculators

Published formulas that turn a model's parameters into capital, provisions or liquidity metrics. Every function
returns the regulatory parameters it applied with their paragraph references and the in-force date of the chapter
they come from; none of them certifies compliance.

| Module | What | Page |
|---|---|---|
| `riskvalidation.regulatory.irb` | IRB risk-weight functions (Basel III final, CRR3, Basel II, CMF RAN 21-6), PD floors, firm-size and financial-institution adjustments, defaulted exposures, expected loss against provisions | [IRB](regulatory/01_irb.md) |

Golden values: BCBS CRE99 Table 1, all 144 cells within 0.01 percentage points (141 to the printed two decimals).
