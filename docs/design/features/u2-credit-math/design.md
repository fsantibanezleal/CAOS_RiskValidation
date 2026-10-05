# U2 credit risk mathematics, design

## Modules

| Module | What | Sources |
|---|---|---|
| `riskvalidation.regulatory.irb` | the IRB risk-weight functions by asset class and regime, the floors, the expected loss against provisions | CRE31, CRE32, CRE35, CAP10.19, CAP30.13; CRR3 Article 153; CMF RAN 21-6 numeral 4 |
| `riskvalidation.engines.vasicek` | the one-factor conditional PD, the large homogeneous portfolio, the implied systematic factor | Vasicek (1987, 1991, 2002); Gordy (2003); BCBS explanatory note (2005) |
| `riskvalidation.engines.low_default` | most prudent estimation, independent and one-factor correlated; scaling (section 5); several years (section 6) | Pluto and Tasche (2005), arXiv cond-mat/0411699v3 |
| `riskvalidation.engines.pd_curve` | the rating model's identities, quasi moment matching, the calibration approaches of cases 1 to 3, the profile chi-square test | Tasche (2013), arXiv 1212.3716; Tasche (2009), arXiv 0905.3928 |
| `riskvalidation.validation.calibration` (`pd.default_profile`) | the chi-square test of the implied default profile with a Monte Carlo p-value | Tasche (2013), sections 3.3 and 4.2.5; Pearson (1900) |

## Regimes

A regime is data, not code: a record of the paragraph-referenced parameters (PD floors, the scaling factor, the LGD
inputs a regime fixes, the correlation family) with the in-force date of its chapter. Four regimes ship: `basel3`
(CRE31 and CRE32 in force 2023-01-01), `crr3` (Regulation (EU) 2024/1623, applying from 2025-01-01; the formula equals
Basel III final), `basel2` (CRE30.4 in force 2019-12-15: the 1.06 factor; 0.03% floors), `cl_ran21_6` (CMF RAN 21-6
numeral 4.4, last amended by Circular N°2.370 of 2026-04-10; correlations of numeral 4.4.2). The regimes are shared
module state, so their tables are read-only mappings (`types.MappingProxyType`) and every result carries its own copy
of the references; `Regime.to_dict()` gives a JSON-safe copy. Nothing in Contraste says that a Chilean bank uses
internal methods for capital (none does; CMF, August 2026).

## Numerics

The normal distribution and its inverse come from SciPy (`scipy.special.ndtr`, `ndtri`), which Pyodide ships. The
correlated most-prudent bound evaluates its one-factor integral by Gauss-Hermite quadrature (256 probabilists' nodes)
inside a Brent root search. Quasi moment matching solves its two equations as nested bracketed roots: for a slope beta
the unconditional PD is strictly decreasing in alpha (one root alpha(beta)), and along alpha(beta) the accuracy ratio
(A.11b) rises from 0 towards the largest value the grade scale allows, so both roots are bracketed around the binormal
values (A.6) and found by Brent's method to machine precision; an unattainable target raises `ImproperModel` instead of
a non-converged answer. The multi-period bound averages its T-dimensional expectation over n_sim factor paths drawn once per call from a seeded generator and reused for every trial p, so the root is smooth in p and reproducible. The scaled-likelihood root (4.8a) is bracketed beyond (4.8b), where it always lies. The
theorem 3.3 root of (3.11a) is bracketed between 0 and the point short of 1 where the function turns negative. The
Monte Carlo p-value of `pd.default_profile` draws multinomial samples in chunks of 50,000 from a seeded NumPy
generator; `n_sim` and `seed` are part of the result and of its inputs hash.

## Tolerances

CRE99 Table 1 is printed to two decimals and three of its 144 cells sit one unit off an exact computation (dossier 10,
section 1): the tolerance is 0.01 percentage points. Pluto and Tasche print two decimals; the independent bounds are
exact except Table 4 at 75% (a computation the paper carries into Table 11), the correlated ones within one unit
(Table 7 in both directions; Table 8 always at or above the exact bound, the paper's integration erring upward). The
scaling tables inherit those inputs (K within one unit, bounds within two, one misprint); the five-year tables are
gated on a deterministic quadrature, the print lying 1% to 5% above the model's bound at the high levels. Tasche (2013):
the computations start from Table 2's counts, never from a rounded intermediate. Table 5 is reproduced within one unit
of its third decimal (15 of 17 cells identical); the paper's own fits sit at a slightly different (alpha, beta) than
the exact root (offsets of 5e-5 to 8e-4, measured on Table 9, where log lambda is affine in the scores), which carries a
relative difference up to about 1e-4 into Tables 6 and 7: the gate there is one unit of the printed decimal plus 0.01%
of the value. Table 8 is within one unit of its printed precision. The p-values of Table 7 are Monte Carlo
approximations of unstated size: the gate assumes at least 2,000 samples (an assumption) and allows three standard
errors of that run, three of ours (100,000 samples, seed 2013) and half a printed unit.
