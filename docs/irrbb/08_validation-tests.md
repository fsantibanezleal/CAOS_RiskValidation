# 08. The validation tests

Module `riskvalidation.validation.irrbb`. Sources: dossier 05 section D.7 and the IRRBB acceptance of the ladder (reproduce the BCBS
worked example, the EVE sum-of-parts reconciliation, the stability of the behavioural parameters, the sensitivity "with and without
behavioural assumptions", the SVB lesson), dossier 01 H.2 and H.5, dossier 03 F.2. Each function returns one `TestResult`
(`riskvalidation.core.result`) that serialises into a row of `common/test-result`.

Acceptance gates are **Contraste internal gates**, stated as such, not regulatory thresholds. Each function records its parameters in
`policy_version` and in `extras["thresholds"]`, so that a view can explain a light. A test that cannot be evaluated returns the light
`not_evaluated` with its reason in `notes` and a null statistic, metric and p-value once serialised, never a fabricated number. The
tests are plain functions that do not read the registry or the default policy. They make no compliance claim: a green light means a
statistic fell inside a gate, nothing more.

| Test | Statistic | Light rule (Contraste internal gate) | Golden |
|---|---|---|---|
| `srp31_92_goldens` (`irrbb.srp31_92_goldens`) | the largest absolute deviation of four values from the printed ones, each in its own unit | green when every deviation is within `tolerance_scalar` (5e-4) and `tolerance_bp` (0.05 bp), red otherwise | green; statistic `0.0393` bp (the flattener, `-1.6393` against `-1.6`) |
| `cmf_annex1_goldens` (`irrbb.cmf_annex1_goldens`) | the number of failed cases of 19 (8 pairs, 5 Hoja 9 rows, the charge, the Annex 1 EVE golden in 3 parts, the Hoja 10 example in 2) | green when none fails (equality to `1e-12`; the two Hoja 10 values at the precision they are printed, `extras["thresholds"]` states both tolerances), red otherwise; a function that raises counts as a failed case | green; statistic `0` |
| `eve_sum_of_parts` (`irrbb.eve_sum_of_parts`) | the absolute residual of `sum eve(item) - whole_eve` (metric: the relative residual) | green when the relative residual is at most `rel_tol` (1e-9), red above | the three-item sheet at a flat 2 percent: green; `whole_eve = 12.9`: red |
| `nmd_parameter_stability` (`irrbb.nmd_parameter_stability`) | the largest sup-norm distance between consecutive windows (metric: the largest change of the weighted average maturity, years) | green at most `gap_amber` (0.10), amber at most `gap_red` (0.25), red above; fewer than two windows: not_evaluated | `0` green; `0.30` red; `0.15` amber |
| `eve_with_without_behavioural` (`irrbb.eve_with_without_behavioural`) | `(contractual - behavioural) / tier1` (metric: the difference in money) | green at most `diff_amber` (0.05), amber at most `diff_red` (0.15), red above; red whatever the size when the behavioural measure passes the outlier test and the contractual one breaches it | 10 against 20 on Tier 1 100: red (hidden breach); 10 against 12: green |
| `behavioural_change_at_breach` (`irrbb.behavioural_change_at_breach`) | the number of changes of a parameter at a date with a breach | red when any, green otherwise; fewer than two dates: not_evaluated | `[0.65, 0.75]` with a breach at the second: red; `[3.0, 3.0]`: green |
| `nii_out_of_sample` (`irrbb.nii_out_of_sample`) | `abs(projected - realised) / abs(realised)` (metric: `projected - realised`) | green at most `err_green` (0.10), amber at most `err_amber` (0.25), red above; realised None or zero: not_evaluated | 90 against 100: green; 70: red; None: not_evaluated |

## Equations

$$\text{stability}=\max_{a}\max_j\big|w_{a+1,j}-w_{a,j}\big|,\qquad
\text{rel. residual}=\frac{\big|\sum_i EVE(\text{item}_i)-EVE_{whole}\big|}{|EVE_{whole}|}$$

$$\text{difference}=\frac{M_{contractual}-M_{behavioural}}{\text{Tier 1}},\qquad
\text{error}=\frac{|\text{projected}-\text{realised}|}{|\text{realised}|}$$

Symbols: $w_{a,j}$ the replicating-portfolio weight of tranche $j$ in window $a$; $M$ the EVE risk measure; the outlier test of the
hidden-breach rule is $M/\text{Tier 1}>0.15$ (SRP31.82 to SRP31.83).

## What these tests do not claim

- A test reproducing the BCBS or CMF worked example shows the engine matches the printed values within the printed precision, not
  that a bank's measure is correct.
- The out-of-sample test compares one projection with one realised change over the 2022 hiking cycle; it is evidence for the
  validator, not a backtest of a distribution.
- The behavioural-change flag is a prompt for a human review (the SVB lesson: duration assumptions can be tuned to fit limits).

## Sources

- BCBS SRP31.82 to SRP31.83, SRP31.90 to SRP31.92 and SRP31.129, https://www.bis.org/basel_framework/chapter/SRP/31.htm (dossier 01 H).
- CMF RAN 21-13 Annex 1 (dossier 03 F.2), https://www.cmfchile.cl/portal/principal/623/articles-29808_doc_pdf.pdf.
- BCBS d368 (2016), https://www.bis.org/bcbs/publ/d368.htm; dossier 05 D.7.

## Limits

The gates are Contraste internal gates and policy parameters of the caller, not regulatory limits. The realised 2022 NII change is a
caller-supplied figure from the call reports; its absence gives not_evaluated, never a number. The UNVERIFIED items of pages 01 to 07
(the overnight bucket edge, the post-shock floor reading for UR, the SRP31.121 notional, the term-deposit remainder, the SIFMA text)
carry over to every test that reproduces them.
