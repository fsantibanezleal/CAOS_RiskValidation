# 03. Downturn LGD

Module `riskvalidation.lgd_ead.downturn`. Source: dossier 04 F.4 (cross-reference only).

The regulatory definitions are in EBA/RTS/2018/04 (RTS on the nature, severity and duration of an economic downturn) and
EBA/GL/2019/03 (Guidelines for the estimation of LGD appropriate for an economic downturn,
https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/policy-statement/2021/october/downturn-lgd-estimation-gl.pdf),
an addendum to EBA/GL/2017/16. The wording of their three calibration approaches is not in the dossier (UNVERIFIED), so this
module claims no regulatory approach and the product labels the add-on an illustration. The Basel II explanatory note explains
why no supervisory LGD mapping function was adopted. The academic method cited is Calabrese (2014), "Downturn Loss Given
Default: Mixture distribution estimation", EJOR 237(1):271-277, doi:10.1016/j.ejor.2014.01.043 (abstract not retrieved:
content UNVERIFIED); systematic co-movement of PD and LGD: Altman, Brady, Resti and Sironi (2005), doi:10.1086/497044.

## The period

`identify_downturn(years, indicator, threshold, higher_is_worse=True)` marks a year as a downturn year when the indicator is
strictly above the threshold (strictly below it when `higher_is_worse` is false); a tie is not a downturn and a NaN raises. The
indicator (for example the unemployment rate in percent) and its source are recorded by the caller in the document. Golden, on
a constructed path (not BLS data): years 2006 to 2011, indicator `[4.6, 4.6, 5.8, 9.3, 9.6, 8.9]`, threshold 7.0 gives
`[False, False, False, True, True, True]`.

## The add-on

With cohort mean LGD $L_c$, weights $w_c$ (the number of defaults) and the downturn set $D$:

$$\mathrm{LR}=\frac{\sum_c w_c L_c}{\sum_c w_c},\qquad
\mathrm{DT}=\frac{\sum_{c\in D} w_c L_c}{\sum_{c\in D} w_c},\qquad \mathrm{add\text{-}on}=\mathrm{DT}-\mathrm{LR}.$$

Symbols: $L_c$ the cohort LGD, $w_c>0$ the number of defaults of the cohort, $D$ the cohorts flagged as downturn. The
add-on may be negative (the downturn cohorts lost less than average) and is reported as is; the result does not depend on the
order of the cohorts. Golden: `cohort_lgd = [0.30, 0.60, 0.40]`, weights `[100, 50, 50]`, mask `[False, True, False]` gives
long-run 0.40, downturn 0.60, add-on 0.20.

## Coverage

`coverage_shortfall(estimate, realised)` returns `(covered, shortfall)` with `covered = estimate >= realised - 1e-12` and
`shortfall = max(0, realised - estimate)`. Goldens: `(0.55, 0.60)` gives `(False, 0.05)`; `(0.70, 0.68)` gives `(True, 0.0)`.
The statistical version is `validation.lgd_ead.downturn_coverage` (page 06).

## Limits

The add-on is Contraste's cohort-based one, not an EBA approach. The period depends on the indicator and the threshold the
caller supplies. The engine does not decide how many downturn years a regulatory calibration needs.
