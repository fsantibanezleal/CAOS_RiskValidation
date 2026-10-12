# 01. PD term structures

Module: `riskvalidation.provisions.term_structure`. Source: research dossier 02 of the Contraste initiative, sections B.1,
B.2, B.4, C.1 and C.3; the contract `engine-provisions-api` 1.0.0 gives the signatures, units and golden values.

IFRS 9 prescribes the measurement objective, not a formula (GPPC 2.1.1.1: IFRS 9 "generally does not prescribe particular
detailed methods"). A lifetime ECL needs the probability of default period by period. This page states the identities that
connect the hazard, the survival function, the marginal and the cumulative PD, the competing exit of prepayment, the forward
PD used by the staging tests, and the one-factor conversion from a through-the-cycle PD to a point-in-time PD.

## Notation

Periods are $t = 1, \dots, T$ after the reporting date; period $t$ covers $(t-1, t]$ and ends at the integer month
`months[t]`. A leading batch dimension is allowed everywhere: arrays are `(..., T)`.

Symbols used on this page:

| Symbol | Meaning | Unit |
|---|---|---|
| $\tau$ | default time | |
| $S(t) = P(\tau > t)$ | survival function, $S(0) = 1$ | decimal |
| $F(t) = 1 - S(t)$ | cumulative PD from the reporting date | decimal |
| $h(t)$ | conditional PD of period $t$ given survival to $t-1$ (the discrete hazard; GPPC calls it the "marginal probability of default") | decimal per period |
| $q(t)$ | unconditional (marginal) PD of period $t$, $q(t) = S(t-1) - S(t)$ | decimal per period |

Contraste reserves "marginal" for $q(t)$ and always prints both names. PDs at an estimate site are open, $0 < \mathrm{pd} < 1$:
a function raises `ValueError` outside the open interval. PDs are per period, not annualised.

## The identities

$$S(t) = \prod_{u=1}^{t} \bigl(1 - h(u)\bigr), \qquad q(t) = S(t-1)\,h(t), \qquad F(t) = \sum_{u=1}^{t} q(u) = 1 - S(t), \qquad h(t) = \frac{q(t)}{S(t-1)}.$$

*Equation 1. The term-structure identities (Botha and Verster 2025, eqs. 7 to 10).* `from_hazard`, `from_marginal` and
`from_cumulative` build a frozen `TermStructure` (months, hazard, survival, marginal, cumulative) from any one of the three
and satisfy these identities to 1e-12 for every leading shape. Round trips agree to 1e-12. Refusals: a hazard or marginal
outside $(0, 1)$; a marginal that sums to one or more; a cumulative that is not strictly increasing inside $(0, 1)$ (the
monotone lifetime PD is an acceptance row of the product); months that are not strictly increasing integers.

Golden value (constant $h = 0.01$ on months 3 and 6): $S = (0.99,\ 0.9801)$, $q = (0.01,\ 0.0099)$, $F = (0.01,\ 0.0199)$.

## Competing exits

With a competing exit (full prepayment or maturity before default), let $h^{D}(t)$ and $h^{P}(t)$ be the cause-specific
conditional probabilities of default and prepayment in period $t$ given the loan is still active.

$$S^{\mathrm{all}}(t) = \prod_{u \le t}\bigl(1 - h^{D}(u) - h^{P}(u)\bigr), \qquad q^{D}(t) = S^{\mathrm{all}}(t-1)\,h^{D}(t), \qquad F^{D}(t) = \sum_{u \le t} q^{D}(u).$$

*Equation 2. Default incidence with prepayment.* $F^{D}$ is the cumulative incidence of default; it is not $1 - S^{\mathrm{all}}$,
which is the probability of default or prepayment. The result of `competing_risks` is a `CompetingRisks` record and never a
`TermStructure`. With `hazard_prepay = 0` the arrays equal those of `from_hazard(hazard_default)`; prepayment never raises
$F^{D}$. Ignoring prepayment overstates the lifetime ECL: IFRS 9 requires "the effect of expected prepayments" (B5.5.51) and
the credit loss definition includes prepayment options (Appendix A). The formalism is that of Fine and Gray (1999) and of the
cause-specific hazard literature; Bellini (2019) devotes a chapter to prepayments and competing risks.

Golden value ($h^{D} = 0.01$, $h^{P} = 0.02$, two periods): $S^{\mathrm{all}} = (0.97,\ 0.9409)$, $q^{D} = (0.01,\ 0.0097)$,
$F^{D} = (0.01,\ 0.0197)$, against $1 - S^{\mathrm{all}} = (0.03,\ 0.0591)$.

## Forward PD and annualisation

$$PD(a, b) = \frac{S(a) - S(b)}{S(a)} \quad\text{(term structure)}, \qquad PD^{D}(a, b) = \frac{F^{D}(b) - F^{D}(a)}{S^{\mathrm{all}}(a)} \quad\text{(competing risks)}.$$

*Equation 3. The forward PD over $(a, b]$ given survival to $a$.* `forward_pd` takes months that are 0 or members of the
structure; from month 0 it equals the cumulative PD. For competing risks it is the default probability given survival of
both exits to $a$, never the default-or-prepayment probability. Golden values: constant $h = 0.01$ per 3-month period,
`forward_pd(ts, 3, 9)` $= 1 - 0.99^2 = 0.0199$; competing risks, `forward_pd(cr, 0, 6)` $= 0.0197$.

$$\overline{PD} = 1 - (1 - PD)^{1/H_{\mathrm{yrs}}}.$$

*Equation 4. Annualisation for the SICR comparison.* The EBA monitoring reports compare "annualised lifetime PD" values
(EBA/Rep/2021/35); the exact annualisation formula the EBA used was not stated in the passages read, so this geometric form
is Contraste's documented convention (UNVERIFIED as the EBA's). Golden value: `annualise_pd(0.19, 2.0)` $= 0.1$.

## From a through-the-cycle PD to a point-in-time PD

Model (BCBS 2005 explanatory note, section 4.2, describing the adaptation of Merton's model used for the IRB function;
Vasicek 2002; Gordy 2003): the obligor asset return is $A_i = \sqrt{\rho}\,Z + \sqrt{1-\rho}\,\varepsilon_i$ with $Z$ and
$\varepsilon_i$ independent $N(0,1)$, and the obligor defaults if $A_i < \Phi^{-1}(PD^{\mathrm{TTC}})$.

$$PD^{\mathrm{PIT}}(z) = \Phi\!\left(\frac{\Phi^{-1}(PD^{\mathrm{TTC}}) - \sqrt{\rho}\, z}{\sqrt{1-\rho}}\right).$$

*Equation 5. The conditional (point-in-time) PD, `ttc_to_pit`.* $\rho$ is the asset correlation, open in $(0, 1)$, and
$z$ the systematic factor in standard-normal units (a low $z$ is a bad state of the economy). The function is strictly
decreasing in $z$; $\mathbb{E}_Z[PD^{\mathrm{PIT}}(Z)] = PD^{\mathrm{TTC}}$, while $PD^{\mathrm{PIT}}(0) < PD^{\mathrm{TTC}}$ for
$PD^{\mathrm{TTC}} < 0.5$.

$$\hat Z_{g,t} = \frac{\Phi^{-1}(PD^{\mathrm{TTC}}_g) - \sqrt{1-\rho}\;\Phi^{-1}(DR_{g,t})}{\sqrt{\rho}}.$$

*Equation 6. Factor extraction from an observed default rate, `implied_z`.* It inverts Equation 5 under the large
homogeneous segment assumption: `implied_z(ttc_to_pit(p, r, z), p, r)` returns $z$ to 1e-9.

Worked example (dossier 02 B.4; parameters illustrative, not calibrated): $PD^{\mathrm{TTC}} = 2\%$, $\rho = 0.12$. The
conditional PDs are 0.526 per cent at $z = +1$ (0.005255059), 1.429 per cent at $z = 0$ (0.014287387) and 7.342 per cent at
$z = -2$ (0.073424082), each tested to 1e-8. The value at $z = 0$ is the median, not the mean, PD (1.43 per cent against
the 2 per cent unconditional mean), a frequent source of bias when a "base" scenario is treated as the expectation.

`weighted_pit_pd(pd_ttc, rho, z, weights)` returns $\sum_s w_s\,PD^{\mathrm{PIT}}(z_s)$ over at least two scenarios whose
weights lie in $[0, 1]$ and sum to one within 1e-9 (B5.5.42); the weighted value is compared with the PD at the weighted
mean factor in [page 03](03_measurement.md). `gauss_hermite_pit_pd` integrates $PD^{\mathrm{PIT}}$ over $Z \sim N(0,1)$ with
probabilists' Gauss-Hermite nodes as the non-linearity cross check (dossier 02 I.1 item 6): it equals $PD^{\mathrm{TTC}}$ to
1e-9 from 32 nodes (golden: `gauss_hermite_pit_pd(0.02, 0.12, 64)` $= 0.02$).

## Limits

- The functions compute from a PD curve the caller supplies; they do not estimate one. Rating migration, survival and vintage
  methods (dossier 02 C.2 to C.4) produce the hazards; this module only converts and combines them.
- The IRB supervisory correlations are calibrated for capital, not for point-in-time dynamics, and should not be used blindly
  as $\rho$ (Contraste position, dossier 02 C.1).
- The engine states no compliance claim: nothing here is an IFRS 9 requirement beyond what the dossier sections cite.

## References

- Botha, A. and Verster, T. (2025). Approaches for modelling the term-structure of default risk under IFRS 9: a tutorial using
  discrete-time survival analysis. arXiv:2507.15441.
- Fine, J.P. and Gray, R.J. (1999). A proportional hazards model for the subdistribution of a competing risk. JASA 94(446),
  496-509. doi:10.1080/01621459.1999.10474144.
- Bellini, T. (2019). IFRS 9 and CECL Credit Risk Modelling and Validation. Academic Press. doi:10.1016/C2017-0-02756-8.
- Gordy, M.B. (2003). A risk-factor model foundation for ratings-based bank capital rules. Journal of Financial Intermediation
  12(3), 199-232. doi:10.1016/S1042-9573(03)00040-8.
- BCBS (2005). An Explanatory Note on the Basel II IRB Risk Weight Functions.
  https://www.bis.org/publications/explanatory-note-basel-ii-irb-risk-weight-functions.pdf.
- Vasicek, O. (2002). The distribution of loan portfolio value. Risk 15(12), December 2002.
- EBA (2021). IFRS 9 implementation by EU institutions, Monitoring report, EBA/Rep/2021/35. doi:10.2853/631076.
- IASB, IFRS 9 Financial Instruments, as endorsed by Commission Regulation (EU) 2016/2067.
  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R2067.

Next: [02. Staging](02_staging.md). Index: [README](README.md).
