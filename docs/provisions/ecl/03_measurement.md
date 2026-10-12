# 03. Measurement of the expected credit loss

Module: `riskvalidation.provisions.ecl`. Source: research dossier 02, sections A.5, A.6, B.3, B.4, B.5, B.6 and B.7; contract
`engine-provisions-api` 1.0.0.

## ECL as a sum of discounted marginal losses

IFRS 9 prescribes the measurement objective, not a formula; the sum-of-marginal-losses structure is the one described by GPPC
2.1.1.2 (IFRS 9 Appendix A "expected credit losses", B5.5.28, B5.5.43, B5.5.44). For exposure $i$ in scenario $s$:

$$\mathrm{ECL}_{i,s}(T) = \sum_{t=1}^{T} q^{D}_{i,s}(t)\; LGD_{i,s}(t)\; EAD_{i,s}(t)\; DF_i(t), \qquad DF_i(t) = (1 + r_i)^{-\delta(t)}.$$

*Equation 1. The scenario ECL, `ecl_scenario`.*

Symbols of Equation 1:

| Symbol | Meaning | Unit |
|---|---|---|
| $q^{D}(t)$ | default incidence of period $t$ (see [page 01](01_term-structures.md), Equation 2) | decimal per period |
| $LGD(t)$ | loss given default as a fraction of $EAD(t)$, closed in $[0, 1]$ | decimal |
| $EAD(t)$ | exposure at default if default occurs in period $t$ | money |
| $r_i$ | original effective interest rate (annual), at least 0 | decimal |
| $\delta(t)$ | years from the reporting date to the assumed loss date of period $t$ | years |
| $T$ | horizon: 12 months for the 12-month ECL; the expected life for the lifetime ECL | months |

`lgd` and `ead` are each a scalar, `(A,)` or `(A, T)`. `horizon_months` must be a member of `months` or at least the last
month (no proration of a straddling period, otherwise `ValueError`); a horizon beyond the last month is the lifetime ECL. The
result is non-negative, nondecreasing in the horizon and linear in EAD and in LGD.

12-month ECL (5.5.5, B5.5.43): $T$ is the periods in 12 months or the shorter remaining life; the LGD term is the lifetime cash
shortfall given default in the window (B5.5.43). Lifetime ECL (5.5.3): $T$ is the expected life (5.5.19, 5.5.20).

Golden value: $q = (0.01, 0.01)$ on months 6 and 12, $LGD = 0.4$, $EAD = 100000$, $r = 0$: horizon 12 gives 800.0, horizon 6
gives 400.0.

## Discounting

$$DF(t) = (1 + r)^{-\delta(t)/12}, \qquad \delta(t) = \begin{cases} m_t - \Delta_t/2 & \text{mid-period},\\ m_t & \text{end-of-period}.\end{cases}$$

*Equation 2. Discount factors, `discount_factors`.* $m_t$ is the end month of period $t$ and $\Delta_t$ its length in months
(the first period's length is `months[0]`). The convention is a documented model setting, not a requirement of the standard
(mid-period is Contraste's convention; end-of-period is the alternative dossier 02 B.3 lists, and it gives smaller discount
factors and so a smaller ECL, as the tests show). Rules of dossier 02 B.7:

- Discount to the reporting date, never to the default date (B5.5.44). Recoveries after default are discounted twice in
  effect: the LGD discounts them to the default date and $DF$ from the default date to the reporting date, both at the EIR.
- The EIR excludes ECL (Appendix A), so the risk premium of the contractual rate is not removed; a risk-free rate in place of
  the EIR overstates the ECL relative to the standard. The caller supplies the right EIR: current for floating rate (B5.4.5), the
  credit-adjusted EIR for POCI (B5.5.45), the EIR of the resulting asset for a loan commitment (B5.5.47).
- The interest revenue unwinding of the discount is part of the Stage 3 allowance reconciliation (IFRS 7 35H, see
  [page 04](04_overlays-and-reconciliation.md)).

Golden values at 6 per cent, months 3 and 6, mid-period: $DF = 1.06^{-0.125} = 0.992742848$ and $1.06^{-0.375} =
0.978386160$. A zero EIR gives exactly 1.

## Probability weighting and non-linearity

$$\mathrm{ECL}_i = \sum_{s=1}^{N_s} w_s\, \mathrm{ECL}_{i,s}, \qquad w_s \ge 0, \quad \sum_s w_s = 1, \quad N_s \ge 2.$$

*Equation 3. The probability-weighted ECL, `probability_weighted_ecl` (5.5.17(a), 5.5.18, B5.5.41, B5.5.42).* Weights lie in
$[0, 1]$ and sum to one within 1e-9; fewer than two scenarios is refused (B5.5.42). Permuting scenarios together with their
weights leaves the result unchanged; zero and maximal weights are valid.

Because $\mathrm{ECL}_{i,s}$ is a non-linear (typically convex in adverse directions) function of the macro path $m_s$,
$\sum_s w_s\, \mathrm{ECL}(m_s) \ne \mathrm{ECL}(\sum_s w_s m_s)$ in general; with convexity Jensen's inequality gives
$\mathbb{E}[\mathrm{ECL}(M)] \ge \mathrm{ECL}(\mathbb{E}[M])$. GPPC 2.8.1.2 draws the consequence: with a non-linear
relationship "more than one forward-looking scenario would need to be incorporated". The ECB (July 2024 report, Section 3)
criticises GDP-only legacy overlays because the approach "ignores the non-linear nature of expected credit losses, thus
systematically underestimating the mean impact".

$$NL = \frac{\mathrm{ECL}_{\mathrm{weighted}}}{\mathrm{ECL}(\bar m)} - 1.$$

*Equation 4. The non-linearity index, `non_linearity_index`.* It raises when the denominator is not positive; $NL \ge 0$ in
the convex regime and $0$ when the weighted and mean-path values coincide.

Worked illustration (dossier 02 B.4, computed with the one-factor conditional PD of
[page 01](01_term-structures.md); illustrative parameters): $PD^{\mathrm{TTC}} = 2\%$, $\rho = 0.12$, scenarios $z = +1$
($w = 0.25$), $z = 0$ ($w = 0.50$), $z = -2$ ($w = 0.25$). The probability-weighted PD is 2.681 per cent (0.026813479); the PD
at the weighted-average factor ($\bar z = -0.25$) is 1.800 per cent (0.017997448); the ratio is 1.49 (1.48985), so
$NL = 0.4898489$ (0.48985 rounded). With $LGD = 40\%$ the weighted 12-month loss rate is 1.07 per cent (0.010725392) against
0.72 per cent (0.007198979) for the "average scenario". Each is tested to 1e-8, the index to 1e-6.

Weight sensitivity (d350 para 36(a): "demonstrate and document how ECL estimates would alter with changes in scenarios"):
since the ECL is linear in the weights with staging frozen,

$$\frac{\partial\, \mathrm{ECL}}{\partial w_s} = \mathrm{ECL}_s - \mathrm{ECL}_{\mathrm{ref}}.$$

*Equation 5. The derivative when weight moves from a reference scenario to $s$, `weight_sensitivity`.* The entry of the
reference is 0 and the weighted ECL changes by $dw$ times the entry. Staging can change non-linearly with the weights when the
SICR test uses probability-weighted PDs (GPPC 2.7.2.10); the staging-frozen version reported here is the linear one. Golden
value: scenario ECLs $[10, 20, 60]$ with reference index 1 give $[-10, 0, 40]$. The 100 per cent single-scenario ECLs are
sensitivities, never the IFRS 9 measure.

## Stage 3, POCI, the simplified approach and commitments

$$\mathrm{ECL}_i = \max\Bigl(GCA_i - \sum_k \mathbb{E}[CF_{i,k}]\,(1+r_i)^{-m_k/12},\ 0\Bigr).$$

*Equation 6. Credit-impaired, non-POCI, `stage3_ecl` (B5.5.33).* The gross carrying amount minus the present value at the
original EIR of expected recoveries, $m_k$ the months from the reporting date; non-increasing in each recovery, equal to the
gross carrying amount with no recovery. Golden value: $GCA = 50000$, one recovery of 30000 at month 12, EIR 8 per cent gives
$50000 - 30000/1.08 = 22222.2222$ (22222.22 to 5e-3), tested to 1e-6.

POCI (5.5.13, B5.5.45): `poci_allowance` returns $\mathrm{ECL}^{\mathrm{life}}(\text{now}) - \mathrm{ECL}^{\mathrm{life}}(\text{initial})$,
both at the credit-adjusted EIR; it may be negative (5.5.14) and is never clipped. Golden values: 120 against 100 gives 20; 80
gives -20.

Simplified approach (5.5.15, 5.5.16, B5.5.35): `simplified_matrix_ecl` returns
$\min\bigl(\sum_b GCA_b,\ m \sum_b GCA_b\,\rho_b\bigr)$ over the past-due buckets with loss rates $\rho_b \in [0, 1]$ and a
forward-looking multiplier $m \ge 0$. Golden value: buckets $[1000, 500, 200]$ with rates $[0.01, 0.05, 0.20]$ give
$10 + 25 + 40 = 75$.

Loan commitments (B5.5.30, B5.5.31): `commitment_ead` returns $\mathrm{drawn} + CCF \cdot \mathrm{undrawn}$ with $CCF \in [0, 1]$.
Golden value: drawn 60, undrawn 40, CCF 0.5 gives 80. Financial guarantees (B5.5.32): the shortfall is the expected payment to
the holder less expected recoveries.

Revolving facilities (5.5.20, B5.5.38 to B5.5.40): the behavioural life is the period over which the bank is exposed and
losses "would not be mitigated by credit risk management actions"; an implementable estimator is a high quantile (or the
mean, as a documented choice) of the account lifetime distribution censored at closure or limit removal, computed by
Kaplan-Meier, with $EAD(t) = \mathrm{Drawn}_0 + CCF(t)(\mathrm{Limit}_0 - \mathrm{Drawn}_0)$. The quantile or mean is not fixed by
IFRS 9 (a policy setting): the engine takes the lifetime horizon as an input of `ecl_scenario`.

## The portfolio allowance

$$\mathrm{LA} = \sum_{i \in S1} \mathrm{ECL}_i(12\text{m}) + \sum_{i \in S2} \mathrm{ECL}_i(T^{\mathrm{life}}_i) + \sum_{i \in S3} \mathrm{ECL}^{D}_i + \sum_{i \in \mathrm{POCI}} \Delta\mathrm{ECL}_i + \sum_k \mathrm{OVL}_k.$$

*Equation 7. The loss allowance, `portfolio_allowance` (dossier 02 B.5).* The result is an `AllowanceBreakdown`: `by_stage`
always holds the four keys `stage1`, `stage2`, `stage3`, `poci` and carries model ECL only; `model_total` is their sum,
`overlay_total` the sum of the overlay amounts, and `total = model_total + overlay_total`. A POCI account is excluded from
stages 1 to 3 whatever its `stage` value and counts once, in `poci`; a POCI account with a non-finite `poci_delta` raises
`ValueError`. Overlays never enter `by_stage` (see [page 04](04_overlays-and-reconciliation.md)).

Golden values: `stage = [1, 2, 3]`, 12-month ECL $[10, 20, 30]$, lifetime $[15, 40, 45]$, Stage 3 $[0, 0, 25]$, one overlay of 5:
`by_stage = {stage1: 10, stage2: 40, stage3: 25, poci: 0}`, `model_total = 75`, `total = 80`. With `poci = [F, F, T]` and
`poci_delta = [nan, nan, 7]` the third account leaves Stage 3: `{stage1: 10, stage2: 40, stage3: 0, poci: 7}`.

The cliff effect at transfer is $\mathrm{ECL}_i(T^{\mathrm{life}}_i) - \mathrm{ECL}_i(12\text{m})$, which grows with the remaining
life and drives the procyclicality analysed by Abad and Suarez (2017).

## Limits

- The engine computes from parameters the caller supplies (PD, LGD, EAD, EIR, weights, horizons). It does not estimate them and
  does not establish that an ECL meets IFRS 9.
- A 100 per cent single-scenario ECL is a sensitivity or a component, never the IFRS 9 measure (5.5.17(a), B5.5.42).
- Behavioural lives, mean-reversion schedules and the discounting convention are policy settings, not requirements of the
  standard (dossier 02 I.5).

## References

- IASB, IFRS 9 Financial Instruments, 5.4, 5.5 and B5.5 (as endorsed by Commission Regulation (EU) 2016/2067).
  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R2067.
- GPPC (2016). The implementation of IFRS 9 impairment requirements by banks, 2.1.1.2, 2.7.2.10, 2.8.1.2.
  https://www.globalpublicpolicycommittee.org/wp-content/uploads/2022/12/gppc-ifrs9-implementation-considerations-20160617.pdf.
- BCBS (2015). Guidance on credit risk and accounting for expected credit losses (d350), para 36(a). https://www.bis.org/bcbs/publ/d350.htm.
- ECB Banking Supervision (2024). IFRS 9 overlays and model improvements for novel risks, Section 3. doi:10.2866/034127.
- Abad, J. and Suarez, J. (2017). Assessing the cyclical implications of IFRS 9: a recursive model. ESRB Occasional Paper No 12.
  doi:10.2849/2685.
- Tong, E.N.C., Mues, C., Brown, I. and Thomas, L.C. (2016). Exposure at default models with and without the credit conversion
  factor. EJOR 252(3), 910-920. doi:10.1016/j.ejor.2016.01.054.
- Botha, A. and Verster, T. (2025). arXiv:2507.15441.

Previous: [02. Staging](02_staging.md). Next: [04. Overlays and reconciliation](04_overlays-and-reconciliation.md).
