# 02. Stage allocation

Module: `riskvalidation.provisions.staging`. Source: research dossier 02, sections B.5, A.3, A.4 and I.5; contract
`engine-provisions-api` 1.0.0.

IFRS 9 puts each exposure in one of three stages (5.5.3, 5.5.5, 5.5.9 to 5.5.11, B5.5.9 to B5.5.11, B5.5.37). Stage 1
carries the 12-month ECL; Stage 2 (a significant increase in credit risk, SICR) and Stage 3 (credit-impaired) carry the
lifetime ECL. This page states how the engine allocates stages, and which parts of the rule are in the standard.

## What is in the standard and what is policy

Only two presumptions are in IFRS 9: more than 30 days past due is a rebuttable presumption of a significant increase in
credit risk, a backstop (5.5.11), and 90 days past due is a rebuttable presumption of default (B5.5.37). The threefold
increase of the annualised lifetime PD is a supervisory expectation (EBA/Rep/2023/36 para 32), not IFRS 9; in the engine it
is the configurable `threefold_ratio` of `StagingPolicy`, never a constant of the code. The multiplier schedule, the absolute
floor, the probation length and the low credit risk threshold are policy parameters the caller supplies.

## The policy

| Field | Meaning | Rule |
|---|---|---|
| `multiplier_schedule` | pairs (upper bound of the origination PD, multiplier $m$) | bounds strictly increasing in $(0, 1]$; $m \ge 1$ and not rising as the origination PD rises |
| `absolute_floor` | $\delta$, absolute annualised PD increase floor | in $[0, 1]$ |
| `threefold_ratio` | backstop ratio | above 1; default 3.0 |
| `dpd_backstop` | days past due; trigger when `dpd` is above it | default 30 (5.5.11) |
| `default_dpd` | Stage 3 when `dpd` is at or above it | default 90 (B5.5.37), above the backstop |
| `probation_months` | cure probation | at least 0 |
| `low_credit_risk_pd` | optional exemption threshold on the current annualised PD | `None` or in $(0, 1)$ |

## The quantitative rule

For an exposure originated at $t_0$ with remaining expected life $H$ at the reporting date $t_r$ (age $a = t_r - t_0$), the
origination expectation for the same window is the forward PD of Equation 3 on the term structure estimated at origination
(GPPC 2.7.2.9); the current estimate comes from the term structure at $t_r$, probability-weighted over scenarios. Both are
annualised (Equation 4 of [page 01](01_term-structures.md)).

$$\mathrm{SICR}^{\mathrm{quant}}_i = \mathbb{1}\Bigl[\frac{\overline{PD}^{\mathrm{rep}}_i}{\overline{PD}^{\mathrm{orig}}_i} \ge m\bigl(\overline{PD}^{\mathrm{orig}}_i\bigr) \;\wedge\; \overline{PD}^{\mathrm{rep}}_i - \overline{PD}^{\mathrm{orig}}_i \ge \delta\Bigr] \;\vee\; \mathbb{1}\Bigl[\frac{\overline{PD}^{\mathrm{rep}}_i}{\overline{PD}^{\mathrm{orig}}_i} \ge 3\Bigr].$$

*Equation 1. The quantitative SICR rule (dossier 02 B.5; BCBS d350 A26).* In the engine the constant 3 is
`threefold_ratio`. The multiplier $m(\cdot)$ is read from the schedule at the first upper bound at or above the origination
PD, and the schedule must cover the largest origination PD. The absolute floor $\delta$ prevents trivial transfers at very low
PD. `sicr_quantitative` is elementwise (row $i$ depends on row $i$ only), monotone in the current PD, false when the two PDs
are equal, and never applies the low credit risk exemption.

Symbols: $\overline{PD}^{\mathrm{orig}}$ and $\overline{PD}^{\mathrm{rep}}$ are the annualised origination and current PDs
(`pd_origination_annualised`, `pd_current_annualised`), open in $(0, 1)$; $m$ the multiplier; $\delta$ the floor.

Golden values (schedule $((0.01, 4.0), (0.05, 2.5), (0.5, 1.5))$, floor 0.005, threefold 3.0): origination 0.02 with current
0.05 gives true (ratio 2.5, difference 0.03); current 0.049 gives false; origination 0.001 with current 0.003 gives true by
the threefold backstop.

## Allocation

`allocate_stages` returns, per account, the stage (int8), the trigger and whether the account is exempt.

$$\mathrm{Stage}_i = \begin{cases} 3 & D_i \ \text{or}\ \mathrm{dpd}_i \ge \text{default\_dpd},\\ 2 & \text{any trigger fires},\\ 1 & \text{otherwise.}\end{cases}$$

*Equation 2. Stage allocation (dossier 02 B.5).* Precedence of the recorded trigger: `default`, `dpd-backstop`,
`forbearance`, `watch-list`, `collective`, `relative-absolute`, `threefold`; the trigger is `none` if and only if the stage is
1. Forbearance (d350 A24(f); EBA definition-of-default guidelines) and the watch list (GPPC 2.7.2.15; d350 A30) are
qualitative triggers the caller supplies as flags; the collective trigger is the mask of the top-down transfer below. Row $i$
of every output depends on row $i$ of the inputs only: changing an unrelated account changes no stage (the isolation test of
dossier 02 I.2).

Low credit risk exemption (5.5.10; d350 A48 to A51; EBA/Rep/2023/36 paras 36 to 39): an account whose current annualised PD is
at or below `low_credit_risk_pd` is not flagged by the two quantitative triggers (`relative-absolute`, `threefold`). The 30
days past due backstop, forbearance, watch-list, collective and default triggers still apply (5.5.11, B5.5.37).
`lcr_exempt[i]` is true if and only if the account is Stage 1 and the quantitative test would have flagged it without the
exemption; the SICR effectiveness report counts them (see [page 05](05_validation-battery.md)).

Golden values: `dpd = 31` with unremarkable PDs gives Stage 2 `dpd-backstop`; `dpd = 30` gives Stage 1; `dpd = 90` gives
Stage 3 `default`. With `low_credit_risk_pd = 0.03`, origination 0.001, current 0.003 and `dpd = 0` the account is Stage 1,
trigger `none`, exempt; the same account at `dpd = 31` is Stage 2 `dpd-backstop`, not exempt.

## Cure probation

Return from Stage 2 to Stage 1 is symmetric (5.5.7), but a configurable probation applies to forborne or SICR exposures,
following B5.5.27 ("consistently good payment behaviour over a period of time"). `apply_cure_probation` keeps in Stage 2, with
trigger `cure-probation`, an account that is Stage 1 now, was Stage 2 before and cured fewer than `probation_months` months
ago; every other account is unchanged, and with zero probation the output equals the current stage. The length is a policy
parameter, not a number fixed by IFRS 9. The required argument `now` (the `StageResult` of `allocate_stages` for the current
stages) carries the triggers of the current allocation so that the unchanged Stage 2 and 3 accounts keep theirs, always inside
`TRIGGERS`. Golden values with a 6-month probation: 3 months since cure gives
Stage 2 `cure-probation`; 6 months gives Stage 1.

## Collective transfer

When a risk driver is known only at segment level, a proportion $\pi_g$ of segment $g$ is moved to Stage 2 (B5.5.6, IE39; ECB
letter of 2020-04-01; ECB July 2024).

$$\mathrm{amount}_g = \pi_g \sum_{i \in g} \mathrm{GCA}_i.$$

*Equation 3. The target Stage 2 money of segment $g$, `collective_transfer`.* It reports only: the amount is never added
to any allowance. The caller turns each target into accounts (the accounts of segment $g$ in descending current PD, ties by
account id, until the cumulative gross carrying amount first reaches the target) and passes them as the `collective` mask of
`allocate_stages`, so the ECL of the transfer reaches the allowance only through the stage of those accounts; applying the
transferred money again would double count. A segment absent from the shares raises `KeyError`. Golden value: segment total
100000 with $\pi = 0.1$ gives 10000.

## Limits

- The engine allocates stages from parameters the caller supplies; it does not establish that a staging policy meets IFRS 9
  or any supervisor's view.
- The multipliers, floors, probation lengths and horizons are not IFRS 9 requirements (dossier 02 I.5).

## References

- IASB, IFRS 9 Financial Instruments, 5.5 and B5.5, as endorsed by Commission Regulation (EU) 2016/2067.
  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R2067.
- EBA (2023). IFRS 9 implementation by EU institutions, 2023 Monitoring report, EBA/Rep/2023/36. doi:10.2853/269836.
- EBA (2021). IFRS 9 implementation by EU institutions, Monitoring report, EBA/Rep/2021/35. doi:10.2853/631076.
- BCBS (2015). Guidance on credit risk and accounting for expected credit losses (d350). https://www.bis.org/bcbs/publ/d350.htm.
- GPPC (2016). The implementation of IFRS 9 impairment requirements by banks.
  https://www.globalpublicpolicycommittee.org/wp-content/uploads/2022/12/gppc-ifrs9-implementation-considerations-20160617.pdf.
- ECB Banking Supervision (2024). IFRS 9 overlays and model improvements for novel risks. doi:10.2866/034127.

Previous: [01. Term structures](01_term-structures.md). Next: [03. Measurement](03_measurement.md).
