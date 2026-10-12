# 04. Overlays and reconciliation

Modules: `riskvalidation.provisions.overlays` and `riskvalidation.provisions.reconciliation`. Source: research dossier 02,
sections I.1 (items 9 and 10), I.2, C.6 and B.5; contract `engine-provisions-api` 1.0.0.

## Overlays as separate lines

Each overlay is an explicit line item: an id, a rationale in English and Spanish, a method (`simulation`, `scenario`,
`expert`), a target parameter (`pd`, `lgd`, `ead`, `staging-share`), the affected segment, a stage, a signed amount in the
document currency, a staging effect (`none`, `transfers-to-stage2`), a review date and the currency. An overlay is reported
separately in every output and never merged into the model ECL (ECB July 2024; BCBS d350 paras 50 and 51). The record
refuses a non-finite amount, a value outside its enum and an invalid ISO date or currency code; ids are unique within a run.

$$\mathrm{LA} = \underbrace{\sum_{S1}\mathrm{ECL}_{12m} + \sum_{S2}\mathrm{ECL}_{life} + \sum_{S3}\mathrm{ECL}^{D} + \sum_{\mathrm{POCI}}\Delta\mathrm{ECL}}_{\text{model ECL, by stage}} + \underbrace{\sum_k \mathrm{OVL}_k}_{\text{overlays, separate lines}}.$$

*Equation 1. The allowance with overlays as separate lines.* `total_allowance(model_ecl, overlays)` returns the model ECL plus
the sum of the overlay amounts and equals `portfolio_allowance(...).total` for the same inputs; the overlay amounts never
enter `by_stage`. `overlay_lines` returns one waterfall line per overlay in id order: `id = "overlay-<overlay id>"`,
`kind = "overlay"`, `overlay_id`, `amount`, `stage` (a null stage stays null); the sum of the line amounts equals the sum of
the overlay amounts exactly. A positive amount raises the allowance. `overlay_share(overlay_total, total_allowance)` is their
ratio and raises when the total allowance is not positive (no placeholder).

Stage rule (a convention of the contract, not an IFRS rule): an overlay with a stage is carried in that stage's allowance
closing; an overlay with no stage (portfolio level) is carried in the stage 1 closing, because the roll-forward has no
portfolio row.

Golden values: one overlay `OVL-RATES` of 1500 at stage 1 gives one line with `id = overlay-OVL-RATES`, `stage = stage1`,
`amount = 1500.0`; `overlay_share(5, 80) = 0.0625`; `total_allowance(75, [5]) = 80`.

An overlay is not a validated model. Climate risk entering only through GDP paths is not "captured" by the model (ECB July
2024); that is why the overlay's rationale, method and review date travel with it.

## The roll-forwards

IFRS 7 35H asks for a reconciliation of the loss allowance by stage and 35I for the drivers of the gross carrying amount.
A `RollForwardRow` holds the stage (`stage1` to `stage3` or `poci`), the opening amount, the signed movements as ordered
`(line_id, amount)` pairs (a positive amount raises the allowance), the closing amount and the residual.

$$r = \mathrm{closing} - \Bigl(\mathrm{opening} + \sum_j \mathrm{movement}_j\Bigr).$$

*Equation 2. The residual of a roll-forward row* (symbols: $r$ the residual, `closing`, `opening` and the signed `movement`
amounts of one stage row, in the document currency). It is always this difference itself, never clipped or adjusted; a row only
stores it, and the identity is judged by `check_identities`. `allowance_rollforward` refuses a negative opening or closing
allowance outside POCI (only POCI may be negative, 5.5.14); `gca_rollforward` accepts the line ids `originations`,
`transfer-1-2`, `transfer-2-3`, `transfer-2-1`, `transfer-3-2`, `repayments`, `write-off`, `derecognition`, `other` and refuses
a negative gross carrying amount. Each allowance `line_id` is the id of a waterfall line (the model lines `originations`,
`transfer-1-2`, ..., and `overlay-<overlay id>`), so the waterfall opens into the 35H table. Golden values: opening 1000 with
originations 200 and write-off -50 and closing 1150 gives residual 0; gross carrying amount opening 100000 with repayments
-5000 and originations 20000 and closing 115000 gives residual 0.

## The identity check

Tolerance basis, one definition for every identity: an identity closes when its absolute difference is at most
$10^{-6}\max(1, G)$, with $G$ the sum of the absolute closing amounts of the gross carrying amount rows (Contraste internal
gate, dossier 02 I.2; not a regulatory threshold).

| Identity name | What it states |
|---|---|
| `allowance-rollforward:<stage>` | the allowance row closes (Equation 2) |
| `gca-rollforward:<stage>` | the gross carrying amount row closes |
| `waterfall-closes` | waterfall opening plus the movement lines equals the closing line |
| `waterfall-equals-stage-closings` | the sum of the stage closings equals the waterfall closing |
| `waterfall-lines-equal-movements` | every movement `line_id` is a waterfall line id and the line amount is the sum of the movements of that id over the stages |
| `stage-closing-equals-ecl:<stage>` | the allowance closing of a stage equals the stage ECL plus the overlays carried in that stage |

`check_identities` returns the names of the identities that fail (an empty list when all close) and never raises on a
mismatch; `identity_residuals` returns the absolute residual of each. The roll-forward identities are exact in principle;
the last one compares two stated amounts that a document rounds, so it is judged on the same tolerance. Golden value: the C06
reporting-date fixture of the product line returns `[]`: its stage 2 closing 1758.76 differs from its ECL 1758.7575 by 0.0025,
inside the tolerance $10^{-6} \times 230000 = 0.23$; moving one movement by 1.0 returns a non-empty list.

## Limits

- The engine reconciles amounts the caller supplies; its output is not an IFRS 7 disclosure and says nothing about whether a
  document discloses what IFRS 7 requires.
- Overlays are the caller's judgements; the engine reports them separately and validates only the shape of the record.

## References

- IASB, IFRS 7 Financial Instruments: Disclosures, 35H and 35I (as amended by IFRS 9, endorsed by Commission Regulation (EU)
  2016/2067). https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R2067.
- ECB Banking Supervision (2024). IFRS 9 overlays and model improvements for novel risks. doi:10.2866/034127.
- ECB Banking Supervision, letter SSM-2020-0744, Identification and measurement of credit risk in the context of the
  coronavirus (COVID-19) pandemic, 2020-12-04.
  https://www.bankingsupervision.europa.eu/press/letterstobanks/shared/pdf/2020/ssm.2020_letter_credit_risk_identification_measurement~734f2a0b84.en.pdf.
- BCBS (2015). Guidance on credit risk and accounting for expected credit losses (d350), paras 50 and 51.
  https://www.bis.org/bcbs/publ/d350.htm.

Previous: [03. Measurement](03_measurement.md). Next: [05. The validation battery](05_validation-battery.md).
