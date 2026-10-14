# 04. The standardised approach (OPE25)

Module `riskvalidation.oprisk.sma`. Source: dossier 01 section F, the Basel Framework standard OPE25 in force 2023-01-01 and
published 2024-07-05 (https://www.bis.org/committees/bcbs/basel-framework/standard/ope/25/inforce/2023-01-01/published/2024-07-05); the version in force 2027-01-01 only rewords
OPE25.10 and OPE25.27(5), the formulas are unchanged. The OPE10 item definitions of the business indicator were amended by BCBS
d610 (2026-03-23, future version in force 2029-04-01); their content was not compared: UNVERIFIED. The engine computes the formulas
and makes no compliance claim; disclosures stay the caller's.

## The formulas

$$\mathrm{ORC} = \mathrm{BIC}\times\mathrm{ILM},\qquad \mathrm{RWA}_{op} = 12.5\times\mathrm{ORC}\quad\text{(OPE25.2)}$$

$$\mathrm{BI} = \mathrm{ILDC} + \mathrm{SC} + \mathrm{FC}\quad\text{(OPE25.4)}$$

$$\mathrm{ILDC} = \min\Big(\overline{|II - IE|},\ 2.25\%\times\overline{IEA}\Big) + \overline{DI}\quad\text{(OPE25.5)}$$

$$\mathrm{SC} = \max\big(\overline{OOI}, \overline{OOE}\big) + \max\big(\overline{FI}, \overline{FE}\big),\qquad
\mathrm{FC} = \overline{|\text{Net P\&L}_{TB}|} + \overline{|\text{Net P\&L}_{BB}|}\quad\text{(OPE25.5)}$$

Symbols: $II, IE$ interest income and expense, $IEA$ interest-earning assets, $DI$ dividend income, $OOI, OOE$ other operating
income and expense, $FI, FE$ fee income and expense, $TB, BB$ trading and banking book; a bar is the three-year average over $t,
t-1, t-2$, and the absolute values of net items are taken **year by year before averaging** (OPE25.5 footnote 1). The functions
`ildc`, `service_component` and `financial_component` take the three years as arrays of length 3. Golden values: with $II = 10$,
$IE = 2$, $DI = 1$ in every year, $\mathrm{ILDC}$ is $\min(8, 2.25) + 1 = 3.25$ for $IEA = 100$ and $\min(8, 22.5) + 1 = 9.0$ for
$IEA = 1000$; $II = [10, 2, 10]$ against $IE = [2, 10, 2]$ gives a year-by-year mean of 8 (the netted value, the absolute value of
the difference of the averages, would be 2.667); $\mathrm{SC} = 5 + 4 = 9$; $\mathrm{FC}$ for trading $[10, -10, 10]$ and banking
$[-3, 3, -3]$ is $10 + 3 = 13$ (netted before the absolute value it would be 4.33); $\mathrm{BI} = 3.25 + 9 + 13 = 25.25$.

## The business indicator component

The marginal coefficients (OPE25.7 Table 1): bucket 1, $\mathrm{BI} \le$ EUR 1bn: 12 percent; bucket 2, 1bn $<\mathrm{BI}\le$ 30bn: 15
percent; bucket 3, above 30bn: 18 percent.

$$\mathrm{BIC} = 0.12\min(\mathrm{BI}, 1) + 0.15\big(\min(\mathrm{BI}, 30) - 1\big)^{+} + 0.18\,(\mathrm{BI} - 30)^{+}\quad(\text{EUR bn})$$

`bic_bucket` returns 1, 2 or 3 (1 up to and including 1e9, 2 up to and including 30e9). Golden value from OPE25.7 footnote 2:
$\mathrm{BI} = $ EUR 35bn gives $\mathrm{BIC} = 1\times12\% + 29\times15\% + 5\times18\% = $ **EUR 5.37bn**; also
$\mathrm{BIC}(0.5\text{bn}) = 0.06\text{bn}$, $\mathrm{BIC}(1\text{bn}) = 0.12\text{bn}$, $\mathrm{BIC}(30\text{bn}) = 4.47\text{bn}$.

## The loss component and the internal loss multiplier

$$\mathrm{ILM} = \ln\!\Big(\exp(1) - 1 + \Big(\frac{LC}{\mathrm{BIC}}\Big)^{0.8}\Big),\qquad LC = 15\times\text{average annual operational risk losses over the previous 10 years}\quad\text{(OPE25.8)}$$

ILM is 1 exactly when $LC = \mathrm{BIC}$ and above 1 when $LC > \mathrm{BIC}$ (**OPE25.9**). `loss_component` takes the annual losses
net of recoveries (FAQ1 to OPE25.10), oldest first, and uses the most recent `window_years` (default 10). Golden values:
ten annual losses of 10 give $LC = 150$; $[20, 10, 30, 40, 0]$ gives $15\times20 = 300$; with $\mathrm{BIC} = 150$ the ILM is 1.0 at
$LC = 150$, 1.241090 at $LC/\mathrm{BIC} = 2$, 0.829700 at 0.5, 0.541325 ($= \ln(e - 1)$) at $LC = 0$ and 1.220648 at $LC = 285$.

## Loss data rules and discretions

| Rule | Value | Paragraph |
|---|---|---|
| LC data window | 10 years of high-quality annual loss data; transitional minimum 5 years; **under 5 years capital is the BIC only** | OPE25.10, **OPE25.15** |
| Losses net of recoveries | yes | OPE25.10 FAQ1 |
| Bucket 1 banks | ILM = 1 (internal losses do not affect capital) unless national discretion allows inclusion | OPE25.11 |
| National discretion ILM = 1 | supervisors may set ILM = 1 for all banks in the jurisdiction; full disclosure still required | OPE25.11 |
| Collection threshold | EUR 20,000 (national discretion EUR 100,000 for buckets 2 and 3) | OPE25.18 |

`sma_capital(bi, annual_losses, bucket1_uses_losses, national_ilm_one, minimum_years, window_years)` applies the rules in this order:

1. fewer than `minimum_years` years of loss data: the capital is the BIC only (OPE25.10, OPE25.15), `basis = "bic-only"`, no LC and no
   ILM, a reason, discretion `transitional-bic-only` - **never an ILM computed on too little data**;
2. `national_ilm_one`: ILM = 1, discretion `national-ilm-one` (the disclosure stays the caller's);
3. bucket 1 and not `bucket1_uses_losses`: ILM = 1, discretion `bucket-1-ilm-one`;
4. otherwise ILM = `ilm(LC, BIC)`, discretion `none`.

In (2) and (3) the loss component is still computed and reported. `minimum_years` below 5 raises: 5 years is the transitional minimum
of OPE25.15, so an ILM on fewer years is unreachable through this function. Golden values: $\mathrm{BI} = 35$bn with ten annual losses
of $5.37\text{bn}/15 = 358$m gives $\mathrm{BIC} = LC = 5.37$bn, ILM $= 1$, $\mathrm{ORC} = 5.37$bn and $\mathrm{RWA} = 67.125$bn;
$\mathrm{BI} = 1.4$bn (bucket 2, BIC 0.18bn) with 3 annual losses is BIC only, ORC 0.18bn, with a reason; $\mathrm{BI} = 0.5$bn
(bucket 1, BIC 0.06bn) has ILM 1 under `bucket-1-ilm-one`, and with `bucket1_uses_losses=True` the ILM is computed.

## Limits

- The SMA formulas are the standard's; the engine does not test a bank's data against OPE25.12 to OPE25.20 (loss data quality,
  event types, reference dates, credit-related losses).
- Non-compliant loss data (capital at least 100 percent of BIC, a supervisory ILM above 1: OPE25.13) is a supervisory outcome, not
  computed here.
- The business indicator items are the caller's; their OPE10 definitions are not applied by the engine.
- The ILM is sensitive to a single large loss (page 05, `sma_ilm_response`); that is a property of the formula, shown and not hidden.

Sources: dossier 01 section F; Basel Framework OPE25, https://www.bis.org/committees/bcbs/basel-framework/standard/ope/25/inforce/2023-01-01/published/2024-07-05; Peters, Shevchenko,
Hassani and Chapelle (2016), J. Operational Risk 11(3):1-49, arXiv 1607.02319 (the critical response on the SMA's stability).
