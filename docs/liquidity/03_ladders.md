# 03. The ladders: contractual gaps, behavioural outflows, the net liquidity position, the survival period and the cliff effect

Module `riskvalidation.liquidity.ladder`. Sources: dossier 05 section E.1 lines 234 to 236 (ECB LiST 2019 final results annex and
methodological note), E.5 rungs 1 and 2; dossier 01 G.3 lines 642 and 643 (SRP50, the contractual maturity mismatch tool); the U11
research addendum L1.6 (paras 84 to 89, 128 to 132; annex slides 8 and 24). Contract `engine-liquidity-api` 1.0.0.

## Equations

The contractual ladder is the static gap per bucket `b` and its cumulative sum (E.5 rung 1):

$$g_b = I_b - O_b,\qquad G_b=\sum_{c\le b} g_c .$$

The behavioural ladder takes the outflow of segment `j` on day `d` from the stock left the day before (see page 01):

$$O_{j,d}=A_j\prod_{s<d}(1-q_{j,s})\;q_{j,d},\qquad \sum_d O_{j,d}=A_j\,F_{j}(H).$$

The net liquidity position (NLP), the survival period (SP), the minimum and the cliff effect, with `CBC(t)` the counterbalancing capacity,
`O(s)` and `I(s)` the stressed outflows and inflows since the reference date and `TA` total assets:

$$NLP(t)=CBC(t)-\sum_{s\le t}\big(O(s)-I(s)\big),\qquad SP=\min\{t:\ NLP(t)<0\},\qquad
\text{cliff}=\frac{NLP(35)-NLP(30)}{TA}.$$

The survival period is "the first day in which the NLP turns negative" (annex slide 8; note paras 87 and 130). A zero NLP is not negative.
The cliff effect is a diagnostic of LCR window optimisation and is negative when the position drops.

Symbols: `B` contractual buckets, `S` segments, `H` ladder days, `A_j` opening balance of segment `j`, days are integers counted from 1 (day 1
is the first relevant day after the valuation date), money in USD thousands.

## The survival period is never a number when the bank survives

`survival_period` returns a frozen `SurvivalResult(survives, day, horizon)`. A position that never turns negative returns `survives = True`,
`day = None` and `horizon = H`: "survives the horizon" with the horizon (the note reports "N/A" when the position stays positive for the
six months). It is never a number, and a validation test that compares survival periods counts `None` as longer than any day.
`minimum_nlp` returns the minimum and the first day it is reached.

## Golden values

| Call | Result |
|---|---|
| `contractual_gaps([10, 20, 30], [40, 10, 5])` | gap `[-30, 10, 25]`, cumulative `[-30, -20, 5]` |
| `behavioural_outflows([100], [[0.1, 0.1]])` | `[[10, 9]]` (the second day is `90 x 0.1`) |
| `net_liquidity_position([100]*4, [30, 40, 50, 60], [0, 10, 0, 0])` | `[70, 40, -10, -70]` (hand computed) |
| `survival_period([70, 40, -10, -70])` | `day = 3`, `survives = False` |
| `survival_period([5, 4, 3])`; `[1, 0, 0]` | `survives = True`, `day = None`, `horizon = 3`; `survives = True` |
| `minimum_nlp([70, 40, -10, -70])`; `[5, 3, 3]` | `(-70, 4)`; `(3, 2)` |
| `cliff_effect`: `NLP(30)=100`, `NLP(35)=40`, `TA=1000`; `NLP(35)=95` | `-0.06`; `-0.005` |

The behavioural rows are checked against the day-by-day stock and against `opening x diff(cumulative path)`; the NLP against a loop.

## Conventions

`near` and `far` are the named parameter `cliff_days` (default 30 and 35). A horizon shorter than `far` raises `ValueError` naming it; the
document then holds the cliff null with its reason. The upper edges of the contractual maturity buckets are a convention (SRP50 names none
for the mismatch tool).

## Limits

- The LiST note does not say whether day 30 and day 35 of the cliff effect count calendar or relevant days (slide 8 does not say; slides
  15 and 24 chart the NLP against calendar days). UNVERIFIED: the named parameter `cliff_days` carries the basis.
- The note prints no closed-form NLP formula (its rows are Excel formulas, paras 84 and 128); the formula above is the definition of
  annex slide 8 and dossier 05 line 234.
- The engine computes: the survival period is a result of the stated scenario and capacity, never a forecast and never a compliance claim.

## Sources

- ECB LiST 2019 final results annex, slides 8, 14, 15, 24 and 32:
  https://www.bankingsupervision.europa.eu/press/pr/date/2019/html/ssm.pr191007_annex~537c259b6d.en.pdf
- ECB LiST 2019 methodological note, paras 84 to 89 and 128 to 132:
  https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.ECB_sensitivity_analysis_of_liquidity_risk-stress_test_2019-methodological_note_20190206~6771e88926.en.pdf
- BCBS, Basel Framework SRP50 (monitoring tools, contractual maturity mismatch):
  https://www.bis.org/committees/bcbs/basel-framework/standard/srp/50/inforce/2019-12-15/published/2019-12-15
