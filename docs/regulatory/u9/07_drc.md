# 07. Default risk charge for non-securitisations (MAR22.1 to MAR22.26)

Module `riskvalidation.regulatory.drc` (unit U19, backlog item BL-076, requirements RV-2006 to RV-2009 in `docs/design/features/u19-frtb-cva/requirements.md`).
Primary text: the Basel Framework chapter MAR22, default risk capital requirement, in force 2023-01-01, published 2020-03-27, read on bis.org on 2026-11-15:
https://www.bis.org/basel_framework/chapter/MAR/22.htm?inforce=20230101&published=20200327 (a paragraph `MAR22.x` is the anchor `#22.x`). The engine computes and
states no compliance claim.

## The steps (MAR22.3)

1. The gross jump-to-default (JTD) risk of each exposure, separately. 2. The offsetting of long and short exposures to the same obligor, where permissible, into a net
long and a net short per obligor. 3. The allocation to the three buckets. 4. Within a bucket the hedge benefit ratio, the risk weights and the sum. 5. The plain sum
of the buckets.

## Gross JTD (MAR22.9 to MAR22.18)

A long exposure is one that results in a loss in the case of a default, whatever the instrument (a sold put option on a bond is long, MAR22.10). With $N$ the
notional (positive for a long, negative for a short) and $P$ the cumulative profit and loss already taken (a loss negative, MAR22.13):

$$JTD_{long}=\max(LGD\cdot N+P,\ 0),\qquad JTD_{short}=\min(LGD\cdot N+P,\ 0)\qquad\text{(MAR22.11)}$$

weighted by $\min(1,\max(0.25,\ \text{maturity in years}))$ (MAR22.15, MAR22.18): the JTD of a position of one year or more is not scaled, a six month position is
weighted by one half, and a position under three months counts as three months. The LGD is 100% for equity and non-senior debt, 75% for senior debt and 25% for
covered bonds (MAR22.12); a price not linked to the recovery rate of the defaulter takes no LGD; legal terms that allow unwinding with no exposure to default risk give
zero (MAR22.13). Cash equity is given the maturity of three months or more than one year at the caller's choice (MAR22.16, and the FAQ that no value between is allowed).

`gross_jtd(direction, notional, pnl, lgd, maturity_years, recovery_linked=True, unwind_allowed=False)` returns the weighted value. Fixtures: the Basel FAQ under MAR22.15
and MAR22.16 (index futures of -10 and the underlying of +10, both counted as three months: $\tfrac14\cdot10-\tfrac14\cdot10=0$); DERIVED DRC-B, a long senior bond of
notional 100 with no P&L gives $0.75\cdot100=75$, and 37.5 at a six month maturity.

## Net JTD and offsetting (MAR22.19 to MAR22.21)

The gross JTD of a long and a short exposure to the same obligor may be offset where the short has the same or lower seniority than the long: "a short exposure in an
equity may offset a long exposure in a bond, but a short exposure in a bond cannot offset a long exposure in the equity" (MAR22.19). Positions under one year are weighted
by their maturity over the one-year horizon, so a three month short against a long of more than one year offsets a quarter of its size (MAR22.19, MAR22.20); that weight is the
one `gross_jtd` already applies (footnote 1 of MAR22.15 states that the scaling is of the gross JTD), so `net_jtd` does not apply it again.

`net_jtd(exposures)` takes `DrcExposure(obligor, bucket, rating, seniority, jtd)` records with the signed weighted gross JTD and returns one `NetJtd` per obligor with
`net_long` (at least 0) and `net_short` (at most 0). Both can remain: a short in a bond cannot offset a long in the equity of the same obligor. The offset taken is the
largest the rule allows: shorts are matched from the most senior to the most junior, each against the most junior longs it may offset (the tests check the totals against a linear
program). The seniority ladder of the engine is `SENIORITY` = equity, non-senior debt, senior debt, covered bond (the most junior first), the engine's ordering of the four classes of
MAR22.12; a caller may pass an integer rank instead (higher is more senior). One obligor has one bucket and one rating category; a clash is refused.

## Hedge benefit ratio, bucket and total (MAR22.22 to MAR22.26)

The three buckets are corporates, sovereigns, and local governments and municipalities (MAR22.22). The default risk weights by credit quality category are those of
MAR22.24 Table 2: AAA 0.5%, AA 2%, A 3%, BBB 6%, BB 15%, B 30%, CCC 50%, unrated 15%, defaulted 100%, for all three buckets. With the sums over the credit quality
categories:

$$HBR=\frac{\sum NetJTD_{long}}{\sum NetJTD_{long}+\sum|NetJTD_{short}|},\qquad DRC_b=\max\Big(\sum_{i\in long}RW_i\,NetJTD_i-HBR\sum_{i\in short}RW_i\,|NetJTD_i|,\ 0\Big)$$

where the bucket requirement is MAR22.25 and the hedge benefit ratio MAR22.23, and the requirement is the plain sum of the $DRC_b$, with no hedging between buckets (MAR22.26) and no diversification between the non-securitisation, the securitisation
(non-CTP) and the securitisation (CTP) requirements (MAR22.4). `hedge_benefit_ratio`, `bucket_drc(long_by_rating, short_by_rating)`, `drc_non_securitisation(buckets)` returning a
`DrcResult` with the charge of each bucket, their hedge benefit ratios and the sum, and `drc_from_exposures(exposures)` for the whole path.

Reading taken where the text is silent: the hedge benefit ratio is defined as 0 when both sums are 0 (the text leaves the ratio undefined there); the bucket's charge is then 0.
Fixtures (DERIVED from the transcription, the Basel text has no number for the bucket formula): DRC-C, a net long of 100 in BBB and a net short of -50 in BB give
$HBR=2/3$ and $DRC=\max(6-\tfrac23\cdot7.5,0)=1.0$; DRC-D, a long of 10 in AAA and a short of -100 in CCC give 0; DRC-E, a charge of 1.0 in each of the three buckets gives 3.0.

## Stated limits

- Securitisations are outside: the non-correlation trading portfolio and the correlation trading portfolio (MAR22.27 to MAR22.45), which the Framework keeps apart
  (MAR22.4), are not computed; the engine returns the non-securitisation requirement alone.
- The look-through of index and multi-underlying instruments (MAR22.5 and its FAQ), the guaranteed-bond rule (MAR22.19, CRE22.71 and CRE22.73), the national discretion of a zero
  weight on sovereigns, public sector entities and multilateral development banks (MAR22.7), the funds of MAR22.8, the removal of the CTP hedges (MAR22.6) and the choice of
  the seniority, rating category, bucket, maturity and LGD of each exposure are the caller's.

## Tests

`tests/regulatory/test_drc.py`: `test_parameters_pinned_to_mar22` (RV-2006, Table 2 against its verbatim rows), `test_gross_jtd_formula_and_the_basel_faq` (RV-2007),
`test_offsetting_by_seniority` (RV-2008), `test_hbr_bucket_and_total` (RV-2009), `test_stated_limits_and_page`.
