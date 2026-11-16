# 08. Residual risk add-on (MAR23.1 to MAR23.8)

Module `riskvalidation.regulatory.rrao` (unit U19, backlog item BL-076, requirement RV-2010 in `docs/design/features/u19-frtb-cva/requirements.md`). Primary text:
the Basel Framework chapter MAR23, in force 2023-01-01, published 2020-03-27, read on bis.org on 2026-11-15:
https://www.bis.org/basel_framework/chapter/MAR/23.htm?inforce=20230101&published=20200327 (a paragraph `MAR23.x` is the anchor `#23.x`). The engine computes and states
no compliance claim.

## The formula (MAR23.8)

The residual risk add-on is calculated in addition to the other components of the standardised approach (MAR23.1). It is the simple sum of the gross notional amounts of the
instruments bearing residual risks, multiplied by a risk weight:

$$RRAO=1.0\%\sum_{i\in\text{exotic underlying}}N_i+0.1\%\sum_{i\in\text{other residual risks}}N_i$$

with the exotic underlying of MAR23.3 (longevity risk, weather, natural disasters, future realised volatility) at 1.0% and the other residual risks of MAR23.4 and
MAR23.5 (gap, correlation and behavioural risk of path-dependent, digital, basket, best-of, spread, Bermudan and quanto options, and the correlation trading portfolio
except its eligible hedges) at 0.1%. There is no offsetting of long and short: the notionals are gross and non-negative and the sums are plain, so a long of 100 and a
short of 100 give 200. A negative notional is refused, naming the field, because a net notional is not a gross one.

`rrao(exotic_gross_notionals, other_gross_notionals)` returns `RraoResult` with the two sums, the two charges and the total; `RW_EXOTIC` and `RW_OTHER` are
`RegulatoryParameter` records naming MAR23.8. Fixtures: RRAO-A, 200 with an exotic underlying and 1,000 with other residual risks give $2.0+1.0=3.0$; RRAO-C, two notionals of 100
with an exotic underlying give 2.0, not 0.

## Exclusions (MAR23.7)

`classify_instrument(exotic_underlying, other_residual_risk, back_to_back, listed_or_cleared)` returns "exotic", "other" or "excluded" and encodes only MAR23.7: in a
back-to-back transaction (an exact match with a third-party transaction) the instruments of both transactions are excluded; an instrument that is listed or eligible for
central clearing is excluded from the RRAO for other residual risks, and is included when its underlying is exotic (RRAO-B). An instrument flagged both exotic and bearing
other residual risks takes the exotic weight only; the caller decides whether both flags apply to it. The flags must be booleans.

## Stated limits

- The instrument-level judgement of MAR23.3 to MAR23.6 is the caller's: whether an underlying is exotic, whether a pay-off bears other residual risks (it cannot be written or
  perfectly replicated as a finite linear combination of vanilla options, MAR23.4), and whether a risk type alone leaves the instrument outside the add-on (cheapest-to-deliver
  option, smile risk, correlation risk of plain vanilla multi-underlying options, dividend risk, index instruments; MAR23.6). The engine takes the answers.
- The fund assumed exposed to exotic underlying exposures and other residual risks to the maximum extent of its mandate (MAR23.6 (5)) is the caller's.
- The scope of the RRAO does not change the risk factors of the delta, vega, curvature or default risk treatments (MAR23.8 (1)); nothing here touches them.
- A supervisor may add a conservative charge under Pillar 2 where the RRAO is not prudent enough (footnote 2 of MAR23.8); the engine does not.

## Tests

`tests/regulatory/test_rrao.py::test_rrao_formula_and_exclusions` (RV-2010): the weights, the sums, the refusals and the full truth table of the exclusions against the cases
written apart in the test.
