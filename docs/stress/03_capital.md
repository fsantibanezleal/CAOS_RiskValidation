# 03. Capital

Layer 3 of a macro stress test aggregates losses and revenues into capital ratios under a balance-sheet assumption (dossier 05 H.1).
`riskvalidation.stress.capital` is that engine on a static balance sheet for the CMF's IAPE (the self-assessment of capital adequacy),
with the rows of its Annex 3 summary. Money is one currency and never converted; shares are shares of RWA.

## The headline path and the PPNR row

The headline CET1 path is loss-only depletion (decision Q10): the CET1 at the jump-off less the cumulated stressed losses, no invented
revenue,

$$
\mathrm{CET1}_y = \mathrm{CET1}_0 - \sum_{j \le y} \ell_j,\qquad \ell_j \ge 0.
$$

Golden value (hand-computed): RWA 100, $\mathrm{CET1}_0 = 11.0$ (11 percent), $\ell = (1.0, 2.0, 1.5)$ give $(10.0, 8.0, 6.5)$ and ratios
$(0.10, 0.08, 0.065)$. A labelled row adds PPNR from a stated constant fee ratio and cost ratio, assumptions of the case:

$$
\text{fee} = f\cdot \mathrm{NII},\qquad \text{cost} = c\,(\mathrm{NII} + \text{fee}),\qquad \mathrm{PPNR} = \mathrm{NII} + \text{fee} - \text{cost},\qquad f, c \in [0, 1].
$$

The definitions of the two ratios are the contract's proposal; the Federal Reserve's efficiency-ratio element of the 2026 PPNR suite
(dossier 05 H.3) is the precedent for a cost ratio, and the dossiers give no model for PPNR (05 l.402). Golden value (computed):
$\mathrm{NII} = 660000$, $f = 0.25$, $c = 0.55$ give fee $165000$, cost $453750$ and PPNR $371250$. PPNR is a labelled row and never the
headline.

## The internal capital target

The CMF requires an internal effective-equity target `OI` of at least the sum, as shares of RWA, of (RAN 21-13 section I; dossier 03
l.152-169):

$$
\mathrm{OI} \;\ge\; 8\% + c_{\text{DSIB}} + c_{\text{P2,bank}} + c_{\text{CCoB}} + c_{\text{CCyB}} + c_{d},\qquad c_{\text{P2,bank}} \ge c_{\text{66 quinquies}}.
$$

`c_p2` is the bank's own estimate of capital for risks not covered by Pillar 1, which "a lo menos debe considerar el capital adicional
exigido por esta Comision de acuerdo con lo establecido en el articulo 66 quinquies"; `c_d` is a voluntary CET1 buffer sized so that CET1
absorbs a severe stress without breaching the legal minima, and "podria ser nula" if the buffers suffice (03 B.2). `internal_capital_target`
refuses a negative component, `c_p2` below `p2r` (the art. 66 quinquies requirement) and `p2r` above the 4 percent cap of RWA (03 l.52,
l.253-254: the additional requirement "no podra exceder el 4% de los activos ponderados por riesgo"); the cap bounds `p2r`, never `c_p2`,
which is the bank's own estimate. `c_p2` and `c_d` are labelled assumptions of the case (decision Q4).

Golden values (Santander Chile 2025-12-31, Form 20-F 2025, Ch\$ million, dossier 03 l.1324-1328; computed, Contraste arithmetic and not a
reported figure): RWA $41{,}715{,}467$, CET1 $4{,}601{,}923$ ($0.110317$ of RWA), effective equity $7{,}047{,}321$, Pillar 2 $52{,}144$ so
$0.125$ percent of RWA, countercyclical $0.50$, systemic $1.50$ and conservation $2.50$ percent. `c_dsib = 0.015`, `c_p2 = 0.00125`,
`c_ccob = 0.025`, `c_ccyb = 0.005`, `c_d = 0.0` give $\mathrm{OI} = 0.12625$; the effective-equity ratio is $0.168938$, so the headroom (Annex 3
row 12, $1.1 - 5$) is $0.042688$. The 4 percent cap is $0.04 \times 41{,}715{,}467 = 1{,}668{,}618.68$, above the $52{,}144$ requirement. The
Santander Chile full phase-in Pillar 2 percentage is UNVERIFIED (dossier 03 l.278).

## The Chilean capital stack

`chilean_cet1_stack` is the minimum CET1/RWA of the legal stack (dossier 03 G.4, l.911-920; RAN 21-13):

$$
\frac{\mathrm{CET1}}{\mathrm{APR}} \;\ge\; 4.5\% + \underbrace{2.5\%}_{\text{66 bis}} + \mathrm{RCC} + c_{\text{DSIB}} + \text{CET1 share of } c_{P2},
$$

the failure point of the reverse stress (decision Q11). The countercyclical buffer is refused above $0.025$ (the contract's bound). Golden
values (computed): `c_ccyb = 0.005`, `c_dsib = 0.015`, `p2_cet1 = 0.0` give $0.090000$, so the Santander Chile CET1 ratio $0.110317$ stands
$0.020317$ above it; with the whole Pillar 2 requirement in CET1 (`p2_cet1 = 0.00125`) the stack is $0.09125$.

## The three-year plan

`capital_plan` returns, for each of three years, `PlanYear` rows for the base (requirement `oi`), the stress without mitigation (CET1 from the
loss-only path, requirement `legal_minimum`) and the stress with mitigation (the loss-only path plus the cumulated mitigation effects,
requirement `legal_minimum`), the deficit being $\max(0, \text{requirement}\cdot \mathrm{RWA} - \mathrm{PE})$ (03 l.197-206):

$$
\mathrm{PE}_y = \mathrm{CET1}_y + \mathrm{AT1} + \mathrm{T2},\qquad \text{headroom}_y = \frac{\mathrm{PE}_y}{\mathrm{RWA}} - \text{requirement},\qquad
\mathrm{CET1}^{\text{mit}}_y = \mathrm{CET1}^{\text{loss-only}}_y + \sum_{j \le y} m_j.
$$

RWA is the same every year and AT1 and T2 stay at their jump-off values (a static balance sheet, paragraph 33: no capital measure after the
reference date). Mitigation effects are not cumulated by the caller and are not negative, so the mitigated PE is never below the
unmitigated PE. Golden value (hand-computed): RWA 100, $\mathrm{CET1}_0 = 11$, AT1 2, T2 3, losses $(1, 2, 1.5)$, mitigation $(0, 1, 1)$, legal
minimum $0.09625$: stress without mitigation CET1 $(10, 8, 6.5)$, PE $(15, 13, 11.5)$, PE ratio $(0.15, 0.13, 0.115)$, headroom $(0.05375, 0.03375,
0.01875)$, deficit $(0, 0, 0)$; with mitigation CET1 $(10, 9, 8.5)$.

## The P2G proxy

$$
\mathrm{P2G} \approx \max\!\Big(0,\; \max_{t \in \{1,2,3\}}\Big[\Big(\frac{\mathrm{CET1}}{\mathrm{RWA}}\Big)_{0} - \Big(\frac{\mathrm{CET1}}{\mathrm{RWA}}\Big)^{\text{severe}}_{t}\Big] - \text{usable buffers and mitigation}\Big)
$$

(dossier 03 l.1001-1005, FAQ 24: "the maximum deterioration inside the three-year projection horizon", referenced to CET1/RWA). A Contraste
approximation: the CMF calibration grid is not public and the value is not a P2G requirement; the document block carries the label
`contraste-approximation`. Golden value (hand-computed): $\mathrm{CET1}/\mathrm{RWA}$ of $0.11$ and severe ratios $(0.10, 0.08, 0.065)$ with usable $0.02$:
the peak depletion is $0.045$ and the proxy $0.025$.

## The Annex 3 summary

`annex3_summary` returns the 34 rows of the CMF summary (Circular 2.365 of 2025-07-08, Resolution 6801; addendum F7;
https://www.cmfchile.cl/portal/principal/623/articles-29808_doc_pdf.pdf, Annex 3 from pdf p.29) with the labels as printed
(`capital.ANNEX3_ROWS`), each with the four columns A (percent of RWA), B (amount), C (legal requirement) and D (internal measurement) as
cells `{"value", "status"}` of status `value`, `RI` (an immaterial risk), `not-to-be-filled` (market and operational risk under the
standardised method) or `not-applicable`. The row ids `2.T2`, `2.AT1` and `2.CET1` are the contract's ids for the three unnumbered
components of row 2. The form states thousand pesos; the document states millions.

Identities: row 3 column B is the RWA; row 5 is `oi`; row 7 is $8\% + $ row 7.1 $+$ row 7.2 (row 7.2 is the target's `p2r`, never `c_p2`); row 8 is row 6 less
row 7.2; row 9 is $c_{\text{CCoB}} + c_{\text{CCyB}}$; row 11 is $c_d$; row 12 is row 1.1 less row 5; row 10.1 is the largest fall of CET1/RWA from the jump-off in
the stress without mitigation and row 10.2 the same with mitigation, `not-applicable` when `mitigation_applies` is false (the form: "N/A if the
stress leads to no mitigation actions"). The legal requirement of rows 4 and 4.1 to 4.3 is 8 percent of the risk's RWA; the internal
measurement of the risks not covered by Pillar 1 (rows 4.4.1 to 4.8) comes from the caller, `None` meaning `RI`; row 6 sums the quantified ones.
Golden values (computed): capital start CET1 $0.11$, AT1 $0.0125$, T2 $0.025$ of RWA and OI $0.145$: row 1.1 is $0.1475$ and row 12 is $0.0025$; row 7 is
$0.09625$ for $0.015$ and $0.00125$.

## The EBA static balance sheet

`eba_static_balance_sheet(stage_balances, new_s3_flows)` applies paragraph 90 (EBA note, addendum F6): new stage 3 exposures move into the
stock of stage 3 and reduce stages 1 and 2 pro rata to their balances, the total exposure is constant, nothing leaves stage 3 (no cures,
charge-offs or write-offs) and nothing is replaced:

$$
S_{1,t} = S_{1,t-1}\Big(1 - \frac{f_t}{S_{1,t-1} + S_{2,t-1}}\Big),\qquad S_{2,t}\ \text{likewise},\qquad S_{3,t} = S_{3,t-1} + f_t.
$$

A flow larger than stages 1 plus 2 is refused. Golden value (hand-computed): stages $(80, 15, 5)$ and flows $(2, 3)$ give stage 3 of $(7, 10)$ and
total $(100, 100)$; stage 1 plus stage 2 is $(93, 90)$.

## Limits

- The engine computes the capital arithmetic of a case; it is not the CMF's IAPE report and states no compliance claim. The P2G proxy
  is not a P2G requirement, the reverse stress failure point is Contraste's, and `c_p2` and `c_d` are labelled assumptions.
- The headline path has no revenue: it is a depletion, not a forecast of capital. PPNR is a separate labelled row with assumed ratios.
- The static balance sheet holds RWA constant; a dynamic balance sheet (30-year climate horizons) is out of scope. The EBA reversion of
  adverse parameters over six years after the horizon (paragraph 133; the paragraph number 132 is UNVERIFIED) matters only for a bake that
  extends beyond three years.
- Rows 7, 7.2 and 8 read `InternalTarget.p2r` (the art. 66 quinquies requirement, at most `c_p2`); `c_p2`, the bank's own Pillar 2
  estimate, only enters the OI of row 5. `annex3_summary` refuses an RWA (the sum of `rwa_by_risk`) that differs from the capital
  plan's beyond 1e-9 relative.
- The Annex 3 sheet 3 capital-planning tables (the A, B.1, B.2 and C blocks by year) are carried by `capital_plan`, not by
  `annex3_summary`.

## Sources

- CMF, Recopilacion Actualizada de Normas, capitulo 21-13, Anexo 3, Circular 2.365 (2025-07-08), Resolution 6801,
  https://www.cmfchile.cl/portal/principal/623/articles-29808_doc_pdf.pdf
- EBA, 2025 EU-wide stress test, Methodological Note of 2025-01-20, paragraphs 24, 33, 34, 90, 91, 92,
  https://www.eba.europa.eu/sites/default/files/2025-01/0246e2f3-fa57-47d1-99f2-7a5b80cae509/2025%20EU-wide%20stress%20test%20-%20Methodological%20Note.pdf
- Banco Santander-Chile, Form 20-F 2025 (capital at 2025-12-31), dossier 03 M; the SEC statement that information on sec.gov may be copied,
  https://www.sec.gov/about/privacy-information
- Dossier 03 of the Contraste initiative (B.2, B.4, G.4, H.2, M) and dossier 05 (H.1, H.3).
