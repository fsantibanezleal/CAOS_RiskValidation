# 02. Shocks

Module `riskvalidation.irrbb.shocks`. Sources: dossier 01 section H.2 (the six shapes, SRP31.90 to SRP31.92), H.3 (the size tables)
and H.5 (the EU floor); dossier 03 section F.2 (RAN 21-13 Annex 1, Hoja 9). Primary texts: BCBS SRP31
https://www.bis.org/basel_framework/chapter/SRP/31.htm; BCBS d368 https://www.bis.org/bcbs/publ/d368.htm; BCBS d578
https://www.bis.org/publications/202407-standards-recalibration-shocks-interest-rate-risk-banking-book.pdf.

## The six shapes

With $t_k$ the bucket midpoint and $x=4$ years (SRP31.91 footnote 8, "for most currencies ... unless otherwise determined by
national supervisors"):

$$\alpha_{short}(t_k)=e^{-t_k/x},\qquad \alpha_{long}(t_k)=1-e^{-t_k/x}$$

$$\Delta S_{parallel}=\pm\bar S_{parallel},\qquad \Delta S_{short}=\pm\bar S_{short}\,\alpha_{short},\qquad
|\Delta S_{long}|=\bar S_{long}\,\alpha_{long}$$

$$\Delta S_{steepener}=-0.65\,|\Delta S_{short}|+0.9\,|\Delta S_{long}|,\qquad
\Delta S_{flattener}=+0.8\,|\Delta S_{short}|-0.6\,|\Delta S_{long}|$$

Symbols: $\bar S_{parallel}$, $\bar S_{short}$, $\bar S_{long}$ the sizes in basis points of the currency; $|\Delta S_{short}|=
\bar S_{short}\,\alpha_{short}$. The long shock enters only inside the two rotations. `shock_curve(shock, parallel_bp, short_bp,
long_bp, midpoints, x)` returns the basis-point shift at each midpoint; a negative size or an unknown shock name raises.

Golden (SRP31.92, dossier 01 H.2): at $t=3.5$ years and $x=4$, $\alpha_{short}=0.417$ (exactly `0.416862`) and $\alpha_{long}=
0.583$; with $\bar S_{short}=\bar S_{long}=100$ bp the short-up shock is `+41.7` bp, the steepener `+25.4` bp and the flattener
`-1.6` bp (unrounded `41.686`, `25.386`, `-1.639`; the tolerance is the printed precision, `0.05` bp). The USD 2024 parallel-up
shock is `200` bp at all 19 midpoints.

## The size tables

`shock_table(calibration)` returns `{currency: (parallel_bp, short_bp, long_bp)}`:

- `"2024"`: SRP31.90 Table 2, in force 2026-01-01, data January 2000 to December 2023, 99.9th percentile of local rolling
  six-month average absolute changes, rounding to 25 bp (d578 introduction and SRP98.60). 21 currencies, no CLP.
- `"2016"`: BCBS d368 Annex 2 Table 1, data 2000 to 2015, 99th percentile of global factors, rounding to 50 bp. The 2016 PDF text
  layer interleaves rows; the row assignment was checked against the column count and against the EUR 200/250/100 and USD
  200/300/150 values, so the 2016 table is verified with that caveat. 21 currencies, no CLP.
- `"cmf-annex1"`: RAN 21-13 Annex 1 Hoja 9, read from page images (dossier 03 F.2): `canasta-1` (200, 300, 150), `canasta-2`
  (400, 500, 300), `CLP` (300, 400, 200), `UR-nii` (300, None, None), `UR-eve` (200, 250, 150); a dash in the source is `None`.
  "Canasta 1 y 2" are the currency baskets of the market-risk chapter.

Golden: `shock_table("2024")["USD"] = (200, 300, 225)`, `shock_table("2016")["USD"] = (200, 300, 150)`,
`shock_table("2016")["EUR"] = (200, 250, 100)`, `shock_table("cmf-annex1")["UR-nii"] = (300, None, None)`. The module data is
compared cell by cell with the rows of dossier 01 H.3 in `tests/irrbb/test_irrbb_shocks.py`.

## Shock sizes and their origin

`shock_size(currency, calibration, estimate=None)` returns a `ShockSize` with the origin `srp31-table-2024`, `srp31-table-2016`,
`cmf-annex1` or `contraste-estimate`. **A size is never invented.** CLP is in neither BCBS table (dossier 01 H.3): CLP under
`"2024"` or `"2016"` raises `ValueError`, unless the caller supplies `estimate = (parallel_bp, short_bp, long_bp, label, method)`,
with a bilingual label (non-empty `en` and `es`) and a method (for example the SRP98.56 to SRP98.63 methodology on BCCh yield
data); the result then has the origin `contraste-estimate`, which is a Contraste estimate and not a regulatory parameter. An
estimate for a currency that has a tabled size is refused, so a tabled size never carries a label. `CLF` takes the `UR-eve` row of
the CMF table (200, 250, 150); `UR-nii` has no short and long size and cannot become a `ShockSize`.

Golden: `shock_size("USD", "2024")` is (200, 300, 225) with origin `srp31-table-2024`; `shock_size("CLP", "cmf-annex1")` is
(300, 400, 200); `shock_size("CLF", "cmf-annex1")` is (200, 250, 150).

## Post-shock floors

$$r_i(t_k)=\max\!\big(r_0(t_k)+\Delta S/10^{4},\ \min(\mathrm{floor},\,r_0(t_k))\big)$$

SRP31.93 lets supervisors floor post-shock rates at a level not above zero; RAN 21-13 Annex 1 floors at `-0.02` for UR rates and
`0.0` for other currencies "or the pre-shock rate if lower". `apply_floor(base_rates, shock_bp, floor)` takes a decimal, an array
(a maturity-dependent floor) or `None`; a floor never lifts a rate above its pre-shock value, and a rate already below the floor
is left at its pre-shock value after a downward shock. Golden: `apply_floor(0.004, -300, 0.0) = 0.0`; `(-0.001, -300, 0.0) =
-0.001`; `(0.03, -300, 0.0) = 0.0`; `(0.004, -300, -0.02) = -0.02` (the unfloored `-0.026` is floored).

`eu_floor(maturity)` is the maturity-dependent floor of Commission Delegated Regulation (EU) 2024/856 Article 3(7), `-0.015` at
immediate maturity rising `0.0003` per year to `0` at 50 years (dossier 01 H.5; verified via EUR-Lex on 2026-10-04,
https://eur-lex.europa.eu/eli/reg_del/2024/856/oj/eng). Golden: `-0.015`, `-0.012`, `0.0`, `0.0` at 0, 10, 50, 60 years.

## Limits

- Whether "or the pre-shock rate if lower" applies to the `-2%` UR floor as well as to the `0%` floor of other currencies is
  UNVERIFIED: the dossier sentence puts the parenthesis after both floors (dossier 03 F.2); the Hoja 9 page image was not
  re-read. The engine applies the `min(floor, r0)` form to every floor.
- The 2016 table carries the row-assignment caveat above. CLP is in neither BCBS table.
- The CRD article number for the EU EVE outlier test is UNVERIFIED (dossier 01 H.5).
