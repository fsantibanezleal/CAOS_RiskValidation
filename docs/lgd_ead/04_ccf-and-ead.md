# 04. CCF and EAD

Module `riskvalidation.lgd_ead.ccf`. Source: dossier 04 G.1 to G.3.

## The convention

With a reference date $t_r$ before the default $t_d$, the drawn amount $E(t_r)$ and the limit $L(t_r)$, the CCF of this engine
is the **EU convention** (`convention.name = "eu-undrawn"`): the share of the undrawn amount drawn by default,

$$\mathrm{CCF}=\frac{\mathrm{EAD}-E(t_r)}{L(t_r)-E(t_r)},\qquad \mathrm{EAD}=E(t_r)+\mathrm{CCF}\,\big(L(t_r)-E(t_r)\big).$$

Qi (2009), "Exposure at default of unsecured credit cards", OCC Economics Working Paper 2009-2
(https://www.occ.gov/publications-and-resources/publications/economics/working-papers-archived/pub-econ-working-paper-2009-2.pdf)
uses US terminology: the ratio above is the **loan equivalent** (LEQ); there "CCF" means EAD as a share of the outstanding
balance and "EADF" EAD as a share of the credit line. `to_us_terms(drawn, limit, ead)` returns `(leq, ccf_us, eadf)` so that a
page can show both conventions and never mix them. Golden: drawn 600, limit 1000, EAD 900 gives `(0.75, 1.5, 0.9)`.

Symbols: $E$ drawn, $L$ limit, $t_r$ reference month, $t_d$ default month, undrawn $=L-E$. Money is a float in the portfolio
currency; the engine never converts currencies.

## Reference-date designs

The three designs are Moral (2006), "EAD estimates for facilities with explicit limits", in Engelmann and Rauhmeier (eds.),
The Basel II Risk Parameters, Springer, pp. 197-242, doi:10.1007/3-540-33087-9_10, as summarised in dossier 04 G.1 (the chapter
was not read: detail UNVERIFIED). `reference_indices` returns the months on the facility path: fixed horizon $[t_d-h]$;
cohort $[\text{start of the calendar cohort containing }t_d]$; variable horizon $t_d-h+k\cdot\text{step}$ for
$k=0..n-1$. A negative index (the reference precedes the first observation) or one not before the default raises. Golden, with
$t_d=21$: `[9]`, `[12]` (cohort start 12) and `[9, 12, 15, 18]`.

## Observations and aggregation

`ccf_observations(drawn, limit, ead, reference_index)` returns `(ccf, undrawn)`. Where the undrawn amount is not positive (a
fully drawn or overdrawn line, dossier 04 G.2) the CCF is undefined and returned as NaN, never as 0 or 1. Values below 0
(repayments, limit cuts) and above 1 (overlimit, capitalised interest and fees, limit increases) are returned as they come.
Golden: limit 1000, EAD 900, drawn at months 9, 12, 15, 18 of `[420, 500, 560, 640]` gives undrawn `[580, 500, 440, 360]` and
CCF `[0.827586, 0.8, 0.772727, 0.722222]`: the fixed-horizon design reads 0.827586 and the cohort design 0.8.
`aggregate_ccf` gives the unweighted mean of the finite CCFs (0.780634) or the pooled ratio
$\sum(\mathrm{CCF}\cdot\text{undrawn})/\sum\text{undrawn}=\sum(\mathrm{EAD}-E)/\sum(L-E)=1480/1880=0.787234$.

## A CCF outside [0, 1]

`treat_ccf(ccf, treatment)` implements the common treatments (dossier 04 G.2): `none` keeps every value (flag it with
`range_flag`); `censor` drops the observations outside [0, 1]; `floor-cap` clips to [0, 1] (it biases EAD); `bounded-scale`
returns the values clipped as the input of the fractional logit, and the estimate is then the model's predicted mean, not a
clip-and-average. NaN entries are dropped first and counted; the counts below 0 and above 1 are taken on the input whatever the
treatment, so a treatment never hides how many observations lay outside. Golden: `[-0.1, 0.0, 0.2, 0.6, 1.3]` has mean 0.4
(none), 0.266667 (censor, two dropped) and 0.36 (floor-cap).

`small_undrawn_rule(drawn, limit, ead, threshold)` switches to EAD over the limit where the undrawn amount is below the
threshold (Qi 2009 switches below USD 50; here the threshold is a parameter in the currency of the amounts): `(980, 1000,
990)` with threshold 50 is switched with ratio 0.99; `(600, 1000, 900)` is not, with ratio 0.75. `ccf_bias(estimate, truth)` is
the signed difference.

## EAD models

`fit_direct_ead` is a fractional logit of the EADF $=\mathrm{EAD}/L$ on a design with an intercept; an EADF outside [0, 1] is
clipped for the fit and counted in `extras` (`n_clipped_below`, `n_clipped_above`), never silently. `fit_utilisation_change`
fits by least squares the change of utilisation $(E(t_d)-E(t_r))/L(t_r)$, a ratio-free model with no undrawn amount in the
denominator (Leow and Crook 2013, dossier 04 G.2 and G.3). `predict_ead` returns $L\cdot\widehat{\mathrm{EADF}}$ (inside
$[0,L]$) or $E(t_r)+L(t_r)\,\widehat{\delta}$ (not clipped; flag it in the document). Goldens: the direct fit on EADF
`[0, 0.5, 0.5, 1]` has coefficients `[-1.098612, 2.197225]` and predicts 750 for limit 1000 in the second group; the utilisation
change fit on drawn `[400, 500, 600]` to `[700, 800, 900]` over limit 1000 has coefficient 0.3 and predicts 900 from drawn 600.
`ead_from_ccf` is the inverse of the observation (golden `(600, 1000, 0.75)` gives 900).

## Sources and limits

Tong, Mues, Brown and Thomas (2016), doi:10.1016/j.ejor.2016.01.054, propose direct EAD models with a zero-adjusted gamma
mixture; that mixture is **not** in this version of the engine (it would be a minor version). Leow and Crook (2013), "Estimation
of credit card exposure at default (EAD)", University of Edinburgh Credit Research Centre
(https://www.crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Estimation-of-credit-card-exposure-at-default-EAD-Leow-and-Crook.pdf),
and Jacobs (2010), doi:10.2139/ssrn.1149407, are the other sources of dossier 04 G. Whether the EBA's CCF guidelines consulted in
2025 are final is UNVERIFIED and not implemented. CCF models are fitted on defaulted facilities only, so the sample is small and
the reference-date design is a material model choice that must be justified. The engine states no compliance claim.
