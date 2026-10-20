# 05. FRTB sensitivities-based method with GIRR, and the liquidity-horizon cascade (MAR21, MAR33)

Module `riskvalidation.regulatory.frtb`. Sources: dossier 01 sections D.2 (MAR21), D.3 (MAR33) and M.1, the research addendum R7 (the GIRR parameters, MAR21.42 to
MAR21.50, MAR21.99) and the contract `engine-regulatory-api` 1.0.0. Primary texts: BCBS MAR21 in force 2023-01-01
https://www.bis.org/basel_framework/chapter/MAR/21.htm?inforce=20230101&published=20260323 (the page the addendum read) and BCBS MAR33, in force 2023-01-01
https://www.bis.org/basel_framework/chapter/MAR/33.htm (the chapter page of the same pattern; this unit transcribes MAR33 from dossier 01 D.3). The existing
`riskvalidation.market.frtb` holds the backtesting and PLA statistics; the two modules do not import each other.

## Bucket and cross-bucket aggregation (MAR21.4)

$$WS_k=RW_k\,s_k,\qquad K_b=\sqrt{\max\Big(0,\ \sum_kWS_k^2+\sum_k\sum_{l\ne k}\rho_{kl}\,WS_kWS_l\Big)},\qquad S_b=\sum_{k\in b}WS_k$$

$$\text{charge}=\sqrt{\max\Big(0,\ \sum_bK_b^2+\sum_b\sum_{c\ne b}\gamma_{bc}\,S_bS_c\Big)}$$

If the radicand of the cross-bucket sum is negative, $S_b$ is replaced by $\max\big[\min\big(\sum_{k\in b}WS_k,\ K_b\big),\ -K_b\big]$ for every bucket and the sum recomputed
(MAR21.4(5)(b)). Symbols: $s_k$ the sensitivity to risk factor $k$, $RW_k$ its risk weight, $WS_k$ the weighted sensitivity, $\rho_{kl}$ the within-bucket
correlation, $\gamma_{bc}$ the cross-bucket correlation. `bucket_charge(ws, rho, scenario)` is $K_b$ (one factor gives $|WS|$; perfectly correlated factors give
$|\sum WS|$; never negative) and `cross_bucket_charge(k_b, s_b, gamma, scenario)` the second formula (one bucket gives its $K_b$; never negative); the diagonal of $\rho$ and
$\gamma$ is ignored. `cross_bucket_charge_detail(k_b, s_b, gamma, scenario)` takes the same arguments and returns
`CrossBucket(charge, fallback_used)`: `charge` equals `cross_bucket_charge` and `fallback_used` is true exactly when the MAR21.4(5)(b) fallback applied (the radicand with the plain
$S_b$ was negative).

## The three correlation scenarios (MAR21.6, MAR21.7)

$$\rho^{medium}=\rho,\qquad \rho^{high}=\min(1.25\,\rho,\ 1),\qquad \rho^{low}=\max(2\rho-1,\ 0.75\,\rho)\qquad\text{(the same for }\gamma\text{)}$$

`scenario_correlation(rho, scenario)` applies this elementwise. The SBM requirement is the maximum over the scenarios of delta plus vega plus curvature (MAR21.7): `sbm_capital(delta,
vega, curvature)` takes, per scenario (`low`, `medium`, `high`), the sum over the risk classes of each measure and returns `SbmCapital(by_scenario, capital, scenario)`.

## GIRR delta (addendum R7)

The GIRR delta sensitivity is a one basis point bump (MAR21.19):

$$s_{k}=\frac{V\big(r_k+0.0001\big)-V\big(r_k\big)}{0.0001}$$

that is the change in value for a 1 bp shift DIVIDED BY 0.0001, not the PV01 per basis point: a PV01 of 4,500 USD per basis point is $s_k=4.5\times10^{7}$ USD per unit of rate. One bucket is
one currency. `girr_parameters(specified_currency=True)` returns a `GirrParameters`:

- tenors $0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30$ years (MAR21.8);
- risk weights (MAR21.42 Table 1) $1.7, 1.7, 1.6, 1.3, 1.2, 1.1, 1.1, 1.1, 1.1, 1.1$ percent; for a specified currency (EUR, USD, GBP, AUD, JPY, SEK, CAD and the bank's domestic
  reporting currency) they may be divided by $\sqrt{2}$ at the bank's discretion (MAR21.44), so the applied weights are Table 1 divided by sqrt(2) and `sqrt2_applied` says so
  (for example 1.1 percent becomes 0.7778 percent; a named convention, with the undivided weights kept in `risk_weights_table`);
- the 10 by 10 correlation matrix of MAR21.46 Table 2 (below), 0.999 for the same tenor on other curves (MAR21.45), and $\gamma=0.5$ across buckets (MAR21.50);
- the curvature shock 0.017, the highest delta risk weight applied to all tenors (MAR21.99), also divided by sqrt(2) for a specified currency (0.012021, DERIVED in the addendum).

MAR21.46 Table 2, as decimals of the printed percentages (rows and columns in the tenor order above):

| T | 0.25 | 0.5 | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1.000 | 0.970 | 0.914 | 0.811 | 0.719 | 0.566 | 0.400 | 0.400 | 0.400 | 0.400 |
| 0.5 | 0.970 | 1.000 | 0.970 | 0.914 | 0.861 | 0.763 | 0.566 | 0.419 | 0.400 | 0.400 |
| 1 | 0.914 | 0.970 | 1.000 | 0.970 | 0.942 | 0.887 | 0.763 | 0.657 | 0.566 | 0.419 |
| 2 | 0.811 | 0.914 | 0.970 | 1.000 | 0.985 | 0.956 | 0.887 | 0.823 | 0.763 | 0.657 |
| 3 | 0.719 | 0.861 | 0.942 | 0.985 | 1.000 | 0.980 | 0.932 | 0.887 | 0.844 | 0.763 |
| 5 | 0.566 | 0.763 | 0.887 | 0.956 | 0.980 | 1.000 | 0.970 | 0.942 | 0.914 | 0.861 |
| 10 | 0.400 | 0.566 | 0.763 | 0.887 | 0.932 | 0.970 | 1.000 | 0.985 | 0.970 | 0.942 |
| 15 | 0.400 | 0.419 | 0.657 | 0.823 | 0.887 | 0.942 | 0.985 | 1.000 | 0.990 | 0.970 |
| 20 | 0.400 | 0.400 | 0.566 | 0.763 | 0.844 | 0.914 | 0.970 | 0.990 | 1.000 | 0.985 |
| 30 | 0.400 | 0.400 | 0.419 | 0.657 | 0.763 | 0.861 | 0.942 | 0.970 | 0.985 | 1.000 |

The matrix is symmetric with a unit diagonal and every off-diagonal entry in [0.40, 1). The reconstructed formula of footnote 13 (the theta of 3 percent)

$$\rho_{kl}=\max\Big(\exp\Big[-0.03\,\frac{|T_k-T_l|}{\min(T_k,T_l)}\Big],\ 0.40\Big)$$

matches all 100 printed entries to the printed 0.1 percent (a reconstructed check; the table is primary).

`girr_delta_charge(sensitivities, tenors, params)` weights the ten sensitivities by the applied risk weights, takes the bucket charge under each scenario and the maximum, and returns
`SbmResult(weighted, k_by_scenario, charge)`: `k_by_scenario` is the tuple of $K_b$ under low, medium, high, in that order (the `girr.scenarios` of `c12-outputs`). The charge with the
sqrt(2)-divided weights is $1/\sqrt2$ times the undivided charge, because the weights scale uniformly; a zero sensitivity vector gives 0.

## Curvature (MAR21.5)

$$K_b^{\pm}=\sqrt{\max\Big(0,\ \sum_k\max(CVR_k^{\pm},0)^2+\sum_k\sum_{l\ne k}\rho_{kl}\,CVR_k^{\pm}CVR_l^{\pm}\,\psi(CVR_k^{\pm},CVR_l^{\pm})\Big)},\qquad K_b=\max(K_b^{+},K_b^{-})$$

$$\text{curvature}=\sqrt{\max\Big(0,\ \sum_bK_b^2+\sum_b\sum_{c\ne b}\gamma_{bc}\,S_bS_c\,\psi(S_b,S_c)\Big)}$$

with $\psi(x,y)=0$ if both are negative and 1 otherwise, and $S_b=\sum_kCVR_k^{+}$ if the upward scenario was selected for bucket $b$, else $\sum_kCVR_k^{-}$. The within-bucket correlations
are the squares of the delta ones (MAR21.100) and so are the cross-bucket ones (MAR21.101), $\gamma_{bc}^2$. `curvature_charge(cvr_plus, cvr_minus, rho, gamma, scenario)` takes the (B, K) positions, the
(K, K) DELTA correlation and the (B, B) DELTA cross-bucket correlation; it squares both and then applies the correlation scenario. The upward scenario is selected for bucket $b$ when
$K_b^{+}>K_b^{-}$; on a tie ($K_b^{+}=K_b^{-}$) it is selected if $\sum_kCVR_k^{+}>\sum_kCVR_k^{-}$, otherwise the downward one (MAR21.5(3)(a)(iii)). All-negative positions give 0 and the
result is never negative.

### Transcribed paragraphs

Source: Basel Framework MAR21, version in force 2023-01-01 (published 2026-03-23 on the page), `https://www.bis.org/basel_framework/chapter/MAR/21.htm?inforce=20230101&published=20200327`,
fetched 2026-10-19. The page renders formulas as SVG images without alt text; the symbols below were read from those images rendered, and the prose is verbatim.

MAR21.101: "For aggregating curvature risk positions across buckets, the curvature risk correlations are determined by squaring the corresponding delta correlation parameters. For instance, when
aggregating [two GIRR buckets b and c], the correlation should be [the square of the delta gamma]. In applying the high and low correlations scenario set out in MAR21.6, the curvature risk
capital requirements are calculated by applying the curvature correlation parameters [gamma_bc] (ie the square of the corresponding delta correlation parameter)."

MAR21.100 (first sentence): "For aggregating curvature risk positions within a bucket, the curvature risk correlations are determined by squaring the corresponding delta correlation parameters."

MAR21.5(3)(a), the selection: "The bucket level capital requirement (K_b) is determined as the greater of the capital requirement under the upward scenario (K_b+) and the capital requirement under
the downward scenario (K_b-). Notably, the selection of upward and downward scenarios is not necessarily the same across the high, medium and low correlations scenarios specified in MAR21.6.
(i) Where [K_b = K_b+], this shall be termed "selecting the upward scenario". (ii) Where [K_b = K_b-], this shall be termed "selecting the downward scenario". (iii) In the specific case where
[K_b+ = K_b-] if [the sum over k of CVR_k+ > the sum over k of CVR_k-], it is deemed that the upward scenario is selected; otherwise the downward scenario is selected."

Both conditions of (iii) are formulas on the page: the first is $K_b^{+}=K_b^{-}$ and the second is $\sum_kCVR_k^{+}>\sum_kCVR_k^{-}$, read from the rendered images
(`MAR21_20220101_32.svg` and `_33.svg`). The engine compares $K_b^{+}$ and $K_b^{-}$ for exact floating-point equality.

## The liquidity-horizon cascade (MAR33.4)

$$ES=\sqrt{\big(ES_T(P)\big)^2+\sum_{j\ge2}\Big(ES_T(P,j)\sqrt{\frac{LH_j-LH_{j-1}}{T}}\Big)^2}$$

with $T=10$ days and the horizons of MAR33.4 Table 1, $LH_j=10, 20, 40, 60, 120$ days; $ES_T(P,j)$ shocks only the risk factors with a liquidity horizon of at least $LH_j$ and holds the others
constant. `liquidity_cascade_es(es_t_full, es_t_by_horizon, base_horizon)` takes the ES per horizon in days; a missing horizon contributes nothing. The liquidity horizon by risk factor class is
module data (`LIQUIDITY_HORIZONS_TABLE_2`, MAR33.12 Table 2, each entry a `RegulatoryParameter`): interest rate specified currencies 10, other 20, volatility 60; credit spread sovereign
investment grade 20, high yield 40, corporate investment grade 40, high yield 60, volatility 120; equity large cap 10, small cap 20, large cap volatility 20, small cap volatility 60; FX
specified pairs 10, other pairs 20, volatility 40; commodity energy and carbon 20, precious and non-ferrous metals 20, other 60.

## Golden values

- One factor: a 10 million USD 10-year payer swap with $s=8.0\times10^{7}$ at the 10-year tenor gives $WS=0.011\times8.0\times10^{7}/\sqrt2$, that is 622,253.97, and a charge of the same size under every scenario.
- Scenarios (fixture entry `sbm-low-and-high-scenarios`): $\rho=0.5$ gives 0.375, 0.5, 0.625; $\rho=0.8$ gives low 0.6 and high 1.0; $\rho=1$ gives 1 in every scenario.
- Aggregation (fixture entries `sbm-bucket-and-cross-bucket-aggregation` and `sbm-cross-bucket-fallback`): the second has two buckets with $WS=(10,10)$ and $(-10,-10)$, within-bucket
  $\rho=0.5$ and $\gamma=0.8$, so $K_b=17.3205$, $S_b=\pm20$, the medium radicand is negative, the fallback gives $S_b=\pm17.3205$ and the charge 10.9545.
- Cascade: $ES_T(P)=10$ with every factor at $LH=10$ gives 10 (entry `cascade-all-lh-ten`); $ES_T(P)=10$, $ES_T(P,2)=6$ gives $\sqrt{100+36\times1}=11.661904$.
- The GIRR blocks of the four C12 documents (weighted sensitivities, the three scenario charges, the divided and undivided delta charge) are reproduced from their sensitivities.

## Conventions

Arrays are `float64`; money is in one currency; rho and gamma lie in [-1, 1]. A shape that does not match, a non-finite value or an unknown scenario is refused with the field named.
In force: MAR21 and MAR33 2023-01-01. The engine computes and states no compliance claim.

## Limits

- UNVERIFIED (addendum R7): the footnote 13 formula of MAR21.46 is reconstructed and matches all 100 entries of Table 2; the table is the primary value. The vega risk-weight formula
  (footnote 24) is not transcribed (a swap book has no vega); vega is not implemented.
- The curvature shock a consumer uses is `curvature_shock_undivided` (0.017). MAR21.99 bases it on "the highest prescribed delta risk weight" and MAR21.44's division by sqrt(2) names the delta
  risk weights of MAR21.42 to MAR21.43, so the divided value 0.012021 is a DERIVED reading (addendum R7), kept exposed as `curvature_shock` and labelled so.
- The row of MAR33.12 Table 2 "commodity volatility, 60 or 120 days" is ambiguous in dossier 01 D.3 and is not transcribed.
- Only the GIRR risk class is parameterised; the other six risk classes, DRC (MAR22), RRAO (MAR23) and SA-CVA are out of scope (decision D1 of the U9 sizing). No official numeric
  example of FRTB SBM exists in the sources read (dossier 01 UNVERIFIED item 8): the tests rest on the invariants of dossier 01 M.1.
- The USD sqrt(2) division is the bank's discretion (MAR21.44); the undivided charge is `girr_delta_charge` with `girr_parameters(specified_currency=False)`, shown beside it.
