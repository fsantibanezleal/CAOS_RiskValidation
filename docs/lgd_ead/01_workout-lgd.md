# 01. Workout LGD

Module `riskvalidation.lgd_ead.workout`. Source: dossier 04 F.1 and F.2.

## The realised LGD

For a defaulted exposure $i$ with exposure at default $\mathrm{EAD}_i$, recoveries $R_{i,t}$, direct and indirect costs
$C_{i,t}$ and additional drawings $A_{i,t}$ after default, at times $t$ (years since the default date) and discounted at
$r_i$:

$$\mathrm{LGD}_i = 1 - \frac{\sum_{t}\big(R_{i,t}-C_{i,t}-A_{i,t}\big)\,(1+r_i)^{-t}}{\mathrm{EAD}_i}.$$

Symbols: $R$ recoveries, $C$ costs, $A$ additional drawings (money, nonnegative, zero-padded where an exposure has fewer
flows), $t > 0$ in years, $r_i$ the discount rate per year (a scalar or one per exposure), $\mathrm{EAD}_i > 0$.
`workout_lgd` returns the value as it comes, **not clipped**: with costs above recoveries the LGD exceeds 1, with
recoveries above the exposure it is negative. The result is nonincreasing in the recoveries and nondecreasing in the costs
and the drawings. It is **not** nondecreasing in the discount rate in general: a higher rate lowers the present value of the
net flows, which raises the LGD when the net flows are positive and lowers it when costs exceed recoveries.

## The discount rate convention

`eba_discount_rate(interbank_rate, add_on=0.05)` returns the interbank rate plus the add-on. It is the reading of
EBA/GL/2017/16, "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures"
(https://www.eba.europa.eu/sites/default/files/documents/10180/2033363/6b062012-45d6-4655-af04-801d26493ed0/Guidelines%20on%20PD%20and%20LGD%20estimation%20(EBA-GL-2017-16).pdf)
in dossier 04 F.1: an annual rate equal to a primary interbank offered rate at the default date plus 5 percentage points.
The paragraph number (section 6.3.1.3 per a search summary) is UNVERIFIED. The caller may pass any other rate to
`workout_lgd`; the function only names the convention.

## Golden values

| Call | Value |
|---|---|
| `eba_discount_rate(0.0325)` | 0.0825 |
| `workout_lgd`, EAD 100, recoveries 40 and 30 and one cost of 10 at 1 and 2 years, 10% | PV = 30/1.1 + 30/1.21 = 52.066116, LGD 0.479339 |
| the same with recoveries 5 and costs 20 | 1.136364 (above-one); at a zero rate 1.15, so the LGD falls as the rate rises for a net cash outflow |
| the same with recoveries 120 and no costs | -0.090909 (below-zero) |

## Range flags and the distribution

`range_flag` returns `below-zero`, `in-range` (0 and 1 included) or `above-one`; a NaN raises. The case documents carry a
realised value as `{value, flag}`. `lgd_distribution(lgd, edges)` counts the values inside [0, 1] per bin (bin $i$ is
$[e_i, e_{i+1})$, the last bin closed), counts the values below 0 and above 1 apart (never in an end bin) and gives the exact
masses at 0 and 1, so that the bimodality of dossier 04 F.2 is visible (LGD is bounded in [0, 1] up to fees and costs, with
masses near 0 for cures and near 1 for no recovery; Tong, Mues and Thomas 2013, doi:10.1016/j.ijforecast.2013.03.003).
Golden: `lgd = [-0.05, 0, 0, 0.25, 0.5, 1, 1, 1.2]`, `edges = [0, 0.5, 1]` gives counts `[3, 3]`, one below 0, one above 1,
two at 0 and two at 1.

## Incomplete workouts and the training cut-off

An open recovery process must be treated explicitly: excluding the incomplete workouts biases LGD down and including them
requires a projection of the future recoveries (dossier 04 F.1). `resolve_incomplete(lgd, complete, treatment, projected)`
takes the treatment as a required argument (`excluded` or `projected`) and returns the counts, which feed the document block
`workout.incomplete_workouts`. Golden: `lgd = [0.2, 0.6, 0.9]` with the last workout incomplete gives mean 0.4 when excluded
and, with the projection 0.7, mean 0.5. `closed_before(close_dates, cutoff)` is true only for a close strictly before the
cutoff; it is the training-set rule of the leakage-safe splits: the training set holds only the defaults whose workout closed
before the training cutoff.

## Limits

The engine states no compliance claim: the discount rate, the treatment of incomplete workouts and the cut-off are choices
of the caller and are recorded in the document. The EBA paragraph of the discount rate is UNVERIFIED. The C08 case on the SBA
7(a) public file has no recovery cash flows (dossier 08 A.6), so it is a proxy that does not call `workout_lgd`.
