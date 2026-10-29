# 06. The PD under informative withdrawals

`riskvalidation.transitions.withdrawals`: `withdrawal_bounds`, `withdrawal_pd`, `withdrawn_defaulters`,
`odds_multiplier`, `tipping_k`, and the constants `ASSUMPTIONS`, `MANSKI_1989`, `CRO_2020`.

A rating withdrawn during the period leaves the cohort, and a withdrawn rating that would have defaulted is not seen to
default. The usual treatments of the [cohort estimator](01_estimators.md) each fix one answer to what the withdrawn
would have done: remove them and renormalise (the industry treatment, Schuermann and Hanson 2004, footnote 4; CEREP's
transition page), keep them in the denominator unfollowed, or follow them (EBA/GL/2017/16 paragraph 76). When
withdrawal is informative (more likely in the year before default) every one understates the PD, and no test of the
matrix sees it. This module does not choose an answer: it states the PD as a function of how informative withdrawal is,
bounds it, finds the informativeness at which a decision flips, and, where a second source counts the defaults
differently, reads the informativeness off that source.

## 1. The family PD(k)

For one grade and one period: $D$ ratings in default at the end, $S$ still rated and not in default, $W$ withdrawn,
$N = D + S + W$ at the start. Of the $W$ withdrawn, $X$ would have defaulted within the period. Let the odds of
withdrawal among the ratings that would default be $k$ times the odds among those that would not:

$$ k = \frac{X/D}{(W - X)/S}, \qquad X(k) = W\,\frac{kD}{kD + S}, \qquad
   \mathrm{PD}(k) = \frac{D + X(k)}{N}. $$

$k = 0$ is the unfollowed treatment $D/N$; $k = 1$ (withdrawal independent of default) the industry treatment
$D/(D + S)$, identical to `cohort`; $k \to \infty$ every withdrawal a default in waiting, $(D + W)/N$, Manski's (1989)
worst-case bound for an outcome missing not at random. $\mathrm{PD}(k)$ increases in $k$. The family is the delta-adjusted
tipping-point analysis of clinical trials with missing outcomes (Cro et al. 2020), with the odds ratio as the sensitivity
parameter; its application to rating withdrawals was not found in the literature (UNVERIFIED, search of 2026-10-28).

```python
from riskvalidation.transitions.withdrawals import withdrawal_bounds, tipping_k

# counts as `cohort` takes them: rows the state at the start, columns at the end, default last
r = withdrawal_bounds(counts, withdrawn, thresholds=0.0075, anchor_defaults=default_page_counts,
                      anchor_obligors=default_page_cohorts)
r["pd_by_k"]       # grades x k_grid, with pd_unfollowed (k=0), pd_industry (k=1), pd_manski_upper (k=inf)
r["identified"]    # the Jeffreys band of the whole family: all the transition page alone supports
r["tipping"]       # the k at which PD(k) crosses the threshold, and whether the decision flips up, down or is robust
r["anchor"]        # the anchored k with its band, the anchored PD and its interval
```

Because $X$ and $k$ are in one-to-one correspondence for a given cohort, $k$ is a way to *state* informativeness, not an
extra assumption: the identifying content is in $X$. What the transition page alone cannot do is say which $k$; that
takes a second source.

## 2. The tipping k and the anchor

The **tipping k** of a decision is the $k$ at which $\mathrm{PD}(k)$ equals a threshold $t$ (a master-scale boundary, an
IRB risk-weight step): $X^* = tN - D$ must lie in $[0, W]$, then $k^* = X^* S / (D (W - X^*))$. The status is
`flips_up` (the industry PD is below $t$ and the family reaches it), `flips_down` (above, and the family goes below) or
`robust` (not crossed for any $k$). A decision that flips only at an implausible $k$ can be taken as it stands.

The **anchor**: where a second source counts the defaults of the same grade differently, for example CEREP's default
page (tab 2: the distinct ratings that defaulted during the period, withdrawn or not) against its transition page
(tab 4), that source gives $A$ defaults over $N_a$ ratings. Setting $A/N_a = \mathrm{PD}(k)$ gives the anchored $k$, and
the Jeffreys interval of $A/N_a$ gives its band; the anchored PD interval runs from the Jeffreys lower bound at the
low end of the band to the Jeffreys upper bound at the high end. It counts the sampling noise of both pages, so it is
conservative; and it is conditional on the two sources' definitions of default being the same (ESMA sets no common
definition), which the real-data check below shows is not the case. A grade with no observed default has no $k$ (the odds
relative to zero defaults), only Manski's bound $W/N$; the anchor still gives its PD. Every result lists its
assumptions; the module is closed form and draws no random numbers.

## 3. The evaluation

Script: `scripts/withdrawal_bounds_evaluation.py` (`python scripts/withdrawal_bounds_evaluation.py --reps 200`); the
measured numbers below are its output at 200 repetitions, seed 20261028, level 95%. Fixtures:
`tests/fixtures/withdrawals_sp_generator.json` (the seven-grade generator that Contraste's C04 case fits to S&P's pooled
annual CEREP counts, with the cohort sizes of that case, 1,391 obligors in B) and
`tests/fixtures/withdrawals_sp_cerep.json` (S&P's 26 annual cohorts, tabs 2 and 4).

### 3.1 Known truth

`generators.paths.RatingPaths` ([rating paths](../generators/05_rating-paths.md)) with withdrawal at 6% a year and
6% $(1 + \text{informative})$ in the year before default, five years; the observed paths are the latent paths from the same
random stream, truncated at the withdrawal. The truth is the generator's one-year PD. The design `first_year` is the
first annual cohort (the model's exact setting: nobody was withdrawn earlier); `pooled` pools the five annual cohorts as
Contraste's C04 does. The compared treatments, each with the Jeffreys interval of its count: **industry** ($k = 1$),
**unfollowed** ($k = 0$), **followed** (the cohort's end read on the latent paths; it needs information no transition
page has, so it is an oracle, and it is the default page's count in the same-cohort anchor), and **anchored** (the
default page counts the true defaults of the same cohort, `anchor="same"`).

Coverage of the true PD of grade B (one-year PD 1.17%), 200 repetitions:

| Design | informative | industry | unfollowed | followed | anchored | anchored width |
|---|---|---|---|---|---|---|
| first year | 0 | 0.965 | 0.965 | 0.965 | 0.980 | 1.95% |
| first year | 1 | 0.960 | 0.965 | 0.965 | 0.985 | 1.98% |
| first year | 3 | 0.940 | 0.945 | 0.965 | 0.990 | 2.03% |
| first year | 9 | 0.835 | 0.835 | 0.965 | 0.995 | 2.14% |
| first year, 10 times the cohort | 9 | 0.200 | 0.055 | 0.915 | 1.000 | 0.72% |
| first year, independent anchor cohort | 9 | 0.835 | 0.835 | 0.965 | 0.995 | 2.11% |
| first year, anchor captures 70% of defaults | 9 | 0.835 | 0.835 | 0.965 | 0.995 | 1.65% |
| pooled five years | 9 | 0.120 | 0.035 | 0.705 | 0.990 | 1.01% |

Relative bias of the point estimate of B's PD at informative 9: industry -23.3%, unfollowed -28.1%, followed -2.1%,
anchored -2.1% (first year); pooled five years: -35.5%, -39.6%, -16.6%, -16.6% (the 34%, 38% and 15% that Contraste's
C04 reports). Monte Carlo standard error of a coverage near 0.95 at 200 repetitions: 0.015. Other grades are in the
script's output; CCC to C at informative 9, first year: industry 0.915, unfollowed 0.780, followed 0.945, anchored
1.000, and at 10 times the cohort 0.425, 0.110, 0.925, 1.000.

### 3.2 Hypothesis and verdict

The plan's hypothesis (i): *the anchored interval covers the true PD in at least 90% of 200 repetitions at informative
1, 3 and 9, whereas the existing treatments cover it in under 50% at 9.* Measured, grade B, 200 repetitions:

- **Anchored coverage at least 90%: holds.** 0.985, 0.990 and 0.995 at informative 1, 3 and 9 (0.980 at 0). The
  interval is conservative: it covers 99.5% where it should cover 95%, and is 1.9 times as wide as the oracle's Jeffreys
  interval (2.14% against 1.12% at informative 9). In the pooled design its point estimate is still 16.6% low; the coverage
  there comes from the width, not from a centred interval, because a rating withdrawn in an earlier year is missing from a
  later cohort, a selection a one-period model does not see.
- **Industry and unfollowed under 50% at informative 9: does not hold at the case's cohort size.** They cover 0.835 each
  at 1,391 obligors in B: the understatement of 23 to 28% is smaller than the sampling interval of 16 expected defaults.
  It holds where the cohort makes the bias detectable: 0.200 and 0.055 at 10 times the cohort, 0.120 and 0.035 in the
  pooled five years. The clause as written is refuted at the size of the case and confirmed at 10 times it.
- **Followed under 50%: does not hold in any design** (lowest 0.705, pooled five years). In the first year it covers
  0.965: following the withdrawn is unbiased when nobody was withdrawn earlier, but it needs the end state of the
  withdrawn, which a transition page never has.
- The plan's refutation condition (coverage not better than the existing treatments') is not met: the anchored
  interval covers more than the industry's in every design at informative 9 (0.995 against 0.835 first year).
- A definition mismatch in the anchor is not repaired by the interval: when the second source counts only 70% of the true
  defaults, the anchored point is 22.8% low (the same as the industry treatment) and the interval, though still
  covering 99.5% because of its width, is centred on the wrong value. The anchor's definition must equal the transition
  page's, as the assumptions say.

### 3.3 Real data: S&P's cohorts

The default page (tab 2: $A$ distinct defaulted ratings over its cohort $N_a$) as the anchor of the transition page
(tab 4: $D$, $S$, $W$), per annual cohort and pooled over the 26 cohorts. Hypothesis (ii): the anchored $k$ for CCC to C
is above 1. The stability criterion was fixed in the script before the run: the anchored $k$ is *stable* when at least
80% of the cohorts' $k$ bands contain the pooled $k$.

| Grade | usable cohorts | pooled k | pooled k band | median cohort k | cohorts containing pooled k | stable |
|---|---|---|---|---|---|---|
| BBB | 9 | 7.9 | 2.3 to 15 | 0 | 8 of 9 | yes |
| BB | 16 | 7.5 | 4.9 to 10.5 | 6.1 | 14 of 16 | yes |
| B | 25 | 12.6 | 11.2 to 14.1 | 17.1 | 17 of 25 (68%) | no |
| CCC to C | 25 | 36.3 | 29.2 to 46.4 | 19.0 | 16 of 25 (64%) | no |

- **(ii) holds**: the pooled anchored $k$ of CCC to C is 36.3, with a band (29.2 to 46.4) far above 1.
- **Stability is refuted for B and CCC to C** under the criterion: only 68% and 64% of the cohorts' bands contain the
  pooled $k$. It holds for BB and BBB. The refutation clause of the plan ("the anchored $k$ is unstable across years")
  is therefore met for the two grades where the correction matters most.
- A $k$ of 36 is not a measure of informative withdrawal alone. Tab 2 and tab 4 define default differently (tab 2
  counts every distinct rating that defaulted during the period, tab 4 the ratings in a default category at its end;
  see the C04 page of Contraste for the definitions), so the anchor absorbs the definition gap into $k$. That
  reading is why the result is reported as an anchored $k$ conditional on the equivalence, not as an estimate of the
  true informativeness of S&P's withdrawals.

## 4. Limits

- One period. A rating withdrawn in an earlier year is missing from a later cohort; pooled cohorts inherit that selection
  and the method does not remove it (the pooled anchored point is 16.6% low).
- The anchor equates two sources' definitions of default; a mismatch moves the anchored point (section 3.2).
- The anchored interval is conservative (1.9 times the oracle's width at the case's size).
- Sparse grades: with few or no observed defaults $k$ is undefined or noisy; the result lists `no_observed_defaults`,
  `no_withdrawals` or `no_cohort`, and the Manski bound remains valid.

## Sources

- Manski (1989), Anatomy of the selection problem, Journal of Human Resources 24(3), doi:10.2307/145818.
- Cro et al. (2020), Sensitivity analysis for clinical trials with missing continuous outcome data using controlled
  multiple imputation: a practical guide, Statistics in Medicine 39(21), doi:10.1002/sim.8569.
- Schuermann and Hanson (2004), Estimating probabilities of default, FRBNY Staff Report 190, footnote 4.
- EBA/GL/2017/16, paragraphs 73 and 76.
- Brown, Cai and DasGupta (2001), doi:10.1214/ss/1009213286 (the Jeffreys interval).
- ESMA, CEREP help file (tabs 2 and 4); the S&P cohorts and the generator are those of Contraste's C04 case,
  https://github.com/fsantibanezleal/CAOS_Contraste.
