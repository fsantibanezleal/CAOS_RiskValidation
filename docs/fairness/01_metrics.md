# 01. Metrics

Module `riskvalidation.fairness.metrics`. Sources: dossier 06 section C.7 l.1029-1040 (fairness for credit decisions on natural
persons) and I.6 l.1268-1270 (conventions are not thresholds); dossier 04 section J.3 l.897-905; the U14 research addendum (H3 the
legal tests). Primary texts: Hardt, Price and Srebro (2016), "Equality of Opportunity in Supervised Learning", arXiv:1610.02413;
29 CFR 1607.4(D), https://www.law.cornell.edu/cfr/text/29/1607.4; Kozodoi, Jacob and Lessmann (2022), "Fairness in credit scoring",
EJOR 297(3):1083-1094, doi:10.1016/j.ejor.2021.06.023, arXiv:2103.01907; Naeini, Cooper and Hauskrecht (2015),
doi:10.1609/aaai.v29i1.9602; Hanley and McNeil (1982), doi:10.1148/radiology.143.1.7063747.

## Selection rates and demographic parity

With protected attribute $A$, decision $\hat Y$ (1 favourable) and outcome $Y$:

$$P(\hat Y = 1 \mid A = a) = P(\hat Y = 1 \mid A = b)\qquad\text{(demographic or statistical parity)}$$

$$r_g=\frac{x_g}{n_g},\qquad \Delta_g = r_g - r_{ref},\qquad \rho_g=\frac{r_g}{r_{ref}}$$

Symbols: $x_g$ the favourable count and $n_g$ the size of group $g$; $r_g$ its selection rate; $r_{ref}$ the rate of the reference group;
$\Delta_g$ the parity difference and $\rho_g$ the parity ratio. `group_selection_rates(decision, group)` returns a frozen
`SelectionRate(group, n, favourable, rate)` per group present, ordered by label; a group with no rows is absent and `rate ==
favourable / n`. `demographic_parity_difference(rates, reference)` and `demographic_parity_ratio(rates, reference)` compute
$\Delta_g$ and $\rho_g$ on the counts, $(x_g n_{ref} - x_{ref} n_g) / (n_g n_{ref})$ and $(x_g n_{ref}) / (n_g x_{ref})$, in exact
integer arithmetic rounded once, so a gap that equals a tolerance is not moved by a float residue (55 of 100 over 50 of 100 is
exactly `0.05`, where the float difference of the two rates is `0.05000000000000004`). The reference maps to 0 and 1; the ratio is
None for every group when $x_{ref} = 0$, never a fabricated number. A reference that is not a group raises `ValueError`.

## Adverse impact and the four-fifths flag

The adverse impact ratio of dossier 06 C.7 l.1033-1037 is the protected group's selection rate over the chosen reference group's,
$\mathrm{AIR}_g=r_g/r_{ref}$; `adverse_impact_ratio` is `demographic_parity_ratio` (an identity). It is **not** the four-fifths
ratio. The four-fifths benchmark of 29 CFR 1607.4(D) says a selection rate "less than four-fifths (4/5) (or eighty percent)" of the
highest group's rate "will generally be regarded by the Federal enforcement agencies as evidence of adverse impact", and that smaller
differences can matter when statistically significant. It is an **employment benchmark** used by analogy in credit and never a
credit-law threshold (dossier 06 I.6 l.1268-1270). `four_fifths_flag(rates)` decides it on the counts in exact arithmetic, against the
highest-rate group $h$:

$$\text{flag}_g \;\Longleftrightarrow\; 5\,x_g\,n_h < 4\,x_h\,n_g$$

In plain text, a group is flagged when 5 x_g n_h < 4 x_h n_g, with integer counts. The highest-rate group is the first, in the order of `rates`, of the groups of the highest rate (found by cross-multiplication); it
has ratio 1 and no flag; a highest rate of 0 raises no flag and its ratios are None. `ratio_to_highest` is carried for display and the
flag never reads it. The result `FourFifths` carries `threshold = 0.8`, `label = 'employment benchmark'`,
`is_credit_law_threshold = False` and `compared_with = 'highest-rate-group'`. The float ratio would misjudge the boundary:
`0.6 / 0.75` is `0.7999999999999999` and flags a pair that is exactly four-fifths.

Golden values (the gate is `tests/fairness/test_fairness_metrics.py`): 60 of 100 against 75 of 100 is exactly four-fifths and is
not flagged; 5999 of 10000 against 75 of 100 is flagged and 6001 of 10000 is not. The scenario White 750 of 1000 (the reference),
Asian 900 of 1000 and Black 650 of 1000: Black is flagged against the highest-rate group (Asian, ratio $650/900 = 0.722$) and would
not be against the reference (ratio $650/750 = 0.867$). A Fraction oracle on 300 random count tables agrees with the flag.

## Equalized odds and equal opportunity

$$P(\hat Y = 1 \mid A = a, Y = y) = P(\hat Y = 1 \mid A = b, Y = y),\ y \in \{0,1\}\qquad\text{(equalized odds)}$$

$$\text{TPR}_g = P(\hat Y = 1 \mid A = g, Y = 1),\qquad \text{FPR}_g = P(\hat Y = 1 \mid A = g, Y = 0)$$

`equalized_odds_difference(y, decision, group)` is $\max(\max_g \text{TPR}_g - \min_g \text{TPR}_g,\ \max_g \text{FPR}_g - \min_g
\text{FPR}_g)$ in $[0, 1]$ and `equal_opportunity_difference` the true positive rate gap alone ($y = 1$ only). The rates are exact
fractions of the counts. Both need an outcome: `y = None` raises `ValueError` naming `y` and the reason (the dataset declares no
outcome); a group without both classes (without positives for equal opportunity) and fewer than two groups raise `InsufficientData`.
`y = 1` is the outcome the favourable decision should predict. Golden: groups with TPR 0.75 and 0.40 and FPR 0.25 and 0.50 give
equalized odds $0.35$ and equal opportunity $0.35$; with the second group's FPR at 1.0 equalized odds is $0.75$ and equal opportunity
stays $0.35$. Both reconcile with fairlearn 0.15.0 (`equalized_odds_difference`, the true-positive-rate difference) to 1e-12.

## Calibration within groups and AUC by group

Dossier 04 J.3 asks for $\Pr(\hat y=1 \mid y, G)$ gaps (separation), calibration within groups (sufficiency) and AUC by group.

$$\text{ECE}_g=\sum_{m}\frac{|B_m|}{n_g}\,\bigl|\bar y_m-\bar p_m\bigr|\qquad \text{AUC}_g=P(s_D>s_N)+\tfrac12 P(s_D=s_N)$$

`calibration_within_groups(p, y, group, n_bins=10)` is the engine's `pd_ece` (quantile bins of the model PD $p$ in $(0, 1)$, $y = 1$
a default) within each group; a group with fewer rows than `n_bins` is not evaluated and is absent from the result (the caller counts
the omitted groups). `auc_by_group(score, y, group)` is the engine's `disc_auc` within each group ($y = 1$ the class the score ranks
higher); a single-class group is omitted. Both need an outcome. Golden: a calibrated simulated group has ECE within Monte Carlo
error of zero (below 0.02 at 10,000 rows per group); a group whose outcomes occur at $p + 0.15$ has ECE above 0.10; `auc_by_group`
equals `disc_auc` on each group (identity).

Limits: the metrics describe disparities in a sample; they do not say why a disparity exists and state no legal claim. Public HMDA has
no outcome, so the four outcome-conditioned metrics never run on it. Groups with very few rows give unstable rates; the conditional
tests of the other pages carry the intervals. UNVERIFIED: the classification of the HMDA `action_taken` codes into approval and
denial is not settled by any source read (addendum H1); the C21 convention is approval for codes 1 and 2 and denial for code 3,
labelled a convention.
