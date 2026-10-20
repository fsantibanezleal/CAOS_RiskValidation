# 02. Inference on the selection rates

Module `riskvalidation.fairness.inference`. Sources: dossier 06 section C.7 l.1036-1039 (two-sample proportion z-test or Fisher exact
test on the rates, with the remark that smaller differences can matter when statistically significant, quoting 29 CFR 1607.4(D)).
Primary texts: https://www.law.cornell.edu/cfr/text/29/1607.4; Fisher (1935), The Design of Experiments; scipy.stats.fisher_exact
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html; statsmodels `proportions_ztest`
https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html.

## The pooled two-proportion z-test

For two groups with $x_1$ of $n_1$ and $x_2$ of $n_2$ favourable decisions (group 1 minus group 2):

$$\hat p=\frac{x_1+x_2}{n_1+n_2},\qquad z=\frac{x_1/n_1-x_2/n_2}{\sqrt{\hat p(1-\hat p)\,(1/n_1+1/n_2)}},\qquad p=2\bigl(1-\Phi(|z|)\bigr)$$

Symbols: $\hat p$ the pooled rate under the null of equal rates; $\Phi$ the standard normal distribution function. `two_proportion_z_test`
returns a frozen `ZResult(statistic, p_value, n1, n2, note)`. A pooled rate of 0 or 1 has no variance: the statistic is 0, the
p-value 1 and `note` says so (`note` is None otherwise). The counts must be integers with $0 \le x \le n$ (a malformed count raises
`ValueError` naming the field; a group of no rows is `InsufficientData`). Golden and reference: the p-value equals
$2\,\mathrm{norm.sf}(|z|)$ of scipy to 1e-12 and the statistic and p-value equal statsmodels `proportions_ztest` to 1e-10 on a grid of
eight tables; swapping the groups flips the sign of $z$ and keeps $p$.

## Fisher's exact test

$$p=\sum_{\text{tables with the margins of the data}}\ \mathbf 1\{P(\text{table})\le P(\text{observed})\}\,P(\text{table}),\qquad
\widehat{\text{OR}}=\frac{x_1\,(n_2-x_2)}{(n_1-x_1)\,x_2}$$

`fisher_exact_test` returns a frozen `FisherResult(odds_ratio, p_value)`: the two-sided exact p-value of `scipy.stats.fisher_exact`
(equal to 1e-12 over the grid of tables) and the **sample** odds ratio, never the conditional maximum likelihood estimate; the odds
ratio is None when its denominator is 0 (infinite, or 0/0). Golden: the table $(30/100, 45/120)$ has odds ratio
$(30 \cdot 75)/(70 \cdot 45) = 0.714$; $(12/12, 3/20)$, $(5/10, 0/10)$ and $(0/10, 0/10)$ have none.

The interval of a controlled effect does not come from here: it comes from the decision model of
[the conditional page](03_conditional-and-replication.md).

Limits: both tests assume independent decisions within and between groups and compare unconditional rates; a significant difference
does not say why the rates differ and the engine states no legal claim. The Interagency Fair Lending Examination Procedures (2009)
note that gross HMDA approval or denial disparities are not appropriate for adverse impact analysis (addendum H3, section 4.3), so
these tests are indicators. The 5 percent and 1 percent cut points of the lights are the engine's default p-value conventions, not
regulatory thresholds. UNVERIFIED: no dossier prints a golden value for either test on credit decisions; the checks are identities
and independent references, plus the size and power runs of [the validation page](06_validation-tests.md).
