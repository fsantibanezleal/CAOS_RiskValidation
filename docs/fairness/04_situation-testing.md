# 04. Situation testing

Module `riskvalidation.fairness.situation`. Sources: dossier 07 section A.3 l.158-177 and l.700 (reimplement mutatis mutandis with
attribution, cite both papers); the U14 research addendum H5 (sections 7.1 to 7.6, transcribed from the papers and the reference
code). Primary texts: Alvarez and Ruggieri, "Counterfactual Situation Testing: From Single to Multidimensional Discrimination",
arXiv:2502.01267v2 = JAIR 82:2279-2323, doi:10.1613/jair.1.17935 (the algorithm, Section 4, equations 3 to 10); Alvarez and
Ruggieri, "Mutatis Mutandis: Revisiting the Comparator in Discrimination Testing", arXiv:2405.13693v4, Computational Intelligence
42(4), doi:10.1111/coin.70238 (the comparators, equations 2 to 7). Reference code: `github.com/SantanderAI/mutatis-mutandis` at
commit `38e2c5bff999d20f9424024019d85396df87ac4f` (Apache-2.0) and the companion repository `cc-jalvarez/counterfactual-situation-testing`
at commit `a94a53f11c90da576d00d8f84fc9a2d7f867eee9` (no licence declared: its data is not stored here).

## The algorithm

Setting: the group indicator is 1 for the protected group (the control search space $\mathcal D_c$) and 0 for the others (the test
search space $\mathcal D_t$); the decision 0 is the negative one. For each complainant $c$ of the protected group:

$$d(x_1,x_2)=\frac{1}{|X|}\sum_{i=1}^{|X|} d_i(x_{1,i},x_{2,i}),\quad d_i=\begin{cases}\mathbf 1\{x_{1,i}\ne x_{2,i}\}&\text{categorical (overlap)}\\[2pt]
\dfrac{|x_{1,i}-x_{2,i}|}{\max X_i-\min X_i}&\text{continuous (equation 5)}\end{cases}$$

$$\text{k-ctr}=\{i\in\mathcal D_c\setminus\{c\}:\operatorname{rank}_d(x_c,x_i)\le k\},\qquad
\text{k-tst}=\{i\in\mathcal D_t:\operatorname{rank}_d(x^{\circ}_c,x_i)\le k\}$$

$$\Delta p=p_c-p_t,\quad p_c=\frac{\#\{\text{negative in k-ctr}\}}{k},\ p_t=\frac{\#\{\text{negative in k-tst}\}}{k}\qquad\text{(equations 8 and 9)}$$

$$\bigl[\Delta p-w_\alpha,\ +\infty\bigr),\qquad w_\alpha=z_\alpha\sqrt{\frac{p_c(1-p_c)+p_t(1-p_t)}{k}},\quad z_\alpha=\Phi^{-1}(1-\alpha)\qquad\text{(equation 10)}$$

Symbols: $x_c$ the complainant's attributes; $x^{\circ}_c = x_c$ for the ceteris paribus (CP) comparator and the counterfactual
$x^{CF}_c$ for the mutatis mutandis (MM) comparator; $k$ the neighbourhood size; $\tau$ the accepted deviation. Neither the protected
attribute nor the decision enters the distance (equations 6 and 7); ties are broken by row order. The complainant is the search center
and is excluded from its own control group (the "CST without centers" version, denominator $k$); the statistic is computed for every
complainant regardless of their own decision. A complainant is a **case** when $\Delta p>\tau$ and the claim is **significant** at level
$\alpha$ when the lower bound $\Delta p-w_\alpha$ exceeds $\tau$ (Definitions 4.4 and 4.5). The numerator of $w_\alpha$ carries the plus
sign (footnote 7: the EAAMO paper's minus sign is a typo). `bonferroni_alpha(alpha, q)` is $\alpha/q$ for $q$ protected attributes
(Definition 4.6).

Functions: `gower_distance`, `k_nearest`, `situation_statistic` (returns `(delta_p, p_c, p_t)`), `wald_interval_one_sided` (returns the
lower bound), `situation_test` (the whole test over the complainants, returning a frozen `SituationResult(n_complainants, n_cases,
n_significant, k, tau, alpha)` with $n_{significant}\le n_{cases}\le n_{complainants}$) and `bonferroni_alpha`.

## The two normalizations

`normalization='range'` (the default, the paper's text and C21's choice): `scale` holds $\max-\min$ of each continuous attribute over the
search space the group is drawn from (control: the protected rows; test: the other rows), and the distance lies in $[0,1]$.
`normalization='standard'` (the reference code's default, a scikit-learn `StandardScaler`): `scale` holds the population standard
deviation (ddof 0) of the attribute over all rows, so the continuous term is the absolute z-score difference and is not bounded by 1.
Any other value raises `ValueError`, as does a zero scale on a continuous attribute (a constant column). Under MM the counterfactual
is read in the same space as the factual rows (the paper's one space); the reference code instead standardizes the counterfactual
dataset with its own scaler (finding F3 of the contract), which the caller reproduces by expressing the counterfactual through that
scaler before the call. The scaler that reproduces the published CST counts is the counterfactual file's own `StandardScaler`
(population standard deviation, ddof 0), applied as `(cf - cf.mean(0)) / cf.std(0) * x.std(0) + x.mean(0)` with `x` the factual
attributes; the engine keeps its one-space reading of the counterfactual and the gated test applies that rescaling before the call.

`situation_test` takes `categorical`, a boolean mask over the columns of `X` (contract 1.1.0; `None`, the default, keeps the 1.0.0
reading of every column as continuous). A categorical column contributes the overlap measure (1 if the two values differ, 0
otherwise) and every other column the normalised Manhattan distance, equations 3 to 5 of the paper; the distance is their mean. Under
`normalization='standard'` the categorical columns still use overlap, as the reference code does for nominal attributes, and the
standardisation applies to the other columns only. A mask that is not one flag per column raises `ValueError`; the scale of a
categorical column is never read, so a constant categorical column is not a zero-scale error. `gower_distance` and `k_nearest`
already took the same explicit mask (a required argument), unchanged in 1.1.0.

Golden (hand-computed, `test_situation_test_categorical_golden_by_hand`): seven rows `(continuous, categorical)`, protected
`(0,2) (6,0) (9,1)` with decisions 1, 0, 1, others `(2,1) (7,0) (6,1) (1,2)` with decisions 0, 0, 1, 1, $k=1$, `range`. The distance
is $(\mathrm{md}+\mathrm{ol})/2$ with scales 9 (control) and 6 (test). Complainant 0: control nearest is row 1 (5/6 against 1), a
negative decision, $p_c=1$; test nearest is the last other row (1/12), favourable, $p_t=0$; $\Delta p=1$. Complainant 1: $\Delta p\le 0$.
Complainant 2: control nearest is row 1 (2/3), $p_c=1$; test nearest is the third other row (1/4), $p_t=0$; $\Delta p=1$. Result: 3
complainants, 2 cases, 2 significant; the same table read as continuous gives 1 case.

## Golden values

- The toy fixture of addendum 7.6 (4), the reference repository's own test: four protected rows with scores 10, 11, 12, 13 and
  decisions 0, 0, 1, 1, four other rows 10.5, 11.5, 12.5, 13.5, all 1; $k=2$, $\alpha=0.05$, $\tau=0$. For every protected row
  $p_c=0.5$, $p_t=0$, $\Delta p=0.5$; with the exact quantile $\Phi^{-1}(0.95)=1.644854$, $w_\alpha=0.58154$ (to 1e-5) and the lower
  bound is $-0.0815$ (to 1e-4), below $\tau=0$, so the claim is not significant: `situation_test` returns 4 complainants, 4 cases,
  0 significant. It holds under both normalizations (one continuous attribute: the distance order is the same). The reference code's
  report for the same fixture rounds $z$ to 1.645 ($w=0.581595$) and prints the symmetric two-sided interval $[-0.082, 1.082]$:
  recorded in the contract's findings, not claimed.
- The loan-application experiment of the companion paper (addendum 7.6 (2); the data is the companion repository's
  `data/LoanApplication_v2.csv`, 4,993 rows, 1,712 with `Gender` 1). The published standard situation testing counts were produced
  with the reference code's StandardScaler, so they are claimed **only under `standard`**; nothing claims them under `range`.
  Ceteris paribus, `normalization='standard'`, $\alpha=0.05$, $\tau=0$, attributes `AnnualSalary` and `AccountBalance`, cases then
  significant:

  | k | 15 | 30 | 50 | 100 | 250 |
  |---|---|---|---|---|---|
  | ST cases, significant | 55, 44 | 65, 57 | 84, 65 | 107, 85 | 204, 148 |
  | CST without centers, cases, significant | 288, 272 | 313, 306 | 342, 331 | 395, 383 | 534, 519 |

  Both rows reproduce exactly, in every cell, in the gated test `test_situation_test_reproduces_the_published_loan_application_counts_under_standard`
  (run when `FAIRNESS_LOAN_APPLICATION_CSV` and `FAIRNESS_LOAN_APPLICATION_CF_CSV` name the two files; skipped with the reason
  otherwise): the CST row once the counterfactual file is expressed in the factual attribute space through its own scaler, as the reference code
  does. With the counterfactual left in the factual space the counts differ (433, 416 at $k=15$): that is finding F3, not an error of
  the algorithm. Under `range`, this engine's counts on the same file are 54, 44; 66, 56; 80, 61; 101, 79 and 191, 137 (measured
  here, not published).
- Not reproduced: the Law School counts of the main paper (addendum 7.6 (1)), whose data is not redistributed, and the CST with
  centers (`include_centers=True`), a path that is broken in the reference code on pandas 3 (finding F3) and is not implemented here.

## What is claimed and what is not

Tests follow the paper's equations and the golden values above, never the reference code's report columns. The **comparator** is a
caller's input: public HMDA gives no structural causal model and no counterfactual dataset, so C21 runs the ceteris paribus comparator
only, with `normalization='range'`. `mutatis-mutandis` without a `counterfactual` aligned row by row raises `ValueError`.

Limits: situation testing is an **individual-level indicator** of comparative evidence: it asks whether comparable individuals of
the two groups were treated differently, not why, and it states no legal claim. It uses the attributes the caller supplies, so an
omitted attribute (the credit score of public HMDA) can drive a difference. The Wald interval is a large-sample approximation and is
unreliable for a small $k$. The significant share of the complainants is correlated across complainants (they share neighbours), so
its variability under the null is larger than a binomial count's (the measured size of the amber rule is on
[the validation page](06_validation-tests.md)). UNVERIFIED: the structural causal model and the counterfactual generator of C21 are not
supplied by the algorithm; ethnicity maps to national origin only by convention (addendum H3); the generating seed of the reference
loan-application file was not traced.
