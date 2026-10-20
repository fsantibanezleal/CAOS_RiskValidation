# 05. Split conformal prediction

Module `riskvalidation.ml.conformal`. Sources: dossier 06 C.7 l.1019-1028 and I.1 l.1194; Angelopoulos and Bates, "A Gentle
Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification", arXiv 2107.07511; Vovk, Gammerman and
Shafer, "Algorithmic Learning in a Random World", 2nd ed. 2022, doi:10.1007/978-3-031-06649-8.

## The quantile and the sets

Scores $s_i=s(x_i,y_i)$ on $n$ calibration points ($1-\hat p_{y_i}(x_i)$ for classes); $\hat q$ is the $k$-th smallest score and
the set is $C(x)=\{y:s(x,y)\le\hat q\}$. Under exchangeability $P(Y_{n+1}\in C(X_{n+1}))\ge1-\alpha$ (06 C.7 l.1019-1028).

$$k=\lceil (n+1)(1-\alpha)\rceil,\qquad \hat q=\texttt{numpy.sort(scores)[k - 1]}\ (+\infty\ \text{if}\ k>n),\qquad
C(x)=\{0:\ p_1\le\hat q\}\cup\{1:\ 1-p_1\le\hat q\}$$

`split_conformal_quantile(scores, alpha)` computes $k$ on the **exact decimal** `alpha` (`fractions.Fraction(str(alpha))`), equals
`numpy.sort(scores)[k - 1]` and `numpy.quantile(scores, k / n, method="inverted_cdf")`; the `higher` method of `numpy.quantile` is
NOT the reference (it is the (k + 1)-th order statistic at the level $k/n$: 92.0 for the scores 1 to 100 at the level 0.91, while at
the level 0.9 it returns 91.0). Golden values (tested): the scores 1 to 100 at $\alpha=0.1$ give
$k=91$ and the 91st smallest score, 91.0; $n=19$ at $\alpha=0.05$ gives $k=19$; $n=5$ at $\alpha=0.1$ gives
$k=6>n$ and $+\infty$ (the whole label space). The quantile is monotone in $\alpha$.

- **Finding F2 (a float trap, found by search).** The contract says the exact decimal arithmetic is there so that `101 * 0.9` cannot
  round the wrong way. In IEEE doubles that product and `20 * 0.95` do not misround; the trap exists for other pairs: of the
  alphas 0.01 to 0.99 and the sizes 1 to 2,999, the float product `ceil((n + 1) * (1 - alpha))` differs from the exact decimal
  rank in 1,738 pairs, the first being $n=149$ and $\alpha=0.18$, where $150\times0.82=123$ exactly and the float product ceils to
  124. The golden of the test is that pair: the 123rd score (the float rank would give the 124th). The exact decimal is kept.

`conformal_sets(p1, qhat)` returns a boolean array `(n, 2)`: label 0 has score $1-(1-p_1)=p_1$ and label 1 has score $1-p_1$; a set
may be empty or hold both labels, and `qhat = +inf` gives both labels everywhere.

## The coverage test

`conformal_coverage(sets, y, alpha, *, n_cal)` returns the empirical coverage (the share of rows whose set holds the true label),
the target $1-\alpha$, `n_cal`, the rank `k` and two two-sided p-values of the covered count $c$ of the $m$ rows.

**The null (decision of the main session after the review, Q1).** Given a calibration of $n_{cal}$ points the coverage of the split
conformal set is Beta$(k,\,n_{cal}+1-k)$ distributed under exchangeability, with $k=\lceil (n_{cal}+1)(1-\alpha)\rceil$ in exact
arithmetic (the `Fraction` path above), so the covered count is Beta-Binomial$(m,\,k,\,n_{cal}+1-k)$. The light p-value
(`p_value`) is the sum of the probabilities of every count no more likely than the observed one (the minimum-likelihood rule of
`scipy.stats.binomtest`, with `scipy.stats.betabinom` for the law). The binomial p-value of the dossier (06 C.7 l.1019-1028) stays
in the result as `p_value_binomial`, beside it and never driving the light: it ignores the randomness of the calibration quantile,
and its size under the null is $2(1-\Phi(1.96/\sqrt{1+m/n_{cal}}))$ (normal theory): 0.062 at $m/n_{cal}=1/10$, about 0.16 at 1:1 and
about 0.38 at 4:1 (page 08). When $k>n_{cal}$ the set is every label, the coverage is 1 by construction and `p_value` is NaN: the
validation test is `not_evaluated` with that reason. A coverage loss is read as an exchangeability violation (drift), never as
proof.

## Reconciliation with MAPIE 1.5.0 (optional extra `ml`)

MAPIE 1.5.0's quantile code was read in `mapie/utils.py` for this unit (the contract marked its parameterisation UNVERIFIED); the
reconciliation tests skip when MAPIE is absent or another version is installed, and the engine core never imports it.

- `_compute_regression_quantile(scores, [1 - alpha])` takes `np.nanquantile(scores, ceil((1 - alpha)(n + 1)) / n, method="lower")`:
  the $k$-th order statistic of this function, except where its float product misrounds (finding F2: it returns the 124th score
  for $n=149$ and $\alpha=0.18$). On the grid $n\in\{19,50,99,100,101,149,250,1000\}$ and $\alpha\in\{0.05,0.1,0.18,0.2\}$ that is the
  only pair that differs.
- **Finding F3.** `_compute_classification_quantile(scores, [alpha])`, the one `SplitConformalClassifier` uses with the LAC score,
  takes `np.quantile(scores, (n + 1)(1 - alpha) / n, method="higher")` with a float product: the $k$-th or the $(k+1)$-th order
  statistic (on the grid above, the $(k+1)$-th in 9 of 32 pairs, never below the $k$-th, never more than one above). Its prediction
  sets therefore contain the sets of `conformal_sets` at this function's quantile (tested on the public
  `SplitConformalClassifier` with a stub classifier), a little more conservative than the definition above. The definition of
  `split_conformal_quantile` is not changed.

## What these checks do not claim

- Marginal coverage at $1-\alpha$ holds on average over calibration sets under exchangeability; it is not conditional coverage for
  a segment, and the test cannot tell drift from a bad calibration draw on its own (the Beta-Binomial law models the draw, not the
  segment).
- $\alpha=0.10$ in the shift kind is a convention of the case document (page 09).
- The engine computes and makes no compliance claim.

## Sources

- Angelopoulos and Bates, arXiv 2107.07511; Vovk, Gammerman and Shafer (2022), doi:10.1007/978-3-031-06649-8.
- Dossier 06 C.7 l.1019-1028; MAPIE 1.5.0 (BSD-3), https://pypi.org/pypi/mapie/json, read in `mapie/utils.py`.

## Limits

Scores are one number per calibration point (the pipeline computes them); the function is exchangeability-based and says nothing
about the model that produced the scores. UNVERIFIED: MAPIE's other split-conformal paths (regression with signed scores, the
conformalized quantile regression) were not read.
