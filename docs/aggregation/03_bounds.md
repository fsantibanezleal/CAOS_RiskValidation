# 03. Bounds on the VaR of the total

Module `riskvalidation.aggregation.bounds`. Primary source: Embrechts, P., Puccetti, G. and Ruschendorf, L., "Model uncertainty and VaR
aggregation", Journal of Banking and Finance 37(8), 2750-2764, 2013, doi:10.1016/j.jbankfin.2013.03.014. Accepted-manuscript PDF
(19 pages): https://people.math.ethz.ch/~embrecht/ftp/MUVaR.pdf, section 2.2, pp.8-11, with Tables 1 and 4 on pp.12 and 14; transcribed
in F5 of the U13 research addendum (read 2026-10-19). Dossier 05 G.3: VaR is not subadditive, so comonotonic aggregation is not a
worst case, and the dependence-uncertainty band shows the bounds beside the diversified figures.

## The algorithm

For an $N\times d$ matrix $X$ of money, $s(X)=\min_i\sum_jx_{ij}$ (the smallest row sum) and $t(X)=\max_i\sum_jx_{ij}$ (the largest).
Two vectors are oppositely ordered when $(a_j-a_k)(b_j-b_k)\le0$ for all $j,k$.

Upper bound, the worst VaR $\overline{\mathrm{VaR}}_\alpha(L^+)$ (steps 1 to 7, p.10): with the accuracy $\varepsilon$, build

$$\underline x_{ij}=F_j^{-1}\!\Big(\alpha+(1-\alpha)\tfrac{i-1}{N}\Big),\qquad \overline x_{ij}=F_j^{-1}\!\Big(\alpha+(1-\alpha)\tfrac{i}{N}\Big)\quad(18)$$

permute each column of both matrices at random, rearrange each column in turn to be oppositely ordered to the sum of the other
columns (one sweep is $j=1,\dots,d$) and repeat sweeps until $s$ changes by less than $\varepsilon$. Then

$$s_N=s(\underline X^*)\le\bar s_N=s(\overline X^*),\qquad\text{both converge to }\overline{\mathrm{VaR}}_\alpha(L^+)\text{ as }N\text{ grows}\quad(19)$$

("in practice we find that"). Lower bound, the best VaR $\underline{\mathrm{VaR}}_\alpha(L^+)$ (p.10): the same steps with
$\underline z_{ij}=F_j^{-1}(\alpha(i-1)/N)$, $\overline z_{ij}=F_j^{-1}(\alpha i/N)$ over $[0,\alpha]$ (20), the loop minimising the largest row
sum $t$, and $t_N\le\bar t_N$ converge to the best VaR (21). The interval $(s_N,\bar s_N)$ is the "rearrangement range" of remark 6
(p.11). No distribution function is inverted: the quantile functions are evaluated on the grid.

Symbols: $F_j^{-1}$ the quantile function of marginal $j$, $\alpha$ the level, $N$ the number of cells, $d$ the number of marginals,
$\varepsilon$ the accuracy.

In code: `rearrangement_algorithm(matrix, objective, epsilon, max_sweeps, seed)` takes an $(N,D)$ matrix and `objective`
`"max-min-row-sum"` (the upper bound, on $\underline X$ or $\overline X$) or `"min-max-row-sum"` (the lower bound); it permutes each column
under the seed, sweeps until the value changes by less than `epsilon` between two successive sweeps (or `max_sweeps`, then
`converged = False`) and returns `(matrix, value, sweeps, converged)`, the matrix a column-wise permutation of the input.
`worst_var_bounds(ppfs, alpha, n, epsilon, max_sweeps, seed)` builds the four matrices, runs the four rearrangements with the seeds
`seed` to `seed + 3` and returns `VarBounds` (`lower_range` $=(t_N,\bar t_N)$, `upper_range` $=(s_N,\bar s_N)$, `sweeps` in that order,
`converged` when all four stopped on `epsilon`). `comonotonic_var(ppfs, q)` is $\sum_kF_k^{-1}(q)$.

## Golden values

Pareto(2) marginals, $F(x)=1-(1+x)^{-2}$ (equation 11 of the paper with $\theta=2$), so $F^{-1}(q)=(1-q)^{-1/2}-1$ and $F^{-1}(0.99)=9$.

Table 1 (PDF p.12): $d=3$, $\alpha=0.99$, $N=50$. The matrix $\underline X$ has first row $(9.00000,9.00000,9.00000)$ with row sum 27.0000, last
row $(69.71068,\dots)$ with row sum 209.1320 and every column sum 851.72901. After the algorithm every column still sums to 851.72901,
the printed smallest row sum is 44.7671 and the largest 87.8122. The random start moves the last digits: the test asserts the column
sums to $5\cdot10^{-5}$ and the smallest and largest row sums within 0.5 of the printed values (a tolerance proposed by the
contract). Measured on seeds 0 to 3: the smallest row sums are 44.95, 44.87, 45.05 and 44.94 and the largest is 87.8122 on all four.

Table 4 (PDF p.14), $d=8$, $N=10^5$, $\varepsilon=10^{-3}$ (comonotonic exact; lower and upper rearrangement ranges printed; the measured
ranges are those of `worst_var_bounds` at seed 0):

| $\alpha$ | comonotonic (exact) | lower, printed | lower, measured | upper exact (printed) | upper, printed | upper, measured |
|---|---|---|---|---|---|---|
| 0.99 | 72.00 | 9.00 - 9.00 | 8.995 - 9.000 | 141.67 | 141.66 - 141.67 | 141.663 - 141.669 |
| 0.995 | 105.14 | 13.13 - 13.14 | 13.128 - 13.142 | 203.66 | 203.65 - 203.66 | 203.656 - 203.664 |
| 0.999 | 244.98 | 30.47 - 30.62 | 30.466 - 30.623 | 465.29 | 465.28 - 465.30 | 465.276 - 465.295 |

The tests assert the ranges within 0.05 of the printed ones at 0.99 and 0.995 and within 0.5 at 0.999 (tolerances proposed by the
contract), and the comonotonic values for $d=8$, 56 and 648 (504.00, 735.96, 1714.88 and 5832.00, 8516.10, 19843.56 for the larger two,
recomputed from the formula). A closed-form check independent of the paper: three $U(0,10)$ marginals at 0.9 have a worst VaR of
$3\times9.5=28.5$ (the upper tail rearranged to a constant row sum), and the algorithm returns it within 0.05.

## Limits

- The exact upper values come from the paper's Proposition 4 for homogeneous Pareto marginals, which F5 does not transcribe: the table
  values are the golden numbers and the proposition is UNVERIFIED as a transcription.
- Remark 6 (p.11): "There does not exist an analytic proof that results (19) and (21) hold for all initial configurations", and
  "a proof of convergence of the RA remains an open problem" (UNVERIFIED as a theorem). The random permutation of step 3 is the
  paper's mitigation, and `converged = False` reports a run stopped by `max_sweeps`.
- For a marginal with unbounded support the last cell of $\overline X$ is $F^{-1}(1)=+\infty$. The engine keeps it as $+\infty$: a row that
  holds it has an infinite row sum, so it is never the smallest row and the smallest row sum stays finite. How the paper treats
  that cell is not in the text read for F5 (UNVERIFIED).
- The contract's invariant "with $D=1$ the value is the minimum entry" cannot hold with the refusal of $D<2$ stated beside it; the
  engine refuses $D<2$ (a single column has no other columns to be oppositely ordered to). The question is returned to the contract owner.
- The paper's computation times (5 s for $d=8$ up to 40 min for $d=648$ on a laptop) are the order of magnitude to expect; the engine
  measures about 1.6 s per level at $d=8$, $N=10^5$ on the development device.
