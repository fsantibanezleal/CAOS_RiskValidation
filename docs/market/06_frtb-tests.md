# 06. The FRTB statistics

Module `riskvalidation.market.frtb`. Sources: dossier 01 section D.4 (MAR32.4 to MAR32.9, MAR32.18 to MAR32.19, MAR32.36 to
MAR32.42), dossier 05 B.4. The engine computes the statistics and the zones of the text and states no compliance claim.

## The binomial table, the zones and the multiplier

$$P(X\le k)=\sum_{i=0}^{k}\binom ni p^i(1-p)^{n-i},\qquad X\sim\mathrm{Bin}(250,\,0.01)$$

`binomial_cumulative_probability(k, n, p)` golden (MAR32.9 Table 1, in percent to two decimals) for `k = 0..10`: 8.11, 28.58,
54.32, 75.81, 89.22, 95.88, 98.63, 99.60, 99.89, 99.97, 99.99. `traffic_light_zone(exceptions, n, coverage)`: the zone design
of the Basel Committee (1996, https://www.bis.org/publ/bcbs22.htm) puts amber where the cumulative probability at true coverage
first reaches 95 percent and red where it reaches 99.99 percent; for `n = 250` and 99 percent that is green 0 to 4, amber 5 to 9,
red 10 or more, and the same rule gives the zones of another window. `mar32_multiplier(exceptions)` is the backtesting-dependent
multiplier of MAR32.9 Table 1: 1.50 for 0 to 4, then 1.70, 1.76, 1.83, 1.88, 1.92 for 5 to 9 and 2.00 for 10 or more. The US
multiplication factors of 12 CFR 217.204 Table 1 (3.00 for 0 to 4, 3.40, 3.50, 3.65, 3.75, 3.85 for 5 to 9, 4.00 for 10 or more;
dossier 05 B.4) are reference values for a document, not a function of the engine.

## The desk limits

`desk_exception_limit(alpha)` is 12 at 0.01 (99 percent VaR) and 30 at 0.025 (97.5 percent VaR) (MAR32.18 to MAR32.19: a desk
with more exceptions in the last 12 months moves to the standardised approach); `ValueError` for another alpha.
`desk_exceptions(loss, var, missing_as_exception)` counts the days with `loss > var` (positive loss numbers), with a NaN in either
input counted as an exception when the flag is true. For the bank-wide test MAR32.4 to MAR32.9 states that a missing P&L or VaR
counts as an outlier; for a desk the dossier does not say (UNVERIFIED), so the flag is explicit. Golden: `loss = [0.01, 0.03, nan,
0.02]`, `var = 0.02` everywhere gives 1 without the flag and 2 with it; equality is no exception.

## The P&L attribution test

$$\rho=\frac{\mathrm{cov}(R_{HPL},R_{RTPL})}{\sigma_{R_{HPL}}\sigma_{R_{RTPL}}},\qquad KS=\sup_x\big|F_{RTPL}(x)-F_{HPL}(x)\big|,\quad F(x)=0.004\,\#\{\text{obs}\le x\}$$

`pla_spearman(hpl, rtpl)` is the Pearson correlation of the average ranks (MAR32.36 to MAR32.38) and needs exactly 250 finite
observations (MAR32.20) and a nonzero spread. Golden: the series `0..249` against itself gives `1.0`, reversed gives `-1.0`, with its
first two values swapped gives `1 - 6 * 2 / (250 (250^2 - 1)) = 0.999999`. `pla_ks(hpl, rtpl)` (MAR32.39 to MAR32.41) is
computed as an exact integer count of ECDF steps, `gap = max_x |#{rtpl <= x} - #{hpl <= x}|` over the union of the observations,
and `KS = gap / 250` (review F5): the literal product `0.004 a - 0.004 b` gives `0.12000000000000002` for an exact 30-step gap,
which would put a desk that MAR32.42 classes as amber (KS not above 0.12) in red; `30 / 250` is exactly the float `0.12`.
Golden: `hpl = 0..249` and `rtpl = hpl + 0.5` give `0.004`; `rtpl = hpl + 29.5` (30 steps) gives exactly `0.12`; `hpl + 30.5` gives
`0.124`. `pla_zone(spearman, ks)` (MAR32.42): green when Spearman is above 0.80 and KS below 0.09; red when Spearman is below 0.70 or
KS above 0.12; amber otherwise, the boundaries belonging to amber. Golden: `(0.81, 0.08)` green, `(0.80, 0.05)` amber, `(0.85,
0.09)` amber, `(0.69, 0.05)` red, `(0.70, 0.05)` amber, `(0.95, 0.121)` red, `(0.95, 0.12)` amber.

Limits: red means the desk is ineligible for the internal models approach and amber triggers a capital surcharge (MAR32.43 and
MAR32.44); the engine states the zone, not the consequence. A desk series of any other length is refused, not padded.
