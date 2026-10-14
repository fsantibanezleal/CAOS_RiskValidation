# 01. Curves

Module `riskvalidation.irrbb.curves`. Sources: dossier 05 section D.5 (term-structure fitting) and dossier 08 section C.2 (the US
Treasury daily par yield curve, whose tenor grid changes over time, so a missing tenor is absent from the input, never filled).
QuantLib is not a dependency of the engine core: the curves are NumPy and SciPy, and QuantLib enters only as an optional test
extra that cross-checks the discount factors.

## Zero rates and discount factors

`zero_rates_at(tenors, zero_rates, at)` interpolates linearly in tenor with flat extrapolation outside the grid, exact at the
pillars. `discount_factors(rates, times)` returns, with continuous compounding (SRP31.129),

$$DF(t)=e^{-R(t)\,t},\qquad DF(0)=1 .$$

Golden: tenors `[1, 2]`, rates `[0.01, 0.03]`: `at = 1.5` gives `0.02`, `0.5` gives `0.01`, `3` gives `0.03`; `R = 0.02`, `t = 2.5`
gives `0.951229`.

## Bootstrap from par yields

`bootstrap_zero_curve(tenors, par_yields, frequency=2)` reads a tenor at or below $1/f$ years as a simple-interest money-market
rate and bootstraps later pillars from the par condition with linear interpolation of the par yield between pillars on the coupon
dates; a pillar above $1/f$ that is not a coupon date is read from the bootstrapped dates by linear interpolation of the zero rate,
so its own par bond reprices at par only on the coupon dates.

$$R(t)=\frac{\ln(1+y\,t)}{t}\quad(t\le 1/f),\qquad 1=\sum_{j=1}^{n}\frac{y_n}{f}\,DF(t_j)+DF(t_n).$$

Symbols: $y$ the par yield (decimal, bond-equivalent with $f$ coupons a year), $t_j=j/f$ the coupon dates, $R$ the continuously
compounded zero rate. A flat par curve gives a flat zero curve at every tenor that is a multiple of $1/f$; below $1/f$ the
money-market rule gives a slightly different rate, by design.

Golden: a flat par curve $y=0.04$, $f=2$ gives $R=2\ln1.02=0.039605$ at the tenors $0.5,1,2,5,10$; at $1/12$ year it gives
$12\ln(1+0.04/12)=0.039933$ and at $0.25$ year $0.039801$. The bootstrapped discount factors agree with QuantLib 1.44
(`FixedRateBondHelper` and `PiecewiseLogLinearDiscount`, 30/360) to $10^{-12}$ in the optional test.

## Nelson-Siegel, Svensson and the fits

$$y(m)=\beta_0+\beta_1\frac{1-e^{-m/\tau}}{m/\tau}+\beta_2\left(\frac{1-e^{-m/\tau}}{m/\tau}-e^{-m/\tau}\right)$$

Svensson adds $\beta_3\left(\frac{1-e^{-m/\tau_2}}{m/\tau_2}-e^{-m/\tau_2}\right)$. The limit $m\to0$ is $\beta_0+\beta_1$ and the limit
$m\to\infty$ is $\beta_0$; the term at $m=0$ is evaluated by its limit, not by a division by zero. `fit_nelson_siegel` solves the
betas by least squares for each $\tau$ of a fixed grid (default $0.25$ to $10.0$ years in steps of $0.25$, which contains $2.0$)
and keeps the best; `fit_svensson` does a grid search over $(\tau_1,\tau_2)$ with $\tau_1<\tau_2$ (steps of $0.25$ from $0.5$ to
$20$, containing $2.0$ and $6.0$), solves the four betas by least squares at each node, and polishes the two decays with a bounded
least-squares step; the family nests Nelson-Siegel, so its RMSE is never above the Nelson-Siegel RMSE of the same data, and a
polish that does not converge raises `ValueError`.

Symbols: $m$ the maturity in years, $\tau$ the decay in years, $\beta_0$ the level, $\beta_1$ the slope, $\beta_2$ the curvature.

Golden: $\beta=(0.05,-0.02,0.01)$, $\tau=2$, $m=1$ gives `0.036065`, $m=10^{-9}$ gives `0.03`, $m=10^{9}$ gives `0.05`; data
generated at maturities `[0.5, 1, 2, 3, 5, 7, 10]` is recovered to `1e-8` in every beta and in $\tau$; Svensson data with
`(0.05, -0.02, 0.01, 0.005, 2.0, 6.0)` over `[0.5, 1, 2, 3, 5, 7, 10, 20]` is recovered to an RMSE below `1e-6`.

## Dynamic Nelson-Siegel (Diebold and Li)

$$y_t(m)=L_t+S_t\,\frac{1-e^{-\lambda m}}{\lambda m}+C_t\left(\frac{1-e^{-\lambda m}}{\lambda m}-e^{-\lambda m}\right),\qquad
\lambda=0.0609\ \text{per month}$$

`dns_loadings` returns the three loadings, `dns_yield` the yield, `fit_dns_factors` the factors by least squares at each date
(rows independent), and `forecast_dns_factors` simulates the factors as independent AR(1) processes
$f_t=c+\phi f_{t-1}+\varepsilon_t$ fitted by least squares, with Gaussian innovations of variance $\mathrm{SSE}/(T-3)$, starting
from the last row of the history (no look-ahead); a history with no spread is forecast as constant. The cross-correlation of the
factor innovations is not modelled.

Diebold and Li fix $\lambda=0.0609$ (maturities in months). Dossier 05 D.5 says the value "maximises the curvature loading at
exactly 30 months". A recomputation shows that the curvature loading $(1-e^{-x})/x-e^{-x}$ peaks at $x^*=1.7932821$, so the
maximiser is $m^*=x^*/\lambda=29.446$ months for $\lambda=0.0609$ (the loading at 30 months is `0.298384` against a maximum of
`0.298426`); 30 months is the paper's rounded figure, and the engine keeps the decay $0.0609$ rather than a decay solved for 30
(`0.059776`). UNVERIFIED against the NBER WP 10048 text in this unit.

Golden: at $m=30$ the slope loading is `0.459280` and the curvature loading `0.298384`; at $m=12$: `0.709464` and `0.227941`; at
$m=60$: `0.266588` and `0.240701`; `(0.05, -0.02, 0.01)` at $m=30$ gives `0.043798`; a constant history gives every path equal
to the row.

## Sources

- Nelson and Siegel (1987), J. Business 60(4):473-489, doi:10.1086/296409.
- Svensson (1994), NBER WP 4871, doi:10.3386/w4871.
- Diebold and Li (2006), J. Econometrics 130(2):337-364, doi:10.1016/j.jeconom.2005.03.005; NBER WP 10048
  https://www.nber.org/system/files/working_papers/w10048/w10048.pdf.
- US Treasury daily par yield curve, https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics
  (dossier 08 C.2).

## Limits

The engine states no claim that a fitted curve is arbitrage-free. The Diebold-Li statement about 30 months is UNVERIFIED against
the paper. The factor innovations are independent across factors.
