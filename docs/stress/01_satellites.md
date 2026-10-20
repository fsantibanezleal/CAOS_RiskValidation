# 01. Satellites

Layer 2 of a macro stress test maps scenario variables to risk parameters (dossier 05 H.1). `riskvalidation.stress.satellites`
implements the satellite of a segment default or loss rate $\mathrm{DR}_{s,t}$ (dossier 04 I.1, a generic ARDL($p$,$q$) form, not a
quoted equation):

$$
g(\mathrm{DR}_{t}) = \alpha + \sum_{k=1}^{p}\phi_{k}\, g(\mathrm{DR}_{t-k}) + \sum_{l=0}^{q}\beta_{l}^{\top} X_{t-l} + \varepsilon_{t},
$$

with $g$ the link: the probit $\Phi^{-1}(\mathrm{DR})$, the logit $\ln\frac{\mathrm{DR}}{1-\mathrm{DR}}$ or the identity.

## The links

| Link | `link(name, p)` | `inverse_link(name, x)` | Domain of `p` |
|---|---|---|---|
| `probit` | $\Phi^{-1}(p)$ | $\Phi(x)$ | $(0, 1)$ |
| `logit` | $\ln\frac{p}{1-p}$ | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ |
| `identity` | $p$ | $x$ | any finite value |

The probit is the Vasicek-consistent choice: under the one-factor model, $\Phi^{-1}(\mathrm{DR}_t) = \big(\Phi^{-1}(\bar p) -
\sqrt{\rho}\,Z_t\big)/\sqrt{1-\rho}$ (dossier 04 I.1, C.4), so regressing $\Phi^{-1}(\mathrm{DR}_t)$ on macro variables is regressing the
latent credit cycle $Z_t$ on them; the round trip DR to $Z$ to DR is `inverse_link(link(p)) = p` to $10^{-12}$. The identity link is
the linear form of the Federal Reserve's own PD regression (dossier 05 l.399, $PD_{r,j,t} = \beta_0 + \beta_1\,BBB_t + \beta_2\,BBB_{t-4} +
\varepsilon$), for the segments whose rate is zero or negative in some quarters (net recoveries: commercial and residential real
estate in the Board's table). A probit or logit link refuses a rate outside $(0, 1)$ with the reason "a probit or logit fit on a
non-positive rate: the link needs a rate in (0, 1); use the identity link".

Golden values: `link("probit", 0.5) = 0.0`; `link("logit", 0.2) = -1.3863` (computed, $\ln 0.25$); `link("identity", -0.0003) = -0.0003`;
`inverse_link("probit", 1.6449) = 0.9500` (to $5\times10^{-5}$); `inverse_link("logit", -1.3863) = 0.2000`.

## The fit

`fit_satellite(rates, regressors, regressor_names, lags, ar_order, link_name, quarters, window_start, window_end)` solves the least
squares problem on the quarters of the window only, with `numpy.linalg.lstsq`, and the classical standard errors

$$
\hat\beta = \arg\min_\beta \lVert y - X\beta\rVert^2,\qquad s^2 = \frac{\lVert y - X\hat\beta\rVert^2}{n - k},\qquad
\widehat{\mathrm{se}}(\hat\beta_j) = \sqrt{s^2\,[(X^\top X)^{-1}]_{jj}},\qquad R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}}.
$$

The design holds an intercept, `ar_order` lags of $g(\mathrm{DR})$ and every regressor at every lag of `lags` (coefficient names
`intercept`, `ar_1`.., `<regressor>_lag<k>`, regressor-major). A lag never reaches before the window start, so `n_obs` is the number
of quarters of the window less the largest lag, $\max(p, \text{lags})$; the models document `fit.satellites[*]` records it and its
checker verifies it. The regressors are the agency series only (decision Q7): `real_gdp_growth`, `unemployment_rate`,
`cpi_inflation`, `treasury_3m`, `treasury_5y`, `treasury_10y`, `mortgage_rate`, `prime_rate`; the BBB corporate yield and the
third-party index levels are not read. The link is `identity` for commercial and residential real estate (windows starting 1991Q1,
the Board's table lacks both series from 1985Q1 to 1990Q4, with 13 and 6 non-positive quarters in 1991Q1 to 2019Q4) and `probit` for
the others (windows starting 1985Q1); the document records `non_positive_quarters_in_window` beside the link.

Refusals (each names the field): a probit or logit fit on a non-positive rate or a rate of 1 or more, a window outside the data, a gap
in `quarters` inside the window (a missing quarter is never filled; one outside the window is out of the window), a regressor outside
the agency list, fewer observations than coefficients (and none left for the residual variance), a constant rate (R squared is
undefined) and a rank-deficient design.

Golden values: the identity fit of `rates = (-0.002, 0.0, 0.002, 0.004)` on `x = (0, 1, 2, 3)` gives estimates $(-0.002, 0.002)$ with
$R^2 = 1$ and the same rates with the probit raise `ValueError`; the exact line $g = 1 + 2x$ over four quarters gives $(1, 2)$ with
standard errors $(0, 0)$, equal to `numpy.linalg.lstsq` to $10^{-10}$.

## The projection

`project_scenario(fit, last_link_values, regressor_history, regressor_path)` plugs the scenario path of $X_t$ into the recursion and
recovers $\mathrm{DR}_t$ (dossier 04 I.1: "plug the scenario path of $X_t$, recover $\mathrm{DR}_{s,t}$, then map to account-level
PDs"). The rates are `inverse_link` of the link values; they are signed on the identity link and read as net recoveries. A lag of a
quarter before the first projected one reads `regressor_history` (the last `max(lags)` observations, most recent last) and
`last_link_values` (the last `ar_order` link values, most recent last). The projection is monotone in the path when every $\beta
\geq 0$.

Golden value (computed): $\alpha = 0$, $\phi_1 = 0.5$, one regressor at lag 0 with $\beta = 0.1$, last link value 1.0 and path
$(2.0, 2.0)$ give link values $(0.7, 0.55)$ and probit rates $(0.758036, 0.708840)$.

## The conditional PD under a scenario

The map from a rate to the account-level PD is not rewritten here. It is the engine's `engines.vasicek.conditional_pd` and
`engines.vasicek.implied_factor` (dossier 04 I.2, C.4):

$$
\mathrm{PD}^{\text{stress}}_t = \Phi\!\left(\frac{\Phi^{-1}(\bar p) - \sqrt{\rho}\, z^{\text{scen}}_t}{\sqrt{1-\rho}}\right).
$$

Golden values (the round trip DR to $Z$ to DR through the reused `implied_factor`): at $\bar p = 0.02$, $\rho = 0.12$ and $z = 0$ the
conditional PD is $0.014287$; at $z = -2$ it is $0.073424$.

## Limits

- The satellites are statistical associations estimated on a stated window; they are not forecasts, and a projection under a
  scenario is scenario analysis. The engine states no compliance claim.
- The ARDL form is generic and is not a quoted equation of Wilson (1998) or Virolainen (2004): their exact specifications are
  UNVERIFIED in the dossier (04 I.1). The logit link and AR(2) macro process usually attributed to CreditPortfolioView are in Wilson
  (1997a, 1997b), not accessed (UNVERIFIED).
- Standard errors are classical (homoskedastic, no autocorrelation correction); with an autoregressive term and overlapping data
  they are optimistic.
- The Federal Reserve's own PD regression (05 l.399) is the special case with the identity link, the BBB spread and its four-quarter
  lag; here the BBB yield is not a readable regressor (Q7), so the Fed model is not replicated (E1).

## Sources

- Wilson, T.C. (1998), Portfolio credit risk, FRBNY Economic Policy Review 4(3):71-82,
  https://www.newyorkfed.org/medialibrary/media/research/epr/98v04n3/9810wils.pdf
- Virolainen, K. (2004), Macro stress testing with a macroeconomic credit risk model for Finland, Bank of Finland Research Discussion
  Paper 18/2004, https://ideas.repec.org/p/zbw/bofrdp/rdp2004_018.html
- Vasicek, O. (2002), Loan portfolio value, Risk, December 2002, 160-162; Gordy, M. (2003), doi:10.1016/S1042-9573(03)00040-8.
- Federal Reserve, Supervisory Stress Test Model Documentation: Credit Risk Models,
  https://www.federalreserve.gov/supervisionreg/files/fall-2026-credit-risk-models-description.pdf (dossier 05 H.3).
