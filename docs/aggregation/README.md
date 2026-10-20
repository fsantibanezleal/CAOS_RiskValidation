# Aggregation: the ladder from the simple sum to the worst-case bounds

`riskvalidation.aggregation` aggregates the economic capital of several risks (credit, market, IRRBB, operational) on the
ladder of dossier 05 G.1: the simple sum, the variance-covariance (square-root) formula, the Gaussian and Student t copulas, and
the rearrangement bounds on the VaR of the total for any dependence; it returns the dependence-uncertainty band whole, the model
risk measures of dossier 06 B.12 and the allocation and diversification quantities of dossier 05 G.4.
`riskvalidation.validation.aggregation` holds the validation tests of dossier 05 G.5. The engine computes and makes no
compliance claim: thresholds and light parameters are supplied by the caller or are Contraste internal gates, not regulatory
requirements.

Conventions of every page:

- Capital is economic capital on the unexpected-loss basis, $EC_q = \mathrm{VaR}_q(L) - EL$ (dossier 04 l.652-653, BCBS 2005
  explanatory note, section 4.4). The expected loss is additive, so the simple sum of the marginal EC equals the comonotonic EC of
  the total. In the text "VaR - EL" is that basis.
- Money is a float in one currency (CLP millions in the case), the currency is the caller's and is never converted. Correlations and
  confidence levels are decimals (99.9 percent is `0.999`). One horizon (one year) and one confidence level apply to every marginal
  and every aggregate (dossier 05 G.5); a marginal read at another level or horizon is converted by the caller, never by the engine.
- The simple sum is the headline capital of the case (`headline-capital`); every other figure of the band is information, not
  capital (`information-not-capital`, dossier 05 l.519).
- A marginal whose severity has an infinite mean (the shape parameter xi of 1 or more, dossier 05 l.482) has no expected loss: the
  caller passes it as `ec = VaR` and `el = 0.0`, and the document labels its basis. No capped value stands in for it.
- Every function is pure and deterministic (a function with randomness takes an integer `seed` and uses
  `numpy.random.Generator(numpy.random.PCG64(seed))`), never mutates its inputs and raises `ValueError` naming the field when an
  input is missing or invalid; a validation test returns the light `not_evaluated` with a null statistic, metric and p-value for an
  undefined ratio.
- Every constant and golden value is transcribed from the source named on its page, or computed from a formula stated on the page
  and labelled as computed. An item the dossiers mark UNVERIFIED is marked UNVERIFIED here and is not silently decided.

| Page | Contents |
|---|---|
| [01. Simple](01_simple.md) | the simple sum, the correlation check, the variance-covariance formula, the hybrid quantile, the diversification factor, the Euler diversification benefit and the standard deviation contributions |
| [02. Copulas](02_copulas.md) | the marginal quantile functions, the Gaussian and Student t copula draws, the t and Clayton tail dependence, the Clayton copula and the copula aggregate |
| [03. Bounds](03_bounds.md) | the comonotonic VaR, the rearrangement algorithm and the best and worst VaR bounds, with Tables 1 and 4 of Embrechts, Puccetti and Ruschendorf |
| [04. Band](04_band.md) | the dependence-uncertainty band and its rows |
| [05. Model risk](05_model-risk.md) | the Barrieu and Scandolo add-on and relative measures and the Glasserman and Xu worst case |
| [06. The validation tests](06_validation-tests.md) | the eight tests with their light rules, and what must not be claimed |

Modules are imported by path, for example `from riskvalidation.aggregation import simple` and
`from riskvalidation.validation import aggregation`; nothing is re-exported from the package root. Requirements `RV-1301` to
`RV-1326` are in `docs/design/features/u13-aggregation/requirements.md`, each naming its test. Sources: dossier 05 of the Contraste
initiative (section G), dossier 06 (B.12), the U13 research addendum (F5) and dossier 04 (l.652-653); each page cites the primary
sources with their DOIs or URLs. Not implemented, by decision: the vine copula, Clayton sampling and the Bayesian VAR (BL-078).
Contract: `engine-aggregation-api` 1.0.0.
