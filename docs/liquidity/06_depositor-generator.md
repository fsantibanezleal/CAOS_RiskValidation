# 06. The depositor generator with known truth

Module `riskvalidation.liquidity.deposits`. Sources: dossier 08 G.2 line 698 (a depositor generator with known truth) and G.4 lines 739 to
742 (its verification); the generator contract of `riskvalidation.generators._base` (a frozen configuration, one dataset per seed, an
exact truth, a verifier; Morris, White and Crowther 2019, doi:10.1002/sim.8086). Contract `engine-liquidity-api` 1.2.0, decision Q10
(placement in the engine, option (a), no concentration parameter).

## The mechanism

A systemic two-state Markov chain over days is shared by all depositors: state 0 (calm) on day 1; calm to stress with probability
`p_enter` and stress to calm with `p_exit` per day. Per segment, a depositor still deposited on day `d` withdraws with hazard `h_calm` or
`h_stress` by the state of day `d`. Balances are equal within a segment, so the cumulative outflow share is the share of depositors who have
withdrawn. With transition matrix `P` and survival weights `u_d` over the states:

$$P=\begin{pmatrix}1-p_{enter}&p_{enter}\\ p_{exit}&1-p_{exit}\end{pmatrix},\qquad
u_1=(1-h_{calm},\,0),\qquad u_{d+1}=(u_d P)\odot(1-h_{calm},\,1-h_{stress}),$$

$$\text{truth}(n)=1-\sum u_n .$$

`DepositorConfig` holds the frozen `DepositorSegment` rows (`segment` one of the five `SEGMENTS`, `share_of_deposits` summing to one and the two
hazards, each in `[0, 1]`), `n_depositors` per segment, `horizon_days` and the two probabilities of the state chain, `p_enter_stress` and
`p_exit_stress`, each in `[0, 1]`; a share vector that does not sum to one, a probability outside `[0, 1]`, a non-positive count or an unknown
segment raises `ValueError` naming the field. Because the chain is shared by every depositor, the two probabilities belong to the configuration
and a `DepositorSegment` does not take them (passing them is a `TypeError`; before `engine-liquidity-api` 2.0.0 each segment carried them and
the segments had to agree). The configuration
serialises through `config()` into the `generator.config` block of the models document.

`DepositorGenerator.draw(rng)` returns `{"state": (H,), "withdrawn_day": {segment: (n,)}, "outflow_rate": {segment: (H,)}}` (0 in
`withdrawn_day` is "not within the horizon"): one seed gives one dataset. `truth()` returns the exact expected cumulative share by horizon.
`verify(n_draws, seed)` draws on its own seeded stream and returns the `Check` list (expected, observed, standard error of a mean) at the
first day, the middle day and the last day of the horizon for each segment.

## Golden values

| Configuration | Truth |
|---|---|
| `h_calm = 0.001`, `h_stress = 0.01`, `p_enter = 1`, `p_exit = 0`, horizons 1, 2, 3 | `0.001`, `0.01099`, `0.0208801` (`1 - 0.999 x 0.99 x 0.99`) |
| `p_enter = 0` | `1 - (1 - h_calm)^n`, the constant compounding of page 01 |
| shares `0.057`, `0.864`, `0.079` | sum to 1.0 |

The tests compare the transfer recursion with an enumeration of every state path for a short horizon, and a large seeded draw with the
truth within four standard errors.

## Conventions

Monte Carlo is seeded (`numpy.random.default_rng`); the verification seed is `20261005`. The generator stores no hidden state: the draw is a
function of the configuration and the generator passed in.

## Limits

- The stress-state transition law (a systemic two-state chain, calm on day 1) has no dossier basis: it is this contract's convention,
  UNVERIFIED, and no concentration parameter is added.
- The generator is a data-generating mechanism with known parameters. It checks that an estimator or a run-off assumption finds what was
  put in; it does not describe how any bank's depositors behave.
- Equal balances within a segment mean the share of depositors equals the share of money; concentration of balances is not modelled.

## Sources

- Dossier 08, G.2 and G.4, and `riskvalidation.generators._base` (the repository's generator contract).
- Morris, White and Crowther, Using simulation studies to evaluate statistical methods, Statistics in Medicine 2019:
  https://doi.org/10.1002/sim.8086
