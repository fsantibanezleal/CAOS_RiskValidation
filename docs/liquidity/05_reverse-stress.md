# 05. Reverse liquidity stress: the run-off multiplier at which the survival period reaches a target

Module `riskvalidation.liquidity.reverse`. Source: dossier 05 section E.5 rung 5, line 279: "the run-off multiplier at which the survival
period falls below a target (for example 30 days)", the named parameter `reverse_target_days`; BCBS Principle 10 for the stress family
(dossier 05 E.2). Contract `engine-liquidity-api` 1.2.0, decision Q9 (the cap).

## Equations

The multiplier `m` scales every daily rate of the base scenario and is capped so that each rate stays at or below 1:

$$q_{j,d}(m)=\min\big(m\,q_{j,d},\,1\big),\qquad m\le m_{\max}=\frac{1}{\max\{q_{j,d}:\ q_{j,d}<1\}} .$$

The survival period `SP(m)` is the first day the net liquidity position of the scaled scenario is negative (page 03). It is non-increasing in
`m`, and "survives the horizon" counts as longer than any day. The reverse stress asks for the smallest multiplier at which `SP(m)` is at
most the target `T`:

$$m^{*}=\min\{m\in[1,m_{\max}]:\ SP(m)\le T\}.$$

`SP(m) <= T` holds exactly when the minimum of `NLP(t)` over `t <= T` is negative, which is a continuous and non-increasing function of
`m`, so `solve_multiplier` brackets its root between 1 and the cap with `scipy.optimize.brentq`, rounds the multiplier up to six decimals
and checks the neighbouring decimals, so that the result is the smallest six-decimal multiplier that reaches the target.
`survival_vs_multiplier(scenario, cbc, multipliers)` returns one `SurvivalResult` per multiplier (increasing, at least 1); a multiplier
beyond the cap is refused (the document omits that point). `max_multiplier(scenario)` returns the cap, infinite when no rate lies in
`(0, 1)`.

## Refusals: never a number

`solve_multiplier` returns a frozen `SolveResult(status, multiplier, survival_at_multiplier, reason)` with `status` `"solved"` or
`"refused"`. A target that is not bracketed is refused with its reason and carries no number: a target not below the horizon, a target at or
above the base survival period (the base already reaches it), and a target that no multiplier up to the cap reaches (the bank survives even
when every scalable rate is 1, or there is no rate to scale).

## Golden values

One segment, opening 1000, `q = 0.01`, constant `cbc = 100`, `H = 125`:

| Call | Result |
|---|---|
| `survival_vs_multiplier` at `m = 1` | `SP = 11` (`0.99^t < 0.9` first at `t = 11`) |
| `survival_vs_multiplier` over `[1, 2, 3]` | non-increasing |
| `solve_multiplier(target_days=5)` | `multiplier = 2.085164` (`1 - 0.9^(1/5) = 0.0208516`, divided by `q`, rounded up), `survival_at_multiplier = 5`, `survival_just_below` is day 6 |
| one segment of 1000 at `q = 0.01`, `monetisation_capacity(1000, [(1, 0.015), (2, 1.0)], 60)`, `target_days = 5` | `multiplier = 1.5`, `survival_at_multiplier = 1`, `survival_just_below` survives the horizon: the curve steps past the target |
| `solve_multiplier(target_days=125)` | refused: not below the horizon |
| `solve_multiplier(target_days=100)` | refused: the base `SP` is 11 |

The tests recompute the survival period day by day on the stock (no engine function) and the multiplier from the closed form
`(1 - (1 - c/A)^(1/T)) / q` for a constant capacity `c`.

## Conventions

The target, the rounding to six decimals and the round-trip tolerance (the named parameter `round_trip_tolerance_days`, applied by the
validation test `reverse_stress_round_trip`) are conventions. The multiplier is applied to every segment, and a rate already at 1 stays.

## Limits

- The multiplier answers "how much faster must the run-off be", not "how likely is that". It is a distance to a stated target in the
  stated scenario, not a probability and not a forecast.
- The cap `1 / max(q < 1)` is a modelling bound (a share cannot leave faster than all of it): beyond it the point is refused rather than
  clipped silently.
- The survival period is an integer day. With a capacity that grows over time (a monetisation schedule) `SP(m)` can step past the
  target, for example from "survives the horizon" straight to day 1, so no multiplier gives exactly the target. `solve_multiplier`
  then returns the threshold multiplier `m*` (the smallest `m` with `SP(m) <= target`), `survival_at_multiplier` below the target
  and `survival_just_below`, the `SurvivalResult` at `m* - 1e-6`. The round trip is the threshold property `SP(m*) <= target` and
  `SP(m* - 1e-6) > target`; `reverse_stress_round_trip(..., survival_just_below=...)` tests it and its notes give the gap `target - SP(m*)`.

## Sources

- Dossier 05, section E.5, rung 5 (line 279) and E.2 (BCBS Principle 10): https://www.bis.org/publ/bcbs144.htm
- SciPy, `scipy.optimize.brentq`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
