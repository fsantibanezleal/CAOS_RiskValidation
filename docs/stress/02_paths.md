# 02. Scenario paths

`riskvalidation.stress.paths` handles the paths a scenario publishes: the Federal Reserve gives baseline and severely adverse paths for
13 quarters from a jump-off quarter (dossier 08 D.1), the EBA gives annual paths for three years (addendum F6), and NGFS runs hold
deviations from a baseline run (addendum F8). A path keeps the unit of its source (percent for the Federal Reserve and EBA paths).

## Transposition

A scenario published after a model's reporting date cannot be used as if it were known then, so the engine transposes the shape of the
path, its change from its jump-off, onto the last observed value (decision Q6; the rule of `cases/c06/macro.py`, R14 of the unit
sizing):

$$
x^{\text{target}}_{k} = x^{\text{target}}_{0} + \big(L_{k} - L_{0}\big),\qquad k = 1, \dots, n,
$$

with $L_0$ the source jump-off (`source_jump_off`) and $x^{\text{target}}_{0}$ the target value (`target_value`); periods are aligned by their
distance from the jump-off, so the first difference of the result is that of the source. `n` above the length of `levels` is refused
(extend with `revert_to_baseline`).

Golden value: `levels = (6, 8, 10)`, `source_jump_off = 5`, `target_value = 3`, `n = 3` give $(4, 6, 8)$.

A level path of a growth regressor (the NGFS GDP level, `baseline-times-percent-deviation`): the transposed quarters of year $y$ all
carry that year's growth in percent, $(\text{level}_y / \text{level}_{y-1} - 1) \times 100$, with $\text{level}_0$ the source jump-off level; the
case checker `semantics.py` verifies it. The engine does not compute that growth.

## Reversion to the baseline

Past the published horizon of $m$ quarters the path reverts linearly to the baseline tail over $R$ quarters:

$$
w_k = \min\!\Big(1, \frac{k - m}{R}\Big),\qquad x_k = x_m + w_k\,\big(\bar x - x_m\big),\qquad k > m,
$$

with $\bar x$ the baseline tail (`baseline_tail`) and $R$ = `reversion_quarters`. `REVERSION_QUARTERS = 12` is the product convention of
`cases/c06/macro.py`, a convention and not a source value; it is the models field `capital_engine.reversion_quarters`. The first $m$
values are unchanged and the path reaches the baseline tail at $k - m = R$ and stays there. The EBA note reverts the adverse
parameters linearly over six years after the horizon (paragraph 133; the paragraph number 132 is UNVERIFIED in the record), a different
rule that the three-year C18 does not need.

Golden value: `path = (8.0,)`, `baseline_tail = 5.0`, `n = 14`, `reversion_quarters = 12` give, at $k - m = 1, 6, 12, 13$, the values
$7.75$, $6.5$, $5.0$, $5.0$.

## Annual to quarterly

`annual_to_quarterly(year_end_levels, jump_off_level)` joins the year-end levels linearly from the jump-off level: quarter $k$
(1 to 4) of year $y$ is

$$
x_{y,k} = x_{y-1} + \big(x_{y} - x_{y-1}\big)\,\frac{k}{4},
$$

and the fourth quarter equals the year-end level exactly. Whether the EBA workbook values are year averages or year-end values is
UNVERIFIED in the research record (F6); the engine treats them as year-end levels and the bake states it.

Golden value: `year_end_levels = (6, 8)`, `jump_off_level = 4` give $(4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)$.

## Deviation paths

NGFS scenario runs hold deviations from the Baseline run (addendum F8): a level is the baseline plus the deviation for rates and
unemployment (`plus`), or the baseline times the percent difference for GDP (`percent`):

$$
x = b + d,\qquad x = b\left(1 + \frac{d}{100}\right).
$$

Golden values: the NGFS Delayed transition, United States, 2031 unemployment, baseline $4.442$ plus deviation $0.106$ is $4.548$; a GDP
baseline $26000.0$ with a percent difference of $-4.239$ is $24897.86$ (computed; the 2031 baseline GDP level is a fixture value, the
record gives 25481.079 for 2030 only).

## The vintage check

`check_vintage(vintage_date, observation_date)` rejects a path whose vintage postdates the model's observation date (SDD l.169, the
`macro_path` family rule); equal dates pass. `("2026-02-04", "2026-08-31")` returns `None`; `("2027-01-01", "2026-08-31")` raises.

## The engine path rule against C06

The test `test_paths_reproduce_the_c06_committed_output` rebuilds the committed macro paths of C06's reporting date 2020-06 (40
quarters of three variables for the baseline, adverse and severely adverse scenarios, copied unchanged into
`tests/stress/fixtures/c06_rd_2020_06_macro_paths.json`) from their first 13 quarters by `revert_to_baseline` with $R = 12$, and checks
that the adverse shape is the midpoint of the two Federal Reserve shapes. The severely adverse prime rate is not compared: the product
floors it at 0 after building the path, so its unfloored 13th value is not in the committed output. The migration of C06 to the engine
function is backlog item BL-079.

## Limits

- The engine moves a path; it does not decide whether a scenario is plausible. The check that a fit window does not reach 2020 and the
  check that a back-cast window does not overlap the episode are the product's contract and `validation.stress.satellite_backcast`.
- The transposition keeps the published change in the variable's own unit; a variable bounded below (unemployment, a rate) can leave
  its natural range after a large negative shape, and the product floors it (1.0 on unemployment, 0.0 on the prime rate in C06). The
  engine applies no floor.
- Year-average against year-end is UNVERIFIED for the EBA workbook and for the NGFS annual values.

## Sources

- Federal Reserve, 2026 stress test scenarios (dossier 08 D.1), https://www.federalreserve.gov/supervisionreg/dfa-stress-tests-2027.htm
- EBA, 2025 EU-wide stress test, Methodological Note of 2025-01-20,
  https://www.eba.europa.eu/sites/default/files/2025-01/0246e2f3-fa57-47d1-99f2-7a5b80cae509/2025%20EU-wide%20stress%20test%20-%20Methodological%20Note.pdf
- NGFS Climate Scenarios Technical Documentation V5.0 (November 2024),
  https://www.ngfs.net/system/files/2025-01/NGFS%20Climate%20Scenarios%20Technical%20Documentation.pdf
- The U13 research addendum, F6 (the EBA note), F8 (the NGFS variables and US values), in the Contraste research record.
