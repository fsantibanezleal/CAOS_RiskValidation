# Fixtures of the U9-E1 regulatory tests

- `calculators-fixture.json`: the regulatory-calculators fixture of Contraste (`docs/contracts/fixtures/regulatory/calculators-fixture.json`,
  branch `task/u9-s`, commit b9543e9, after its Opus review's fix round). It holds the five CRE99 netting sets with their transcribed trade
  inputs, the RBC20.13 floor examples, the invariant cases, the parameter tables and the 79-cell CRE20 grid. Copied byte for byte.
- `c12-blocks.json`: the `sa_ccr`, `ba_cva` and `girr` blocks of the four C12 variant fixtures of the same commit
  (`docs/contracts/fixtures/c12/variant-swap-exposure-<variant>-fixture.json`, variants csa-mpor-10d, csa-mpor-20d, netting-set,
  single-swap), extracted unchanged.

The tests start from the trades and the formulas; the fixture values are what they are compared with.
