# 03. Suites, errors and reproducibility

## Running a battery

```python
from riskvalidation import run_suite, default_policy

run = run_suite(
    [(test_a, {"n": 1200, "d": 31, "pd": 0.021}), (test_b, {"scores": s, "defaults": y})],
    policy=default_policy(),
    model_id="retail-pd-v3",
    segment="cards",
)
run.counts()   # results per light
run.worst()    # the most severe light present (error outranks red)
run.table()    # JSON-safe rows for a report
```

The runner injects the shared policy, model id and segment into every task. A test that raises (an empty
segment, inconsistent inputs) becomes an `error` result with the exception text, and the remaining tests run:
a validator needs the whole picture, and a battery that stops at its first failure hides the rest.

## One result, one sample

`inputs_hash` is a SHA-256 over the test's inputs in order. Arrays contribute their dtype, shape and bytes in
little-endian order; mappings are hashed with sorted keys; scalars through canonical JSON. Equal inputs give the
same hash on every platform; changing a value, the dtype, the shape or the order changes it. A validation result
is valid only for the sample it tested, and the hash makes that checkable when a report is reviewed later.

## Two lanes, one wheel

The core imports NumPy and SciPy only. The same wheel runs in an offline Python pipeline and in the browser under
Pyodide, which ships NumPy and SciPy; a test recomputed live in a page therefore uses the code that produced the
published result, not a port of it.
