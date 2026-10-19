"""ML model validation: perturbation robustness and shift resilience, seed stability, explanation checks, effect curves,
generalisation, split conformal prediction, the leakage screen, data quality, the label-free performance estimate, the
champion and challenger comparison and the planted-defect generators.

Modules are imported by path (for example ``from riskvalidation.ml import robustness``); nothing is re-exported here. The
engine imports no learning library: a model enters as a predict callable on arrays, and attributions as arrays computed in
the pipeline. Every function is pure and deterministic (randomness takes a ``seed``), never mutates its inputs and raises
``ValueError`` naming the field when data is missing or invalid. Tolerances, grids, counts and cut points are conventions,
never regulatory thresholds. The engine computes and states no compliance claim. The ``TestResult`` functions are in
``riskvalidation.validation.ml``. Contract: ``engine-ml-api`` 1.0.0.
"""
