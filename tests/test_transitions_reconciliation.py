"""RV-415: the cohort estimator reconciles with transitionMatrix 0.5.1, the optional reference (never in the core).

The reference is Open Risk's ``transitionMatrix`` (Apache-2.0), installed with the ``transitions`` extra. The cohort
estimator is compared on one seeded rating panel with withdrawals, element-wise within 1e-12, once the package's
conventions are aligned (each one is a test below, with both values where they cannot be aligned). The package has no
withdrawn-rating treatment, no maximum likelihood generator from paths and its Aalen-Johansen estimator counts the
risk set after the exits; those are findings, stated by the tests that pin them.

Package functions called, with their source in the installed 0.5.1 (``transitionMatrix/estimators/``):
``CohortEstimator.fit`` (cohort_estimator.py l.48; the average matrix l.183, the per-period matrices
``matrix_set``), ``AalenJohansenEstimator.fit`` (aalen_johansen_estimator.py l.41; the risk set l.189).
"""

from __future__ import annotations

import warnings
from importlib import metadata

import numpy as np
import pytest

from riskvalidation.transitions.estimators import aalen_johansen, cohort, pooled_cohort

REASON = "transitionMatrix is the optional reference (transitions extra)"

cohort_module = pytest.importorskip(
    "transitionMatrix.estimators.cohort_estimator", reason=REASON
)
aj_module = pytest.importorskip(
    "transitionMatrix.estimators.aalen_johansen_estimator", reason=REASON
)
statespace_module = pytest.importorskip("transitionMatrix.statespaces.statespace", reason=REASON)
pd = pytest.importorskip("pandas", reason="pandas comes with the transitions extra")

TOL = 1e-12
M = 5  # four grades and default (the last state)
T = 6  # six observation dates, five periods


# ------------------------------------------------------------------------------------------------------------------
# the conversion of one panel into the form each estimator takes
# ------------------------------------------------------------------------------------------------------------------


def _panel(n: int = 600, seed: int = 20261020) -> np.ndarray:
    """A seeded rating panel: row i is entity i's state at the T dates, -1 once its rating is withdrawn.

    Four grades (0 best) and default (M - 1, absorbing, never withdrawn); the withdrawal probability per period is 6
    percent for a performing entity. The last entity is set to a path that never withdraws and ends in a grade (see the
    finding on the package's last record).
    """
    p = np.array(
        [
            [0.90, 0.07, 0.02, 0.009, 0.001],
            [0.05, 0.85, 0.07, 0.02, 0.01],
            [0.02, 0.08, 0.80, 0.07, 0.03],
            [0.00, 0.03, 0.10, 0.77, 0.10],
            [0.00, 0.00, 0.00, 0.00, 1.00],
        ]
    )
    rng = np.random.default_rng(seed)
    panel = np.full((n, T), -1, dtype=int)
    panel[:, 0] = rng.integers(0, M - 1, n)
    for t in range(1, T):
        for i in range(n):
            prev = panel[i, t - 1]
            if prev < 0:
                continue
            if prev != M - 1 and rng.random() < 0.06:
                continue  # withdrawn between t - 1 and t
            panel[i, t] = rng.choice(M, p=p[prev])
    panel[-1] = [1, 1, 2, 1, 1, 2]
    return panel


def _engine_input(panel: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The engine's form: per period, the M x M count table (state at date k by state at date k + 1) and the vector of
    ratings withdrawn in the period by start state (observed at k, absent at k + 1)."""
    tables, withdrawn = [], []
    for k in range(T - 1):
        a, b = panel[:, k], panel[:, k + 1]
        both = (a >= 0) & (b >= 0)
        table = np.zeros((M, M))
        np.add.at(table, (a[both], b[both]), 1.0)
        gone = (a >= 0) & (b < 0)
        w = np.bincount(a[gone], minlength=M).astype(float)
        tables.append(table)
        withdrawn.append(w)
    return tables, withdrawn


def _package_cohort(panel: np.ndarray, sentinel: bool = True):
    """The package's form: a long table (ID, Time, State) of the observed records only, sorted by ID then Time, which
    ``CohortEstimator.fit`` (cohort_estimator.py l.48) reads with the dates as the integers 0 to T - 1 and the cohort
    bounds as the list of dates. A withdrawn rating simply has no record after its last one. The sentinel is an entity
    in default at the last two dates, appended last, so the package's handling of the final record (l.150 to l.155)
    falls on a row whose unit-vector result is unchanged."""
    ids, times, states = np.nonzero(panel >= 0)[0], np.nonzero(panel >= 0)[1], panel[panel >= 0]
    rows = list(zip(ids.tolist(), times.tolist(), states.tolist(), strict=True))
    if sentinel:
        rows += [(panel.shape[0], T - 2, M - 1), (panel.shape[0], T - 1, M - 1)]
    data = pd.DataFrame(rows, columns=["ID", "Time", "State"])
    space = statespace_module.StateSpace(definition=[(str(i), str(i)) for i in range(M)])
    ci = {"method": "goodman", "alpha": 0.05}
    est = cohort_module.CohortEstimator(states=space, cohort_bounds=list(range(T)), ci=ci)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # the package's confidence interval divides by empty rows
        est.fit(data)
    return est


def _tally(est) -> np.ndarray:
    """The package's start tally by state, summed over the periods (the denominators of its average matrix)."""
    return np.sum(est.count_normalization[: T - 1], axis=0).astype(float)


def test_reference_is_the_pinned_version():
    assert metadata.version("transitionMatrix") == "0.5.1"


# ------------------------------------------------------------------------------------------------------------------
# the cohort estimator on one panel with withdrawals
# ------------------------------------------------------------------------------------------------------------------


def test_cohort_state_treatment_equals_package_per_period_and_pooled():
    panel = _panel()
    tables, withdrawn = _engine_input(panel)
    pkg = _package_cohort(panel)
    # per period: the engine keeps the withdrawn as a last state; the package leaves their mass out of the row sum
    for k in range(T - 1):
        eng = cohort(tables[k], withdrawn[k], treatment="state")["matrix"][:M, :M]
        np.testing.assert_allclose(eng[: M - 1], pkg.matrix_set[k][: M - 1], rtol=0, atol=TOL, err_msg=f"period {k}")
    # pooled: Anderson and Goodman (2.8) is the package's count-averaged matrix
    pooled = pooled_cohort(tables, withdrawn, treatment="state")["matrix"][:M, :M]
    np.testing.assert_allclose(pooled, pkg.average_matrix, rtol=0, atol=TOL)
    # the withdrawn column is what the package's rows lack
    gone = pooled_cohort(tables, withdrawn, treatment="state")["matrix"][:M, M]
    np.testing.assert_allclose(gone, 1.0 - pkg.average_matrix.sum(axis=1), rtol=0, atol=TOL)


def test_cohort_exclude_treatment_is_the_package_matrix_renormalised():
    """The package has no exclude treatment; the industry standard (withdrawn removed, rows renormalised) is its
    block renormalised on the performing columns, which is how the engine defines it (derived equivalence)."""
    panel = _panel()
    tables, withdrawn = _engine_input(panel)
    pkg = _package_cohort(panel)
    eng = pooled_cohort(tables, withdrawn, treatment="exclude")["matrix"]
    block = pkg.average_matrix[: M - 1]
    np.testing.assert_allclose(eng[: M - 1], block / block.sum(axis=1, keepdims=True), rtol=0, atol=TOL)
    for k in range(T - 1):
        eng_k = cohort(tables[k], withdrawn[k], treatment="exclude")["matrix"][: M - 1]
        blk = pkg.matrix_set[k][: M - 1]
        np.testing.assert_allclose(eng_k, blk / blk.sum(axis=1, keepdims=True), rtol=0, atol=TOL, err_msg=f"period {k}")


def test_default_row_is_absorbing_in_both_when_the_panel_keeps_default_absorbing():
    """Pooled, the package's default row is the observed default to default share, one here; the engine sets it."""
    panel = _panel()
    tables, withdrawn = _engine_input(panel)
    pkg = _package_cohort(panel)
    unit = np.eye(M)[M - 1]
    np.testing.assert_allclose(pkg.average_matrix[M - 1], unit, rtol=0, atol=TOL)
    np.testing.assert_allclose(pooled_cohort(tables, withdrawn)["matrix"][M - 1], unit, rtol=0, atol=TOL)


def test_reinstated_default_is_a_convention_difference_with_both_values():
    """The package reports what the panel shows after default (a reinstatement); the engine forces the unit vector."""
    panel = _panel()
    d = np.argwhere(panel[:, 1] == M - 1)[0, 0]  # an obligor in default at date 1
    panel[d, 2:] = 0  # reinstated to the best grade from date 2
    tables, withdrawn = _engine_input(panel)
    pkg = _package_cohort(panel)
    row = pkg.average_matrix[M - 1]
    assert row[M - 1] < 1.0 and row[0] > 0.0  # the package: the reinstatement is a migration out of default
    np.testing.assert_allclose(pooled_cohort(tables, withdrawn)["matrix"][M - 1], np.eye(M)[M - 1], rtol=0, atol=TOL)
    # engine counts (not the forced row) hold the same reinstatement the package saw
    assert pooled_cohort(tables, withdrawn)["counts"][M - 1, 0] == pytest.approx(row[0] * _tally(pkg)[M - 1])


def test_empty_row_is_nan_in_the_engine_and_zero_in_the_package():
    """A grade nobody occupies: the engine returns NaN and lists the row, the package fills zeros. Both documented."""
    panel = _panel()
    panel[panel == 2] = 1  # grade 2 is never occupied
    tables, withdrawn = _engine_input(panel)
    pkg = _package_cohort(panel)
    out = pooled_cohort(tables, withdrawn, treatment="state")
    assert out["empty_rows"] == [2]
    assert np.all(np.isnan(out["matrix"][2]))
    np.testing.assert_array_equal(pkg.average_matrix[2], np.zeros(M))
    keep = [0, 1, 3]
    np.testing.assert_allclose(out["matrix"][keep][:, :M], pkg.average_matrix[keep], rtol=0, atol=TOL)


def test_package_double_counts_the_last_record_of_its_last_entity():
    """Finding. cohort_estimator.py l.150 to l.155 handle the last record again after the loop of l.138 has already
    counted its transition: the final migration is counted twice and the final state is added to the start tally of
    the period before. Unguarded, the package differs from the engine by more than 1e-6; the engine's own counts, with
    that one migration added again to its numerator and that one state added to the tally of the final state's row,
    give the package's matrix to 1e-12; with a sentinel entity (an absorbing default, whose rows are the unit vector
    in both) as the last record the package agrees with the engine without any adjustment."""
    panel = _panel()
    tables, withdrawn = _engine_input(panel)
    raw = _package_cohort(panel, sentinel=False)
    exact = pooled_cohort(tables, withdrawn, treatment="state")["matrix"][:M, :M]
    assert np.max(np.abs(raw.average_matrix - exact)) > 1e-6
    last = panel[-1]
    assert last[-1] >= 0 and last[-2] >= 0 and last[-1] != M - 1  # both records present, no default
    a, b = last[-2], last[-1]
    pooled = pooled_cohort(tables, withdrawn, treatment="state")
    num = pooled["counts"][:M, :M].copy()
    size = pooled["row_sizes"][:M].copy()
    num[a, b] += 1.0  # the migration a to b counted a second time, in row a's numerator
    size[b] += 1.0  # and the final state b added to the start tally of row b
    np.testing.assert_allclose(raw.average_matrix, num / size[:, None], rtol=0, atol=TOL)
    # the sentinel keeps the package clear of it: the last record is then a default that stays in default
    np.testing.assert_allclose(_package_cohort(panel).average_matrix, exact, rtol=0, atol=TOL)


# ------------------------------------------------------------------------------------------------------------------
# what the package does not have, and the Aalen-Johansen difference
# ------------------------------------------------------------------------------------------------------------------


def test_package_has_no_maximum_likelihood_generator_from_paths():
    """duration_generator (q_ij = K_ij / S_i from the rating paths) has no equivalent: the package's only generator is
    the matrix logarithm of a transition matrix (model.py l.231), a different estimator; its KaplanMeierEstimator is
    an empty class (kaplan_meier_estimator.py l.20)."""
    from transitionMatrix.estimators.kaplan_meier_estimator import KaplanMeierEstimator

    assert not hasattr(KaplanMeierEstimator, "fit")
    import pkgutil

    import transitionMatrix.estimators as est

    modules = {m.name for m in pkgutil.iter_modules(est.__path__)}
    assert modules == {"aalen_johansen_estimator", "cohort_estimator", "kaplan_meier_estimator", "simple_estimator"}


def test_aalen_johansen_package_counts_the_risk_set_after_the_exits():
    """Finding. Four obligors start in grade 0 and one moves to grade 1 at time 2. The Nelson-Aalen increment is
    1 / 4 (at risk just before the move); the engine returns it. The package divides by the risk set after the exit
    (aalen_johansen_estimator.py l.189, l.201), 1 / 3. The padding rows below only give the package the initial states
    and the two later time points its loops skip; they are not events."""
    seg = {
        "obligor": np.array([0, 0, 1, 2, 3]),
        "start": np.array([0.0, 2.0, 0.0, 0.0, 0.0]),
        "stop": np.array([2.0, 5.0, 5.0, 5.0, 5.0]),
        "state": np.array([0, 1, 0, 0, 0]),
        "to": np.array([1, -1, -1, -1, -1]),
    }
    eng = aalen_johansen(seg, 2, 0.0, 5.0)["matrix"]
    assert eng[0, 1] == 0.25 and eng[0, 0] == 0.75
    rows = [(i, 0.0, 0, 0) for i in range(4)] + [(0, 2.0, 0, 1), (1, 3.0, 0, 0), (2, 4.0, 0, 0), (3, 5.0, 0, 0)]
    data = pd.DataFrame(rows, columns=["ID", "Time", "From", "To"]).sort_values("Time", kind="stable")
    states = statespace_module.StateSpace(definition=[("0", "0"), ("1", "1")])
    est = aj_module.AalenJohansenEstimator(states=states)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data, labels={"Time": "Time", "From": "From", "To": "To", "ID": "ID"})
    pkg = est.etm[:, :, -1]
    assert pkg[0, 1] == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert abs(pkg[0, 1] - eng[0, 1]) > 0.08
