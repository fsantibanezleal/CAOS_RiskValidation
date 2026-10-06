"""Rating-system generators: migration matrices and grade frequencies, with the planted defects of the ECB tests.

The ECB's migration z-tests (2019, section 2.5.5.2) expect transition frequencies that do not increase moving away from
the diagonal; the concentration test (section 2.5.5.3) compares the Herfindahl index of the grade distribution with the
one at development. The generators here draw multinomial data from a known matrix or distribution, so whether the null
holds, and by how much it fails, is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from riskvalidation.generators._base import Check, GeneratorBase, frozen_array, mean_check
from riskvalidation.validation.rating_system import concentration

ECB_2019 = (
    "ECB (2019), Instructions for reporting the validation results of internal models: IRB Pillar I models for credit "
    "risk, sections 2.5.5.2 and 2.5.5.3"
)


def decaying_matrix(k: int, stay: float = 0.8, decay: float = 0.5) -> np.ndarray:
    """A ``k`` x ``k`` migration matrix with ``stay`` on the diagonal and off-diagonal cells falling geometrically,
    ``decay`` per notch, on both sides; every row sums to one and the ECB's null holds strictly when ``decay`` < 1."""
    if k < 2 or not 0.0 < stay < 1.0 or not 0.0 < decay <= 1.0:
        raise ValueError("k >= 2, stay in (0, 1) and decay in (0, 1] are required")
    m = np.zeros((k, k))
    for i in range(k):
        w = np.array([0.0 if j == i else decay ** (abs(i - j) - 1) for j in range(k)])
        m[i] = (1.0 - stay) * w / w.sum()
        m[i, i] = stay
    return m


def neighbour(row: int, col: int) -> int:
    """The cell of the same row one notch closer to the diagonal (the ECB test's comparison cell)."""
    return col + 1 if col < row else col - 1


def break_monotonicity(matrix: Any, row: int, col: int, ratio: float) -> np.ndarray:
    """Set cell (``row``, ``col``), two or more notches from the diagonal, to ``ratio`` times its neighbour closer to
    the diagonal, the mass coming from the diagonal cell. ``ratio`` 1 is the boundary of the ECB null; above 1 the
    null fails."""
    m = np.array(matrix, dtype=float)
    if abs(row - col) < 2:
        raise ValueError("the planted cell must be at least two notches from the diagonal")
    nb = neighbour(row, col)
    delta = ratio * m[row, nb] - m[row, col]
    m[row, col] += delta
    m[row, row] -= delta
    if m[row, row] < 0.0 or m[row, col] < 0.0:
        raise ValueError("the ratio leaves a negative probability")
    return m


def monotone_cells(matrix: Any) -> dict[str, int]:
    """Counts of off-diagonal cells strictly inside, on the boundary of, and outside the ECB null."""
    m = np.asarray(matrix, dtype=float)
    inside = boundary = outside = 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i == j:
                continue
            nb = neighbour(i, j)
            if np.isclose(m[i, j], m[i, nb], rtol=1e-12, atol=1e-15):
                boundary += 1
            elif m[i, j] < m[i, nb]:
                inside += 1
            else:
                outside += 1
    return {"inside": inside, "boundary": boundary, "outside": outside}


@dataclass(frozen=True)
class Migrations(GeneratorBase):
    """``obligors[i]`` obligors start in grade ``i`` and migrate by the rows of ``matrix`` (multinomial)."""

    name: ClassVar[str] = "migrations"
    obligors: Any
    matrix: Any

    def __post_init__(self) -> None:
        n = np.asarray(self.obligors)
        m = np.asarray(self.matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or n.shape != (m.shape[0],):
            raise ValueError("matrix must be square with one obligor count per starting grade")
        if np.any(m < 0) or not np.allclose(m.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError("every row of the matrix must be a probability vector")
        if np.any(n < 0) or not np.all(np.equal(np.mod(n, 1), 0)):
            raise ValueError("obligors must be non-negative integers")
        object.__setattr__(self, "obligors", frozen_array(n, dtype=np.int64))
        object.__setattr__(self, "matrix", frozen_array(m))

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """One multinomial row per starting grade, in grade order."""
        m = np.asarray(self.matrix)
        counts = np.vstack([rng.multinomial(int(n), m[i]) for i, n in enumerate(self.obligors)])
        return {"counts": counts}

    def truth(self) -> dict[str, Any]:
        m = np.asarray(self.matrix)
        n = np.asarray(self.obligors, dtype=float)
        cells = monotone_cells(m)
        return {"matrix": m.tolist(), "expected_counts": (n[:, None] * m).tolist(), "cells": cells,
                "null_holds": cells["outside"] == 0, "reference": ECB_2019}

    def verify(self, n_draws: int = 4_000, seed: int = 20261005) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 306]))
        c = np.stack([self.draw(rng)["counts"] for _ in range(n_draws)])
        exp = np.asarray(self.truth()["expected_counts"])
        k = exp.shape[0]
        return [mean_check(f"count from {i + 1} to {j + 1}", exp[i, j], c[:, i, j])
                for i in range(k) for j in range(k) if exp[i, j] > 0]


def concentrate(probabilities: Any, share: float) -> np.ndarray:
    """Move ``share`` of every grade's mass into the modal grade: (1 - share) p + share e_mode."""
    p = np.asarray(probabilities, dtype=float)
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must lie in [0, 1]")
    out = (1.0 - share) * p
    out[int(np.argmax(p))] += share
    return out


@dataclass(frozen=True)
class GradeFrequencies(GeneratorBase):
    """``obligors`` obligors over grades with the current population's ``probabilities``; ``initial`` is the grade
    distribution at development, given to the test as fixed (the current population when omitted: the boundary of the
    ECB concentration null)."""

    name: ClassVar[str] = "grade_frequencies"
    obligors: int
    probabilities: Any
    initial: Any = None

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        q = p if self.initial is None else np.asarray(self.initial, dtype=float)
        for arr, label in ((p, "probabilities"), (q, "initial")):
            if arr.ndim != 1 or arr.size < 2 or np.any(arr < 0) or not np.isclose(arr.sum(), 1.0, atol=1e-12):
                raise ValueError(f"{label} must be a probability vector over at least two grades")
        if p.shape != q.shape:
            raise ValueError("current and initial distributions must cover the same grades")
        if int(self.obligors) < 1:
            raise ValueError("at least one obligor is required")
        object.__setattr__(self, "obligors", int(self.obligors))
        object.__setattr__(self, "probabilities", frozen_array(p))
        object.__setattr__(self, "initial", frozen_array(q))

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        f = rng.multinomial(self.obligors, np.asarray(self.probabilities))
        return {"freq_current": f, "freq_initial": np.asarray(self.initial)}

    def truth(self) -> dict[str, Any]:
        cv_c, hi_c = concentration(self.probabilities)
        cv_i, hi_i = concentration(self.initial)
        return {"cv_current": cv_c, "hi_current": hi_c, "cv_initial": cv_i, "hi_initial": hi_i,
                "null_holds": hi_c <= hi_i + 1e-15, "reference": ECB_2019}

    def verify(self, n_draws: int = 4_000, seed: int = 20261005) -> list[Check]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 307]))
        f = np.stack([self.draw(rng)["freq_current"] for _ in range(n_draws)])
        p = np.asarray(self.probabilities)
        return [mean_check(f"share of grade {i + 1}", p[i], f[:, i] / self.obligors) for i in range(p.size)]


__all__ = ["ECB_2019", "GradeFrequencies", "Migrations", "break_monotonicity", "concentrate", "decaying_matrix",
           "monotone_cells", "neighbour"]
