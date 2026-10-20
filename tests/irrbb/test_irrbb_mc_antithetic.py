"""BL-074: ``ear_evar_mc_error(..., antithetic=True)`` keeps the mirror pairs of an antithetic path set together in
one batch (contract `engine-irrbb-api`, module simulation)."""

from __future__ import annotations

import numpy as np
import pytest

from riskvalidation.irrbb import simulation as sim


def test_antithetic_batches_hold_whole_pairs():
    p, nb = 26, 4  # half = 13, 3 pairs per batch, pairs 0..11 used, pair 12 dropped
    idx = sim._batch_indices(p, nb, antithetic=True)
    half = p // 2
    assert idx.shape == (nb, 6)
    assert sorted(idx.ravel().tolist()) == sorted(list(range(12)) + [half + i for i in range(12)])
    for row in idx:
        members = set(row.tolist())
        for i in members:
            mirror = i + half if i < half else i - half
            assert mirror in members, (i, mirror)
    # different batches share no pair
    pair_ids = [{i % half for i in row.tolist()} for row in idx]
    for a in range(nb):
        for b in range(a + 1, nb):
            assert not pair_ids[a] & pair_ids[b]
    # contiguous pairs per batch, in order
    assert [sorted(i % half for i in row.tolist())[::2] for row in idx][0] == [0, 1, 2]


def test_contiguous_batches_unchanged():
    idx = sim._batch_indices(41, 4, antithetic=False)
    assert np.array_equal(idx, np.arange(40).reshape(4, 10))


def test_antithetic_false_regression_pinned():
    r = np.random.default_rng(42)
    a, b = r.standard_normal(1000), 3.0 * r.standard_normal(1000)
    o = sim.ear_evar_mc_error(a, b, 0.99, 20, antithetic=False)
    assert o == sim.ear_evar_mc_error(a, b, 0.99, 20)
    assert (o.nii_quantile_se, o.eve_quantile_se) == (0.0904768930814362, 0.20762454258013258)
    o = sim.ear_evar_mc_error(a, b, 0.95, 7, antithetic=False)
    assert (o.nii_quantile_se, o.eve_quantile_se) == (0.0533153010549719, 0.17487615602586162)


def test_antithetic_uses_the_same_batching_for_nii_and_eve():
    r = np.random.default_rng(5)
    half = r.standard_normal(500)
    x = np.r_[half, -half]
    out = sim.ear_evar_mc_error(x, 4.0 * x, 0.99, 10, antithetic=True)
    assert out.eve_quantile_se == pytest.approx(4.0 * out.nii_quantile_se, rel=1e-12)
    assert out == sim.ear_evar_mc_error(x, 4.0 * x, 0.99, 10, antithetic=True)  # deterministic


def test_antithetic_validation():
    x = np.arange(41.0)
    with pytest.raises(ValueError, match="even"):
        sim.ear_evar_mc_error(x, x, 0.99, 4, antithetic=True)
    y = np.arange(30.0)  # half = 15 < 20
    with pytest.raises(ValueError, match="20"):
        sim.ear_evar_mc_error(y, y, 0.99, 20, antithetic=True)
    # the minimum is half == n_batches
    z = np.r_[np.arange(20.0), -np.arange(20.0)]
    assert sim.ear_evar_mc_error(z, z, 0.5, 20, antithetic=True).ear_se >= 0.0


def _nii_of(paths: np.ndarray) -> np.ndarray:
    """A rate-sensitive NII proxy: 10 000 of notional earning the mean monthly short rate over the horizon."""
    return 1.0e4 * paths[:, 1:].mean(axis=1)


def test_antithetic_error_tracks_the_true_quantile_sd(capsys):
    zr, ten = np.array([0.03, 0.04, 0.045]), np.array([1.0, 5.0, 10.0])
    # 10 000 paths: with 2 000 (batches of 100) both batchings sit about 22 percent below the truth, because a batch
    # then holds about one path below the 1 percent quantile; at 10 000 they reach 0.96 and 0.95, at 40 000 1.04 and
    # 1.07 (measured 2026-10-17). On this NII proxy the two batchings differ within noise; keeping the pairs together
    # makes the batches independent by construction, which is what the error assumes.
    p, nb, seeds, level = 10000, 20, 200, 0.99
    q_true, anti, contig = [], [], []
    for seed in range(seeds):
        nii = _nii_of(sim.hw1f_scenarios(zr, ten, 0.1, 0.012, 24, p, seed))
        q_true.append(np.quantile(nii, 1.0 - level))
        anti.append(sim.ear_evar_mc_error(nii, nii, level, nb, antithetic=True).ear_se)
        contig.append(sim.ear_evar_mc_error(nii, nii, level, nb, antithetic=False).ear_se)
    truth, a_mean, c_mean = float(np.std(q_true, ddof=1)), float(np.mean(anti)), float(np.mean(contig))
    with capsys.disabled():
        print(f"\nBL-074 truth (sd of the 99% NII quantile over {seeds} seeds): {truth:.4f}")
        print(f"BL-074 antithetic=True mean error: {a_mean:.4f} (ratio {a_mean / truth:.3f})")
        print(f"BL-074 antithetic=False mean error: {c_mean:.4f} (ratio {c_mean / truth:.3f})")
    assert abs(a_mean / truth - 1.0) < 0.15
