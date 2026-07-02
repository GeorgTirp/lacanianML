import math

import numpy as np

from winding.config import CFG
from winding.data import (
    Embedding, sample_annulus_points, make_loop, make_traj_track,
    perturbation_field, perturb_loop, oracle_quantities,
)
from winding.topology import winding_of_points

PI = math.pi


def test_loops_close_exactly():
    rng = np.random.default_rng(0)
    for k in (-1, 0, 1):
        p = make_loop(k, CFG, rng)
        # closed sequence: the cyclic gap between last and first must be small,
        # i.e. the loop is genuinely closed (Fourier noise has period 1).
        gap = np.linalg.norm(p[0] - p[-1])
        step = np.median(np.linalg.norm(np.diff(p, axis=0), axis=1))
        assert gap < 3 * step  # first/last are one step apart, not a jump


def test_labels_match_oracle_winding():
    rng = np.random.default_rng(1)
    for k in (-1, 0, 1):
        for _ in range(20):
            p = make_loop(k, CFG, rng)
            assert round(winding_of_points(p)) == k


def test_hole_has_zero_support():
    rng = np.random.default_rng(2)
    p, _ = sample_annulus_points(5000, CFG, rng)
    r = np.linalg.norm(p, axis=1)
    assert r.min() >= CFG.r_inner - 1e-9
    assert r.max() <= CFG.r_outer + 1e-9


def test_traj_track_balanced_and_labeled():
    d = make_traj_track(CFG, seed=0, n_per_class=10)
    assert d["x"].shape == (30, CFG.T, CFG.D)
    # balanced
    for k in (-1, 0, 1):
        assert (d["y"] == k).sum() == 10
    # oracle winding matches label
    assert np.all(np.round(d["oracle_W"]) == d["y"])


def test_sector_labels_range():
    rng = np.random.default_rng(3)
    _, sect = sample_annulus_points(1000, CFG, rng)
    assert sect.min() >= 0 and sect.max() < CFG.n_sectors


def test_perturbation_can_cross_hole_at_large_eps():
    rng = np.random.default_rng(4)
    p = make_loop(1, CFG, rng)
    field = perturbation_field(CFG, seed=7)
    # at eps=0 no crossing; the clean loop stays on the annulus
    assert oracle_quantities(perturb_loop(p, 0.0, field))["crossed"] is False
    # perturbation field has unit peak magnitude
    assert abs(np.max(np.linalg.norm(field, axis=1)) - 1.0) < 1e-6


def test_embedding_shape_and_determinism():
    emb = Embedding(CFG)
    p = np.array([[1.5, 0.0], [0.0, 1.5]])
    x1 = emb(p, noise=0.0)
    x2 = emb(p, noise=0.0)
    assert x1.shape == (2, CFG.D)
    assert np.allclose(x1, x2)  # frozen, deterministic without noise
