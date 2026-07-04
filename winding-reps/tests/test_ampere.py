import numpy as np
import torch

from winding.config import CFG
from winding import ampere as A
from winding.topology import winding_of_points
from winding.uncertainty import train_ensemble_regression


def test_components_finds_two_islands():
    mask = np.zeros((30, 30), bool)
    mask[5:10, 5:10] = True          # island 1
    mask[20:25, 20:25] = True        # island 2
    comps = A._components(mask)
    big = [c for c in comps if len(c) >= 3]
    assert len(big) == 2


def test_charge_integration_ratio_two_blobs():
    # synthetic disagreement: two gaussian blobs, blob1 4x the excess mass of blob2
    gn = 90; axis = np.linspace(-A.DOMAIN, A.DOMAIN, gn)
    gx, gy = np.meshgrid(axis, axis); grid = np.stack([gx, gy], -1)
    d1 = np.linalg.norm(grid - np.array([-1.2, 0]), axis=2)
    d2 = np.linalg.norm(grid - np.array([1.6, 0]), axis=2)
    dis = 0.001 + 0.08 * np.exp(-(d1 ** 2) / 0.3) + 0.02 * np.exp(-(d2 ** 2) / 0.3)
    bg = 0.001; cell = (2 * A.DOMAIN / (gn - 1)) ** 2
    w = A.World()
    oq = A.oracle_charges_from_map(grid, dis, bg, w, cell)
    # constructed excess-mass ratio is (0.08/0.02)=4x (equal widths) -> expect ~4
    assert 2.5 < oq[0] / oq[1] < 6.0, oq


def test_loop_families_have_correct_linking():
    w = A.default_worlds()[1]
    for kind, l1, l2 in [("A1", 1, 0), ("A2", 0, 1), ("both", 1, 1), ("neither", 0, 0)]:
        for p in A.deformed_loops(kind, w, n_reps=3, seed=0):
            assert round(winding_of_points(p, w.c1)) == l1, (kind, "c1")
            assert round(winding_of_points(p, w.c2)) == l2, (kind, "c2")


def test_estimate_charges_ordering_and_localization():
    w = A.default_worlds()[1]
    reg = A.make_regression_data(w, CFG, seed=0, n=2500)
    models = train_ensemble_regression(reg, CFG, seed=0)
    charged, mp = A.estimate_charges(models, w, CFG, gn=70)
    assert len(charged) == 2
    (c1, q1), (c2, q2) = charged
    assert q1 > q2                                   # A1 more charged
    assert np.linalg.norm(c1 - np.array(w.c1)) < 0.6  # localized on A1
    assert np.linalg.norm(c2 - np.array(w.c2)) < 0.8  # localized on A2


def test_install_produces_unit_windings():
    w = A.default_worlds()[1]
    m = A.TwoHead(CFG)
    A.install(m, w, [np.array(w.c1), np.array(w.c2)], CFG, steps=400, seed=0)
    wind = A.verify_winding(m, w, [np.array(w.c1), np.array(w.c2)], CFG)
    assert all(round(x) == 1 for x in wind), wind
