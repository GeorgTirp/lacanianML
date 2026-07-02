import numpy as np

from winding.uncertainty import convex_hull, in_hull, interior_peak


def test_convex_hull_square():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    hull = convex_hull(pts)
    # interior point excluded, 4 corners retained
    assert len(hull) == 4


def test_in_hull():
    hull = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    q = np.array([[1, 1], [3, 1], [-1, 1]], dtype=float)
    mask = in_hull(q, hull)
    assert mask.tolist() == [True, False, False]


def test_interior_peak_on_synthetic_map():
    # synthetic disagreement: high near origin (interior singularity) AND high
    # far outside; training data is an annulus so its hull is a disk of r~2.
    lim, gn = 2.5, 81
    axis = np.linspace(-lim, lim, gn)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    r = np.linalg.norm(grid, axis=1)
    interior = np.exp(-(r ** 2) / 0.2)            # sharp peak at origin
    exterior = np.exp(-((r - 3.0) ** 2) / 0.3)    # ring outside the hull
    dis = interior + exterior

    theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    train_pts = np.stack([1.5 * np.cos(theta), 1.5 * np.sin(theta)], axis=1)

    c_hat, info = interior_peak(grid, dis, train_pts, top_frac=0.05)
    # interior peak must be recovered near the origin, not pulled to the exterior
    assert np.linalg.norm(c_hat) < 0.3
    assert info["n_high"] > 0
