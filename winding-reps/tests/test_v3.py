import numpy as np
import torch

from winding.config import CFG
from winding.data import (make_traj_track, radius_class_labels,
                          start_sector_labels, make_point_regression)
from winding.models import PointRegressor, MeanPoolHead, GRUBaseline, PhaseEncoder
from winding.uncertainty import (train_ensemble_regression,
                                 disagreement_map_regression, interior_peak)


# ---- interfering-task labels ----------------------------------------------
def test_radius_class_labels_range_and_binning():
    d = make_traj_track(CFG, seed=0, n_per_class=20)
    lab = radius_class_labels(d["p"], CFG)
    assert set(np.unique(lab)).issubset({0, 1, 2})
    # a loop centered at r~1.5 lands in the middle bin
    mean_r = np.linalg.norm(d["p"], axis=2).mean(axis=1)
    for r, l in zip(mean_r, lab):
        expected = 0 if r < 1.35 else (1 if r < 1.65 else 2)
        assert l == expected


def test_start_sector_labels_match_geometry():
    d = make_traj_track(CFG, seed=1, n_per_class=20)
    lab = start_sector_labels(d["p"], CFG)
    assert lab.min() >= 0 and lab.max() < CFG.n_sectors
    start = d["p"][:, 0, :]
    theta = np.mod(np.arctan2(start[:, 1], start[:, 0]), 2 * np.pi)
    expected = np.floor(theta / (2 * np.pi / CFG.n_sectors)).astype(int)
    assert np.all(lab == expected)


def test_regression_track_shapes():
    rt = make_point_regression(CFG, seed=0)
    assert rt["x"].shape == (CFG.n_points, CFG.D)
    assert rt["y"].shape == (CFG.n_points,)
    # target ~ sin(2 theta): bounded roughly in [-1.5, 1.5]
    assert rt["y"].min() > -2.0 and rt["y"].max() < 2.0


# ---- heads / models --------------------------------------------------------
def test_mean_pool_head_shapes():
    head = MeanPoolHead(in_dim=2, n_classes=3, hidden=16)
    feats = torch.randn(5, CFG.T, 2)
    assert head(feats).shape == (5, 3)


def test_gru_trunk_features_shape():
    net = GRUBaseline(CFG)
    x = torch.randn(4, CFG.T, CFG.D)
    assert net.trunk_features(x).shape == (4, CFG.T, CFG.gru_hidden)


def test_point_regressor_shape():
    net = PointRegressor(CFG)
    assert net(torch.randn(7, CFG.D)).shape == (7,)


# ---- regression EU locates the hole ---------------------------------------
def test_regression_eu_peak_inside_hole():
    # cheap config: smaller grid/ensemble for test speed via monkey values
    import dataclasses
    cfg = dataclasses.replace(CFG, grid_n=41, n_points=1500, ens_epochs=25)
    rt = make_point_regression(cfg, seed=0)
    models = train_ensemble_regression(rt, cfg, seed=0)
    grid, dis, axis = disagreement_map_regression(models, cfg)
    c_hat, info = interior_peak(grid, dis, rt["p"], top_frac=0.1)
    # its only requirement (per exp2 P1b): c_hat lands inside the hole
    assert np.linalg.norm(c_hat) < cfg.r_inner
