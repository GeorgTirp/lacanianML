import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import numpy as np
import torch

from winding.config import CFG
from winding.data import radius_class_labels
from winding.topology import winding_of_points
import exp3b_probe_control as e3b


def test_v3b_task2_not_degenerate():
    d = e3b.make_v3b_traj(CFG, seed=0, n_per_class=100)
    counts = np.bincount(radius_class_labels(d["p"], CFG), minlength=3)
    frac = counts / counts.sum()
    # all three mean-radius classes populated, none trivially dominant (< 0.7)
    assert np.all(counts > 0), counts
    assert frac.max() < 0.7, frac
    # a middle-class predictor must fall well short of the 0.95 fine-tune target
    assert frac.max() < 0.95


def test_v3b_labels_match_oracle_winding():
    d = e3b.make_v3b_traj(CFG, seed=1, n_per_class=30)
    for pp, k in zip(d["p"][:20], d["y"][:20]):
        assert round(winding_of_points(pp)) == int(k)


def test_probe_pipeline_recovers_from_trained_trunk():
    # smoke: a fresh probe on an untouched (well-trained) trunk recovers winding
    # near-perfectly (P_pre sanity). Small + short for speed.
    import dataclasses
    cfg = CFG
    tr = e3b.make_v3b_traj(cfg, 0, 120)
    te = e3b.make_v3b_traj(cfg, 5, 60)
    net, clean = e3b.train_C(cfg, tr, te, seed=0, max_steps=1500)
    assert clean > 0.9                      # trunk actually learned winding
    P = e3b.probe_all(net, cfg, tr, te, seed=0)
    assert P["primary"] > 0.9               # fresh probe recovers it (pipeline OK)
