"""Shared helpers for experiments: path setup, dirs, seeding."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)


def seed_all(seed):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


def fig_path(name):
    return os.path.join(FIGS, name)


def result_path(name):
    return os.path.join(RESULTS, name)
