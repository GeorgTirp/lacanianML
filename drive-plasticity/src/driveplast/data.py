"""MNIST loader + Permuted-MNIST task stream (no torchvision — raw IDX)."""
import gzip
import os
import struct
import urllib.request

import numpy as np
import torch

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
_FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
          "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]


def _ensure(fname):
    path = os.path.join(_DIR, fname + ".gz")
    if not os.path.exists(path):
        os.makedirs(_DIR, exist_ok=True)
        urllib.request.urlretrieve(f"{_URL}/{fname}.gz", path)
    return path


def _load_idx(fname):
    with gzip.open(_ensure(fname), "rb") as f:
        b = f.read()
    magic, = struct.unpack(">I", b[:4])
    if magic == 2051:
        n, r, c = struct.unpack(">III", b[4:16])
        return np.frombuffer(b[16:], np.uint8).reshape(n, r * c).astype(np.float32) / 255.0
    n, = struct.unpack(">I", b[4:8])
    return np.frombuffer(b[8:], np.uint8).astype(np.int64)


def load_mnist():
    Xtr = torch.tensor(_load_idx(_FILES[0]))
    ytr = torch.tensor(_load_idx(_FILES[1]))
    Xte = torch.tensor(_load_idx(_FILES[2]))
    yte = torch.tensor(_load_idx(_FILES[3]))
    return Xtr, ytr, Xte, yte


class PermutedMNIST:
    """A stream of pixel-permutation tasks. Each task exposes a fixed train/test
    subset under its own permutation. Deterministic given seed."""

    def __init__(self, n_tasks, n_train=2000, n_test=2000, seed=0):
        self.Xtr, self.ytr, self.Xte, self.yte = load_mnist()
        self.n_tasks = n_tasks
        self.n_train = n_train
        self.n_test = n_test
        self.rng = np.random.default_rng(seed)
        self.perms = [self.rng.permutation(784) for _ in range(n_tasks)]
        # fixed train/test index pools per task
        self.tr_idx = [self.rng.permutation(len(self.Xtr))[:n_train] for _ in range(n_tasks)]
        self.te_idx = self.rng.permutation(len(self.Xte))[:n_test]

    def task(self, i):
        p = self.perms[i]
        xtr = self.Xtr[self.tr_idx[i]][:, p]
        ytr = self.ytr[self.tr_idx[i]]
        xte = self.Xte[self.te_idx][:, p]
        yte = self.yte[self.te_idx]
        return xtr, ytr, xte, yte
