import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn

from driveplast.model import MLP, flat_grad
from driveplast.curvature import hvp, top_k_subspace
from driveplast.drive import Drive
from driveplast.probe import probe_acc
from driveplast.data import PermutedMNIST


def test_hvp_matches_linear_mse_hessian():
    torch.manual_seed(0)
    d = 8; N = 64
    lin = nn.Linear(d, 1, bias=False)
    x = torch.randn(N, d); y = torch.randn(N, 1)
    lossf = lambda pred, t: ((pred - t) ** 2).mean()
    v = torch.randn(d)
    Hv = hvp(lin, lossf, x, y, v)
    Hv_true = (2.0 / N) * (x.T @ x) @ v          # Hessian of MSE wrt weight = 2/N XᵀX
    assert torch.allclose(Hv, Hv_true, atol=1e-4), (Hv[:3], Hv_true[:3])


def test_top_k_recovers_dominant_curvature():
    torch.manual_seed(0)
    d = 12; N = 200
    # make one input direction dominate the curvature (XᵀX top eigvec known)
    x = torch.randn(N, d); x[:, 0] *= 6.0
    y = torch.randn(N, 1)
    lin = nn.Linear(d, 1, bias=False)
    lossf = lambda p, t: ((p - t) ** 2).mean()
    V = top_k_subspace(lin, lossf, x, y, k=1, iters=8)
    # top Hessian eigenvector should align with e_0 (the amplified input dim)
    assert abs(float(V[0, 0])) > 0.9


def test_drive_orthogonal_to_gradient_and_unit():
    torch.manual_seed(0)
    n = 500
    g = torch.randn(n)
    dr = Drive(n, mode="drive", seed=99)         # seed != g's seed (independent dirs)
    D, leak = dr.step(g, V=None)
    assert abs(float(D @ (g / g.norm()))) < 1e-5       # D ⊥ g
    assert abs(float(D.norm()) - 1.0) < 1e-5           # unit


def test_drive_confinement_removes_S_hi_component():
    torch.manual_seed(0)
    n = 500
    g = torch.randn(n)
    V, _ = torch.linalg.qr(torch.randn(n, 5))          # a fake committed subspace
    D_drive, leak_drive = Drive(n, mode="drive", seed=1).step(g, V)
    D_iso, leak_iso = Drive(n, mode="iso", seed=1).step(g, V)
    # DRIVE has ~zero component in V; ISO keeps its component
    assert float((V.T @ D_drive).norm()) < 1e-5
    assert float((V.T @ D_iso).norm()) > 1e-3


def test_probe_recovers_from_trained_trunk():
    stream = PermutedMNIST(n_tasks=1, n_train=1000, n_test=1000, seed=0)
    xtr, ytr, xte, yte = stream.task(0)
    m = MLP(width=100, depth=2)
    opt = torch.optim.SGD(m.parameters(), lr=0.3); lossf = nn.CrossEntropyLoss()
    for _ in range(400):
        idx = torch.randint(0, len(xtr), (128,))
        loss = lossf(m(xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    assert probe_acc(m, xtr, ytr, xte, yte, seed=0) > 0.8
