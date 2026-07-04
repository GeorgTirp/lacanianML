"""Probe-recovery layer (§9): linear probe on frozen trunk features.

Decomposes forgetting into readout-access drift (recoverable) vs true trunk
erosion (not), following the v3b control. Retention measured only through the
live head OVERSTATES forgetting; the drive's signature prediction is near-zero
EROSION (motion confined to the null space of what's learned).
"""
import torch
import torch.nn as nn

from .model import MLP


@torch.no_grad()
def _feats(model, x, bs=2000):
    return torch.cat([model.features(x[i:i + bs]) for i in range(0, len(x), bs)])


def probe_acc(model, xtr, ytr, xte, yte, steps=300, lr=1e-2, seed=0):
    """Held-out accuracy of a fresh LINEAR probe on the frozen trunk's features."""
    ftr = _feats(model, xtr); fte = _feats(model, xte)
    torch.manual_seed(seed)
    head = nn.Linear(ftr.shape[1], int(max(yte.max(), ytr.max())) + 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    N = len(ftr)
    for _ in range(steps):
        idx = torch.randint(0, N, (256,))
        loss = lossf(head(ftr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((head(fte).argmax(1) == yte).float().mean())


def random_trunk_probe(width, depth, xtr, ytr, xte, yte, seed=0):
    """P_rand anchor: same probe on a random-init trunk (the reservoir baseline)."""
    torch.manual_seed(seed + 12345)
    m = MLP(width=width, depth=depth)
    return probe_acc(m, xtr, ytr, xte, yte, seed=seed)
