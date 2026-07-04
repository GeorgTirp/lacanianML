"""Small MLP with feature access + plasticity mediators."""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, width=100, depth=2, n_in=784, n_out=10):
        super().__init__()
        layers = [nn.Linear(n_in, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.body = nn.Sequential(*layers)          # ends on ReLU -> penultimate feats
        self.head = nn.Linear(width, n_out)
        self.width = width

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))

    @torch.no_grad()
    def mediators(self, x):
        """dead-unit fraction (ReLU units never active on batch), participation
        ratio of penultimate features, over a batch x."""
        h = x
        dead = []
        for l in self.body:
            h = l(h)
            if isinstance(l, nn.ReLU):
                dead.append(float((h.sum(0) == 0).float().mean()))
        f = (h - h.mean(0, keepdim=True)).cpu().numpy()
        cov = (f.T @ f) / max(len(f) - 1, 1)
        lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
        pr = float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-12)) if lam.sum() > 0 else 0.0
        return dict(dead=float(np.mean(dead)), rank=pr)


def flat_params(model):
    return torch.cat([p.reshape(-1) for p in model.parameters()])


def flat_grad(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()])


def add_flat_(model, vec, scale):
    """theta += scale * vec, in place, from a flat vector."""
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.add_(scale * vec[i:i + n].view_as(p))
            i += n
