"""Continual-learning baselines: shrink-perturb, L2-init, CBP (selective reinit)."""
import torch
import torch.nn as nn


def shrink_perturb_(model, lam, sigma, gen):
    """theta <- (1-lam) theta + sigma * noise  (isotropic; Ash & Adams 2020)."""
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1 - lam).add_(sigma * torch.randn(p.shape, generator=gen))


def l2_init_grad_(model, init_params, lam):
    """Add lam*(theta - theta_init) to .grad (regularize toward initialization)."""
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), init_params):
            if p.grad is not None:
                p.grad.add_(lam * (p.data - p0))


class CBP:
    """Continual Backprop (Dohare et al. 2024), simplified: track a running
    utility per hidden unit and periodically reinitialize the lowest-utility
    units (reset incoming weights, zero outgoing weights). SOTA plasticity method.
    """

    def __init__(self, model, reinit_period=100, reinit_frac=0.01, decay=0.99):
        self.period = reinit_period; self.frac = reinit_frac; self.decay = decay
        self.lins = [m for m in model.body if isinstance(m, nn.Linear)]
        self.nexts = self.lins[1:] + [model.head]
        self.util = [torch.zeros(l.out_features) for l in self.lins]
        self.age = [torch.zeros(l.out_features) for l in self.lins]

    @torch.no_grad()
    def observe(self, model, x):
        h = x
        li = 0
        for m in model.body:
            h = m(h)
            if isinstance(m, nn.ReLU):
                act = h.abs().mean(0)                       # mean |activation| per unit
                out_norm = self.nexts[li].weight.abs().mean(0)  # outgoing magnitude
                u = act * out_norm
                self.util[li] = self.decay * self.util[li] + (1 - self.decay) * u
                self.age[li] += 1
                li += 1

    @torch.no_grad()
    def maybe_reinit(self, model, step, gen):
        if step % self.period != 0:
            return
        for li, lin in enumerate(self.lins):
            nu = lin.out_features
            n_re = max(1, int(self.frac * nu))
            mature = self.age[li] > self.period          # only reinit matured units
            u = self.util[li].clone(); u[~mature] = float("inf")
            idx = torch.argsort(u)[:n_re]
            idx = idx[torch.isfinite(u[idx])]
            for j in idx.tolist():
                lin.weight[j].normal_(0, (1.0 / lin.in_features) ** 0.5, generator=gen)
                lin.bias[j].zero_()
                self.nexts[li].weight[:, j].zero_()       # new unit starts silent
                self.util[li][j] = 0.0; self.age[li][j] = 0
