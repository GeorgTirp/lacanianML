"""Top-k curvature eigenspace via Hessian-vector products + subspace iteration.

Reuses the curvature-probe idea from paper 2. The top-k eigenvectors span the
COMMITTED / low-EU subspace S_hi; the drive is confined to its complement (the
flat / high-EU directions good optimizers vacate). Basis error grows with width
(paper 2) — a known risk tested via the width sweep (§7).
"""
import torch


def hvp(model, lossf, x, y, vec):
    """Hessian-vector product H·vec (flat), H = Hessian of L(model(x),y)."""
    params = list(model.parameters())
    g = torch.autograd.grad(lossf(model(x), y), params, create_graph=True)
    gflat = torch.cat([gg.reshape(-1) for gg in g])
    hv = torch.autograd.grad(gflat @ vec, params, retain_graph=False)
    return torch.cat([h.reshape(-1) for h in hv]).detach()


def top_k_subspace(model, lossf, x, y, k, iters=3, V0=None):
    """Orthogonal (subspace) iteration -> orthonormal basis (n,k) of the top-k
    Hessian eigenspace. Warm-started from V0 (previous basis) => 1-2 iters suffice.
    """
    n = sum(p.numel() for p in model.parameters())
    if V0 is None:
        V = torch.randn(n, k)
    else:
        V = V0.clone()
    V, _ = torch.linalg.qr(V)
    for _ in range(iters):
        HV = torch.stack([hvp(model, lossf, x, y, V[:, j]) for j in range(k)], dim=1)
        V, _ = torch.linalg.qr(HV)
    return V                                            # (n, k), orthonormal
