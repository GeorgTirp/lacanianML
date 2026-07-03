"""Losses (torch). Installation loss, gate barrier, wrap.

Design constraints:
  - W1: installation signal must span the *full loop* (see train.py: L_inst is
    averaged over all T points of a loop against the oracle angle, not over
    small local increments).
  - W3: never optimize the integer winding directly. L_inst is a continuous
    surrogate; the integer is only monitored.
"""
import math
import torch

PI = math.pi


def wrap_torch(d):
    """Wrap to (-pi, pi] for torch tensors."""
    return torch.remainder(d + PI, 2 * PI) - PI


def installation_loss(phi, ang_target):
    """L_inst = E[1 - cos(phi - ang_target)].

    phi: predicted phase angle(s) of phi_theta(x) = f/||f||.
    ang_target: oracle angle ang_c(p) at the same points.
    Averaged over every point supplied (full-loop supervision, W1).
    """
    return torch.mean(1.0 - torch.cos(phi - ang_target))


def gate_barrier(f, margin, lam_b):
    """L_bar = lam_b * E[max(0, m - ||f||)^2].  Guards the gate f=0."""
    norm = torch.linalg.norm(f, dim=-1)
    return lam_b * torch.mean(torch.clamp(margin - norm, min=0.0) ** 2)


def stabilizer_loss(phi_v1, phi_v2):
    """v5 label-free class stabilizer: multi-view phase agreement
    L_stab = E[1 - cos(phi_v1 - phi_v2)] over two independent observation-noise
    views of the same underlying point. Anchors the phase field against
    shift-driven scrambling. The integer winding is monitored, never optimized
    (W3): this is a continuous surrogate whose side effect is class stability."""
    return torch.mean(1.0 - torch.cos(phi_v1 - phi_v2))


def phase_of(f, eps=1e-8):
    """Angle of the 2D vector(s) f = (u, v): atan2(v, u). Also returns norm."""
    norm = torch.linalg.norm(f, dim=-1, keepdim=True)
    fn = f / (norm + eps)
    phi = torch.atan2(fn[..., 1], fn[..., 0])
    return phi, norm.squeeze(-1)
