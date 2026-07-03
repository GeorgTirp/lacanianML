"""The drive (Tier-3): a closed non-exact term in the UPDATE RULE.

A real-valued loss provably cannot circulate (dL/dt = -||grad L||^2 <= 0 is
monotone), so persistent directed motion cannot come from any loss. The drive D
is added directly to the update:

    theta_{t+1} = theta_t - eta * grad(L_ssl + L_bar) + eta_d * D(theta_t)

D is the pullback, through the evaluation map theta -> f_theta(x) = (u, v), of
the angular 1-form dtheta = (u dv - v du)/(u^2 + v^2), averaged over a batch:

    D(theta) = mean_B [ u grad_theta v - v grad_theta u ] / (u^2 + v^2)

Implemented with a stop-gradient trick (exact, not an approximation):

    w_u = stopgrad(-v / max(u^2+v^2, m^2))
    w_v = stopgrad( u / max(u^2+v^2, m^2))
    D   = grad_theta mean_B( w_u * u + w_v * v )

D is *locally* a gradient (of the stop-gradded linear functional at that point)
but *globally* has no potential: its loop integral around the U(1) orbit is 2*pi
(a nonzero period) -- closed but not exact. Adding +eta_d*D advances every
sample's phase counterclockwise. The gate barrier stays PERMANENTLY active in all
deployment regimes; the max(., m^2) clip is numerical safety near the gate.
"""
import math

import torch
import torch.nn as nn

from .losses import phase_of

PI = math.pi


# --------------------------------------------------------------------------- #
#  relational (U(1)-invariant) streaming loss + its offset head               #
# --------------------------------------------------------------------------- #
class QHead(nn.Module):
    """Learned expected phase advance q_psi(k) over step offset k in {1..kmax}.

    L_ssl depends only on phase DIFFERENCES, so the global phase rotation U(1) is
    an exact symmetry -- the absolute phase is the data-unidentified fiber the
    drive sweeps. (Do NOT feed absolute (cos phi, sin phi) as state anywhere: that
    would anchor the phase and put a spurious dL component along the orbit.)
    """

    def __init__(self, kmax):
        super().__init__()
        self.kmax = kmax
        self.q = nn.Embedding(kmax, 1)
        nn.init.zeros_(self.q.weight)

    def forward(self, k):                      # k: (B,) longs in {1..kmax}
        return self.q(k - 1).squeeze(-1)


def sample_pairs(x_traj, kmax, gen):
    """Sample (x_t, x_{t+k}, k) along closed loops. x_traj: (B,T,D)."""
    B, T, _ = x_traj.shape
    t = torch.randint(0, T, (B,), generator=gen)
    k = torch.randint(1, kmax + 1, (B,), generator=gen)
    ar = torch.arange(B)
    x_t = x_traj[ar, t]
    x_tk = x_traj[ar, (t + k) % T]
    return x_t, x_tk, k


def lssl(encoder, q, x_t, x_tk, k):
    """L_ssl = E[1 - cos(phi(x_{t+k}) - phi(x_t) - q(k))] (relational)."""
    phi_t, _ = phase_of(encoder(x_t))
    phi_tk, _ = phase_of(encoder(x_tk))
    return torch.mean(1.0 - torch.cos(phi_tk - phi_t - q(k)))


# --------------------------------------------------------------------------- #
#  the drive field D                                                          #
# --------------------------------------------------------------------------- #
def _drive_scalar(encoder, x, margin):
    f = encoder(x).reshape(-1, 2)
    u, v = f[:, 0], f[:, 1]
    denom = torch.clamp(u * u + v * v, min=margin ** 2)
    w_u = (-v / denom).detach()
    w_v = (u / denom).detach()
    return torch.mean(w_u * u + w_v * v)


def drive_vector(encoder, x, margin):
    """D as a tuple of grads aligned with encoder.parameters()."""
    scalar = _drive_scalar(encoder, x, margin)
    return torch.autograd.grad(scalar, list(encoder.parameters()))


def apply_drive(encoder, grads, eta_d):
    """theta += eta_d * D (outside the optimizer; a separate flow, not through it)."""
    with torch.no_grad():
        for p, g in zip(encoder.parameters(), grads):
            p.add_(eta_d * g)


# --------------------------------------------------------------------------- #
#  exp4a period-test utilities: rigid rotation of the final-layer output rows  #
# --------------------------------------------------------------------------- #
def _final(encoder):
    return encoder.net[-1]           # Linear(H, 2): weight (2,H), bias (2,)


def _rot(s):
    c, sn = math.cos(s), math.sin(s)
    return torch.tensor([[c, -sn], [sn, c]], dtype=torch.float32)


def _rot_prime(s):
    c, sn = math.cos(s), math.sin(s)
    return torch.tensor([[-sn, -c], [c, -sn]], dtype=torch.float32)


def set_rotated_final(encoder, W0, b0, s):
    """Rotate the two output rows by s -> rotates every (u,v) rigidly (phase += s).
    s in [0, 2*pi] is an EXACT closed loop in parameter space."""
    R = _rot(s)
    with torch.no_grad():
        _final(encoder).weight.copy_(R @ W0)
        _final(encoder).bias.copy_(R @ b0)


def _grads_final(scalar, encoder):
    return torch.autograd.grad(scalar, [_final(encoder).weight, _final(encoder).bias])


def period_integrals(encoder, x, pairs, margin, n=200):
    """Discretize the U(1) loop s in [0,2pi) and integrate the two 1-forms along
    it: (circ_D, circ_Lssl). Expected: circ_D ~ 2pi (closed non-exact),
    circ_Lssl ~ 0 (exact). `pairs` = (x_t, x_tk, k, q) for the L_ssl form."""
    x_t, x_tk, k, q = pairs
    W0 = _final(encoder).weight.detach().clone()
    b0 = _final(encoder).bias.detach().clone()
    ss = torch.linspace(0, 2 * PI, n + 1)[:-1]
    ds = float(2 * PI / n)
    circ_D = circ_L = 0.0
    for s in ss:
        s = float(s)
        set_rotated_final(encoder, W0, b0, s)
        dW = _rot_prime(s) @ W0
        db = _rot_prime(s) @ b0
        gW, gb = _grads_final(_drive_scalar(encoder, x, margin), encoder)
        circ_D += float((gW * dW).sum() + (gb * db).sum()) * ds
        lW, lb = _grads_final(lssl(encoder, q, x_t, x_tk, k), encoder)
        circ_L += float((lW * dW).sum() + (lb * db).sum()) * ds
    with torch.no_grad():                       # restore
        _final(encoder).weight.copy_(W0)
        _final(encoder).bias.copy_(b0)
    return circ_D, circ_L


# --------------------------------------------------------------------------- #
#  diagnostics: cumulative phase, plasticity                                   #
# --------------------------------------------------------------------------- #
def circular_mean_phase(encoder, x_probe):
    """Circular mean of phi over a fixed probe set (a single angle in (-pi,pi])."""
    with torch.no_grad():
        phi, _ = phase_of(encoder(x_probe))
    return float(torch.atan2(torch.sin(phi).mean(), torch.cos(phi).mean()))


def penultimate(encoder, x):
    """Activations after the last hidden Tanh (before the R^2 output)."""
    with torch.no_grad():
        return encoder.net[:4](x)


def plasticity_metrics(encoder, x):
    """participation ratio of penultimate features, fraction of saturated tanh
    units (|h|>0.95). Both are standard plasticity-loss proxies."""
    h = penultimate(encoder, x).numpy()
    hc = h - h.mean(0, keepdims=True)
    cov = (hc.T @ hc) / max(len(hc) - 1, 1)
    import numpy as np
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    pr = float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-12)) if lam.sum() > 0 else 0.0
    sat = float((np.abs(h) > 0.95).mean())
    return pr, sat
