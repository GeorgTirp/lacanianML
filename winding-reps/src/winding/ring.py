"""Ring attractor (v6): topology in the DYNAMICS, not the readout map.

Tiers 2-5 painted S^1 onto a static readout phi = f/||f||. Every deployment-side
failure (uninstallable without oracle angles, unrepairable after a shift P12,
shattered by distribution jumps P9, 2-unit bottleneck P7) traces to that
staticness. v6 moves the circle into recurrent dynamics: N ring units with FIXED
cosine connectivity form a continuous S^1 attractor by construction. The encoder
e_psi only learns to inject input current onto that ring — descent never has to
create the hole; the recurrence supplies it (resolving the gradient-dead
obstruction structurally). An attractor projects a noisy/shifted state back onto
its manifold, so the winding can REPAIR itself by relaxation (P15) — the thing a
static map provably could not do.

Biological existence proofs (not evidence): head-direction ring attractors and
grid-cell toroidal attractors, topology recoverable from population activity and
persisting through sensory shutdown (Chaudhuri et al. 2019; Gardner et al. 2022).
"""
import math

import numpy as np
import torch
import torch.nn as nn

from .config import CFG

PI = math.pi


def preferred_angles(N):
    return torch.arange(N, dtype=torch.float32) * (2 * PI / N)


def ring_weights(cfg=CFG, eta=None):
    """Fixed recurrent connectivity W_ij = (J0 + J1 cos(dϑ) + eta sin(dϑ)) / N.

    The symmetric cosine part builds the RING of marginally-stable bump states
    (a continuous S^1 attractor when J1 is above the bump-formation threshold).
    The antisymmetric sin part (eta > 0) makes the bump circulate spontaneously —
    the intrinsic drive (P16), a property of the module, not an optimizer term.
    NOT learned.
    """
    N = cfg.N_ring
    th = preferred_angles(N)
    d = th[:, None] - th[None, :]
    e = cfg.ring_eta if eta is None else eta
    W = (cfg.ring_J0 + cfg.ring_J1 * torch.cos(d) + e * torch.sin(d)) / N
    return W


def _phi(u):
    """Rectified-tanh transfer: bounded in [0,1) -> stable bump amplitude."""
    return torch.relu(torch.tanh(u))


def relax(W, b, cfg=CFG, R=None, r0=None):
    """Euler relaxation of the rate dynamics for R steps.

        r <- r + (dt/tau) [ -r + Phi( W r + b + h0 ) ]

    W: (N,N); b: (B,N) input current (may be 0). Returns settled r (B,N).
    Differentiable in b, so L_ssl can train the encoder through the relaxation.
    """
    R = cfg.ring_R if R is None else R
    B, N = b.shape
    r = torch.zeros(B, N) if r0 is None else r0
    a = cfg.ring_dt / cfg.ring_tau
    for _ in range(R):
        r = r + a * (-r + _phi(r @ W.T + b + cfg.ring_h0))
    return r


def decode_phase(r, thetas):
    """Population-vector angle psi(r) = atan2(sum r_i sin ϑ_i, sum r_i cos ϑ_i)."""
    s = (r * torch.sin(thetas)).sum(-1)
    c = (r * torch.cos(thetas)).sum(-1)
    return torch.atan2(s, c)


def decode_amplitude(r, thetas):
    """Bump amplitude rho = |sum r_i e^{i ϑ_i}| / sum r_i (the magnitude register;
    the ring analogue of ||f|| — rho -> 0 is the bump-collapse 'gate' event)."""
    s = (r * torch.sin(thetas)).sum(-1)
    c = (r * torch.cos(thetas)).sum(-1)
    tot = r.sum(-1).clamp_min(1e-6)
    return torch.sqrt(s * s + c * c) / tot


class RingEncoder(nn.Module):
    """e_psi: R^D -> R^N. The ONLY learned map in v6 (plus q_phi offset head).
    Injects input current onto the ring; the recurrence supplies the topology."""

    def __init__(self, cfg=CFG):
        super().__init__()
        h = cfg.enc_hidden
        self.gain = cfg.ring_input_gain
        self.net = nn.Sequential(
            nn.Linear(cfg.D, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, cfg.N_ring),
        )

    def forward(self, x):
        return self.gain * self.net(x)


def settle_sequence(enc, W, x_seq, cfg=CFG, R=None, carry=True):
    """Settle the bump along a trajectory of inputs x_seq (T, D).

    carry=True continues relaxation from the previous step's state (the ring
    integrates the path — how a head-direction system tracks); carry=False
    settles each input from rest. Returns psi (T,), rho (T,), states (T,N).
    """
    thetas = preferred_angles(cfg.N_ring)
    T = x_seq.shape[0]
    b = enc(x_seq)                                   # (T, N)
    r = None
    psi, rho, states = [], [], []
    for t in range(T):
        r = relax(W, b[t:t + 1], cfg, R=R, r0=(r if carry else None))
        psi.append(decode_phase(r, thetas)[0])
        rho.append(decode_amplitude(r, thetas)[0])
        states.append(r[0])
    return torch.stack(psi), torch.stack(rho), torch.stack(states)


def settle_batch(enc, W, x, cfg=CFG, R=None, carry=True):
    """Batched settle over trajectories x (B,T,D). carry=True integrates the path
    (the ring tracks like a head-direction system). Returns psi (B,T), rho (B,T),
    final states r (B,N). Differentiable through the relaxation for training."""
    thetas = preferred_angles(cfg.N_ring)
    R = cfg.ring_R if R is None else R
    a = cfg.ring_dt / cfg.ring_tau
    b = enc(x)                                       # (B,T,N)
    B, T, N = b.shape
    r = torch.zeros(B, N)
    psis, rhos = [], []
    for t in range(T):
        if not carry:
            r = torch.zeros(B, N)
        bt = b[:, t, :]
        for _ in range(R):
            r = r + a * (-r + _phi(r @ W.T + bt + cfg.ring_h0))
        s = (r * torch.sin(thetas)).sum(-1); c = (r * torch.cos(thetas)).sum(-1)
        psis.append(torch.atan2(s, c))
        rhos.append(torch.sqrt(s * s + c * c) / r.sum(-1).clamp_min(1e-6))
    return torch.stack(psis, 1), torch.stack(rhos, 1), r


def decode_current(enc, x, cfg=CFG):
    """STATIC readout (P15 baseline): decode phase from the feedforward current
    directly (population vector of relu(e_psi(x))), with NO recurrent relaxation —
    the exp2-style 'read the map immediately' path."""
    thetas = preferred_angles(cfg.N_ring)
    b = torch.relu(enc(x))                            # (B,T,N)
    s = (b * torch.sin(thetas)).sum(-1); c = (b * torch.cos(thetas)).sum(-1)
    return torch.atan2(s, c)                          # (B,T)


# --------------------------------------------------------------------------- #
#  b1 (ring-topology) certificate  (cheap proxy; report the diagram, §6)       #
# --------------------------------------------------------------------------- #
def ring_certificate(states, n_bins=36):
    """Certify the settled-state cloud is a 1D closed curve (b1 = 1).

    PCA to 2D; a ring covers all angular sectors at roughly constant radius,
    whereas a blob (b1=0) sits near the centre and two clusters (b1=0) clump in
    angle. Pre-committed (§6): b1=1 iff angular coverage > 0.9 AND radial CV < 0.5
    AND top-2 PCA variance fraction > 0.5. Also returns a k-NN graph cycle-rank
    proxy and a loop-persistence ratio for transparency.
    """
    X = np.asarray(states, dtype=float)
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pca = X @ Vt[:2].T
    ang = np.arctan2(pca[:, 1], pca[:, 0])
    rad = np.linalg.norm(pca, axis=1)
    hist, _ = np.histogram(ang, bins=n_bins, range=(-PI, PI))
    coverage = float(np.mean(hist > 0))
    radial_cv = float(np.std(rad) / (np.mean(rad) + 1e-9))
    var2 = float((S[:2] ** 2).sum() / ((S ** 2).sum() + 1e-9))
    # loop-persistence proxy: cross-cloud extent / local spacing (thin ring >> 1)
    sub = X[np.random.default_rng(0).choice(len(X), min(200, len(X)), replace=False)]
    d = np.linalg.norm(sub[:, None] - sub[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn_dist = float(np.median(d.min(1)))
    d2 = d.copy(); d2[np.isinf(d2)] = 0
    loop_persist = float(np.median(d2[d2 > 0]) / (nn_dist + 1e-9))
    b1 = int(coverage > 0.9 and radial_cv < 0.5 and var2 > 0.5)
    return dict(b1=b1, coverage=coverage, radial_cv=radial_cv, var2=var2,
                loop_persist=loop_persist, pca=pca)
