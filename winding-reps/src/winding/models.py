"""Models (torch): phase encoder, GRU baseline, point-track MLP, product latent.

Architectures and budgets are matched across arms (§4). Only the phase center
differs between A/B/D; C is the supervised baseline with matched parameter count.
"""
import torch
import torch.nn as nn

from .config import CFG


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


class PhaseEncoder(nn.Module):
    """Per-step encoder f_theta: R^D -> R^2 (MLP, 2 hidden layers of `hidden`).

    phi_theta(x) = f/||f|| in S^1; the point f=0 is the gate. Used by arms A/B/D.
    Applied pointwise to every step of a trajectory.
    """

    def __init__(self, cfg=CFG, gate_init_scale=0.02):
        super().__init__()
        h = cfg.enc_hidden
        self.net = nn.Sequential(
            nn.Linear(cfg.D, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 2),
        )
        # Start the phase head near the gate (f ~ 0) so the winding class is
        # *born* by crossing the gate during installation (Tier-1 protocol).
        # This is what makes the conservation law observable: winding changes
        # coincide with min||f|| -> 0 rather than being aliased by the encoder
        # emitting a large winding at initialization. Shared across A/B/D (§7.2).
        with torch.no_grad():
            self.net[-1].weight.mul_(gate_init_scale)
            self.net[-1].bias.mul_(gate_init_scale)

    def forward(self, x):
        """x: (..., D) -> f: (..., 2). Works on (N,T,D) or (N,D)."""
        return self.net(x)


class PointMLP(nn.Module):
    """Small MLP classifier for the point track (used by the EU ensemble)."""

    def __init__(self, cfg=CFG):
        super().__init__()
        h = cfg.ens_hidden
        self.net = nn.Sequential(
            nn.Linear(cfg.D, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, cfg.n_sectors),
        )

    def forward(self, x):
        return self.net(x)


class GRUBaseline(nn.Module):
    """Baseline C: GRU over the embedded trajectory -> 3-class winding softmax.

    Classes {-1,0,+1} mapped to {0,1,2}. Parameter count matched to the phase
    encoder within ~2x (see models.match_report)."""

    def __init__(self, cfg=CFG):
        super().__init__()
        self.gru = nn.GRU(cfg.D, cfg.gru_hidden, batch_first=True)
        self.head = nn.Linear(cfg.gru_hidden, 3)

    def forward(self, x):
        """x: (N, T, D) -> logits (N, 3)."""
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class ProductEncoder(nn.Module):
    """exp1 product latent: R^D -> R^8 (geometric) x R^2 (phase/S^1).

    The guard's norm barrier acts on the R^2 phase part; the open question is
    whether that rank floor helps the geometric R^8 part.
    """

    def __init__(self, cfg=CFG, geom_dim=8):
        super().__init__()
        h = cfg.enc_hidden
        self.geom_dim = geom_dim
        self.trunk = nn.Sequential(
            nn.Linear(cfg.D, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
        )
        self.geom = nn.Linear(h, geom_dim)
        self.phase = nn.Linear(h, 2)

    def forward(self, x):
        z = self.trunk(x)
        return self.geom(z), self.phase(z)


def match_report(cfg=CFG):
    """Report parameter counts to document the C-vs-A budget match (§4)."""
    enc = _count_params(PhaseEncoder(cfg))
    gru = _count_params(GRUBaseline(cfg))
    return dict(phase_encoder=enc, gru_baseline=gru, ratio=gru / max(enc, 1))
